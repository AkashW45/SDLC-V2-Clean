"""
Phase 6 — Delivery Agent
Pushes generated code changes to GitHub and creates PRs.
Human approval gate — engineers must review PR before merge.
"""

import os
import time
import subprocess
import tempfile
import shutil
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
# Auto-register newly-created repos into the knowledge layer so subsequent
# pipelines (Phase 0 routing, manual repo picker, semantic search) can find them.
try:
    from knowledge_layer.project_registry import register_project
    from agents.indexer_queue import get_queue
    _REGISTRY_AVAILABLE = True
except ImportError as _e:
    print(f"[Phase 6] Auto-register dependencies missing: {_e}")
    _REGISTRY_AVAILABLE = False

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")
# If set, new repos are created INSIDE this organization via POST /orgs/{org}/repos.
# If empty/unset, repos are created under the token owner's personal account via
# POST /user/repos. This is the difference between a repo showing up in your org
# vs. silently landing in your personal account.
GITHUB_ORG = os.getenv("GITHUB_ORG", "").strip()
# The account that actually owns the repos: the org when creating in an org,
# otherwise the personal owner. All repo existence checks, clone URLs, and PR
# URLs must use THIS, so they point at wherever the repo was actually created.
EFFECTIVE_OWNER = GITHUB_ORG or GITHUB_OWNER


# -----------------------------------------
# State
# -----------------------------------------

class DeliveryState(TypedDict, total=False):
    requirement: str
    generated_changes: list
    test_files: list
    branch_name: str
    repo_url: str
    pr_urls: list
    # Cross-repo additions:
    per_repo_push_status: dict
    per_repo_pr: dict

    # ── PR merge bookkeeping (populated by merge_prs node) ───────────────
    # After the human approves Phase 6, the merge_prs node merges each PR
    # and records the resulting commit SHA so Phase 7 can `git checkout <sha>`
    # for a reproducible build.
    merged_shas: dict          # {repo_name: sha}  — SHAs to deploy from
    merge_errors: list         # list of failure messages (empty on success)

    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# Git Helpers
# -----------------------------------------

def run_git(cmd: list, cwd: str) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Git error: {result.stderr.strip()}")
    return result.stdout.strip()


def create_github_repo(
        repo_name: str,
        private: bool = True
) -> dict:
    """
    Create a new GitHub repository for greenfield projects.

    Creates inside GITHUB_ORG (POST /orgs/{org}/repos) when that env var is set,
    otherwise under the token owner's personal account (POST /user/repos).
    NOTE: POST /user/repos ALWAYS creates under the account that owns the token —
    it cannot create org repos. Org creation requires the /orgs/{org}/repos
    endpoint and a token with the 'repo' scope (classic) or repository-creation
    permission on that org (fine-grained), plus org membership/permission to
    create repos.
    """

    if GITHUB_ORG:
        url = f"https://api.github.com/orgs/{GITHUB_ORG}/repos"
    else:
        url = "https://api.github.com/user/repos"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "name": repo_name,
        "private": private,
        "auto_init": True
    }

    response = requests.post(url, headers=headers, json=payload)

    # Repo created successfully
    if response.status_code == 201:
        repo = response.json()
        target = repo.get("full_name", f"{GITHUB_ORG or '?'}/{repo_name}")
        print(f"  ✅ Created repo at {url} → {target}")
        # auto_init creates the initial README commit ASYNCHRONOUSLY. The 201
        # response can return before that commit exists, so a clone fired
        # immediately afterwards may hit an empty repo (no default branch ref).
        # Block until the initial commit is visible before returning.
        _wait_for_initial_commit(repo.get("full_name", ""), headers)
        return repo

    # Repo already exists
    if response.status_code == 422 and "name already exists" in response.text:
        return {"status": "ALREADY_EXISTS"}

    raise Exception(
        f"GitHub repo creation failed: "
        f"{response.status_code} - {response.text}"
    )


def _wait_for_initial_commit(
        full_name: str,
        headers: dict,
        attempts: int = 10,
        delay: float = 1.0
) -> bool:
    """Poll GitHub until the repo has at least one commit.

    Returns True once a commit is visible, False if it never appeared within
    the budget. Called after auto_init repo creation to defeat the clone race.
    """
    if not full_name:
        return False

    commits_url = f"https://api.github.com/repos/{full_name}/commits"
    for _ in range(attempts):
        resp = requests.get(commits_url, headers=headers)
        # 200 with a non-empty list → initial commit exists.
        # 409 ("Git Repository is empty") → auto_init hasn't landed yet.
        if resp.status_code == 200 and resp.json():
            return True
        time.sleep(delay)

    print(f"  ⚠️  Initial commit for '{full_name}' not visible after "
          f"{attempts}s — proceeding anyway (clone may still race).")
    return False


def push_files_to_branch(
        repo_url: str,
        branch_name: str,
        files: list,
        commit_message: str
) -> dict:
    """Clone repo, write files, push to branch."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Clone. On a freshly auto_init'd repo this normally works, but if the
        # initial commit still hasn't landed git prints a warning about cloning
        # an empty repository and leaves HEAD on an unborn branch — handled below.
        run_git(["git", "clone", repo_url, "."], cwd=temp_dir)

        # Configure git identity early so commits work even on an unborn branch.
        run_git(["git", "config", "user.email", "ai-sdlc@bot.com"], cwd=temp_dir)
        run_git(["git", "config", "user.name", "AI SDLC Bot"], cwd=temp_dir)

        # Detect whether the clone actually has any commits. A just-created repo
        # whose auto_init commit hasn't propagated yet clones as "empty": HEAD
        # points at an unborn branch and `git branch -r` lists nothing.
        rev_check = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=temp_dir, capture_output=True, text=True
        )
        repo_has_commits = rev_check.returncode == 0

        if repo_has_commits:
            run_git(["git", "fetch", "--all"], cwd=temp_dir)
            remote_branches = run_git(["git", "branch", "-r"], cwd=temp_dir)
            if f"origin/{branch_name}" in remote_branches:
                run_git(["git", "checkout", branch_name], cwd=temp_dir)
                run_git(["git", "pull", "origin", branch_name], cwd=temp_dir)
            else:
                run_git(["git", "checkout", "-b", branch_name], cwd=temp_dir)
        else:
            # Empty repo → put HEAD on the target branch directly. The first
            # commit will create it. Use -B so it works whether or not the
            # unborn branch already happens to be named branch_name.
            run_git(["git", "checkout", "-B", branch_name], cwd=temp_dir)

        # Write files
        files_written = 0
        for file in files:
            file_path = file["file_path"].replace("\\", os.sep)
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file["content"])
            files_written += 1

        # Commit and push
        status = run_git(["git", "status", "--porcelain"], cwd=temp_dir)
        if not status:
            return {"status": "NO_CHANGES", "branch": branch_name,
                    "files_pushed": 0}

        run_git(["git", "add", "."], cwd=temp_dir)
        run_git(["git", "commit", "-m", commit_message], cwd=temp_dir)
        run_git(
            ["git", "push", "--set-upstream", "origin", branch_name, "--force"],
            cwd=temp_dir
        )

        return {
            "status": "PUSH_SUCCESS",
            "branch": branch_name,
            "files_pushed": files_written
        }

    except Exception as e:
        # Surface the real git error instead of letting it disappear. The
        # caller (push_code) inspects status and must treat this as a failure.
        return {"status": "PUSH_FAILED", "branch": branch_name,
                "files_pushed": 0, "error": str(e)}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_github_pr(
        repo_name: str,
        branch_name: str,
        title: str,
        body: str
) -> dict:
    """Create a GitHub PR via API."""
    url = f"https://api.github.com/repos/{EFFECTIVE_OWNER}/{repo_name}/pulls"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # Auto-detect the repo's default branch instead of hardcoding "main".
    # Older repos use "master", newer ones use "main". GitHub rejects PRs
    # with status 422 "field:base invalid" when base doesn't exist.
    try:
        repo_info_url = f"https://api.github.com/repos/{EFFECTIVE_OWNER}/{repo_name}"
        info_resp = requests.get(repo_info_url, headers=headers)
        if info_resp.status_code == 200:
            base_branch = info_resp.json().get("default_branch", "main")
        else:
            base_branch = "main"  # fallback
    except Exception:
        base_branch = "main"

    payload = {
        "title": title,
        "head": branch_name,
        "base": base_branch,
        "body": body
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        return response.json()

    # PR already exists
    if response.status_code == 422 and "already exists" in response.text:
        pr_list_url = f"https://api.github.com/repos/{EFFECTIVE_OWNER}/{repo_name}/pulls?head={EFFECTIVE_OWNER}:{branch_name}&state=open"
        pr_resp = requests.get(pr_list_url, headers=headers)
        prs = pr_resp.json()
        if prs:
            return prs[0]

    raise Exception(f"GitHub PR creation failed: {response.text}")


# -----------------------------------------
# Nodes
# -----------------------------------------
def _push_single_repo(state: DeliveryState, repo_name: str, payload: dict) -> dict:
    """
    Push to a single repo. Used by both single-repo path and cross-repo
    parallel path (called once per repo).

    Returns: {"status": "CODE_PUSHED" | "PUSH_FAILED", "repo_name": ..., "error": ...}
    """
    branch_name = state["branch_name"]
    repo_url = state["repo_url"]

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise Exception("GITHUB_TOKEN is missing")

    # Inject auth into clone URL
    if repo_url.startswith("https://github.com/"):
        repo_url = repo_url.replace(
            "https://github.com/",
            f"https://x-access-token:{github_token}@github.com/"
        )

    # Check if repo exists on GitHub
    repo_check_url = f"https://api.github.com/repos/{EFFECTIVE_OWNER}/{repo_name}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    repo_check = requests.get(repo_check_url, headers=headers)

    newly_created = False
    if repo_check.status_code == 404:
        print(f"  ℹ️  Repo '{repo_name}' does not exist. Creating...")
        try:
            create_github_repo(repo_name=repo_name)
            print(f"  ✅ Repository created: {repo_name}")
            newly_created = True
        except Exception as e:
            print(f"  ❌ Repo creation failed: {e}")
            return {"status": "PUSH_FAILED", "error": str(e), "repo_name": repo_name}

        # Early-register so Phase 0 can find it on the next pipeline
        if _REGISTRY_AVAILABLE:
            try:
                print(f"  [Phase 6] Registering '{repo_name}' in knowledge layer...")
                register_project(
                    project_id=repo_name,
                    project_name=repo_name.replace("-", " ").title(),
                    description=(state.get("requirement", "")[:300]
                                 or "Auto-created by SDLC-V2 pipeline"),
                    domain="generated",
                    tech_stack=[],
                    repos=[repo_name],
                    owner_team="SDLC-V2",
                )
                print(f"  [Phase 6] ✅ Registered '{repo_name}' in Postgres + Qdrant")
            except Exception as e:
                print(f"  [Phase 6] ⚠️  Early registration failed: {e}")
    elif repo_check.status_code not in (200, 301):
        msg = (f"Cannot access repo '{EFFECTIVE_OWNER}/{repo_name}': "
               f"HTTP {repo_check.status_code} - {repo_check.text[:200]}")
        print(f"  ❌ {msg}")
        return {"status": "PUSH_FAILED", "error": msg, "repo_name": repo_name}

    # Build the file list — code changes + test files for THIS repo
    all_files = []
    for change in payload.get("changes", []):
        if isinstance(change, dict) and change.get("file_path") and change.get("content"):
            all_files.append({
                "file_path": change["file_path"],
                "content": change["content"],
            })

    for test_file in payload.get("tests", []):
        if isinstance(test_file, dict) and test_file.get("content"):
            all_files.append({
                "file_path": test_file.get("test_file_path", ""),
                "content": test_file["content"],
            })

    if not all_files:
        return {
            "status": "PUSH_FAILED",
            "error": f"No files to push for {repo_name}",
            "repo_name": repo_name,
        }

    print(f"  Pushing {len(all_files)} files to {repo_name} branch: {branch_name}")

    push_result = push_files_to_branch(
        repo_url=repo_url,
        branch_name=branch_name,
        files=all_files,
        commit_message=f"feat: AI-generated changes — {state.get('requirement', '')[:60]}"
    )

    # PUSH_SUCCESS or NO_CHANGES (resume scenario) both count as success
    if (push_result["status"] == "PUSH_SUCCESS" or
            (push_result["status"] == "NO_CHANGES" and len(all_files) > 0)):
        print(f"  ✅ Pushed {push_result.get('files_pushed', 0)} files to {repo_name}")

        # Enqueue indexer for code symbol/embedding extraction (async, best-effort)
        if newly_created and _REGISTRY_AVAILABLE:
            try:
                queue = get_queue()
                repo_url_for_index = f"https://github.com/{EFFECTIVE_OWNER}/{repo_name}.git"
                job_id = queue.enqueue(
                    repo_name=repo_name,
                    repo_url=repo_url_for_index,
                    branch="main",
                    force=False,
                )
                print(f"  [Phase 6] ✅ Queued indexer job {job_id} for '{repo_name}'")
            except Exception as queue_err:
                print(f"  [Phase 6] ⚠️  Indexer queue: {queue_err}")

        return {"status": "CODE_PUSHED", "repo_name": repo_name}

    # Push failed
    err = push_result.get("error") or push_result["status"]
    print(f"  ❌ Push did not succeed for {repo_name}: {err}")
    return {"status": "PUSH_FAILED", "error": err, "repo_name": repo_name}

def push_code(state: DeliveryState) -> DeliveryState:
    """
    Production-grade cross-repo push for 10-15 repo scale.

    Groups changes by target_repo, pushes IN PARALLEL with per-repo PR
    creation. Concurrency capped at MAX_PARALLEL_PUSHES (default 4) to
    respect GitHub API rate limits.

    Falls through to single-repo path when only 1 target.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n[Phase 6] Pushing code to GitHub...")

    all_changes = state.get("generated_changes", [])
    test_files = state.get("test_files", [])

    # Group by target_repo
    by_repo = {}
    fallback_repo = None
    if state.get("repo_url"):
        fallback_repo = state["repo_url"].split("/")[-1].replace(".git", "")

    for c in all_changes:
        if isinstance(c, dict):
            target = c.get("target_repo") or fallback_repo
            by_repo.setdefault(target, {"changes": [], "tests": []})
            by_repo[target]["changes"].append(c)
        else:
            by_repo.setdefault(fallback_repo, {"changes": [], "tests": []})
            by_repo[fallback_repo]["changes"].append(c)

    for t in test_files:
        if isinstance(t, dict):
            target = t.get("target_repo") or fallback_repo
            by_repo.setdefault(target, {"changes": [], "tests": []})
            by_repo[target]["tests"].append(t)

    if not by_repo:
        return {**state, "status": "PUSH_FAILED", "error": "No changes to push"}

    # Single-repo: unchanged behavior
    if len(by_repo) == 1:
        return _push_single_repo(state, list(by_repo.keys())[0], next(iter(by_repo.values())))

    # Cross-repo: parallel fan-out
    n = len(by_repo)
    max_parallel = int(os.getenv("MAX_PARALLEL_PUSHES", "4"))
    concurrency = min(n, max_parallel)

    print(f"[Phase 6] Cross-repo delivery: {n} repos, concurrency={concurrency}")

    def run_push(repo_name, payload):
        if not repo_name:
            return repo_name, {"status": "SKIPPED", "error": "no target_repo"}
        try:
            print(f"[Phase 6]   → pushing to {repo_name}")
            sub_state = dict(state)
            sub_state["repo_url"] = f"https://github.com/{EFFECTIVE_OWNER}/{repo_name}.git"
            sub_state["generated_changes"] = payload["changes"]
            sub_state["test_files"] = payload["tests"]
            return repo_name, _push_single_repo(sub_state, repo_name, payload)
        except Exception as e:
            import traceback; traceback.print_exc()
            return repo_name, {"status": "FAILED", "error": str(e)}

    per_repo_results = {}
    pr_urls = []
    merged_shas = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_push, repo, payload): repo
            for repo, payload in by_repo.items()
        }
        for future in as_completed(futures):
            repo_name, result = future.result()
            per_repo_results[repo_name] = result.get("status", "UNKNOWN")
            if result.get("pr_url"):
                pr_urls.append(result["pr_url"])
            if result.get("merged_sha"):
                merged_shas[repo_name] = result["merged_sha"]

            icon = "✅" if result.get("status") in ("CODE_PUSHED", "PR_OPENED") else "❌"
            print(f"[Phase 6]   {icon} {repo_name}: {result.get('status')}")

    SUCCESS = {"CODE_PUSHED", "PR_OPENED"}
    succeeded = sum(1 for s in per_repo_results.values() if s in SUCCESS)
    total = len(per_repo_results)
    success_rate = succeeded / total if total else 0

    if success_rate == 1.0:
        overall_status = "CODE_PUSHED"
    elif success_rate >= 0.5:
        overall_status = "PARTIAL_PUSH"
    else:
        overall_status = "PUSH_FAILED"

    print(f"\n[Phase 6] Cross-repo summary: {succeeded}/{total} pushed "
          f"({success_rate*100:.0f}%), overall: {overall_status}")

    return {
        **state,
        "status": overall_status,
        "per_repo_push_status": per_repo_results,
        "pr_urls": pr_urls,
        "merged_shas": merged_shas,
    }



def create_pr(state: DeliveryState) -> DeliveryState:
    """
    Create GitHub PR(s) for the changes.

    Cross-repo: iterates over per_repo_push_status (set by parallel push_code)
    and opens one PR per repo. Single-repo: original behavior preserved.
    """
    print("\n[Phase 6] Creating GitHub PR(s)...")

    if state.get("status") in ("PUSH_FAILED", "PARTIAL_PUSH") and not state.get("per_repo_push_status"):
        return {**state, "status": "PR_SKIPPED"}

    branch_name = state["branch_name"]

    if branch_name in ("main", "master"):
        msg = (f"Refusing to open a PR from '{branch_name}' into itself. "
               f"Delivery must use a feature branch (feature/<thread>). "
               f"This indicates run_phase6 set the wrong branch_name.")
        print(f"  ❌ {msg}")
        return {**state, "status": "PR_FAILED", "error": msg}

    # Determine target repos. Cross-repo case: read from per_repo_push_status
    # (populated by parallel push_code). Single-repo: fall back to state["repo_url"].
    per_repo_status = state.get("per_repo_push_status") or {}

    if per_repo_status:
        # Cross-repo path — only open PRs for repos that successfully pushed
        target_repos = [
            repo_name for repo_name, status in per_repo_status.items()
            if status in ("CODE_PUSHED", "PR_OPENED")
        ]
        print(f"[Phase 6] Cross-repo PR creation for {len(target_repos)} repos")
    else:
        # Single-repo legacy path
        repo_url = state.get("repo_url", "")
        if not repo_url:
            return {**state, "status": "PR_FAILED", "error": "No repo_url and no per_repo_push_status"}
        target_repos = [repo_url.split("/")[-1].replace(".git", "")]

    if not target_repos:
        return {**state, "status": "PR_SKIPPED", "error": "No successfully-pushed repos to open PRs for"}

    # Group generated changes by target_repo for the PR body
    changes_by_repo = {}
    for c in state.get("generated_changes", []):
        if not isinstance(c, dict):
            continue
        repo = c.get("target_repo") or target_repos[0]
        changes_by_repo.setdefault(repo, []).append(c)

    tests_by_repo = {}
    for t in state.get("test_files", []):
        if not isinstance(t, dict):
            continue
        repo = t.get("target_repo") or target_repos[0]
        tests_by_repo.setdefault(repo, []).append(t)

    # Open one PR per repo
    pr_urls = []
    per_repo_pr = {}
    failed_repos = []

    for repo_name in target_repos:
        repo_changes = changes_by_repo.get(repo_name, [])
        repo_tests = tests_by_repo.get(repo_name, [])

        changes_summary = "\n".join([
            f"- `{c['file_path']}`: {c.get('change_summary', '')[:80]}"
            for c in repo_changes
        ]) or "_(no per-repo change list available)_"

        pr_body = f"""## AI-Generated Changes

**Requirement:** {state['requirement']}

**Target Repo:** `{repo_name}`

## Files Changed
{changes_summary}

## Tests Added
{len(repo_tests)} test file(s) generated

## Generated by
SDLC Automation Platform V2 — LangGraph Pipeline

---
*This PR was automatically generated. Please review carefully before merging.*
"""

        try:
            pr = create_github_pr(
                repo_name=repo_name,
                branch_name=branch_name,
                title=f"feat: {state['requirement'][:60]}",
                body=pr_body,
            )
            pr_url = pr.get("html_url", "")
            print(f"  ✅ PR created for {repo_name}: {pr_url}")
            pr_urls.append(pr_url)
            per_repo_pr[repo_name] = pr_url
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ PR creation failed for {repo_name}: {e}")
            failed_repos.append(repo_name)
            per_repo_pr[repo_name] = f"FAILED: {e}"

    # Determine overall status
    if pr_urls and not failed_repos:
        overall_status = "PR_CREATED"
    elif pr_urls and failed_repos:
        overall_status = "PR_PARTIAL"
    else:
        overall_status = "PR_FAILED"

    print(f"\n[Phase 6] PR creation summary: {len(pr_urls)}/{len(target_repos)} succeeded, overall: {overall_status}")

    return {
        **state,
        "pr_urls": pr_urls,
        "per_repo_pr": per_repo_pr,
        "status": overall_status,
    }

def human_approval_gate(state: DeliveryState) -> DeliveryState:
    """Human reviews the PR before merge is approved."""
    print("\n[Phase 6] ⏸ Waiting for PR review approval...")

    for url in state.get("pr_urls", []):
        print(f"  PR to review: {url}")

    human_input = interrupt("Review the PR and approve or reject")

    approved = False
    feedback = ""
    if isinstance(human_input, dict):
        approved = human_input.get("approved", False)
        feedback = human_input.get("feedback", "")

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_PR_APPROVAL"
    }


def process_approval(state: DeliveryState) -> DeliveryState:
    if state.get("approved"):
        print(f"\n[Phase 6] ✅ PR approved — ready for deployment")
        return {**state, "status": "APPROVED_FOR_DEPLOYMENT"}
    else:
        print(f"\n[Phase 6] ❌ PR rejected — {state.get('human_feedback', '')}")
        return {**state, "status": "PR_REJECTED"}


# -----------------------------------------
# PR Merge helpers — used by the merge_prs node
# -----------------------------------------
# Why these live in delivery_agent.py and not in pr_manager.py:
#   pr_manager.py is explicitly designed to never auto-merge (see its
#   module-level docstring). That's a sensible invariant — PR creation
#   should be cheap and reversible. Merging, by contrast, only happens
#   AFTER a human has approved Phase 6 on the dashboard, so it belongs
#   in the delivery flow.

def _parse_pr_url(pr_url: str) -> tuple:
    """Parse 'https://github.com/<owner>/<repo>/pull/<n>' → (owner, repo, n).

    Returns (None, None, None) if the URL doesn't look like a PR. The
    'PR_SKIPPED_NEW_PROJECT' case (where the URL points to the repo root
    instead of a /pull/<n> path) is detected here and reported as None.
    """
    import re as _re
    m = _re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url or "")
    if not m:
        return None, None, None
    return m.group(1), m.group(2), int(m.group(3))


def _merge_one_pr(owner: str, repo: str, pr_number: int,
                  commit_title: str, commit_message: str) -> dict:
    """Call GitHub's PUT /repos/{owner}/{repo}/pulls/{n}/merge.

    Returns:
        {"merged": bool, "sha": str|None, "error": str|None}

    GitHub merge_method = "squash" so each AI-generated PR lands as a single
    clean commit on main, regardless of how many small commits the feature
    branch had. Two other options are "merge" (preserves all commits) and
    "rebase" (replays each commit onto main).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/merge"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "merge_method": "squash",
        "commit_title": commit_title[:240],     # GitHub caps these at 240/4096
        "commit_message": commit_message[:4000],
    }

    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return {"merged": False, "sha": None,
                "error": f"network error calling merge API: {e}"}

    if resp.status_code == 200:
        data = resp.json()
        return {"merged": True, "sha": data.get("sha"), "error": None}

    # GitHub merge failures we want clean messages for:
    #   405 — PR not mergeable (conflicts, failing required checks, or branch
    #         protection that requires reviews the bot doesn't have)
    #   409 — head SHA changed since the merge was attempted
    #   422 — validation error (e.g. PR already closed)
    try:
        err = resp.json().get("message", resp.text[:200])
    except Exception:
        err = resp.text[:200]
    return {"merged": False, "sha": None,
            "error": f"HTTP {resp.status_code}: {err}"}


def merge_prs_for_state(state: dict) -> dict:
    """Standalone helper — same logic as the merge_prs graph node, callable
    directly from anywhere (e.g. api/main.py during the Phase 6→7 transition).

    Takes a state-shaped dict (or anything with .get('approved'), .get('pr_urls'),
    .get('status'), .get('requirement')) and returns a new state-shaped dict
    with 'merged_shas', 'merge_errors', and a possibly-updated 'status'.

    Returned status values:
        'READY_FOR_DEPLOYMENT'   — merges all succeeded (or nothing to merge)
        'MERGE_FAILED'           — at least one PR failed to merge
        'PR_REJECTED'            — input had approved=False; nothing done
    """
    # Reject path
    if state.get("status") == "PR_REJECTED" or not state.get("approved", True):
        # default approved=True so callers that don't pass it (like the API
        # transition handler, which only runs on approval anyway) still work
        if state.get("status") != "PR_REJECTED":
            # Caller passed approved=False explicitly
            return {**state, "merged_shas": {}, "merge_errors": [],
                    "status": "PR_REJECTED"}
        return {**state, "merged_shas": {}, "merge_errors": []}

    # New-project path
    if state.get("status") == "PR_SKIPPED_NEW_PROJECT":
        print("\n[merge_prs] New project — code already on default branch, "
              "no PRs to merge")
        return {
            **state, "merged_shas": {}, "merge_errors": [],
            "status": "READY_FOR_DEPLOYMENT",
        }

    pr_urls = state.get("pr_urls") or []
    if not pr_urls:
        print("\n[merge_prs] ⚠ No PR URLs — nothing to merge")
        return {
            **state, "merged_shas": {}, "merge_errors": [],
            "status": "READY_FOR_DEPLOYMENT",
        }

    print(f"\n[merge_prs] Merging {len(pr_urls)} approved PR(s)...")
    merged_shas: dict = {}
    failures: list = []
    requirement = (state.get("requirement") or "")[:80]

    for pr_url in pr_urls:
        owner, repo, pr_number = _parse_pr_url(pr_url)
        if owner is None:
            failures.append(f"{pr_url}: not recognizable as a /pull/<n> URL")
            print(f"  ⚠ Skipped non-PR URL: {pr_url}")
            continue

        result = _merge_one_pr(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            commit_title=f"feat: {requirement}",
            commit_message=(
                f"Merged via SDLC-V2 Phase 6.\n\n"
                f"Requirement: {state.get('requirement', '')}\n"
                f"PR: {pr_url}"
            ),
        )

        if result["merged"]:
            merged_shas[repo] = result["sha"]
            short = (result["sha"] or "?")[:7]
            print(f"  ✅ Merged {owner}/{repo} PR #{pr_number} → {short}")
        else:
            failures.append(f"{owner}/{repo} PR #{pr_number}: {result['error']}")
            print(f"  ❌ Merge failed for {owner}/{repo} PR #{pr_number}: "
                  f"{result['error']}")

    if failures:
        print(f"\n[merge_prs] ⚠ {len(failures)} merge failure(s) — Phase 7 must NOT run")
        return {
            **state, "merged_shas": merged_shas, "merge_errors": failures,
            "status": "MERGE_FAILED",
        }

    print(f"\n[merge_prs] ✅ All PRs merged. SHAs to deploy:")
    for repo, sha in merged_shas.items():
        print(f"    {repo} → {(sha or '?')[:7]}")

    return {
        **state, "merged_shas": merged_shas, "merge_errors": [],
        "status": "READY_FOR_DEPLOYMENT",
    }


# -----------------------------------------
# merge_prs — the graph node (thin wrapper over the standalone helper)
# -----------------------------------------

def merge_prs(state: DeliveryState) -> DeliveryState:
    """Graph node that merges approved PRs. Delegates to merge_prs_for_state
    so the same logic is reachable both via the graph and directly from
    api/main.py during the Phase 6 → Phase 7 transition.
    """
    return merge_prs_for_state(dict(state))  # type: ignore[return-value]


def route_after_approval(state: DeliveryState) -> str:
    if state["status"] == "APPROVED_FOR_DEPLOYMENT":
        return "approved"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_delivery_graph():
    builder = StateGraph(DeliveryState)

    builder.add_node("push_code", push_code)
    builder.add_node("create_pr", create_pr)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)
    builder.add_node("merge_prs", merge_prs)                  # ← NEW

    builder.set_entry_point("push_code")
    builder.add_edge("push_code", "create_pr")
    builder.add_edge("create_pr", "human_approval_gate")
    builder.add_edge("human_approval_gate", "process_approval")

    # Approved → run the merge step. Rejected → end immediately (no merge).
    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {
            "approved": "merge_prs",                          # ← was END
            "rejected": END
        }
    )

    # After merge (success OR failure), Phase 6 is done. Phase 7 reads
    # state.status to decide whether to proceed.
    builder.add_edge("merge_prs", END)                        # ← NEW

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]
    )


# -----------------------------------------
# Run
# -----------------------------------------

def run_delivery(
        requirement: str,
        generated_changes: list,
        test_files: list,
        repo_url: str,
        branch_name: str,
        thread_id: str = "thread-delivery"
) -> dict:
    graph = build_delivery_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = DeliveryState(
        requirement=requirement,
        generated_changes=generated_changes,
        test_files=test_files,
        branch_name=branch_name,
        repo_url=repo_url,
        pr_urls=[],
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 6 — Delivery ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\nStatus after interrupt: {result['status']}")
    for url in result.get("pr_urls", []):
        print(f"  PR: {url}")

    return graph, config, result


def resume_delivery(graph, config, approved: bool, feedback: str = "") -> dict:
    print(f"\n--- Resuming Phase 6 (approved={approved}) ---")
    result = graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config
    )
    print(f"Final status: {result['status']}")
    return result


if __name__ == "__main__":
    # Mock generated changes from Phase 4
    mock_changes = [
        {
            "file_path": "app/models.py",
            "content": """from pydantic import BaseModel
from typing import Optional

class LeaveRequest(BaseModel):
    employee_name: str
    leave_type: str
    start_date: str
    end_date: str
    reason: str
    balance: int = 20

class LeaveStatus(BaseModel):
    leave_id: str
    status: str
    leave_balance: Optional[int] = None

class Employee(BaseModel):
    employee_id: str
    name: str
    leave_balance: int = 20
""",
            "change_summary": "Added Employee model and balance fields",
            "new_symbols_added": ["Employee"],
            "existing_symbols_modified": ["LeaveRequest", "LeaveStatus"]
        }
    ]

    mock_test_files = [
        {
            "test_file_path": "tests/test_app_models.py",
            "content": """import pytest
from app.models import LeaveRequest, LeaveStatus, Employee

def test_leave_request_default_balance():
    req = LeaveRequest(
        employee_name="John",
        leave_type="sick",
        start_date="2026-01-01",
        end_date="2026-01-02",
        reason="sick"
    )
    assert req.balance == 20

def test_employee_default_balance():
    emp = Employee(employee_id="E001", name="John")
    assert emp.leave_balance == 20
"""
        }
    ]

    requirement = "Add leave balance tracker. Each employee gets 20 days per year."

    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        raise Exception("GITHUB_TOKEN is missing")

    repo_url = (
        f"https://x-access-token:{github_token}@github.com/"
        "AkashW45/leave-mgmt-backend.git"
    )

    graph6, config6, result6 = run_delivery(
        requirement=requirement,
        generated_changes=mock_changes,
        test_files=mock_test_files,
        repo_url=repo_url,
        branch_name="feature/leave-balance-test",
        thread_id="test-delivery-1"
    )

    print(f"\nStatus after interrupt: {result6['status']}")
    for url in result6.get("pr_urls", []):
        print(f"  PR: {url}")

    print("\n--- Simulating Human PR Approval ---")
    final = resume_delivery(graph6, config6, approved=True, feedback="Looks good")

    print(f"\n✅ Phase 6 Test Complete")
    print(f"Status: {final['status']}")
    for url in final.get("pr_urls", []):
        print(f"PR: {url}")