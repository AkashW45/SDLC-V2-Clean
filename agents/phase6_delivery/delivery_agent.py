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

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")


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
    """

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
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/pulls"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "title": title,
        "head": branch_name,
        "base": "main",
        "body": body
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        return response.json()

    # PR already exists
    if response.status_code == 422 and "already exists" in response.text:
        pr_list_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/pulls?head={GITHUB_OWNER}:{branch_name}&state=open"
        pr_resp = requests.get(pr_list_url, headers=headers)
        prs = pr_resp.json()
        if prs:
            return prs[0]

    raise Exception(f"GitHub PR creation failed: {response.text}")


# -----------------------------------------
# Nodes
# -----------------------------------------

def push_code(state: DeliveryState) -> DeliveryState:
    """Push all generated changes to GitHub branch."""
    print("\n[Phase 6] Pushing code to GitHub...")

    branch_name = state["branch_name"]
    repo_url = state["repo_url"]

    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        raise Exception("GITHUB_TOKEN is missing")

    # Inject auth into GitHub HTTPS URL
    if repo_url.startswith("https://github.com/"):
        repo_url = repo_url.replace(
            "https://github.com/",
            f"https://x-access-token:{github_token}@github.com/"
        )

    # -----------------------------------------
    # Greenfield repo creation
    # -----------------------------------------

    # Extract repo name
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    # Check if repo exists
    repo_check_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}"

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }

    repo_check = requests.get(repo_check_url, headers=headers)
    # Repo does not exist → create it (and wait for auto_init commit inside).
    if repo_check.status_code == 404:
        print(f"  ℹ️  Repo '{repo_name}' does not exist. Creating...")
        try:
            create_github_repo(repo_name=repo_name)
            print(f"  ✅ Repository created: {repo_name}")
        except Exception as e:
            print(f"  ❌ Repo creation failed: {e}")
            return {**state, "status": "PUSH_FAILED", "error": str(e)}
    elif repo_check.status_code not in (200, 301):
        # 401/403 here almost always means the token lacks repo scope. Fail
        # loudly instead of silently sailing into a push that can't work.
        msg = (f"Cannot access repo '{GITHUB_OWNER}/{repo_name}': "
               f"HTTP {repo_check.status_code} - {repo_check.text[:200]}")
        print(f"  ❌ {msg}")
        return {**state, "status": "PUSH_FAILED", "error": msg}

    # Prepare all files to push
    all_files = []

    # Generated code changes
    for change in state.get("generated_changes", []):
        all_files.append({
            "file_path": change["file_path"],
            "content": change["content"]
        })

    # Generated test files
    for test_file in state.get("test_files", []):
        if test_file.get("content"):
            all_files.append({
                "file_path": test_file["test_file_path"],
                "content": test_file["content"]
            })

    # Nothing to push is itself a failure for a greenfield project — an empty
    # repo is exactly the symptom we're trying to eliminate.
    if not all_files:
        msg = "No files to push (generated_changes and test_files are empty)"
        print(f"  ❌ {msg}")
        return {**state, "status": "PUSH_FAILED", "error": msg}

    print(f"  Pushing {len(all_files)} files to branch: {branch_name}")

    push_result = push_files_to_branch(
        repo_url=repo_url,
        branch_name=branch_name,
        files=all_files,
        commit_message=f"feat: AI-generated changes — {state['requirement'][:60]}"
    )
    print(f"  Push result: {push_result['status']}")

    # Only PUSH_SUCCESS counts. NO_CHANGES and PUSH_FAILED must NOT advance the
    # pipeline as if code landed — that was the bug that hid empty repos.
    if push_result["status"] == "PUSH_SUCCESS":
        print(f"  ✅ Pushed {push_result.get('files_pushed', 0)} files")
        return {**state, "status": "CODE_PUSHED"}

    err = push_result.get("error") or push_result["status"]
    print(f"  ❌ Push did not succeed: {err}")
    return {**state, "status": "PUSH_FAILED", "error": err}


def create_pr(state: DeliveryState) -> DeliveryState:
    """Create GitHub PR for the changes."""
    print("\n[Phase 6] Creating GitHub PR...")

    if state["status"] == "PUSH_FAILED":
        return {**state, "status": "PR_SKIPPED"}

    branch_name = state["branch_name"]
    repo_url = state["repo_url"]

    # Extract repo name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    # New project: code was pushed directly to default branch — no PR needed.
    # GitHub can't open a PR from main → main, so we record the repo URL instead.
    # But only AFTER confirming the branch really has our commit — otherwise we'd
    # hand back a link to an empty/nonexistent repo (the original bug).
    if branch_name in ("main", "master"):
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        verify_url = (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/"
            f"branches/{branch_name}"
        )
        verify = requests.get(verify_url, headers=headers)
        if verify.status_code != 200:
            msg = (f"Push verification failed — branch '{branch_name}' not found "
                   f"on {GITHUB_OWNER}/{repo_name} (HTTP {verify.status_code}). "
                   f"Code did not land.")
            print(f"  ❌ {msg}")
            return {**state, "status": "PUSH_FAILED", "error": msg}

        push_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}"
        sha = (verify.json().get("commit", {}).get("sha") or "")[:7]
        print(f"  ℹ️  New project — code pushed to {branch_name} ({sha}), no PR needed.")
        print(f"  Repo: {push_url}")
        return {**state, "pr_urls": [push_url], "status": "PR_SKIPPED_NEW_PROJECT"}

    # Build PR body
    changes_summary = "\n".join([
        f"- `{c['file_path']}`: {c.get('change_summary', '')[:80]}"
        for c in state["generated_changes"]
    ])

    pr_body = f"""## AI-Generated Changes

**Requirement:** {state['requirement']}

## Files Changed
{changes_summary}

## Tests Added
{len(state.get('test_files', []))} test file(s) generated

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
            body=pr_body
        )

        pr_url = pr.get("html_url", "")
        print(f"  ✅ PR created: {pr_url}")

        return {
            **state,
            "pr_urls": [pr_url],
            "status": "PR_CREATED"
        }

    except Exception as e:
        print(f"  ❌ PR creation failed: {e}")
        return {**state, "status": "PR_FAILED"}


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