"""
agents/pr_manager.py
Idempotent PR creation. The same (asp_id + artifact_id + agent_version) will
NEVER create two PRs — the second call returns the existing PR.

Flow:
  1. Compute unique_request_id = sha256(asp_id + artifact_id + agent_version)
  2. Check pr_registry — if exists, return that PR (idempotent short-circuit)
  3. Ensure repo exists (create via GitHub API if new project)
  4. Create branch from default branch
  5. Commit all artifact files to the branch
  6. Open PR (never auto-merge)
  7. Persist to pr_registry

Never auto-merges. Never force-pushes.
"""
import os
import base64
import hashlib
import requests
from dotenv import load_dotenv

from agents.stage2_store import get_pr_by_request_id, save_pr, audit

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_ORG = os.getenv("GITHUB_ORG", "").strip()
# Effective owner: the org when set (repos created via /orgs/{org}/repos),
# else the token owner's personal account (/user/repos).
GITHUB_OWNER = GITHUB_ORG or os.getenv("GITHUB_REPO_OWNER", "AkashW45")
AGENT_VERSION = os.getenv("AGENT_VERSION", "v2.0")

from agents.github_auth import get_github_headers


def _request_id(asp_id: str, artifact_id: str) -> str:
    raw = f"{asp_id}:{artifact_id}:{AGENT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ────────────────────────────────────────────────────────────────────
# GitHub primitives
# ────────────────────────────────────────────────────────────────────
def _repo_exists(repo_name: str) -> bool:
    r = requests.get(f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}", headers=get_github_headers(), timeout=10)
    return r.status_code == 200


def _create_repo(repo_name: str, description: str) -> bool:
    create_url = (f"{GH_API}/orgs/{GITHUB_ORG}/repos"
                  if GITHUB_ORG else f"{GH_API}/user/repos")
    r = requests.post(
        create_url,
        headers=get_github_headers(),
        json={
            "name": repo_name,
            "description": description[:200],
            "private": False,
            "auto_init": True,
        },
        timeout=15,
    )
    if r.status_code != 201:
        print(f"[pr_manager] repo create failed at {create_url}: "
              f"HTTP {r.status_code} - {r.text[:200]}")
    return r.status_code == 201


def _get_default_branch(repo_name: str) -> str:
    r = requests.get(f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}", headers=get_github_headers(), timeout=10)
    if r.status_code == 200:
        return r.json().get("default_branch", "main")
    return "main"


def _get_branch_sha(repo_name: str, branch: str) -> str:
    r = requests.get(
        f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/git/ref/heads/{branch}",
        headers=get_github_headers(),
        timeout=10,
    )
    if r.status_code == 200:
        return r.json()["object"]["sha"]
    return None


def _branch_exists(repo_name: str, branch: str) -> bool:
    return _get_branch_sha(repo_name, branch) is not None


def _create_branch(repo_name: str, new_branch: str, from_sha: str) -> bool:
    r = requests.post(
        f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/git/refs",
        headers=get_github_headers(),
        json={"ref": f"refs/heads/{new_branch}", "sha": from_sha},
        timeout=10,
    )
    return r.status_code in (200, 201)


def _get_file_sha(repo_name: str, path: str, branch: str) -> str:
    """Return the blob sha of an existing file, or None if it doesn't exist."""
    r = requests.get(
        f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/contents/{path}?ref={branch}",
        headers=get_github_headers(),
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def _commit_file(repo_name: str, path: str, content: str, branch: str,
                 message: str) -> bool:
    """Create or update a single file on a branch."""
    existing_sha = _get_file_sha(repo_name, path, branch)
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha  # required to update an existing file

    r = requests.put(
        f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/contents/{path}",
        headers=get_github_headers(),
        json=payload,
        timeout=15,
    )
    return r.status_code in (200, 201)


def _open_pr(repo_name: str, head_branch: str, base_branch: str,
             title: str, body: str) -> dict:
    r = requests.post(
        f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/pulls",
        headers=get_github_headers(),
        json={"title": title, "head": head_branch, "base": base_branch, "body": body},
        timeout=15,
    )
    if r.status_code == 201:
        data = r.json()
        return {"pr_url": data["html_url"], "pr_number": data["number"]}
    # PR may already exist for this branch pair
    if r.status_code == 422:
        existing = requests.get(
            f"{GH_API}/repos/{GITHUB_OWNER}/{repo_name}/pulls?head={GITHUB_OWNER}:{head_branch}&state=open",
            headers=get_github_headers(),
            timeout=10,
        )
        if existing.status_code == 200 and existing.json():
            d = existing.json()[0]
            return {"pr_url": d["html_url"], "pr_number": d["number"]}
    return {"pr_url": None, "pr_number": None, "error": r.text[:300]}


# ────────────────────────────────────────────────────────────────────
# Main entry point — idempotent
# ────────────────────────────────────────────────────────────────────
def create_pr_for_artifact(artifact: dict, asp: dict, thread_id: str,
                           target_repo: dict) -> dict:
    """
    Idempotently create a PR for an accepted CODE/PATCH artifact.

    target_repo: {"name": "...", "url": "...", "exists": bool, "type": "backend"}

    Returns:
      {
        "status": "created|existing|error",
        "pr_url": "...",
        "pr_number": int,
        "branch": "...",
        "unique_request_id": "...",
        "error": "..."  (only on error)
      }
    """
    asp_id = asp.get("asp_id", "no-asp")
    artifact_id = artifact.get("artifact_id", "no-artifact")
    unique_request_id = _request_id(asp_id, artifact_id)

    # ── IDEMPOTENCY CHECK ──
    existing = get_pr_by_request_id(unique_request_id)
    if existing:
        print(f"  [PR Manager] Idempotent hit — PR already exists: {existing['pr_url']}")
        return {
            "status": "existing",
            "pr_url": existing["pr_url"],
            "pr_number": existing["pr_number"],
            "branch": existing["branch"],
            "unique_request_id": unique_request_id,
        }

    repo_name = target_repo["name"]
    is_new = not target_repo.get("exists", True) or asp.get("_is_new_project", False)

    # ── ENSURE REPO EXISTS ──
    if not _repo_exists(repo_name):
        print(f"  [PR Manager] Creating repo: {repo_name}")
        if not _create_repo(repo_name, asp.get("user_input", "")):
            return {"status": "error", "error": f"Failed to create repo {repo_name}",
                    "unique_request_id": unique_request_id}
        audit(thread_id, "pr_manager", "REPO_CREATED", {"repo": repo_name})

    # ── BRANCH ──
    default_branch = _get_default_branch(repo_name)
    # New projects push straight to default; existing projects get a feature branch
    if is_new:
        target_branch = default_branch
    else:
        target_branch = f"feature/{thread_id}-{artifact_id[:8]}"
        if not _branch_exists(repo_name, target_branch):
            base_sha = _get_branch_sha(repo_name, default_branch)
            if not base_sha:
                return {"status": "error", "error": f"Cannot read default branch of {repo_name}",
                        "unique_request_id": unique_request_id}
            if not _create_branch(repo_name, target_branch, base_sha):
                return {"status": "error", "error": f"Failed to create branch {target_branch}",
                        "unique_request_id": unique_request_id}

    # ── COMMIT FILES ──
    files = artifact.get("files", [])
    if not files:
        return {"status": "error", "error": "Artifact has no files to commit",
                "unique_request_id": unique_request_id}

    committed = 0
    failed_files = []
    commit_msg = f"[{artifact.get('type', 'CODE')}] {artifact.get('title', 'Generated by SDLC-V2')}"
    for f in files:
        path = f.get("path") or f.get("file_path")
        content = f.get("content", "")
        if not path:
            continue
        if _commit_file(repo_name, path, content, target_branch, commit_msg):
            committed += 1
        else:
            failed_files.append(path)

    if committed == 0:
        return {"status": "error", "error": f"Failed to commit any files. Failed: {failed_files}",
                "unique_request_id": unique_request_id}

    if failed_files:
        print(f"  [PR Manager] Warning — some files failed: {failed_files}")

    # ── OPEN PR (unless pushed straight to default branch for new project) ──
    if is_new and target_branch == default_branch:
        # New project — code is on main, no PR needed, but we still register it
        pr_url = f"https://github.com/{GITHUB_OWNER}/{repo_name}"
        pr_number = 0
        save_pr(unique_request_id, artifact_id, thread_id, repo_name, target_branch, pr_url, pr_number)
        audit(thread_id, "pr_manager", "PUSHED_TO_MAIN",
              {"repo": repo_name, "files": committed})
        return {
            "status": "created",
            "pr_url": pr_url,
            "pr_number": 0,
            "branch": target_branch,
            "files_committed": committed,
            "note": "New project — code pushed directly to default branch, no PR",
            "unique_request_id": unique_request_id,
        }

    pr_title = f"[SDLC-V2] {artifact.get('title', 'Generated changes')}"
    pr_body = _build_pr_body(artifact, asp)
    pr_result = _open_pr(repo_name, target_branch, default_branch, pr_title, pr_body)

    if not pr_result.get("pr_url"):
        return {"status": "error", "error": f"PR creation failed: {pr_result.get('error', 'unknown')}",
                "branch": target_branch, "unique_request_id": unique_request_id}

    # ── PERSIST ──
    save_pr(unique_request_id, artifact_id, thread_id, repo_name,
            target_branch, pr_result["pr_url"], pr_result["pr_number"])
    audit(thread_id, "pr_manager", "PR_CREATED", {
        "repo": repo_name, "pr_url": pr_result["pr_url"],
        "files": committed, "branch": target_branch,
    })

    return {
        "status": "created",
        "pr_url": pr_result["pr_url"],
        "pr_number": pr_result["pr_number"],
        "branch": target_branch,
        "files_committed": committed,
        "unique_request_id": unique_request_id,
    }


def _build_pr_body(artifact: dict, asp: dict) -> str:
    lines = [
        "## Generated by SDLC-V2",
        "",
        f"**Artifact type:** {artifact.get('type', 'CODE')}",
        f"**Category:** {artifact.get('category', 'mvp')}",
        f"**Units estimate:** {artifact.get('units_estimate', 'N/A')}",
        f"**Confidence:** {artifact.get('confidence', 'N/A')}",
        f"**Evidence resolved:** {artifact.get('evidence_resolved', 'N/A')} ({artifact.get('evidence_note', '')})",
        "",
        f"**Depth level:** {asp.get('depth_level', 'N/A')}",
        f"**Policy mode:** {asp.get('policy_mode', 'N/A')}",
        "",
        "### Justification",
        artifact.get("justification", "No justification provided"),
        "",
        "### Traces to",
    ]
    for t in artifact.get("traces_to", []):
        lines.append(f"- {t}")
    lines += [
        "",
        "### Files in this PR",
    ]
    for f in artifact.get("files", []):
        path = f.get("path") or f.get("file_path", "unknown")
        lines.append(f"- `{path}`")
    lines += [
        "",
        "---",
        "_This PR was generated automatically. **Do not auto-merge** — human review required._",
    ]
    return "\n".join(lines)