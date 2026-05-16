"""
RepoWorkspaceManager — Production helper for brownfield code access.

Responsibilities:
1. Resolve repo_name → local clone path (auto-clones if missing).
2. Read file contents with two-tier fallback: local clone → GitHub API.
3. Provide a stable WORKSPACE_ROOT directory layout.

Used by Phase 4 (load_existing_code) and Phase 6 (push_files_to_branch).
"""

import os
import shutil
import subprocess
import tempfile
import base64
import logging
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Default workspace root — override via env var
WORKSPACE_ROOT = os.path.abspath(os.getenv("WORKSPACE_ROOT", "./repos"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER", "AkashW45")

from agents.github_auth import get_github_headers, get_clone_url


def get_repo_local_path(repo_name: str) -> str:
    """Return the canonical local path for a repo within WORKSPACE_ROOT."""
    return os.path.join(WORKSPACE_ROOT, repo_name)


def ensure_repo_cloned(repo_name: str, repo_url: Optional[str] = None,
                      branch: str = "main") -> Optional[str]:
    """
    Ensure repo is cloned locally. Returns the local path, or None on failure.

    Order of operations:
    1. If WORKSPACE_ROOT/<repo_name> exists and is a git repo → return it
    2. Else clone from repo_url (or constructed GitHub URL)
    3. On clone failure → return None (caller should fall back to GitHub API)
    """
    local = get_repo_local_path(repo_name)
    git_dir = os.path.join(local, ".git")

    if os.path.isdir(git_dir):
        logger.info(f"[RepoWorkspace] {repo_name} already cloned at {local}")
        return local

    if os.path.isdir(local) and not os.path.isdir(git_dir):
        # Directory exists but not a git repo — wipe and re-clone
        logger.warning(f"[RepoWorkspace] {local} exists but is not a git repo — removing")
        shutil.rmtree(local, ignore_errors=True)

    os.makedirs(WORKSPACE_ROOT, exist_ok=True)

    if not repo_url:
       repo_url = get_clone_url(repo_name)

    try:
        logger.info(f"[RepoWorkspace] cloning {repo_name} from GitHub...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo_url, local],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"[RepoWorkspace] clone failed (branch={branch}): {result.stderr.strip()}")
            # Retry without branch (some repos use master, some have no default yet)
            shutil.rmtree(local, ignore_errors=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, local],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.error(f"[RepoWorkspace] clone failed entirely: {result.stderr.strip()}")
                return None
        logger.info(f"[RepoWorkspace] cloned {repo_name} → {local}")
        return local
    except subprocess.TimeoutExpired:
        logger.error(f"[RepoWorkspace] clone timed out for {repo_name}")
        return None
    except Exception as e:
        logger.error(f"[RepoWorkspace] clone exception: {e}")
        return None


def read_file(repo_name: str, file_path: str,
              repo_url: Optional[str] = None) -> str:
    """
    Read a file's content. Two-tier fallback:
    1. Local clone (fast, no API quota)
    2. GitHub Contents API (network, slower, counts against rate limit)

    Returns '' if both fail.
    """
    # Tier 1: local clone
    local = ensure_repo_cloned(repo_name, repo_url=repo_url)
    if local:
        full = os.path.join(local, file_path.replace("\\", os.sep).replace("/", os.sep))
        if os.path.exists(full):
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"[RepoWorkspace] local read failed for {file_path}: {e}")

    # Tier 2: GitHub Contents API
    if not GITHUB_TOKEN:
        logger.warning("[RepoWorkspace] no GITHUB_TOKEN — cannot fall back to API")
        return ""

    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}/contents/{file_path}"
    try:
        r = requests.get(api_url, headers=get_github_headers(), timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        logger.warning(f"[RepoWorkspace] GitHub API returned {r.status_code} for {file_path}")
    except Exception as e:
        logger.warning(f"[RepoWorkspace] GitHub API call failed: {e}")
    return ""