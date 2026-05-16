"""
GitHub App authentication module.

Replaces the long-lived PAT (GITHUB_TOKEN) with short-lived installation tokens
generated via JWT signing of the GitHub App's private key.

Why this matters:
- PATs are long-lived; if leaked, an attacker has indefinite access.
- Installation tokens auto-expire in 1 hour and are scoped to the installed repos.
- GitHub Apps have higher rate limits (15k/hr vs 5k/hr for PATs).

Usage:
    from agents.github_auth import get_github_headers, get_installation_token

    headers = get_github_headers()  # cached, auto-refreshes
    r = requests.get(f"{GH_API}/repos/...", headers=headers)
"""

import os
import time
import logging
import threading
from typing import Optional
import requests
import jwt
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GH_API = "https://api.github.com"

GITHUB_APP_ID = os.getenv("GITHUB_APP_ID", "")
GITHUB_APP_INSTALLATION_ID = os.getenv("GITHUB_APP_INSTALLATION_ID", "")
GITHUB_APP_PRIVATE_KEY_PATH = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH", "./github_app.pem")

# Legacy PAT fallback (used only if GitHub App env vars are missing)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Token cache
_token_cache: dict = {"token": None, "expires_at": 0}
_cache_lock = threading.Lock()


def _load_private_key() -> Optional[str]:
    """Read the App's private key PEM file."""
    if not os.path.exists(GITHUB_APP_PRIVATE_KEY_PATH):
        logger.error(f"[GitHubAuth] Private key not found at {GITHUB_APP_PRIVATE_KEY_PATH}")
        return None
    try:
        with open(GITHUB_APP_PRIVATE_KEY_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[GitHubAuth] Failed to read private key: {e}")
        return None


def _generate_jwt() -> Optional[str]:
    """
    Create a short-lived JWT signed with the App's private key.
    Valid for ~10 minutes per GitHub's spec.
    Used ONLY to fetch installation tokens — not for direct API calls.
    """
    private_key = _load_private_key()
    if not private_key or not GITHUB_APP_ID:
        return None

    now = int(time.time())
    payload = {
        "iat": now - 60,        # issued 60s ago (clock-skew tolerance)
        "exp": now + 600,       # expires in 10 minutes
        "iss": GITHUB_APP_ID,   # the App ID
    }
    try:
        return jwt.encode(payload, private_key, algorithm="RS256")
    except Exception as e:
        logger.error(f"[GitHubAuth] JWT encode failed: {e}")
        return None


def get_installation_token(force_refresh: bool = False) -> Optional[str]:
    """
    Returns a cached installation access token, refreshing if expired or near expiry.
    Installation tokens are valid for ~1 hour. We refresh at 50 minutes for safety.
    """
    with _cache_lock:
        now = time.time()
        if not force_refresh and _token_cache["token"] and _token_cache["expires_at"] > now + 600:
            return _token_cache["token"]

        if not GITHUB_APP_INSTALLATION_ID:
            logger.warning("[GitHubAuth] GITHUB_APP_INSTALLATION_ID not set")
            return None

        app_jwt = _generate_jwt()
        if not app_jwt:
            return None

        try:
            r = requests.post(
                f"{GH_API}/app/installations/{GITHUB_APP_INSTALLATION_ID}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=10,
            )
        except Exception as e:
            logger.error(f"[GitHubAuth] Token request failed: {e}")
            return None

        if r.status_code != 201:
            logger.error(f"[GitHubAuth] Installation token error {r.status_code}: {r.text[:300]}")
            return None

        data = r.json()
        _token_cache["token"] = data["token"]
        # Parse 'expires_at' but be conservative: assume ~55 min validity
        _token_cache["expires_at"] = now + 55 * 60
        logger.info("[GitHubAuth] Got fresh installation token")
        return _token_cache["token"]


def get_github_headers() -> dict:
    """
    Returns headers for GitHub API calls. Prefers App auth, falls back to PAT.
    Use this everywhere instead of building headers manually.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Try GitHub App first (production path)
    if GITHUB_APP_ID and GITHUB_APP_INSTALLATION_ID:
        token = get_installation_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            return headers
        logger.warning("[GitHubAuth] App auth failed, falling back to PAT")

    # Fallback to PAT (legacy)
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        return headers

    logger.error("[GitHubAuth] No GitHub auth configured (neither App nor PAT)")
    return headers


def get_clone_url(repo_name: str, owner: Optional[str] = None) -> str:
    """
    Returns an authenticated clone URL.
    For GitHub App, uses `x-access-token:<token>@github.com/...` format.
    """
    owner = owner or os.getenv("GITHUB_REPO_OWNER", "")
    token = get_installation_token() if GITHUB_APP_ID else GITHUB_TOKEN
    if token and (GITHUB_APP_ID and GITHUB_APP_INSTALLATION_ID):
        return f"https://x-access-token:{token}@github.com/{owner}/{repo_name}.git"
    elif token:
        return f"https://{token}@github.com/{owner}/{repo_name}.git"
    return f"https://github.com/{owner}/{repo_name}.git"