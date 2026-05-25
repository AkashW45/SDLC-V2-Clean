"""
Deployment executor — does the real work of Phase 7.

Responsibilities:
  - verify_clis(): check required CLI tools are installed
  - fetch_repos(): git clone (or pull) each repo into a workspace
  - build_image(): docker build per repo using its DockerfileConfig
  - trigger_deploy(): invoke the configured deploy target
                      (compose | kubernetes | webhook | shell | dry-run)

Every action returns a structured result dict so the agent can record
exactly what happened, and every action respects `config.dry_run` —
in dry-run mode commands are logged but not executed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.deployment_config import (
    DeploymentConfig,
    DeployTarget,
    EnvironmentDeploy,
    RegistryConfig,
    RepoConfig,
    SmokeTest,
)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    # `datetime.utcnow()` is deprecated in 3.12+; use a tz-aware UTC stamp.
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _ok(action: str, **extra: Any) -> Dict[str, Any]:
    return {"action": action, "status": "SUCCESS", "at": _now(), **extra}


def _fail(action: str, error: str, **extra: Any) -> Dict[str, Any]:
    return {"action": action, "status": "FAILED", "at": _now(), "error": error, **extra}


# ---------------------------------------------------------------------------
# Shell runner
# ---------------------------------------------------------------------------

def _run(
        cmd: List[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        stdin: Optional[str] = None,
        timeout: int = 600,
        dry_run: bool = False,
        check: bool = True,
        secret: bool = False,
) -> Dict[str, Any]:
    """Run a shell command and return a structured result.

    `dry_run=True` means: log what would run, return a synthetic success.
    `check=False` returns the failure dict instead of raising.
    `stdin` is piped to the command's stdin (use for password-stdin).
    `secret=True` redacts the printed command — use when args carry a token.
    """
    import sys
    is_win = sys.platform == "win32"

    
    pretty = "<redacted>" if secret else " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(f"  [dry-run] would run: {pretty} (cwd={cwd or os.getcwd()})")
        return {"cmd": pretty, "returncode": 0, "stdout": "", "stderr": "", "dry_run": True}

    print(f"  $ {pretty}")
    run_cmd = " ".join(cmd) if is_win else cmd
    try:
        proc = subprocess.run(
            run_cmd,
            shell=is_win,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        result = {"cmd": pretty, "returncode": 127, "stdout": "",
                  "stderr": f"command not found: {e}", "dry_run": False}
        if check:
            raise RuntimeError(f"Command failed: {pretty}\n{result['stderr']}")
        return result
    except subprocess.TimeoutExpired as e:
        result = {"cmd": pretty, "returncode": 124, "stdout": "",
                  "stderr": f"timeout after {e.timeout}s", "dry_run": False}
        if check:
            raise RuntimeError(f"Command timed out: {pretty}")
        return result

    result = {
        "cmd": pretty,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip()[-2000:],   # cap for state size
        "stderr": (proc.stderr or "").strip()[-2000:],
        "dry_run": False,
    }
    # Always report the outcome so a failing command is never silent in the log.
    # (Previously a failed command with check=False returned a dict whose error
    # was never printed — making the trace "cut off" with no visible cause.)
    if proc.returncode == 0:
        print(f"    ✓ exit 0")
    else:
        print(f"    ✗ exit {proc.returncode}")
        if result["stderr"]:
            # Show the tail of stderr — this is the actual failure reason.
            print(f"    ↳ stderr: {result['stderr'][-800:]}")
    if proc.returncode != 0 and check:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {pretty}\n"
            f"stderr: {result['stderr']}"
        )
    return result


# ---------------------------------------------------------------------------
# Step 1 — Verify CLIs
# ---------------------------------------------------------------------------

def verify_clis(config: DeploymentConfig) -> Dict[str, Any]:
    """Check that each required CLI tool is on PATH.

    Auto-install behavior:
      - `config.cli_auto_install` must be True (opt-in)
      - The CLI's `install_command` must be set (no command = no auto-install)
      - We run the install_command, then re-check presence

    If auto-install is off OR install_command isn't set, missing required CLIs
    fail the phase with their install_hint. Auto-installing system packages on
    a shared host can be destructive, so the operator has to opt in.
    """
    print("\n[executor] Verifying required CLIs...")
    checks: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    auto_installed: List[str] = []
    auto_install_failures: List[Dict[str, str]] = []

    for req in config.required_clis:
        present = shutil.which(req.name) is not None

        # Auto-install attempt
        if (not present and req.required and config.cli_auto_install
                and req.install_command):
            print(f"  ⏳ {req.name}: missing — attempting auto-install")
            print(f"     (cli_auto_install=true; running: {req.install_command})")
            # sh -c so install commands can use shell features (pipes, &&)
            install_result = _run(
                ["sh", "-c", req.install_command],
                dry_run=config.dry_run, check=False, timeout=900,
            )
            if install_result["returncode"] == 0 or config.dry_run:
                present = shutil.which(req.name) is not None or config.dry_run
                if present:
                    auto_installed.append(req.name)
                    print(f"  ✅ {req.name}: auto-installed")
                else:
                    print(f"  ⚠ {req.name}: install command exited 0 but "
                          f"binary still not on PATH (PATH update needed?)")
            else:
                auto_install_failures.append({
                    "cli": req.name,
                    "stderr": install_result["stderr"][-500:],
                })
                print(f"  ❌ {req.name}: auto-install failed — "
                      f"{install_result['stderr'][-200:]}")

        check: Dict[str, Any] = {
            "name": req.name,
            "present": present,
            "required": req.required,
            "auto_installed": req.name in auto_installed,
        }
        if present and req.version_command and not config.dry_run:
            v = _run(shlex.split(req.version_command),
                     dry_run=False, check=False, timeout=10)
            check["version"] = v["stdout"].splitlines()[0] if v["stdout"] else ""
        if not present:
            check["install_hint"] = req.install_hint
            if req.required:
                missing_required.append(req.name)
            print(f"  ❌ {req.name}: NOT FOUND ({req.install_hint or 'no hint'})")
        else:
            if req.name not in auto_installed:
                print(f"  ✅ {req.name}: {check.get('version', 'present')}")
        checks.append(check)

    return {
        "checks": checks,
        "missing_required": missing_required,
        "auto_installed": auto_installed,
        "auto_install_failures": auto_install_failures,
        "status": "OK" if not missing_required else "MISSING_REQUIRED_CLIS",
    }


# ---------------------------------------------------------------------------
# Step 2 — Fetch repos
# ---------------------------------------------------------------------------

def _auth_url(url: str) -> str:
    """Inject GITHUB_TOKEN into an https github URL for non-interactive clone.

    We never log the token. If the URL is already SSH or non-github, leave it.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not url.startswith("https://github.com/"):
        return url
    # https://github.com/owner/repo.git -> https://x-access-token:TOKEN@github.com/owner/repo.git
    return url.replace("https://", f"https://x-access-token:{token}@", 1)


def fetch_repo(repo: RepoConfig, workspace: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    """Clone (or pull) a single repo into `workspace/<repo.name>`."""
    if not repo.git_url:
        return _fail("fetch_repo", f"repo '{repo.name}' has no git_url in config",
                     repo=repo.name)

    target = workspace / repo.name
    workspace.mkdir(parents=True, exist_ok=True)

    if (target / ".git").is_dir():
        print(f"  [{repo.name}] existing clone — fetching latest")
        _run(["git", "fetch", "--all", "--tags", "--prune"],
             cwd=target, dry_run=dry_run)
    else:
        print(f"  [{repo.name}] cloning {repo.git_url}")
        auth_url = _auth_url(repo.git_url)
        url_has_token = auth_url != repo.git_url
        _run(["git", "clone", auth_url, str(target)],
             dry_run=dry_run, secret=url_has_token)

    ref = repo.ref or repo.branch
    _run(["git", "checkout", ref], cwd=target, dry_run=dry_run)
    if not repo.ref:  # only pull if tracking a branch
        _run(["git", "pull", "--ff-only", "origin", repo.branch],
             cwd=target, dry_run=dry_run, check=False)

    # Capture the commit SHA we actually built
    sha = "dry-run-sha"
    if not dry_run:
        sha_result = _run(["git", "rev-parse", "HEAD"], cwd=target,
                          dry_run=False, check=False)
        sha = sha_result["stdout"].strip() or "unknown"

    return _ok("fetch_repo", repo=repo.name, path=str(target), ref=ref, commit=sha)


def fetch_repos(repos: List[RepoConfig], config: DeploymentConfig) -> List[Dict[str, Any]]:
    """Clone all repos in parallel-friendly sequence (sequential for log clarity)."""
    print(f"\n[executor] Fetching {len(repos)} repo(s)...")
    workspace = Path(config.workspace_dir)
    results: List[Dict[str, Any]] = []
    for repo in repos:
        if repo.skip:
            print(f"  [{repo.name}] skipped (skip=true in config)")
            results.append({"action": "fetch_repo", "status": "SKIPPED",
                            "repo": repo.name, "at": _now()})
            continue
        try:
            results.append(fetch_repo(repo, workspace, dry_run=config.dry_run))
        except Exception as e:
            results.append(_fail("fetch_repo", str(e), repo=repo.name))
    return results


# ---------------------------------------------------------------------------
# Step 3 — Registry login + image push helpers
# ---------------------------------------------------------------------------

def _registry_qualified_image(repo: RepoConfig,
                              registry: Optional[RegistryConfig]) -> str:
    """Return image ref qualified with registry URL when push is configured.

    ECR     -> <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag>
    Hub     -> docker.io/<username>/<image>:<tag>
    GHCR    -> ghcr.io/<owner>/<image>:<tag>
    Generic -> <url>/<image>:<tag>
    None    -> <image>:<tag>     (no registry — local-only)
    """
    base = f"{repo.image_name}:{repo.image_tag}"
    if registry is None:
        return base
    if registry.kind == "ecr":
        return f"{registry.url}/{repo.image_name}:{repo.image_tag}"
    if registry.kind == "dockerhub":
        ns = registry.username or "library"
        return f"docker.io/{ns}/{repo.image_name}:{repo.image_tag}"
    if registry.kind == "ghcr":
        owner = registry.username
        if not owner:
            return f"ghcr.io/{repo.image_name}:{repo.image_tag}"
        return f"ghcr.io/{owner}/{repo.image_name}:{repo.image_tag}"
    if registry.kind == "generic":
        host = registry.url.rstrip("/")
        return f"{host}/{repo.image_name}:{repo.image_tag}"
    return base


def registry_login(registry: RegistryConfig,
                   config: DeploymentConfig) -> Dict[str, Any]:
    """Authenticate `docker` to a registry. Password is piped via stdin so it
    never lands in shell history or process listings."""
    print(f"\n[executor] Logging in to registry '{registry.name}' "
          f"(kind={registry.kind})...")

    if registry.kind == "ecr":
        if not registry.region or not registry.account_id:
            return _fail("registry_login",
                         "ECR registry needs `region` and `account_id`",
                         registry=registry.name)
        registry_url = registry.url or (
            f"{registry.account_id}.dkr.ecr.{registry.region}.amazonaws.com"
        )
        # 1) Get login password from AWS
        pw_result = _run(
            ["aws", "ecr", "get-login-password", "--region", registry.region],
            dry_run=config.dry_run, check=False, timeout=60,
        )
        if pw_result["returncode"] != 0 and not config.dry_run:
            return _fail("registry_login",
                         f"aws ecr get-login-password failed: {pw_result['stderr']}",
                         registry=registry.name)
        password = pw_result["stdout"] or "dry-run-token"
        # 2) docker login via stdin
        login = _run(
            ["docker", "login", "--username", "AWS",
             "--password-stdin", registry_url],
            stdin=password, dry_run=config.dry_run, check=False, timeout=60,
        )
        if login["returncode"] != 0 and not config.dry_run:
            return _fail("registry_login",
                         f"docker login failed: {login['stderr']}",
                         registry=registry.name)
        return _ok("registry_login", registry=registry.name,
                   url=registry_url, kind="ecr")

    if registry.kind in ("dockerhub", "ghcr", "generic"):
        host = {
            "dockerhub": "docker.io",
            "ghcr": "ghcr.io",
            "generic": registry.url,
        }[registry.kind]
        if not registry.username or not registry.password:
            return _fail("registry_login",
                         f"{registry.kind} needs username + password",
                         registry=registry.name)
        login = _run(
            ["docker", "login", host, "--username", registry.username,
             "--password-stdin"],
            stdin=registry.password,
            dry_run=config.dry_run, check=False, timeout=60,
        )
        if login["returncode"] != 0 and not config.dry_run:
            return _fail("registry_login",
                         f"docker login failed: {login['stderr']}",
                         registry=registry.name)
        return _ok("registry_login", registry=registry.name,
                   url=host, kind=registry.kind)

    return _fail("registry_login",
                 f"unsupported registry kind '{registry.kind}'",
                 registry=registry.name)


def registry_logins(repos: List[RepoConfig],
                    config: DeploymentConfig) -> List[Dict[str, Any]]:
    """Log in to every distinct registry referenced by these repos."""
    seen: set = set()
    results: List[Dict[str, Any]] = []
    for repo in repos:
        if repo.skip or not repo.push or not repo.registry:
            continue
        if repo.registry in seen:
            continue
        seen.add(repo.registry)
        reg = config.get_registry(repo.registry)
        if reg is None:
            results.append(_fail("registry_login",
                                 f"registry '{repo.registry}' not in config",
                                 registry=repo.registry))
            continue
        results.append(registry_login(reg, config))
    return results


def _ensure_ecr_repository(repo: RepoConfig, registry: RegistryConfig,
                           config: DeploymentConfig) -> Dict[str, Any]:
    """`aws ecr describe-repositories`; if missing, `aws ecr create-repository`."""
    if not registry.create_repository_if_missing:
        return _ok("ensure_ecr_repo", repo=repo.name, skipped=True)

    describe = _run(
        ["aws", "ecr", "describe-repositories",
         "--repository-names", repo.image_name,
         "--region", registry.region],
        dry_run=config.dry_run, check=False, timeout=30,
    )
    if describe["returncode"] == 0:
        return _ok("ensure_ecr_repo", repo=repo.name, existed=True)

    print(f"  [ECR] repository '{repo.image_name}' not found, creating...")
    create = _run(
        ["aws", "ecr", "create-repository",
         "--repository-name", repo.image_name,
         "--region", registry.region,
         "--image-scanning-configuration", "scanOnPush=true"],
        dry_run=config.dry_run, check=False, timeout=60,
    )
    if create["returncode"] != 0 and not config.dry_run:
        return _fail("ensure_ecr_repo",
                     f"aws ecr create-repository failed: {create['stderr']}",
                     repo=repo.name)
    return _ok("ensure_ecr_repo", repo=repo.name, created=True)


def push_image(repo: RepoConfig, registry: RegistryConfig,
               config: DeploymentConfig) -> Dict[str, Any]:
    """`docker push` the registry-qualified image. For ECR, ensure the repo
    exists first."""
    qualified = _registry_qualified_image(repo, registry)
    print(f"\n[executor] Pushing {qualified}...")

    if registry.kind == "ecr":
        ensure = _ensure_ecr_repository(repo, registry, config)
        if ensure["status"] != "SUCCESS":
            return _fail("push_image",
                         f"ECR repo ensure failed: {ensure.get('error')}",
                         repo=repo.name, image=qualified)

    push = _run(["docker", "push", qualified],
                dry_run=config.dry_run, check=False, timeout=900)
    if push["returncode"] != 0 and not config.dry_run:
        return _fail("push_image",
                     f"docker push failed: {push['stderr'][-300:]}",
                     repo=repo.name, image=qualified)
    return _ok("push_image", repo=repo.name, image=qualified,
               registry=registry.name)


def push_images(repos: List[RepoConfig],
                config: DeploymentConfig) -> List[Dict[str, Any]]:
    print(f"\n[executor] Pushing docker images...")
    results: List[Dict[str, Any]] = []
    for r in repos:
        if r.skip:
            continue
        if not r.push:
            results.append({"action": "push_image", "status": "SKIPPED",
                            "repo": r.name, "reason": "push=false",
                            "at": _now()})
            continue
        if not r.registry:
            results.append({"action": "push_image", "status": "SKIPPED",
                            "repo": r.name, "reason": "no registry configured",
                            "at": _now()})
            continue
        reg = config.get_registry(r.registry)
        if reg is None:
            results.append(_fail("push_image",
                                 f"registry '{r.registry}' not in config",
                                 repo=r.name))
            continue
        results.append(push_image(r, reg, config))
    return results


# ---------------------------------------------------------------------------
# Step 3 — Build docker images
# ---------------------------------------------------------------------------

def build_image(repo: RepoConfig, workspace: Path,
                registry: Optional[RegistryConfig] = None,
                *, dry_run: bool = False) -> Dict[str, Any]:
    """`docker build` the repo. When a registry is configured we tag with both
    the local name (`image:tag`) and the registry-qualified name
    (`<registry>/<image>:<tag>`), so the push step needs no extra tag command
    and local-only targets (compose/k8s) keep working off the short name."""
    repo_path = workspace / repo.name
    if not repo_path.is_dir() and not dry_run:
        return _fail("build_image", f"repo path missing: {repo_path}", repo=repo.name)

    dockerfile = repo_path / repo.dockerfile
    if not dockerfile.is_file() and not dry_run:
        # Search one level deep for Dockerfile in subdirectories
        found = list(repo_path.rglob("Dockerfile"))
        if found:
            dockerfile = found[0]
            # Update docker_context to the directory containing the Dockerfile
            repo.docker_context = str(dockerfile.parent.relative_to(repo_path))
            print(f"  [{repo.name}] Dockerfile found at subdirectory: {dockerfile}")
        else:
            return _fail("build_image",
                        f"Dockerfile not found at {dockerfile} or any subdirectory",
                        repo=repo.name)

    local_tag = f"{repo.image_name}:{repo.image_tag}"
    qualified_tag = _registry_qualified_image(repo, registry)

    cmd = ["docker", "build", "-t", local_tag]
    if qualified_tag != local_tag:
        cmd.extend(["-t", qualified_tag])
    cmd.extend(["-f", str(dockerfile)])
    for k, v in repo.build_args.items():
        cmd.extend(["--build-arg", f"{k}={v}"])
    cmd.append(str(dockerfile.parent) if repo.docker_context == "." else str(repo_path / repo.docker_context))

    try:
        result = _run(cmd, dry_run=dry_run, timeout=1800)
        return _ok("build_image", repo=repo.name,
                   local_tag=local_tag, qualified_tag=qualified_tag,
                   build_log_tail=result.get("stdout", "")[-500:])
    except Exception as e:
        return _fail("build_image", str(e), repo=repo.name,
                     local_tag=local_tag)


def build_images(repos: List[RepoConfig], config: DeploymentConfig) -> List[Dict[str, Any]]:
    print(f"\n[executor] Building docker images...")
    workspace = Path(config.workspace_dir)
    out = []
    for r in repos:
        if r.skip:
            continue
        registry = config.get_registry(r.registry) if r.registry else None
        out.append(build_image(r, workspace, registry, dry_run=config.dry_run))
    return out


# ---------------------------------------------------------------------------
# Step 4 — Trigger deployment
# ---------------------------------------------------------------------------

def trigger_deploy(repo: RepoConfig, config: DeploymentConfig) -> Dict[str, Any]:
    """Dispatch to the deploy target configured for this repo."""
    target_name = repo.deploy_target or "default"
    target = config.deploy_targets.get(target_name)
    if target is None:
        return _fail("trigger_deploy",
                     f"deploy target '{target_name}' not defined in config",
                     repo=repo.name)

    print(f"\n  [{repo.name}] deploying via target='{target.name}' kind='{target.kind}'")
    handler = _DEPLOY_HANDLERS.get(target.kind)
    if handler is None:
        return _fail("trigger_deploy",
                     f"unsupported deploy kind '{target.kind}' "
                     f"(supported: {sorted(_DEPLOY_HANDLERS)})",
                     repo=repo.name)
    try:
        return handler(repo, target, config)
    except Exception as e:
        return _fail("trigger_deploy", str(e), repo=repo.name, target=target.name)


# ---------------------------------------------------------------------------
# Multi-environment promotion
# ---------------------------------------------------------------------------

def trigger_deploy_to_env(repo: RepoConfig, env: EnvironmentDeploy,
                          config: DeploymentConfig) -> Dict[str, Any]:
    """Deploy `repo` to a specific environment (one rung of its promotion
    ladder). Same logic as `trigger_deploy`, but uses `env.target` and merges
    `env.overrides` on top of any existing per-repo overrides.

    The image is NOT rebuilt — this just calls the deploy target with the
    image that was already built and pushed.
    """
    target = config.deploy_targets.get(env.target)
    if target is None:
        return _fail("trigger_deploy_env",
                     f"environment '{env.name}' references unknown deploy "
                     f"target '{env.target}'",
                     repo=repo.name, environment=env.name)

    # Stack overrides: existing _deploy_overrides + this env's overrides.
    # Env-level overrides win (more specific).
    existing = getattr(repo, "_deploy_overrides", {}) or {}
    combined = {**existing, **(env.overrides or {})}
    setattr(repo, "_deploy_overrides", combined)

    print(f"\n  [{repo.name}] deploying to env='{env.name}' "
          f"target='{target.name}' kind='{target.kind}'")
    handler = _DEPLOY_HANDLERS.get(target.kind)
    if handler is None:
        return _fail("trigger_deploy_env",
                     f"unsupported deploy kind '{target.kind}'",
                     repo=repo.name, environment=env.name)
    try:
        result = handler(repo, target, config)
        result["environment"] = env.name
        return result
    except Exception as e:
        return _fail("trigger_deploy_env", str(e),
                     repo=repo.name, environment=env.name, target=target.name)
    finally:
        # Restore the original overrides so the next env sees a clean slate
        setattr(repo, "_deploy_overrides", existing)


def run_smoke_test(repo: RepoConfig, env: EnvironmentDeploy,
                   config: DeploymentConfig) -> Dict[str, Any]:
    """Run the smoke test for an environment. Returns {status, ...details}.

    Two modes (env.smoke_test.command wins if both set):
      - HTTP GET on env.smoke_test.url, must return env.smoke_test.expect_status.
      - Shell: run env.smoke_test.command via `sh -c`, must exit 0.

    Honors retries with retry_delay between attempts. config.dry_run skips
    the test entirely and reports SUCCESS.
    """
    test = env.smoke_test
    if test is None:
        return {"action": "smoke_test", "status": "SKIPPED",
                "reason": "no smoke_test configured",
                "environment": env.name, "at": _now()}

    if config.dry_run:
        print(f"  [dry-run] would smoke-test {env.name}: "
              f"{test.command or test.url}")
        return _ok("smoke_test", repo=repo.name, environment=env.name,
                   dry_run=True)

    import time
    import urllib.request

    last_error = ""
    for attempt in range(1, test.retries + 1):
        print(f"  [{repo.name}] smoke test {env.name} "
              f"(attempt {attempt}/{test.retries})...")

        if test.command:
            r = _run(["sh", "-c", test.command],
                     dry_run=False, check=False, timeout=test.timeout_seconds)
            if r["returncode"] == 0:
                return _ok("smoke_test", repo=repo.name, environment=env.name,
                           mode="shell", command=test.command,
                           stdout_tail=r["stdout"][-300:])
            last_error = f"exit {r['returncode']}: {r['stderr'][-200:]}"
        elif test.url:
            try:
                req = urllib.request.Request(test.url, method="GET")
                with urllib.request.urlopen(req, timeout=test.timeout_seconds) as resp:  # noqa: S310
                    if resp.status == test.expect_status:
                        return _ok("smoke_test", repo=repo.name,
                                   environment=env.name, mode="http",
                                   url=test.url, http_status=resp.status)
                    last_error = (f"got HTTP {resp.status}, "
                                  f"expected {test.expect_status}")
            except Exception as e:
                last_error = str(e)
        else:
            return _fail("smoke_test",
                         "smoke_test has neither `command` nor `url`",
                         repo=repo.name, environment=env.name)

        if attempt < test.retries:
            print(f"    ✗ {last_error} — retrying in "
                  f"{test.retry_delay_seconds}s")
            time.sleep(test.retry_delay_seconds)

    return _fail("smoke_test",
                 f"smoke test failed after {test.retries} attempts: {last_error}",
                 repo=repo.name, environment=env.name)


# ---- handlers -------------------------------------------------------------

def _opt(repo: RepoConfig, target: DeployTarget, key: str, default: Any = None) -> Any:
    """Look up a deploy option. Repo-level overrides (from .deploy.yaml's
    `deploy:` block, attached as `repo._deploy_overrides`) win over the
    platform target's config. This is how a repo customizes its deploy
    without owning the whole target."""
    overrides = getattr(repo, "_deploy_overrides", {}) or {}
    if key in overrides:
        return overrides[key]
    return target.config.get(key, default)


def _deploy_dry_run(repo: RepoConfig, target: DeployTarget,
                    config: DeploymentConfig) -> Dict[str, Any]:
    print(f"  [dry-run] would deploy {repo.image_name}:{repo.image_tag}")
    return _ok("trigger_deploy", repo=repo.name, target=target.name, kind="dry-run")


def _deploy_compose(repo: RepoConfig, target: DeployTarget,
                    config: DeploymentConfig) -> Dict[str, Any]:
    """`docker compose up -d` for a service in a compose file."""
    compose_file = target.config.get("compose_file", "docker-compose.yml")
    service = target.config.get("service", repo.name)
    cmd = ["docker", "compose", "-f", compose_file, "up", "-d", service]
    _run(cmd, dry_run=config.dry_run, timeout=900)
    return _ok("trigger_deploy", repo=repo.name, target=target.name,
               kind="compose", service=service, compose_file=compose_file)


def _deploy_kubernetes(repo: RepoConfig, target: DeployTarget,
                       config: DeploymentConfig) -> Dict[str, Any]:
    """`kubectl set image` then `kubectl rollout status` to wait for healthy."""
    namespace = target.config.get("namespace", "default")
    deployment = target.config.get("deployment", repo.name)
    container = target.config.get("container", repo.name)
    registry = config.get_registry(repo.registry) if repo.registry else None
    image = _registry_qualified_image(repo, registry)

    set_cmd = ["kubectl", "-n", namespace, "set", "image",
               f"deployment/{deployment}", f"{container}={image}"]
    _run(set_cmd, dry_run=config.dry_run, timeout=120)

    wait_cmd = ["kubectl", "-n", namespace, "rollout", "status",
                f"deployment/{deployment}", "--timeout=300s"]
    _run(wait_cmd, dry_run=config.dry_run, timeout=360)

    return _ok("trigger_deploy", repo=repo.name, target=target.name,
               kind="kubernetes", namespace=namespace,
               deployment=deployment, image=image)


def _deploy_webhook(repo: RepoConfig, target: DeployTarget,
                    config: DeploymentConfig) -> Dict[str, Any]:
    """POST to a webhook (n8n / TeamCity REST / Jenkins remote trigger / etc).

    Payload is built from target.config['payload'] with `{image}`, `{repo}`,
    `{tag}`, `{commit}` substitutions.
    """
    import urllib.request

    url = target.config.get("url")
    if not url:
        return _fail("trigger_deploy",
                     "webhook target needs config.url", repo=repo.name)

    payload_template = target.config.get("payload", {})
    registry = config.get_registry(repo.registry) if repo.registry else None
    payload = _substitute(payload_template, {
        "image": _registry_qualified_image(repo, registry),
        "repo": repo.name,
        "tag": repo.image_tag,
        "commit": repo.ref or repo.branch,
    })
    headers = target.config.get("headers", {"Content-Type": "application/json"})
    headers = _substitute(headers, {})  # env-vars already resolved at load time

    if config.dry_run:
        print(f"  [dry-run] would POST to {url}")
        print(f"  [dry-run] payload: {json.dumps(payload)[:200]}")
        return _ok("trigger_deploy", repo=repo.name, target=target.name,
                   kind="webhook", url=url, dry_run=True)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")[:1000]
            return _ok("trigger_deploy", repo=repo.name, target=target.name,
                       kind="webhook", url=url,
                       http_status=resp.status, response_tail=body)
    except Exception as e:
        return _fail("trigger_deploy", f"webhook POST failed: {e}",
                     repo=repo.name, target=target.name, url=url)


def _deploy_shell(repo: RepoConfig, target: DeployTarget,
                  config: DeploymentConfig) -> Dict[str, Any]:
    """Run an arbitrary shell command. The escape hatch for unusual targets."""
    cmd_template = target.config.get("command")
    if not cmd_template:
        return _fail("trigger_deploy",
                     "shell target needs config.command", repo=repo.name)
    registry = config.get_registry(repo.registry) if repo.registry else None
    rendered = cmd_template.format(
        image=_registry_qualified_image(repo, registry),
        repo=repo.name,
        tag=repo.image_tag,
        commit=repo.ref or repo.branch,
    )
    _run(shlex.split(rendered), dry_run=config.dry_run, timeout=900)
    return _ok("trigger_deploy", repo=repo.name, target=target.name,
               kind="shell", command=rendered)


# ---- AWS handlers --------------------------------------------------------

def _aws_env(target: DeployTarget) -> Dict[str, str]:
    """Build env dict for aws CLI from the deploy target config. If config
    values are empty, falls back to whatever's already in the process env."""
    cfg = target.config
    env: Dict[str, str] = {}
    if cfg.get("access_key_id"):
        env["AWS_ACCESS_KEY_ID"] = cfg["access_key_id"]
    if cfg.get("secret_access_key"):
        env["AWS_SECRET_ACCESS_KEY"] = cfg["secret_access_key"]
    region = cfg.get("region", "")
    if region:
        env["AWS_DEFAULT_REGION"] = region
    return env


def _aws_region_args(target: DeployTarget) -> List[str]:
    region = target.config.get("region", "")
    return ["--region", region] if region else []


def _ecs_describe_json(cmd: List[str], env: Dict[str, str], dry_run: bool) -> Optional[dict]:
    """Run an aws describe/list command and parse JSON stdout. Returns None on
    any failure or in dry_run (caller treats None as 'not found / unknown')."""
    r = _run(cmd, env=env, dry_run=dry_run, check=False, timeout=60)
    if dry_run or r["returncode"] != 0:
        return None
    try:
        return json.loads(r["stdout"])
    except Exception:
        return None


def _ensure_ecs_infra(repo: RepoConfig, target: DeployTarget,
                      config: DeploymentConfig, image: str) -> Dict[str, Any]:
    """Create the minimum ECS infrastructure if it doesn't already exist, so a
    brand-new account can deploy without manual setup.

    SCOPE (deliberately minimal to limit blast radius):
      - Reuses the account's DEFAULT VPC + its subnets (does NOT create a VPC,
        subnets, gateways, or a load balancer).
      - Creates, only if missing: a Fargate ECS cluster, a CloudWatch log group,
        an execution IAM role (or reuses one named by config), a task definition,
        a security group allowing the container port, and a Fargate service with
        a public IP.
    Every step is idempotent (check-then-create) and honours dry_run.

    Returns {"ok": True, "task_def_arn": <arn or None>} on success, or a _fail
    dict on error.
    """
    env = _aws_env(target)
    region_args = _aws_region_args(target)
    dry = config.dry_run
    cluster = _opt(repo, target, "cluster", "sdlc-cluster")
    service = _opt(repo, target, "service", repo.name)
    container_port = int(_opt(repo, target, "container_port", 8000))
    cpu = str(_opt(repo, target, "cpu", "256"))
    memory = str(_opt(repo, target, "memory", "512"))
    exec_role = _opt(repo, target, "execution_role_arn", "")
    log_group = f"/ecs/{service}"

    print(f"\n  [ecs-provision] ensuring infra for service='{service}' "
          f"cluster='{cluster}' (dry_run={dry})")

    # 1) Cluster — create if missing (create-cluster is idempotent: returns the
    #    existing cluster if it already exists).
    _run(["aws", "ecs", "create-cluster", "--cluster-name", cluster,
          "--capacity-providers", "FARGATE", *region_args],
         env=env, dry_run=dry, check=False, timeout=60)

    # 2) CloudWatch log group — create if missing (ignore "already exists").
    _run(["aws", "logs", "create-log-group", "--log-group-name", log_group,
          *region_args], env=env, dry_run=dry, check=False, timeout=60)

    # 3) Execution role — required so ECS can pull from ECR and write logs.
    #    If the caller didn't supply one, reuse/create the AWS-conventional
    #    'ecsTaskExecutionRole'. Creating IAM roles needs iam permissions; if
    #    that fails we surface a clear message rather than a cryptic ECS error.
    if not exec_role:
        role_name = "ecsTaskExecutionRole"
        got = _ecs_describe_json(
            ["aws", "iam", "get-role", "--role-name", role_name],
            env=env, dry_run=dry)
        if got is None and not dry:
            # Try to create it with the standard trust + managed policy.
            trust = ('{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
                     '"Principal":{"Service":"ecs-tasks.amazonaws.com"},'
                     '"Action":"sts:AssumeRole"}]}')
            create = _run(["aws", "iam", "create-role", "--role-name", role_name,
                           "--assume-role-policy-document", trust],
                          env=env, dry_run=dry, check=False, timeout=60)
            if create["returncode"] != 0 and "EntityAlreadyExists" not in create.get("stderr", ""):
                return _fail("ensure_ecs_infra",
                             f"could not create execution role '{role_name}': "
                             f"{create['stderr'][-300:]} — the IAM user needs "
                             f"iam:CreateRole/AttachRolePolicy, or set "
                             f"execution_role_arn in config.", repo=repo.name)
            _run(["aws", "iam", "attach-role-policy", "--role-name", role_name,
                  "--policy-arn",
                  "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
                 env=env, dry_run=dry, check=False, timeout=60)
        # Resolve the ARN for the task definition.
        if not dry:
            role_json = _ecs_describe_json(
                ["aws", "iam", "get-role", "--role-name", role_name],
                env=env, dry_run=dry)
            if role_json:
                exec_role = role_json["Role"]["Arn"]
        if not exec_role and not dry:
            return _fail("ensure_ecs_infra",
                         f"execution role '{role_name}' unavailable",
                         repo=repo.name)

    # 4) Default VPC subnets + a security group opening the container port.
    subnets: List[str] = []
    sg_id = ""
    vpc_id = "" # <--- ADD THIS INITIALIZATION
    if not dry:
        # Check for pre-configured subnets first
        configured_subnets = _opt(repo, target, "subnets", [])
        if configured_subnets:
            subnets = configured_subnets
            vpc_id = _opt(repo, target, "vpc_id", "") # <--- ADD THIS LINE
            print(f"  [ecs-provision] using pre-configured subnets: {subnets}")
        else:
            vpc_json = _ecs_describe_json(
                ["aws", "ec2", "describe-vpcs", "--filters",
                "Name=isDefault,Values=true", *region_args],
                env=env, dry_run=dry)
            vpc_id = ""
            if vpc_json and vpc_json.get("Vpcs"):
                vpc_id = vpc_json["Vpcs"][0]["VpcId"]
            if not vpc_id:
                return _fail("ensure_ecs_infra",
                            "no default VPC found in this region. Either create a "
                            "default VPC or set subnets in config.", repo=repo.name)
            sn_result = _run(
                ["aws", "ec2", "describe-subnets",
                "--filters", f"Name=vpc-id,Values={vpc_id}",
                "--filters", "Name=defaultForAz,Values=true",
                *region_args],
                env=env, dry_run=dry, check=False, timeout=30)
            try:
                import json as _j
                sn_data = _j.loads(sn_result["stdout"])
                subnets = [s["SubnetId"] for s in sn_data.get("Subnets", [])][:3]
            except Exception:
                subnets = []
            if not subnets:
                return _fail("ensure_ecs_infra",
                            f"default VPC {vpc_id} has no subnets", repo=repo.name)
        # Security group: reuse one named sdlc-<service>-sg or create it.
        sg_name = f"sdlc-{service}-sg"
        sg_json = _ecs_describe_json(
            ["aws", "ec2", "describe-security-groups", "--filters",
             f"Name=group-name,Values={sg_name}",
             f"Name=vpc-id,Values={vpc_id}", *region_args],
            env=env, dry_run=dry)
        if sg_json and sg_json.get("SecurityGroups"):
            sg_id = sg_json["SecurityGroups"][0]["GroupId"]
        else:
            created = _ecs_describe_json(
                ["aws", "ec2", "create-security-group", "--group-name", sg_name,
                 "--description", f"SDLC deploy SG for {service}",
                 "--vpc-id", vpc_id, *region_args],
                env=env, dry_run=dry)
            sg_id = (created or {}).get("GroupId", "")
            if sg_id:
                _run(["aws", "ec2", "authorize-security-group-ingress",
                      "--group-id", sg_id, "--protocol", "tcp",
                      "--port", str(container_port), "--cidr", "0.0.0.0/0",
                      *region_args], env=env, dry_run=dry, check=False, timeout=60)

    # 5) Register a task definition for this image (Fargate).
    task_def_arn: Optional[str] = None
    td = {
        "family": service,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": cpu,
        "memory": memory,
        "executionRoleArn": exec_role or "ROLE_PLACEHOLDER",
        "containerDefinitions": [{
            "name": service,
            "image": image,
            "essential": True,
            "portMappings": [{"containerPort": container_port, "protocol": "tcp"}],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": log_group,
                    "awslogs-region": target.config.get("region", "us-east-1"),
                    "awslogs-stream-prefix": "ecs",
                },
            },
        }],
    }
    tmp_td = Path(config.workspace_dir) / f"{service}-taskdef.json"
    tmp_td.parent.mkdir(parents=True, exist_ok=True)
    tmp_td.write_text(json.dumps(td))
    reg = _run(["aws", "ecs", "register-task-definition",
                "--cli-input-json", f"file://{tmp_td}", *region_args],
               env=env, dry_run=dry, check=False, timeout=60)
    if reg["returncode"] != 0 and not dry:
        return _fail("ensure_ecs_infra",
                     f"register-task-definition failed: {reg['stderr'][-300:]}",
                     repo=repo.name)
    if not dry:
        try:
            task_def_arn = json.loads(reg["stdout"])["taskDefinition"]["taskDefinitionArn"]
        except Exception:
            pass

    # 6) Service — create if missing, else the caller's update path handles it.
    svc_json = _ecs_describe_json(
        ["aws", "ecs", "describe-services", "--cluster", cluster,
         "--services", service, *region_args], env=env, dry_run=dry)
    svc_exists = bool(
        svc_json and svc_json.get("services")
        and svc_json["services"][0].get("status") == "ACTIVE"
    )
    if not svc_exists:
        net_cfg = (
            "awsvpcConfiguration={"
            f"subnets=[{','.join(subnets)}],"
            f"securityGroups=[{sg_id}],"
            "assignPublicIp=ENABLED}"
        ) if not dry else "awsvpcConfiguration={subnets=[...],securityGroups=[...],assignPublicIp=ENABLED}"
        create_svc = _run(
            ["aws", "ecs", "create-service", "--cluster", cluster,
             "--service-name", service,
             "--task-definition", task_def_arn or service,
             "--desired-count", "1", "--launch-type", "FARGATE",
             "--network-configuration", net_cfg, *region_args],
            env=env, dry_run=dry, check=False, timeout=120)
        if create_svc["returncode"] != 0 and not dry:
            return _fail("ensure_ecs_infra",
                         f"create-service failed: {create_svc['stderr'][-300:]}",
                         repo=repo.name)
        print(f"  [ecs-provision] created service '{service}'")
        return {"ok": True, "task_def_arn": task_def_arn, "created_service": True}

    return {"ok": True, "task_def_arn": task_def_arn, "created_service": False}


def _deploy_ecs(repo: RepoConfig, target: DeployTarget,
                config: DeploymentConfig) -> Dict[str, Any]:
    """ECS (Fargate) deployment via aws CLI, with create-if-missing infra.

    If `auto_provision` is enabled (default True), missing infra — cluster,
    execution role, task definition, security group, and the Fargate service —
    is created automatically using the account's DEFAULT VPC/subnets. No VPC,
    gateways, or load balancer are created. Then:
      - first deploy: the service is created already pointing at the new image.
      - later deploys: update-service --force-new-deployment (optionally with a
        freshly registered task def revision).
    """
    cluster = _opt(repo, target, "cluster", "sdlc-cluster")
    service = _opt(repo, target, "service", repo.name)

    registry = config.get_registry(repo.registry) if repo.registry else None
    image = _registry_qualified_image(repo, registry)
    env = _aws_env(target)
    region_args = _aws_region_args(target)

    auto_provision = bool(_opt(repo, target, "auto_provision", True))
    provisioned_task_def: Optional[str] = None
    just_created_service = False
    if auto_provision:
        infra = _ensure_ecs_infra(repo, target, config, image)
        if not infra.get("ok"):
            return infra  # already a _fail dict
        provisioned_task_def = infra.get("task_def_arn")
        just_created_service = infra.get("created_service", False)

    # If provisioning just created the service, it's already running the new
    # image — no update needed (and update-service would race the creation).
    if just_created_service:
        if _opt(repo, target, "wait_for_stable", True):
            _run(["aws", "ecs", "wait", "services-stable",
                  "--cluster", cluster, "--services", service, *region_args],
                 env=env, dry_run=config.dry_run, check=False, timeout=900)
        return _ok("trigger_deploy", repo=repo.name, target=target.name,
                   kind="ecs", cluster=cluster, service=service, image=image,
                   task_definition_arn=provisioned_task_def, created=True)

    if not cluster:
        return _fail("trigger_deploy", "ecs target needs `cluster` "
                                       "(in target.config or repo's .deploy.yaml `deploy:` block)",
                     repo=repo.name)

    new_task_def_arn: Optional[str] = provisioned_task_def
    td_template_path = _opt(repo, target, "task_definition_template")
    if td_template_path and not new_task_def_arn:
        # Render the template. We use plain str.replace (not str.format)
        # because the template is JSON — every `{` and `}` in JSON would
        # otherwise be parsed as a format placeholder.
        td_path = Path(td_template_path)
        if not td_path.is_file() and not config.dry_run:
            return _fail("trigger_deploy",
                         f"task_definition_template not found at {td_path}",
                         repo=repo.name)
        td_text = td_path.read_text() if td_path.is_file() else "{}"
        td_rendered = (
            td_text
            .replace("{image}", image)
            .replace("{repo}", repo.name)
            .replace("{tag}", repo.image_tag)
            .replace("{commit}", repo.ref or repo.branch)
        )
        tmp_td = Path(config.workspace_dir) / f"{repo.name}-taskdef.json"
        tmp_td.parent.mkdir(parents=True, exist_ok=True)
        tmp_td.write_text(td_rendered)

        register = _run(
            ["aws", "ecs", "register-task-definition",
             "--cli-input-json", f"file://{tmp_td}", *region_args],
            env=env, dry_run=config.dry_run, check=False, timeout=60,
        )
        if register["returncode"] != 0 and not config.dry_run:
            return _fail("trigger_deploy",
                         f"register-task-definition failed: {register['stderr'][-300:]}",
                         repo=repo.name)
        if not config.dry_run:
            try:
                payload = json.loads(register["stdout"])
                new_task_def_arn = payload["taskDefinition"]["taskDefinitionArn"]
            except Exception:
                pass

    update_cmd = ["aws", "ecs", "update-service",
                  "--cluster", cluster, "--service", service,
                  "--force-new-deployment", *region_args]
    if new_task_def_arn:
        update_cmd.extend(["--task-definition", new_task_def_arn])

    update = _run(update_cmd, env=env, dry_run=config.dry_run,
                  check=False, timeout=120)
    if update["returncode"] != 0 and not config.dry_run:
        return _fail("trigger_deploy",
                     f"update-service failed: {update['stderr'][-300:]}",
                     repo=repo.name)

    if _opt(repo, target, "wait_for_stable", True):
        wait = _run(
            ["aws", "ecs", "wait", "services-stable",
             "--cluster", cluster, "--services", service, *region_args],
            env=env, dry_run=config.dry_run, check=False, timeout=900,
        )
        if wait["returncode"] != 0 and not config.dry_run:
            return _fail("trigger_deploy",
                         f"ecs services-stable wait failed (deploy may "
                         f"still be in progress): {wait['stderr'][-300:]}",
                         repo=repo.name, cluster=cluster,
                         service=service, image=image)

    return _ok("trigger_deploy", repo=repo.name, target=target.name,
               kind="ecs", cluster=cluster, service=service,
               image=image, task_definition_arn=new_task_def_arn)


def _deploy_lambda(repo: RepoConfig, target: DeployTarget,
                   config: DeploymentConfig) -> Dict[str, Any]:
    """Lambda container deployment. Function must already exist (configured
    for container image deployment) — we update its image URI."""
    function_name = _opt(repo, target, "function_name", repo.name)
    registry = config.get_registry(repo.registry) if repo.registry else None
    image = _registry_qualified_image(repo, registry)
    env = _aws_env(target)
    region_args = _aws_region_args(target)

    update = _run(
        ["aws", "lambda", "update-function-code",
         "--function-name", function_name, "--image-uri", image,
         "--publish", *region_args],
        env=env, dry_run=config.dry_run, check=False, timeout=120,
    )
    if update["returncode"] != 0 and not config.dry_run:
        return _fail("trigger_deploy",
                     f"lambda update-function-code failed: {update['stderr'][-300:]}",
                     repo=repo.name, function=function_name)

    if _opt(repo, target, "wait_for_updated", True):
        _run(["aws", "lambda", "wait", "function-updated",
              "--function-name", function_name, *region_args],
             env=env, dry_run=config.dry_run, check=False, timeout=300)

    # Optional: bump an alias to the new version
    alias = _opt(repo, target, "alias")
    version: Optional[str] = None
    if alias and not config.dry_run:
        try:
            version = json.loads(update["stdout"]).get("Version")
        except Exception:
            version = None
        if version:
            _run(["aws", "lambda", "update-alias",
                  "--function-name", function_name, "--name", alias,
                  "--function-version", version, *region_args],
                 env=env, dry_run=config.dry_run, check=False, timeout=60)

    return _ok("trigger_deploy", repo=repo.name, target=target.name,
               kind="lambda", function=function_name, image=image,
               version=version, alias=alias)


_DEPLOY_HANDLERS = {
    "dry-run":    _deploy_dry_run,
    "compose":    _deploy_compose,
    "kubernetes": _deploy_kubernetes,
    "k8s":        _deploy_kubernetes,
    "webhook":    _deploy_webhook,
    "shell":      _deploy_shell,
    "ecs":        _deploy_ecs,
    "lambda":     _deploy_lambda,
}


def _substitute(obj: Any, mapping: Dict[str, str]) -> Any:
    """Recursive `{key}` substitution in dict/list/str values."""
    if isinstance(obj, str):
        try:
            return obj.format(**mapping)
        except (KeyError, IndexError):
            return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, mapping) for v in obj]
    return obj
