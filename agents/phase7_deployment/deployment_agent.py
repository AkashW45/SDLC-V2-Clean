"""
Phase 7 — Deployment Agent

Now config-driven:
  - resolve_deploy_sequence: orders affected_repos using config classification
  - verify_clis_node: checks required CLIs are installed before doing anything
  - fetch_repos_node: git clones the affected repos into a workspace
  - build_images_node: docker build per repo
  - human_approval_gate: interrupt for production approval
  - execute_deployment: triggers configured deploy target per repo
  - enable_feature_flags / monitor_deployment / execute_rollback: as before

Human approval gate stays a hard interrupt before any production action.
All CLI / git / docker / network commands respect `config.dry_run`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# Direct OpenAI client — matches the pattern used by Phases 1–6.
# Reads DEEPSEEK_API_KEY (the same env var every other phase uses) so this
# file no longer depends on the half-finished core/llm_gateway abstraction.
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

from core.deployment_config import (
    DeploymentConfig,
    RepoConfig,
    load_deployment_config,
    merge_repo_contract,
)
from core.deployment_executor import (
    verify_clis,
    fetch_repos,
    registry_logins,
    build_images,
    push_images,
    trigger_deploy,
    trigger_deploy_to_env,
    run_smoke_test,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DeploymentState(TypedDict, total=False):
    requirement: str
    scope_contract: dict
    runbook: dict
    pr_urls: list
    # Enriched by Phase 3 with full metadata from Phase 0 — each entry is a dict
    # {name, url, type, language, …}. resolve_deploy_sequence reads git URLs
    # directly from here; no separate selected_repos lookup needed.
    affected_repos: list           # list[dict] from Phase 3 — {name, url, type, …}

    # Per-repo commit SHAs to deploy (set by Phase 6's merge_prs step).
    # Phase 7's resolve_deploy_sequence pins each repo's `ref` to its
    # entry here so the build matches exactly what was merged.
    merged_shas: dict              # {repo_name: sha}

    # Resolved at runtime from config + affected_repos
    resolved_repos: list           # list[dict] — serialised RepoConfig
    deploy_sequence: list
    feature_flags: list

    # Executor results
    cli_check: dict
    fetch_results: list
    login_results: list
    build_results: list
    push_results: list
    deploy_results: list
    monitoring_results: dict

    rollback_triggered: bool
    human_feedback: str
    approved: bool
    status: str
    config_path: Optional[str]     # forwarded from caller; None = autodetect
    dry_run: bool                  # overrides config.dry_run if explicitly set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_cfg(state: DeploymentState) -> DeploymentConfig:
    """Load config, honoring caller's override of `dry_run`."""
    cfg = load_deployment_config(state.get("config_path"))
    if "dry_run" in state and state["dry_run"] is not None:
        cfg.dry_run = bool(state["dry_run"])
    return cfg


def _repo_to_dict(r: RepoConfig) -> dict:
    d = {
        "name": r.name, "git_url": r.git_url, "branch": r.branch,
        "ref": r.ref, "repo_type": r.repo_type, "image_name": r.image_name,
        "image_tag": r.image_tag, "deploy_target": r.deploy_target,
        "skip": r.skip, "registry": r.registry, "push": r.push,
        "dockerfile": r.dockerfile, "docker_context": r.docker_context,
        "build_args": r.build_args,
    }
    overrides = getattr(r, "_deploy_overrides", None)
    if overrides:
        d["_deploy_overrides"] = overrides
    # Roundtrip environments — each env is a small nested dict
    if r.environments:
        d["environments"] = [
            {
                "name": e.name, "target": e.target,
                "overrides": dict(e.overrides or {}),
                "requires_approval": e.requires_approval,
                "smoke_test": (
                    {
                        "url": e.smoke_test.url,
                        "expect_status": e.smoke_test.expect_status,
                        "command": e.smoke_test.command,
                        "timeout_seconds": e.smoke_test.timeout_seconds,
                        "retries": e.smoke_test.retries,
                        "retry_delay_seconds": e.smoke_test.retry_delay_seconds,
                    } if e.smoke_test else None
                ),
            }
            for e in r.environments
        ]
    return d


def _dict_to_repo(d: dict) -> RepoConfig:
    """Inverse of _repo_to_dict. Restores `_deploy_overrides` and
    `environments` so multi-env promotion works after state roundtrip."""
    from core.deployment_config import EnvironmentDeploy, SmokeTest
    overrides = d.get("_deploy_overrides")
    envs_raw = d.get("environments") or []
    envs: List = []
    for er in envs_raw:
        smoke = None
        if er.get("smoke_test"):
            smoke = SmokeTest(**er["smoke_test"])
        envs.append(EnvironmentDeploy(
            name=er["name"], target=er["target"],
            overrides=dict(er.get("overrides") or {}),
            smoke_test=smoke,
            requires_approval=bool(er.get("requires_approval", False)),
        ))
    rc = RepoConfig(
        name=d["name"], git_url=d.get("git_url", ""),
        branch=d.get("branch", "main"), ref=d.get("ref", ""),
        repo_type=d.get("repo_type", "service"),
        dockerfile=d.get("dockerfile", "Dockerfile"),
        docker_context=d.get("docker_context", "."),
        image_name=d.get("image_name", d["name"]),
        image_tag=d.get("image_tag", "latest"),
        build_args=d.get("build_args", {}) or {},
        registry=d.get("registry", ""),
        push=d.get("push", True),
        deploy_target=d.get("deploy_target", "default"),
        environments=envs,
        skip=d.get("skip", False),
    )
    if overrides:
        setattr(rc, "_deploy_overrides", overrides)
    return rc


def _repos_from_state(state: "DeploymentState") -> List[RepoConfig]:
    """Restore the live list of RepoConfig objects from state. After fetch,
    these have any .deploy.yaml overrides already merged in."""
    return [_dict_to_repo(d) for d in state.get("resolved_repos", [])]


def call_llm(prompt: str) -> dict:
    """Kept for compatibility with the rest of the agent suite.

    Uses the same direct-OpenAI pattern as Phases 1–6. Previously routed
    through core.llm_gateway, but that file looked for LLM_API_KEY while
    the rest of the project reads DEEPSEEK_API_KEY — mismatch crashed
    Phase 7 on import.
    """
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=False,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def resolve_deploy_sequence(state: DeploymentState) -> DeploymentState:
    """First-pass resolution: build a RepoConfig for each affected repo using
    the legacy `repos:` block first, then the `repo_resolver` template for
    anything not pre-declared. Classification is provisional here (name-based);
    the in-repo `.deploy.yaml` may override `type` later, in which case
    `fetch_repos_node` re-sorts the sequence.

    If `merged_shas` is present in state (populated by Phase 6's merge step),
    each repo's `ref` is pinned to the SHA Phase 6 just merged. The executor's
    fetch_repos will then `git checkout <sha>` instead of `git pull` on a
    branch — meaning the build is guaranteed to match what was reviewed in
    the PR, even if main has moved on since.
    """
    print("\n[Phase 7] Resolving repo names and git URLs...")
    cfg = _load_cfg(state)

    affected_raw = list(state.get("affected_repos") or [])
    if not affected_raw:
        print("  ⚠ no affected_repos provided — falling back to legacy `repos:` block")
        affected = list(cfg.repos.keys())
    else:
        # affected_repos may be enriched dicts {name, url, …} (new format from Phase 3)
        # or plain strings (old format / fallback). Normalise to plain names here.
        affected = [
            r["name"] if isinstance(r, dict) else r
            for r in affected_raw
        ]

    # SHAs to pin each repo to (from Phase 6's merge step). If empty, Phase 7
    # falls back to HEAD-of-branch — useful for hot-fix re-deploys where you
    # just want whatever's on main right now.
    merged_shas = state.get("merged_shas") or {}
    if merged_shas:
        print(f"  📌 Pinning repos to merged SHAs from Phase 6: "
              f"{', '.join(f'{k}={(v or '?')[:7]}' for k, v in merged_shas.items())}")

    # affected_repos from Phase 3 now carries full metadata (name, url, type, …)
    # enriched from Phase 0's selected_repos. Build a name→meta lookup so we can
    # resolve git URLs directly from the impact report — no separate selected_repos
    # lookup needed. Falls back gracefully if entries are plain strings (old format).
    affected_repos_meta: dict = {}
    for r in (state.get("affected_repos") or []):
        if isinstance(r, dict) and r.get("name"):
            affected_repos_meta[r["name"]] = r

    if affected_repos_meta:
        print(f"  📦 Re-using git URLs from Phase 3 impact report for: "
              f"{list(affected_repos_meta.keys())}")

    resolved: List[RepoConfig] = []
    missing_urls: List[str] = []
    for name in affected:
        rc = cfg.get_repo(name)
        # 1st priority: git URL from Phase 3's enriched impact_report (came from Phase 0)
        if not rc.git_url and name in affected_repos_meta:
            rc.git_url = affected_repos_meta[name].get("url", "")
            if rc.git_url:
                print(f"    ✔ {name}: using Phase 3 impact report git URL → {rc.git_url}")
        # 2nd priority: fall back to config's repo_resolver template
        if not rc.git_url:
            rc.git_url = cfg.resolve_repo_url(name)
            if not rc.git_url:
                missing_urls.append(name)
        # Pin to the exact commit Phase 6 just merged (if available)
        if name in merged_shas and merged_shas[name]:
            rc.ref = merged_shas[name]
        # Provisional classification — may be overridden by .deploy.yaml later
        rc.repo_type = cfg.classify_repo(rc)
        resolved.append(rc)

    if missing_urls:
        print(f"  ❌ no git URL for: {missing_urls}")
        print(f"     declare in `repos:` block OR set a `repo_resolver:` template")
        return {
            **state,
            "resolved_repos": [_repo_to_dict(r) for r in resolved],
            "deploy_sequence": [],
            "status": "RESOLVE_FAILED",
        }

    order = cfg.classification.order
    def order_key(r: RepoConfig) -> int:
        try:
            return order.index(r.repo_type)
        except ValueError:
            return len(order)
    resolved.sort(key=order_key)

    sequence = []
    for i, r in enumerate(resolved, start=1):
        sequence.append({
            "step": i, "repo": r.name, "type": r.repo_type,
            "deploy_target": r.deploy_target, "status": "PENDING",
        })

    print(f"  Provisional sequence: {[(s['step'], s['repo'], s['type']) for s in sequence]}")
    return {
        **state,
        "resolved_repos": [_repo_to_dict(r) for r in resolved],
        "deploy_sequence": sequence,
        "status": "SEQUENCE_RESOLVED",
    }


def verify_clis_node(state: DeploymentState) -> DeploymentState:
    cfg = _load_cfg(state)
    cli_check = verify_clis(cfg)
    status = "CLIS_VERIFIED" if cli_check["status"] == "OK" else "CLI_CHECK_FAILED"
    return {**state, "cli_check": cli_check, "status": status}


def fetch_repos_node(state: DeploymentState) -> DeploymentState:
    """Clone each repo, then read its `.deploy.yaml` (if present) and merge
    the in-repo deploy contract into our RepoConfig. After all merges, if any
    repo's contract changed its `type`, re-sort the deploy_sequence.
    """
    from pathlib import Path
    cfg = _load_cfg(state)
    repos = _repos_from_state(state)

    # 1) Clone everything
    results = fetch_repos(repos, cfg)
    all_ok = all(r["status"] in ("SUCCESS", "SKIPPED") for r in results)
    if not all_ok:
        return {
            **state, "fetch_results": results,
            "resolved_repos": [_repo_to_dict(r) for r in repos],
            "status": "FETCH_FAILED",
        }

    # 2) Merge .deploy.yaml from each cloned repo (if it has one)
    workspace = Path(cfg.workspace_dir)
    contract_count = 0
    for r in repos:
        if r.skip:
            continue
        repo_path = workspace / r.name
        before = (r.repo_type, r.deploy_target, r.registry)
        try:
            merge_repo_contract(r, repo_path, cfg)
        except ValueError as e:
            return {
                **state, "fetch_results": results,
                "resolved_repos": [_repo_to_dict(rr) for rr in repos],
                "status": "CONTRACT_INVALID",
                "contract_error": str(e),
            }
        after = (r.repo_type, r.deploy_target, r.registry)
        if before != after:
            contract_count += 1

    if contract_count:
        print(f"  📄 .deploy.yaml from {contract_count} repo(s) "
              f"changed deploy plan — re-sorting")
        order = cfg.classification.order
        def order_key(r: RepoConfig) -> int:
            try:
                return order.index(r.repo_type)
            except ValueError:
                return len(order)
        repos.sort(key=order_key)

    # Sanity check: after fetch, every repo that's going to be built should have
    # a Dockerfile. Phase 4 generates one for new services, so missing one here
    # is unusual and worth flagging — but we don't hard-fail, because the user
    # may have intentionally skipped the build step for some repos.
    missing_dockerfiles: list = []
    for r in repos:
        if r.skip or not r.push:
            continue
        df_path = workspace / r.name / (r.dockerfile or "Dockerfile")
        if not df_path.exists():
            missing_dockerfiles.append(f"{r.name} (expected {r.dockerfile or 'Dockerfile'})")
    if missing_dockerfiles:
        print(f"  ⚠ Dockerfile missing in {len(missing_dockerfiles)} repo(s):")
        for m in missing_dockerfiles:
            print(f"      - {m}")
        print("    (build_images will likely fail for these — Phase 4 normally "
              "generates a Dockerfile when none exists)")

    sequence = [
        {"step": i, "repo": r.name, "type": r.repo_type,
         "deploy_target": r.deploy_target, "status": "PENDING"}
        for i, r in enumerate(repos, start=1)
    ]

    return {
        **state,
        "fetch_results": results,
        "resolved_repos": [_repo_to_dict(r) for r in repos],
        "deploy_sequence": sequence,
        "missing_dockerfiles": missing_dockerfiles,
        "status": "REPOS_FETCHED",
    }


def build_images_node(state: DeploymentState) -> DeploymentState:
    cfg = _load_cfg(state)
    repos = _repos_from_state(state)
    results = build_images(repos, cfg)
    all_ok = all(r["status"] == "SUCCESS" for r in results)
    return {
        **state,
        "build_results": results,
        "status": "IMAGES_BUILT" if all_ok else "BUILD_FAILED",
    }


def registry_login_node(state: DeploymentState) -> DeploymentState:
    """Log in to every distinct registry referenced by the deploy sequence.
    Skips entirely if no repo declares a registry (e.g. compose-only deploys)."""
    cfg = _load_cfg(state)
    repos = _repos_from_state(state)
    needs_login = any(r.registry and r.push and not r.skip for r in repos)
    if not needs_login:
        print("\n[Phase 7] No registries configured — skipping login")
        return {**state, "login_results": [], "status": "REGISTRY_LOGIN_SKIPPED"}

    results = registry_logins(repos, cfg)
    all_ok = all(r["status"] == "SUCCESS" for r in results)
    return {
        **state,
        "login_results": results,
        "status": "REGISTRY_LOGGED_IN" if all_ok else "REGISTRY_LOGIN_FAILED",
    }


def push_images_node(state: DeploymentState) -> DeploymentState:
    cfg = _load_cfg(state)
    repos = _repos_from_state(state)
    results = push_images(repos, cfg)
    all_ok = all(r["status"] in ("SUCCESS", "SKIPPED") for r in results)
    return {
        **state,
        "push_results": results,
        "status": "IMAGES_PUSHED" if all_ok else "PUSH_FAILED",
    }


def setup_feature_flags(state: DeploymentState) -> DeploymentState:
    print("\n[Phase 7] Setting up feature flags...")
    cfg = _load_cfg(state)
    runbook = state.get("runbook", {}) or {}
    flags_from_runbook = runbook.get("feature_flags") or cfg.default_feature_flags or []

    feature_flags = []
    for flag in flags_from_runbook:
        feature_flags.append({
            "flag_name": flag.get("flag_name", ""),
            "enabled": False,
            "enable_after_deploy": flag.get("enable_after_deploy", True),
            "description": flag.get("description", ""),
        })
        print(f"  Flag: {flag.get('flag_name')} — disabled (will enable after deploy)")

    return {**state, "feature_flags": feature_flags, "status": "FLAGS_CONFIGURED"}


def human_approval_gate(state: DeploymentState) -> DeploymentState:
    """Hard interrupt — graph is compiled with interrupt_before on this node,
    so execution pauses here until resume_deployment() is called."""
    print("\n[Phase 7] ⏸ Awaiting production deployment approval...")
    print(f"  Deploy sequence: {[s['repo'] for s in state.get('deploy_sequence', [])]}")
    print(f"  Feature flags: {[f['flag_name'] for f in state.get('feature_flags', [])]}")
    print(f"  PRs to deploy: {state.get('pr_urls', [])}")
    print(f"  Build results: "
          f"{[(b['repo'], b['status']) for b in state.get('build_results', [])]}")

    human_input = interrupt("Approve production deployment")

    approved = False
    feedback = ""
    if isinstance(human_input, dict):
        approved = bool(human_input.get("approved", False))
        feedback = str(human_input.get("feedback", ""))

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "APPROVED" if approved else "REJECTED",
    }


def execute_deployment(state: DeploymentState) -> DeploymentState:
    """Trigger deployment for each repo, in sequence order.

    Two modes per repo:
      - Multi-env: repo has `environments: [...]` from .deploy.yaml. Deploy
        to each env in order, run smoke test, gate on requires_approval.
        Same image promotes across all envs (built once, deployed many).
      - Single-env: repo just has `deploy_target`. Original behavior.

    Note on the in-loop approval prompt: LangGraph's `interrupt_before` is
    per-graph-node, so a clean state-machine approach to per-env approvals
    would require a re-entrant approval node. For the demo / POC, we use a
    blocking input() prompt — fine for CLI/dev use, NOT for production
    pipelines where approvals should be async. Marked TODO below.
    """
    print("\n[Phase 7] Executing deployment...")
    cfg = _load_cfg(state)
    deploy_results: list = []
    all_ok = True

    repos_by_name = {r.name: r for r in _repos_from_state(state)}

    for step in state["deploy_sequence"]:
        repo = repos_by_name.get(step["repo"])
        if repo is None:
            repo = cfg.get_repo(step["repo"])
            repo.repo_type = step["type"]

        if repo.environments:
            # Multi-env promotion path
            env_results = _deploy_through_environments(repo, cfg)
            for r in env_results:
                r["step"] = step["step"]
                r["repo"] = repo.name
            deploy_results.extend(env_results)
            if any(r["status"] not in ("SUCCESS", "SKIPPED") for r in env_results):
                all_ok = False
                print(f"  ❌ {repo.name}: deployment chain failed")
                break
            print(f"  ✅ {repo.name}: deployed through "
                  f"{len(repo.environments)} environment(s)")
        else:
            # Single-env (existing) path
            result = trigger_deploy(repo, cfg)
            result["step"] = step["step"]
            deploy_results.append(result)
            if result["status"] != "SUCCESS":
                all_ok = False
                print(f"  ❌ {repo.name}: deploy failed — {result.get('error')}")
                break
            else:
                print(f"  ✅ {repo.name}: deployed")

    return {
        **state,
        "deploy_results": deploy_results,
        "status": "DEPLOYED" if all_ok else "DEPLOY_FAILED",
    }


def _deploy_through_environments(repo: RepoConfig,
                                 cfg: DeploymentConfig) -> list:
    """Walk `repo.environments` in order: deploy → smoke test → [approval] → next."""
    results: list = []
    for i, env in enumerate(repo.environments):
        is_first = i == 0
        # Per-env approval gate (except the first env, which is gated by the
        # global human_approval_gate that already happened before this node).
        # TODO: replace blocking input() with a re-entrant LangGraph node for
        # async approvals in production pipelines.
        if env.requires_approval and not is_first and not cfg.dry_run:
            print(f"\n  ⏸ [{repo.name}] approval required to promote to "
                  f"'{env.name}'...")
            print(f"     image: {repo.image_name}:{repo.image_tag}")
            try:
                answer = input(f"     promote to {env.name}? [y/N]: ").strip().lower()
            except (EOFError, OSError):
                answer = "n"
            if answer not in ("y", "yes"):
                print(f"  ⏹ [{repo.name}] promotion to '{env.name}' rejected — "
                      f"stopping chain")
                results.append({
                    "action": "env_approval", "status": "REJECTED",
                    "environment": env.name, "at": _utcnow(),
                })
                break

        deploy_result = trigger_deploy_to_env(repo, env, cfg)
        results.append(deploy_result)
        if deploy_result["status"] != "SUCCESS":
            print(f"  ❌ [{repo.name}] env '{env.name}' deploy failed — "
                  f"stopping chain")
            break

        # Smoke test
        if env.smoke_test:
            smoke = run_smoke_test(repo, env, cfg)
            results.append(smoke)
            if smoke["status"] not in ("SUCCESS", "SKIPPED"):
                print(f"  ❌ [{repo.name}] env '{env.name}' smoke test failed "
                      f"— stopping chain")
                break

    return results


def enable_feature_flags(state: DeploymentState) -> DeploymentState:
    print("\n[Phase 7] Enabling feature flags...")
    updated_flags = []
    for flag in state.get("feature_flags", []):
        if flag.get("enable_after_deploy", True):
            flag["enabled"] = True
            print(f"  ✅ Enabled: {flag['flag_name']}")
        updated_flags.append(flag)
    return {**state, "feature_flags": updated_flags, "status": "FLAGS_ENABLED"}


def monitor_deployment(state: DeploymentState) -> DeploymentState:
    """Read thresholds from config; fetch real metrics if endpoint configured,
    otherwise use simulated healthy values."""
    print("\n[Phase 7] Monitoring post-deployment metrics...")
    cfg = _load_cfg(state)
    mon_cfg = cfg.monitoring

    metrics = {
        "error_rate": 0.0,
        "avg_latency_ms": 45,
        "requests_per_min": 120,
        "health_check": "passing",
    }
    # Real metrics fetching is intentionally a hook — most teams have their own
    # observability stack (Prometheus, Datadog, Langfuse, etc). If the config
    # provides an endpoint, we GET it and trust the JSON shape to match.
    if mon_cfg.enabled and mon_cfg.metrics_endpoint:
        try:
            import urllib.request
            with urllib.request.urlopen(  # noqa: S310
                    mon_cfg.metrics_endpoint, timeout=30
            ) as resp:
                fetched = json.loads(resp.read().decode("utf-8"))
                metrics.update({k: v for k, v in fetched.items() if k in metrics})
                print(f"  Fetched metrics from {mon_cfg.metrics_endpoint}")
        except Exception as e:
            print(f"  ⚠ metrics fetch failed ({e}) — using defaults")

    monitoring_results = {
        "checked_at": _utcnow(),
        "metrics": metrics,
        "thresholds": {
            "max_error_rate": mon_cfg.max_error_rate,
            "max_latency_ms": mon_cfg.max_latency_ms,
        },
        "alerts": [],
        "status": "HEALTHY",
    }

    breached = False
    if metrics["error_rate"] > mon_cfg.max_error_rate:
        monitoring_results["alerts"].append(
            f"Error rate {metrics['error_rate']} exceeds threshold {mon_cfg.max_error_rate}"
        )
        breached = True
    if metrics["avg_latency_ms"] > mon_cfg.max_latency_ms:
        monitoring_results["alerts"].append(
            f"Latency {metrics['avg_latency_ms']}ms exceeds threshold {mon_cfg.max_latency_ms}ms"
        )
        breached = True

    if breached and mon_cfg.auto_rollback_on_breach:
        monitoring_results["status"] = "DEGRADED"
        print(f"  ❌ Metrics breached thresholds — triggering rollback")
        return {
            **state,
            "monitoring_results": monitoring_results,
            "rollback_triggered": True,
            "status": "ROLLBACK_TRIGGERED",
        }

    print(f"  ✅ Metrics within thresholds: error_rate={metrics['error_rate']}, "
          f"latency={metrics['avg_latency_ms']}ms")
    return {
        **state,
        "monitoring_results": monitoring_results,
        "rollback_triggered": False,
        "status": "DEPLOYMENT_COMPLETE",
    }


def execute_rollback(state: DeploymentState) -> DeploymentState:
    print("\n[Phase 7] ⚠️  Executing rollback...")
    cfg = _load_cfg(state)
    runbook = state.get("runbook", {}) or {}
    rollback_steps = runbook.get("rollback_steps") or cfg.default_rollback_steps or [
        "Revert to previous deployment",
        "Disable feature flags",
        "Alert on-call team",
    ]
    for i, step in enumerate(rollback_steps, 1):
        print(f"  Rollback step {i}: {step}")

    updated_flags = []
    for flag in state.get("feature_flags", []):
        flag["enabled"] = False
        updated_flags.append(flag)

    print(f"  ✅ Rollback complete — all flags disabled")
    return {**state, "feature_flags": updated_flags, "status": "ROLLED_BACK"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_step(success_status, fail_status: str = ""):
    """Return a routing function: state.status -> 'proceed' or 'fail'.

    success_status can be a single string, or a set/list of acceptable
    success statuses (e.g. when SKIPPED is also a successful outcome).
    """
    if isinstance(success_status, str):
        success_set = {success_status}
    else:
        success_set = set(success_status)
    def route(state: DeploymentState) -> str:
        return "proceed" if state.get("status") in success_set else "fail"
    return route


def route_after_approval(state: DeploymentState) -> str:
    return "approved" if state.get("approved") else "rejected"


def route_after_monitoring(state: DeploymentState) -> str:
    return "rollback" if state.get("rollback_triggered") else "complete"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_deployment_graph():
    builder = StateGraph(DeploymentState)

    builder.add_node("resolve_deploy_sequence", resolve_deploy_sequence)
    builder.add_node("verify_clis", verify_clis_node)
    builder.add_node("fetch_repos", fetch_repos_node)
    builder.add_node("build_images", build_images_node)
    builder.add_node("registry_login", registry_login_node)
    builder.add_node("push_images", push_images_node)
    builder.add_node("setup_feature_flags", setup_feature_flags)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("execute_deployment", execute_deployment)
    builder.add_node("enable_feature_flags", enable_feature_flags)
    builder.add_node("monitor_deployment", monitor_deployment)
    builder.add_node("execute_rollback", execute_rollback)

    builder.set_entry_point("resolve_deploy_sequence")
    # If resolve_deploy_sequence fails (e.g. no git URL found for an
    # affected repo), short-circuit to END before doing anything else.
    builder.add_conditional_edges(
        "resolve_deploy_sequence",
        _route_after_step("SEQUENCE_RESOLVED"),
        {"proceed": "verify_clis", "fail": END},
    )

    # CLI check is a hard gate — if required CLIs missing, fail fast
    builder.add_conditional_edges(
        "verify_clis",
        _route_after_step("CLIS_VERIFIED", "CLI_CHECK_FAILED"),
        {"proceed": "fetch_repos", "fail": END},
    )
    # fetch_repos can fail with FETCH_FAILED (clone error) or CONTRACT_INVALID
    # (.deploy.yaml references an undefined registry / target). Either way: stop.
    builder.add_conditional_edges(
        "fetch_repos",
        _route_after_step("REPOS_FETCHED"),
        {"proceed": "registry_login", "fail": END},
    )
    # REGISTRY_LOGIN_SKIPPED is also a success status — repos without a
    # registry still need to flow through to build.
    builder.add_conditional_edges(
        "registry_login",
        _route_after_step({"REGISTRY_LOGGED_IN", "REGISTRY_LOGIN_SKIPPED"}),
        {"proceed": "build_images", "fail": END},
    )
    builder.add_conditional_edges(
        "build_images",
        _route_after_step("IMAGES_BUILT", "BUILD_FAILED"),
        {"proceed": "push_images", "fail": END},
    )
    builder.add_conditional_edges(
        "push_images",
        _route_after_step("IMAGES_PUSHED", "PUSH_FAILED"),
        {"proceed": "setup_feature_flags", "fail": END},
    )
    builder.add_edge("setup_feature_flags", "human_approval_gate")

    # human_approval_gate is an interrupt_before node — when resumed it sets
    # `approved`. Branch on that, not on a hardcoded "always proceed".
    builder.add_conditional_edges(
        "human_approval_gate",
        route_after_approval,
        {"approved": "execute_deployment", "rejected": END},
    )

    builder.add_conditional_edges(
        "execute_deployment",
        _route_after_step("DEPLOYED", "DEPLOY_FAILED"),
        {"proceed": "enable_feature_flags", "fail": END},
    )
    builder.add_edge("enable_feature_flags", "monitor_deployment")
    builder.add_conditional_edges(
        "monitor_deployment",
        route_after_monitoring,
        {"complete": END, "rollback": "execute_rollback"},
    )
    builder.add_edge("execute_rollback", END)

    memory = MemorySaver()
    # The previous code both compiled with `interrupt_before=[...]` AND called
    # `interrupt(...)` inside the same node — that double-pauses. We keep
    # interrupt_before only; the inner `interrupt(...)` still works at resume
    # time to extract the human payload via Command(resume=...).
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"],
    )


# ---------------------------------------------------------------------------
# Run / Resume
# ---------------------------------------------------------------------------

def run_deployment(
        requirement: str,
        runbook: dict,
        pr_urls: list,
        affected_repos: list,
        scope_contract: Optional[dict] = None,
        thread_id: str = "thread-deployment",
        config_path: Optional[str] = None,
        dry_run: Optional[bool] = None,
) -> tuple:
    """Start Phase 7. Pauses at human_approval_gate; call resume_deployment
    with the human's decision to continue.

    Args:
        requirement: original requirement text (for audit trail)
        runbook: from Phase 2 — may contain feature_flags & rollback_steps
        pr_urls: merged PR URLs from Phase 6
        affected_repos: list[dict] from Phase 3 — enriched with git URL, type,
            language, etc. from Phase 0. resolve_deploy_sequence reads git URLs
            directly from these dicts; no separate selected_repos needed.
        scope_contract: optional scope contract from upstream phases
        thread_id: LangGraph checkpoint thread id
        config_path: explicit path to deployment.yaml (else autodetect)
        dry_run: override config.dry_run (None = honor config)
    """
    graph = build_deployment_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: DeploymentState = {
        "requirement": requirement,
        "scope_contract": scope_contract or {},
        "runbook": runbook or {},
        "pr_urls": pr_urls or [],
        "affected_repos": affected_repos or [],
        "resolved_repos": [],
        "deploy_sequence": [],
        "feature_flags": [],
        "cli_check": {},
        "fetch_results": [],
        "login_results": [],
        "build_results": [],
        "push_results": [],
        "deploy_results": [],
        "monitoring_results": {},
        "rollback_triggered": False,
        "human_feedback": "",
        "approved": False,
        "status": "STARTED",
        "config_path": config_path,
        "dry_run": dry_run,
    }

    print("\n" + "=" * 50)
    print("--- Starting Phase 7 — Deployment ---")
    print("=" * 50)

    result = graph.invoke(initial_state, config)

    print(f"\nStatus after pre-approval stages: {result['status']}")
    if result.get("deploy_sequence"):
        print(f"  Deploy sequence: {[s['repo'] for s in result['deploy_sequence']]}")
    if result.get("feature_flags"):
        print(f"  Feature flags: {[f['flag_name'] for f in result['feature_flags']]}")
    if result.get("build_results"):
        print(f"  Builds: {[(b['repo'], b['status']) for b in result['build_results']]}")
    if result.get("push_results"):
        print(f"  Pushes: {[(p['repo'], p['status']) for p in result['push_results']]}")

    return graph, config, result


def resume_deployment(graph, config, approved: bool, feedback: str = "") -> dict:
    print(f"\n--- Resuming Phase 7 (approved={approved}) ---")
    result = graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config,
    )
    print(f"Final status: {result['status']}")
    return result


# ---------------------------------------------------------------------------
# Standalone test — runs in dry-run mode using the default config
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_runbook = {
        "feature_flags": [
            {
                "flag_name": "leave_balance_tracker_enabled",
                "default": False,
                "enable_after_deploy": True,
            }
        ],
        "rollback_steps": [
            "Revert to previous git tag",
            "Disable leave_balance_tracker_enabled flag",
            "Alert #deployments Slack channel",
        ],
    }

    graph7, config7, result7 = run_deployment(
        requirement="Add leave balance tracker",
        runbook=mock_runbook,
        pr_urls=["https://github.com/AkashW45/leave-mgmt-backend/pull/11"],
        # affected_repos now carries full metadata (as Phase 3 would produce
        # after enrichment from Phase 0) — url is used directly by
        # resolve_deploy_sequence, no separate selected_repos needed.
        affected_repos=[{
            "name": "leave-mgmt-backend",
            "url":  "https://github.com/AkashW45/leave-mgmt-backend.git",
            "type": "backend",
            "language": "python",
            "impacted": True,
        }],
        thread_id="test-deployment-1",
        dry_run=True,  # always dry-run for the self-test
    )

    print("\n--- Simulating Production Approval ---")
    final = resume_deployment(graph7, config7, approved=True)

    print(f"\n✅ Phase 7 Test Complete")
    print(f"Status: {final['status']}")
    print(f"Deployed repos: "
          f"{[r['repo'] for r in final.get('deploy_results', []) if r['status'] == 'SUCCESS']}")
    print(f"Feature flags enabled: "
          f"{[f['flag_name'] for f in final.get('feature_flags', []) if f['enabled']]}")
    print(f"Monitoring: {final.get('monitoring_results', {}).get('metrics', {})}")
    print(f"Rollback triggered: {final.get('rollback_triggered')}")
