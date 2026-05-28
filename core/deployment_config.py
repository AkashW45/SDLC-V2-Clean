"""
Deployment configuration loader.

Loads a YAML/JSON deployment config that describes:
  - Per-repo settings: git URL, default branch, docker build args, deploy target
  - Required CLI tools and their version checks
  - Deploy targets (compose / kubernetes / teamcity-webhook / shell)
  - Classification rules (replaces the hardcoded `"lib" in r` substring checks)
  - Monitoring thresholds

The config is intentionally separate from the agent so the same agent can
deploy any project just by swapping the config file. Env-var interpolation
(`${VAR}` or `${VAR:-default}`) keeps secrets out of the YAML.

Lookup order for the config path:
  1. Explicit `config_path` arg to `load_deployment_config()`
  2. `$SDLC_DEPLOY_CONFIG` env var
  3. `config/deployment.yaml` relative to the project root
  4. `config/deployment.yml`
  5. `config/deployment.json`
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CLIRequirement:
    """A CLI tool that must be present on the deploy host."""
    name: str                       # e.g. "docker"
    version_command: str = ""       # e.g. "docker --version"; empty = just check presence
    install_hint: str = ""          # human-readable install instructions
    install_command: str = ""       # shell command to install; only run when cli_auto_install=true
    required: bool = True           # if False, log a warning instead of failing


@dataclass
class RegistryConfig:
    """A container registry to push to.

    `kind` selects the auth flow:
      - "ecr"        AWS ECR. Uses `aws ecr get-login-password` against region + account_id.
      - "dockerhub"  Docker Hub. Uses username + password (typically env-interpolated).
      - "ghcr"       GitHub Container Registry. Uses username + token.
      - "generic"    Any OCI registry; supply `url`, `username`, `password`.
    """
    name: str
    kind: str                                # "ecr" | "dockerhub" | "ghcr" | "generic"
    url: str = ""                            # auto-derived for ECR from account_id+region
    region: str = ""                         # ECR
    account_id: str = ""                     # ECR
    username: str = ""                       # dockerhub / ghcr / generic
    password: str = ""                       # env-interpolated; never hardcoded
    create_repository_if_missing: bool = True   # ECR convenience


@dataclass
class RepoResolver:
    """How to find a repo's git URL from just its name. Used when the repo
    isn't pre-declared in the legacy `repos:` block.

    `url_template`: a string with `{name}` and `{org}` placeholders, e.g.
        "https://github.com/{org}/{name}.git"
    `org`: the GitHub/GitLab org/user. Required when the template uses `{org}`.

    With both set, Phase 7 can deploy any repo just from its name — no need
    to pre-declare it in the central config.
    """
    url_template: str = ""
    org: str = ""

    def resolve(self, repo_name: str) -> str:
        """Render the template for a given repo name. Returns "" if no
        template is configured."""
        if not self.url_template:
            return ""
        return self.url_template.format(name=repo_name, org=self.org)


@dataclass
class RepoDefaults:
    """Platform-wide defaults applied to every repo unless the repo's own
    `.deploy.yaml` (or legacy `repos:` entry) overrides them. Keep these
    minimal — they're the "we agree across the org" knobs.
    """
    branch: str = "main"
    dockerfile: str = "Dockerfile"
    docker_context: str = "."
    registry: str = ""                        # default registry name
    deploy_target: str = ""                   # default deploy target name
    image_tag: str = "latest"
    deploy_contract_path: str = ".deploy.yaml"  # where to look in each repo


@dataclass
class SmokeTest:
    """Post-deploy smoke test to gate promotion to the next environment.

    Two modes (pick one — if both set, `command` wins):
      - HTTP GET: hit `url`, require status in `expect_status` (default 200).
      - Shell:    run `command` (via sh -c), require exit 0.

    `timeout_seconds` applies to both modes. `retries` lets transient
    failures resolve (e.g. service still warming up).
    """
    url: str = ""
    expect_status: int = 200
    command: str = ""
    timeout_seconds: int = 30
    retries: int = 3
    retry_delay_seconds: int = 10


@dataclass
class EnvironmentDeploy:
    """A single rung of a repo's promotion ladder.

    Each rung says: deploy THIS image to THAT target with these overrides,
    optionally smoke-test afterwards, optionally require a human approval
    before promoting to the next rung.
    """
    name: str                                 # e.g. "staging", "prod", "canary"
    target: str                               # references DeployTarget.name
    overrides: Dict[str, Any] = field(default_factory=dict)  # extra deploy.target.config overrides
    smoke_test: Optional[SmokeTest] = None
    requires_approval: bool = False           # human gate BEFORE this env deploys


@dataclass
class RepoConfig:
    """Per-repo deployment settings."""
    name: str
    git_url: str = ""
    branch: str = "main"
    ref: str = ""                   # commit/tag; overrides branch if set
    repo_type: str = "service"      # library | backend | frontend | batch | service
    dockerfile: str = "Dockerfile"
    docker_context: str = "."
    image_name: str = ""            # defaults to repo name
    image_tag: str = "latest"
    build_args: Dict[str, str] = field(default_factory=dict)
    registry: str = ""              # references RegistryConfig.name; empty = no push
    push: bool = True               # set false to build but skip push
    deploy_target: str = "default"  # references DeployTarget.name (single-env mode)
    environments: List[EnvironmentDeploy] = field(default_factory=list)  # multi-env mode
    skip: bool = False              # set true to exclude from this run


@dataclass
class DeployTarget:
    """How to deploy. Supported `kind` values:
      - "dry-run"     log only, never execute
      - "compose"     `docker compose up -d`
      - "kubernetes"  `kubectl set image` + `rollout status`
      - "webhook"     POST to URL (n8n / TeamCity / Jenkins)
      - "shell"       arbitrary command
      - "ecs"         AWS ECS service update via aws CLI
      - "lambda"      AWS Lambda update-function-code with a container image
    """
    name: str
    kind: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    enabled: bool = True
    metrics_endpoint: str = ""      # if empty, uses the simulated metrics
    max_error_rate: float = 0.05
    max_latency_ms: int = 500
    check_duration_seconds: int = 60
    auto_rollback_on_breach: bool = True


@dataclass
class ClassificationRules:
    """How to classify a repo by name when RepoConfig.repo_type is missing.

    Order matters — first match wins. Each rule is a list of substrings;
    any match assigns the type.
    """
    library: List[str] = field(default_factory=lambda: ["lib", "common", "shared", "sdk"])
    backend: List[str] = field(default_factory=lambda: ["backend", "api", "service", "server"])
    frontend: List[str] = field(default_factory=lambda: ["frontend", "ui", "web", "client"])
    batch: List[str] = field(default_factory=lambda: ["batch", "job", "worker", "cron"])

    # Deploy order — earlier types deploy first
    order: List[str] = field(default_factory=lambda: ["library", "backend", "frontend", "batch", "service"])


@dataclass
class DeploymentConfig:
    """Root deployment config."""
    project_name: str = "sdlc-project"
    workspace_dir: str = "/tmp/sdlc-deploy-workspace"
    required_clis: List[CLIRequirement] = field(default_factory=list)
    cli_auto_install: bool = False  # when true, run install_command for missing CLIs
    registries: Dict[str, RegistryConfig] = field(default_factory=dict)
    repos: Dict[str, RepoConfig] = field(default_factory=dict)  # legacy/fallback
    repo_resolver: RepoResolver = field(default_factory=RepoResolver)
    defaults: RepoDefaults = field(default_factory=RepoDefaults)
    deploy_targets: Dict[str, DeployTarget] = field(default_factory=dict)
    classification: ClassificationRules = field(default_factory=ClassificationRules)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    default_feature_flags: List[Dict[str, Any]] = field(default_factory=list)
    default_rollback_steps: List[str] = field(default_factory=list)
    dry_run: bool = False           # if True, log commands but don't execute
    deploy_parallelism: int = 4     # max concurrent repos per dependency level
    # (1 = fully sequential)

    def get_repo(self, repo_name: str) -> RepoConfig:
        """Get repo config by name. Returns a default config if not declared
        in the legacy `repos:` block. The default has the platform's defaults
        applied — caller can detect "not in legacy block" because the entry
        in `self.repos` won't exist."""
        if repo_name in self.repos:
            return self.repos[repo_name]
        # Build a default RepoConfig with platform defaults applied. Caller
        # is responsible for filling in git_url (via resolve_repo_url) and
        # merging the in-repo .deploy.yaml.
        return RepoConfig(
            name=repo_name,
            image_name=repo_name,
            branch=self.defaults.branch,
            dockerfile=self.defaults.dockerfile,
            docker_context=self.defaults.docker_context,
            image_tag=self.defaults.image_tag,
            registry=self.defaults.registry,
            deploy_target=self.defaults.deploy_target,
        )

    def resolve_repo_url(self, repo_name: str) -> str:
        """Look up the git URL for a repo name. Priority order:
          1. Legacy `repos:` block (if the repo is pre-declared with a URL)
          2. `repo_resolver` template (e.g. "https://github.com/{org}/{name}.git")
          3. Empty string — caller must handle this (no URL = can't deploy)
        """
        if repo_name in self.repos and self.repos[repo_name].git_url:
            return self.repos[repo_name].git_url
        return self.repo_resolver.resolve(repo_name)

    def get_registry(self, registry_name: str) -> Optional[RegistryConfig]:
        """Get registry config by name. Returns None if no registry is configured
        for this repo (which means: build locally, don't push)."""
        if not registry_name:
            return None
        return self.registries.get(registry_name)

    def classify_repo(self, repo: RepoConfig) -> str:
        """Return the repo_type, honoring an explicit setting then falling
        back to name-substring rules."""
        if repo.repo_type and repo.repo_type != "service":
            return repo.repo_type
        name = repo.name.lower()
        for rtype in ("library", "backend", "frontend", "batch"):
            keywords: List[str] = getattr(self.classification, rtype)
            if any(kw in name for kw in keywords):
                return rtype
        return "service"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace `${VAR}` or `${VAR:-default}` with env values."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _find_default_config() -> Optional[Path]:
    """Walk up from cwd looking for `config/deployment.{yaml,yml,json}`."""
    cwd = Path.cwd()
    candidates = []
    for parent in [cwd, *cwd.parents]:
        for name in ("deployment.yaml", "deployment.yml", "deployment.json"):
            candidates.append(parent / "config" / name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_raw(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError(
                f"Cannot load {path}: PyYAML is not installed. "
                f"`pip install pyyaml` or use a .json config."
            )
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_deployment_config(config_path: Optional[str] = None) -> DeploymentConfig:
    """Load deployment config from a file path, env, or default location.

    If no config file is found anywhere, returns a sensible default config
    that runs in dry-run mode so the pipeline never hard-fails on a missing
    file — instead Phase 7 will log what it *would* do.
    """
    path: Optional[Path] = None
    if config_path:
        path = Path(config_path)
    elif os.environ.get("SDLC_DEPLOY_CONFIG"):
        path = Path(os.environ["SDLC_DEPLOY_CONFIG"])
    else:
        path = _find_default_config()

    if path is None or not path.is_file():
        print(
            f"[deployment_config] No config file found "
            f"(searched arg, $SDLC_DEPLOY_CONFIG, config/deployment.{{yaml,yml,json}}). "
            f"Using built-in dry-run defaults."
        )
        return _default_dry_run_config()

    raw = _interpolate_env(_load_raw(path))
    return _parse(raw, source=str(path))


def _parse(raw: Dict[str, Any], source: str = "") -> DeploymentConfig:
    """Convert raw dict to DeploymentConfig with light validation."""
    cfg = DeploymentConfig()
    cfg.project_name = raw.get("project_name", cfg.project_name)
    cfg.workspace_dir = raw.get("workspace_dir", cfg.workspace_dir)
    cfg.dry_run = bool(raw.get("dry_run", False))
    cfg.deploy_parallelism = int(raw.get("deploy_parallelism", 4) or 4)
    cfg.cli_auto_install = bool(raw.get("cli_auto_install", False))

    cfg.required_clis = [
        CLIRequirement(**c) for c in raw.get("required_clis", [])
    ]

    cfg.registries = {}
    for reg_raw in raw.get("registries", []):
        if "name" not in reg_raw or "kind" not in reg_raw:
            raise ValueError(f"{source}: registry needs `name` and `kind`")
        reg = RegistryConfig(**reg_raw)
        # For ECR, derive URL from account_id + region if not given
        if reg.kind == "ecr" and not reg.url and reg.account_id and reg.region:
            reg.url = f"{reg.account_id}.dkr.ecr.{reg.region}.amazonaws.com"
        cfg.registries[reg.name] = reg

    if "repo_resolver" in raw:
        cfg.repo_resolver = RepoResolver(**raw["repo_resolver"])

    if "defaults" in raw:
        cfg.defaults = RepoDefaults(**raw["defaults"])
        if cfg.defaults.registry and cfg.defaults.registry not in cfg.registries:
            raise ValueError(
                f"{source}: defaults.registry '{cfg.defaults.registry}' "
                f"not in registries (defined: {list(cfg.registries)})"
            )

    cfg.repos = {}
    for repo_raw in raw.get("repos", []):
        if "name" not in repo_raw:
            raise ValueError(f"{source}: every repo must have a `name`")
        rc = RepoConfig(**repo_raw)
        if not rc.image_name:
            rc.image_name = rc.name
        if rc.registry and rc.registry not in cfg.registries:
            raise ValueError(
                f"{source}: repo '{rc.name}' references unknown registry "
                f"'{rc.registry}' (defined: {list(cfg.registries)})"
            )
        cfg.repos[rc.name] = rc

    cfg.deploy_targets = {}
    for tgt_raw in raw.get("deploy_targets", []):
        if "name" not in tgt_raw or "kind" not in tgt_raw:
            raise ValueError(f"{source}: deploy_target needs `name` and `kind`")
        tgt = DeployTarget(
            name=tgt_raw["name"],
            kind=tgt_raw["kind"],
            config=tgt_raw.get("config", {}),
        )
        cfg.deploy_targets[tgt.name] = tgt

    if "classification" in raw:
        cfg.classification = ClassificationRules(**raw["classification"])

    if "monitoring" in raw:
        cfg.monitoring = MonitoringConfig(**raw["monitoring"])

    cfg.default_feature_flags = raw.get("default_feature_flags", [])
    cfg.default_rollback_steps = raw.get("default_rollback_steps", [])

    print(f"[deployment_config] Loaded config from {source}")
    print(f"  project={cfg.project_name} legacy_repos={len(cfg.repos)} "
          f"resolver={'yes' if cfg.repo_resolver.url_template else 'no'} "
          f"registries={list(cfg.registries)} "
          f"targets={list(cfg.deploy_targets)} dry_run={cfg.dry_run}")
    return cfg


def _default_dry_run_config() -> DeploymentConfig:
    """Built-in fallback so the agent runs end-to-end without a config file."""
    return DeploymentConfig(
        project_name="sdlc-default",
        workspace_dir="/tmp/sdlc-deploy-workspace",
        required_clis=[
            CLIRequirement(name="git", version_command="git --version",
                           install_hint="apt-get install git / brew install git"),
            CLIRequirement(name="docker", version_command="docker --version",
                           install_hint="https://docs.docker.com/get-docker/"),
        ],
        deploy_targets={
            "default": DeployTarget(name="default", kind="dry-run", config={})
        },
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# In-repo deploy contract (`.deploy.yaml`)
# ---------------------------------------------------------------------------
#
# Each repo can carry its own deploy contract — a small YAML file at
# `.deploy.yaml` (path overridable via platform `defaults.deploy_contract_path`).
# This lets the platform team avoid declaring every repo centrally.
#
# Contract shape (all fields optional — anything missing falls back to
# platform defaults or the repo's legacy entry):
#
#   kind: service                 # service | library | batch
#   type: backend                 # repo_type override (backend/frontend/etc)
#   dockerfile: Dockerfile        # path inside repo
#   docker_context: .
#   image_name: my-app            # defaults to repo name
#   image_tag: "${BUILD_TAG:-latest}"
#   build_args:
#     BUILD_ENV: production
#   registry: ecr-prod            # references platform registries[].name
#   push: true
#   deploy:
#     target: ecs-prod            # references platform deploy_targets[].name
#     # Any extra fields here get merged into the deploy target's config
#     # for THIS repo only — e.g. for ECS:
#     service: my-app-svc
#     cluster: my-cluster         # override platform target's cluster
#
# Resolution order for any field: repo's .deploy.yaml > legacy `repos:` >
# platform `defaults` > built-in fallback.

def merge_repo_contract(
        base: RepoConfig,
        repo_path: "Path",
        platform: DeploymentConfig,
) -> RepoConfig:
    """Read `.deploy.yaml` (if present) from a cloned repo and merge it into
    the RepoConfig that was built from platform defaults + legacy block.

    Returns the merged RepoConfig. If no `.deploy.yaml` exists, returns `base`
    unchanged — the platform defaults still apply.

    The repo's contract WINS over platform defaults (repo owners know best
    what their service needs), but the legacy `repos:` block in the central
    config wins over the in-repo contract (intentional: it's an escape hatch
    for repos that haven't migrated to the new pattern yet — operator can
    force-override from the central config).
    """
    contract_path = repo_path / platform.defaults.deploy_contract_path
    if not contract_path.is_file():
        return base

    try:
        raw = _interpolate_env(_load_raw(contract_path))
    except Exception as e:
        print(f"  ⚠ [{base.name}] failed to read {contract_path}: {e}")
        return base

    if not isinstance(raw, dict):
        print(f"  ⚠ [{base.name}] {contract_path} is not a YAML mapping; ignoring")
        return base

    print(f"  📄 [{base.name}] merging in-repo {platform.defaults.deploy_contract_path}")

    # If this repo was pre-declared in the legacy `repos:` block, that wins.
    # Otherwise apply the in-repo contract on top of base.
    was_legacy = base.name in platform.repos
    merged = base
    if not was_legacy:
        # Apply scalar overrides from the contract
        merged.repo_type = raw.get("type", merged.repo_type)
        merged.dockerfile = raw.get("dockerfile", merged.dockerfile)
        merged.docker_context = raw.get("docker_context", merged.docker_context)
        merged.image_name = raw.get("image_name", merged.image_name)
        merged.image_tag = raw.get("image_tag", merged.image_tag)
        merged.registry = raw.get("registry", merged.registry)
        merged.push = bool(raw.get("push", merged.push))
        if "build_args" in raw and isinstance(raw["build_args"], dict):
            merged.build_args = {**merged.build_args, **raw["build_args"]}

        # Top-level deploy knobs that Phase 7's ALB/target-group setup reads via
        # _opt(). The CICD generator writes container_port / health_check_path at
        # the TOP LEVEL of .deploy.yaml (not inside a `deploy:` block), so we must
        # collect them here into _deploy_overrides — otherwise Phase 7 falls back
        # to the target's defaults (8000, /health), which for an nginx static
        # site (port 80, no /health route) means the health check fails forever
        # → target unhealthy → ECS crash-loop. This is exactly that bug.
        _top_level_overrides = {}
        for _k in ("container_port", "health_check_path",
                   "hc_interval_seconds", "hc_healthy_threshold",
                   "hc_timeout_seconds"):
            if _k in raw:
                _top_level_overrides[_k] = raw[_k]
        if _top_level_overrides:
            existing = getattr(merged, "_deploy_overrides", {}) or {}
            setattr(merged, "_deploy_overrides", {**existing, **_top_level_overrides})
        # Top-level deploy_target (the generator writes it at top level too).
        if "deploy_target" in raw:
            merged.deploy_target = raw["deploy_target"]

        # `kind: library` means: build but don't push or deploy.
        kind = raw.get("kind", "service").lower()
        if kind == "library":
            merged.push = False
            merged.deploy_target = "dry-run-only"
            merged.repo_type = merged.repo_type or "library"

        # Deploy block. Two shapes supported:
        #   1) Single-env: deploy.target + extra keys → deploy_target + overrides
        #   2) Multi-env:  deploy.environments[] → environments list
        # If both are present, environments wins.
        deploy = raw.get("deploy", {}) or {}
        envs_raw = deploy.get("environments")
        if isinstance(envs_raw, list) and envs_raw:
            parsed_envs: List[EnvironmentDeploy] = []
            for env_raw in envs_raw:
                if "target" not in env_raw:
                    raise ValueError(
                        f"repo '{merged.name}': each environment needs a `target`"
                    )
                smoke_raw = env_raw.get("smoke_test")
                smoke = SmokeTest(**smoke_raw) if isinstance(smoke_raw, dict) else None
                parsed_envs.append(EnvironmentDeploy(
                    name=env_raw.get("name", env_raw["target"]),
                    target=env_raw["target"],
                    overrides=env_raw.get("overrides", {}) or {},
                    smoke_test=smoke,
                    requires_approval=bool(env_raw.get("requires_approval", False)),
                ))
            merged.environments = parsed_envs
            # Use the LAST env's target as the legacy single deploy_target
            # — keeps downstream code that reads deploy_target happy.
            merged.deploy_target = parsed_envs[-1].target
        elif "target" in deploy:
            # Single-env mode (existing behavior)
            merged.deploy_target = deploy["target"]
            deploy_overrides = {
                k: v for k, v in deploy.items()
                if k not in ("target", "environments")
            }
            if deploy_overrides:
                existing = getattr(merged, "_deploy_overrides", {}) or {}
                setattr(merged, "_deploy_overrides", {**existing, **deploy_overrides})

    # Validation: registry must exist in platform config (if set)
    if merged.registry and merged.registry not in platform.registries:
        raise ValueError(
            f"repo '{merged.name}': .deploy.yaml references unknown "
            f"registry '{merged.registry}' "
            f"(defined: {list(platform.registries)})"
        )
    if merged.deploy_target and merged.deploy_target not in platform.deploy_targets:
        raise ValueError(
            f"repo '{merged.name}': .deploy.yaml references unknown "
            f"deploy_target '{merged.deploy_target}' "
            f"(defined: {list(platform.deploy_targets)})"
        )
    for env in merged.environments:
        if env.target not in platform.deploy_targets:
            raise ValueError(
                f"repo '{merged.name}': environment '{env.name}' references "
                f"unknown deploy_target '{env.target}' "
                f"(defined: {list(platform.deploy_targets)})"
            )

    return merged
