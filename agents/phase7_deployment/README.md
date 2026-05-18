# Phase 7 — Deployment

Config-driven deployment with **decentralized repo ownership**: each
service repo carries its own `.deploy.yaml` contract; the central platform
config only declares the infrastructure (registries, AWS targets,
required CLIs). Adding a new repo to the deploy system does NOT require
editing the central config.

## Architecture

Three layers, clear ownership:

1. **Platform config** (`config/deployment.yaml`) — owned by platform team.
   Lists *what infrastructure exists* (registries, deploy targets, default
   CLIs). Does NOT enumerate every repo.

2. **Per-repo deploy contract** (`.deploy.yaml` in each repo) — owned by
   each service team. Says *what this repo wants*: kind, type, build
   args, which platform target, etc.

3. **Resolver** — bridges the two: when Phase 7 sees a repo name in
   `affected_repos`, it (a) figures out the git URL via the resolver
   template, (b) clones the repo, (c) reads the in-repo contract, (d)
   merges with platform defaults.

## Flow

```
resolve_deploy_sequence  ← takes `affected_repos`, resolves git URLs
  → verify_clis          ← installs missing CLIs if cli_auto_install: true
  → fetch_repos          ← git clone, then merge each repo's .deploy.yaml
                           (re-sorts deploy_sequence if any type changed)
  → registry_login       ← ECR / Docker Hub / GHCR
  → build_images         ← docker build with merged settings
  → push_images          ← docker push to configured registry
  → setup_feature_flags
  → human_approval_gate  ← interrupt; operator confirms
  → execute_deployment   ← aws CLI: ECS update-service / Lambda update-function-code
                           uses per-repo overrides from .deploy.yaml deploy: block
  → enable_feature_flags
  → monitor_deployment
  → [rollback?]
```

Every pre-approval stage is a hard gate.

## Files

| File | Purpose |
|------|---------|
| `agents/phase7_deployment/deployment_agent.py` | LangGraph state machine |
| `core/deployment_config.py` | Config loader + contract merge logic |
| `core/deployment_executor.py` | git, docker, registry, AWS shell calls |
| `config/deployment.yaml` | **Platform config — infrastructure only** |
| `examples/deploy-contract-*.yaml` | Example `.deploy.yaml` files for repos |

## Platform config (`config/deployment.yaml`)

This file declares infrastructure, not repos. The new sections are:

```yaml
# How to find a repo's git URL from just its name
repo_resolver:
  url_template: "https://github.com/{org}/{name}.git"
  org: ${GITHUB_REPO_OWNER}

# Defaults applied to every repo unless its .deploy.yaml overrides
defaults:
  branch: main
  dockerfile: Dockerfile
  registry: ecr-prod
  deploy_target: ecs-prod
  image_tag: ${BUILD_TAG:-latest}
  deploy_contract_path: .deploy.yaml

registries:
  - name: ecr-prod
    kind: ecr
    region: ${AWS_REGION}
    account_id: ${AWS_ACCOUNT_ID}

deploy_targets:
  - name: ecs-prod
    kind: ecs
    config:
      region: ${AWS_REGION}
      cluster: my-cluster
      # service name comes from each repo's .deploy.yaml

# Legacy escape hatch — pre-declare repos here only when the in-repo
# pattern doesn't fit. Empty in the new pattern.
repos: []
```

**Adding new infrastructure** (new registry, new ECS cluster, new env)
→ edit this file.

**Adding a new repo** → don't touch this file. Just add `.deploy.yaml`
to the repo.

## Per-repo contract (`.deploy.yaml`)

Lives in the root of each service repo. Every field optional.

```yaml
kind: service                # service | library | batch
type: backend                # repo_type override
dockerfile: Dockerfile
build_args:
  BUILD_ENV: production

registry: ecr-prod           # references platform registries[].name
push: true

deploy:
  target: ecs-prod           # references platform deploy_targets[].name
  # Any extra keys here override the platform target's config for this repo
  service: my-app-svc
  cluster: special-cluster   # optional override
```

See `examples/` for service, lambda, and library variants.

**Resolution precedence** (highest to lowest):

1. Legacy `repos:` entry in `config/deployment.yaml` (escape hatch)
2. Repo's own `.deploy.yaml`
3. Platform `defaults:` block
4. Built-in dataclass defaults

## CLI auto-install

```yaml
cli_auto_install: true
required_clis:
  - name: docker
    install_command: "curl -fsSL https://get.docker.com | sh"
  - name: aws
    install_command: "pip install --break-system-packages awscli"
```

Off by default. Auto-installing system packages on a shared host is
destructive — turn it on for ephemeral CI runners only.

## Registries

- **ECR**: `aws ecr get-login-password` piped to `docker login --password-stdin`.
  URL auto-derived from account_id + region. ECR repo auto-created on
  first push.
- **Docker Hub / GHCR / generic**: username + password from config
  (env-interpolated), piped to `docker login --password-stdin`.

Build dual-tags every image: local (`name:tag`) + registry-qualified
(`<registry>/<image>:<tag>`).

## AWS deploy targets

### ECS

```yaml
# Platform config — shared across services in the cluster
- name: ecs-prod
  kind: ecs
  config:
    region: us-east-1
    cluster: my-cluster
    wait_for_stable: true
```

```yaml
# In each service's .deploy.yaml
deploy:
  target: ecs-prod
  service: leave-mgmt-backend-svc          # this service's name
  task_definition_template: config/ecs/backend-taskdef.json   # optional
```

Two modes (selected by presence of `task_definition_template`):

1. **With template**: render JSON, register new task def revision,
   `update-service` pointing at it. Immutable image refs, audit log.
2. **Without template**: just `update-service --force-new-deployment`.

### Lambda

```yaml
# Platform config — region-wide
- name: lambda-prod
  kind: lambda
  config:
    region: us-east-1
    wait_for_updated: true
```

```yaml
# In each Lambda's .deploy.yaml
deploy:
  target: lambda-prod
  function_name: my-worker
  alias: production              # optional
```

## Other deploy targets

`compose`, `kubernetes` / `k8s`, `webhook` (n8n / TeamCity / Jenkins),
`shell` (escape hatch), `dry-run`. All accept per-repo overrides via the
`deploy:` block in `.deploy.yaml`.

## Multi-environment promotion

For services that need staging→prod (or dev→staging→canary→prod), declare
an `environments:` list in `.deploy.yaml` instead of a single `target:`.

```yaml
deploy:
  environments:
    - name: staging
      target: ecs-staging
      overrides:
        service: my-app-staging-svc
      smoke_test:
        url: https://staging.my-app.example.com/health
        retries: 5

    - name: prod
      target: ecs-prod
      overrides:
        service: my-app-prod-svc
      requires_approval: true
      smoke_test:
        url: https://my-app.example.com/health
```

**Key invariant:** the image is built and pushed ONCE. Promotion is just
re-deploying the same image to different targets. No rebuilds between envs.

**Flow per environment:**
1. Deploy to env (`aws ecs update-service`, etc.)
2. Smoke test (HTTP GET or shell command) — required to pass before next env
3. If next env has `requires_approval: true`, prompt operator

**Smoke test modes** (pick one per env):
- `url:` + `expect_status:` — HTTP GET, must return expected status
- `command:` — shell command (via `sh -c`), must exit 0

Both honor `retries`, `retry_delay_seconds`, `timeout_seconds`.

See `examples/deploy-contract-multi-env.yaml`.

**Caveat:** per-env approval currently uses a blocking `input()` prompt
inside `execute_deployment`. Fine for CLI/dev use, not for async pipelines.
A re-entrant LangGraph approval node is the right fix for production.

## Dry-run

```yaml
dry_run: true
```

or `run_deployment(..., dry_run=True)`. Every shell command, HTTP POST,
docker / aws / kubectl call is logged as `[dry-run] would run: ...` —
graph walks every node but executes nothing. Always do this first on a
new config.

## Adding a new repo to the deploy pipeline

Old way:
1. Edit `config/deployment.yaml` to add a `repos:` entry
2. Get PR review from platform team
3. Wait for merge
4. Pipeline can deploy

New way:
1. Add `.deploy.yaml` to your repo
2. Pipeline can deploy

## Result shape

After `resume_deployment(...)`, state contains:

```python
{
  "resolved_repos":   [{repo dicts, with .deploy.yaml overrides merged}],
  "cli_check":        {"checks": [...], "auto_installed": [...], "status": "OK"},
  "fetch_results":    [{"action": "fetch_repo", "status": "SUCCESS", ...}],
  "login_results":    [{"action": "registry_login", "status": "SUCCESS", ...}],
  "build_results":    [{"action": "build_image", "status": "SUCCESS",
                        "local_tag": "...", "qualified_tag": "..."}],
  "push_results":     [{"action": "push_image", "status": "SUCCESS", ...}],
  "deploy_results":   [{"action": "trigger_deploy", "status": "SUCCESS",
                        "kind": "ecs", "cluster": "...", "service": "...",
                        "task_definition_arn": "..."}],
  "status":           "DEPLOYMENT_COMPLETE",
}
```
