# Phase 4 → Phase 7 CI/CD Pipeline Integration

## What changed

Phase 4 now generates deployment artifacts as part of code generation. Phase 7
then consumes them to deploy to AWS infrastructure. No tests are executed in
either phase — only structural validation.

## New files

| File | Purpose |
|---|---|
| `agents/phase4_codegen/cicd_generator.py` | LLM-aided decision + deterministic template synthesis for Dockerfile, `.deploy.yaml`, and GitHub Actions workflow |

## Modified files

| File | Change |
|---|---|
| `agents/phase4_codegen/codegen_agent.py` | Added `generate_cicd_node` and `validate_cicd_node` to the graph; fixed `route_after_validation` to accept real success statuses; added `cicd_decision` and `cicd_warnings` to state |
| `agents/phase7_deployment/deployment_agent.py` | Removed broken `llm_gateway` import (matches Phases 1–6 pattern); `fetch_repos_node` now warns when a Dockerfile is missing |
| `api/test_cases_export.py` | Removed dead `llm_gateway` import (was triggering crash on missing `LLM_API_KEY`) |
| `config/deployment.yaml` | `cli_auto_install: true` so Phase 7 can install missing CLIs (docker, aws) on a fresh runner |

## What Phase 4 generates

When code generation produces a service-shaped project (long-running process,
network port, deployable), Phase 4 now also generates:

1. **`Dockerfile`** — multi-stage where appropriate; based on detected language
   (Python/Node/Java/Go); uses safe base images (`python:3.11-slim`,
   `node:20-alpine`, `eclipse-temurin:17`)
2. **`.dockerignore`** — keeps build context small and fast
3. **`.deploy.yaml`** — the in-repo deploy contract Phase 7 already reads;
   declares `deploy_target`, `service_name`, `registry`, `dockerfile`,
   `branch`, etc. (all keys Phase 7's `merge_repo_contract` understands)
4. **`.github/workflows/ci-cd.yml`** — GitHub Actions pipeline:
   - On PR: checkout, install deps, lint (non-fatal). **No tests run.**
   - On push to main: same plus AWS credentials, ECR login, build, push, ECS
     deploy with `--force-new-deployment`, wait for `services-stable`

## When Phase 4 skips generating

- Repo already has a `Dockerfile` (won't overwrite hand-tuned infra)
- Repo already has `.deploy.yaml`
- Repo already has `.github/workflows/*.yml`
- The project is non-deployable (library, docs, static frontend)

## How the decision is made

A single LLM call (DeepSeek, deterministic temperature 0.1) reads:
- the requirement text
- the ADR (architecture decisions)
- the architecture nodes (which services exist)
- a summary of the just-generated files
- a fingerprint of which deployment artifacts already exist

…and returns a JSON decision packet describing `language`, `framework`,
`deploy_target`, `service_name`, `container_port`, `start_command`, etc.

The actual **file content** is then synthesized from deterministic templates
keyed by that decision — NOT from free-form LLM output. This is intentional:
LLM-generated Dockerfiles hallucinate base images, COPY paths, and CMD
syntax with surprising frequency. Templates are reviewable, predictable, and
easy to fix in one place.

If the LLM call fails for any reason (network, parse error), a safe default
fires: if a Python file exists in the changeset, generate a Python/FastAPI
ECS service contract.

## Validation step

`validate_cicd_node` runs immediately after generation and checks:
- Dockerfile has `FROM` and (`CMD` or `ENTRYPOINT`)
- YAML files parse to mappings
- `.deploy.yaml` contains `deploy_target` and `service_name`
- GitHub workflow contains `on` and `jobs`

Failures here are recorded as warnings on `state.cicd_warnings` — they don't
fail the pipeline, because deployment artifacts being suboptimal shouldn't
kill an otherwise-successful codegen run.

**No tests are executed at any point** — per directive. CI workflow includes
a lint step but no `pytest`/`jest`/`mvn test` invocation.

## Phase 7 side — what changed

The existing Phase 7 already supported config-driven AWS deployment via
`core/deployment_executor.py`. The only changes needed:

1. **`cli_auto_install: true`** in `config/deployment.yaml` so missing
   `docker` / `aws` / `kubectl` get installed on the runner before the
   `docker build` step. Already supported in the code via the
   `install_command` field per CLI — we just flipped the global toggle on.

2. **Missing Dockerfile warning** in `fetch_repos_node`. After cloning each
   repo, we check whether the Dockerfile Phase 7 expects to find is actually
   there. If not, we print a warning (build_images will fail anyway, but the
   warning makes the cause explicit). This is purely diagnostic — Phase 7's
   build step still runs; this just makes the failure mode legible.

3. **Removed broken `from core.llm_gateway import gateway`** — the gateway
   abstraction wasn't wired up for this codebase's env vars (it reads
   `LLM_API_KEY` while everyone else uses `DEEPSEEK_API_KEY`). Phase 7 now
   uses the same direct-OpenAI pattern as Phases 1–6.

## End-to-end flow

```
Phase 4 codegen
  └─ generate code changes (existing or fresh)
  └─ generate_cicd ⬅️ NEW: produces Dockerfile, .deploy.yaml, workflow
  └─ validate_cicd ⬅️ NEW: structural checks, no tests
  └─ validate_changes (existing)

Phase 6 delivery
  └─ pushes generated files (now including Dockerfile + .deploy.yaml +
     workflow) to a feature branch, opens PR

Phase 7 deployment
  └─ resolve_deploy_sequence (existing)
  └─ verify_clis ⬅️ now auto-installs missing docker/aws/kubectl
  └─ fetch_repos ⬅️ reads the new .deploy.yaml from the cloned repo;
                    warns if Dockerfile is missing
  └─ registry_login (existing — handles ECR auth)
  └─ build_images ⬅️ uses the Phase-4-generated Dockerfile
  └─ push_images (existing — pushes to ECR)
  └─ human_approval_gate
  └─ execute_deployment ⬅️ uses the Phase-4-generated .deploy.yaml
                            to know it should target ECS
```

## Required GitHub repo secrets (for the generated workflow to run on GitHub)

The generated `ci-cd.yml` expects these secrets to be set in the repo:

| Secret | Example |
|---|---|
| `AWS_ACCESS_KEY_ID` | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | `...` |
| `AWS_REGION` | `us-east-1` |
| `AWS_ACCOUNT_ID` | `123456789012` |
| `ECR_REPOSITORY` | `leave-mgmt-backend` |
| `ECS_CLUSTER` | `leave-mgmt-cluster` |
| `ECS_SERVICE` | `leave-mgmt-backend` |

These are independent of Phase 7's own credentials (which it gets from
`.env` / `config/deployment.yaml`). Phase 7 deploys directly from your
SDLC-V2 runner; the GitHub Actions workflow deploys from GitHub's runners
on every push to main. Both work; you can use either or both.
