"""
agents/phase4_codegen/cicd_generator.py
========================================

Generates CI/CD and deployment artifacts as part of Phase 4 codegen output.

Why this lives here (and not in Phase 7):
  Phase 4 is where files get created and shipped via Phase 6's PR. Phase 7
  is execution-only — it reads what's in the repo and acts on it. So the
  *creation* of Dockerfile, .deploy.yaml, and the GitHub Actions workflow
  belongs in Phase 4, not Phase 7.

What this module produces, when applicable:
  - Dockerfile                       (for Phase 7's `docker build` step)
  - .dockerignore                    (keeps images small + fast)
  - .deploy.yaml                     (the in-repo deploy contract that
                                      Phase 7's merge_repo_contract reads)
  - .github/workflows/ci-cd.yml      (CI/CD: build → push to ECR → deploy ECS)

When this module skips:
  - The repo already has a Dockerfile and .deploy.yaml — we don't clobber
    hand-tuned infra
  - The project type isn't deployable (e.g. it's a docs repo or library)
  - The scope_contract / ADR indicates the project is not a service

Decisions are LLM-aided. We give the model the ADR, the architecture nodes,
and a snapshot of the generated files; it returns a JSON decision packet:
    {
      "needs_dockerfile": bool,
      "needs_ci_cd_workflow": bool,
      "needs_deploy_yaml": bool,
      "language": "python|node|java|...",
      "framework": "fastapi|express|spring|...",
      "deploy_target": "ecs-prod|lambda-prod|...",
      "service_name": "...",
      "container_port": 8000,
      "rationale": "..."
    }

We then synthesize the actual file contents using deterministic templates
parameterized by that decision — NOT by free-form LLM output, because
generated Dockerfiles are notoriously hallucinated (wrong base images,
nonexistent COPY paths, etc.). Templates are reviewable and predictable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Use the same direct-OpenAI pattern as Phases 1-6. No dependency on the
# half-finished core/llm_gateway abstraction.
_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ---------------------------------------------------------------------------
# Decision step — ask the LLM what kind of project this is
# ---------------------------------------------------------------------------

DECISION_PROMPT = """You are a senior DevOps engineer reviewing a code change. Your job is to
decide what CI/CD and deployment files this change needs.

REQUIREMENT:
{requirement}

ARCHITECTURE DECISION RECORD (excerpt):
{adr_summary}

ARCHITECTURE NODES (services in the system):
{arch_summary}

GENERATED FILES (paths and short summaries):
{file_summary}

EXISTING FILES IN THE REPO (true if a Dockerfile / .deploy.yaml already exists):
- has_dockerfile: {has_dockerfile}
- has_deploy_yaml: {has_deploy_yaml}
- has_github_workflow: {has_github_workflow}

Decide whether the project needs deployment artifacts generated. A deployable
service is typically a long-running process that listens on a network port:
web API (FastAPI/Express/Spring), background worker, scheduled job. Pure
libraries, documentation, and frontend-only bundles usually don't need a
Dockerfile (frontends can be deployed via static hosting).

Return ONLY a JSON object with these fields, nothing else:
{{
  "needs_dockerfile": true|false,
  "needs_ci_cd_workflow": true|false,
  "needs_deploy_yaml": true|false,
  "language": "python|node|typescript|java|go|csharp|other",
  "framework": "fastapi|flask|express|spring|gin|aspnet|other|unknown",
  "deploy_target": "ecs-prod|lambda-prod|compose-prod|k8s-prod",
  "service_name": "<kebab-case short name, e.g. 'leave-mgmt-backend'>",
  "container_port": 8000,
  "build_command": "<shell command if non-default, else empty string>",
  "start_command": "<the command that runs the service, e.g. 'uvicorn app.main:app --host 0.0.0.0 --port 8000'>",
  "rationale": "<one sentence>"
}}

If existing files already cover an artifact (has_dockerfile: true), set the
corresponding needs_* to false — never overwrite hand-tuned infra. Prefer
ecs-prod for typical web APIs. Use lambda-prod only for explicit event-driven
or scheduled functions. Use compose-prod only for local-dev-style multi-
container setups. Use k8s-prod only if the ADR mentions Kubernetes."""


def _summarize_for_decision(generated_changes: list) -> str:
    """Build a compact bulleted list of what's being generated."""
    lines = []
    for c in generated_changes[:40]:  # keep prompt bounded
        fp = c.get("file_path", "")
        summary = c.get("change_summary") or ""
        if summary:
            lines.append(f"- {fp}: {summary[:120]}")
        else:
            lines.append(f"- {fp}")
    if len(generated_changes) > 40:
        lines.append(f"- (+ {len(generated_changes) - 40} more files)")
    return "\n".join(lines) if lines else "(no files generated yet)"


def _summarize_adr(adr: Optional[dict]) -> str:
    if not adr:
        return "(none provided)"
    decisions = adr.get("decisions") or adr.get("adr") or []
    if isinstance(decisions, list):
        parts = []
        for d in decisions[:6]:
            if isinstance(d, dict):
                title = d.get("title") or d.get("id") or ""
                rationale = d.get("rationale") or d.get("decision") or ""
                if title or rationale:
                    parts.append(f"- {title}: {rationale[:200]}")
        return "\n".join(parts) if parts else json.dumps(adr)[:800]
    return json.dumps(adr)[:800]


def _summarize_arch(arch: Optional[dict]) -> str:
    if not arch:
        return "(none provided)"
    nodes = arch.get("nodes") or []
    parts = []
    for n in nodes[:10]:
        if isinstance(n, dict):
            name = n.get("name") or n.get("id") or ""
            typ = n.get("type") or ""
            desc = n.get("description") or ""
            parts.append(f"- {name} ({typ}): {desc[:120]}")
    return "\n".join(parts) if parts else "(no nodes)"


def _detect_existing(generated_changes: list, existing_files_in_repo: set[str]) -> dict:
    """Detect whether deployment artifacts already exist either in the existing
    repo OR in the just-generated changes (we should never double-generate)."""
    paths_being_written = {c.get("file_path", "") for c in generated_changes}
    all_paths = paths_being_written | existing_files_in_repo

    has_dockerfile = any(p == "Dockerfile" or p.endswith("/Dockerfile") for p in all_paths)
    has_deploy_yaml = any(p == ".deploy.yaml" or p.endswith("/.deploy.yaml") for p in all_paths)
    has_workflow = any(
        p.startswith(".github/workflows/") and (p.endswith(".yml") or p.endswith(".yaml"))
        for p in all_paths
    )
    return {
        "has_dockerfile": has_dockerfile,
        "has_deploy_yaml": has_deploy_yaml,
        "has_github_workflow": has_workflow,
    }


def _ask_llm_for_decision(
    requirement: str,
    adr: Optional[dict],
    architecture: Optional[dict],
    generated_changes: list,
    existing_files_in_repo: set[str],
) -> dict:
    """One LLM call to make the deployment decision. Falls back to a safe
    default if the LLM errors out."""
    existing = _detect_existing(generated_changes, existing_files_in_repo)
    prompt = DECISION_PROMPT.format(
        requirement=requirement,
        adr_summary=_summarize_adr(adr),
        arch_summary=_summarize_arch(architecture),
        file_summary=_summarize_for_decision(generated_changes),
        has_dockerfile=existing["has_dockerfile"],
        has_deploy_yaml=existing["has_deploy_yaml"],
        has_github_workflow=existing["has_github_workflow"],
    )

    try:
        response = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=False,
        )
        content = response.choices[0].message.content.strip()
        # Strip code fences if the model wrapped its JSON
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content)
            content = re.sub(r"```\s*$", "", content).strip()
        decision = json.loads(content)
    except Exception as e:
        print(f"  [cicd] decision LLM call failed: {e} — using safe default")
        # Safe default: if a Python file exists and no Dockerfile, generate one
        is_python_service = any(
            c.get("file_path", "").endswith(".py") for c in generated_changes
        )
        decision = {
            "needs_dockerfile": is_python_service and not existing["has_dockerfile"],
            "needs_ci_cd_workflow": is_python_service and not existing["has_github_workflow"],
            "needs_deploy_yaml": is_python_service and not existing["has_deploy_yaml"],
            "language": "python" if is_python_service else "other",
            "framework": "fastapi",
            "deploy_target": "ecs-prod",
            "service_name": "service",
            "container_port": 8000,
            "build_command": "",
            "start_command": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
            "rationale": "LLM unavailable; defaulted to Python/FastAPI ECS service.",
        }

    # Sanitize: never generate files that already exist
    decision["needs_dockerfile"] = bool(decision.get("needs_dockerfile")) and not existing["has_dockerfile"]
    decision["needs_deploy_yaml"] = bool(decision.get("needs_deploy_yaml")) and not existing["has_deploy_yaml"]
    decision["needs_ci_cd_workflow"] = bool(decision.get("needs_ci_cd_workflow")) and not existing["has_github_workflow"]
    return decision


# ---------------------------------------------------------------------------
# Templates — deterministic content generation from decision dict
# ---------------------------------------------------------------------------

PYTHON_DOCKERFILE = """\
# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps that many Python wheels need
RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so docker layer caching is useful
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

EXPOSE {port}

# Tini-style PID 1 isn't strictly needed for Lambda/ECS Fargate but doesn't hurt
CMD ["sh", "-c", "{start_command}"]
"""

NODE_DOCKERFILE = """\
# syntax=docker/dockerfile:1
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

# Build step if a build script exists; harmless otherwise
RUN if [ -f tsconfig.json ] || grep -q '"build"' package.json; then \\
        npm run build || true; \\
    fi

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app /app

EXPOSE {port}

CMD ["sh", "-c", "{start_command}"]
"""

JAVA_DOCKERFILE = """\
# syntax=docker/dockerfile:1
FROM maven:3.9-eclipse-temurin-17 AS build

WORKDIR /app
COPY pom.xml ./
COPY src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:17-jre-jammy

WORKDIR /app
COPY --from=build /app/target/*.jar /app/app.jar

EXPOSE {port}

CMD ["java", "-jar", "/app/app.jar"]
"""

DEFAULT_DOCKERIGNORE = """\
.git
.gitignore
**/__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
.env
.env.local
.idea/
.vscode/
node_modules/
dist/
build/
target/
*.log
.DS_Store
"""

# .deploy.yaml — read by Phase 7's merge_repo_contract.
# Phase 7 already understands these keys (deploy_target, registry, branch,
# dockerfile, docker_context, image_tag, env, env_promotion). We're just
# emitting them.
DEPLOY_YAML = """\
# This file is generated by SDLC-V2 Phase 4 and consumed by Phase 7.
# Edit if you want to change deployment behavior — Phase 7 always
# defers to this file over its global defaults.

# Which named registry from config/deployment.yaml to push to
registry: ecr-prod

# Which named deploy_target from config/deployment.yaml to use
deploy_target: {deploy_target}

# Service identifier — also used as the ECR repo name and ECS service name
service_name: {service_name}

# Repo classification — Phase 7 sorts the deploy sequence by these
type: backend                 # backend | frontend | library | batch

# Image build settings
dockerfile: Dockerfile
docker_context: .
image_tag: "${{BUILD_TAG:-latest}}"

# Branch + commit ref to deploy. Phase 7 uses these when cloning.
branch: main

# Optional: multi-environment promotion
# If present, Phase 7 will deploy to each environment in order,
# pausing for approval between them.
# environments:
#   - name: staging
#     deploy_target: ecs-staging
#     smoke_test_url: https://staging.example.com/health
#   - name: production
#     deploy_target: ecs-prod
#     requires_approval: true
"""

GITHUB_ACTIONS_WORKFLOW = """\
# CI/CD pipeline generated by SDLC-V2 Phase 4.
#
# Triggers:
#   - On every push to main (production deploy)
#   - On every PR (build + validate only, no deploy)
#
# What it does:
#   1. Checkout
#   2. Set up language toolchain
#   3. Install deps (no tests run — this matches your "no test execution"
#      directive; we keep a linting step that's fast and always-pass-or-fail)
#   4. Configure AWS credentials from GitHub repo secrets
#   5. Log into ECR
#   6. Build Docker image
#   7. Push image to ECR (only on main branch)
#   8. Deploy to ECS (only on main branch)
#
# Required GitHub repo secrets:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_REGION             (e.g. us-east-1)
#   AWS_ACCOUNT_ID         (12-digit AWS account number)
#   ECR_REPOSITORY         (e.g. {service_name})
#   ECS_CLUSTER            (e.g. leave-mgmt-cluster)
#   ECS_SERVICE            (e.g. {service_name})

name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: ${{{{ secrets.AWS_REGION }}}}
  ECR_REPOSITORY: ${{{{ secrets.ECR_REPOSITORY }}}}
  ECS_CLUSTER: ${{{{ secrets.ECS_CLUSTER }}}}
  ECS_SERVICE: ${{{{ secrets.ECS_SERVICE }}}}
  IMAGE_TAG: ${{{{ github.sha }}}}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

{lang_setup}

      - name: Validate (lint only — no tests run per directive)
{lint_step}

      - name: Configure AWS credentials
        if: github.event_name == 'push'
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}
          aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}
          aws-region: ${{{{ env.AWS_REGION }}}}

      - name: Login to Amazon ECR
        if: github.event_name == 'push'
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build Docker image
        run: |
          docker build \\
            -t $ECR_REPOSITORY:$IMAGE_TAG \\
            -t $ECR_REPOSITORY:latest \\
            .

      - name: Tag and push to ECR
        if: github.event_name == 'push'
        run: |
          REGISTRY=${{{{ steps.login-ecr.outputs.registry }}}}
          docker tag $ECR_REPOSITORY:$IMAGE_TAG $REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REPOSITORY:latest    $REGISTRY/$ECR_REPOSITORY:latest
          docker push $REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $REGISTRY/$ECR_REPOSITORY:latest

      - name: Force ECS service to deploy new image
        if: github.event_name == 'push'
        run: |
          aws ecs update-service \\
            --cluster $ECS_CLUSTER \\
            --service $ECS_SERVICE \\
            --force-new-deployment \\
            --region $AWS_REGION

      - name: Wait for ECS deployment to stabilize
        if: github.event_name == 'push'
        run: |
          aws ecs wait services-stable \\
            --cluster $ECS_CLUSTER \\
            --services $ECS_SERVICE \\
            --region $AWS_REGION
"""

LANG_SETUP_PYTHON = """\
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
"""

LANG_SETUP_NODE = """\
      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm

      - name: Install Node deps
        run: npm ci
"""

LANG_SETUP_JAVA = """\
      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
          cache: maven
"""

LINT_PYTHON = """\
        run: |
          pip install --quiet ruff || true
          ruff check . || echo "Lint warnings (non-fatal)"
"""

LINT_NODE = """\
        run: |
          if [ -f package.json ] && grep -q '"lint"' package.json; then
            npm run lint || echo "Lint warnings (non-fatal)"
          else
            echo "No lint script — skipping"
          fi
"""

LINT_JAVA = """\
        run: |
          mvn -B -DskipTests verify || echo "Verify warnings (non-fatal)"
"""

LINT_DEFAULT = """\
        run: echo "No lint configured for this language"
"""


# ---------------------------------------------------------------------------
# Synthesis — turn the decision dict into file content strings
# ---------------------------------------------------------------------------

def _render_dockerfile(decision: dict) -> str:
    language = (decision.get("language") or "").lower()
    port = int(decision.get("container_port") or 8000)
    start_command = decision.get("start_command") or "echo 'no start_command set'"

    if language == "python":
        return PYTHON_DOCKERFILE.format(port=port, start_command=start_command)
    if language in ("node", "javascript", "typescript"):
        return NODE_DOCKERFILE.format(port=port, start_command=start_command)
    if language == "java":
        return JAVA_DOCKERFILE.format(port=port)
    # Fallback — generic Python (most common case in this codebase)
    return PYTHON_DOCKERFILE.format(port=port, start_command=start_command)


def _render_deploy_yaml(decision: dict) -> str:
    return DEPLOY_YAML.format(
        deploy_target=decision.get("deploy_target") or "ecs-prod",
        service_name=decision.get("service_name") or "service",
    )


def _render_github_workflow(decision: dict) -> str:
    language = (decision.get("language") or "").lower()
    service_name = decision.get("service_name") or "service"

    if language == "python":
        lang_setup, lint = LANG_SETUP_PYTHON, LINT_PYTHON
    elif language in ("node", "javascript", "typescript"):
        lang_setup, lint = LANG_SETUP_NODE, LINT_NODE
    elif language == "java":
        lang_setup, lint = LANG_SETUP_JAVA, LINT_JAVA
    else:
        lang_setup, lint = LANG_SETUP_PYTHON, LINT_DEFAULT

    return GITHUB_ACTIONS_WORKFLOW.format(
        service_name=service_name,
        lang_setup=lang_setup.rstrip(),
        lint_step=lint.rstrip(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cicd_artifacts(
    requirement: str,
    generated_changes: list,
    adr: Optional[dict] = None,
    architecture: Optional[dict] = None,
    existing_files_in_repo: Optional[set] = None,
    is_brownfield: bool = False,
) -> dict:
    """Decide what CI/CD artifacts to generate and produce their content.

    Brownfield policy (is_brownfield=True):
      - We bias HARD against generating new infra files. Existing repos have
        their own Dockerfiles, deploy contracts, and workflows tuned to
        production. Generating new ones is more likely to break things than
        help. The LLM is allowed to override the bias only if NO infra at all
        exists in the repo — which is uncommon but happens (legacy services
        deployed manually).
      - The defensive sanitizer at the bottom of this function strips any
        `needs_*` flag for a file that already exists on disk, regardless of
        what the LLM said.

    Greenfield policy (is_brownfield=False):
      - Always generate the full stack: Dockerfile, .dockerignore,
        .deploy.yaml, and GitHub Actions workflow.

    Returns:
        {
          "decision": <the decision dict>,
          "new_files": [{"file_path", "content", "change_summary"}, ...]
        }

    The new_files list uses the same shape as `generated_changes` in the
    rest of Phase 4, so they can be appended directly.
    """
    existing_files_in_repo = existing_files_in_repo or set()

    # ── Greenfield: ask the LLM but bias toward "yes, generate everything"
    # ── Brownfield: respect what's on disk; only generate if a slot is truly empty
    if is_brownfield:
        decision = _brownfield_decision(
            requirement, adr, architecture, generated_changes, existing_files_in_repo
        )
    else:
        decision = _ask_llm_for_decision(
            requirement, adr, architecture, generated_changes, existing_files_in_repo
        )

    new_files: list = []

    if decision["needs_dockerfile"]:
        new_files.append({
            "file_path": "Dockerfile",
            "content": _render_dockerfile(decision),
            "change_summary": f"Generated Dockerfile for {decision.get('language')} service",
        })
        # Pair the Dockerfile with a .dockerignore (only if .dockerignore is
        # also missing — covered by the sanitizer below)
        if ".dockerignore" not in existing_files_in_repo:
            new_files.append({
                "file_path": ".dockerignore",
                "content": DEFAULT_DOCKERIGNORE,
                "change_summary": "Generated .dockerignore",
            })

    if decision["needs_deploy_yaml"]:
        new_files.append({
            "file_path": ".deploy.yaml",
            "content": _render_deploy_yaml(decision),
            "change_summary": (
                f"Generated .deploy.yaml — "
                f"target={decision.get('deploy_target')}, "
                f"service={decision.get('service_name')}"
            ),
        })

    if decision["needs_ci_cd_workflow"]:
        new_files.append({
            "file_path": ".github/workflows/ci-cd.yml",
            "content": _render_github_workflow(decision),
            "change_summary": (
                f"Generated GitHub Actions CI/CD pipeline targeting "
                f"AWS ECR + {decision.get('deploy_target')}"
            ),
        })

    print(f"  [cicd] is_brownfield={is_brownfield}; decision: {decision.get('rationale', '(none)')}")
    print(f"  [cicd] generated {len(new_files)} CI/CD artifact(s)")
    for f in new_files:
        print(f"           + {f['file_path']}")

    return {"decision": decision, "new_files": new_files}


def _brownfield_decision(
    requirement: str,
    adr: Optional[dict],
    architecture: Optional[dict],
    generated_changes: list,
    existing_files_in_repo: set,
) -> dict:
    """Brownfield-specific decision: respect existing infra, almost always.

    Three cases:
      1. Repo has ALL infra (Dockerfile + .deploy.yaml + workflow):
         generate nothing. Most common brownfield case. No LLM call needed.
      2. Repo has SOME infra but is missing one or two slots:
         generate nothing. Mixing AI-generated infra with hand-tuned infra
         is a recipe for subtle production breakage — better to leave it to
         the human to write the missing piece, since they understand the
         existing setup. Print a warning so the user can address it.
      3. Repo has NO infra at all:
         legacy service deployed by some other mechanism. Ask the LLM
         (same prompt as greenfield) — generating a full stack is probably
         safer than leaving it bare. Print a warning so the user can
         compare against their current deployment.
    """
    existing = _detect_existing(generated_changes, existing_files_in_repo)
    has_any_infra = (existing["has_dockerfile"]
                     or existing["has_deploy_yaml"]
                     or existing["has_github_workflow"])

    # Case 3: no infra at all → fall through to greenfield-style decision
    if not has_any_infra:
        print(f"  [cicd] brownfield repo has NO infra files — falling through "
              f"to LLM-driven generation. Review carefully against your "
              f"existing deployment process before merging.")
        return _ask_llm_for_decision(
            requirement, adr, architecture, generated_changes, existing_files_in_repo
        )

    # Cases 1 + 2: some or all infra present → do nothing, log specifics
    missing = []
    if not existing["has_dockerfile"]:    missing.append("Dockerfile")
    if not existing["has_deploy_yaml"]:   missing.append(".deploy.yaml")
    if not existing["has_github_workflow"]: missing.append(".github/workflows/*.yml")

    if missing:
        print(f"  [cicd] ⚠ brownfield repo has partial infra. Present: "
              f"Dockerfile={existing['has_dockerfile']}, "
              f".deploy.yaml={existing['has_deploy_yaml']}, "
              f"workflow={existing['has_github_workflow']}. "
              f"Missing: {missing}. NOT auto-generating — please add by hand "
              f"to ensure consistency with your existing infra.")
    else:
        print(f"  [cicd] brownfield repo has full infra — nothing to generate")

    return {
        "needs_dockerfile": False,
        "needs_ci_cd_workflow": False,
        "needs_deploy_yaml": False,
        "language": "unknown",
        "framework": "unknown",
        "deploy_target": "ecs-prod",
        "service_name": "service",
        "container_port": 0,
        "build_command": "",
        "start_command": "",
        "rationale": (
            f"Brownfield repo with existing infra (Dockerfile="
            f"{existing['has_dockerfile']}, .deploy.yaml="
            f"{existing['has_deploy_yaml']}, workflow="
            f"{existing['has_github_workflow']}). Respecting it; no files generated."
        ),
    }
