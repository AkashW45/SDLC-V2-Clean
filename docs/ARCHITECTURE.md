# Architecture

This doc explains how SDLC-V2 works internally. Audience: tech leads, reviewers, anyone debugging beyond the basics.

## High-Level Design
┌─────────────────────────────────────────────────────────────────────┐
│                            DASHBOARD                                │
│             (Vanilla JS, polls /pipeline/status every 3s)           │
└────────────────────────────┬────────────────────────────────────────┘
│ HTTP/REST
▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI (api/main.py)                         │
│   /pipeline/start  /pipeline/preview-routing  /pipeline/approve     │
│   /knowledge/projects/sync  /webhooks/github  /pipeline/list        │
└────┬─────────────────────┬──────────────────────────────────────────┘
│                     │
▼                     ▼
┌──────────────┐    ┌──────────────────────────────────────────────┐
│  Indexer     │    │            LangGraph Orchestration            │
│  Queue       │    │   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
│  (ThreadPool)│    │   │ P0 │→│ P1 │→│ P2 │→│ P3 │→│ P4 │→│ P5 │ │
└──────┬───────┘    │   └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ │
│            │             ↓     ↓     ↓     ↓     ↓        │
│            │           [Approval gates after 1, 2, 3, 6, 7]│
│            └──────────────────────┬──────────────────────-─┘
│                                   │
▼                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   Qdrant     │  │   Neo4j      │  │      PostgreSQL          │ │
│  │ (vectors)    │  │  (graph)     │  │  symbols, contracts,     │ │
│  │              │  │              │  │  projects, pipelines,    │ │
│  │ 4 collections│  │ File/Symbol  │  │  audit_log, asp,         │ │
│  │              │  │ /Repo nodes  │  │  artifacts, pr_registry  │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
▲                                   ▲
│                                   │
└─── GitHub repos (cloned to repos/) ─── auto-indexed on push
via webhook + hourly reconciliation
## Phase Boundaries

### Phase 0 — Routing
Picks where the requirement goes. Three modes:
- **Existing project** — User picked from dashboard modal
- **Manual repos** — User selected specific repos
- **New project** — Slug created from requirement, fresh repos provisioned

Auto-match fallback uses Qdrant `project_embeddings` with threshold `PHASE0_MATCH_THRESHOLD` (default 0.55).

### Phase 0.5 — Adaptive Scope Profile (ASP)
Hidden between Phase 0 and Phase 1. The classifier emits:
- `depth_level` — 1 (toy) to 5 (enterprise)
- `policy_mode` — `lite` | `managed` | `strict`
- `build_mode` — `greenfield` | `modify_existing`
- `repo_summary.matched_repo` — for brownfield, the target repo
- Unit budgets, allowed/forbidden elements

ASP is injected as `scope_contract` into every downstream phase, so artifacts auto-scale.

### Phase 1 — Discovery
LLM generates four artifacts in parallel:
- BRD (business requirements, stakeholders, RACI, KPIs, risks)
- PRD (functional + non-functional + technical reqs, acceptance criteria)
- ADR (architecture decision records)
- Architecture (nodes, edges, Mermaid diagram, security considerations)

Each uses ASP-aware system prompts in `agents/prompts/system_prompts.py`. The "body unwrap" pattern flattens nested LLM output into top-level fields.

### Phase 2 — Planning
Generates:
- Sprint plan (Epics → Stories → Tasks/Subtasks, proportional to ASP scope)
- Real Jira tickets (creates them in your Jira via API)
- Deployment runbook (steps, rollback, feature flags)

Powered by `SPRINT_PLANNER_SYSTEM` prompt + body unwrap.

### Phase 3 — Impact Analysis (Brownfield only)
Three parallel queries scoped to `selected_repos`:
- **Qdrant** — semantic search → top-K affected files (filtered by repo)
- **Neo4j** — dependency graph → file dependents
- **Postgres** — protocol contracts (OpenAPI, gRPC) → contract impact

Then LLM risk assessment combines all three.

**Key fix in current version:** `semantic_search` now accepts `repo_names` filter so impact analysis never bleeds into unselected projects.

### Phase 4 — Code Generation
Two modes:
- **Greenfield** — Fresh scaffold from ASP + PRD
- **Brownfield** — Auto-clones target repo via `RepoWorkspaceManager`, reads affected files (local-first, GitHub API fallback), builds RAG context packet from Qdrant + Postgres + full source files, sends to LLM, gets `search_block` / `replace_block` diffs, applies with retry.

### Phase 5 — Validation
Concurrent pipeline:
- AST parse (Python only; HTML/MD/JSON skipped)
- Semgrep security scan
- pytest sandbox (creates temp dir, runs subprocess with timeout)
- Critic LLM does final review

Early-exits cleanly for non-Python projects.

### Phase 6 — Delivery
- Pushes generated files to a feature branch (`feature/pipeline-<thread_id>`)
- Opens PR via GitHub App auth
- Idempotent via SHA-256 request_id (re-runs don't create duplicates)
- For greenfield (new repo, no master), pushes directly to main

### Phase 7 — Deployment Plan
Generates the deployment + monitoring + rollback plan. **Currently simulated** — outputs the plan but does not execute `docker build / kubectl apply`. See [DEPLOYMENT.md](DEPLOYMENT.md) for how to make this real.

## State Persistence

Every pipeline's full state is stored as JSONB in PostgreSQL's `pipelines` table. On uvicorn restart, all in-progress pipelines auto-resume from where they paused (typically at an approval gate).

Resume API: `POST /pipeline/<thread_id>/resume` with `{"phase": N}` — re-runs Phase N using saved state, no LLM re-cost.

## Indexing Strategy — Change-Driven, Not Run-Driven

**Critical principle:** Pipeline runs NEVER trigger indexing. They only read from the Knowledge Layer.

Indexing happens on:
1. **GitHub webhook** — push to default branch, repository created/deleted
2. **Scheduled reconciliation** — APScheduler runs hourly, compares GitHub SHA → indexed SHA, enqueues drifted repos
3. **Manual sync** — `/knowledge/projects/sync` admin endpoint

SHA-skip logic in `index_repo(repo_path, repo_name, force=False)` ensures unchanged repos are not re-embedded.

**Cost model:** Pipeline cost scales with developer activity (commits), not pipeline runs. 1000 pipelines/day on a stable codebase = zero embedding cost.

## Knowledge Layer Data Model

### Qdrant collections

| Collection | Used by | Stores |
|---|---|---|
| `project_embeddings` | Phase 0 | One vector per registered project |
| `code_embeddings` | Phase 3, Phase 4 | One vector per code symbol |
| `contract_embeddings` | Phase 3 | One vector per API contract |
| `repo_map_embeddings` | Phase 0.5 | One vector per repo summary |

### Postgres tables

See `knowledge_layer/db_setup.py` — 12 tables. Key ones:
- `pipelines` — Full state JSONB
- `audit_log` — Every phase event
- `asp` — Adaptive Scope Profiles per pipeline
- `pr_registry` — Idempotent PR tracking
- `indexer_jobs` — Indexer queue state (survives restart)
- `repo_overrides` — Manual include/exclude decisions per repo

### Neo4j graph
(Repo)-[:CONTAINS]->(File)-[:DEFINES]->(Symbol)
(File)-[:IMPORTS]->(File)
## Adaptive Scope Profile (ASP) — How It Auto-Scales

The classifier emits depth 1-5:
- **Depth 1** — Toy/hello-world. ~1 ticket, 1 file, smoke test only.
- **Depth 2** — Small utility. ~3 tickets, 2-3 files.
- **Depth 3** — Real feature. ~5-10 tickets, 5-10 files, full test matrix.
- **Depth 4** — Multi-service. ~15-25 tickets, 20-30 files.
- **Depth 5** — Enterprise platform. 30+ tickets, 50+ files, deep integration.

All prompts in `system_prompts.py` consume the depth via `_SCOPE_BINDING` and emit proportionally. "No fixed count, no cap" is enforced everywhere.

## GitHub App Authentication

`agents/github_auth.py` handles the full flow:
1. JWT signed with the App's private key (RS256, valid 10 min) → used to authenticate as the App
2. Exchange JWT for installation token (`POST /app/installations/{id}/access_tokens`) → valid 1 hour
3. Token cached for 55 min, auto-refreshed before expiry
4. Used as `Authorization: Bearer <token>` for all GitHub API calls
5. Used as `https://x-access-token:<token>@github.com/...` for git clones

Falls back to PAT if App env vars are missing (legacy mode).

## Security

- API key required on every write endpoint (`X-API-Key` header)
- GitHub App secret rotates installation tokens hourly
- Webhook payload verified with HMAC-SHA256 against `GITHUB_WEBHOOK_SECRET`
- LLM never sees `.env`, `.pem`, or secrets in repo
- Generated code is validated (AST + Semgrep) before push

## Reading the code

| Question | Start here |
|---|---|
| "How does Phase X work?" | `agents/phaseN_*/...` |
| "How does the LLM get called?" | `core/llm_gateway.py` |
| "How does Qdrant search work?" | `agents/phase3_impact/impact_analyzer.py:semantic_search` |
| "How is brownfield context built?" | `agents/context_packet_builder.py` |
| "How does indexing happen?" | `knowledge_layer/indexer.py` |
| "How does the dashboard work?" | `dashboard/index.html` (single file) |
| "How are GitHub PRs created?" | `agents/pr_manager.py` + `agents/github_auth.py` |