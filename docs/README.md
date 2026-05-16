<!-- # SDLC-V2 — AI-Powered Software Development Lifecycle Automation Platform

> **Plain English in → Working code, PRs, Jira tickets, runbooks, and deployment plans out.**
> Eight LangGraph agent phases. Five human approval gates. Full audit trail. Survives server restarts.

---

## What This Is

SDLC-V2 takes a single business requirement written in plain English and automates the entire software development lifecycle — from writing the BRD through to pushing code to GitHub and generating a production deployment plan. It understands your existing codebase through a vector + graph knowledge layer, respects your architectural decisions, and never merges anything without a human reviewing it first.

It is not a chatbot. It is a multi-agent pipeline orchestrated by LangGraph, backed by DeepSeek V4 Pro (1M context, thinking mode), and connected to your real Jira, GitHub, PostgreSQL, Qdrant, and Neo4j instances.

---

## How It Works — End to End

```
User submits requirement (plain English)
        │
        ▼
Phase 0  — Project Router
        Embeds requirement → searches Qdrant project_embeddings
        Score ≥ 0.7 → matches existing project + repos
        Score < 0.7 → creates new project slug + fresh GitHub repos
        │
        ▼
Phase 0.5 — Intent Classifier (inside Phase 1)
        Reads requirement + repo_summary
        Produces Adaptive Scope Profile (ASP):
          depth level 1–5, policy mode, forbidden elements,
          mandatory elements, budget estimate in work-units
        This ASP binds every downstream agent
        │
        ▼
Phase 1  — Discovery  ──────────────────── ⏸ HUMAN GATE 1
        BRD  → PRD  → ADR  → Architecture (Mermaid diagram)
        Each artifact validated by CriticAgent
          (ACCEPT / COMPRESS / EXPAND / REGENERATE)
        Reject → SurgicalReplay regenerates only the rejected artifact
        Approve → Phase 2
        │
        ▼
Phase 2  — Planning   ──────────────────── ⏸ HUMAN GATE 2
        Sprint plan (epics + stories)
        Real Jira tickets created via REST API
        Deployment runbook (steps, rollback, feature flags)
        Approve → Phase 3 (existing project) or Phase 4 (new project)
        │
        ▼
Phase 3  — Impact Analysis ─────────────── ⏸ HUMAN GATE 3
        (skipped for brand-new projects)
        Qdrant semantic search → top 3 affected files
        Neo4j dependency traversal (batched, concurrent)
        Protocol contract check (OpenAPI / gRPC / AsyncAPI)
        LLM risk assessment → low / medium / high
        Approve → Phase 4
        │
        ▼
Phase 4  — Code Generation  (automatic, no gate)
        Brownfield: RAG context packet from Qdrant + GitHub API
                    diff-based patches (search_block → replace_block)
        Greenfield: full scaffold per architecture nodes + ADR tech stack
        Polyglot: Python, Java, TypeScript, Go, C#, etc.
        AST validation + up to 3 auto-retries per file
        ExpansionDecisionEngine checks work-unit budget
        │
        ▼
Phase 5  — Validation  (automatic, no gate)
        Concurrent test generation per changed file (ThreadPoolExecutor)
        pytest files with Given/When/Then acceptance criteria
        Syntax validation + basic lint
        Semgrep security scan → blocks on critical findings
        │
        ▼
Phase 6  — Delivery   ──────────────────── ⏸ HUMAN GATE 4
        Pushes code to GitHub branch (creates repo via API if new)
        Opens pull request with full change summary
        PRManager is idempotent: SHA-256(asp_id + artifact_id + version)
          prevents duplicate PRs on retry
        Human reviews PR in GitHub → Approve or Reject
        │
        ▼
Phase 7  — Deployment ──────────────────── ⏸ HUMAN GATE 5
        Resolves deploy order: shared libs → backend → frontend → batch
        Configures feature flags (all disabled by default)
        Human approves production deploy
        Simulates deployment per repo in sequence
        Enables feature flags post-deploy
        Monitors error rate + latency
        Auto-rollback if thresholds exceeded
```

---

## The 8 Phases at a Glance

| # | Phase | Human Gate | Auto | Status |
|---|-------|-----------|------|--------|
| 0 | Project & Repo Router | — | ✅ | Done |
| 0.5 | Intent Classifier → ASP | — | ✅ | Done |
| 1 | Discovery: BRD / PRD / ADR / Architecture | ✅ Gate 1 | — | Done |
| 2 | Planning: Sprint / Jira / Runbook | ✅ Gate 2 | — | Done |
| 3 | Impact Analysis | ✅ Gate 3 | — | Done (skipped for new projects) |
| 4 | Code Generation | — | ✅ | Done |
| 5 | Validation + Test Generation | — | ✅ | Done |
| 6 | Delivery: GitHub Push + PR | ✅ Gate 4 | — | Done |
| 7 | Deployment + Monitoring + Rollback | ✅ Gate 5 | — | Done |

---

## Architecture Deep Dive

### Two Layers

**Layer 1 — Knowledge Layer (offline, runs after every repo merge)**

Indexes your codebase into three stores:

- **PostgreSQL** — AST symbols (class/function/method names, signatures, docstrings), protocol contracts (OpenAPI/gRPC/AsyncAPI), repo maps, project registry, pipeline state, audit log
- **Qdrant** — vector embeddings of code symbols (`code_embeddings`), project descriptions (`project_embeddings`), API contracts (`contract_embeddings`), repo summaries (`repo_map_embeddings`)
- **Neo4j** — file dependency graph (`File -[IMPORTS]→ Module`, `File -[DEFINES]→ Symbol`)

Supported languages for indexing: Python (AST module), JavaScript/TypeScript/Java/C# (Tree-sitter).

**Layer 2 — Execution Layer (runtime, per requirement)**

Eight LangGraph agent phases. Each phase queries the Knowledge Layer for the minimum context needed — the LLM never sees the full codebase, only a precision-targeted context packet.

### LangGraph vs n8n Boundary

LangGraph owns everything AI-related: agent graphs, state management, all INTERRUPT gates, PostgreSQL persistence after every node, audit logging, reject-and-regenerate loops.

n8n is planned for external integrations only: Slack notifications, TeamCity build triggers, Confluence updates, nightly indexer scheduling.

**Rule: never put AI reasoning in n8n. Never put external API calls inside LangGraph nodes (except the GitHub and Jira clients we wrap directly).**

### Adaptive Scope Profile (ASP)

The ASP is generated once in Phase 0.5 by the Intent Classifier and is immutable for the rest of the pipeline. It contains:

- `depth_level` (1–5) — drives how many FRs, nodes, tickets, files are appropriate
- `policy_mode` — open / managed / conservative — controls auto-approval vs human gates
- `forbidden_elements` — technologies that must never appear (e.g. `kubernetes` at depth 1)
- `mandatory_elements` — things every phase must include (e.g. `HIPAA_audit_trail` at depth 5)
- `budget_estimates.work_units` — total unit budget for the entire pipeline
- `build_mode` — `greenfield` or `modify_existing` based on repo_summary overlap score

Depth scoring is mechanical, not subjective: "enterprise" and "platform" add nothing. Explicit compliance terms (PCI/HIPAA/SOC2) force depth 5. "For our company" is not scale.

### ExpansionDecisionEngine

Every generated artifact passes through the engine before being accepted. It uses depth-aware unit weights stored in PostgreSQL (`units_weights` table). Example weights at depth 3:

| Unit Type | Weight |
|-----------|--------|
| FR | 1.2 |
| CODE_FILE | 1.0 |
| ENDPOINT | 4.0 |
| DB_TABLE | 10.0 |
| INTEGRATION | 20.0 |

Verdict options: `accept` / `queue_for_approval` / `reject`. Hard rejections fire on forbidden elements, critic REGENERATE verdict, critical violations, or unresolved evidence on load-bearing artifacts.

### CriticAgent

Validates every BRD, PRD, ADR, Architecture, Sprint Plan, and Code artifact against the ASP before it leaves Phase 1 or 2. Checks:

1. Traceability — every element must trace to a real anchor phrase in the ASP
2. Forbidden elements — immediate REGENERATE if found
3. Mandatory elements — EXPAND if missing
4. Groundedness — no hallucinated features or integrations
5. Proportionality — untagged bloat → COMPRESS; genuine thinness at high depth → EXPAND
6. Drift — PRD FRs must match BRD FRs exactly; Architecture nodes must trace to PRD FR-IDs

### SurgicalReplay

When a human rejects Phase 1, instead of regenerating everything, the API computes which specific artifact (BRD / PRD / ADR / Architecture) the feedback targets, creates a `replay_job` record, and regenerates only that artifact with the feedback baked into the prompt. The pipeline returns to `WAITING_PHASE_1_APPROVAL` without touching the others.

### ContextPacketBuilder (RAG for Codegen)

For brownfield modifications, before the codegen LLM call:
1. Embeds the requirement
2. Searches `code_embeddings` in Qdrant for top-K relevant code chunks in the target repo
3. Enriches chunks with Postgres symbols metadata (file_path, line numbers, kind)
4. Fetches full file content from `indexed_files` table or GitHub API for top-N files
5. Formats a deterministic "RELEVANT EXISTING CODE" block prepended to the codegen prompt

This ensures codegen sees the actual function bodies it needs to modify, not just symbol names.

### EvidenceResolver

Every artifact carries `evidence_links` — typed references that prove its elements are grounded:

- `requirement` — exact phrase from user input (fuzzy word-overlap check)
- `doc` — Qdrant point ID (cosine similarity ≥ 0.7)
- `commit` — repo + SHA + path + lines (GitHub API verification)
- `symbol` — symbol name in Postgres symbols table

An artifact's `evidence_resolved` is true when ≥50% of links resolve and average score ≥ 0.7. Load-bearing artifacts (CODE, PATCH) are hard-rejected if evidence is unresolved.

### PRManager (Idempotent)

`unique_request_id = SHA-256(asp_id + artifact_id + agent_version)[:32]`

Before creating any PR, the manager checks `pr_registry` for this ID. If found, returns the existing PR. This means retrying a failed pipeline never creates duplicate PRs. New projects push directly to the default branch (no PR needed); existing projects always get a feature branch.

---

## Project Structure

```
SDLC-V2/
│
├── api/
│   ├── main.py                  # FastAPI app — all HTTP endpoints, pipeline orchestration
│   ├── persistence.py           # PostgreSQL helpers: save/load pipeline, audit log
│   ├── jira_client.py           # Jira REST API v3: fetch metadata, create issues
│   ├── runbook_export.py        # Export BRD/PRD/ADR/Architecture/Sprint to Markdown + Excel
│   └── test_cases_export.py     # Generate test cases per Jira story → Excel
│
├── agents/
│   ├── phase0_selector/
│   │   └── selector_agent.py    # Qdrant project search, new project slug generation
│   │
│   ├── phase1_discovery/
│   │   └── discovery_agent.py   # classify_intent → BRD → PRD → ADR → Architecture
│   │                            # CriticAgent validation at each step
│   │
│   ├── phase2_planning/
│   │   └── planning_agent.py    # Sprint plan → Jira tickets (real API) → Runbook
│   │
│   ├── phase3_impact/
│   │   ├── impact_analyzer.py   # Qdrant search + Neo4j batched + PG contracts + LLM risk
│   │   └── graph.py             # LangGraph graph for Phase 3 with INTERRUPT
│   │
│   ├── phase4_codegen/
│   │   └── codegen_agent.py     # Fresh scaffold or diff-based patches, AST validation
│   │
│   ├── phase5_validation/
│   │   └── validation_agent.py  # Concurrent test gen, syntax check, Semgrep gate
│   │
│   ├── phase6_delivery/
│   │   └── delivery_agent.py    # Git push to branch, GitHub PR creation
│   │
│   ├── phase7_deployment/
│   │   └── deployment_agent.py  # Deploy order, feature flags, monitoring, rollback
│   │
│   ├── prompts/
│   │   └── system_prompts.py    # All LLM system prompts (CLASSIFIER, BRD, PRD, ADR,
│   │                            # ARCHITECTURE, SPRINT, CODEGEN, TESTGEN, DEPLOYMENT, CRITIC)
│   │
│   ├── critic/
│   │   └── critic_agent.py      # CriticAgent: validates artifacts against ASP
│   │
│   ├── context_packet_builder.py  # RAG context for brownfield codegen
│   ├── evidence_resolver.py       # Validates evidence_links in artifacts
│   ├── expansion_engine.py        # Depth-aware unit budget gating
│   ├── units_model.py             # Extracts unit_counts from any artifact type
│   ├── pr_manager.py              # Idempotent GitHub PR creation
│   ├── stage2_store.py            # Persistence helpers for ASP, artifacts, decisions, PR registry
│   └── pipeline.py                # CLI orchestrator: runs all phases interactively
│
├── core/
│   ├── llm_gateway.py           # LLMGateway wrapper: telemetry, token logging
│   └── context_engine.py        # ContextOptimizationEngine: YAML projection, AST pruning
│
├── knowledge-layer/
│   ├── indexer.py               # Repo indexer: AST (Python) + Tree-sitter (JS/TS/Java/C#)
│   │                            # → PostgreSQL symbols + Qdrant embeddings + Neo4j graph
│   ├── db_setup.py              # Creates all PG tables and Qdrant collections
│   ├── project_registry.py     # Register/search projects in PG + Qdrant
│   └── repo_summary.py          # Builds repo_summary JSON for the ASP classifier
│
├── db/
│   └── stage2_migration.py      # Creates Stage 2 tables: asp, artifacts, decisions,
│                                # units_weights (depth-aware), pr_registry, audit_log
│
├── dashboard/
│   └── index.html               # Vanilla JS live dashboard — polls every 3s,
│                                # renders Mermaid diagrams, approval gates, download links
│
├── docs/
│   ├── README.md                # This file
│   └── architecture/
│       ├── overview.md          # LangGraph vs n8n boundary, phase descriptions
│       ├── data-flow.md         # Request lifecycle, state persistence, audit trail
│       └── tech-stack.md        # Full tech stack table, env vars, DB schemas
│
├── docker-compose.yml           # Qdrant + Neo4j + PostgreSQL + FastAPI API
├── Dockerfile                   # Python 3.11 slim, installs requirements
├── requirements.txt             # All Python dependencies
├── setup.py                     # Provisions 12-project enterprise portfolio,
│                                # clones + indexes real GitHub repos end to end
└── approve.json / start.json    # Sample CLI payloads for testing
```

---

## Infrastructure

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI + Dashboard | `localhost:8001` | Backend API + live control tower |
| Qdrant | `localhost:6333` | Vector search (4 collections) |
| Neo4j | `localhost:7474` | Dependency graph (bolt: 7687) |
| PostgreSQL | `localhost:5437` (host) / `5432` (Docker internal) | All relational data |
| n8n | `localhost:5678` | External integrations (planned) |

### PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `projects` | Project registry: id, name, description, repos, tech stack |
| `repo_maps` | Indexed repo metadata: language, file count, last indexed |
| `symbols` | AST-extracted symbols: name, type, file, line, signature, docstring |
| `protocol_contracts` | OpenAPI / gRPC / AsyncAPI / Avro contract files |
| `pipelines` | Full pipeline state JSONB — survives server restart |
| `audit_log` | Every event with actor, phase, timestamp (90-day retention) |
| `asp` | Adaptive Scope Profiles per pipeline |
| `artifacts` | Generated artifacts with status, unit counts, confidence |
| `artifact_decisions` | Expansion engine decisions per artifact |
| `units_weights` | Depth-aware unit weights (name, depth_level, weight) |
| `pr_registry` | Idempotent PR tracking by unique_request_id |
| `replay_jobs` | Surgical replay job tracking |

### Qdrant Collections

| Collection | Dimensions | Payload |
|-----------|-----------|---------|
| `project_embeddings` | 384 | project_id, name, description, repos |
| `code_embeddings` | 384 | repo_name, file_path, symbol_name, type |
| `contract_embeddings` | 384 | repo_name, file_path, contract_type |
| `repo_map_embeddings` | 384 | repo_name, summary |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally).

---

## Quick Start

### 1. Prerequisites

- Docker Desktop running
- Python 3.11+
- Git
- DeepSeek API key
- GitHub personal access token (repo scope)
- Jira Cloud account + API token

### 2. Clone and configure

```bash
git clone https://github.com/AkashW45/SDLC-V2.git
cd SDLC-V2
cp .env.example .env
```

Edit `.env`:

```env
DEEPSEEK_API_KEY=sk-...
LLM_API_KEY=sk-...           # same key, used by LLMGateway
LLM_BASE_URL=https://api.deepseek.com

GITHUB_TOKEN=ghp_...
GITHUB_REPO_OWNER=YourGitHubUsername

JIRA_EMAIL=you@company.com
JIRA_API_TOKEN=...
JIRA_BASE_URL=yourcompany.atlassian.net
JIRA_PROJECT_KEY=DEV

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5437
POSTGRES_USER=sdlc
POSTGRES_PASSWORD=sdlc1234
POSTGRES_DB=sdlc_knowledge
POSTGRES_PASSWORD=sdlc1234

NEO4J_PASSWORD=password1234

API_SECRET_KEY=sdlc-dev-key-12345
```

### 3. Start infrastructure

```bash
docker-compose up -d --build
```

Wait ~30 seconds for all services to be healthy.

### 4. Initialize databases

```bash
# Create all PostgreSQL tables and Qdrant collections
python knowledge-layer/db_setup.py

# Create Stage 2 tables (ASP, artifacts, units_weights, pr_registry)
python db/stage2_migration.py
```

### 5. Register your projects

Option A — Register the 12-project demo portfolio (clones real GitHub repos, indexes them):

```bash
python setup.py
```

Option B — Register a single existing project manually:

```bash
python knowledge-layer/project_registry.py
```

Option C — Index one of your own repos:

```bash
python knowledge-layer/indexer.py \
  --repo-path /path/to/your/repo \
  --repo-name your-repo-name
```

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### 7. Open the dashboard

```
http://localhost:8001/dashboard
```

Type a requirement and click **Launch Pipeline**.

---

## API Reference

All write endpoints require the header `X-API-Key: <your API_SECRET_KEY>`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check all service connections |
| GET | `/dashboard` | Serve the live control tower UI |
| POST | `/pipeline/start` | Start a pipeline: `{"requirement": "..."}` |
| GET | `/pipeline/status/{thread_id}` | Poll pipeline state |
| POST | `/pipeline/approve/{thread_id}` | Approve or reject: `{"approved": true, "feedback": "..."}` |
| GET | `/pipeline/list` | List all pipelines |
| DELETE | `/pipeline/{thread_id}` | Remove a pipeline |
| GET | `/pipeline/{thread_id}/audit` | Full audit log for a pipeline |
| GET | `/pipeline/{thread_id}/download/brd` | BRD as Markdown |
| GET | `/pipeline/{thread_id}/download/prd` | PRD as Markdown |
| GET | `/pipeline/{thread_id}/download/adr` | ADR as Markdown |
| GET | `/pipeline/{thread_id}/download/architecture` | Architecture as Markdown + Mermaid |
| GET | `/pipeline/{thread_id}/download/sprint-plan` | Sprint plan as Markdown |
| GET | `/pipeline/{thread_id}/download/impact` | Impact report as Markdown |
| GET | `/pipeline/{thread_id}/download/runbook` | Runbook as Excel |
| GET | `/pipeline/{thread_id}/download/test-cases` | Test cases as Excel |
| GET | `/pipeline/{thread_id}/download/all` | ZIP of all artifacts + code + tests |
| POST | `/knowledge/index` | Index a repo: `{"repo_path": "...", "repo_name": "..."}` |
| POST | `/knowledge/search` | Semantic code search: `{"query": "...", "top_k": 10}` |
| GET | `/knowledge/repos` | List all indexed repos |

### Approval Payload Options

```json
{
  "approved": true,
  "feedback": "Looks good",
  "actor": "engineer@company.com"
}
```

For rejection with surgical replay on Phase 1:

```json
{
  "approved": false,
  "feedback": "The ADR should use PostgreSQL not SQLite",
  "artifact": "ADR"
}
```

Valid `artifact` values: `BRD`, `PRD`, `ADR`, `ARCHITECTURE`. If omitted, the system infers from the feedback text.

---

## Dashboard Guide

The dashboard at `/dashboard` polls `/pipeline/status/{thread_id}` every 3 seconds.

**Pipeline bar** — shows live progress across all 7 phases with color coding:
- Green dot = running
- Amber pause = awaiting human approval
- Tick = done

**Phase cards** each show:
- Sub-stage progress within the phase (e.g. "BRD ✓ PRD ✓ ADR ○ Arch ○")
- Full artifact content rendered inline (RACI tables, FR cards, risk matrix, etc.)
- Mermaid architecture diagram rendered client-side (anti-flicker cached)
- Approval zone with optional feedback textarea

**Download bar** — every artifact has a download button. The "Download Full Package" button top-right produces a ZIP containing all Markdown files, the Excel runbook, test cases XLSX, and all generated code files organized in folders.

---

## How the Knowledge Layer Feeds Codegen

When the requirement targets an existing project, Phase 4 does this before calling the LLM:

```
Embed requirement
    → Qdrant search code_embeddings (top 8 chunks, filtered by repo)
    → Enrich with Postgres symbols (file path, line numbers, kind, signature)
    → GitHub API fetch full content for top 4 files
    → Format as "RELEVANT EXISTING CODE" block
    → Prepend to codegen prompt
```

This means the LLM sees the actual function bodies it needs to patch, not just their names. It emits `search_block` / `replace_block` diffs, not full file rewrites. The pipeline applies each diff, validates Python syntax (AST), and retries up to 3 times if the `search_block` doesn't match.

For a new (greenfield) project, there is no existing code — the LLM instead generates a full scaffold guided by the architecture nodes and the ADR tech stack decisions.

---

## Supported Languages

| Language | Knowledge Layer Indexing | Code Generation |
|----------|--------------------------|-----------------|
| Python | AST module (native) | DeepSeek V4 Pro |
| JavaScript | Tree-sitter | DeepSeek V4 Pro |
| TypeScript | Tree-sitter | DeepSeek V4 Pro |
| Java | Tree-sitter | DeepSeek V4 Pro |
| C# | Tree-sitter | DeepSeek V4 Pro |
| Go | — (planned) | DeepSeek V4 Pro |

Protocol contracts indexed: OpenAPI (YAML/JSON), gRPC (`.proto`), AsyncAPI (YAML), Avro (`.avsc`).

---

## State Persistence

Every node completion in every LangGraph graph writes to the `pipelines` table:

```sql
INSERT INTO pipelines (thread_id, requirement, status, phase, sub_stage, current_state, pr_urls, error, updated_at)
ON CONFLICT (thread_id) DO UPDATE SET ...
```

`current_state` is a JSONB column holding the entire pipeline state — BRD, PRD, ADR, architecture, sprint plan, impact report, generated code, test files, PR URLs, deploy results. On server restart, `load_all_pipelines()` restores all active pipelines into memory from the last 50 records ordered by `updated_at`.

Every state transition also writes to `audit_log`:

```sql
INSERT INTO audit_log (thread_id, phase, event, actor, details, created_at)
```

The audit log is append-only (DELETE revoked from PUBLIC). Retention enforcement runs as a privileged admin cron job (90 days).

---

## Reject and Regenerate

When a human rejects any phase with feedback:

1. Feedback is saved to `human_feedback` in pipeline state
2. For Phase 1: SurgicalReplay targets the specific artifact (BRD/PRD/ADR/Architecture)
3. For other phases: the full phase re-runs with feedback prepended to all prompts:

```
IMPORTANT — USER FEEDBACK on previous attempt (you MUST address all of this):
<feedback>
```

4. New artifacts replace old ones in the pipeline state
5. Pipeline returns to `WAITING_PHASE_X_APPROVAL` for re-review

---

## Known Issues Fixed in Current Version

| File | Issue | Fix Applied |
|------|-------|-------------|
| `codegen_agent.py` | `call_llm` didn't accept `max_tokens`, wrong return type | Signature fixed, uses `gateway.generate` |
| `codegen_agent.py` | `generate_fresh_project` defined twice (nested duplicate) | Inner duplicate removed |
| `codegen_agent.py` | `build_codegen_graph` had two `set_entry_point` calls | Second one removed |
| `codegen_agent.py` | `run_critic_check` node referenced but never defined | Node removed from graph |
| `codegen_agent.py` | `CodegenState` had `existing_code` field twice | Duplicate removed |
| `codegen_agent.py` | `run_codegen` missing `scope_contract` kwarg | Added with default `None` |
| `validation_agent.py` | `run_validation_phase` missing `scope_contract` kwarg | Added with default `None` |
| `validation_agent.py` | `ValidationState` missing `scope_contract` field | Field added |
| `validation_agent.py` | Test gen used Groq model on DeepSeek client | Changed to `deepseek-chat` |
| `discovery_agent.py` | `safe_parse_json` missing `label` parameter | Parameter restored |
| `discovery_agent.py` | Fallback functions defined inside `build_discovery_graph` | Moved to module level |
| `discovery_agent.py` | `classify_intent` node added twice | Duplicate removed |
| `main.py` | `run_phase5` referenced but never defined | Stub function added |
| `main.py` | `audit()` called with 3 args instead of 5 | Missing args added |

---

## Critical Rules — Never Violate

1. **LangGraph owns all AI reasoning.** n8n handles only external integrations (Slack, TeamCity).
2. **All human gates use LangGraph `interrupt()`.** Never n8n Wait nodes.
3. **The LLM never sees the full codebase.** Only the Knowledge Layer context packet.
4. **Never auto-merge PRs.** Phase 6 always requires human review before merge.
5. **Always validate generated Python with `ast.parse()`.** Retry up to 3 times.
6. **Architecture nodes must trace to specific PRD FR-IDs.** No invented technologies.
7. **ASP is frozen after Phase 0.5.** No phase may add to forbidden_elements or change depth_level.
8. **PRManager is always idempotent.** Check `pr_registry` before any GitHub API call.

---

## Running the Pipeline from CLI

For local testing without the dashboard:

```bash
python agents/pipeline.py
```

This runs the full 8-phase pipeline interactively — it pauses at each human gate and waits for you to press Enter.

To test individual phases:

```bash
# Phase 3 only
python agents/phase3_impact/impact_analyzer.py

# Phase 6 only
python agents/phase6_delivery/delivery_agent.py

# Phase 7 only
python agents/phase7_deployment/deployment_agent.py
```

---

## Environment Variables Reference

```env
# LLM
DEEPSEEK_API_KEY=sk-...
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com

# GitHub
GITHUB_TOKEN=ghp_...
GITHUB_REPO_OWNER=YourUsername
AGENT_VERSION=v2.0            # used in PRManager idempotency key

# Jira
JIRA_EMAIL=you@company.com
JIRA_API_TOKEN=...
JIRA_BASE_URL=yourcompany.atlassian.net
JIRA_PROJECT_KEY=DEV

# PostgreSQL
POSTGRES_HOST=127.0.0.1       # set to sdlc_postgres inside Docker
POSTGRES_PORT=5437             # host port (5432 inside Docker)
POSTGRES_USER=sdlc
POSTGRES_PASSWORD=sdlc1234
POSTGRES_DB=sdlc_knowledge

# Qdrant
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password1234

# API Security
API_SECRET_KEY=sdlc-dev-key-12345

# Codegen
REPO_PATH=C:\path\to\local\repo   # local checkout path for brownfield codegen

# Evidence
EVIDENCE_THRESHOLD=0.7             # minimum score for evidence_link to resolve

# Notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

---

## What Changed from V1

| V1 | V2 |
|----|-----|
| n8n orchestrator | LangGraph orchestrator with INTERRUPT gates |
| Single Leave Management repo | Multi-project, multi-repo, auto-create new repos |
| 4 phases | 8 phases (Phase 0 selector + Phase 7 deployment added) |
| Groq / gpt-oss-120b (8K context) | DeepSeek V4 Pro (1M context, thinking mode) |
| In-memory state only | PostgreSQL persistence — survives server restarts |
| No reject-regenerate | SurgicalReplay — regenerate only the failed artifact |
| Static documents | Mermaid architecture diagrams generated deterministically |
| No project intelligence | Qdrant project_embeddings for smart repo routing |
| No budget control | ExpansionDecisionEngine with depth-aware unit weights |
| No artifact validation | CriticAgent validates every artifact before acceptance |
| No evidence checking | EvidenceResolver validates all artifact evidence links |
| No duplicate PR protection | PRManager with SHA-256 idempotency key |
| Monolingual (Python only) | Polyglot codegen and indexing (Python/JS/TS/Java/C#) | -->
# Current Working State — as of 2026-05-15

This file documents what is verified end-to-end. The main `README.md` describes the
full design intent; this file is the ground truth on what runs today.

ndexing policy: change-driven, never run-driven.

Pipeline runs (/pipeline/start) NEVER trigger indexing — they only read from the pre-built Knowledge Layer.
Indexing is triggered by: (a) GitHub webhook on push-to-default-branch, (b) GitHub webhook on repository created/deleted, (c) explicit admin call to /knowledge/projects/sync.
All three paths compare current commit SHA against repo_maps.last_indexed_sha and skip when equal. Force with force_reindex: true in the sync API.
Cost scales with repo changes, not with pipeline runs. A team running 10 pipelines/day against a stable codebase pays zero embedding cost beyond the initial index.

## Verified end-to-end

| Phase | Greenfield (new project) | Brownfield (existing project) |
|------:|--------------------------|-------------------------------|
| 0     | ✅ Routes to new slug, creates repo entries | ✅ Matches by Qdrant score ≥ 0.7 |
| 0.5   | ✅ Generates ASP with `build_mode=greenfield` | ✅ Generates ASP with `build_mode=modify_existing` when repo_summary.symbol_overlap > 0.3; injects `repo_summary` into ASP |
| 1     | ✅ BRD + PRD + ADR + Architecture, body fields flattened to top-level | ✅ Same, plus traces against existing repo |
| 2     | ✅ Sprint + Jira + Runbook | ✅ Same |
| 3     | Skipped for new projects | ✅ Qdrant + Neo4j + Postgres affected-files |
| 4     | ✅ Fresh polyglot scaffold; non-Python files (HTML/MD/JSON) no longer parsed with Python AST | ✅ Auto-clones repo via RepoWorkspaceManager; reads files local-first with GitHub API fallback; diff-based search_block/replace_block |
| 5     | ✅ Concurrent test gen, semgrep, pytest sandbox; early-exits cleanly when no Python source exists | ✅ Same |
| 6     | ✅ Pushes directly to main (no PR), idempotent | ✅ Creates feature branch, opens PR via PRManager |
| 7     | ✅ Deployment plan + monitoring + rollback | ✅ Same |

## Production-grade brownfield path

When a requirement targets an indexed project:

1. **Phase 0** matches the project via `project_embeddings` (score ≥ 0.7).
2. **Phase 0.5** builds a `repo_summary` from the Knowledge Layer and sets `build_mode = modify_existing`.
3. **Phase 3** runs Impact Analysis: Qdrant semantic search → top-K affected files; Neo4j → dependents; Postgres → OpenAPI/gRPC contracts; LLM → risk_level.
4. **Phase 4**:
   - `RepoWorkspaceManager` auto-clones the target repo into `WORKSPACE_ROOT/<repo_name>` if missing.
   - `load_existing_code` reads each affected file: local clone first, GitHub Contents API fallback.
   - `context_packet_builder` injects top-8 RAG chunks + top-4 full files into the codegen prompt.
   - LLM emits `search_block` / `replace_block` diffs; pipeline applies them with up to 3 retries per file.
5. **Phase 5** runs syntax check + Semgrep + pytest sandbox on the patched code.
6. **Phase 6** opens a feature branch and PR (idempotent via SHA-256 request ID).
7. After push, the Knowledge Layer auto-reindexes the repo so the next pipeline sees the new symbols.

## What is NOT yet verified

- Brownfield end-to-end on a non-trivial repo (the indexed conduit/flask/django repos haven't been pipeline-tested yet).
- Multi-repo changes in a single requirement (Phase 4 loops affected files but Phase 6 only pushes to one repo at a time).
- Neo4j path traversal under high fan-out (no perf benchmark).
- Phase 7 actual deployment — currently simulated.

## Known issues that don't block the pipeline

- `audit_log` schema mismatch: `agents/stage2_store.audit()` writes column `action`, table has column `event`. Audit rows from Stage 2 store are silently dropped. Pipeline continues. Fix with `ALTER TABLE audit_log ADD COLUMN action VARCHAR(100)` OR change the stage2_store column name to `event`.
- Phase 4's `goldenset.yaml` check warns when missing — this is intentional ("soft mode"); only matters if you want eval-driven codegen.

## Setup quickstart for brownfield

```bash
# 1. Index every repo you want the system to be able to modify
python knowledge_layer/indexer.py --repo-path ./repos/<repo> --repo-name <repo>

# 2. Register the project that groups those repos
python knowledge_layer/project_registry.py
# (edit the file to add your project definition)

# 3. Verify it's discoverable
curl http://localhost:8001/knowledge/repos

# 4. Launch a brownfield pipeline
curl -X POST http://localhost:8001/pipeline/start \
  -H "Content-Type: application/json" \
  -d '{"requirement": "Add a /v1/orders endpoint that returns paginated orders to conduit-django-api"}'
```

## Resume after a crash

POST `/pipeline/{thread_id}/resume` with `{"phase": N}` re-runs Phase N using saved state from PostgreSQL. No re-payment of earlier-phase LLM tokens.