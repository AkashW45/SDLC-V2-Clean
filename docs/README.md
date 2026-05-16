# SDLC-V2 — Autonomous Multi-Agent SDLC Pipeline

**Take a one-line business requirement → ship validated, tested code as a GitHub PR.**

SDLC-V2 is a production-grade AI platform that automates the full software development lifecycle for both new and existing codebases. It routes requirements to the right repos, generates BRD/PRD/architecture, plans sprints with Jira tickets, performs code-change impact analysis, generates code and tests, and opens pull requests — all with human approval gates at every phase.
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   "Add a DELETE endpoint to flask-contacts-api"                    │
│                          ↓                                         │
│   Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → 6→7 │
│   Routing  Discovery  Planning   Impact   Codegen  Validate  PR    │
│                          ↓                                         │
│   ✅ GitHub PR ready for review on feature/pipeline-xxx            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
## What It Does

| Phase | What happens | Time |
|------:|--------------|------|
| 0 | Routes requirement to existing project OR creates new one | 2s |
| 1 | Generates BRD, PRD, ADR, Architecture diagram | 30-60s |
| 2 | Generates sprint plan + Jira tickets + deployment runbook | 30-45s |
| 3 | Impact analysis (Qdrant + Neo4j + Postgres) | 15-30s |
| 4 | Generates code using RAG over indexed repo | 60-120s |
| 5 | Validates: AST + Semgrep + pytest sandbox | 30-60s |
| 6 | Pushes to GitHub feature branch + opens PR | 10s |
| 7 | Generates deployment plan with rollback | 10s |

**Total:** 3-6 minutes for a working PR.

## Two Modes

- **Greenfield** — New project. Creates fresh repos, scaffolds code from scratch.
- **Brownfield** — Existing project. Indexes your real GitHub repos, performs impact analysis, generates diff-based patches that fit your codebase.

## Key Features

- 🧠 **Adaptive Scope Profile (ASP)** — Right-sizes output: a hello-world gets 1 ticket; an enterprise platform gets 30. No fixed counts.
- 🔍 **Knowledge Layer** — Auto-discovers your GitHub org, indexes all repos to Qdrant (vectors) + Neo4j (graph) + Postgres (symbols).
- 🔐 **GitHub App authentication** — Production-grade JWT auth, not long-lived PATs.
- ⚡ **Change-driven indexing** — Webhooks + reconciliation. Indexing cost scales with repo *changes*, not pipeline *runs*.
- 👤 **Human-in-the-loop** — Approval gates at phases 1, 2, 3, 6, 7.
- 📊 **Live dashboard** — Real-time status, Mermaid architecture diagrams, PR links.
- ♻️ **Resumable** — Crash anywhere; resume the phase without re-paying earlier LLM tokens.

## Quick Links

| If you want to... | Read |
|---|---|
| Get it running locally | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Understand the internals | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Deploy to production | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Run the Morgan Stanley demo | [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) |
| Debug something | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + FastAPI |
| LLM | DeepSeek V3 (configurable via `core/llm_gateway.py`) |
| Vector DB | Qdrant |
| Graph DB | Neo4j |
| Relational DB | PostgreSQL 16 |
| Embeddings | sentence-transformers (MiniLM-L6-v2, local) |
| Auth | GitHub App (JWT + installation tokens) |
| Frontend | Vanilla JS + Mermaid.js (no build step) |

## License

Proprietary — internal use only.