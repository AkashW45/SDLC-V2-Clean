# AI-Powered SDLC Automation Platform V2

## What Is This?
An enterprise-grade AI automation platform that takes a plain English business requirement and runs the entire software development lifecycle — from BRD generation through to production deployment — across multiple repos with human approval gates at every critical step.

## Key Capabilities
- **Plain English in, working code out** — no developer needed for routine changes
- **Knowledge-aware** — understands your codebase via AST + dependency graph + vector search
- **Smart repo routing** — auto-detects existing project match OR creates fresh repos for new ideas
- **5 human approval gates** — humans stay in control at Discovery, Planning, Impact, Delivery, Deployment
- **Persistent state** — pipelines survive server restarts (PostgreSQL backed)
- **Full audit trail** — every action logged with actor + timestamp
- **Reject-and-regenerate** — feedback flows back into LLM prompts for retry
- **Live dashboard** — sub-stage progress, Mermaid diagrams, Jira links, downloadable artifacts

## Architecture
- **LangGraph** — orchestrates all 8 phases (Phase 0-7) with INTERRUPT-based human gates
- **DeepSeek V4 Pro** — primary LLM with thinking mode for high-quality reasoning
- **Qdrant** — vector search across code symbols + project descriptions
- **Neo4j** — file/symbol dependency graph for impact analysis
- **PostgreSQL** — AST symbol index, protocol contracts, repo maps, projects, pipelines, audit log
- **n8n** — external workflow integrations (planned: Slack, TeamCity webhooks)
- **FastAPI** — HTTP layer + dashboard host

## The 8 Phases

| Phase | Name | Human Gate | Status |
|-------|------|------------|--------|
| 0 | Project & Repo Selector | ✅ | ✅ Done |
| 1 | Discovery — BRD / PRD / ADR / Architecture | ✅ | ✅ Done |
| 2 | Planning — Sprint + Jira + Runbook | ✅ | ✅ Done |
| 3 | Impact Analysis | ✅ | ✅ Done |
| 4 | Code Generation (existing OR fresh project) | ❌ Auto | ✅ Done |
| 5 | Validation + Test Generation (Jira-driven) | ❌ Auto | ✅ Done |
| 6 | Delivery — Push + GitHub PR | ✅ | ✅ Done |
| 7 | Deployment — Sequence + Flags + Monitor + Rollback | ✅ | ✅ Done |

## Infrastructure

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | localhost:8001 | Backend API + Dashboard |
| Dashboard | localhost:8001/dashboard | Live pipeline control tower |
| Qdrant | localhost:6333 | Vector search |
| Neo4j | localhost:7474 | Dependency graph (auth: neo4j/password1234) |
| PostgreSQL | localhost:5433 | Symbol index + Pipelines + Audit (db: sdlc_knowledge) |
| n8n | localhost:5678 | External workflow integrations |

## Quick Start

```bash
# 1. Start infrastructure (Qdrant + Neo4j + PostgreSQL + API)
docker-compose up -d --build

# 2. Set environment variables
cp .env.example .env
# Edit .env with: DEEPSEEK_API_KEY, GITHUB_TOKEN, JIRA_EMAIL, JIRA_API_TOKEN

# 3. Initialize databases
python knowledge-layer/db_setup.py

# 4. Register demo projects
python knowledge-layer/project_registry.py

# 5. Index existing repos (optional — only for matching against existing codebases)
python knowledge-layer/indexer.py --repo-path C:\path\to\repo --repo-name my-repo

# 6. Open dashboard
open http://localhost:8001/dashboard
```

## How It Works (End-to-End)

1. **User submits requirement** in dashboard ("Build a payment reconciliation system")
2. **Phase 0** searches existing projects via vector similarity → either matches existing OR marks as new project (auto-generates fresh repo names)
3. **Phase 1** generates BRD → PRD → ADR → Architecture (with traced-to-requirement Mermaid diagram). Human approves OR rejects with feedback (regenerates with feedback baked in)
4. **Phase 2** creates sprint plan + real Jira tickets + deployment runbook. Human approves
5. **Phase 3** analyzes impact via Knowledge Layer (only for existing projects). Human approves
6. **Phase 4** generates code — modifies existing files OR scaffolds fresh project structure
7. **Phase 5** auto-generates pytest tests + Jira-driven manual test cases (XLSX export)
8. **Phase 6** pushes to GitHub branch (creates new repo via API if needed) + opens PR. Human reviews PR
9. **Phase 7** resolves deploy sequence → feature flags → human approves → simulated deploy → health monitoring → auto-rollback if metrics degrade

## Documentation Index

- [Architecture Overview](architecture/overview.md)
- [Tech Stack](architecture/tech-stack.md)
- [Phase Details](phases/) — one file per phase
- [Infrastructure Setup](infrastructure/docker-setup.md)
- [API Reference](api/endpoints.md)
- [Dashboard Guide](dashboard/usage.md)
- [Demo Script](demo/demo-script.md)

## What's New in V2 vs V1

| V1 | V2 |
|----|-----|
| n8n orchestrator | LangGraph orchestrator |
| Single Leave Mgmt repo | Multi-project + multi-repo + auto-create new repos |
| 4 phases | 8 phases (added Phase 0 selector + Phase 7 deployment) |
| Groq gpt-oss-120b (8K context) | DeepSeek V4 Pro (1M context, thinking mode) |
| In-memory state | PostgreSQL persistence + audit log |
| No reject regeneration | Reject-and-regenerate with LLM feedback |
| Static documents | Mermaid architecture diagrams + sub-stage live progress |
| No project intelligence | Qdrant project_embeddings for smart routing |

## Critical Rules (NEVER VIOLATE)
1. LangGraph orchestrates everything; n8n only handles external API calls
2. Human gates use LangGraph INTERRUPT — not n8n Wait nodes
3. LLM never sees full codebase — only Knowledge Layer context packets
4. Never auto-merge PRs — Phase 6 always requires human review
5. Always validate generated code with `ast.parse()` + retry x3
6. Architecture nodes must trace to specific requirements (no random tech inventions)