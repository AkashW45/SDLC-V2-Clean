# AI-Powered SDLC Automation Platform V2

## What Is This?
An enterprise-grade AI automation platform that takes a plain English 
business requirement and automatically manages the entire software 
development lifecycle across 50+ repos in multiple languages.

## Architecture
- **LangGraph** — orchestrates all 7 agent phases
- **n8n** — handles all external API calls (Jira, GitHub, Slack, TeamCity)
- **Qdrant** — vector search across all repos
- **Neo4j** — dependency graph across repos
- **PostgreSQL** — AST symbol index, protocol contracts, repo maps
- **Groq** — LLM for all AI generation tasks

## The 7 Phases
| Phase | Name | Status |
|-------|------|--------|
| 1 | Discovery — BRD/PRD/ADR | 🔨 Building |
| 2 | Planning — Jira + Runbook | 🔨 Building |
| 3 | Impact Analysis | 🔨 Building |
| 4 | Code Generation | 🔨 Building |
| 5 | Validation | 🔨 Building |
| 6 | Delivery — PR + Build | 🔨 Building |
| 7 | Deployment | 🔨 Building |

## Infrastructure
| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI | localhost:8000 | Backend API |
| n8n | localhost:5678 | Workflow orchestration |
| Qdrant | localhost:6333 | Vector search |
| Neo4j | localhost:7474 | Dependency graph |
| PostgreSQL | localhost:5433 | Symbol index |

## Quick Start
```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your keys

# 4. Start API
cd api
uvicorn main:app --reload
```

## Documentation
- [Architecture Overview](architecture/overview.md)
- [Tech Stack](architecture/tech-stack.md)
- [Phase 1 — Discovery](phases/phase1-discovery.md)
- [Phase 2 — Planning](phases/phase2-planning.md)
- [Phase 3 — Impact Analysis](phases/phase3-impact-analysis.md)
- [Phase 4 — Code Generation](phases/phase4-code-generation.md)
- [Phase 5 — Validation](phases/phase5-validation.md)
- [Phase 6 — Delivery](phases/phase6-delivery.md)
- [Phase 7 — Deployment](phases/phase7-deployment.md)
- [Infrastructure Setup](infrastructure/docker-setup.md)
- [Demo Script](demo/demo-script.md)