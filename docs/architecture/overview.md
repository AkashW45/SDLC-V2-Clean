# Architecture Overview

## Core Principle
The LLM does NOT need the entire codebase.
It needs the RIGHT SUBSET of context for the specific change.

Requirement → Project router → Knowledge Layer retrieval →
Context packet → LLM plans changes → Codemods apply →
Tests validate → PR created → Deploy with monitoring
## LangGraph vs n8n Boundary

### LangGraph owns
- All AI reasoning and decision-making
- Agent graph flow + state management
- All 5 human approval INTERRUPT points (Phase 1, 2, 3, 6, 7)
- Persistence to PostgreSQL after every node
- Audit logging on every state transition
- Reject-and-regenerate flow with feedback injection

### n8n owns (planned)
- Slack notifications
- TeamCity build triggers
- Confluence updates
- Nightly indexer schedule
- Rollback alerts

### Critical Rule
- NEVER put external API calls inside LangGraph nodes (except GitHub/Jira clients we wrap directly)
- NEVER put AI reasoning inside n8n
- LangGraph node → HTTP POST to n8n webhook → n8n executes → returns result

## Two Layers

### Layer 1 — Knowledge Layer (Offline)
Runs as scheduled job after every repo merge. Builds and maintains code intelligence.

Stores:
- Project registry → PostgreSQL `projects` table + Qdrant `project_embeddings`
- AST symbol index → PostgreSQL `symbols` table
- Dependency graph → Neo4j (File-IMPORTS-File, File-DEFINES-Symbol)
- Code embeddings → Qdrant `code_embeddings`
- Protocol contracts → PostgreSQL `protocol_contracts` + Qdrant `contract_embeddings`
- Repo maps → PostgreSQL `repo_maps`

### Layer 2 — Execution Layer (Runtime)
8 LangGraph agent phases triggered per requirement. Each phase queries Knowledge Layer for context.

State persisted to PostgreSQL `pipelines` table after every node — survives server restart.

## Smart Repo Routing (Phase 0)
Requirement → Embed → Search project_embeddings (top 3)
↓
Score >= 0.4? ─── Yes ──→ Use existing project + indexed repos
│
No ──→ Auto-generate slug → Create fresh repos:
- {slug}-backend
- {slug}-frontend
(GitHub repos created on-demand in Phase 6)

## Human Approval Gates

| Gate | Phase | Reviews | Reject Action |
|------|-------|---------|----------------|
| Gate 1 | Phase 1 | BRD + PRD + ADR + Architecture diagram | Regenerate Phase 1 with feedback |
| Gate 2 | Phase 2 | Sprint plan + Jira tickets + Runbook | Regenerate Phase 2 with feedback |
| Gate 3 | Phase 3 | Impact report + Risk level | Regenerate Phase 3 with feedback |
| Gate 4 | Phase 6 | GitHub PR | Reject (no auto-merge) |
| Gate 5 | Phase 7 | Production deployment | Halt deployment |

All gates use LangGraph `interrupt()` (not n8n Wait nodes).

## Persistence & Audit

Every pipeline state mutation persists to PostgreSQL:
- `pipelines` table — full state JSONB, survives restart
- `audit_log` table — every event with actor, phase, details, timestamp

API endpoint `/pipeline/{thread_id}/audit` returns full timeline.

## Reject-and-Regenerate

When user rejects a phase with feedback:
1. Feedback saved to `human_feedback` field in state
2. Background task re-runs the phase
3. Phase prompt prepends feedback block:

IMPORTANT — USER FEEDBACK on previous attempt (you MUST address all of this):
{feedback}

4. New artifacts replace old ones in pipeline state

## Architecture Diagram Generation

Phase 1 generates Mermaid diagrams deterministically from architecture nodes/edges (server-side, not LLM-generated). Each node traces to a specific PRD requirement — no invented technologies.

Dashboard renders Mermaid client-side via mermaid.js.