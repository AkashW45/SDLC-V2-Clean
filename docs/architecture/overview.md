# Architecture Overview

## Core Principle
The LLM does NOT need the entire codebase.
It needs the RIGHT SUBSET of context for the specific change.

Flow:Requirement → Retrieve relevant files → Build context packet → LLM plans changes → Codemods apply changes → Tests validate → PR created
## LangGraph vs n8n Boundary

### LangGraph owns:
- All AI reasoning and decision making
- Agent graph flow and state management  
- All 5 human approval INTERRUPT points
- Calling n8n via HTTP webhooks for external actions

### n8n owns:
- All external API calls (Jira, GitHub, Slack, TeamCity, Confluence)
- Webhooks received from external systems
- Scheduled jobs (nightly re-index, health checks)
- Notifications and alerts

### Critical Rule:
- NEVER put external API calls inside LangGraph nodes
- NEVER put AI reasoning inside n8n
- LangGraph node → HTTP POST to n8n webhook → n8n executes → returns result

## Two Layers

### Layer 1 — Knowledge Layer (Offline)
Runs as scheduled job after every repo merge.
Builds and maintains the code intelligence backend.

Stores:
- AST symbol index → PostgreSQL
- Dependency graph → Neo4j  
- Vector embeddings → Qdrant
- Protocol contracts → PostgreSQL + Qdrant
- Repo maps → PostgreSQL

### Layer 2 — Execution Layer (Runtime)
7 LangGraph agent phases triggered per requirement.
Each phase queries the Knowledge Layer for context.

## Human Approval Gates
| Gate | After Phase | What Human Reviews |
|------|------------|-------------------|
| Gate 1 | Phase 1 | BRD + PRD documents |
| Gate 2 | Phase 2 | Sprint plan + Runbook |
| Gate 3 | Phase 3 | Impact report + Risk level |
| Gate 4 | Phase 6 | PRs in each repo |
| Gate 5 | Phase 7 | Production deployment |

All gates = LangGraph INTERRUPT nodes (not n8n Wait nodes).