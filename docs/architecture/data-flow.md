# Data Flow

## Request Lifecycle
POST /pipeline/start
{ "requirement": "..." }
↓
Phase 0 — Project Router
├─ Embed requirement (sentence-transformers)
├─ Qdrant search project_embeddings (top 3)
├─ Score >= 0.4? Use existing : Create fresh
└─ Save state → PostgreSQL pipelines
↓
Phase 1 — Discovery
├─ generate_brd → DeepSeek V4 Pro
├─ Save state, audit BRD_GENERATED
├─ generate_prd → DeepSeek V4 Pro
├─ Save state, audit PRD_GENERATED
├─ generate_adr → DeepSeek V4 Pro
├─ Save state, audit ADR_GENERATED
├─ generate_architecture → DeepSeek V4 Pro + deterministic Mermaid
├─ Save state, audit ARCHITECTURE_GENERATED
└─ INTERRUPT → wait for human approval
↓
POST /pipeline/approve/{thread_id}
{ "approved": true|false, "feedback": "..." }
↓
Approved → Phase 2
Rejected → re-run Phase 1 with feedback
↓
Phase 2 — Planning (analogous to Phase 1)
Phase 3 — Impact Analysis (only for existing projects)
Phase 4+5 — Code generation + tests (auto, no gate)
Phase 6 — Push + PR (creates repo via GitHub API if new)
Phase 7 — Deploy with monitoring + rollback

## State Persistence

Every node completion saves to PostgreSQL `pipelines` table:
```sql
INSERT INTO pipelines (thread_id, requirement, status, phase, sub_stage, current_state, ...)
ON CONFLICT (thread_id) DO UPDATE SET ...
```

Server restart → `load_all_pipelines()` restores everything to memory.

## Audit Trail

```sql
INSERT INTO audit_log (thread_id, phase, event, actor, details, created_at)
VALUES (...);
```

Events logged: PHASE_STARTED, BRD_GENERATED, PRD_GENERATED, ADR_GENERATED,
ARCHITECTURE_GENERATED, APPROVED, REJECTED, REPO_CREATED, ERROR

## Dashboard Polling

Dashboard polls `GET /pipeline/status/{thread_id}` every 3 seconds:
```json
{
  "thread_id": "...",
  "phase": 1,
  "status": "PHASE_1_PRD_DONE",
  "sub_stage": "PRD Done — Generating ADR...",
  "is_new_project": false,
  "selected_repos": [...],
  "current_state": { "brd": {...}, "prd": {...} }
}
```

Sub-stage chips show live progress within Phase 1 (BRD ✓ PRD ✓ ADR ○ Arch ○).

## Download Endpoints

| Endpoint | Returns |
|----------|---------|
| `/download/brd` | BRD markdown |
| `/download/prd` | PRD markdown |
| `/download/adr` | ADR markdown |
| `/download/architecture` | Architecture markdown with Mermaid |
| `/download/sprint-plan` | Sprint plan markdown |
| `/download/impact` | Impact report markdown |
| `/download/runbook` | Excel runbook |
| `/download/test-cases` | Excel test cases (Jira-driven) |
| `/download/all` | ZIP with everything + generated code + tests |