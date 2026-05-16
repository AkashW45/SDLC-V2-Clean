# Troubleshooting

Common issues and exact fixes.

## "API: offline" on dashboard

**Symptom:** Red dot, status says "offline".

**Causes:**
- Uvicorn not running → start with `uvicorn api.main:app --port 8001 --reload`
- Wrong port in dashboard → check `const API = ''` in `dashboard/index.html` (empty = same-origin, works on both `localhost` and `127.0.0.1`)
- CORS blocking → if accessing from a different origin, set `CORS_ORIGINS` in `.env`

## Pipeline starts but nothing renders on dashboard

**Symptom:** Modal closes, new pipeline doesn't appear in sidebar.

**Cause:** `watchPipeline()` not registering the thread ID.

**Fix:** Verify these two functions exist in `dashboard/index.html`:
```js
function watchPipeline(threadId) { ... }
async function discoverActivePipelines() { ... }
```

And on page load:
```js
checkHealth();
discoverActivePipelines();  // ← must be here
```

## "Not authenticated" on curl requests

**Symptom:** `{"detail":"Not authenticated"}`

**Fix:** Add the API key header:
```bash
curl -H "X-API-Key: sdlc-dev-key-12345" ...
```

Every write endpoint requires it. GET endpoints don't.

## Impact analysis returns files from unrelated repos

**Symptom:** Selected `flask-contacts-api`, but impact report shows `leave-mgmt-core` files.

**Cause:** `semantic_search` not filtering by repo.

**Fix:** Verify in `agents/phase3_impact/impact_analyzer.py`:
```python
def semantic_search(query: str, top_k: int = 3, repo_names: list = None) -> list:
```
Should accept `repo_names`. And `run_impact_analysis` should call it with:
```python
hits = semantic_search(requirement, top_k=8, repo_names=selected_repo_names or None)
```

Also confirm `graph.py` passes `selected_repos` through state, and `main.py`'s `run_phase3` initial dict includes `"selected_repos": prev_state.get("selected_repos", [])`.

## Mermaid diagram shows raw text, not rendered

**Symptom:** Architecture section shows `graph TD FLASK_API[...]` as text.

**Cause:** Either base64 encoding failed, or mermaid CDN didn't load.

**Fix:**
1. Open browser DevTools Console. Look for "Mermaid" errors.
2. Verify `<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>` is in `index.html` `<head>`.
3. Verify the architecture mermaid block uses base64-encoded `data-mermaid` attribute and `mermaid-container` class.

## Phase 4 generates 0 files

**Symptom:** "Code: 0 files, Tests: 0 files" after Phase 5 completes.

**Causes (in order of likelihood):**
1. **Impact analyzer found wrong files** → fix scoping (see above)
2. **`RepoWorkspaceManager` couldn't clone the repo** → check `WORKSPACE_ROOT` exists, GitHub App has access to the repo
3. **LLM returned invalid JSON** → check uvicorn log for `Skipping ticket — invalid LLM response`

Debug:
```bash
docker exec -it sdlc_postgres psql -U sdlc -d sdlc_knowledge \
  -c "SELECT jsonb_pretty(current_state->'generated_changes') FROM pipelines WHERE thread_id='<your_id>';"
```

## GitHub push fails

**Symptom:** Phase 6 status = ERROR. Logs show "401 Unauthorized" or "Repository not found".

**Causes:**
1. **GitHub App not installed on target repo** → go to `https://github.com/settings/installations/<id>` → confirm "All repositories" or include the target
2. **Wrong owner** → check `GITHUB_REPO_OWNER` in `.env`
3. **Private key missing** → verify `github_app.pem` is at `GITHUB_APP_PRIVATE_KEY_PATH`
4. **Installation token expired** → it shouldn't be (auto-refresh), but restart uvicorn to force a fresh token

## Indexer jobs stuck in QUEUED forever

**Symptom:** `/knowledge/jobs` shows jobs but no progression.

**Cause:** Indexer worker pool not started, or `index_repo` is hanging.

**Fix:**
```bash
# Check if the queue process is running
ps aux | grep indexer

# If not, kill uvicorn and restart — workers initialize on first /sync call
```

If a job is stuck in RUNNING for >10 minutes, the worker may have crashed. Reset:
```bash
docker exec -it sdlc_postgres psql -U sdlc -d sdlc_knowledge \
  -c "UPDATE indexer_jobs SET status='QUEUED' WHERE status='RUNNING' AND started_at < NOW() - INTERVAL '10 minutes';"
```

Then restart uvicorn — pending jobs auto-resume.

## Postgres port mismatch

**Symptom:** Various "connection refused" errors after `docker-compose up`.

**Cause:** `docker-compose.yml` exposes Postgres on port 5437 but `.env` says 5433 (or vice versa).

**Fix:** Both must match. Check:
```bash
docker ps | grep postgres
# Note the port mapping: 0.0.0.0:XXXX->5432
# Make POSTGRES_PORT in .env match XXXX
```

## "Audit failed: can't adapt type 'dict'"

**Symptom:** Warning in uvicorn log, pipelines still work.

**Cause:** `stage2_store.audit()` writes column `action` but table has `event`.

**Fix:** Non-blocking, but to silence:
```bash
docker exec -it sdlc_postgres psql -U sdlc -d sdlc_knowledge \
  -c "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS action VARCHAR(100);"
```

## Old pipelines clutter the sidebar

**Symptom:** Dashboard shows 30+ old pipelines from previous testing.

**Fix:**
```bash
docker exec -it sdlc_postgres psql -U sdlc -d sdlc_knowledge \
  -c "DELETE FROM pipelines WHERE status IN ('ERROR') OR created_at < NOW() - INTERVAL '7 days';"
```

Then restart uvicorn (pipelines reload from DB on startup).

## Phase 7 says "Deployment Complete" but nothing was deployed

**Symptom:** Phase 7 shows green ✅, but `docker ps` shows no new container.

**Cause:** **By design.** Phase 7 currently simulates deployment — it generates the plan but doesn't execute.

**Fix:** See [DEPLOYMENT.md](DEPLOYMENT.md) — "Making Phase 7 Actually Deploy" section.

## How to nuke everything and start fresh

```bash
docker-compose down -v   # -v removes volumes (deletes all data)
rm -rf repos/
rm -rf __pycache__ agents/__pycache__ knowledge_layer/__pycache__
docker-compose up -d --build
python knowledge_layer/db_setup.py
# Then re-sync via /knowledge/projects/sync
```

---

## Logs to check when debugging

| Where | What |
|---|---|
| Uvicorn terminal | Real-time Phase logs, LLM responses, errors |
| `audit_log` table | Phase events, user decisions |
| `pipelines.current_state` JSONB | Full state of any pipeline |
| `indexer_jobs` table | Indexing queue status |
| Browser DevTools Console | Frontend errors, network failures |
| Docker logs | `docker logs sdlc_postgres` etc. for DB issues |

---

## Where to get help

1. Check this doc first
2. Search `audit_log` for the failed thread ID
3. Reproduce in isolation: re-run with `routing_choice` explicit
4. Slack channel: `#sdlc-v2-platform`