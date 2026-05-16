# Quickstart — 15 Minutes to First Pipeline

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Docker Desktop | latest | Postgres, Qdrant, Neo4j |
| Python | 3.11+ | API server |
| Git | any | Repo operations |
| GitHub account | with admin on target org | App installation |

You will also need:
- **DeepSeek API key** (or any OpenAI-compatible endpoint)
- **Jira Cloud account** + API token (for Phase 2 ticket creation)
- A **GitHub App** installed on the org/account whose repos you want to operate on

---

## Step 1 — Clone and configure

```bash
git clone https://github.com/AkashW45/SDLC-V2-Clean.git
cd SDLC-V2-Clean
cp .env.example .env
```

Edit `.env`:

```env
# ── LLM ──────────────────────────────────────────────
DEEPSEEK_API_KEY=sk-...
LLM_API_KEY=sk-...                  # same as above
LLM_BASE_URL=https://api.deepseek.com

# ── Databases ────────────────────────────────────────
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5437                  # MUST match docker-compose
POSTGRES_USER=sdlc
POSTGRES_PASSWORD=sdlc1234
POSTGRES_DB=sdlc_knowledge

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password1234

QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# ── GitHub App (production auth) ─────────────────────
GITHUB_APP_ID=<your_app_id>
GITHUB_APP_INSTALLATION_ID=<your_installation_id>
GITHUB_APP_PRIVATE_KEY_PATH=./github_app.pem
GITHUB_REPO_OWNER=<your_org_or_user>
GITHUB_WEBHOOK_SECRET=<random_32_byte_hex>

# ── Jira ─────────────────────────────────────────────
JIRA_EMAIL=you@company.com
JIRA_API_TOKEN=...
JIRA_BASE_URL=yourcompany.atlassian.net
JIRA_PROJECT_KEY=DEV

# ── API key for dashboard + CLI ──────────────────────
API_SECRET_KEY=sdlc-dev-key-12345

# ── Workspace ────────────────────────────────────────
WORKSPACE_ROOT=./repos             # where repos get cloned
INDEXER_WORKERS=4                  # parallel indexing
PHASE0_MATCH_THRESHOLD=0.55        # auto-routing threshold (0-1)
```

## Step 2 — Set up GitHub App

This replaces the legacy PAT-based auth with production-grade GitHub App auth.

1. Go to `https://github.com/settings/apps/new`
2. Fill in:
   - **Name:** `<your-org>-sdlc-pipeline`
   - **Homepage URL:** `http://localhost:8001` (or production URL)
   - **Webhook URL:** `https://<ngrok-or-public-url>/webhooks/github` (skip if no public endpoint yet)
   - **Webhook Secret:** the value of `GITHUB_WEBHOOK_SECRET`
3. Permissions:
   - **Repository:** Contents (R/W), Metadata (R), Pull requests (R/W), Webhooks (R/W)
   - **Organization:** Members (R)
4. Events: ✅ Push, ✅ Repository, ✅ Pull request
5. Create app. Note the **App ID**.
6. Generate a **private key** — saves `<app-name>.pem`. Move it to project root as `github_app.pem`.
7. Click **Install App** → All repositories. Note the **Installation ID** from the URL.
8. Update `.env` with App ID, Installation ID.

## Step 3 — Start infrastructure

```bash
docker-compose up -d
```

Wait ~30 seconds. Verify:

```bash
docker ps
# Should show: sdlc_postgres, sdlc_qdrant, sdlc_neo4j (all healthy)
```

## Step 4 — Install Python deps + initialize databases

```bash
pip install -r requirements.txt
python knowledge_layer/db_setup.py
```

Expected output:
[PostgreSQL] ✅ All tables created successfully
[Qdrant] ✅ All collections ready
[Neo4j] ✅ Setup complete
## Step 5 — Start the API server

```bash
uvicorn api.main:app --port 8001 --reload
```

## Step 6 — Discover and sync your GitHub repos

```bash
# Preview (no writes)
curl -X POST http://localhost:8001/knowledge/projects/discover \
  -H "X-API-Key: sdlc-dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"min_group_size": 2}'

# Real sync — clones + indexes all repos
curl -X POST http://localhost:8001/knowledge/projects/sync \
  -H "X-API-Key: sdlc-dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"min_group_size": 2}'

# Watch progress
curl http://localhost:8001/knowledge/jobs | python -m json.tool
```

When all jobs show `SUCCESS`, the Knowledge Layer is ready.

## Step 7 — Open the dashboard

Visit `http://localhost:8001/dashboard`. You should see:
- API: connected (green dot)
- List of sample/existing pipelines in left sidebar
- Empty input box at top

## Step 8 — Run your first pipeline

1. Type a requirement:
Add a DELETE /contacts/{id} endpoint to flask-contacts-api
2. Click **▶ Launch Pipeline**
3. Modal appears — pick `Flask Contacts API` as the routing target → **Launch**
4. Watch phases progress in real-time
5. At each approval gate, click **✓ Approve & Continue**
6. After Phase 6, find your PR link at the bottom of the page

That's it.

## Verify the PR was actually pushed

Go to `https://github.com/<your-org>/flask-contacts-api/branches`. You should see `feature/pipeline-<thread_id>` with recent commits.

---

## Common First-Run Issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).