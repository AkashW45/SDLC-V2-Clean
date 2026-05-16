# Deployment Guide

This doc covers two scenarios:
1. **Deploying SDLC-V2 itself** (the platform)
2. **Making Phase 7 actually deploy generated code** (currently simulated)

---

## Scenario 1: Deploying SDLC-V2 Platform

### Recommended: Single-host Docker Compose

For dev, staging, or low-traffic production:

```bash
# On the target server
git clone https://github.com/AkashW45/SDLC-V2-Clean.git
cd SDLC-V2-Clean

# Copy .env with production values
# Mount github_app.pem with strict perms
chmod 600 github_app.pem

# Start everything
docker-compose up -d --build

# Verify
docker-compose ps
curl http://localhost:8001/health
```

Default `docker-compose.yml` exposes:
- `8001` — API + dashboard
- `5437` — Postgres (host-mapped)
- `6333` — Qdrant
- `7687` — Neo4j Bolt
- `7474` — Neo4j Browser

For production, put a reverse proxy (nginx, Caddy, Traefik) in front of port 8001 with TLS termination.

### Scaling considerations

| Component | Bottleneck | Scale by |
|---|---|---|
| API | LLM API rate limits | Run multiple uvicorn workers behind nginx |
| Indexer | CPU + I/O for embeddings | Bump `INDEXER_WORKERS` in `.env` |
| Postgres | Connection count | Use PgBouncer; increase `max_connections` |
| Qdrant | Memory (indices in RAM) | Vertical scale; Qdrant supports clustering for large fleets |
| Neo4j | Single-instance default | Neo4j Aura or enterprise cluster for HA |

### Production checklist

- [ ] Strong DB passwords (rotate `POSTGRES_PASSWORD`, `NEO4J_PASSWORD`)
- [ ] Rotate `API_SECRET_KEY` to a random 32-byte hex
- [ ] Rotate `GITHUB_WEBHOOK_SECRET` (any leaked value invalidates webhook auth)
- [ ] `github_app.pem` permissions = 600, not committed to git
- [ ] `.env` permissions = 600
- [ ] Daily Postgres backups (the `pipelines` table is your source of truth)
- [ ] Daily Qdrant snapshots (rebuilding embeddings is expensive)
- [ ] Webhook URL publicly accessible (load balancer or ngrok-like tunnel)
- [ ] Reverse proxy with TLS in front of port 8001
- [ ] Container restart policy = `unless-stopped` (already set in docker-compose.yml)
- [ ] Monitor disk: cloned repos in `./repos/` grow with org size

### Kubernetes

For 1000+ repos or HA needs, port to Kubernetes:
- Each Postgres, Qdrant, Neo4j as its own StatefulSet with PVC
- API as Deployment (3+ replicas)
- Indexer queue moves out of in-process ThreadPool → Celery/RQ on Redis
- Ingress (e.g. nginx-ingress) → TLS → API service

This is ~1 day of work. Helm chart is on the roadmap.

---

## Scenario 2: Making Phase 7 Actually Deploy

Phase 7 currently *generates* a deployment plan (Docker build, kubectl apply, etc.) but does NOT execute it. To make it real, wire `subprocess.run` calls into `agents/phase7_deployment/`.

### Why it's simulated by default

- Deploying generated code from an AI is risky → most teams want a manual gate
- Different orgs have different deployment systems (Docker, K8s, Heroku, AWS ECS, etc.)
- Real deploy needs credentials (cloud, registry, kubectl) which the platform shouldn't accumulate

### Minimal real-deploy implementation

If you're running everything locally and want to actually deploy after Phase 6 merges:

```python
# In agents/phase7_deployment/deployment_agent.py, add a new node:

import subprocess

def execute_deployment_real(state):
    """Actually run the deploy steps from the runbook."""
    runbook = state.get("runbook", {})
    for step in runbook.get("deployment_sequence", []):
        cmd = step.get("command", "")
        if not cmd:
            continue
        # Only run safe-allowlisted commands
        if not any(cmd.startswith(safe) for safe in ["docker ", "kubectl ", "git "]):
            print(f"[Phase 7] Skipping non-allowlisted: {cmd}")
            continue
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {**state, "status": "DEPLOY_FAILED", "error": result.stderr}
    return {**state, "status": "DEPLOY_COMPLETE"}
```

Then add the node to the Phase 7 graph **after** the human approval gate. Production-grade approach:
- Use a sandboxed runner (e.g. dedicated CI runner with limited credentials)
- Allowlist exact commands, not just prefixes
- Log every deploy command + output to `audit_log`
- Roll back automatically on healthcheck failure

### Recommended path: GitHub Actions

Better than running deploys inside SDLC-V2:

1. SDLC-V2 opens a PR (Phase 6 — works today)
2. Reviewer approves and merges
3. GitHub Actions workflow on `main` deploys (your existing CI/CD)
4. Phase 7 just outputs the plan + tracks the deployment via GitHub Deployments API

This way SDLC-V2 stays *out* of the deployment runtime — exactly where it should be for a Morgan-Stanley-grade audit story.

---

## Backups and Restore

### What to back up

| Data | Where | Frequency |
|---|---|---|
| Pipeline state | Postgres `pipelines` table | Daily |
| Audit log | Postgres `audit_log` table | Daily |
| Project registry | Postgres `projects` + Qdrant `project_embeddings` | After every registry change |
| Code embeddings | Qdrant `code_embeddings` collection | Daily (or rebuild from indexer) |
| Cloned repos | `./repos/` directory | Don't back up — re-clone on demand |

### Quick backup

```bash
# Postgres
docker exec sdlc_postgres pg_dump -U sdlc sdlc_knowledge > backup_$(date +%Y%m%d).sql

# Qdrant — uses its own snapshot API
curl -X POST http://localhost:6333/collections/code_embeddings/snapshots
```

### Restore

```bash
# Postgres
cat backup_20260516.sql | docker exec -i sdlc_postgres psql -U sdlc sdlc_knowledge

# Qdrant — see https://qdrant.tech/documentation/concepts/snapshots/
```