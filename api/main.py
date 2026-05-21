"""
SDLC Automation Platform V2 — FastAPI HTTP Layer
Unified Version: Blends Surgical Replays & API Security with Phase 0 Smart Routing. details in main.py
"""
import io
import os
import re
import hashlib
from time import time
import uuid
import json
import asyncio
import requests
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, Depends, Security, Response, Request, logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import zipfile

from streamlit import feedback
from agents.repo_workspace import get_repo_local_path
import time
import threading
from langgraph.types import Command
import logging

# ADD THESE TWO LINES TO FIX THE CRASH
logger = logging.getLogger("reconciliation")
logger.setLevel(logging.INFO)
from api.runbook_export import (
    export_runbook_excel, export_brd_markdown, export_prd_markdown,
    export_adr_markdown, export_architecture_markdown, export_sprint_plan_markdown,
    export_impact_markdown
)

load_dotenv()

from api.persistence import (
    init_persistence_tables, save_pipeline, load_all_pipelines, audit, get_audit_log
)

# Safely import Shantanu's persistence tools (fallback if not yet implemented in DB)
try:
    from api.persistence import create_replay_job, update_replay_job, get_artifact, save_artifact
    SURGICAL_REPLAY_ENABLED = True
except ImportError:
    SURGICAL_REPLAY_ENABLED = False
    print("[Warning] Surgical Replay persistence functions missing. Will fallback to standard regeneration.")


# ── In-memory pipeline state store ────────────────────────────────────────────
pipeline_store: dict = {}

init_persistence_tables()
try:
    restored = load_all_pipelines()
    pipeline_store.update(restored)
    print(f"[Startup] Restored {len(restored)} pipelines from DB")
except Exception as e:
    print(f"[Startup] DB restore failed: {e}")


# ── App & Security ────────────────────────────────────────────────────────────
app = FastAPI(
    title="SDLC Automation Platform V2",
    description="AI-powered SDLC pipeline with Surgical Replays & Smart Routing",
    version="2.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_methods=["*"],
    allow_headers=["*"]
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
VALID_API_KEY = os.getenv("API_SECRET_KEY", "sdlc-dev-key-12345")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorised")


# ── Request Models ────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    requirement: str
    thread_id: Optional[str] = None

class ApproveRequest(BaseModel):
    approved: bool
    feedback: Optional[str] = ""

class IndexRequest(BaseModel):
    repo_path: str
    repo_name: str

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class PipelineStartRequest(BaseModel):
    requirement: str
    # If set, Phase 0 is BYPASSED and this choice is used directly.
    routing_choice: Optional[dict] = None
    # Examples of routing_choice:
    #   {"mode": "existing", "project_id": "conduit", "repo_names": ["conduit-django-api"]}
    #   {"mode": "new"}    -> creates new project
    #   {"mode": "manual", "repo_names": ["flask-contacts-api"]} -> use raw repos


# In run_phase1, near the top, after thread_id is known:
def _routing_choice_to_repos(routing_choice: dict) -> list:
    """Convert the user's modal pick into a selected_repos list."""
    if not routing_choice:
        return []
    mode = routing_choice.get("mode", "").lower()

    if mode == "existing":
        # User picked an existing project
        project_id = routing_choice.get("project_id")
        repo_names = routing_choice.get("repo_names", [])
        if repo_names:
            return [{"name": n} for n in repo_names]
        elif project_id:
            return [{"name": project_id}]
        return []

    elif mode == "manual":
        # User manually picked repo names
        return [{"name": n} for n in routing_choice.get("repo_names", [])]

    elif mode == "new":
        # New project — no existing repos
        return []

    return []

def _warn_if_stale_index(thread_id: str, matched_repo: str):
    """Non-blocking drift check; logs warning only."""
    if not matched_repo:
        return
    try:
        from agents.github_auth import get_github_headers
        from knowledge_layer.indexer import _get_last_indexed_sha
        import requests
        owner = os.getenv("GITHUB_REPO_OWNER", "")
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{matched_repo}",
            headers=get_github_headers(), timeout=5,
        )
        if resp.status_code != 200:
            return
        default_branch = resp.json().get("default_branch", "main")
        br = requests.get(
            f"https://api.github.com/repos/{owner}/{matched_repo}/branches/{default_branch}",
            headers=get_github_headers(), timeout=5,
        )
        if br.status_code != 200:
            return
        current_sha = br.json()["commit"]["sha"]
        stored_sha = _get_last_indexed_sha(matched_repo)
        if stored_sha and current_sha != stored_sha:
            logger.warning(f"[Phase 0] Stale index for {matched_repo}: "
                           f"stored={stored_sha[:8]} current={current_sha[:8]}")
            audit(thread_id, "phase0", "STALE_INDEX_WARNING",
                  {"repo": matched_repo, "stored": stored_sha, "current": current_sha})
    except Exception as e:
        logger.debug(f"[Phase 0] drift check skipped: {e}")

# At top of api/main.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

_scheduler: BackgroundScheduler = None

from fastapi import Header

from fastapi import Header

def get_user_id(x_user_id: str = Header(default="anonymous", alias="X-User-Id")) -> str:
    return (x_user_id or "anonymous").strip().lower()

def _reconciliation_job():
    """Detect drift between GitHub default-branch SHAs and what we have indexed."""
    try:
        from agents.github_discovery import list_all_repos
        from agents.indexer_queue import get_queue
        from agents.github_auth import get_github_headers
        from knowledge_layer.indexer import _get_last_indexed_sha
        import requests

        repos = list_all_repos()
        queue = get_queue()
        drift = 0
        for r in repos:
            try:
                resp = requests.get(
                    f"https://api.github.com/repos/{r['full_name']}/branches/{r['default_branch']}",
                    headers=get_github_headers(), timeout=10,
                )
                if resp.status_code != 200:
                    continue
                current_sha = resp.json()["commit"]["sha"]
                stored_sha = _get_last_indexed_sha(r["name"])
                if current_sha != stored_sha:
                    queue.enqueue(r["name"], r["url"], r["default_branch"])
                    drift += 1
            except Exception as e:
                logger.warning(f"[Reconciliation] {r['name']}: {e}")
        logger.info(f"[Reconciliation] checked {len(repos)} repos, enqueued {drift} drifted")
    except Exception as e:
        logger.error(f"[Reconciliation] global error: {e}")


@app.on_event("startup")
def warmup_embedder():
    # Pay the one-time model load (download + init) at server boot rather than
    # on the first user request. Without this, the first /pipeline/preview-routing
    # call appears to "hang" for seconds while the model loads. Runs in a daemon
    # thread so it never blocks the server from accepting connections.
    import threading

    def _warm():
        try:
            from core.embeddings import warmup
            warmup()
            logger.info("[startup] embedding model warmed up")
        except Exception as e:
            logger.warning(f"[startup] embedder warmup skipped: {e}")

    threading.Thread(target=_warm, name="embedder-warmup", daemon=True).start()


@app.on_event("startup")
def start_reconciliation():
    global _scheduler
    interval_min = int(os.getenv("RECONCILIATION_INTERVAL_MIN", "60"))
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _reconciliation_job,
        IntervalTrigger(minutes=interval_min, jitter=300),
        id="indexing_reconciliation",
        max_instances=1,           # don't pile up if one run is slow
        coalesce=True,             # collapse missed runs into one
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(f"[Reconciliation] scheduled every {interval_min} min")


@app.on_event("shutdown")
def stop_reconciliation():
    if _scheduler:
        _scheduler.shutdown(wait=False)


@app.post("/admin/reconciliation/run")
def trigger_reconciliation(api_key: str = Depends(verify_api_key)):
    """Manual trigger for the drift check. Runs in background."""
    if _scheduler:
        _scheduler.add_job(_reconciliation_job, id=f"recon-manual-{int(time.time())}")
    return {"status": "TRIGGERED"}

# ── Shantanu's Surgical Replay Logic ──────────────────────────────────────────
def extract_target_artifact(payload: dict) -> str:
    """Determine exactly which artifact needs re-generation from human feedback."""
    artifact = payload.get("artifact")
    if isinstance(artifact, str) and artifact.strip():
        normalized = artifact.strip().upper()
        if normalized == "ARCHITECTURE":
            return "Architecture"
        return normalized

    feedback = str(payload.get("feedback", "")).lower()
    if "adr" in feedback: return "ADR"
    if "prd" in feedback: return "PRD"
    if "brd" in feedback: return "BRD"
    if "architecture" in feedback or "arch" in feedback: return "Architecture"
    return "ADR" # Default fallback


def execute_surgical_replay(job_id: int, thread_id: str, target_artifact: str, feedback: str):
    """Regenerates ONLY the rejected artifact instead of the whole phase."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase1_discovery.discovery_agent import (
            generate_brd, generate_prd, generate_adr, generate_architecture
        )

        old_version = 0
        if SURGICAL_REPLAY_ENABLED:
            previous_artifact = get_artifact(thread_id, target_artifact)
            old_version = previous_artifact.get("version", 0)

        state_data = pipeline_store[thread_id].get("current_state", {})

        state = {
            "requirement": pipeline_store[thread_id]["requirement"],
            "brd": state_data.get("brd", {}),
            "prd": state_data.get("prd", {}),
            "adr": state_data.get("adr", {}),
            "architecture": state_data.get("architecture", {}),
            "human_feedback": feedback,
            "approved": False,
            "status": "STARTED",
            "thread_id": thread_id
        }

        generator_map = {
            "BRD": generate_brd,
            "PRD": generate_prd,
            "ADR": generate_adr,
            "ARCHITECTURE": generate_architecture
        }

        generator = generator_map.get(target_artifact.upper())
        if not generator:
            raise ValueError(f"Unsupported artifact for replay: {target_artifact}")

        print(f"[Surgical Replay] Regenerating {target_artifact} for {thread_id}...")
        result = generator(state)

        artifact_key = "architecture" if target_artifact.upper() == "ARCHITECTURE" else target_artifact.lower()
        pipeline_store[thread_id]["current_state"][artifact_key] = result.get(artifact_key, {})

        diff_summary = f"Surgically updated {artifact_key} based on feedback."

        if SURGICAL_REPLAY_ENABLED:
            artifact_body = result.get(artifact_key, {})
            content = json.dumps(artifact_body, indent=2, ensure_ascii=False)
            new_version = save_artifact(thread_id, artifact_key, "Phase 1 - Discovery Replay", content)
            diff_summary = f"Updated {artifact_key} from version {old_version} to version {new_version}"
            update_replay_job(job_id, "COMPLETED", diff_summary)

        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_url:
            try:
                requests.post(slack_url, json={"text": f"✅ Replay Job {job_id} completed for {thread_id}. Artifact: {artifact_key}. Diff: {diff_summary}"})
            except Exception as slack_error:
                print(f"[Slack] notification failed: {slack_error}")

        audit(thread_id, "phase1", "REPLAY_COMPLETED", "system", {"job_id": job_id, "artifact": artifact_key, "diff_summary": diff_summary})

        pipeline_store[thread_id]["status"] = "WAITING_PHASE_1_APPROVAL"
        pipeline_store[thread_id]["sub_stage"] = f"Replay completed for {artifact_key} — Awaiting Review"
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))

    except Exception as e:
        error_message = str(e)
        if SURGICAL_REPLAY_ENABLED:
            update_replay_job(job_id, "FAILED", error_message)
        audit(thread_id, "phase1", "REPLAY_FAILED", "system", {"job_id": job_id, "error": error_message})
        pipeline_store[thread_id]["status"] = "ERROR"
        pipeline_store[thread_id]["error"] = f"Replay failed: {error_message}"
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))


# ── Core Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check all service connections."""
    services = {}
    try:
        from core.db_clients import qdrant_client
        collections = qdrant_client.get_collections()
        services["qdrant"] = {"status": "ok", "collections": len(collections.collections)}
    except Exception as e:
        services["qdrant"] = {"status": "error", "error": str(e)}

    try:
        from core.db_clients import neo4j_driver
        with neo4j_driver.session() as s:
            s.run("RETURN 1")
        services["neo4j"] = {"status": "ok"}
    except Exception as e:
        services["neo4j"] = {"status": "error", "error": str(e)}

    try:
        from core.db_clients import pg_conn, pg_pool_stats
        with pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
        services["postgres"] = {"status": "ok", "pool": pg_pool_stats()}
    except Exception as e:
        services["postgres"] = {"status": "error", "error": str(e)}

    overall = "ok" if all(s["status"] == "ok" for s in services.values()) else "degraded"
    return {"status": overall, "services": services, "version": "2.0.1"}

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'index.html')
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Inject API key into the dashboard so it can authenticate
    api_key_script = f"const API_KEY = '{VALID_API_KEY}';\nconsole.log('API_KEY injected');"
    html = html.replace("const API = 'http://localhost:8001';", f"const API = 'http://localhost:8001';\n{api_key_script}")
    return HTMLResponse(content=html)



@app.post("/pipeline/preview-routing")
def preview_routing(req: dict = Body(...), api_key: str = Depends(verify_api_key)):
    """
    Run Phase 0 search ONLY, return candidates. Don't commit anything.
    The dashboard shows these to the user, who picks one (or 'new', or 'manual').
    """
    requirement = req.get("requirement", "").strip()
    if not requirement:
        raise HTTPException(400, "requirement is required")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.phase0_selector.selector_agent import search_projects, slugify
    from knowledge_layer.project_registry import list_all_projects

    # Top-5 semantic matches — give the user a real menu
    candidates = search_projects(requirement, top_k=5)

    # Also list ALL projects (so user can pick something semantic search didn't surface)
    all_projects = list_all_projects()

    return {
        "requirement": requirement,
        "suggested_match": candidates[0] if candidates else None,
        "candidates": candidates,                  # ranked by similarity
        "all_projects": all_projects,              # full menu
        "new_project_preview": {
            "slug": slugify(requirement),
            "backend_repo": f"{slugify(requirement)}-backend",
            "frontend_repo": f"{slugify(requirement)}-frontend",
        },
    }


# Update existing POST /pipeline/start to accept an explicit routing choice:




@app.post("/pipeline/start")
def pipeline_start(req: PipelineStartRequest, background_tasks: BackgroundTasks,
                   api_key: str = Depends(verify_api_key),
                   user_id: str = Depends(get_user_id)):
    thread_id = f"pipeline-{uuid.uuid4().hex[:8]}"
    pipeline_store[thread_id] = {
        "thread_id": thread_id,
        "user_id": user_id,                # ← NEW
        "requirement": req.requirement,
        "routing_choice": req.routing_choice,
        "phase": 0,
        "status": "STARTED",
        "current_state": {},
    }

    save_pipeline(thread_id, pipeline_store[thread_id], {})
    background_tasks.add_task(run_phase0_and_phase1, thread_id, req.requirement)
    return {"thread_id": thread_id, "status": "STARTED"}

@app.get("/pipeline/status/{thread_id}")
def pipeline_status(thread_id: str,
                    request: Request,
                    response: Response,
                    api_key: str = Depends(verify_api_key),
                    user_id: str = Depends(get_user_id)):
    # Ownership check
    entry = pipeline_store.get(thread_id)
    if entry and entry.get("user_id") and entry["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not your pipeline")


    entry = pipeline_store[thread_id]
    safe = _safe_state(entry.get("current_state", {}))
    payload = {
        "thread_id": thread_id,
        "phase": entry.get("phase", ""),
        "status": entry.get("status", ""),
        "sub_stage": entry.get("sub_stage", ""),
        "requirement": entry.get("requirement", ""),
        "pr_urls": entry.get("pr_urls", []),
        "error": entry.get("error"),
        "is_new_project": safe.get("is_new_project", False),
        "selected_project": safe.get("selected_project", {}),
        "selected_repos": safe.get("selected_repos", []),
        "target_repo": safe.get("target_repo", ""),
        "current_state": safe
    }

    # ── ETag handling ────────────────────────────────────────────────────
    # This is the hottest polling endpoint — fires every few seconds per
    # open dashboard. Returning 304 when state hasn't changed cuts payload
    # bytes to ~0 and avoids the JSON.parse on the browser side.
    #
    # The hash is computed from the full payload (sort_keys for determinism
    # across uvicorn workers / Python dict ordering). Any field changing
    # invalidates the ETag, so the browser sees a fresh 200 with new data.
    body = json.dumps(payload, sort_keys=True, default=str)
    etag = '"' + hashlib.md5(body.encode("utf-8")).hexdigest() + '"'

    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"
    return payload

@app.get("/pipeline/list")
def pipeline_list(request: Request, response: Response, user_id: str = Depends(get_user_id)):
    """
    List pipelines for the requesting user. user_id is the boundary.

    Supports HTTP ETag: the browser sends back the previous ETag in the
    If-None-Match header, and if the data hasn't changed we return 304 with
    an empty body. Saves bandwidth when the dashboard polls and nothing has
    moved since last time.
    """
    from core.db_clients import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
                    SELECT thread_id, requirement, status, sub_stage, phase, updated_at
                    FROM pipelines
                    WHERE user_id = %s
                    ORDER BY updated_at DESC NULLS LAST
                        LIMIT 50
                    """, (user_id,))
        rows = cur.fetchall()
        cur.close()

    payload = [
        {
            "thread_id": r[0],
            "requirement": r[1] or "",
            "status": r[2] or "",
            "sub_stage": r[3] or "",
            "phase": r[4] or "",
            "updated_at": str(r[5]) if r[5] else "",
        }
        for r in rows
    ]

    # ── ETag handling ────────────────────────────────────────────────────
    # Hash the rows (each row already contains updated_at, which changes when
    # anything about a pipeline changes). The hash is deterministic across
    # processes — two uvicorn workers compute the same ETag for identical data.
    body = json.dumps(payload, sort_keys=True, default=str)
    etag = '"' + hashlib.md5(body.encode("utf-8")).hexdigest() + '"'

    # If the browser sends If-None-Match matching our current ETag, the data
    # hasn't changed — return 304 with zero body. Browser reuses its cache.
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})

    response.headers["ETag"] = etag
    # no-cache (not no-store) tells the browser: revalidate every time, but
    # you may keep a cached copy to use after a 304.
    response.headers["Cache-Control"] = "no-cache"
    return payload

@app.post("/pipeline/approve/{thread_id}")
def pipeline_approve(thread_id: str, body: dict = Body(...), background_tasks: BackgroundTasks = None, api_key: str = Depends(verify_api_key)):
    if thread_id not in pipeline_store:
        raise HTTPException(404, "Not found")

    approved = body.get("approved", False)
    feedback = body.get("feedback", "")
    actor = body.get("actor", "user")

    entry = pipeline_store[thread_id]
    current_phase = str(entry.get("phase", "1"))

    # REJECTION LOGIC
    if not approved:
        audit(thread_id, f"phase{current_phase}", "REJECTED", actor, {"feedback": feedback})

        # Shantanu's Surgical Replay applies only to Phase 1
        if current_phase == "1" and SURGICAL_REPLAY_ENABLED:
            target_artifact = extract_target_artifact(body)
            job_id = create_replay_job(thread_id, target_artifact)
            entry["status"] = "REPLAYING"
            entry["sub_stage"] = f"Surgical replay for {target_artifact}"
            save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

            background_tasks.add_task(execute_surgical_replay, job_id, thread_id, target_artifact, feedback)
            return {"status": "REPLAYING", "phase": current_phase, "job_id": job_id, "target_artifact": target_artifact}

        # Fallback Standard Regeneration for other phases
        entry["status"] = "REGENERATING"
        entry["sub_stage"] = f"Regenerating Phase {current_phase} with feedback..."
        entry["human_feedback"] = feedback
        save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

        if current_phase == "1": background_tasks.add_task(run_phase1, thread_id, entry["requirement"], feedback)
        elif current_phase == "2": background_tasks.add_task(run_phase2, thread_id, feedback)
        elif current_phase == "3": background_tasks.add_task(run_phase3, thread_id, feedback)
        elif current_phase == "6": background_tasks.add_task(run_phase6, thread_id, feedback)
        elif current_phase == "7": background_tasks.add_task(run_phase7, thread_id, feedback)

        return {"status": "REGENERATING", "phase": current_phase, "feedback_applied": feedback}

    # APPROVAL LOGIC
    audit(thread_id, f"phase{current_phase}", "APPROVED", actor, {"feedback": feedback})
    entry["status"] = f"PHASE_{current_phase}_APPROVED"
    entry["sub_stage"] = f"Phase {current_phase} approved — proceeding..."
    save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

    next_phase = str(int(current_phase) + 1)
    entry["phase"] = int(next_phase)

    # ── Phase 6 → Phase 7 transition: merge the approved PR(s) first ──
    # Phase 7 then deploys from the exact merged commit SHA, not from
    # HEAD-of-main (which might have drifted). If any merge fails, we
    # refuse to start Phase 7 — the user must fix the PR manually first.
    if next_phase == "7":
        from agents.phase6_delivery.delivery_agent import merge_prs_for_state
        cur_state = entry.get("current_state", {})
        merge_input = {
            "requirement": entry.get("requirement", ""),
            "pr_urls": entry.get("pr_urls", []) or cur_state.get("pr_urls", []),
            "approved": True,
            "status": cur_state.get("phase6_status") or "APPROVED_FOR_DEPLOYMENT",
        }
        try:
            merge_result = merge_prs_for_state(merge_input)
        except Exception as e:
            import traceback; traceback.print_exc()
            entry["status"] = "ERROR"
            entry["error"] = f"PR merge step crashed: {e}"
            save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))
            return {"status": "MERGE_CRASHED", "error": str(e)}

        # Persist merged_shas + merge_errors into the pipeline state so
        # run_phase7 can read them later.
        cur_state["merged_shas"] = merge_result.get("merged_shas", {})
        cur_state["merge_errors"] = merge_result.get("merge_errors", [])
        cur_state["phase6_final_status"] = merge_result.get("status")
        entry["current_state"] = cur_state

        if merge_result.get("status") == "MERGE_FAILED":
            entry["status"] = "MERGE_FAILED"
            entry["sub_stage"] = (
                "PR merge failed — see merge_errors. Fix the PR on GitHub "
                "and use /pipeline/{id}/resume?phase=7 to retry."
            )
            entry["error"] = "; ".join(merge_result.get("merge_errors", []))[:500]
            audit(thread_id, "phase6", "MERGE_FAILED", actor,
                  {"errors": merge_result.get("merge_errors", [])})
            save_pipeline(thread_id, entry, _safe_state(cur_state))
            return {
                "status": "MERGE_FAILED",
                "next_phase": next_phase,
                "merge_errors": merge_result.get("merge_errors", []),
            }

        audit(thread_id, "phase6", "MERGED", actor,
              {"merged_shas": merge_result.get("merged_shas", {})})
        save_pipeline(thread_id, entry, _safe_state(cur_state))

    if next_phase == "2": background_tasks.add_task(run_phase2, thread_id, "")
    elif next_phase == "3":
        # Main's Fix: Skip Phase 3 for new projects
        if entry.get("current_state", {}).get("is_new_project", False):
            print(f"  [Approve] New project — skipping Phase 3, jumping to Phase 4")
            entry["phase"] = 4
            background_tasks.add_task(run_phase4, thread_id)
        else:
            background_tasks.add_task(run_phase3, thread_id, "")
    elif next_phase == "4": background_tasks.add_task(run_phase4, thread_id)
    elif next_phase == "5": background_tasks.add_task(run_phase5, thread_id)
    elif next_phase == "6": background_tasks.add_task(run_phase6, thread_id, "")
    elif next_phase == "7": background_tasks.add_task(run_phase7, thread_id, "")

    return {"status": "APPROVED", "next_phase": next_phase}
@app.post("/pipeline/{thread_id}/resume")
def pipeline_resume(
        thread_id: str,
        body: dict = Body(default={}),
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(verify_api_key),
):
    """
    Resume a stuck/errored pipeline from a specific phase, reusing all saved state
    (BRD/PRD/ADR/Architecture/Sprint/Impact/Code/Tests) from PostgreSQL.

    Body: {"phase": 2}   # which phase to re-run; saved state from earlier phases is kept.
    If "phase" omitted, resumes from the phase recorded in DB.
    """
    if thread_id not in pipeline_store:
        raise HTTPException(404, f"Thread {thread_id} not found")

    entry = pipeline_store[thread_id]
    requested_phase = body.get("phase")
    try:
        phase = int(requested_phase) if requested_phase is not None else int(entry.get("phase", 1) or 1)
    except (TypeError, ValueError):
        raise HTTPException(400, f"Invalid phase: {requested_phase!r}")

    if phase < 1 or phase > 7:
        raise HTTPException(400, "phase must be between 1 and 7")

    # Clear error fields; keep current_state intact so prior artifacts are reused
    entry["status"] = f"PHASE_{phase}_RUNNING"
    entry["sub_stage"] = f"Resuming Phase {phase}..."
    entry["error"] = None
    entry["phase"] = phase
    audit(thread_id, f"phase{phase}", "RESUMED", body.get("actor", "user"),
          {"from_status": entry.get("status", ""), "requested_phase": phase})
    save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

    if   phase == 1: background_tasks.add_task(run_phase1, thread_id, entry["requirement"], "")
    elif phase == 2: background_tasks.add_task(run_phase2, thread_id, "")
    elif phase == 3: background_tasks.add_task(run_phase3, thread_id, "")
    elif phase == 4: background_tasks.add_task(run_phase4, thread_id)
    elif phase == 5: background_tasks.add_task(run_phase5, thread_id)
    elif phase == 6: background_tasks.add_task(run_phase6, thread_id, "")
    elif phase == 7: background_tasks.add_task(run_phase7, thread_id, "")

    return {"thread_id": thread_id, "status": "RESUMED", "resuming_from_phase": phase}

@app.get("/pipeline/list")
def pipeline_list():
    return {
        "pipelines": [{"thread_id": tid, "phase": e.get("phase", ""), "status": e.get("status", "")} for tid, e in pipeline_store.items()],
        "total": len(pipeline_store)
    }

@app.delete("/pipeline/{thread_id}")
def pipeline_delete(thread_id: str):
    if thread_id not in pipeline_store:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    del pipeline_store[thread_id]
    return {"message": f"Pipeline {thread_id} removed"}


# ── DOWNLOAD ENDPOINTS ─────────────────────────────────────────────
def _get_pipeline_or_404(thread_id: str):
    if thread_id not in pipeline_store:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    return pipeline_store[thread_id]

@app.get("/pipeline/{thread_id}/download/brd")
def download_brd(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_brd_markdown(entry["current_state"].get("brd", {}))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=BRD_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/prd")
def download_prd(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_prd_markdown(entry["current_state"].get("prd", {}))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=PRD_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/adr")
def download_adr(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_adr_markdown(entry["current_state"].get("adr", {}))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=ADR_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/architecture")
def download_architecture(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_architecture_markdown(entry["current_state"].get("architecture", {}))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=Architecture_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/sprint-plan")
def download_sprint_plan(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_sprint_plan_markdown(entry["current_state"].get("sprint_plan", {}), entry["current_state"].get("jira_tickets", []))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=SprintPlan_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/test-cases")
def download_test_cases(thread_id: str):
    from api.test_cases_export import export_test_cases_excel
    entry = _get_pipeline_or_404(thread_id)
    xlsx_bytes = export_test_cases_excel(entry)
    return Response(content=xlsx_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=TestCases_{thread_id}.xlsx"})

@app.get("/pipeline/{thread_id}/download/impact")
def download_impact(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_impact_markdown(entry["current_state"].get("impact_report", {}))
    return Response(content=md, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=ImpactReport_{thread_id}.md"})

@app.get("/pipeline/{thread_id}/download/runbook")
def download_runbook(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    xlsx_bytes = export_runbook_excel(entry)
    return Response(content=xlsx_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=Runbook_{thread_id}.xlsx"})

@app.get("/pipeline/{thread_id}/download/all")
def download_all(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    state = entry["current_state"]

    def _safe_write(z, name, producer):
        """Run an exporter; on failure, log it and write a placeholder so the zip still builds."""
        try:
            content = producer()
            if content is None:
                content = f"# {name}\n\n_Not generated yet (phase has not run)._"
            z.writestr(name, content)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Download] '{name}' skipped: {e}")
            placeholder = f"# {name}\n\n_Could not be exported — {type(e).__name__}: {e}_"
            try:
                z.writestr(name, placeholder)
            except Exception:
                pass

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        _safe_write(z, "01_BRD.md",          lambda: export_brd_markdown(state.get("brd", {})))
        _safe_write(z, "02_PRD.md",          lambda: export_prd_markdown(state.get("prd", {})))
        _safe_write(z, "03_ADR.md",          lambda: export_adr_markdown(state.get("adr", {})))
        _safe_write(z, "04_Architecture.md", lambda: export_architecture_markdown(state.get("architecture", {})))

        # Phase 2+ artifacts — may legitimately not exist yet
        if state.get("sprint_plan") or state.get("jira_tickets"):
            _safe_write(z, "05_SprintPlan.md",
                        lambda: export_sprint_plan_markdown(state.get("sprint_plan", {}),
                                                            state.get("jira_tickets", [])))
        if state.get("impact_report"):
            _safe_write(z, "06_ImpactReport.md",
                        lambda: export_impact_markdown(state.get("impact_report", {})))
        if state.get("sprint_plan") or state.get("runbook") or state.get("jira_tickets"):
            _safe_write(z, "07_Runbook.xlsx", lambda: export_runbook_excel(entry))

        # Phase 4 — generated code
        for change in state.get("generated_changes", []) or []:
            fname = (change.get("file_path") or "unknown.txt").replace("/", "_").replace("\\", "_")
            _safe_write(z, f"08_code/{fname}", lambda c=change: c.get("content", ""))

        # Phase 5 — tests
        for test in state.get("test_files", []) or []:
            fname = (test.get("test_file_path") or "test.py").replace("/", "_").replace("\\", "_")
            _safe_write(z, f"09_tests/{fname}", lambda t=test: t.get("content", ""))

        # Test cases excel — only if Jira tickets exist
        if state.get("jira_tickets"):
            from api.test_cases_export import export_test_cases_excel
            _safe_write(z, "10_TestCases.xlsx", lambda: export_test_cases_excel(entry))

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=Pipeline_{thread_id}_FullPackage.zip"},
    )

# ── Knowledge Layer Endpoints ─────────────────────────────────────────────────
@app.post("/knowledge/index")
def knowledge_index(req: IndexRequest, background_tasks: BackgroundTasks):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    background_tasks.add_task(run_indexer, req.repo_path, req.repo_name)
    return {"status": "INDEXING_STARTED", "repo_name": req.repo_name, "repo_path": req.repo_path}

@app.post("/knowledge/search")
def knowledge_search(req: SearchRequest):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.phase3_impact.impact_analyzer import semantic_search
    hits = semantic_search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": hits, "total": len(hits)}

@app.get("/knowledge/repos")
def knowledge_repos():
    from core.db_clients import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT repo_name, language, file_count, last_indexed FROM repo_maps ORDER BY last_indexed DESC")
        repos = [{"repo_name": r[0], "language": r[1], "file_count": r[2], "last_indexed": str(r[3])} for r in cur.fetchall()]
        cur.close()
    return {"repos": repos, "total": len(repos)}

# ─────────────────────────────────────────────────────────────────────
# GitHub-driven project discovery + sync + indexer queue
# ─────────────────────────────────────────────────────────────────────
from pydantic import BaseModel


class DiscoverRequest(BaseModel):
    owner: Optional[str] = None
    min_group_size: int = 2


class SyncRequest(BaseModel):
    owner: Optional[str] = None
    min_group_size: int = 2
    dry_run: bool = False
    project_ids: Optional[list] = None   # if set, only sync these; otherwise all discovered
    force_reindex: bool = False   # NEW — defaults to False

@app.post("/knowledge/projects/discover")
def projects_discover(req: DiscoverRequest, api_key: str = Depends(verify_api_key)):
    """
    Preview what GitHub-driven discovery would register.
    Returns the project groupings WITHOUT writing anything.
    Use this to review before calling /sync.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.github_discovery import discover_projects
    try:
        preview = discover_projects(owner=req.owner, min_group_size=req.min_group_size)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Discovery failed: {e}")
    return preview


@app.post("/knowledge/projects/sync")
def projects_sync(req: SyncRequest, api_key: str = Depends(verify_api_key)):
    """
    Discover GitHub repos, group by prefix, register each project, and queue
    each repo for cloning + indexing. Returns immediately with a job manifest.

    Use dry_run=true to see exactly what would happen without writing.
    Use project_ids=["billing", "conduit"] to limit the sync to specific groups.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.github_discovery import discover_projects
    from knowledge_layer.project_registry import register_project
    from agents.indexer_queue import get_queue

    try:
        preview = discover_projects(owner=req.owner, min_group_size=req.min_group_size)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Discovery failed: {e}")

    projects = preview["projects"]
    if req.project_ids:
        filter_set = set(req.project_ids)
        projects = [p for p in projects if p["project_id"] in filter_set]

    if req.dry_run:
        return {
            "dry_run": True,
            "would_register": len(projects),
            "would_index_repos": sum(len(p["repos"]) for p in projects),
            "projects": projects,
        }

    queue = get_queue()
    registered = []
    queued_jobs = []
    errors = []

    for proj in projects:
        try:
            repo_names = [r["name"] for r in proj["repos"]]
            register_project(
                project_id=proj["project_id"],
                project_name=proj["project_name"],
                description=proj["description"],
                domain=proj["domain"],
                tech_stack=proj["tech_stack"],
                repos=repo_names,
                owner_team=proj["owner_team"],
            )
            registered.append(proj["project_id"])

            for repo in proj["repos"]:
                job_id = queue.enqueue(
                    repo_name=repo["name"],
                    repo_url=repo["url"],
                    branch=repo.get("branch", "main"),
                    force=req.force_reindex,     # NEW
                )
                queued_jobs.append({"repo": repo["name"], "job_id": job_id})
        except Exception as e:
            errors.append({"project_id": proj["project_id"], "error": str(e)})

    return {
        "status": "SYNC_STARTED",
        "registered_projects": registered,
        "queued_indexing_jobs": queued_jobs,
        "total_queued": len(queued_jobs),
        "errors": errors,
        "tip": "Poll GET /knowledge/jobs to track indexing progress.",
    }


@app.get("/knowledge/jobs")
def indexer_jobs_list(status: Optional[str] = None):
    """List indexer jobs. Filter with ?status=RUNNING|SUCCESS|FAILED|QUEUED|RETRYING."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.indexer_queue import get_queue
    q = get_queue()
    return {"summary": q.summary(), "jobs": q.list_jobs(status=status)}


@app.get("/knowledge/jobs/{job_id}")
def indexer_job_detail(job_id: str):
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from agents.indexer_queue import get_queue
    q = get_queue()
    job = q.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


# ─────────────────────────────────────────────────────────────────────
# GitHub webhook (stub — NOT active until you configure GitHub to call it)
# ─────────────────────────────────────────────────────────────────────
import hmac, hashlib
from fastapi import Request

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def _verify_github_signature(payload_body: bytes, signature_header: str) -> bool:
    """Verify the GitHub webhook signature. Required for production webhook use."""
    if not GITHUB_WEBHOOK_SECRET:
        return False
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@app.post("/webhooks/github")
async def github_webhook(request: Request):
    """
    GitHub webhook handler — STUB. Wire up later by:
      1. Setting GITHUB_WEBHOOK_SECRET in .env
      2. Configuring your GitHub org webhook to POST here on:
         repository.created, repository.deleted, push, repository.archived
      3. Exposing this endpoint publicly (ngrok / load balancer)
    """
    if not GITHUB_WEBHOOK_SECRET:
        raise HTTPException(503, "Webhooks not configured — set GITHUB_WEBHOOK_SECRET in .env")

    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_github_signature(body, sig):
        raise HTTPException(401, "Invalid webhook signature")

    event = request.headers.get("X-GitHub-Event", "")
    payload = await request.json()
    action = payload.get("action", "")
    repo = payload.get("repository", {}) or {}
    repo_name = repo.get("name", "")
    repo_url = repo.get("clone_url", "")
    branch = repo.get("default_branch", "main")

    # Minimal handling — wire to your queue when you're ready
    if event == "repository" and action == "created" and repo_name:
        from agents.indexer_queue import get_queue
        job_id = get_queue().enqueue(repo_name=repo_name, repo_url=repo_url, branch=branch)
        return {"received": True, "action": "indexed", "job_id": job_id}

    if event == "push" and repo_name:
        # Only re-index if the push was to the repo's default branch
        ref = payload.get("ref", "")  # e.g. "refs/heads/main"
        default_branch = repo.get("default_branch", "main")
        if ref != f"refs/heads/{default_branch}":
            return {"received": True, "action": "ignored",
                    "reason": f"push to non-default branch ({ref})"}

        from agents.indexer_queue import get_queue
        job_id = get_queue().enqueue(repo_name=repo_name, repo_url=repo_url, branch=default_branch)
        return {"received": True, "action": "reindexed", "job_id": job_id}

class RepoOverrideRequest(BaseModel):
    action: str  # "include" or "exclude"
    reason: str = ""


@app.post("/knowledge/repos/{repo_name}/override")
def set_repo_override(
        repo_name: str,
        req: RepoOverrideRequest,
        api_key: str = Depends(verify_api_key),
):
    """Manually mark a repo as include or exclude. Persists across syncs."""
    if req.action not in ("include", "exclude"):
        raise HTTPException(400, "action must be 'include' or 'exclude'")
    from core.db_clients import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS repo_overrides (
                                                                  repo_name VARCHAR(255) PRIMARY KEY,
                        action VARCHAR(20) NOT NULL,
                        reason TEXT,
                        set_by VARCHAR(255),
                        set_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
        cur.execute("""
                    INSERT INTO repo_overrides (repo_name, action, reason, set_by)
                    VALUES (%s, %s, %s, %s)
                        ON CONFLICT (repo_name) DO UPDATE
                                                       SET action = EXCLUDED.action,
                                                       reason = EXCLUDED.reason,
                                                       set_by = EXCLUDED.set_by,
                                                       set_at = NOW()
                    """, (repo_name, req.action, req.reason, "api"))
        cur.close()
    return {"repo_name": repo_name, "action": req.action, "reason": req.reason}


@app.delete("/knowledge/repos/{repo_name}/override")
def clear_repo_override(repo_name: str, api_key: str = Depends(verify_api_key)):
    """Remove the manual override so default heuristics apply again."""
    from core.db_clients import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM repo_overrides WHERE repo_name = %s", (repo_name,))
        deleted = cur.rowcount
        cur.close()
    return {"repo_name": repo_name, "cleared": deleted > 0}


@app.get("/knowledge/repos/overrides")
def list_repo_overrides(api_key: str = Depends(verify_api_key)):
    """List every manual override currently active."""
    from core.db_clients import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS repo_overrides (
                                                                  repo_name VARCHAR(255) PRIMARY KEY,
                        action VARCHAR(20) NOT NULL,
                        reason TEXT,
                        set_by VARCHAR(255),
                        set_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
        cur.execute("SELECT repo_name, action, reason, set_by, set_at FROM repo_overrides ORDER BY set_at DESC")
        rows = cur.fetchall()
        cur.close()
    return {
        "overrides": [
            {"repo_name": r[0], "action": r[1], "reason": r[2],
             "set_by": r[3], "set_at": str(r[4])}
            for r in rows
        ]
    }

# ── Background Pipeline Runners ───────────────────────────────────────────────

def run_phase0_and_phase1(thread_id: str, requirement: str):
    """Phase 0 (project/repo routing) + Phase 1 (Discovery) runner."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase0_selector.selector_agent import search_projects, slugify

        # ── Phase 0: route to existing project or create new ────────────
        pipeline_store[thread_id]["phase"] = 0
        pipeline_store[thread_id]["status"] = "PHASE_0_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Routing requirement..."
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))
        audit(thread_id, "phase0", "PHASE_STARTED")

        entry = pipeline_store[thread_id]
        choice = entry.get("routing_choice")
        github_owner = os.getenv("GITHUB_REPO_OWNER", "AkashW45")

        # ── Path A: User made an explicit choice — honor it ─────────────
        if choice:
            mode = choice.get("mode")

            if mode == "existing":
                project_id = choice["project_id"]
                # Look up real repos for this project from the registry
                from knowledge_layer.project_registry import list_all_projects
                all_proj = {p["project_id"]: p for p in list_all_projects()}
                proj = all_proj.get(project_id, {})

                selected = {
                    "id": project_id,
                    "name": proj.get("project_name", project_id),
                    "description": proj.get("description", ""),
                }
                repo_names = choice.get("repo_names") or proj.get("repos", [])
                selected_repos = [
                    {"name": n, "type": "backend", "exists": True,
                     "url": f"https://github.com/{github_owner}/{n}.git"}
                    for n in repo_names
                ]
                is_new = False
                print(f"[Phase 0] ✅ User chose existing project: {selected['name']} with repos {repo_names}")

            elif mode == "manual":
                repo_names = choice.get("repo_names", [])
                selected = {"id": "manual", "name": "User-selected repos",
                            "description": "Manual repo selection"}
                selected_repos = [
                    {"name": n, "type": "backend", "exists": True,
                     "url": f"https://github.com/{github_owner}/{n}.git"}
                    for n in repo_names
                ]
                is_new = False
                print(f"[Phase 0] ✅ User picked repos manually: {repo_names}")

            elif mode == "new":
                slug = slugify(requirement)
                selected = {"id": f"new-{slug}", "name": requirement[:60],
                            "description": requirement}
                selected_repos = [
                    {"name": f"{slug}-backend", "type": "backend",
                     "url": f"https://github.com/{github_owner}/{slug}-backend.git",
                     "exists": False},
                    {"name": f"{slug}-frontend", "type": "frontend",
                     "url": f"https://github.com/{github_owner}/{slug}-frontend.git",
                     "exists": False},
                ]
                is_new = True
                print(f"[Phase 0] ✅ User chose NEW project: {slug}")

            else:
                raise ValueError(f"Unknown routing mode: {mode}")

        # ── Path B: No explicit choice — fall back to auto-matching ─────
        else:
            candidates = search_projects(requirement, top_k=3)
            THRESHOLD = float(os.getenv("PHASE0_MATCH_THRESHOLD", "0.55"))

            if candidates and candidates[0]["score"] >= THRESHOLD:
                top = candidates[0]
                selected = top
                repo_names = top.get("repos", [])
                selected_repos = [
                    {"name": n if isinstance(n, str) else n.get("name"),
                     "type": "backend", "exists": True,
                     "url": f"https://github.com/{github_owner}/{n if isinstance(n, str) else n.get('name')}.git"}
                    for n in repo_names
                ]
                is_new = False
                print(f"[Phase 0] ✅ Auto-matched: {selected.get('name')} (score={top['score']})")

                # Fire-and-forget drift check
                matched_repo = selected_repos[0]["name"] if selected_repos else None
                if matched_repo:
                    threading.Thread(target=_warn_if_stale_index,
                                     args=(thread_id, matched_repo),
                                     daemon=True).start()
            else:
                slug = slugify(requirement)
                selected = {"id": f"new-{slug}", "name": requirement[:60],
                            "description": requirement}
                selected_repos = [
                    {"name": f"{slug}-backend", "type": "backend",
                     "url": f"https://github.com/{github_owner}/{slug}-backend.git",
                     "exists": False},
                    {"name": f"{slug}-frontend", "type": "frontend",
                     "url": f"https://github.com/{github_owner}/{slug}-frontend.git",
                     "exists": False},
                ]
                is_new = True
                print(f"[Phase 0] 🆕 No match (best score < {THRESHOLD}) — creating NEW project: {slug}")

        # ── Persist Phase 0 result ──────────────────────────────────────
        pipeline_store[thread_id]["selected_project"] = selected
        pipeline_store[thread_id]["selected_repos"] = selected_repos
        pipeline_store[thread_id]["is_new_project"] = is_new
        pipeline_store[thread_id]["current_state"] = {
            **(pipeline_store[thread_id].get("current_state") or {}),
            "selected_project": selected,
            "selected_repos": selected_repos,
            "is_new_project": is_new,
        }
        audit(thread_id, "phase0", "PHASE_COMPLETED",
              {"project": selected.get("name"), "is_new": is_new,
               "repos": [r["name"] for r in selected_repos]})
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id]["current_state"]))

        # ── Phase 1: Discovery (BRD/PRD/ADR/Architecture) ───────────────
        run_phase1(thread_id, requirement, "")

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))


def run_phase1(thread_id: str, requirement: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase1_discovery.discovery_agent import build_discovery_graph, DiscoveryState

        pipeline_store[thread_id].update({"phase": 1, "status": "PHASE_1_RUNNING", "sub_stage": "Generating BRD..."})

        graph = build_discovery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p1"}}
        # Pull routing context from pipeline entry — works whether Phase 0 set
        # it on current_state, on the entry directly, or via routing_choice
        entry = pipeline_store[thread_id]
        prev_state = entry.get("current_state", {})
        routing_choice = entry.get("routing_choice", {}) or {}

        # Resolve selected_repos from THREE possible sources, in priority order
        selected_repos = (
                prev_state.get("selected_repos")                   # Phase 0 stored it on state
                or entry.get("selected_repos")                     # Phase 0 stored it on entry
                or _routing_choice_to_repos(routing_choice)        # Direct fallback from user choice
                or []
        )

        # Resolve is_new_project — explicit False if user picked existing/manual, True if 'new'
        mode = routing_choice.get("mode", "").lower() if routing_choice else ""
        if mode == "new":
            is_new_project = True
        elif mode in ("existing", "manual"):
            is_new_project = False
        else:
            is_new_project = prev_state.get("is_new_project",
                                            entry.get("is_new_project",
                                                      not bool(selected_repos)))

        print(f"[Phase 1] selected_repos={[r.get('name') if isinstance(r,dict) else r for r in selected_repos]}, is_new={is_new_project}")

        initial_state = DiscoveryState(
            requirement=requirement,
            scope_contract={},
            classifier_output={},
            brd={},
            prd={},
            adr={},
            architecture={},
            human_feedback=feedback,
            approved=False,
            status="STARTED",
            selected_repos=selected_repos,
            is_new_project=is_new_project,
        )

        for chunk in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict): continue
                current = pipeline_store[thread_id].get("current_state", {})
                for k, v in node_state.items():
                    if k in ("brd", "prd", "adr", "architecture", "status",
                             "scope_contract", "classifier_output"): current[k] = v

                pipeline_store[thread_id]["current_state"] = current

                if node_name == "generate_brd": pipeline_store[thread_id]["sub_stage"] = "BRD Done — Generating PRD..."
                elif node_name == "generate_prd": pipeline_store[thread_id]["sub_stage"] = "PRD Done — Generating ADR..."
                elif node_name == "generate_adr": pipeline_store[thread_id]["sub_stage"] = "ADR Done — Generating Architecture..."
                elif node_name == "generate_architecture": pipeline_store[thread_id]["sub_stage"] = "Architecture Done — Awaiting Approval"
                save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(current))

        pipeline_store[thread_id].update({"status": "WAITING_PHASE_1_APPROVAL", "graph": graph, "config": config})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id]["current_state"]))

    except Exception as e:
        # GraphInterrupt is how LangGraph signals a pause for human approval — it is NOT an error.
        from langgraph.errors import GraphInterrupt
        if isinstance(e, GraphInterrupt):
            pipeline_store[thread_id].update({
                "status": "WAITING_PHASE_1_APPROVAL",
                "sub_stage": "Discovery complete — Awaiting Review",
            })
            save_pipeline(
                thread_id,
                pipeline_store[thread_id],
                _safe_state(pipeline_store[thread_id].get("current_state", {})),
            )
            return
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )

def run_phase2(thread_id: str, feedback: str = ""):
    try:
        from agents.phase2_planning.planning_agent import build_planning_graph, PlanningState
        entry = pipeline_store[thread_id]
        p1_state = entry["current_state"]

        pipeline_store[thread_id].update({"status": "PHASE_2_RUNNING", "sub_stage": "Generating sprint plan...", "phase": 2})

        graph = build_planning_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p2"}}
        initial = PlanningState(requirement=entry["requirement"], brd=p1_state.get("brd", {}), prd=p1_state.get("prd", {}), sprint_plan={}, runbook={}, jira_tickets=[], human_feedback=feedback, approved=False, status="STARTED")

        result = graph.invoke(initial, config)
        merged = {**p1_state}
        for k in ("sprint_plan", "runbook", "jira_tickets"):
            if k in result: merged[k] = result[k]

        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "status": "WAITING_PHASE_2_APPROVAL", "sub_stage": "Sprint plan ready — Awaiting approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
    except Exception as e:
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )

def run_phase3(thread_id: str, feedback: str = ""):
    try:
        from agents.phase3_impact.graph import get_graph
        entry = pipeline_store[thread_id]
        prev_state = entry["current_state"]

        pipeline_store[thread_id].update({
            "status": "PHASE_3_RUNNING",
            "sub_stage": "Analyzing code impact...",
            "phase": 3
        })

        graph = get_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p3"}}
        initial = {
            "requirement": entry["requirement"],
            "prd": prev_state.get("prd", {}),                   # NEW
            "selected_repos": prev_state.get("selected_repos", []),   # NEW
            "scope_contract": prev_state.get("scope_contract", {}),   # NEW
            "impact_report": {},
            "human_approved": False,
            "human_feedback": feedback,
            "status": "STARTED"
        }
        result = graph.invoke(initial, config)

        merged = {**prev_state}
        if "impact_report" in result:
            merged["impact_report"] = result["impact_report"]

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": merged,
            "status": "WAITING_PHASE_3_APPROVAL",
            "sub_stage": "Impact report ready — Awaiting approval"
        })
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))

def run_phase4(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase4_codegen.codegen_agent import run_codegen
        from agents.phase5_validation.validation_agent import run_validation_phase
        from agents.expansion_engine import decide_single
        from agents.units_model import attach_units

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        def update_substage(msg, status=None):
            pipeline_store[thread_id]["sub_stage"] = msg
            if status: pipeline_store[thread_id]["status"] = status
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))

        update_substage("Preparing context for codegen...", "PHASE_4_RUNNING")
        pipeline_store[thread_id]["phase"] = 4

        impact = state.get("impact_report", {})
        if state.get("is_new_project", False) and not impact.get("affected_files"):
            impact = {"requirement": entry["requirement"], "affected_files": [], "affected_repos": [r["name"] for r in state.get("selected_repos", [])], "architecture": state.get("architecture", {}), "risk_assessment": {"risk_level": "low", "breaking_changes": [], "recommendation": "proceed"}}

        result4 = run_codegen(requirement=entry["requirement"], impact_report=impact, workspace_path="/repos", thread_id=f"{thread_id}-p4", adr=state.get("adr", {}), scope_contract=state.get("scope_contract", {}))

        # Accept multiple success statuses from codegen
        SUCCESS_STATUSES = {
            "VALIDATED",
            "CODE_GENERATED",
            "CODE_GENERATED_WITH_WARNINGS",
            "PARTIAL_SUCCESS_BELOW_THRESHOLD",  # still has partial output to use
        }

        if result4["status"] not in SUCCESS_STATUSES:
            pipeline_store[thread_id].update({
                "status": "ERROR",
                "error": f"Phase 4 failed: status={result4['status']}, errors={result4.get('errors') or result4.get('validation_errors', [])}"
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
            return

        # Save the generated changes immediately so dashboard sees them
        generated_changes = result4.get("generated_changes", [])
        if not generated_changes:
            pipeline_store[thread_id].update({
                "status": "ERROR",
                "error": "Phase 4 returned no generated_changes despite success status"
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
            return

        # Persist code BEFORE moving on
        pipeline_store[thread_id]["current_state"]["generated_changes"] = generated_changes
        if result4.get("warning"):
            pipeline_store[thread_id]["warning"] = result4["warning"]
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id]["current_state"]))

        print(f"[Phase 4] ✅ Saved {len(generated_changes)} generated files to DB")


        # Route directly to Phase 5
        run_phase5(thread_id)




    except Exception as e:
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )
def run_phase5(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase5_validation.validation_agent import run_validation_phase
        from agents.expansion_engine import decide_single
        from agents.units_model import attach_units

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id]["phase"] = 5
        pipeline_store[thread_id]["sub_stage"] = "Generating pytest test files..."
        pipeline_store[thread_id]["status"] = "PHASE_5_RUNNING"
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))

        result5 = run_validation_phase(
            requirement=entry["requirement"],
            generated_changes=state.get("generated_changes", []),
            scope_contract=state.get("scope_contract", {}),
            workspace_path="/repos", # <--- CLAUDE'S BUG FIX RE-ADDED HERE
            thread_id=f"{thread_id}-p5"
        )

        merged = {**state, "test_files": result5.get("test_files", [])}
        pipeline_store[thread_id]["current_state"] = merged

        # ── EXPANSION DECISION ENGINE GUARD (The Missing Brain!) ──
        pipeline_store[thread_id]["sub_stage"] = "Calculating Work-Unit Budget (Expansion Engine)..."
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

        code_artifact = {
            "artifact_id": f"code-{thread_id}",
            "artifact_type": "CODE",
            "category": "mvp",
            "content": {"files": state.get("generated_changes", [])},
            "confidence": 95
        }

        # Calculate how many "units" this code costs
        attach_units(code_artifact, state.get("scope_contract", {}))

        # Ask the engine if we are allowed to push this to GitHub
        decision = decide_single(
            thread_id=thread_id,
            asp=state.get("scope_contract", {}),
            artifact=code_artifact,
            accepted_units_so_far=0
        )

        if decision["verdict"] == "reject":
            pipeline_store[thread_id].update({
                "status": "ERROR",
                "error": f"Expansion Engine Rejected: {decision.get('reason')}",
                "sub_stage": "Code Rejected by Engine"
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
            return

        elif decision["verdict"] == "queue_for_approval":
            pipeline_store[thread_id].update({
                "status": "WAITING_PHASE_4_APPROVAL",
                "sub_stage": f"Budget Exceeded. Reason: {decision.get('reason')}. Awaiting manual approval."
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
            return

        pipeline_store[thread_id]["sub_stage"] = f"Budget check passed! Cost: {decision.get('recomputed_units')} units. Moving to Delivery."
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
        # ─────────────────────────────────────────────────────────

        run_phase6(thread_id, "")
    except Exception as e:
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )


def run_phase6(thread_id: str, feedback: str = ""):
    try:
        from agents.phase6_delivery.delivery_agent import build_delivery_graph, DeliveryState
        import requests as req_lib

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id].update({"status": "PHASE_6_RUNNING", "sub_stage": "Determining target repository...", "phase": 6})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))

        is_new_project = state.get("is_new_project", False)
        selected_repos = state.get("selected_repos", [])

        if not selected_repos:
            pipeline_store[thread_id].update({"status": "ERROR", "error": "No selected_repos. Phase 0 failed."})
            return

        target_repo = next((r for r in selected_repos if r.get("type") == "backend"), selected_repos[0])
        target_repo_name = target_repo["name"]
        target_repo_url = target_repo.get("url", f"https://github.com/{os.getenv('GITHUB_REPO_OWNER','AkashW45')}/{target_repo_name}.git")

        github_token = os.getenv("GITHUB_TOKEN")
        github_owner = os.getenv("GITHUB_REPO_OWNER", "AkashW45")
        gh_headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}

        if is_new_project or not target_repo.get("exists", True):
            check_resp = req_lib.get(f"https://api.github.com/repos/{github_owner}/{target_repo_name}", headers=gh_headers)
            if check_resp.status_code == 404:
                create_resp = req_lib.post(
                    "https://api.github.com/user/repos",
                    headers=gh_headers,
                    json={"name": target_repo_name, "private": False, "auto_init": True},
                )
                # Do NOT ignore the result — a 401/403/422 here is the #1 reason
                # a greenfield repo never appears. Surface it as a hard error.
                if create_resp.status_code not in (201, 422):
                    err = f"GitHub repo creation failed: HTTP {create_resp.status_code} - {create_resp.text[:300]}"
                    print(f"  ❌ {err}")
                    pipeline_store[thread_id].update({"status": "ERROR", "error": err})
                    save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
                    return
                # auto_init's initial commit is async; wait for it so the
                # delivery agent's clone doesn't race an empty repo.
                commits_url = f"https://api.github.com/repos/{github_owner}/{target_repo_name}/commits"
                for _ in range(10):
                    c = req_lib.get(commits_url, headers=gh_headers)
                    if c.status_code == 200 and c.json():
                        break
                    time.sleep(1)
            elif check_resp.status_code not in (200, 301):
                err = f"Cannot access repo {github_owner}/{target_repo_name}: HTTP {check_resp.status_code} - {check_resp.text[:200]}"
                print(f"  ❌ {err}")
                pipeline_store[thread_id].update({"status": "ERROR", "error": err})
                save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
                return

        branch_name = "main" if is_new_project else f"feature/{thread_id}"

        graph = build_delivery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p6"}}
        initial = DeliveryState(requirement=entry["requirement"], generated_changes=state.get("generated_changes", []), test_files=state.get("test_files", []), branch_name=branch_name, repo_url=target_repo_url, pr_urls=[], human_feedback=feedback, approved=False, status="STARTED")

        result = graph.invoke(initial, config)
        delivery_status = result.get("status", "")
        pr_urls = result.get("pr_urls", [])

        # If the push/PR step failed, do NOT report "Awaiting PR approval" — that
        # false-success was masking empty/missing repos. Stop and report the error.
        if delivery_status in ("PUSH_FAILED", "PR_SKIPPED", "PR_FAILED"):
            err = result.get("error") or f"Phase 6 delivery failed with status {delivery_status}"
            print(f"  ❌ Phase 6 push/PR failed: {err}")
            merged = {**state, "pr_urls": pr_urls, "target_repo": target_repo_name}
            pipeline_store[thread_id].update({"current_state": merged, "status": "ERROR", "error": err, "sub_stage": f"Delivery failed: {delivery_status}"})
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
            return

        merged = {**state, "pr_urls": pr_urls, "target_repo": target_repo_name}
        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "pr_urls": pr_urls, "status": "WAITING_PHASE_6_APPROVAL", "sub_stage": f"Pushed to {target_repo_name} — Awaiting PR approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
        # ── FIX 5: RE-INDEX BROWNFIELD REPOS (BACKGROUND THREAD) ──
        if not state.get("is_new_project", False):
            import threading
            def _bg_index(r_name):
                # Calculate the path to your local 'repos' folder
                repo_dir = os.path.join(os.getcwd(), "repos", r_name)
                if os.path.isdir(repo_dir):
                    print(f"  [Phase 6] Re-indexing {r_name} in background...")
                    run_indexer(repo_dir, r_name)

            for repo in state.get("selected_repos", []):
                rname = repo.get("name") if isinstance(repo, dict) else repo
                if rname:
                    threading.Thread(target=_bg_index, args=(rname,)).start()
        # ──────────────────────────────────────────────────────────
    except Exception as e:
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )


def run_phase7(thread_id: str, feedback: str = ""):
    try:
        from agents.phase7_deployment.deployment_agent import build_deployment_graph, DeploymentState
        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        # Refuse to deploy if Phase 6's merge step failed. The PR is still
        # open on GitHub; the user must resolve conflicts / fix branch
        # protection there, then re-trigger Phase 7 via /pipeline/{id}/resume.
        phase6_final = state.get("phase6_final_status")
        if phase6_final == "MERGE_FAILED":
            pipeline_store[thread_id].update({
                "status": "ERROR",
                "error": (
                        "Cannot run Phase 7: PR merge failed in Phase 6. "
                        "Errors: " + "; ".join(state.get("merge_errors", []))[:400]
                ),
                "sub_stage": "Phase 7 blocked — PR not merged",
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
            return

        pipeline_store[thread_id].update({"status": "PHASE_7_RUNNING", "sub_stage": "Resolving deployment sequence...", "phase": 7})

        # affected_repos in impact_report is now a list of enriched dicts
        # (name, url, type, …) produced by Phase 3 using Phase 0's metadata.
        # No need to separately read selected_repos for git URL resolution.
        affected_repos = state.get("impact_report", {}).get("affected_repos", [])
        if not affected_repos:
            # Fallback: use selected_repos names if impact report is empty
            affected_repos = [r["name"] for r in state.get("selected_repos", [])]

        # Pull the SHAs Phase 6 merged so Phase 7 deploys exactly that code,
        # not whatever HEAD-of-main happens to be at clone time.
        merged_shas = state.get("merged_shas", {}) or {}

        graph = build_deployment_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p7"}}
        initial = DeploymentState(
            requirement=entry["requirement"],
            runbook=state.get("runbook", {}),
            pr_urls=entry.get("pr_urls", []),
            affected_repos=affected_repos,
            scope_contract=state.get("scope_contract", {}),
            merged_shas=merged_shas,                  # ← SHA per repo to git-checkout
            deploy_sequence=[], feature_flags=[],
            deploy_results=[], monitoring_results={},
            rollback_triggered=False,
            human_feedback=feedback, approved=False,
            status="STARTED",
        )

        result = graph.invoke(initial, config)
        merged = {**state}
        for k in ("deploy_sequence", "feature_flags", "deploy_results", "monitoring_results", "rollback_triggered"):
            if k in result: merged[k] = result[k]

        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "status": "WAITING_PHASE_7_APPROVAL", "sub_stage": "Deployment plan ready — Awaiting production approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
    except Exception as e:
        import traceback
        traceback.print_exc()   # so the real error shows in the uvicorn log
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(
            thread_id,
            pipeline_store[thread_id],
            _safe_state(pipeline_store[thread_id].get("current_state", {})),
        )


def _safe_state(state: dict) -> dict:
    safe_keys = ["selected_project", "selected_repos", "is_new_project", "candidates", "scope_contract", "classifier_output", "brd", "prd", "adr", "architecture", "sprint_plan", "runbook", "jira_tickets", "impact_report", "generated_changes", "test_files", "pr_urls", "target_repo", "deploy_results", "monitoring_results", "deploy_sequence", "feature_flags", "rollback_triggered", "status", "requirement", "merged_shas", "merge_errors", "phase6_final_status"]
    result = {}
    for k in safe_keys:
        if k in state:
            v = state[k]
            try:
                json.dumps(v)
                result[k] = v
            except: result[k] = str(v)
    return result

@app.get("/pipeline/{thread_id}/audit")
def pipeline_audit(thread_id: str):
    return {"thread_id": thread_id, "audit_log": get_audit_log(thread_id)}

def run_indexer(repo_path: str, repo_name: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from knowledge_layer.indexer import index_repo
        index_repo(repo_path, repo_name)
    except Exception as e:
        print(f"[Indexer] Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)