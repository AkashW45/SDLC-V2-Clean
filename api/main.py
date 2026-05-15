"""
SDLC Automation Platform V2 — FastAPI HTTP Layer
Unified Version: Blends Surgical Replays & API Security with Phase 0 Smart Routing.
"""
import io
import os
import re
import uuid
import json
import asyncio
import requests
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import zipfile

from langgraph.types import Command

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
        from qdrant_client import QdrantClient
        q = QdrantClient(url="http://127.0.0.1:6333", timeout=5)
        collections = q.get_collections()
        services["qdrant"] = {"status": "ok", "collections": len(collections.collections)}
    except Exception as e:
        services["qdrant"] = {"status": "error", "error": str(e)}

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password1234"))
        )
        with driver.session() as s:
            s.run("RETURN 1")
        driver.close()
        services["neo4j"] = {"status": "ok"}
    except Exception as e:
        services["neo4j"] = {"status": "error", "error": str(e)}

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=os.getenv("POSTGRES_PORT", "5433"),
            user=os.getenv("POSTGRES_USER", "sdlc"),
            password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
            dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
        )
        conn.close()
        services["postgres"] = {"status": "ok"}
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


@app.post("/pipeline/start")
def pipeline_start(req: StartRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """Starts Phase 0 routing, then automatically chains to Phase 1."""
    thread_id = req.thread_id or f"pipeline-{uuid.uuid4().hex[:8]}"

    if thread_id in pipeline_store:
        return JSONResponse(status_code=400, content={"error": f"Thread {thread_id} exists."})

    pipeline_store[thread_id] = {
        "thread_id": thread_id,
        "requirement": req.requirement,
        "phase": 0,
        "status": "STARTING",
        "sub_stage": "Routing to project...",
        "current_state": {},
        "graph": None,
        "config": None,
        "pr_urls": [],
        "error": None
    }

    # IMPORTANT: Run Phase 0 routing first
    background_tasks.add_task(run_phase0_and_phase1, thread_id, req.requirement)

    return {
        "thread_id": thread_id,
        "status": "STARTED",
        "message": "Pipeline started. Polling Phase 0 -> Phase 1.",
        "next": f"/pipeline/status/{thread_id}"
    }

@app.get("/pipeline/status/{thread_id}")
def pipeline_status(thread_id: str):
    if thread_id not in pipeline_store:
        raise HTTPException(404, f"Thread {thread_id} not found")
    entry = pipeline_store[thread_id]
    safe = _safe_state(entry.get("current_state", {}))
    return {
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

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"01_BRD.md", export_brd_markdown(state.get("brd", {})))
        z.writestr(f"02_PRD.md", export_prd_markdown(state.get("prd", {})))
        z.writestr(f"03_ADR.md", export_adr_markdown(state.get("adr", {})))
        z.writestr(f"04_Architecture.md", export_architecture_markdown(state.get("architecture", {})))
        z.writestr(f"05_SprintPlan.md", export_sprint_plan_markdown(state.get("sprint_plan", {}), state.get("jira_tickets", [])))
        z.writestr(f"06_ImpactReport.md", export_impact_markdown(state.get("impact_report", {})))
        z.writestr(f"07_Runbook.xlsx", export_runbook_excel(entry))

        for change in state.get("generated_changes", []):
            fname = change.get("file_path", "unknown.txt").replace("/", "_").replace("\\", "_")
            z.writestr(f"08_code/{fname}", change.get("content", ""))

        for test in state.get("test_files", []):
            fname = test.get("test_file_path", "test.py").replace("/", "_").replace("\\", "_")
            z.writestr(f"09_tests/{fname}", test.get("content", ""))

        from api.test_cases_export import export_test_cases_excel
        try:
            z.writestr("10_TestCases.xlsx", export_test_cases_excel(entry))
        except Exception as e:
            print(f"[Download] test cases skipped: {e}")

    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=Pipeline_{thread_id}_FullPackage.zip"})


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
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )
    cur = conn.cursor()
    cur.execute("SELECT repo_name, language, file_count, last_indexed FROM repo_maps ORDER BY last_indexed DESC")
    repos = [{"repo_name": r[0], "language": r[1], "file_count": r[2], "last_indexed": str(r[3])} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"repos": repos, "total": len(repos)}


# ── Background Pipeline Runners ───────────────────────────────────────────────

def run_phase0_and_phase1(thread_id: str, requirement: str):
    """Main's Smart Repo Routing logic."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase0_selector.selector_agent import search_projects, slugify

        pipeline_store[thread_id]["phase"] = 0
        pipeline_store[thread_id]["status"] = "PHASE_0_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Searching for matching project..."
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))
        audit(thread_id, "phase0", "PHASE_STARTED")

        candidates = search_projects(requirement, top_k=3)

        if candidates and candidates[0]["score"] >= 0.7:
            selected = candidates[0]
            is_new = False
            selected_repos = [{"name": r, "type": "backend", "exists": True} if isinstance(r, str) else r for r in selected.get("repos", [])]
            print(f"[Phase 0] ✅ Matched existing project: {selected['name']}")
        else:
            slug = slugify(requirement)
            is_new = True
            github_owner = os.getenv("GITHUB_REPO_OWNER", "AkashW45")
            selected_repos = [
                {"name": f"{slug}-backend", "type": "backend", "url": f"https://github.com/{github_owner}/{slug}-backend.git", "exists": False},
                {"name": f"{slug}-frontend", "type": "frontend", "url": f"https://github.com/{github_owner}/{slug}-frontend.git", "exists": False}
            ]
            selected = {"id": f"new-{slug}", "name": requirement[:60], "repos": selected_repos, "is_new": True}
            print(f"[Phase 0] 🆕 NEW PROJECT: {slug}")

        current = pipeline_store[thread_id].get("current_state", {})
        current.update({"selected_project": selected, "selected_repos": selected_repos, "is_new_project": is_new, "candidates": candidates})
        pipeline_store[thread_id].update({"current_state": current, "status": "PHASE_0_DONE", "sub_stage": f"Project routed to: {selected['name'][:50]}"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(current))

        run_phase1(thread_id, requirement)

    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": f"Phase 0 error: {str(e)}"})

def run_phase1(thread_id: str, requirement: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase1_discovery.discovery_agent import build_discovery_graph, DiscoveryState

        pipeline_store[thread_id].update({"phase": 1, "status": "PHASE_1_RUNNING", "sub_stage": "Generating BRD..."})

        graph = build_discovery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p1"}}
        initial_state = DiscoveryState(requirement=requirement, brd={}, prd={}, adr={}, architecture={}, human_feedback=feedback, approved=False, status="STARTED")

        for chunk in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict): continue
                current = pipeline_store[thread_id].get("current_state", {})
                for k, v in node_state.items():
                    if k in ("brd", "prd", "adr", "architecture", "status"): current[k] = v
                pipeline_store[thread_id]["current_state"] = current

                if node_name == "generate_brd": pipeline_store[thread_id]["sub_stage"] = "BRD Done — Generating PRD..."
                elif node_name == "generate_prd": pipeline_store[thread_id]["sub_stage"] = "PRD Done — Generating ADR..."
                elif node_name == "generate_adr": pipeline_store[thread_id]["sub_stage"] = "ADR Done — Generating Architecture..."
                elif node_name == "generate_architecture": pipeline_store[thread_id]["sub_stage"] = "Architecture Done — Awaiting Approval"
                save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(current))

        pipeline_store[thread_id].update({"status": "WAITING_PHASE_1_APPROVAL", "graph": graph, "config": config})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id]["current_state"]))

    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": f"Phase 1 error: {str(e)}"})

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
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})

def run_phase3(thread_id: str, feedback: str = ""):
    try:
        from agents.phase3_impact.graph import get_graph
        entry = pipeline_store[thread_id]
        prev_state = entry["current_state"]

        pipeline_store[thread_id].update({"status": "PHASE_3_RUNNING", "sub_stage": "Analyzing code impact...", "phase": 3})

        graph = get_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p3"}}
        initial = {"requirement": entry["requirement"], "impact_report": {}, "human_approved": False, "human_feedback": feedback, "status": "STARTED"}
        result = graph.invoke(initial, config)

        merged = {**prev_state}
        if "impact_report" in result: merged["impact_report"] = result["impact_report"]

        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "status": "WAITING_PHASE_3_APPROVAL", "sub_stage": "Impact report ready — Awaiting approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})

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

        if result4["status"] != "VALIDATED":
            pipeline_store[thread_id].update({"status": "ERROR", "error": f"Phase 4 failed: {result4.get('validation_errors', [])}"})
            return

        # Save generated code to state before moving to Phase 5
        pipeline_store[thread_id]["current_state"]["generated_changes"] = result4["generated_changes"]
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id]["current_state"]))
        
        # Route directly to Phase 5
        run_phase5(thread_id)
        

       
        
    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
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
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))
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

        if is_new_project or not target_repo.get("exists", True):
            check_resp = req_lib.get(f"https://api.github.com/repos/{github_owner}/{target_repo_name}", headers={"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"})
            if check_resp.status_code == 404:
                req_lib.post("https://api.github.com/user/repos", headers={"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}, json={"name": target_repo_name, "private": False, "auto_init": True})

        branch_name = "main" if is_new_project else f"feature/{thread_id}"

        graph = build_delivery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p6"}}
        initial = DeliveryState(requirement=entry["requirement"], generated_changes=state.get("generated_changes", []), test_files=state.get("test_files", []), branch_name=branch_name, repo_url=target_repo_url, pr_urls=[], human_feedback=feedback, approved=False, status="STARTED")

        result = graph.invoke(initial, config)
        pr_urls = result.get("pr_urls", [])

        merged = {**state, "pr_urls": pr_urls, "target_repo": target_repo_name}
        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "pr_urls": pr_urls, "status": "WAITING_PHASE_6_APPROVAL", "sub_stage": f"Pushed to {target_repo_name} — Awaiting PR approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})


def run_phase7(thread_id: str, feedback: str = ""):
    try:
        from agents.phase7_deployment.deployment_agent import build_deployment_graph, DeploymentState
        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id].update({"status": "PHASE_7_RUNNING", "sub_stage": "Resolving deployment sequence...", "phase": 7})

        affected_repos = [r["name"] for r in state.get("selected_repos", [])] or state.get("impact_report", {}).get("affected_repos", [])

        graph = build_deployment_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p7"}}
        initial = DeploymentState(requirement=entry["requirement"], runbook=state.get("runbook", {}), pr_urls=entry.get("pr_urls", []), affected_repos=affected_repos, scope_contract=state.get("scope_contract", {}), deploy_sequence=[], feature_flags=[], deploy_results=[], monitoring_results={}, rollback_triggered=False, human_feedback=feedback, approved=False, status="STARTED")

        result = graph.invoke(initial, config)
        merged = {**state}
        for k in ("deploy_sequence", "feature_flags", "deploy_results", "monitoring_results", "rollback_triggered"):
            if k in result: merged[k] = result[k]

        pipeline_store[thread_id].update({"graph": graph, "config": config, "current_state": merged, "status": "WAITING_PHASE_7_APPROVAL", "sub_stage": "Deployment plan ready — Awaiting production approval"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
    except Exception as e:
        pipeline_store[thread_id].update({"status": "ERROR", "error": str(e)})


def _safe_state(state: dict) -> dict:
    safe_keys = ["selected_project", "selected_repos", "is_new_project", "candidates", "scope_contract", "classifier_output", "brd", "prd", "adr", "architecture", "sprint_plan", "runbook", "jira_tickets", "impact_report", "generated_changes", "test_files", "pr_urls", "target_repo", "deploy_results", "monitoring_results", "deploy_sequence", "feature_flags", "rollback_triggered", "status", "requirement"]
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