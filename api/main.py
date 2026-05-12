"""
SDLC Automation Platform V2 — FastAPI HTTP Layer
Exposes the LangGraph pipeline as HTTP endpoints.

ARCHITECTURAL FIX 2026-05-08:
  - Phase 0 (project selector) is now ACTUALLY invoked before Phase 1
  - is_new_project + selected_repos populated from Phase 0 propagate through state
  - Phase 6 routes to correct repo (no silent fallback to leave-mgmt-backend)
"""
import io
import os
import re
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.types import Command
from api.runbook_export import (
    export_runbook_excel,
    export_brd_markdown,
    export_prd_markdown,
    export_adr_markdown,
    export_architecture_markdown,
    export_sprint_plan_markdown,
    export_impact_markdown
)
import zipfile

load_dotenv()

# ── In-memory pipeline state store ────────────────────────────────────────────
pipeline_store: dict = {}

from api.persistence import (
    init_persistence_tables, save_pipeline, load_all_pipelines,
    audit, get_audit_log
)

# Initialize on startup
init_persistence_tables()

# Restore pipelines from DB on server start
try:
    restored = load_all_pipelines()
    pipeline_store.update(restored)
    print(f"[Startup] Restored {len(restored)} pipelines from DB")
except Exception as e:
    print(f"[Startup] DB restore failed: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SDLC Automation Platform V2",
    description="AI-powered SDLC pipeline — Phases 0-7",
    version="2.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Request / Response Models ─────────────────────────────────────────────────
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


# ── Slug helper (used as fallback if Phase 0 fails) ───────────────────────────
def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower())[:max_len].strip('-')
    return slug or "new-project"


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
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
            auth=(os.getenv("NEO4J_USER", "neo4j"),
                  os.getenv("NEO4J_PASSWORD", "password1234"))
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
        return HTMLResponse(content=f.read())


# ── Pipeline Endpoints ────────────────────────────────────────────────────────
@app.post("/pipeline/start")
def pipeline_start(req: StartRequest, background_tasks: BackgroundTasks):
    """
    Start the SDLC pipeline.
    FLOW: Phase 0 (project selector) → Phase 1 (discovery)
    """
    thread_id = req.thread_id or f"pipeline-{uuid.uuid4().hex[:8]}"

    if thread_id in pipeline_store:
        return JSONResponse(
            status_code=400,
            content={"error": f"Thread {thread_id} already exists."}
        )

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

    # FIX: Run Phase 0 FIRST, which then auto-chains to Phase 1
    background_tasks.add_task(run_phase0_and_phase1, thread_id, req.requirement)

    return {
        "thread_id": thread_id,
        "status": "STARTED",
        "message": "Pipeline started. Phase 0 routing → Phase 1 discovery.",
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
def pipeline_approve(thread_id: str, body: dict = Body(...), background_tasks: BackgroundTasks = None):
    if thread_id not in pipeline_store:
        raise HTTPException(404, "Not found")

    approved = body.get("approved", False)
    feedback = body.get("feedback", "")
    actor = body.get("actor", "user")

    entry = pipeline_store[thread_id]
    current_phase = str(entry.get("phase", "1"))

    if not approved:
        audit(thread_id, f"phase{current_phase}", "REJECTED", actor, {"feedback": feedback})
        entry["status"] = "REGENERATING"
        entry["sub_stage"] = f"Regenerating Phase {current_phase} with feedback..."
        entry["human_feedback"] = feedback
        save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

        if current_phase == "1":
            background_tasks.add_task(run_phase1, thread_id, entry["requirement"], feedback)
        elif current_phase == "2":
            background_tasks.add_task(run_phase2, thread_id, feedback)
        elif current_phase == "3":
            background_tasks.add_task(run_phase3, thread_id, feedback)
        elif current_phase == "6":
            background_tasks.add_task(run_phase6, thread_id, feedback)
        elif current_phase == "7":
            background_tasks.add_task(run_phase7, thread_id, feedback)

        return {"status": "REGENERATING", "phase": current_phase, "feedback_applied": feedback}

    # APPROVAL
    audit(thread_id, f"phase{current_phase}", "APPROVED", actor, {"feedback": feedback})
    entry["status"] = f"PHASE_{current_phase}_APPROVED"
    entry["sub_stage"] = f"Phase {current_phase} approved — proceeding..."
    save_pipeline(thread_id, entry, _safe_state(entry.get("current_state", {})))

    next_phase = str(int(current_phase) + 1)
    entry["phase"] = int(next_phase)

    if next_phase == "2":
        background_tasks.add_task(run_phase2, thread_id, "")
    elif next_phase == "3":
        # FIX: skip Phase 3 for new projects — there's nothing to impact-analyze
        is_new = entry.get("current_state", {}).get("is_new_project", False)
        if is_new:
            print(f"  [Approve] New project — skipping Phase 3, jumping to Phase 4")
            entry["phase"] = 4
            background_tasks.add_task(run_phase4, thread_id)
        else:
            background_tasks.add_task(run_phase3, thread_id, "")
    elif next_phase == "4":
        background_tasks.add_task(run_phase4, thread_id)
    elif next_phase == "5":
        background_tasks.add_task(run_phase5, thread_id)
    elif next_phase == "6":
        background_tasks.add_task(run_phase6, thread_id, "")
    elif next_phase == "7":
        background_tasks.add_task(run_phase7, thread_id, "")

    return {"status": "APPROVED", "next_phase": next_phase}


@app.get("/pipeline/list")
def pipeline_list():
    return {
        "pipelines": [
            {
                "thread_id": tid,
                "phase": e.get("phase", ""),
                "status": e.get("status", ""),
                "requirement": e["requirement"][:60] + "..." if len(e["requirement"]) > 60 else e["requirement"]
            }
            for tid, e in pipeline_store.items()
        ],
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
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=BRD_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/prd")
def download_prd(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_prd_markdown(entry["current_state"].get("prd", {}))
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=PRD_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/adr")
def download_adr(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_adr_markdown(entry["current_state"].get("adr", {}))
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=ADR_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/architecture")
def download_architecture(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_architecture_markdown(entry["current_state"].get("architecture", {}))
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=Architecture_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/sprint-plan")
def download_sprint_plan(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_sprint_plan_markdown(
        entry["current_state"].get("sprint_plan", {}),
        entry["current_state"].get("jira_tickets", [])
    )
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=SprintPlan_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/test-cases")
def download_test_cases(thread_id: str):
    from api.test_cases_export import export_test_cases_excel
    entry = _get_pipeline_or_404(thread_id)
    xlsx_bytes = export_test_cases_excel(entry)
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=TestCases_{thread_id}.xlsx"}
    )


@app.get("/pipeline/{thread_id}/download/impact")
def download_impact(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    md = export_impact_markdown(entry["current_state"].get("impact_report", {}))
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=ImpactReport_{thread_id}.md"})


@app.get("/pipeline/{thread_id}/download/runbook")
def download_runbook(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    xlsx_bytes = export_runbook_excel(entry)
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=Runbook_{thread_id}.xlsx"}
    )


@app.get("/pipeline/{thread_id}/download/all")
def download_all(thread_id: str):
    entry = _get_pipeline_or_404(thread_id)
    state = entry["current_state"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("01_BRD.md", export_brd_markdown(state.get("brd", {})))
        z.writestr("02_PRD.md", export_prd_markdown(state.get("prd", {})))
        z.writestr("03_ADR.md", export_adr_markdown(state.get("adr", {})))
        z.writestr("04_Architecture.md", export_architecture_markdown(state.get("architecture", {})))
        z.writestr("05_SprintPlan.md", export_sprint_plan_markdown(
            state.get("sprint_plan", {}), state.get("jira_tickets", [])))
        z.writestr("06_ImpactReport.md", export_impact_markdown(state.get("impact_report", {})))
        z.writestr("07_Runbook.xlsx", export_runbook_excel(entry))

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
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=Pipeline_{thread_id}_FullPackage.zip"}
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
    repos = [{"repo_name": r[0], "language": r[1], "file_count": r[2], "last_indexed": str(r[3])}
             for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"repos": repos, "total": len(repos)}


# ── PHASE 0 — Project Selector (NEW: actually invoked) ───────────────────────
def run_phase0_and_phase1(thread_id: str, requirement: str):
    """
    Run Phase 0 (project selector) FIRST, then chain into Phase 1.
    This is the architectural fix — Phase 0 was previously dead code.
    """
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        pipeline_store[thread_id]["phase"] = 0
        pipeline_store[thread_id]["status"] = "PHASE_0_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Searching for matching project..."
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))
        audit(thread_id, "phase0", "PHASE_STARTED")

        # Use selector logic directly (no graph needed — it's a single decision)
        from agents.phase0_selector.selector_agent import search_projects, slugify

        candidates = search_projects(requirement, top_k=3)

        if candidates and candidates[0]["score"] >= 0.7:
            selected = candidates[0]
            is_new = False
            selected_repos = selected.get("repos", [])
            # Normalize repos to dict format
            normalized_repos = []
            for r in selected_repos:
                if isinstance(r, str):
                    normalized_repos.append({
                        "name": r,
                        "type": "backend" if "backend" in r else "frontend" if "frontend" in r else "service",
                        "url": f"https://github.com/{os.getenv('GITHUB_REPO_OWNER','AkashW45')}/{r}.git",
                        "exists": True
                    })
                else:
                    normalized_repos.append(r)
            selected_repos = normalized_repos

            print(f"[Phase 0] ✅ Matched existing project: {selected['name']} (score: {selected['score']})")
            audit(thread_id, "phase0", "PROJECT_MATCHED",
                  details={"project": selected["name"], "score": selected["score"]})
        else:
            # NEW PROJECT
            slug = slugify(requirement)
            is_new = True
            github_owner = os.getenv("GITHUB_REPO_OWNER", "AkashW45")
            selected_repos = [
                {
                    "name": f"{slug}-backend",
                    "type": "backend",
                    "language": "python",
                    "url": f"https://github.com/{github_owner}/{slug}-backend.git",
                    "exists": False
                },
                {
                    "name": f"{slug}-frontend",
                    "type": "frontend",
                    "language": "typescript",
                    "url": f"https://github.com/{github_owner}/{slug}-frontend.git",
                    "exists": False
                }
            ]
            selected = {
                "id": f"new-{slug}",
                "name": requirement[:60],
                "description": requirement,
                "repos": selected_repos,
                "is_new": True,
                "score": 0.0
            }
            print(f"[Phase 0] 🆕 NEW PROJECT: {slug}")
            audit(thread_id, "phase0", "NEW_PROJECT_CREATED",
                  details={"slug": slug, "repos": [r["name"] for r in selected_repos]})

        # Persist Phase 0 results into state
        current = pipeline_store[thread_id].get("current_state", {}) or {}
        current["selected_project"] = selected
        current["selected_repos"] = selected_repos
        current["is_new_project"] = is_new
        current["candidates"] = candidates
        pipeline_store[thread_id]["current_state"] = current
        pipeline_store[thread_id]["status"] = "PHASE_0_DONE"
        pipeline_store[thread_id]["sub_stage"] = (
            f"{'New project' if is_new else 'Matched existing'}: {selected['name'][:50]}"
        )
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(current))

        # Now chain into Phase 1
        run_phase1(thread_id, requirement)

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 0 error: {str(e)}"
        })
        audit(thread_id, "phase0", "ERROR", details={"error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))


# ── Background Tasks ──────────────────────────────────────────────────────────
def run_phase1(thread_id: str, requirement: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase1_discovery.discovery_agent import (
            build_discovery_graph, DiscoveryState
        )

        pipeline_store[thread_id]["phase"] = 1
        pipeline_store[thread_id]["status"] = "PHASE_1_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Generating BRD..."
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))
        audit(thread_id, "phase1", "PHASE_STARTED")

        graph = build_discovery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p1"}}

        initial_state = DiscoveryState(
            requirement=requirement,
            brd={}, prd={}, adr={}, architecture={},
            human_feedback=feedback,
            approved=False,
            status="STARTED"
        )

        for chunk in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict):
                    continue

                # FIX: Preserve Phase 0 state when merging Phase 1 outputs
                current = pipeline_store[thread_id].get("current_state", {}) or {}
                # Phase 1 only contributes brd/prd/adr/architecture; preserve everything else
                for k, v in node_state.items():
                    if k in ("brd", "prd", "adr", "architecture", "status"):
                        current[k] = v
                pipeline_store[thread_id]["current_state"] = current

                if node_name == "generate_brd":
                    pipeline_store[thread_id]["sub_stage"] = "BRD Done — Generating PRD..."
                    pipeline_store[thread_id]["status"] = "PHASE_1_BRD_DONE"
                    audit(thread_id, "phase1", "BRD_GENERATED",
                          details={"title": node_state.get("brd", {}).get("title", "")})
                elif node_name == "generate_prd":
                    pipeline_store[thread_id]["sub_stage"] = "PRD Done — Generating ADR..."
                    pipeline_store[thread_id]["status"] = "PHASE_1_PRD_DONE"
                    audit(thread_id, "phase1", "PRD_GENERATED",
                          details={"frs": len(node_state.get("prd", {}).get("functional_requirements", []))})
                elif node_name == "generate_adr":
                    pipeline_store[thread_id]["sub_stage"] = "ADR Done — Generating Architecture..."
                    pipeline_store[thread_id]["status"] = "PHASE_1_ADR_DONE"
                    audit(thread_id, "phase1", "ADR_GENERATED",
                          details={"decisions": len(node_state.get("adr", {}).get("decisions", []))})
                elif node_name == "generate_architecture":
                    pipeline_store[thread_id]["sub_stage"] = "Architecture Done — Awaiting Approval"
                    pipeline_store[thread_id]["status"] = "WAITING_PHASE_1_APPROVAL"
                    audit(thread_id, "phase1", "ARCHITECTURE_GENERATED",
                          details={"nodes": len(node_state.get("architecture", {}).get("nodes", []))})

                save_pipeline(thread_id, pipeline_store[thread_id],
                              _safe_state(pipeline_store[thread_id]["current_state"]))

        pipeline_store[thread_id]["status"] = "WAITING_PHASE_1_APPROVAL"
        pipeline_store[thread_id]["graph"] = graph
        pipeline_store[thread_id]["config"] = config
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id]["current_state"]))

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 1 error: {str(e)}"
        })
        audit(thread_id, "phase1", "ERROR", details={"error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id],
                      _safe_state(pipeline_store[thread_id].get("current_state", {})))


def run_phase2(thread_id: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase2_planning.planning_agent import build_planning_graph, PlanningState

        entry = pipeline_store[thread_id]
        prev_state = entry["current_state"]

        pipeline_store[thread_id]["status"] = "PHASE_2_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Generating sprint plan..."
        pipeline_store[thread_id]["phase"] = 2
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(prev_state))

        graph = build_planning_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p2"}}

        initial = PlanningState(
            requirement=entry["requirement"],
            scope_contract=prev_state.get("scope_contract", {}),  # NEW
            brd=prev_state.get("brd", {}),
            prd=prev_state.get("prd", {}),
            sprint_plan={}, runbook={}, jira_tickets=[],
            human_feedback=feedback,
            approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)

        # FIX: merge instead of replace — preserve all Phase 0/1 state
        merged = {**prev_state}
        for k in ("sprint_plan", "runbook", "jira_tickets"):
            if k in result:
                merged[k] = result[k]

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": merged,
            "status": "WAITING_PHASE_2_APPROVAL",
            "sub_stage": "Sprint plan ready — Awaiting approval"
        })
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 2 error: {str(e)}"
        })


def run_phase3(thread_id: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase3_impact.graph import get_graph

        entry = pipeline_store[thread_id]
        prev_state = entry["current_state"]

        # FIX: For new projects, skip Phase 3 (nothing to impact analyze)
        if prev_state.get("is_new_project", False):
            print(f"[Phase 3] Skipping — new project has no existing code to analyze")
            empty_impact = {
                "requirement": entry["requirement"],
                "affected_repos": [r["name"] for r in prev_state.get("selected_repos", [])],
                "affected_files": [],
                "affected_symbols": [],
                "dependents": {},
                "protocol_contracts": [],
                "risk_assessment": {
                    "risk_level": "low",
                    "risk_reasons": ["New project — no existing code to risk"],
                    "breaking_changes": [],
                    "recommendation": "proceed"
                },
                "status": "SKIPPED_NEW_PROJECT"
            }
            merged = {**prev_state, "impact_report": empty_impact}
            pipeline_store[thread_id].update({
                "current_state": merged,
                "status": "PHASE_3_SKIPPED_NEW_PROJECT",
                "sub_stage": "New project — Phase 3 not applicable",
                "phase": 3
            })
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))
            # Auto-advance to Phase 4
            run_phase4(thread_id)
            return

        pipeline_store[thread_id]["status"] = "PHASE_3_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Analyzing code impact..."
        pipeline_store[thread_id]["phase"] = 3

        graph = get_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p3"}}

        initial = {
            "requirement": entry["requirement"],
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
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 3 error: {str(e)}"
        })


def run_phase4(thread_id: str):
    """Phase 4 + 5 — auto, with detailed sub-stage telemetry."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase4_codegen.codegen_agent import run_codegen
        from agents.phase5_validation.validation_agent import run_validation_phase

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        def update_substage(msg, status=None):
            pipeline_store[thread_id]["sub_stage"] = msg
            if status:
                pipeline_store[thread_id]["status"] = status
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
            print(f"  [Phase 4/5] {msg}")

        update_substage("Preparing context for codegen...", "PHASE_4_RUNNING")
        pipeline_store[thread_id]["phase"] = 4

        impact = state.get("impact_report", {})
        is_new = state.get("is_new_project", False)
        if is_new and not impact.get("affected_files"):
            impact = {
                "requirement": entry["requirement"],
                "affected_files": [],
                "affected_repos": [r["name"] for r in state.get("selected_repos", [])],
                "architecture": state.get("architecture", {}),
                "risk_assessment": {"risk_level": "low", "breaking_changes": [], "recommendation": "proceed"}
            }
            update_substage(f"NEW project — generating fresh scaffold for {len(state.get('selected_repos',[]))} repo(s)...")
        else:
            update_substage(f"Modifying {len(impact.get('affected_files', []))} existing files...")

        update_substage("Calling DeepSeek V4 Pro for code generation...")
        result4 = run_codegen(
            requirement=entry["requirement"],
            impact_report=impact,
            thread_id=f"{thread_id}-p4"
        )

        if result4["status"] != "VALIDATED":
            errors = result4.get("validation_errors", [])
            update_substage(f"Codegen failed: {len(errors)} validation errors", "ERROR")
            pipeline_store[thread_id]["error"] = f"Phase 4 failed: {errors[:3]}"
            return

        files_generated = len(result4.get("generated_changes", []))
        update_substage(f"✅ {files_generated} files generated and validated")
        
        # Phase 5
        pipeline_store[thread_id]["phase"] = 5
        update_substage("Generating pytest test files...", "PHASE_5_RUNNING")
        
        result5 = run_validation_phase(
            requirement=entry["requirement"],
            generated_changes=result4["generated_changes"],
            scope_contract=state.get("scope_contract", {}),
            thread_id=f"{thread_id}-p5"
        )

        tests_generated = len(result5.get("test_files", []))
        update_substage(f"✅ {tests_generated} test files generated")

        merged = {
            **state,
            "generated_changes": result4["generated_changes"],
            "test_files": result5.get("test_files", [])
        }
        pipeline_store[thread_id]["current_state"] = merged
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

        update_substage(f"Phases 4+5 complete: {files_generated} code files + {tests_generated} test files")
        run_phase6(thread_id, "")

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({"status": "ERROR", "error": f"Phase 4/5 error: {str(e)}"})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))


def run_phase6(thread_id: str, feedback: str = ""):
    """Phase 6 — Delivery with full visibility."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase6_delivery.delivery_agent import build_delivery_graph, DeliveryState
        import requests as req_lib

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        def update_substage(msg, status=None):
            pipeline_store[thread_id]["sub_stage"] = msg
            if status:
                pipeline_store[thread_id]["status"] = status
            save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(state))
            print(f"  [Phase 6] {msg}")

        update_substage("Determining target repository...", "PHASE_6_RUNNING")
        pipeline_store[thread_id]["phase"] = 6

        is_new_project = state.get("is_new_project", False)
        selected_repos = state.get("selected_repos", [])

        if not selected_repos:
            update_substage("ERROR: No selected_repos. Phase 0 did not run.", "ERROR")
            pipeline_store[thread_id]["error"] = "No selected_repos in state"
            return

        target_repo = next((r for r in selected_repos if r.get("type") == "backend"), selected_repos[0])
        target_repo_name = target_repo["name"]
        target_repo_url = target_repo.get("url",
            f"https://github.com/{os.getenv('GITHUB_REPO_OWNER','AkashW45')}/{target_repo_name}.git")

        github_token = os.getenv("GITHUB_TOKEN")
        github_owner = os.getenv("GITHUB_REPO_OWNER", "AkashW45")

        update_substage(f"Target: {target_repo_name} ({'NEW' if is_new_project else 'existing'})")

        if is_new_project or not target_repo.get("exists", True):
            update_substage(f"Checking if {target_repo_name} exists on GitHub...")

            check_url = f"https://api.github.com/repos/{github_owner}/{target_repo_name}"
            check_resp = req_lib.get(check_url, headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json"
            })

            if check_resp.status_code == 404:
                update_substage(f"Creating new GitHub repo: {target_repo_name}...")
                create_resp = req_lib.post(
                    "https://api.github.com/user/repos",
                    headers={
                        "Authorization": f"Bearer {github_token}",
                        "Accept": "application/vnd.github+json"
                    },
                    json={
                        "name": target_repo_name,
                        "description": entry["requirement"][:200],
                        "private": False,
                        "auto_init": True
                    }
                )
                if create_resp.status_code == 201:
                    update_substage(f"✅ GitHub repo created: {target_repo_name}")
                    audit(thread_id, "phase6", "REPO_CREATED", details={"repo": target_repo_name})
                else:
                    update_substage(f"❌ Failed to create repo: {create_resp.status_code}", "ERROR")
                    pipeline_store[thread_id]["error"] = f"Repo creation failed: {create_resp.text[:200]}"
                    return
            else:
                update_substage(f"ℹ️ Repo {target_repo_name} already exists, will push to it")

        branch_name = "main" if is_new_project else f"feature/{thread_id}"
        update_substage(f"Pushing {len(state.get('generated_changes',[]))} files to {target_repo_name}/{branch_name}...")

        graph = build_delivery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p6"}}

        initial = DeliveryState(
            requirement=entry["requirement"],
            generated_changes=state.get("generated_changes", []),
            test_files=state.get("test_files", []),
            branch_name=branch_name,
            repo_url=target_repo_url,
            pr_urls=[],
            human_feedback=feedback,
            approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)
        pr_urls = result.get("pr_urls", [])

        merged = {**state, "pr_urls": pr_urls, "target_repo": target_repo_name}
        pipeline_store[thread_id].update({
            "graph": graph, "config": config,
            "current_state": merged,
            "pr_urls": pr_urls,
            "status": "WAITING_PHASE_6_APPROVAL"
        })

        if pr_urls:
            update_substage(f"✅ Pushed to {target_repo_name} — PR: {pr_urls[0][:60]}...")
        else:
            update_substage(f"✅ Pushed to {target_repo_name}/{branch_name} — Awaiting PR approval")

        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({"status": "ERROR", "error": f"Phase 6 error: {str(e)}"})
        audit(thread_id, "phase6", "ERROR", details={"error": str(e)})
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(pipeline_store[thread_id].get("current_state", {})))


def run_phase7(thread_id: str, feedback: str = ""):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase7_deployment.deployment_agent import build_deployment_graph, DeploymentState

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id]["status"] = "PHASE_7_RUNNING"
        pipeline_store[thread_id]["sub_stage"] = "Resolving deployment sequence..."
        pipeline_store[thread_id]["phase"] = 7

        # Use selected_repos from Phase 0, not impact_report
        affected_repos = [r["name"] for r in state.get("selected_repos", [])]
        if not affected_repos:
            affected_repos = state.get("impact_report", {}).get("affected_repos", [])

        graph = build_deployment_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p7"}}

        initial = DeploymentState(
            requirement=entry["requirement"],
            runbook=state.get("runbook", {}),
            pr_urls=entry.get("pr_urls", []),
            affected_repos=affected_repos,
            scope_contract=state.get("scope_contract", {}),
            deploy_sequence=[],
            feature_flags=[],
            deploy_results=[],
            monitoring_results={},
            rollback_triggered=False,
            human_feedback=feedback,
            approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)

        merged = {**state}
        for k in ("deploy_sequence", "feature_flags", "deploy_results", "monitoring_results", "rollback_triggered"):
            if k in result:
                merged[k] = result[k]

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": merged,
            "status": "WAITING_PHASE_7_APPROVAL",
            "sub_stage": "Deployment plan ready — Awaiting production approval"
        })
        save_pipeline(thread_id, pipeline_store[thread_id], _safe_state(merged))

    except Exception as e:
        import traceback
        traceback.print_exc()
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 7 error: {str(e)}"
        })


def run_indexer(repo_path: str, repo_name: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from knowledge_layer.indexer import index_repo
        index_repo(repo_path, repo_name)
    except Exception as e:
        print(f"[Indexer] Error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_state(state: dict) -> dict:
    """Strip non-serialisable items. CRITICAL: must include Phase 0 keys."""
    safe_keys = [
        # Phase 0
        "selected_project", "selected_repos", "is_new_project", "candidates",
        # Phase 0.5 — NEW
        "scope_contract", "classifier_output",
        # Phase 1
        "brd", "prd", "adr", "architecture",
        # Phase 2
        "sprint_plan", "runbook", "jira_tickets",
        # Phase 3
        "impact_report",
        # Phase 4 / 5
        "generated_changes", "test_files",
        # Phase 6
        "pr_urls", "target_repo",
        # Phase 7
        "deploy_results", "monitoring_results", "deploy_sequence",
        "feature_flags", "rollback_triggered",
        # Common
        "status", "requirement"
    ]
    if not isinstance(state, dict):
        return {}
    result = {}
    for k in safe_keys:
        if k in state:
            v = state[k]
            try:
                import json
                json.dumps(v)
                result[k] = v
            except Exception:
                result[k] = str(v)
    return result


@app.get("/pipeline/{thread_id}/audit")
def pipeline_audit(thread_id: str):
    return {"thread_id": thread_id, "audit_log": get_audit_log(thread_id)}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)