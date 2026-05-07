"""
SDLC Automation Platform V2 — FastAPI HTTP Layer
Exposes the LangGraph pipeline as HTTP endpoints.
"""

import os
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SDLC Automation Platform V2",
    description="AI-powered SDLC pipeline — Phases 1-7",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── In-memory pipeline state store ────────────────────────────────────────────
# Stores: { thread_id: { graph, config, result, phase, status } }
pipeline_store: dict = {}


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


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check all service connections."""
    services = {}

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        q = QdrantClient(url="http://127.0.0.1:6333", timeout=5)
        collections = q.get_collections()
        services["qdrant"] = {
            "status": "ok",
            "collections": len(collections.collections)
        }
    except Exception as e:
        services["qdrant"] = {"status": "error", "error": str(e)}

    # Check Neo4j
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

    # Check PostgreSQL
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

    overall = "ok" if all(
        s["status"] == "ok" for s in services.values()
    ) else "degraded"

    return {
        "status": overall,
        "services": services,
        "version": "2.0.0"
    }

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(
        os.path.dirname(__file__), '..', 'dashboard', 'index.html'
    )
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())
    

# ── Pipeline Endpoints ────────────────────────────────────────────────────────
@app.post("/pipeline/start")
def pipeline_start(req: StartRequest, background_tasks: BackgroundTasks):
    """
    Start the SDLC pipeline from a plain English requirement.
    Returns immediately with thread_id.
    Pipeline runs Phase 1 until first INTERRUPT, then waits.
    """
    thread_id = req.thread_id or f"pipeline-{uuid.uuid4().hex[:8]}"

    if thread_id in pipeline_store:
        return JSONResponse(
            status_code=400,
            content={"error": f"Thread {thread_id} already exists. Use a different thread_id."}
        )

    # Initialize store entry
    pipeline_store[thread_id] = {
        "thread_id": thread_id,
        "requirement": req.requirement,
        "phase": 1,
        "status": "STARTING",
        "current_state": {},
        "graph": None,
        "config": None,
        "pr_urls": [],
        "error": None
    }

    # Run Phase 1 in background
    background_tasks.add_task(run_phase1, thread_id, req.requirement)

    return {
        "thread_id": thread_id,
        "status": "STARTED",
        "message": "Pipeline started. Poll /pipeline/status/{thread_id} for updates.",
        "next": f"/pipeline/status/{thread_id}"
    }


@app.get("/pipeline/status/{thread_id}")
def pipeline_status(thread_id: str):
    """Get current pipeline state."""
    if thread_id not in pipeline_store:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    entry = pipeline_store[thread_id]

    # Build clean response without internal graph objects
    return {
        "thread_id": thread_id,
        "phase": entry["phase"],
        "status": entry["status"],
        "requirement": entry["requirement"],
        "pr_urls": entry.get("pr_urls", []),
        "error": entry.get("error"),
        "current_state": _safe_state(entry.get("current_state", {}))
    }


@app.post("/pipeline/approve/{thread_id}")
def pipeline_approve(thread_id: str, req: ApproveRequest, background_tasks: BackgroundTasks):
    """
    Resume pipeline after human approval.
    Call this when human approves or rejects at any gate.
    """
    if thread_id not in pipeline_store:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    entry = pipeline_store[thread_id]

    if "WAITING" not in entry["status"] and "INTERRUPT" not in entry["status"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Pipeline is not waiting for approval. Current status: {entry['status']}"
            }
        )

    if not req.approved:
        pipeline_store[thread_id]["status"] = "REJECTED_BY_HUMAN"
        pipeline_store[thread_id]["error"] = f"Rejected at phase {entry['phase']}: {req.feedback}"
        return {
            "thread_id": thread_id,
            "status": "REJECTED_BY_HUMAN",
            "message": f"Pipeline rejected at phase {entry['phase']}"
        }

    # Resume in background
    background_tasks.add_task(
        resume_current_phase,
        thread_id,
        req.approved,
        req.feedback or ""
    )

    return {
        "thread_id": thread_id,
        "status": "RESUMING",
        "message": "Approval received. Pipeline resuming.",
        "next": f"/pipeline/status/{thread_id}"
    }


@app.get("/pipeline/list")
def pipeline_list():
    """List all active pipelines."""
    return {
        "pipelines": [
            {
                "thread_id": tid,
                "phase": e["phase"],
                "status": e["status"],
                "requirement": e["requirement"][:60] + "..."
                if len(e["requirement"]) > 60 else e["requirement"]
            }
            for tid, e in pipeline_store.items()
        ],
        "total": len(pipeline_store)
    }


@app.delete("/pipeline/{thread_id}")
def pipeline_delete(thread_id: str):
    """Remove a pipeline from the store."""
    if thread_id not in pipeline_store:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    del pipeline_store[thread_id]
    return {"message": f"Pipeline {thread_id} removed"}


# ── Knowledge Layer Endpoints ─────────────────────────────────────────────────
@app.post("/knowledge/index")
def knowledge_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Index a repository into the Knowledge Layer."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    background_tasks.add_task(run_indexer, req.repo_path, req.repo_name)

    return {
        "status": "INDEXING_STARTED",
        "repo_name": req.repo_name,
        "repo_path": req.repo_path,
        "message": "Indexing running in background."
    }


@app.post("/knowledge/search")
def knowledge_search(req: SearchRequest):
    """Semantic search across indexed repos."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from agents.phase3_impact.impact_analyzer import semantic_search
    hits = semantic_search(req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "results": hits,
        "total": len(hits)
    }


@app.get("/knowledge/repos")
def knowledge_repos():
    """List all indexed repos."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT repo_name, language, file_count, last_indexed
        FROM repo_maps ORDER BY last_indexed DESC
    """)
    repos = [
        {
            "repo_name": r[0],
            "language": r[1],
            "file_count": r[2],
            "last_indexed": str(r[3])
        }
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return {"repos": repos, "total": len(repos)}


# ── Background Tasks ──────────────────────────────────────────────────────────
def run_phase1(thread_id: str, requirement: str):
    """Run Phase 1 until INTERRUPT."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        from agents.phase1_discovery.discovery_agent import (
            build_discovery_graph, DiscoveryState
        )

        pipeline_store[thread_id]["status"] = "PHASE_1_RUNNING"

        graph = build_discovery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p1"}}

        initial_state = DiscoveryState(
            requirement=requirement,
            brd={}, prd={}, adr={},
            human_feedback="", approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial_state, config)

        pipeline_store[thread_id].update({
            "phase": 1,
            "status": "WAITING_PHASE_1_APPROVAL",
            "graph": graph,
            "config": config,
            "current_state": result
        })

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 1 error: {str(e)}"
        })


def resume_current_phase(thread_id: str, approved: bool, feedback: str):
    """Resume whichever phase is currently waiting."""
    entry = pipeline_store[thread_id]
    phase = entry["phase"]

    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        graph = entry["graph"]
        config = entry["config"]

        # Resume current phase
        result = graph.invoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config
        )

        entry["current_state"] = result
        status = result.get("status", "")

        # Move to next phase based on approval status
        if phase == 1 and status == "APPROVED_FOR_PLANNING":
            pipeline_store[thread_id]["phase"] = 1
            pipeline_store[thread_id]["status"] = "PHASE_1_APPROVED"
            run_phase2(thread_id)

        elif phase == 2 and status == "APPROVED_FOR_IMPACT_ANALYSIS":
            pipeline_store[thread_id]["status"] = "PHASE_2_APPROVED"
            run_phase3(thread_id)

        elif phase == 3 and status == "APPROVED_FOR_CODE_GENERATION":
            pipeline_store[thread_id]["status"] = "PHASE_3_APPROVED"
            run_phase4(thread_id)

        elif phase == 6 and status == "APPROVED_FOR_DEPLOYMENT":
            pipeline_store[thread_id]["status"] = "PHASE_6_APPROVED"
            run_phase7(thread_id)

        elif phase == 7 and status in ("DEPLOYMENT_COMPLETE", "ROLLED_BACK"):
            pipeline_store[thread_id]["status"] = "PIPELINE_COMPLETE"

        else:
            pipeline_store[thread_id]["status"] = status

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase {phase} resume error: {str(e)}"
        })


def run_phase2(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase2_planning.planning_agent import build_planning_graph, PlanningState

        entry = pipeline_store[thread_id]
        p1_state = entry["current_state"]

        pipeline_store[thread_id]["status"] = "PHASE_2_RUNNING"
        pipeline_store[thread_id]["phase"] = 2

        graph = build_planning_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p2"}}

        initial = PlanningState(
            requirement=entry["requirement"],
            brd=p1_state.get("brd", {}),
            prd=p1_state.get("prd", {}),
            sprint_plan={}, runbook={},
            human_feedback="", approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": {**p1_state, **result},
            "status": "WAITING_PHASE_2_APPROVAL"
        })

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 2 error: {str(e)}"
        })


def run_phase3(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase3_impact.graph import get_graph

        entry = pipeline_store[thread_id]
        pipeline_store[thread_id]["status"] = "PHASE_3_RUNNING"
        pipeline_store[thread_id]["phase"] = 3

        graph = get_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p3"}}

        initial = {
            "requirement": entry["requirement"],
            "impact_report": {},
            "human_approved": False,
            "human_feedback": "",
            "status": "STARTED"
        }

        result = graph.invoke(initial, config)

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": {**entry["current_state"], **result},
            "status": "WAITING_PHASE_3_APPROVAL"
        })

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 3 error: {str(e)}"
        })


def run_phase4(thread_id: str):
    """Phase 4 + 5 run automatically — no human gate."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase4_codegen.codegen_agent import run_codegen
        from agents.phase5_validation.validation_agent import run_validation_phase

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        # Phase 4
        pipeline_store[thread_id]["status"] = "PHASE_4_RUNNING"
        pipeline_store[thread_id]["phase"] = 4

        result4 = run_codegen(
            requirement=entry["requirement"],
            impact_report=state.get("impact_report", {}),
            thread_id=f"{thread_id}-p4"
        )

        if result4["status"] != "VALIDATED":
            pipeline_store[thread_id].update({
                "status": "ERROR",
                "error": f"Phase 4 failed: {result4.get('validation_errors', [])}"
            })
            return

        # Phase 5
        pipeline_store[thread_id]["status"] = "PHASE_5_RUNNING"
        pipeline_store[thread_id]["phase"] = 5

        result5 = run_validation_phase(
            requirement=entry["requirement"],
            generated_changes=result4["generated_changes"],
            thread_id=f"{thread_id}-p5"
        )

        # Update state and move to Phase 6
        pipeline_store[thread_id]["current_state"].update({
            "generated_changes": result4["generated_changes"],
            "test_files": result5["test_files"]
        })

        run_phase6(thread_id)

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 4/5 error: {str(e)}"
        })


def run_phase6(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase6_delivery.delivery_agent import build_delivery_graph, DeliveryState

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id]["status"] = "PHASE_6_RUNNING"
        pipeline_store[thread_id]["phase"] = 6

        graph = build_delivery_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p6"}}

        initial = DeliveryState(
            requirement=entry["requirement"],
            generated_changes=state.get("generated_changes", []),
            test_files=state.get("test_files", []),
            branch_name=f"feature/{thread_id}",
            repo_url=os.getenv(
                "REPO_URL",
                "https://github.com/AkashW45/leave-mgmt-backend.git"
            ),
            pr_urls=[],
            human_feedback="", approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": {**state, **result},
            "pr_urls": result.get("pr_urls", []),
            "status": "WAITING_PHASE_6_APPROVAL"
        })

    except Exception as e:
        pipeline_store[thread_id].update({
            "status": "ERROR",
            "error": f"Phase 6 error: {str(e)}"
        })


def run_phase7(thread_id: str):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from agents.phase7_deployment.deployment_agent import build_deployment_graph, DeploymentState

        entry = pipeline_store[thread_id]
        state = entry["current_state"]

        pipeline_store[thread_id]["status"] = "PHASE_7_RUNNING"
        pipeline_store[thread_id]["phase"] = 7

        graph = build_deployment_graph()
        config = {"configurable": {"thread_id": f"{thread_id}-p7"}}

        initial = DeploymentState(
            requirement=entry["requirement"],
            runbook=state.get("runbook", {}),
            pr_urls=entry.get("pr_urls", []),
            affected_repos=state.get(
                "impact_report", {}
            ).get("affected_repos", ["leave-mgmt-backend"]),
            deploy_sequence=[],
            feature_flags=[],
            deploy_results=[],
            monitoring_results={},
            rollback_triggered=False,
            human_feedback="", approved=False,
            status="STARTED"
        )

        result = graph.invoke(initial, config)

        pipeline_store[thread_id].update({
            "graph": graph,
            "config": config,
            "current_state": {**state, **result},
            "status": "WAITING_PHASE_7_APPROVAL"
        })

    except Exception as e:
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
    """Strip non-serialisable items (graph objects etc)."""
    safe_keys = [
        "brd", "prd", "adr", "sprint_plan", "runbook",
        "impact_report", "generated_changes", "test_files",
        "pr_urls", "deploy_results", "monitoring_results",
        "status", "requirement"
    ]
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


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)