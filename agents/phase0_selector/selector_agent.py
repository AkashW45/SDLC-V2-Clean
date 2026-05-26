import os
import sys
import re


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, List, Dict, Any
from qdrant_client import QdrantClient
# NOTE: SentenceTransformer is no longer imported here. The model is provided
# by the shared singleton in core/embeddings.py (lazy + warmed at startup).
import psycopg2
import json


class SelectorState(TypedDict):
    requirement: str
    candidates: List[Dict[str, Any]]
    selected_project: Dict[str, Any]
    selected_repos: List[Dict[str, Any]]
    is_new_project: bool
    human_input: Dict[str, Any]
    status: str


def get_qdrant():
    # Process-wide singleton Qdrant client.
    from core.db_clients import qdrant_client
    return qdrant_client


def get_postgres():
    # Pooled connection — .close() returns it to the pool.
    from core.db_clients import PooledConn
    return PooledConn()


def get_embedder():
    # Shared process-wide singleton (see core/embeddings.py). Loaded once,
    # warmed at server startup, so this call is effectively free.
    from core.embeddings import get_embedder as _shared
    return _shared()


def search_projects(requirement: str, top_k: int = 3):
    qdrant = get_qdrant()
    embedder = get_embedder()

    try:
        vector = embedder.encode(requirement).tolist()
        results = get_qdrant().query_points(
            collection_name="project_embeddings",
            query=vector,
            limit=top_k * 3,           # over-fetch since we'll dedupe
            with_payload=True,
        )

        # ─── DEDUPE by project_id keeping highest score (B1 fix) ───
        seen = {}
        for r in results.points:
            p = r.payload or {}
            pid = p.get("project_id", "")
            if not pid:
                continue
            if pid not in seen or r.score > seen[pid].score:
                seen[pid] = r

        # Build candidate list from deduped results, top_k highest
        deduped = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]
        out = []
        for r in deduped:
            p = r.payload or {}
            out.append({
                "project_id": p.get("project_id", ""),
                "project_name": p.get("project_name", ""),
                "description": p.get("description", ""),
                "domain": p.get("domain", ""),
                "repos": p.get("repos", []),
                "score": round(r.score, 4),
            })
        return out
    except Exception as e:
        print(f"[Phase 0] Qdrant search failed: {e}")
        return []


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower())[:max_len].strip('-')
    return slug or "new-project"


def select_project_node(state: SelectorState) -> SelectorState:
    print("\n[Phase 0] Searching for matching project...")
    requirement = state["requirement"]

    candidates = search_projects(requirement, top_k=3)

    if candidates:
        print(f"  Found {len(candidates)} candidates:")
        for c in candidates:
            print(f"    - {c['name']} (score: {c['score']})")

    threshold = float(os.getenv("PHASE0_MATCH_THRESHOLD", "0.55"))
    if candidates and candidates[0]["score"] >= threshold:
        selected = candidates[0]
        print(f"  ✅ Auto-selected: {selected['name']} (score: {selected['score']})")
        return {
            **state,
            "candidates": candidates,
            "selected_project": selected,
            "selected_repos": selected.get("repos", []),
            "is_new_project": False,
            "status": "PROJECT_SELECTED"
        }

    # No good match — create NEW project
    slug = slugify(requirement)
    print(f"  🆕 NEW PROJECT — slug: {slug}")

    new_repos = [
        {
            "name": f"{slug}-backend",
            "type": "backend",
            "language": "python",
            "url": f"https://github.com/AkashW45/{slug}-backend.git",
            "exists": False
        },
        {
            "name": f"{slug}-frontend",
            "type": "frontend",
            "language": "typescript",
            "url": f"https://github.com/AkashW45/{slug}-frontend.git",
            "exists": False
        }
    ]

    new_project = {
        "id": f"new-{slug}",
        "name": requirement[:60],
        "description": requirement,
        "repos": new_repos,
        "is_new": True,
        "score": 0.0
    }

    return {
        **state,
        "candidates": candidates,
        "selected_project": new_project,
        "selected_repos": new_repos,
        "is_new_project": True,
        "status": "NEW_PROJECT_CREATED"
    }


def human_gate_node(state: SelectorState) -> SelectorState:
    interrupt({
        "type": "PROJECT_SELECTION",
        "selected_project": state["selected_project"],
        "selected_repos": state["selected_repos"],
        "is_new_project": state.get("is_new_project", False),
        "candidates": state.get("candidates", [])
    })
    return state


def process_approval(state: SelectorState) -> SelectorState:
    human_input = state.get("human_input", {})
    if not human_input.get("approved"):
        return {**state, "status": "REJECTED"}
    return {**state, "status": "APPROVED"}


def build_selector_graph():
    g = StateGraph(SelectorState)
    g.add_node("select_project", select_project_node)
    g.add_node("human_gate", human_gate_node)
    g.add_node("process_approval", process_approval)

    g.set_entry_point("select_project")
    g.add_edge("select_project", "human_gate")
    g.add_edge("human_gate", "process_approval")
    g.add_edge("process_approval", END)

    return g.compile(checkpointer=MemorySaver())