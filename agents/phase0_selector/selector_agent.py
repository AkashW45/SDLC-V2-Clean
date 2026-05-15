import os
import sys
import re


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import TypedDict, List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
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
    return QdrantClient(url="http://127.0.0.1:6333", timeout=60)


def get_postgres():
    return psycopg2.connect(
        host="127.0.0.1", port=5433,
        user="sdlc", password="sdlc1234",
        dbname="sdlc_knowledge"
    )


_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def search_projects(requirement: str, top_k: int = 3):
    qdrant = get_qdrant()
    embedder = get_embedder()

    try:
        vector = embedder.encode(requirement).tolist()
        results = qdrant.query_points(
            collection_name="project_embeddings",
            query=vector,
            limit=top_k,
            with_payload=True
        )

        candidates = []
        for r in results.points:
            payload = r.payload or {}
            candidates.append({
                "id": payload.get("project_id", ""),
                "name": payload.get("name", ""),
                "description": payload.get("description", ""),
                "repos": payload.get("repos", []),
                "score": round(r.score, 4)
            })
        return candidates
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

    if candidates and candidates[0]["score"] >= 0.4:
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