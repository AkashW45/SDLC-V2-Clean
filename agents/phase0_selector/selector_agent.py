"""
Phase 0 — Project & Repo Selector
Finds the right project and repos for a given requirement.
Human confirms the selection before pipeline continues.
"""

"""
Phase 0 — Project & Repo Selector
Finds the right project and repos for a given requirement.
Human confirms the selection before pipeline continues.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, r"C:\Users\user\SDLC-V2")

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------
# State
# -----------------------------------------

class SelectorState(TypedDict):
    requirement: str
    candidate_projects: list
    selected_project: dict
    selected_repos: list
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# Nodes
# -----------------------------------------

def search_projects(state: SelectorState) -> SelectorState:
    """Semantic search to find top candidate projects."""
    print("\n[Phase 0] Searching for matching projects...")
    import sys
    sys.path.insert(0, r"C:\Users\user\SDLC-V2")

    from knowledge_layer.project_registry import search_projects as do_search

    candidates = do_search(state["requirement"], top_k=3)

    print(f"  Found {len(candidates)} candidate projects:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c['project_name']} (score: {c['score']})")
        print(f"     Repos: {', '.join(c['repos'])}")

    return {
        **state,
        "candidate_projects": candidates,
        "status": "PROJECTS_FOUND"
    }


def auto_select_project(state: SelectorState) -> SelectorState:
    """
    Auto-select top scoring project if score is high enough.
    If top score > 0.4 — auto select.
    If top score < 0.4 — flag for human to decide.
    """
    candidates = state["candidate_projects"]

    if not candidates:
        print("  ⚠️  No projects found — human must specify")
        return {**state, "status": "NO_PROJECT_FOUND"}

    top = candidates[0]

    if top["score"] >= 0.4:
        print(f"\n[Phase 0] Auto-selected: {top['project_name']} (score: {top['score']})")
        return {
            **state,
            "selected_project": top,
            "selected_repos": top["repos"],
            "status": "PROJECT_AUTO_SELECTED"
        }
    else:
        print(f"\n[Phase 0] Low confidence ({top['score']}) — human selection needed")
        return {
            **state,
            "selected_project": top,
            "selected_repos": top["repos"],
            "status": "PROJECT_NEEDS_CONFIRMATION"
        }


def human_approval_gate(state: SelectorState) -> SelectorState:
    """Human confirms project and repo selection."""
    print("\n[Phase 0] ⏸ Waiting for project selection confirmation...")

    candidates = state["candidate_projects"]
    selected = state["selected_project"]

    print(f"\n  Top match: {selected['project_name']}")
    print(f"  Repos: {', '.join(selected['repos'])}")
    print(f"  Score: {selected['score']}")

    human_input = interrupt({
        "message": "Confirm project selection",
        "candidates": candidates,
        "auto_selected": selected,
        "stage": "phase0_project_selection"
    })

    approved = False
    feedback = ""
    override_project_id = None

    if isinstance(human_input, dict):
        approved = human_input.get("approved", False)
        feedback = human_input.get("feedback", "")
        override_project_id = human_input.get("project_id")

    # Handle override — human picked a different project
    if approved and override_project_id:
        override = next(
            (c for c in candidates if c["project_id"] == override_project_id),
            None
        )
        if override:
            return {
                **state,
                "selected_project": override,
                "selected_repos": override["repos"],
                "approved": True,
                "human_feedback": feedback,
                "status": "PROJECT_CONFIRMED"
            }

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_CONFIRMATION"
    }


def process_approval(state: SelectorState) -> SelectorState:
    if state.get("approved"):
        project = state["selected_project"]
        print(f"\n[Phase 0] ✅ Project confirmed: {project['project_name']}")
        print(f"  Repos in scope: {', '.join(state['selected_repos'])}")
        return {**state, "status": "PROJECT_CONFIRMED"}
    else:
        print(f"\n[Phase 0] ❌ Rejected: {state.get('human_feedback')}")
        return {**state, "status": "REJECTED"}


# -----------------------------------------
# Routing
# -----------------------------------------

def route_after_auto_select(state: SelectorState) -> str:
    if state["status"] == "PROJECT_AUTO_SELECTED":
        return "high_confidence"
    elif state["status"] == "NO_PROJECT_FOUND":
        return "no_match"
    return "low_confidence"


def route_after_approval(state: SelectorState) -> str:
    if state["status"] == "PROJECT_CONFIRMED":
        return "confirmed"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_selector_graph():
    builder = StateGraph(SelectorState)

    builder.add_node("search_projects", search_projects)
    builder.add_node("auto_select_project", auto_select_project)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("search_projects")
    builder.add_edge("search_projects", "auto_select_project")

    builder.add_conditional_edges(
        "auto_select_project",
        route_after_auto_select,
        {
            "high_confidence": "human_approval_gate",
            "low_confidence": "human_approval_gate",
            "no_match": END
        }
    )

    builder.add_edge("human_approval_gate", "process_approval")

    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {
            "confirmed": END,
            "rejected": END
        }
    )

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]
    )


# -----------------------------------------
# Run
# -----------------------------------------

def run_selector(requirement: str, thread_id: str = "thread-selector"):
    graph = build_selector_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = SelectorState(
        requirement=requirement,
        candidate_projects=[],
        selected_project={},
        selected_repos=[],
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Phase 0 — Project Selector ---")
    print("="*50)

    result = graph.invoke(initial_state, config)
    return graph, config, result


def resume_selector(graph, config, approved: bool,
                    feedback: str = "", project_id: str = None):
    payload = {"approved": approved, "feedback": feedback}
    if project_id:
        payload["project_id"] = project_id

    result = graph.invoke(Command(resume=payload), config)
    return result


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":
    req = "Add leave balance tracker. Each employee gets 20 days per year."

    graph, config, result = run_selector(req, "test-selector-1")

    print(f"\nPaused at: {result['status']}")
    print(f"Auto-selected: {result['selected_project'].get('project_name')}")
    print(f"Repos: {result['selected_repos']}")
    print(f"\nAll candidates:")
    for c in result['candidate_projects']:
        print(f"  {c['project_name']} — {c['score']} — {c['repos']}")

    print("\n--- Simulating Human Confirmation ---")
    final = resume_selector(graph, config, approved=True)

    print(f"\n✅ Phase 0 Complete")
    print(f"Status: {final['status']}")
    print(f"Project: {final['selected_project']['project_name']}")
    print(f"Repos: {final['selected_repos']}")