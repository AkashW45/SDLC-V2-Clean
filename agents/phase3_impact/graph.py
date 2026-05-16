"""
Phase 3 — Impact Analysis LangGraph Agent
Uses INTERRUPT for human approval before code generation begins.

Optimizations vs original:
  - Removed duplicate resume_impact_analysis function (dead code; resume_with_approval
    is the canonical path used by the API)
  - Singleton graph built exactly once via get_graph() — unchanged, already correct
  - Docstrings tightened; no logic changes to the graph itself
"""

import json
import sys
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.phase3_impact.impact_analyzer import run_impact_analysis

# -----------------------------------------
# SHARED memory — module-level so start and
# resume operate on the SAME MemorySaver instance
# -----------------------------------------
_memory = MemorySaver()
_graph = None  # built once, reused


# -----------------------------------------
# State Definition
# -----------------------------------------

class ImpactState(TypedDict):
    requirement: str
    impact_report: dict
    adr: dict
    prd: dict                    # NEW
    selected_repos: list         # NEW
    scope_contract: dict         # NEW
    human_approved: bool
    human_feedback: str
    status: str


# -----------------------------------------
# Nodes
# -----------------------------------------

def analyze_impact(state: ImpactState) -> ImpactState:
    print("[Node] analyze_impact — running...")
    report = run_impact_analysis(
        state["requirement"],
        prd=state.get("prd", {}),
        adr=state.get("adr", {}),
        selected_repos=state.get("selected_repos", []),
    )
    return {
        **state,
        "impact_report": report,
        "status": "IMPACT_ANALYZED",
    }


def human_approval_gate(state: ImpactState) -> ImpactState:
    print("[Node] human_approval_gate — INTERRUPTED, waiting for human...")

    human_input = interrupt("Waiting for human approval of impact report")

    approved = False
    feedback = ""

    if isinstance(human_input, dict):
        approved = human_input.get("approved") or human_input.get("human_approved") or False
        feedback = human_input.get("feedback", "")

    return {
        **state,
        "human_approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_APPROVAL",
    }


def process_approval(state: ImpactState) -> ImpactState:
    print(f"[Node] process_approval — human_approved={state.get('human_approved')}")
    if state.get("human_approved"):
        return {**state, "status": "APPROVED_FOR_CODE_GENERATION"}
    return {
        **state,
        "status": "REJECTED_BY_HUMAN",
        "impact_report": {
            **state["impact_report"],
            "rejection_reason": state.get("human_feedback", "No reason given"),
        },
    }


def route_after_approval(state: ImpactState) -> str:
    return "approved" if state.get("human_approved") else "rejected"


# -----------------------------------------
# Build Graph ONCE
# -----------------------------------------

def get_graph():
    """Return the singleton compiled graph (builds on first call)."""
    global _graph
    if _graph is not None:
        return _graph

    builder = StateGraph(ImpactState)

    builder.add_node("analyze_impact", analyze_impact)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("analyze_impact")
    builder.add_edge("analyze_impact", "human_approval_gate")
    builder.add_edge("human_approval_gate", "process_approval")

    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {"approved": END, "rejected": END},
    )

    _graph = builder.compile(
        checkpointer=_memory,
        interrupt_before=["human_approval_gate"],
    )

    return _graph


# -----------------------------------------
# FastAPI / CLI Helpers
# -----------------------------------------

def start_impact_analysis(requirement: str, thread_id: str) -> dict:
    """Start a new impact-analysis run and block at the human-approval gate."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: ImpactState = {
        "requirement": requirement,
        "impact_report": {},
        "human_approved": False,
        "human_feedback": "",
        "status": "STARTED",
    }

    result = graph.invoke(initial_state, config)

    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "impact_report": result.get("impact_report"),
        "message": "Impact analysis complete. Awaiting human approval.",
    }


def resume_with_approval(thread_id: str, approved: bool, feedback: str = "") -> dict:
    """Resume an interrupted graph run after the human has reviewed the report."""
    graph = get_graph()  # same graph, same _memory
    config = {"configurable": {"thread_id": thread_id}}

    # Inject human decision into the existing checkpoint
    graph.update_state(
        config,
        {"human_approved": approved, "human_feedback": feedback},
    )

    # Resume from the INTERRUPT point
    result = graph.invoke(None, config)

    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "approved": approved,
        "message": (
            "Approved — ready for code generation"
            if approved
            else "Rejected — pipeline stopped"
        ),
    }


# -----------------------------------------
# CLI Test
# -----------------------------------------

if __name__ == "__main__":
    import uuid

    thread_id = str(uuid.uuid4())
    requirement = "Add leave balance tracker. Each employee gets 20 days per year."

    print("\n--- Starting Phase 3 ---")
    result = start_impact_analysis(requirement, thread_id)
    print(f"\nStatus: {result['status']}")
    print(f"Risk: {result['impact_report'].get('risk_assessment', {}).get('risk_level')}")

    print("\n--- Simulating Human Approval ---")
    final = resume_with_approval(thread_id, approved=True, feedback="Looks good")
    print(f"Final status: {final['status']}")
    print(f"Message: {final['message']}")
