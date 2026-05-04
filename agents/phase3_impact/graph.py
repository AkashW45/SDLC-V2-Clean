"""
Phase 3 — Impact Analysis LangGraph Agent
Uses INTERRUPT for human approval before code generation begins.
"""


import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.phase3_impact.impact_analyzer import run_impact_analysis
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# -----------------------------------------
# State Definition
# -----------------------------------------

class ImpactState(TypedDict):
    requirement: str
    impact_report: dict
    human_approved: bool
    human_feedback: str
    status: str


# -----------------------------------------
# Nodes
# -----------------------------------------

def analyze_impact(state: ImpactState) -> ImpactState:
    """Run full impact analysis and store report in state."""
    print("[Node] analyze_impact — running...")

    report = run_impact_analysis(state["requirement"])

    return {
        **state,
        "impact_report": report,
        "status": "IMPACT_ANALYZED"
    }


def human_approval_gate(state: ImpactState) -> ImpactState:
    """
    LangGraph INTERRUPT node.
    Pipeline pauses here — human reviews impact report and approves/rejects.
    Human response is injected via graph.update_state() when they respond.
    """
    print("[Node] human_approval_gate — INTERRUPTED, waiting for human...")

    # This node just marks the state as waiting
    # LangGraph INTERRUPT will pause execution here
    return {
        **state,
        "status": "AWAITING_HUMAN_APPROVAL"
    }


def process_approval(state: ImpactState) -> ImpactState:
    """Process human decision after INTERRUPT resumes."""
    print(f"[Node] process_approval — human_approved={state.get('human_approved')}")

    if state.get("human_approved"):
        return {
            **state,
            "status": "APPROVED_FOR_CODE_GENERATION"
        }
    else:
        return {
            **state,
            "status": "REJECTED_BY_HUMAN",
            "impact_report": {
                **state["impact_report"],
                "rejection_reason": state.get("human_feedback", "No reason given")
            }
        }


def route_after_approval(state: ImpactState) -> str:
    """Route based on human decision."""
    if state.get("human_approved"):
        return "approved"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_phase3_graph():
    from langgraph.types import interrupt

    builder = StateGraph(ImpactState)

    # Add nodes
    builder.add_node("analyze_impact", analyze_impact)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    # Add edges
    builder.set_entry_point("analyze_impact")
    builder.add_edge("analyze_impact", "human_approval_gate")

    # INTERRUPT at human_approval_gate
    builder.add_edge("human_approval_gate", "process_approval")

    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {
            "approved": END,
            "rejected": END
        }
    )

    # Compile with memory checkpointer for INTERRUPT support
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]
    )

    return graph


# -----------------------------------------
# FastAPI Endpoint Helpers
# -----------------------------------------

def start_impact_analysis(requirement: str, thread_id: str) -> dict:
    """
    Start Phase 3 analysis.
    Returns impact report and pauses at human approval gate.
    """
    graph = build_phase3_graph()

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "requirement": requirement,
        "impact_report": {},
        "human_approved": False,
        "human_feedback": "",
        "status": "STARTED"
    }

    # Run until INTERRUPT
    result = graph.invoke(initial_state, config)

    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "impact_report": result.get("impact_report"),
        "message": "Impact analysis complete. Awaiting human approval."
    }


def resume_with_approval(thread_id: str, approved: bool, feedback: str = "") -> dict:
    """
    Resume Phase 3 after human approves or rejects.
    Called from dashboard when human clicks Approve/Reject.
    """
    graph = build_phase3_graph()

    config = {"configurable": {"thread_id": thread_id}}

    # Inject human decision into state
    graph.update_state(
        config,
        {
            "human_approved": approved,
            "human_feedback": feedback
        }
    )

    # Resume from INTERRUPT
    result = graph.invoke(None, config)

    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "approved": approved,
        "message": "Approved — ready for code generation" if approved else "Rejected — pipeline stopped"
    }


# -----------------------------------------
# Test
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