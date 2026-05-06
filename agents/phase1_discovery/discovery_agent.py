"""
Phase 1 — Discovery Agent
Generates BRD, PRD, ADR from raw business requirement.
Human approval INTERRUPT after PRD generated.
"""

import os
import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from groq import Groq
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------------------
# State
# -----------------------------------------

class DiscoveryState(TypedDict):
    requirement: str
    brd: dict
    prd: dict
    adr: dict
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# LLM Helper
# -----------------------------------------

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}


# -----------------------------------------
# Nodes
# -----------------------------------------

def generate_brd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating BRD...")

    brd = call_llm(f"""
You are a senior Business Analyst.
Generate a complete BRD for this requirement.
Return ONLY valid JSON with these fields:
{{
  "title": "...",
  "business_objectives": ["..."],
  "scope": {{"in_scope": ["..."], "out_of_scope": ["..."]}},
  "functional_requirements": [{{"id": "FR1", "title": "...", "description": "...", "priority": "High"}}],
  "non_functional_requirements": [{{"id": "NFR1", "title": "...", "description": "..."}}],
  "stakeholders": ["..."],
  "assumptions": ["..."],
  "risks": ["..."]
}}

Requirement: {state['requirement']}
""")

    print(f"  ✅ BRD generated: {brd.get('title', 'untitled')}")
    return {**state, "brd": brd, "status": "BRD_GENERATED"}


def generate_prd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating PRD...")

    prd = call_llm(f"""
You are a Product Owner.
Generate a PRD from this BRD.
Return ONLY valid JSON with these exact fields:
{{
  "title": "...",
  "product_vision": "...",
  "functional_requirements": [
    {{
      "id": "FR1",
      "title": "...",
      "description": "...",
      "priority": "High",
      "acceptance_criteria": ["..."]
    }}
  ],
  "non_functional_requirements": [
    {{
      "id": "NFR1",
      "title": "...",
      "description": "...",
      "priority": "Medium"
    }}
  ],
  "stakeholders": ["..."],
  "scope": "..."
}}

The "title" field is mandatory. Do not rename it.

BRD:
{json.dumps(state['brd'], indent=2)}
""")

    # Handle different title keys LLM might return
    if not prd.get("title"):
        prd["title"] = (
            prd.get("projectTitle") or
            prd.get("product_title") or
            prd.get("name") or
            state['brd'].get('title', 'Untitled PRD')
        )

    print(f"  ✅ PRD generated: {prd.get('title', 'untitled')}")
    return {**state, "prd": prd, "status": "PRD_GENERATED"}


def generate_adr(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating ADR...")

    adr = call_llm(f"""
You are a senior software architect.
Generate Architecture Decision Records for this requirement.
Return ONLY valid JSON:
{{
  "decisions": [
    {{
      "id": "ADR-001",
      "title": "...",
      "status": "Accepted",
      "context": "...",
      "decision": "...",
      "consequences": ["..."],
      "alternatives_considered": ["..."]
    }}
  ]
}}

Requirement: {state['requirement']}
PRD Summary: {state['prd'].get('product_vision', '')}
""")

    print(f"  ✅ ADR generated: {len(adr.get('decisions', []))} decisions")
    return {**state, "adr": adr, "status": "ADR_GENERATED"}


def human_approval_gate(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] ⏸ Waiting for human approval...")
    print(f"  BRD: {state['brd'].get('title', '')}")
    print(f"  PRD: {state['prd'].get('title', '') or state['prd'].get('projectTitle', '')}")
    print(f"  ADR decisions: {len(state['adr'].get('decisions', []))}")

    # INTERRUPT — human responds with approved True/False
    human_input = interrupt("Waiting for human approval")

    # human_input contains what was passed in resume
    approved = human_input.get("approved", False) if isinstance(human_input, dict) else False
    feedback = human_input.get("feedback", "") if isinstance(human_input, dict) else ""

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_APPROVAL"
    }


def process_approval(state: DiscoveryState) -> DiscoveryState:
    approved = state.get("approved", False)
    feedback = state.get("human_feedback", "")

    if approved:
        print(f"\n[Phase 1] ✅ Approved — moving to Phase 2")
        return {**state, "status": "APPROVED_FOR_PLANNING"}
    else:
        print(f"\n[Phase 1] ❌ Rejected — {feedback}")
        return {**state, "status": "REJECTED"}


# -----------------------------------------
# Routing
# -----------------------------------------

def route_after_approval(state: DiscoveryState) -> str:
    if state["status"] == "APPROVED_FOR_PLANNING":
        return "approved"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_discovery_graph():
    builder = StateGraph(DiscoveryState)

    builder.add_node("generate_brd", generate_brd)
    builder.add_node("generate_prd", generate_prd)
    builder.add_node("generate_adr", generate_adr)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("generate_brd")
    builder.add_edge("generate_brd", "generate_prd")
    builder.add_edge("generate_prd", "generate_adr")
    builder.add_edge("generate_adr", "human_approval_gate")
    builder.add_edge("human_approval_gate", "process_approval")

    builder.add_conditional_edges(
        "process_approval",
        route_after_approval,
        {
            "approved": END,
            "rejected": END
        }
    )

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]  # pause BEFORE approval node
    )


# -----------------------------------------
# Run
# -----------------------------------------

def start_discovery(requirement: str, thread_id: str = "thread-1"):
    graph = build_discovery_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = DiscoveryState(
        requirement=requirement,
        brd={},
        prd={},
        adr={},
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 1 — Discovery ---")
    print("="*50)

    # Run until INTERRUPT
    result = graph.invoke(initial_state, config)
    print(f"\nStatus after interrupt: {result['status']}")
    print(f"BRD: {result['brd'].get('title', '')}")
    print(f"PRD: {result['prd'].get('title', '')}")

    return graph, config, result


def resume_discovery(graph, config, approved: bool, feedback: str = ""):
    print(f"\n--- Resuming Phase 1 (approved={approved}) ---")

    result = graph.invoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config
    )

    print(f"Final status: {result['status']}")
    return result


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":
    requirement = "Add leave balance tracker to Leave Management System. Each employee gets 20 days per year. Balance decreases when leave is approved."

    graph, config, result = start_discovery(requirement)

    print(f"\nPaused at: {result['status']}")
    print(f"BRD: {result['brd'].get('title')}")
    print(f"PRD requirements: {len(result['prd'].get('functional_requirements', []))}")
    print(f"ADR decisions: {len(result['adr'].get('decisions', []))}")

    print("\n--- Simulating Human Approval ---")
    final = resume_discovery(graph, config, approved=True, feedback="Looks good")

    print(f"\n✅ Phase 1 Complete")
    print(f"Final status: {final['status']}")