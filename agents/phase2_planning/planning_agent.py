"""
Phase 2 — Planning Agent
Generates Jira epics/stories and deployment runbook from approved PRD.
Human approval INTERRUPT after sprint plan generated.
"""

import os
import json
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------------------
# State
# -----------------------------------------

class PlanningState(TypedDict):
    requirement: str
    brd: dict
    prd: dict
    sprint_plan: dict
    runbook: dict
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

def generate_sprint_plan(state: PlanningState) -> PlanningState:
    print("\n[Phase 2] Generating sprint plan...")

    prd = state["prd"]
    requirements = prd.get("functional_requirements", [])

    sprint_plan = call_llm(f"""
You are a senior Scrum Master and Product Owner.
Generate a sprint plan from this PRD.
Return ONLY valid JSON:
{{
  "project": "LEAVE-MGMT",
  "sprint_duration": "2 weeks",
  "epics": [
    {{
      "epic_id": "EP-001",
      "title": "...",
      "description": "...",
      "business_goal": "...",
      "affected_repos": ["leave-mgmt-backend"],
      "risk_level": "medium",
      "stories": [
        {{
          "story_id": "US-001",
          "title": "...",
          "description": "As a [user] I want [goal] so that [benefit]",
          "acceptance_criteria": ["...", "..."],
          "story_points": 3,
          "affected_repo": "leave-mgmt-backend",
          "labels": ["backend"]
        }}
      ]
    }}
  ]
}}

PRD functional requirements:
{json.dumps(requirements, indent=2)}

Requirement: {state['requirement']}
""")

    epics = sprint_plan.get("epics", [])
    total_stories = sum(len(e.get("stories", [])) for e in epics)

    print(f"  ✅ Sprint plan generated: {len(epics)} epics, {total_stories} stories")
    return {**state, "sprint_plan": sprint_plan, "status": "SPRINT_PLAN_GENERATED"}


def generate_runbook(state: PlanningState) -> PlanningState:
    print("\n[Phase 2] Generating runbook...")

    runbook = call_llm(f"""
You are a senior DevOps engineer.
Generate a deployment runbook for this feature.
Return ONLY valid JSON:
{{
  "feature": "...",
  "version": "1.0.0",
  "pre_deployment_checklist": ["...", "..."],
  "deployment_sequence": [
    {{
      "step": 1,
      "repo": "leave-mgmt-backend",
      "action": "...",
      "command": "...",
      "rollback_command": "..."
    }}
  ],
  "feature_flags": [
    {{
      "flag_name": "...",
      "default": false,
      "enable_after_deploy": true
    }}
  ],
  "smoke_test_checklist": ["...", "..."],
  "rollback_decision_criteria": ["...", "..."],
  "rollback_steps": ["...", "..."],
  "on_call_escalation": {{
    "primary": "team-lead",
    "secondary": "platform-team",
    "slack_channel": "#deployments"
  }}
}}

Feature: {state['requirement']}
Sprint plan epics: {json.dumps([e['title'] for e in state['sprint_plan'].get('epics', [])], indent=2)}
""")

    print(f"  ✅ Runbook generated: {len(runbook.get('deployment_sequence', []))} deployment steps")
    return {**state, "runbook": runbook, "status": "RUNBOOK_GENERATED"}


def human_approval_gate(state: PlanningState) -> PlanningState:
    print("\n[Phase 2] ⏸ Waiting for human approval...")

    epics = state['sprint_plan'].get('epics', [])
    total_stories = sum(len(e.get('stories', [])) for e in epics)

    print(f"  Epics: {len(epics)}")
    print(f"  Stories: {total_stories}")
    print(f"  Deployment steps: {len(state['runbook'].get('deployment_sequence', []))}")

    human_input = interrupt("Waiting for human approval of sprint plan and runbook")

    approved = human_input.get("approved", False) if isinstance(human_input, dict) else False
    feedback = human_input.get("feedback", "") if isinstance(human_input, dict) else ""

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_APPROVAL"
    }


def process_approval(state: PlanningState) -> PlanningState:
    approved = state.get("approved", False)
    feedback = state.get("human_feedback", "")

    if approved:
        print(f"\n[Phase 2] ✅ Approved — moving to Impact Analysis")
        return {**state, "status": "APPROVED_FOR_IMPACT_ANALYSIS"}
    else:
        print(f"\n[Phase 2] ❌ Rejected — {feedback}")
        return {**state, "status": "REJECTED"}


# -----------------------------------------
# Routing
# -----------------------------------------

def route_after_approval(state: PlanningState) -> str:
    if state["status"] == "APPROVED_FOR_IMPACT_ANALYSIS":
        return "approved"
    return "rejected"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_planning_graph():
    builder = StateGraph(PlanningState)

    builder.add_node("generate_sprint_plan", generate_sprint_plan)
    builder.add_node("generate_runbook", generate_runbook)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("generate_sprint_plan")
    builder.add_edge("generate_sprint_plan", "generate_runbook")
    builder.add_edge("generate_runbook", "human_approval_gate")
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
        interrupt_before=["human_approval_gate"]
    )


# -----------------------------------------
# Run
# -----------------------------------------

def start_planning(requirement: str, brd: dict, prd: dict, thread_id: str = "thread-1"):
    graph = build_planning_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = PlanningState(
        requirement=requirement,
        brd=brd,
        prd=prd,
        sprint_plan={},
        runbook={},
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 2 — Planning ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\nStatus after interrupt: {result['status']}")
    print(f"Epics: {len(result['sprint_plan'].get('epics', []))}")
    print(f"Runbook steps: {len(result['runbook'].get('deployment_sequence', []))}")

    return graph, config, result


def resume_planning(graph, config, approved: bool, feedback: str = ""):
    print(f"\n--- Resuming Phase 2 (approved={approved}) ---")

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
    requirement = "Add leave balance tracker. Each employee gets 20 days per year."

    # Simulate approved Phase 1 output
    mock_brd = {"title": "Leave Balance Tracker", "functional_requirements": []}
    mock_prd = {
        "title": "Leave Balance Tracker",
        "product_vision": "Track employee leave balances automatically",
        "functional_requirements": [
            {
                "id": "FR1",
                "title": "Balance Tracking",
                "description": "Track leave balance per employee",
                "priority": "High",
                "acceptance_criteria": ["Balance decreases on approval"]
            }
        ]
    }

    graph, config, result = start_planning(
        requirement, mock_brd, mock_prd, "thread-phase2-1"
    )

    print("\n--- Simulating Human Approval ---")
    final = resume_planning(graph, config, approved=True)

    print(f"\n✅ Phase 2 Complete")
    print(f"Status: {final['status']}")
    print(f"Epics: {len(final['sprint_plan'].get('epics', []))}")
    print(f"Runbook steps: {len(final['runbook'].get('deployment_sequence', []))}")