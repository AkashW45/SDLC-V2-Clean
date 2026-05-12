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
from agents.prompts.system_prompts import SPRINT_PLANNER_SYSTEM
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from api.jira_client import fetch_jira_metadata
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# -----------------------------------------
# State
# -----------------------------------------

class PlanningState(TypedDict):
    requirement: str
    scope_contract: dict      # <-- ADD THIS LINE
    brd: dict
    prd: dict
    sprint_plan: dict
    jira_tickets: list
    runbook: dict
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# LLM Helper
# -----------------------------------------

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
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

    user_msg = json.dumps({
        "scope_contract": state["scope_contract"],
        "brd": state["brd"],
        "prd": state["prd"]
    }, indent=2)

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": SPRINT_PLANNER_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
    try:
        sprint_plan = json.loads(content)
    except Exception:
        sprint_plan = {"epics": []}

    epics = sprint_plan.get("epics", [])
    total_stories = sum(len(e.get("stories", [])) for e in epics)

    print(f"  ✅ Sprint plan generated: {len(epics)} epics, {total_stories} stories")
    return {**state, "sprint_plan": sprint_plan, "status": "SPRINT_PLAN_GENERATED"}

def create_jira_tickets(state: PlanningState) -> PlanningState:
    """Create Jira epics and stories from sprint plan."""
    print("\n[Phase 2] Creating Jira tickets...")

    try:
        import os
        import base64
        import httpx

        from api.jira_client import fetch_jira_metadata

        project_key = os.getenv("JIRA_PROJECT_KEY", "DEV")
        jira_meta = fetch_jira_metadata(project_key)

        email = os.getenv("JIRA_EMAIL")
        token = os.getenv("JIRA_API_TOKEN")
        domain = os.getenv("JIRA_BASE_URL")
        credentials = base64.b64encode(f"{email}:{token}".encode()).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        epics = state["sprint_plan"].get("epics", [])
        created_tickets = []

        for epic in epics:
            # Create Epic
            try:
                epic_resp = httpx.post(
                    f"https://{domain}/rest/api/3/issue",
                    headers=headers,
                    json={"fields": {
                        "project": {"key": project_key},
                        "summary": epic["title"],
                        "description": {
                            "type": "doc", "version": 1,
                            "content": [{"type": "paragraph", "content": [
                                {"type": "text", "text": epic.get("description", "")}
                            ]}]
                        },
                        "issuetype": {"id": jira_meta["issue_types"].get("Epic", "")},
                        "priority": {"id": jira_meta["priorities"].get("Medium", "")},
                        "labels": ["ai-generated", "sdlc-v2"]
                    }},
                    timeout=15
                )
                if epic_resp.status_code == 201:
                    epic_key = epic_resp.json()["key"]
                    created_tickets.append(epic_key)
                    print(f"  ✅ Epic: {epic_key} — {epic['title']}")
                else:
                    print(f"  ⚠️  Epic failed: {epic_resp.text[:100]}")
                    continue

            except Exception as e:
                print(f"  ⚠️  Epic error: {e}")
                continue

            # Create Stories under Epic
            for story in epic.get("stories", []):
                try:
                    ac_text = "\n".join(
                        f"- {ac}" for ac in story.get("acceptance_criteria", [])
                    )
                    full_desc = (
                        f"{story.get('description', '')}"
                        f"\n\nAcceptance Criteria:\n{ac_text}"
                    )

                    story_resp = httpx.post(
                        f"https://{domain}/rest/api/3/issue",
                        headers=headers,
                        json={"fields": {
                            "project": {"key": project_key},
                            "summary": story["title"],
                            "description": {
                                "type": "doc", "version": 1,
                                "content": [{"type": "paragraph", "content": [
                                    {"type": "text", "text": full_desc}
                                ]}]
                            },
                            "issuetype": {"id": jira_meta["issue_types"].get("Story", "")},
                            "priority": {"id": jira_meta["priorities"].get("Medium", "")},
                            "labels": ["ai-generated", "sdlc-v2"]
                        }},
                        timeout=15
                    )

                    if story_resp.status_code == 201:
                        story_key = story_resp.json()["key"]
                        created_tickets.append(story_key)
                        print(f"    ✅ Story: {story_key} — {story['title']}")
                    else:
                        print(f"    ⚠️  Story failed: {story_resp.text[:100]}")

                except Exception as e:
                    print(f"    ⚠️  Story error: {e}")

        print(f"\n  ✅ Total tickets created: {len(created_tickets)}")
        return {
            **state,
            "jira_tickets": created_tickets,
            "status": "JIRA_TICKETS_CREATED"
        }

    except Exception as e:
        print(f"  ⚠️  Jira creation skipped: {e}")
        return {**state, "jira_tickets": [], "status": "JIRA_SKIPPED"}
    
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
    builder.add_node("create_jira_tickets", create_jira_tickets)
    builder.add_node("generate_runbook", generate_runbook)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("generate_sprint_plan")
    builder.add_edge("generate_sprint_plan", "create_jira_tickets")
    builder.add_edge("create_jira_tickets", "generate_runbook")
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