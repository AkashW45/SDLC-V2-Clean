"""
Phase 2 — Planning Agent
Generates Jira epics/stories and deployment runbook from approved PRD.
Human approval INTERRUPT after sprint plan generated.
"""

import concurrent.futures
import os
import json
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from groq import Groq
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from api.jira_client import fetch_jira_metadata
from api.persistence import save_artifact

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
    thread_id: str


def _get_thread_id(state) -> str:
    if isinstance(state, dict):
        thread_id = state.get("thread_id")
        if thread_id:
            return thread_id
        config = state.get("config") or state.get("__config__") or state.get("_config") or {}
        if isinstance(config, dict):
            return config.get("thread_id") or config.get("configurable", {}).get("thread_id")
    return "unknown-thread"


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
    from agents.prompts.system_prompts import SPRINT_PLANNER_SYSTEM

    scope_contract = state.get("scope_contract", {})
    prd = state["prd"]
    selected_repos = state.get("selected_repos", [])

    prompt = f"""{SPRINT_PLANNER_SYSTEM}

SCOPE CONTRACT:
{json.dumps(scope_contract, indent=2)}

PRD:
{json.dumps(prd, indent=2)}

TARGET REPOSITORIES (use real repo names in affected_repos):
{json.dumps(selected_repos, indent=2)}

USER REQUIREMENT: {state['requirement']}

Generate the sprint plan and Jira tickets per the ABSOLUTE RULES above.
"""

    sprint_plan = call_llm(prompt)

    # SPRINT_PLANNER_SYSTEM returns the data wrapped in `body` — unwrap it
    # to keep backward compatibility with the rest of the planning_agent code.
    if isinstance(sprint_plan, dict) and "body" in sprint_plan:
        body = sprint_plan["body"]
        # Convert ASP shape (sprint_plan + jira_tickets) → legacy shape (epics + stories)
        if "sprint_plan" in body or "jira_tickets" in body:
            tickets = body.get("jira_tickets", [])
            sprints = body.get("sprint_plan", {}).get("sprints", [])

            # Group tickets by epic if hierarchy exists, else flatten as single epic
            epics_map = {}
            standalone = []
            for t in tickets:
                if t.get("type") == "Epic":
                    epics_map[t["id"]] = {
                        "epic_id": t["id"],
                        "title": t.get("summary", ""),
                        "description": t.get("description", ""),
                        "business_goal": t.get("description", "")[:200],
                        "affected_repos": [r.get("name") for r in selected_repos if isinstance(r, dict)],
                        "risk_level": "medium",
                        "stories": [],
                    }
                elif t.get("type") in ("Story", "Task", "Subtask"):
                    standalone.append(t)

            for t in standalone:
                story = {
                    "story_id": t["id"],
                    "title": t.get("summary", ""),
                    "description": t.get("description", ""),
                    "acceptance_criteria": t.get("acceptance_criteria", []),
                    "story_points": t.get("story_points", 3),
                    "affected_repo": (selected_repos[0].get("name") if selected_repos
                                      and isinstance(selected_repos[0], dict) else ""),
                    "labels": t.get("labels", []),
                    "priority": t.get("priority", "P2"),
                    "depends_on": t.get("depends_on", []),
                    "traces_to_prd": t.get("traces_to_prd", ""),
                    "sprint": t.get("sprint", 1),
                }
                parent = t.get("parent_ticket")
                if parent and parent in epics_map:
                    epics_map[parent]["stories"].append(story)
                else:
                    # No parent Epic — create a default one if none exist
                    if not epics_map:
                        epics_map["EP-001"] = {
                            "epic_id": "EP-001",
                            "title": prd.get("title", "Main Epic"),
                            "description": "Auto-generated default epic",
                            "business_goal": "",
                            "affected_repos": [r.get("name") for r in selected_repos
                                               if isinstance(r, dict)],
                            "risk_level": "medium",
                            "stories": [],
                        }
                    list(epics_map.values())[0]["stories"].append(story)

            sprint_plan = {
                "project": prd.get("title", "PROJECT"),
                "sprint_duration": f"{body.get('sprint_plan', {}).get('sprint_length_days', 14)} days",
                "epics": list(epics_map.values()),
                "_total_sprints": body.get("sprint_plan", {}).get("total_sprints", 1),
                "_raw_sprints": sprints,
                "_unit_counts": sprint_plan.get("unit_counts", {}),
            }

    epics = sprint_plan.get("epics", [])
    total_stories = sum(len(e.get("stories", [])) for e in epics)

    thread_id = _get_thread_id(state)
    if thread_id != "unknown-thread":
        try:
            save_artifact(
                thread_id=thread_id,
                key="SprintPlan",
                phase="Phase 2 - Planning",
                content=json.dumps(sprint_plan, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            print(f"[Persistence] save_artifact SprintPlan failed: {e}")

    print(f"  ✅ Sprint plan generated: {len(epics)} epics, {total_stories} stories")
    return {**state, "sprint_plan": sprint_plan, "status": "SPRINT_PLAN_GENERATED"}

def create_jira_tickets(state: PlanningState) -> PlanningState:
    """Create Jira epics and stories from sprint plan, in parallel."""
    print("\n[Phase 2] Creating Jira tickets...")

    try:
        import os
        import base64
        import httpx
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # ─── Helper: create one ticket (epic or story) ──────────────
        def _create_one_ticket(ticket_data: dict, ticket_type: str) -> dict:
            """Returns {'key': 'DEV-123', 'title': '...', 'type': 'Epic'} or {'error': ...}"""
            try:
                if ticket_type == "Epic":
                    payload = {"fields": {
                        "project": {"key": project_key},
                        "summary": ticket_data["title"],
                        "description": {
                            "type": "doc", "version": 1,
                            "content": [{"type": "paragraph", "content": [
                                {"type": "text", "text": ticket_data.get("description", "")}
                            ]}]
                        },
                        "issuetype": {"id": jira_meta["issue_types"].get("Epic", "")},
                        "priority": {"id": jira_meta["priorities"].get("Medium", "")},
                        "labels": ["ai-generated", "sdlc-v2"]
                    }}
                else:  # Story
                    ac_list = ticket_data.get("acceptance_criteria", [])
                    ac_text = ""
                    if ac_list:
                        ac_text = "\n\nAcceptance Criteria:\n" + "\n".join(f"- {ac}" for ac in ac_list)
                    full_desc = ticket_data.get("description", "") + ac_text

                    payload = {"fields": {
                        "project": {"key": project_key},
                        "summary": ticket_data["title"],
                        "description": {
                            "type": "doc", "version": 1,
                            "content": [{"type": "paragraph", "content": [
                                {"type": "text", "text": full_desc}
                            ]}]
                        },
                        "issuetype": {"id": jira_meta["issue_types"].get("Story", "")},
                        "priority": {"id": jira_meta["priorities"].get("Medium", "")},
                        "labels": ["ai-generated", "sdlc-v2"]
                    }}

                resp = httpx.post(
                    f"https://{domain}/rest/api/3/issue",
                    headers=headers,
                    json=payload,
                    timeout=15,
                )

                if resp.status_code == 201:
                    return {
                        "key": resp.json()["key"],
                        "title": ticket_data["title"],
                        "type": ticket_type,
                    }
                else:
                    return {"error": f"{ticket_type} '{ticket_data['title']}' failed: {resp.text[:200]}"}
            except Exception as e:
                return {"error": f"{ticket_type} '{ticket_data.get('title', '?')}' exception: {e}"}

        # ─── Main flow ──────────────────────────────────────────────
        epics = state["sprint_plan"].get("epics", [])
        created_tickets = []

        if not epics:
            print("  ⚠️  No epics to create — skipping Jira phase")
            return {**state, "jira_tickets": [], "status": "JIRA_SKIPPED"}

        # Step 1: Create all Epics in parallel
        print(f"  [Phase 2] Creating {len(epics)} epics in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            epic_futures = {executor.submit(_create_one_ticket, e, "Epic"): e for e in epics}
            for fut in as_completed(epic_futures):
                result = fut.result()
                if "error" in result:
                    print(f"    ⚠️  {result['error']}")
                else:
                    created_tickets.append(result["key"])
                    print(f"    ✅ Epic: {result['key']} — {result['title']}")

        # Step 2: Collect all stories across all epics
        all_stories = []
        for epic in epics:
            for story in epic.get("stories", []):
                all_stories.append(story)

        # Step 3: Create all Stories in parallel
        if all_stories:
            print(f"  [Phase 2] Creating {len(all_stories)} stories in parallel...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                story_futures = {executor.submit(_create_one_ticket, s, "Story"): s for s in all_stories}
                for fut in as_completed(story_futures):
                    result = fut.result()
                    if "error" in result:
                        print(f"    ⚠️  {result['error']}")
                    else:
                        created_tickets.append(result["key"])
                        print(f"    ✅ Story: {result['key']} — {result['title']}")

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

    thread_id = _get_thread_id(state)
    if thread_id != "unknown-thread":
        try:
            save_artifact(
                thread_id=thread_id,
                key="Runbook",
                phase="Phase 2 - Planning",
                content=json.dumps(runbook, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            print(f"[Persistence] save_artifact Runbook failed: {e}")

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