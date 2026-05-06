"""
Phase 7 — Deployment Agent
Handles deployment sequencing, feature flags, monitoring and rollback.
Human approval gate before production deployment.
"""

import os
import json
import re
import datetime
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

class DeploymentState(TypedDict):
    requirement: str
    runbook: dict
    pr_urls: list
    affected_repos: list
    deploy_sequence: list
    feature_flags: list
    deploy_results: list
    monitoring_results: dict
    rollback_triggered: bool
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
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

def resolve_deploy_sequence(state: DeploymentState) -> DeploymentState:
    """
    Determine correct deployment order.
    Rule: shared libs first → backend services → frontend → batch jobs
    """
    print("\n[Phase 7] Resolving deployment sequence...")

    affected_repos = state.get("affected_repos", ["leave-mgmt-backend"])

    # Classify repos by type
    sequence = []
    libs = [r for r in affected_repos if "lib" in r or "common" in r or "shared" in r]
    backends = [r for r in affected_repos if "backend" in r or "api" in r or "service" in r]
    frontends = [r for r in affected_repos if "frontend" in r or "ui" in r or "web" in r]
    batches = [r for r in affected_repos if "batch" in r or "job" in r or "worker" in r]
    others = [r for r in affected_repos if r not in libs + backends + frontends + batches]

    # Build sequence
    step = 1
    for repo in libs + backends + frontends + batches + others:
        sequence.append({
            "step": step,
            "repo": repo,
            "type": (
                "library" if repo in libs else
                "backend" if repo in backends else
                "frontend" if repo in frontends else
                "batch" if repo in batches else
                "service"
            ),
            "status": "PENDING"
        })
        step += 1

    print(f"  Deploy sequence: {[s['repo'] for s in sequence]}")

    return {
        **state,
        "deploy_sequence": sequence,
        "status": "SEQUENCE_RESOLVED"
    }


def setup_feature_flags(state: DeploymentState) -> DeploymentState:
    """Setup feature flags for safe deployment."""
    print("\n[Phase 7] Setting up feature flags...")

    runbook = state.get("runbook", {})
    flags_from_runbook = runbook.get("feature_flags", [])

    if not flags_from_runbook:
        # Generate default feature flag
        flags_from_runbook = [
            {
                "flag_name": "leave_balance_tracker_enabled",
                "default": False,
                "enable_after_deploy": True,
                "description": "Enable leave balance tracking feature"
            }
        ]

    feature_flags = []
    for flag in flags_from_runbook:
        feature_flags.append({
            "flag_name": flag.get("flag_name", ""),
            "enabled": False,  # Always start disabled
            "enable_after_deploy": flag.get("enable_after_deploy", True),
            "description": flag.get("description", "")
        })
        print(f"  Flag: {flag.get('flag_name')} — disabled (will enable after deploy)")

    return {
        **state,
        "feature_flags": feature_flags,
        "status": "FLAGS_CONFIGURED"
    }


def human_approval_gate(state: DeploymentState) -> DeploymentState:
    """Human must approve before production deployment."""
    print("\n[Phase 7] ⏸ Awaiting production deployment approval...")
    print(f"  Deploy sequence: {[s['repo'] for s in state['deploy_sequence']]}")
    print(f"  Feature flags: {[f['flag_name'] for f in state['feature_flags']]}")
    print(f"  PRs to deploy: {state.get('pr_urls', [])}")

    human_input = interrupt("Approve production deployment")

    approved = False
    feedback = ""
    if isinstance(human_input, dict):
        approved = human_input.get("approved", False)
        feedback = human_input.get("feedback", "")

    return {
        **state,
        "approved": approved,
        "human_feedback": feedback,
        "status": "WAITING_FOR_DEPLOY_APPROVAL"
    }


def execute_deployment(state: DeploymentState) -> DeploymentState:
    """
    Execute deployment in sequence order.
    In production this calls TeamCity REST API via n8n.
    Currently simulates deployment steps.
    """
    print("\n[Phase 7] Executing deployment...")

    deploy_results = []
    all_passed = True

    for step in state["deploy_sequence"]:
        repo = step["repo"]
        print(f"  Step {step['step']}: Deploying {repo}...")

        # Simulate deployment
        # In production: POST to n8n webhook which triggers TeamCity
        result = {
            "step": step["step"],
            "repo": repo,
            "status": "SUCCESS",
            "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "message": f"{repo} deployed successfully"
        }

        deploy_results.append(result)
        print(f"  ✅ {repo}: deployed")

    return {
        **state,
        "deploy_results": deploy_results,
        "status": "DEPLOYED" if all_passed else "DEPLOY_FAILED"
    }


def enable_feature_flags(state: DeploymentState) -> DeploymentState:
    """Enable feature flags after successful deployment."""
    print("\n[Phase 7] Enabling feature flags...")

    updated_flags = []
    for flag in state["feature_flags"]:
        if flag.get("enable_after_deploy", True):
            flag["enabled"] = True
            print(f"  ✅ Enabled: {flag['flag_name']}")
        updated_flags.append(flag)

    return {
        **state,
        "feature_flags": updated_flags,
        "status": "FLAGS_ENABLED"
    }


def monitor_deployment(state: DeploymentState) -> DeploymentState:
    """
    Monitor post-deployment metrics.
    In production: queries monitoring system for error rates, latency etc.
    """
    print("\n[Phase 7] Monitoring post-deployment metrics...")

    # Simulate monitoring checks
    monitoring_results = {
        "checked_at": datetime.datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "error_rate": 0.0,
            "avg_latency_ms": 45,
            "requests_per_min": 120,
            "health_check": "passing"
        },
        "thresholds": {
            "max_error_rate": 0.05,
            "max_latency_ms": 500
        },
        "alerts": [],
        "status": "HEALTHY"
    }

    error_rate = monitoring_results["metrics"]["error_rate"]
    max_error_rate = monitoring_results["thresholds"]["max_error_rate"]

    if error_rate > max_error_rate:
        monitoring_results["alerts"].append(
            f"Error rate {error_rate} exceeds threshold {max_error_rate}"
        )
        monitoring_results["status"] = "DEGRADED"
        print(f"  ❌ Metrics degraded — triggering rollback")
        return {
            **state,
            "monitoring_results": monitoring_results,
            "rollback_triggered": True,
            "status": "ROLLBACK_TRIGGERED"
        }

    print(f"  ✅ All metrics healthy")
    print(f"     Error rate: {error_rate}")
    print(f"     Latency: {monitoring_results['metrics']['avg_latency_ms']}ms")

    return {
        **state,
        "monitoring_results": monitoring_results,
        "rollback_triggered": False,
        "status": "DEPLOYMENT_COMPLETE"
    }


def execute_rollback(state: DeploymentState) -> DeploymentState:
    """Execute rollback if metrics degrade."""
    print("\n[Phase 7] ⚠️  Executing rollback...")

    runbook = state.get("runbook", {})
    rollback_steps = runbook.get("rollback_steps", [
        "Revert to previous deployment",
        "Disable feature flags",
        "Alert on-call team"
    ])

    for i, step in enumerate(rollback_steps, 1):
        print(f"  Rollback step {i}: {step}")

    # Disable all feature flags
    updated_flags = []
    for flag in state["feature_flags"]:
        flag["enabled"] = False
        updated_flags.append(flag)

    print(f"  ✅ Rollback complete — all flags disabled")

    return {
        **state,
        "feature_flags": updated_flags,
        "status": "ROLLED_BACK"
    }


# -----------------------------------------
# Routing
# -----------------------------------------

def route_after_approval(state: DeploymentState) -> str:
    if state.get("approved"):
        return "approved"
    return "rejected"


def route_after_monitoring(state: DeploymentState) -> str:
    if state.get("rollback_triggered"):
        return "rollback"
    return "complete"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_deployment_graph():
    builder = StateGraph(DeploymentState)

    builder.add_node("resolve_deploy_sequence", resolve_deploy_sequence)
    builder.add_node("setup_feature_flags", setup_feature_flags)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("execute_deployment", execute_deployment)
    builder.add_node("enable_feature_flags", enable_feature_flags)
    builder.add_node("monitor_deployment", monitor_deployment)
    builder.add_node("execute_rollback", execute_rollback)

    builder.set_entry_point("resolve_deploy_sequence")
    builder.add_edge("resolve_deploy_sequence", "setup_feature_flags")
    builder.add_edge("setup_feature_flags", "human_approval_gate")
    builder.add_edge("human_approval_gate", "execute_deployment")

    builder.add_conditional_edges(
        "execute_deployment",
        lambda s: "proceed" if s["status"] == "DEPLOYED" else "fail",
        {
            "proceed": "enable_feature_flags",
            "fail": END
        }
    )

    builder.add_edge("enable_feature_flags", "monitor_deployment")

    builder.add_conditional_edges(
        "monitor_deployment",
        route_after_monitoring,
        {
            "complete": END,
            "rollback": "execute_rollback"
        }
    )

    builder.add_edge("execute_rollback", END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"]
    )


# -----------------------------------------
# Run
# -----------------------------------------

def run_deployment(
    requirement: str,
    runbook: dict,
    pr_urls: list,
    affected_repos: list,
    thread_id: str = "thread-deployment"
) -> tuple:
    graph = build_deployment_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = DeploymentState(
        requirement=requirement,
        runbook=runbook,
        pr_urls=pr_urls,
        affected_repos=affected_repos,
        deploy_sequence=[],
        feature_flags=[],
        deploy_results=[],
        monitoring_results={},
        rollback_triggered=False,
        human_feedback="",
        approved=False,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 7 — Deployment ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\nStatus after interrupt: {result['status']}")
    print(f"  Deploy sequence: {[s['repo'] for s in result['deploy_sequence']]}")
    print(f"  Feature flags: {[f['flag_name'] for f in result['feature_flags']]}")

    return graph, config, result


def resume_deployment(graph, config, approved: bool, feedback: str = "") -> dict:
    print(f"\n--- Resuming Phase 7 (approved={approved}) ---")
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
    mock_runbook = {
        "feature_flags": [
            {
                "flag_name": "leave_balance_tracker_enabled",
                "default": False,
                "enable_after_deploy": True
            }
        ],
        "rollback_steps": [
            "Revert to previous git tag",
            "Disable leave_balance_tracker_enabled flag",
            "Alert #deployments Slack channel"
        ]
    }

    graph7, config7, result7 = run_deployment(
        requirement="Add leave balance tracker",
        runbook=mock_runbook,
        pr_urls=["https://github.com/AkashW45/leave-mgmt-backend/pull/11"],
        affected_repos=["leave-mgmt-backend"],
        thread_id="test-deployment-1"
    )

    print("\n--- Simulating Production Approval ---")
    final = resume_deployment(graph7, config7, approved=True)

    print(f"\n✅ Phase 7 Test Complete")
    print(f"Status: {final['status']}")
    print(f"Deployed repos: {[r['repo'] for r in final['deploy_results']]}")
    print(f"Feature flags enabled: {[f['flag_name'] for f in final['feature_flags'] if f['enabled']]}")
    print(f"Monitoring: {final['monitoring_results'].get('metrics', {})}")
    print(f"Rollback triggered: {final['rollback_triggered']}")