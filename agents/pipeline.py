"""
SDLC Automation Pipeline — Full Orchestrator
Runs Phase 1 → Phase 2 → Phase 3 in sequence with human approval gates.
"""

import os
import json
from dotenv import load_dotenv
from agents.phase1_discovery.discovery_agent import start_discovery, resume_discovery, build_discovery_graph
from agents.phase2_planning.planning_agent import start_planning, resume_planning
from agents.phase3_impact.impact_analyzer import run_impact_analysis
from agents.phase3_impact.graph import get_graph

load_dotenv()


def run_pipeline(requirement: str):
    print("\n" + "="*60)
    print("SDLC AUTOMATION PIPELINE — STARTING")
    print(f"Requirement: {requirement}")
    print("="*60)

    # ─────────────────────────────────────────
    # PHASE 1 — Discovery
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 1 — DISCOVERY")
    print("─"*60)

    graph1 = build_discovery_graph()
    config1 = {"configurable": {"thread_id": "pipeline-phase1"}}

    from langgraph.types import Command
    from agents.phase1_discovery.discovery_agent import DiscoveryState

    initial_state = DiscoveryState(
        requirement=requirement,
        brd={}, prd={}, adr={},
        human_feedback="", approved=False,
        status="STARTED"
    )

    result1 = graph1.invoke(initial_state, config1)
    print(f"\n[Pipeline] Phase 1 paused — status: {result1['status']}")
    print(f"  BRD: {result1['brd'].get('title', '')}")
    print(f"  PRD requirements: {len(result1['prd'].get('functional_requirements', []))}")

    # Simulate human approval (in production this waits for real human)
    print("\n[Pipeline] ⏸ Awaiting human approval for Phase 1...")
    input("[Pipeline] Press Enter to approve Phase 1 and continue...")

    result1 = graph1.invoke(
        Command(resume={"approved": True, "feedback": "Approved"}),
        config1
    )
    print(f"[Pipeline] Phase 1 complete — {result1['status']}")

    if result1['status'] != "APPROVED_FOR_PLANNING":
        print("[Pipeline] ❌ Phase 1 rejected — stopping pipeline")
        return

    # ─────────────────────────────────────────
    # PHASE 2 — Planning
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 2 — PLANNING")
    print("─"*60)

    graph2, config2, result2 = start_planning(
        requirement=requirement,
        brd=result1['brd'],
        prd=result1['prd'],
        thread_id="pipeline-phase2"
    )

    print(f"\n[Pipeline] Phase 2 paused — status: {result2['status']}")
    print(f"  Epics: {len(result2['sprint_plan'].get('epics', []))}")
    print(f"  Runbook steps: {len(result2['runbook'].get('deployment_sequence', []))}")

    print("\n[Pipeline] ⏸ Awaiting human approval for Phase 2...")
    input("[Pipeline] Press Enter to approve Phase 2 and continue...")

    result2 = resume_planning(graph2, config2, approved=True)
    print(f"[Pipeline] Phase 2 complete — {result2['status']}")

    if result2['status'] != "APPROVED_FOR_IMPACT_ANALYSIS":
        print("[Pipeline] ❌ Phase 2 rejected — stopping pipeline")
        return

    # ─────────────────────────────────────────
    # PHASE 3 — Impact Analysis
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 3 — IMPACT ANALYSIS")
    print("─"*60)

    graph3 = get_graph()
    config3 = {"configurable": {"thread_id": "pipeline-phase3"}}

    initial_state3 = {
        "requirement": requirement,
        "impact_report": {},
        "human_approved": False,
        "human_feedback": "",
        "status": "STARTED"
    }

    result3 = graph3.invoke(initial_state3, config3)

    print(f"\n[Pipeline] Phase 3 paused — status: {result3['status']}")
    print(f"  Risk: {result3['impact_report']['risk_assessment']['risk_level']}")
    print(f"  Affected files: {[f['file_path'] for f in result3['impact_report']['affected_files']]}")

    print("\n[Pipeline] ⏸ Awaiting human approval for Phase 3...")
    input("[Pipeline] Press Enter to approve Phase 3 and continue...")

    result3 = graph3.invoke(
        Command(resume={"approved": True, "feedback": "Approved"}),
        config3
    )

    print(f"[Pipeline] Phase 3 complete — {result3['status']}")

    if result3['status'] != "APPROVED_FOR_CODE_GENERATION":
        print("[Pipeline] ❌ Phase 3 rejected — stopping pipeline")
        return
    
    # ─────────────────────────────────────────
    # PHASE 4 — Code Generation
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 4 — CODE GENERATION")
    print("─"*60)

    from agents.phase4_codegen.codegen_agent import run_codegen

    result4 = run_codegen(
        requirement=requirement,
        impact_report=result3["impact_report"],
        thread_id="pipeline-phase4"
    )

    if result4["status"] != "VALIDATED":
        print("[Pipeline] ❌ Phase 4 failed — stopping pipeline")
        return

    print(f"[Pipeline] Phase 4 complete — {result4['status']}")
    print(f"  Files changed: {len(result4['generated_changes'])}")

    
    # ─────────────────────────────────────────
    # PHASE 5 — Validation
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 5 — VALIDATION")
    print("─"*60)

    from agents.phase5_validation.validation_agent import run_validation_phase

    result5 = run_validation_phase(
        requirement=requirement,
        generated_changes=result4["generated_changes"],
        thread_id="pipeline-phase5"
    )

    if result5["status"] != "VALIDATION_PASSED":
        print("[Pipeline] ❌ Phase 5 failed — stopping pipeline")
        return

    print(f"[Pipeline] Phase 5 complete — {result5['status']}")
    print(f"  Test files: {len(result5['test_files'])}")
    
    # ─────────────────────────────────────────
    # PHASE 6 — Delivery
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("PHASE 6 — DELIVERY")
    print("─"*60)

    from agents.phase6_delivery.delivery_agent import run_delivery, resume_delivery

    graph6, config6, result6 = run_delivery(
        requirement=requirement,
        generated_changes=result4["generated_changes"],
        test_files=result5["test_files"],
        repo_url="https://github.com/AkashW45/leave-mgmt-backend.git",
        branch_name="feature/leave-balance-v2",
        thread_id="pipeline-phase6"
    )

    print(f"\n[Pipeline] Phase 6 paused — status: {result6['status']}")
    for url in result6.get("pr_urls", []):
        print(f"  PR: {url}")

    print("\n[Pipeline] ⏸ Awaiting human PR review...")
    input("[Pipeline] Press Enter after reviewing PR to approve...")

    result6 = resume_delivery(graph6, config6, approved=True)
    print(f"[Pipeline] Phase 6 complete — {result6['status']}")

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ PIPELINE PHASES 1-6 COMPLETE")
    print("="*60)
    print(f"BRD: {result1['brd'].get('title')}")
    print(f"PRD requirements: {len(result1['prd'].get('functional_requirements', []))}")
    print(f"Sprint epics: {len(result2['sprint_plan'].get('epics', []))}")
    print(f"Files generated: {len(result4['generated_changes'])}")
    print(f"Tests generated: {len(result5['test_files'])}")
    print(f"PRs created: {result6.get('pr_urls', [])}")
    print(f"Final status: {result6['status']}")
    print("\nReady for Phase 7 — Deployment")

    return {
        "brd": result1['brd'],
        "prd": result1['prd'],
        "adr": result1['adr'],
        "sprint_plan": result2['sprint_plan'],
        "runbook": result2['runbook'],
        "impact_report": result3['impact_report']
    }


if __name__ == "__main__":
    requirement = "Add leave balance tracker to Leave Management System. Each employee gets 20 days per year. Balance decreases when leave is approved."
    run_pipeline(requirement)