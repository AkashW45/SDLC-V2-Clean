import os
import sys
import json
import re
sys.path.insert(0, r"C:\Users\user\SDLC-V2")

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from typing import TypedDict, Dict, Any
from agents.prompts.system_prompts import (
    CLASSIFIER_SYSTEM, BRD_SYSTEM, PRD_SYSTEM,
    ADR_SYSTEM, ARCHITECTURE_SYSTEM, CRITIC_SYSTEM
)
import json
from agents.critic.critic_agent import critique
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-v4-flash"
EXTRA_PARAMS = {
    "stream": False,
    "reasoning_effort": "high",
    "extra_body": {"thinking": {"type": "enabled"}}
}


class DiscoveryState(TypedDict):
    requirement: str
    scope_contract: dict      # NEW — frozen after Phase 0.5
    classifier_output: dict   # NEW — full classifier response
    brd: Dict[str, Any]
    prd: Dict[str, Any]
    adr: Dict[str, Any]
    architecture: Dict[str, Any]
    human_feedback: str
    approved: bool
    status: str


def _llm_json(prompt: str, max_tokens: int = 16384) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=max_tokens
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
    try:
        return json.loads(content)
    except Exception:
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            return json.loads(content[start:end])
        except Exception:
            print(f"  [LLM] JSON parse failed. Raw response (first 500 chars): {content[:500]}")
            return {}
        
# ── Fallbacks ─────────────────────────────────────────────────
def _make_fallback_brd(requirement: str, contract: dict) -> dict:
    return {
        "title": requirement[:80],
        "executive_summary": requirement,
        "business_objectives": [requirement],
        "scope": {"in_scope": [requirement], "out_of_scope": []},
        "kpis": [],
        "risks": [],
        "stakeholders": []
    }

def _make_fallback_prd(requirement: str, contract: dict) -> dict:
    return {
        "title": requirement[:80],
        "functional_requirements": [],
        "non_functional_requirements": [],
        "technical_requirements": []
    }

def _make_fallback_adr(requirement: str, contract: dict) -> dict:
    return {"decisions": []}

def _make_fallback_architecture(requirement: str, contract: dict) -> dict:
    return {
        "system_name": requirement[:50],
        "nodes": [],
        "edges": [],
        "mermaid": "graph LR\n    A[System]\n"
    }



def _feedback_block(state) -> str:
    fb = state.get("human_feedback", "")
    if fb:
        return f"\n\nIMPORTANT — USER FEEDBACK on previous attempt (you MUST address all of this):\n{fb}\n"
    return ""

def safe_parse_json(raw: str, label: str = "json") -> dict:
    if not raw:
        return {}

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()

    try:
        return json.loads(raw)
    except Exception:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception:
            pass

    print(f"  [{label}] JSON parse failed. Raw response head: {raw[:500]}")
    return {}

def classify_intent(state: DiscoveryState) -> DiscoveryState:
    """Phase 0.5 — produces Scope Contract that binds all later phases."""
    print("[Phase 0.5] Classifying intent + producing Scope Contract...")

    user_msg = f"PROJECT_REQUEST: {state['requirement']}"

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=4096,
        **EXTRA_PARAMS
    )

    raw = response.choices[0].message.content
    parsed = safe_parse_json(raw, "classifier")  # use your existing safe parser

    # If ambiguity → still produce a reasonable default contract so pipeline doesn't halt
    if parsed.get("ambiguity_detected") or not parsed.get("scope_contract"):
        print(f"  ⚠️ Ambiguity detected. Questions: {parsed.get('clarifying_questions', [])}")
        # Fallback contract — depth 3 default
        parsed["scope_contract"] = {
            "depth_level": 3,
            "depth_rationale": "Default — ambiguous requirement",
            "scope_anchor": {
                "primary_domain": state["requirement"][:50],
                "user_types": ["primary user"],
                "core_capabilities": [state["requirement"][:100]],
                "explicit_integrations": [],
                "explicit_compliance": [],
                "explicit_scale": "unspecified",
                "production_intent": False
            },
            "hard_limits": {
                "max_functional_requirements": 10,
                "max_architecture_nodes": 8,
                "max_jira_tickets": 15,
                "max_code_files": 20,
                "max_sprints": 2
            },
            "forbidden_elements": ["microservices", "kafka", "kubernetes", "multi_region", "service_mesh"],
            "mandatory_elements": []
        }

    contract = parsed["scope_contract"]
    print(f"  ✅ Depth: {contract['depth_level']} | Domain: {contract['scope_anchor']['primary_domain']}")
    print(f"  Hard limits: {contract['hard_limits']}")
    print(f"  Forbidden: {contract['forbidden_elements'][:5]}")

    return {
        **state,
        "classifier_output": parsed,
        "scope_contract": contract,
        "status": "INTENT_CLASSIFIED"
    }

def generate_brd(state: DiscoveryState) -> DiscoveryState:
    print("[Phase 1] Generating BRD...")

    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")

    user_msg = json.dumps({
        "requirement": state["requirement"],
        "scope_contract": contract,
        "human_feedback": feedback
    }, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": BRD_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=16384,
        **EXTRA_PARAMS
    )

    raw = response.choices[0].message.content
    brd = safe_parse_json(raw, "brd")

    if not brd:
        brd = _make_fallback_brd(state["requirement"], contract)

        # Critic check
    verdict = critique(brd, "brd", contract, state["requirement"])
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join(
            [v["fix_instruction"] for v in verdict.get("violations", [])]
        )
        return generate_brd(state)  # one retry    

    print(f"  ✅ BRD generated: {brd.get('title', 'Untitled')}")
    return {**state, "brd": brd, "status": "BRD_GENERATED"}

def generate_prd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating PRD...")

    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")

    user_msg = json.dumps({
        "requirement": state["requirement"],
        "scope_contract": contract,
        "brd": state["brd"],
        "human_feedback": feedback
    }, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PRD_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=16384,
        **EXTRA_PARAMS
    )

    raw = response.choices[0].message.content
    prd = safe_parse_json(raw, "prd")

    if not prd:
        prd = _make_fallback_prd(state["requirement"], contract)

    # Critic check
    verdict = critique(prd, "prd", contract, state["requirement"],
                       prior_artifacts={"brd": state["brd"]})
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join(
            [v["fix_instruction"] for v in verdict.get("violations", [])]
        )
        return generate_prd(state)  # one retry

    print(f"  ✅ PRD: {prd.get('title', 'Untitled')} — {len(prd.get('functional_requirements', []))} FRs, "
          f"{len(prd.get('non_functional_requirements', []))} NFRs")
    return {**state, "prd": prd, "status": "PRD_GENERATED"}


def generate_adr(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating ADR...")

    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")

    user_msg = json.dumps({
        "requirement": state["requirement"],
        "scope_contract": contract,
        "brd": state["brd"],
        "prd": state["prd"],
        "human_feedback": feedback
    }, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ADR_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=16384,
        **EXTRA_PARAMS
    )

    raw = response.choices[0].message.content
    adr = safe_parse_json(raw, "adr")

    if not adr:
        adr = _make_fallback_adr(state["requirement"], contract)

    # Critic check
    verdict = critique(adr, "adr", contract, state["requirement"],
                       prior_artifacts={"brd": state["brd"], "prd": state["prd"]})
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join(
            [v["fix_instruction"] for v in verdict.get("violations", [])]
        )
        return generate_adr(state)  # one retry

    print(f"  ✅ ADR generated: {len(adr.get('decisions', []))} decisions")
    return {**state, "adr": adr, "status": "ADR_GENERATED"}




# def generate_prd(state: DiscoveryState) -> DiscoveryState:
#     print("\n[Phase 1] Generating PRD...")

#     brd = state["brd"]
#     prd = _llm_json(f"""
# You are a Senior Product Owner writing a PRD for engineering execution.
# This must read like a Stripe/Linear PRD — specific user stories, not vague generalities.
# {_feedback_block(state)}
# Return ONLY valid JSON:

# {{
#   "title": "...",
#   "executive_summary": "2-paragraph summary for product leadership",
#   "product_vision": "Clear vision statement — who, what, why",
#   "target_users": [
#     {{"persona": "User type", "needs": "what they need", "pain_points": "current problems"}}
#   ],
#   "user_journeys": [
#     {{"journey": "End-to-end flow name", "steps": ["step 1", "step 2"]}}
#   ],
#   "functional_requirements": [
#     {{
#       "id": "FR1",
#       "title": "...",
#       "user_story": "As a [persona] I want [goal] so that [benefit]",
#       "description": "Detailed description",
#       "priority": "P0|P1|P2",
#       "acceptance_criteria": ["Given...When...Then..."],
#       "edge_cases": ["edge case 1", "edge case 2"],
#       "dependencies": ["depends on FR2"]
#     }}
#   ],
#   "non_functional_requirements": [
#     {{"id": "NFR1", "title": "...", "description": "Quantified", "priority": "High", "verification_method": "load test / monitoring / etc"}}
#   ],
#   "technical_requirements": [
#     {{"id": "TR1", "title": "...", "description": "Specific technical constraint", "rationale": "why this constraint exists"}}
#   ],
#   "success_metrics": [
#     {{"metric": "...", "baseline": "current value", "target": "desired value", "timeline": "when measured"}}
#   ],
#   "release_phases": [
#     {{"phase": "MVP", "timeline": "Week 1-4", "scope": ["FR1", "FR2"]}},
#     {{"phase": "Phase 2", "timeline": "Week 5-8", "scope": ["FR3"]}}
#   ],
#   "open_questions": ["question that needs decision"]
# }}

# Minimum:
# - 6+ FRs with user stories AND edge cases
# - 4+ NFRs with verification methods
# - 4+ TRs with rationale
# - 3+ user personas
# - 4+ success metrics with baseline AND target
# - 3+ release phases

# BRD Context:
# Title: {brd.get('title','')}
# Summary: {brd.get('executive_summary','')[:500]}
# Objectives: {json.dumps(brd.get('business_objectives', []))}
# KPIs: {json.dumps(brd.get('kpis', []))}
# """)

#     if not prd.get("title"):
#         prd["title"] = brd.get("title", "Untitled")
#     for k in ["functional_requirements", "non_functional_requirements", "technical_requirements"]:
#         if not prd.get(k):
#             prd[k] = []

#     print(f"  ✅ PRD: {prd['title']} — {len(prd['functional_requirements'])} FRs, {len(prd['non_functional_requirements'])} NFRs")
#     return {**state, "prd": prd, "status": "PRD_GENERATED"}


# def generate_adr(state: DiscoveryState) -> DiscoveryState:
#     print("\n[Phase 1] Generating ADR...")

#     prd = state["prd"]
#     adr = _llm_json(f"""
# You are a Senior Software Architect writing ADRs in MADR format.
# Each ADR must include alternatives considered with trade-offs — not just the chosen path.
# {_feedback_block(state)}
# Return ONLY valid JSON:

# {{
#   "decisions": [
#     {{
#       "id": "ADR-001",
#       "title": "Specific architectural decision",
#       "status": "Accepted|Proposed|Deprecated",
#       "date": "2026-05-08",
#       "context": "Why this decision was needed — 2-3 sentences with specific drivers",
#       "decision": "What was decided — be specific (technology, pattern, approach)",
#       "rationale": "Why this option over others — paragraph",
#       "alternatives_considered": [
#         {{"option": "Alt 1", "pros": ["pro 1"], "cons": ["con 1"], "rejected_because": "specific reason"}}
#       ],
#       "consequences": {{
#         "positive": ["positive consequence 1", "..."],
#         "negative": ["negative consequence 1", "..."],
#         "neutral": ["neutral implication"]
#       }},
#       "compliance_notes": "any regulatory/security implications",
#       "supersedes": "ADR-XXX or null"
#     }}
#   ]
# }}

# Minimum 7 decisions covering:
# 1. Tech stack (language/framework)
# 2. Data store (type and rationale)
# 3. Deployment platform (cloud/on-prem/hybrid)
# 4. Authentication & authorization approach
# 5. Observability strategy (logging/monitoring/tracing)
# 6. API design (REST/GraphQL/gRPC)
# 7. Async processing (queue/streaming/batch)

# Each MUST have at least 2 alternatives_considered with detailed pros/cons.

# PRD Context:
# Project: {prd.get('title','')}
# NFRs: {json.dumps(prd.get('non_functional_requirements', [])[:6])}
# Tech Requirements: {json.dumps(prd.get('technical_requirements', [])[:6])}
# """)

#     if not adr.get("decisions"):
#         adr["decisions"] = []

#     print(f"  ✅ ADR generated: {len(adr['decisions'])} decisions")
#     return {**state, "adr": adr, "status": "ADR_GENERATED"}

# import re

def _build_mermaid(arch: dict) -> str:
    nodes = arch.get("nodes", [])
    edges = arch.get("edges", [])

    def safe_id(node_id):
        return re.sub(r'[^A-Z0-9_]', '_', str(node_id).upper())

    def shape(node):
        nid = safe_id(node.get("id", ""))
        name = node.get("name", "").replace('"', "'")[:35]
        ntype = (node.get("type") or "service").lower()
        if ntype == "database":
            return f'        {nid}[("{name}")]'
        elif ntype == "external":
            return f'        {nid}(["{name}"])'
        elif ntype == "queue":
            return f'        {nid}[/"{name}"\\]'
        elif ntype == "cache":
            return f'        {nid}[("{name}")]'
        else:
            return f'        {nid}["{name}"]'

    # Group nodes by zone
    zones = {}
    for n in nodes:
        zone = (n.get("zone") or "core").lower()
        zones.setdefault(zone, []).append(n)

    # Render order — controls visual flow left to right
    zone_order = ["external", "edge", "dmz", "core", "pci", "data", "observability"]
    zone_titles = {
        "external": "External",
        "edge": "Edge",
        "dmz": "DMZ",
        "core": "Core Services",
        "pci": "PCI Zone",
        "data": "Data Layer",
        "observability": "Observability"
    }

    lines = ["graph LR"]

    # Subgraphs per zone
    for zone in zone_order:
        if zone not in zones:
            continue
        title = zone_titles.get(zone, zone.title())
        lines.append(f'    subgraph {zone.upper()}["{title}"]')
        lines.append(f'    direction TB')
        for n in zones[zone]:
            lines.append(shape(n))
        lines.append('    end')

    # Edges
    for e in edges:
        src = safe_id(e.get("source", ""))
        tgt = safe_id(e.get("target", ""))
        proto = (e.get("protocol", "") or "").replace('|', '/')[:12]
        if proto:
            lines.append(f'    {src} -->|{proto}| {tgt}')
        else:
            lines.append(f'    {src} --> {tgt}')

    # Styling — professional palette
    lines.append('')
    lines.append('    classDef service fill:#1E40AF,stroke:#3B82F6,stroke-width:2px,color:#fff')
    lines.append('    classDef database fill:#065F46,stroke:#10B981,stroke-width:2px,color:#fff')
    lines.append('    classDef external fill:#6B21A8,stroke:#A855F7,stroke-width:2px,color:#fff')
    lines.append('    classDef queue fill:#9A3412,stroke:#F97316,stroke-width:2px,color:#fff')
    lines.append('    classDef cache fill:#854D0E,stroke:#EAB308,stroke-width:2px,color:#fff')

    # Apply classes
    for n in nodes:
        nid = safe_id(n.get("id", ""))
        ntype = (n.get("type") or "service").lower()
        if ntype in ("service", "database", "external", "queue", "cache"):
            lines.append(f'    class {nid} {ntype}')
        else:
            lines.append(f'    class {nid} service')

    return "\n".join(lines)


def generate_architecture(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating architecture design...")

    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")

    user_msg = json.dumps({
        "requirement": state["requirement"],
        "scope_contract": contract,
        "brd": state["brd"],
        "prd": state["prd"],
        "adr": state["adr"],
        "human_feedback": feedback
    }, indent=2)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ARCHITECTURE_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=8192,
        **EXTRA_PARAMS
    )

    raw = response.choices[0].message.content
    arch = safe_parse_json(raw, "architecture")

    if not arch:
        arch = _make_fallback_architecture(state["requirement"], contract)

    # Add Mermaid diagram
    arch["mermaid"] = _build_mermaid(arch)

    # Critic check
    verdict = critique(arch, "architecture", contract, state["requirement"],
                       prior_artifacts={"brd": state["brd"], "prd": state["prd"], "adr": state["adr"]})
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join(
            [v["fix_instruction"] for v in verdict.get("violations", [])]
        )
        return generate_architecture(state)

    print(f"  ✅ Architecture: {len(arch.get('nodes', []))} nodes, {len(arch.get('edges', []))} edges")
    return {**state, "architecture": arch, "status": "ARCHITECTURE_GENERATED"}

# def generate_architecture(state: DiscoveryState) -> DiscoveryState:
#     print("\n[Phase 1] Generating architecture design...")

#     prd = state["prd"]

#     # Build canonical requirement list (V1 pattern)
#     functional_reqs = []
#     for r in prd.get("functional_requirements", []):
#         if isinstance(r, dict):
#             text = f"{r.get('title','')} {r.get('description','')}".strip()
#             if text:
#                 functional_reqs.append(text)
#         elif isinstance(r, str):
#             functional_reqs.append(r)

#     non_functional_reqs = []
#     for r in prd.get("non_functional_requirements", []):
#         if isinstance(r, dict):
#             text = r.get("description") or r.get("title") or ""
#             if text:
#                 non_functional_reqs.append(text)
#         elif isinstance(r, str):
#             non_functional_reqs.append(r)

#     project_name = prd.get("title", "SYSTEM")

#     arch = _llm_json(f"""
# You are a strict software architect.
# {_feedback_block(state)}
# RULES:
# 1. Every node MUST be justified by a functional or non-functional requirement
# 2. Do NOT invent technologies not implied by requirements
# 3. 4-8 nodes maximum
# 4. Every node MUST include "traced_to" — the exact requirement text it fulfils
# 5. Return ONLY valid JSON

# Project: {project_name}

# Functional Requirements:
# {json.dumps(functional_reqs, indent=2)}

# Non-Functional Requirements:
# {json.dumps(non_functional_reqs, indent=2)}

# Return:
# {{
#   "system_name": "{project_name}",
#   "architecture_style": "microservices | monolith | layered | event-driven",
#   "deployment_model": "kubernetes | docker | serverless | vm",
#   "nodes": [
#     {{
#       "id": "UPPERCASE_ID",
#       "name": "Human Name",
#       "type": "service | database | external | queue | cache",
#       "zone": "external | core | data | observability",
#       "description": "What it does",
#       "tech_stack": ["Tech1", "Tech2"],
#       "responsibilities": ["resp1", "resp2"],
#       "traced_to": "exact requirement text"
#     }}
#   ],
#   "edges": [
#     {{
#       "source": "ID",
#       "target": "ID",
#       "protocol": "REST | gRPC | Kafka | SQL | HTTPS | internal",
#       "description": "what flows"
#     }}
#   ],
#   "security_considerations": ["..."],
#   "scalability_notes": "..."
# }}
# """)

#     if not arch.get("nodes"):
#         arch["nodes"] = []
#     if not arch.get("edges"):
#         arch["edges"] = []

    

#     arch["mermaid"] = _build_mermaid(arch)

#     print(f"  ✅ Architecture: {len(arch['nodes'])} nodes, {len(arch['edges'])} edges")
#     return {**state, "architecture": arch, "status": "ARCHITECTURE_GENERATED"}


def human_approval_gate(state: DiscoveryState) -> DiscoveryState:
    interrupt({
        "type": "DISCOVERY_REVIEW",
        "brd": state["brd"],
        "prd": state["prd"],
        "adr": state["adr"],
        "architecture": state["architecture"]
    })
    return state


def process_approval(state: DiscoveryState) -> DiscoveryState:
    if state.get("approved"):
        return {**state, "status": "APPROVED_FOR_PLANNING"}
    return {**state, "status": "REJECTED"}


def build_discovery_graph():
    g = StateGraph(DiscoveryState)
    g.add_node("classify_intent", classify_intent)   # NEW
    g.add_node("generate_brd", generate_brd)
    g.add_node("generate_prd", generate_prd)
    g.add_node("generate_adr", generate_adr)
    g.add_node("generate_architecture", generate_architecture)
    g.add_node("human_approval_gate", human_approval_gate)
    g.add_node("process_approval", process_approval)

    g.set_entry_point("classify_intent")               # was: "generate_brd"
    g.add_edge("classify_intent", "generate_brd")      # NEW
    g.add_edge("generate_brd", "generate_prd")
    g.add_edge("generate_prd", "generate_adr")
    g.add_edge("generate_adr", "generate_architecture")
    g.add_edge("generate_architecture", "human_approval_gate")
    g.add_edge("human_approval_gate", "process_approval")
    g.add_edge("process_approval", END)

    return g.compile(checkpointer=MemorySaver())