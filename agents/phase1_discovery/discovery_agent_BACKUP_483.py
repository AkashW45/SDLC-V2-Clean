import os
import sys
import json
import re
sys.path.insert(0, r"C:\Users\user\SDLC-V2")

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
from tenacity import retry, stop_after_attempt, wait_exponential
from core.llm_gateway import gateway
from core.context_engine import coe

load_dotenv()

<<<<<<< HEAD
=======
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

>>>>>>> origin/main

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


<<<<<<< HEAD
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def _generate_validated_json(prompt: str, max_tokens: int = 8192) -> dict:
    """Uses LLMGateway with strict JSON enforcement and retries."""
    
    system_prompt = "You are a strict JSON outputter. Respond ONLY with valid JSON. Do not include markdown formatting like ```json."
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    try:
        raw_response = gateway.generate(
            prompt=full_prompt, 
            model="deepseek-chat", # or deepseek-reasoner
            temperature=0.1, 
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            tag="phase1_discovery"
            # Note: Do not pass 'extra_params' here, it crashes the OpenAI SDK
        )
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"  [LLM] JSON decode error, retrying... Error: {e}")
        raise # Triggers Tenacity retry
=======
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


>>>>>>> origin/main

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

<<<<<<< HEAD
    brd = _generate_validated_json(f"""
You are a Senior Business Analyst writing a BRD for executive review.
The BRD must read like a McKinsey document — specific, quantified, no fluff.
{_feedback_block(state)}
Return ONLY valid JSON with this exact structure:
=======
    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")
>>>>>>> origin/main

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

<<<<<<< HEAD
    # Use COE to extract and compress BRD context into YAML
    brd_context_yaml = coe.optimize_for_prompt(state, [
        "brd.title",
        "brd.business_objectives",
        "brd.kpis"
    ])
    
    prd = _generate_validated_json(f"""
You are a Senior Product Owner writing a PRD for engineering execution.
This must read like a Stripe/Linear PRD — specific user stories, not vague generalities.
{_feedback_block(state)}
Return ONLY valid JSON:
=======
    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")
>>>>>>> origin/main

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

<<<<<<< HEAD
BRD Context (YAML):
{brd_context_yaml}
""")
=======
    raw = response.choices[0].message.content
    prd = safe_parse_json(raw, "prd")
>>>>>>> origin/main

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

<<<<<<< HEAD
    # Use COE to extract and compress PRD context into YAML
    # COE acts as firewall, automatically handling key extraction and compression
    prd_context_yaml = coe.optimize_for_prompt(state, [
        "prd.title",
        "prd.non_functional_requirements",
        "prd.technical_requirements"
    ])
    
    adr = _generate_validated_json(f"""
You are a Senior Software Architect writing ADRs in MADR format.
Each ADR must include alternatives considered with trade-offs — not just the chosen path.
{_feedback_block(state)}
Return ONLY valid JSON:
=======
    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")
>>>>>>> origin/main

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

<<<<<<< HEAD
PRD Context (YAML):
{prd_context_yaml}
""")
=======
    if not adr:
        adr = _make_fallback_adr(state["requirement"], contract)
>>>>>>> origin/main

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

<<<<<<< HEAD
    prd = state["prd"]
    adr = state.get("adr", {})
=======
    contract = state["scope_contract"]
    feedback = state.get("human_feedback", "")
>>>>>>> origin/main

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

<<<<<<< HEAD
    # Use COE to extract and compress ADR decisions into YAML
    adr_context_yaml = coe.optimize_for_prompt(state, ["adr.decisions"])

    project_name = prd.get("title", "SYSTEM")

    arch = _generate_validated_json(f"""
You are a strict software architect.
{_feedback_block(state)}
RULES:
1. Every node MUST be justified by a functional or non-functional requirement
2. Do NOT invent technologies not implied by requirements
3. 4-8 nodes maximum
4. Every node MUST include "traced_to" — the exact requirement text it fulfils
5. You MUST strictly adhere to the technologies and patterns defined in the ADR Context below. Do not invent conflicting nodes.
6. Return ONLY valid JSON
=======
    raw = response.choices[0].message.content
    arch = safe_parse_json(raw, "architecture")

    if not arch:
        arch = _make_fallback_architecture(state["requirement"], contract)
>>>>>>> origin/main

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

<<<<<<< HEAD
Non-Functional Requirements:
{json.dumps(non_functional_reqs, indent=2)}

ADR Context (Architectural Decisions - YAML):
{adr_context_yaml}

Return:
{{
  "system_name": "{project_name}",
  "architecture_style": "microservices | monolith | layered | event-driven",
  "deployment_model": "kubernetes | docker | serverless | vm",
  "nodes": [
    {{
      "id": "UPPERCASE_ID",
      "name": "Human Name",
      "type": "service | database | external | queue | cache",
      "zone": "external | core | data | observability",
      "description": "What it does",
      "tech_stack": ["Tech1", "Tech2"],
      "responsibilities": ["resp1", "resp2"],
      "traced_to": "exact requirement text"
    }}
  ],
  "edges": [
    {{
      "source": "ID",
      "target": "ID",
      "protocol": "REST | gRPC | Kafka | SQL | HTTPS | internal",
      "description": "what flows"
    }}
  ],
  "security_considerations": ["..."],
  "scalability_notes": "..."
}}
""")

    if not arch.get("nodes"):
        arch["nodes"] = []
    if not arch.get("edges"):
        arch["edges"] = []

    # Generate mermaid deterministically (FIXED syntax)
    mermaid_lines = ["graph TD"]
    for node in arch.get("nodes", []):
        nid = re.sub(r'[^A-Z0-9_]', '_', node["id"].upper())
        # Escape quotes in name
        name = node["name"].replace('"', "'")[:40]
        ntype = node.get("type", "service")

        if ntype == "database":
            mermaid_lines.append(f'    {nid}[("{name}")]')
        elif ntype == "external":
            mermaid_lines.append(f'    {nid}(["{name}"])')
        elif ntype == "queue":
            mermaid_lines.append(f'    {nid}>"{name}"]')  # asymmetric shape, valid mermaid
        elif ntype == "cache":
            mermaid_lines.append(f'    {nid}[/"{name}"/]')
        else:
            mermaid_lines.append(f'    {nid}["{name}"]')

    for edge in arch.get("edges", []):
        src = re.sub(r'[^A-Z0-9_]', '_', edge["source"].upper())
        tgt = re.sub(r'[^A-Z0-9_]', '_', edge["target"].upper())
        proto = edge.get("protocol", "").replace('|', '/')[:15]
        if proto:
            mermaid_lines.append(f'    {src} -->|{proto}| {tgt}')
        else:
            mermaid_lines.append(f'    {src} --> {tgt}')

    # Add styling
    mermaid_lines.append("    classDef serviceStyle fill:#1a4a8a,stroke:#2E86DE,color:#fff")
    mermaid_lines.append("    classDef dbStyle fill:#1a6b3a,stroke:#27AE60,color:#fff")
    mermaid_lines.append("    classDef extStyle fill:#7d3c98,stroke:#9b59b6,color:#fff")
    for node in arch.get("nodes", []):
        nid = re.sub(r'[^A-Z0-9_]', '_', node["id"].upper())
        ntype = node.get("type", "service")
        if ntype == "database":
            mermaid_lines.append(f"    class {nid} dbStyle")
        elif ntype == "external":
            mermaid_lines.append(f"    class {nid} extStyle")
        else:
            mermaid_lines.append(f"    class {nid} serviceStyle")

    arch["mermaid"] = "\n".join(mermaid_lines)

    print(f"  ✅ Architecture: {len(arch['nodes'])} nodes, {len(arch['edges'])} edges")
=======
    print(f"  ✅ Architecture: {len(arch.get('nodes', []))} nodes, {len(arch.get('edges', []))} edges")
>>>>>>> origin/main
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