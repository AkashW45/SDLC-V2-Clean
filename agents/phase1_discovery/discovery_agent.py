import os
import sys
import json
import re
import yaml
sys.path.insert(0, r"C:\Users\user\SDLC-V2")

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv

from agents.prompts.system_prompts import (
    CLASSIFIER_SYSTEM, BRD_SYSTEM, PRD_SYSTEM,
    ADR_SYSTEM, ARCHITECTURE_SYSTEM
)
from agents.critic.critic_agent import critique

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-v4-flash"

class DiscoveryState(TypedDict):
    requirement: str
    scope_contract: dict
    classifier_output: dict
    brd: Dict[str, Any]
    prd: Dict[str, Any]
    adr: Dict[str, Any]
    architecture: Dict[str, Any]
    human_feedback: str
    approved: bool
    status: str


# -----------------------------------------
# Merged Helpers
# -----------------------------------------

def _llm_json(prompt: str, max_tokens: int = 8192) -> dict:
    """Combines Shantanu's strict JSON generation with native retries."""
    system_prompt = "You are a strict JSON outputter. Respond ONLY with valid JSON. Do not include markdown formatting like ```json."

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"}, # Strict JSON enforcement
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  [LLM] JSON decode error on attempt {attempt+1}/3: {e}")
            if attempt == 2:
                return {}

def safe_parse_json(raw: str) -> dict:
    if not raw: return {}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}

def _feedback_block(state) -> str:
    fb = state.get("human_feedback", "")
    return f"\n\nIMPORTANT — USER FEEDBACK on previous attempt:\n{fb}\n" if fb else ""

def _extract_yaml_context(artifact: dict, keys: list) -> str:
    """Replicates Shantanu's COE: Extracts specific keys and converts to YAML for token savings."""
    if not artifact: return ""
    extracted = {k: artifact.get(k) for k in keys if k in artifact}
    return yaml.dump(extracted, sort_keys=False, default_flow_style=False)


# -----------------------------------------
# Nodes
# -----------------------------------------

def classify_intent(state: DiscoveryState) -> DiscoveryState:
    """Phase 0.5 — Produces Scope Contract (From Main)"""
    print("[Phase 0.5] Classifying intent + producing Scope Contract...")
    user_msg = f"PROJECT_REQUEST: {state['requirement']}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=4096,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    parsed = safe_parse_json(response.choices[0].message.content)

    if parsed.get("ambiguity_detected") or not parsed.get("scope_contract"):
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
            "hard_limits": {"max_functional_requirements": 10, "max_architecture_nodes": 8, "max_jira_tickets": 15, "max_code_files": 20, "max_sprints": 2},
            "forbidden_elements": ["microservices", "kafka", "kubernetes", "multi_region", "service_mesh"],
            "mandatory_elements": []
        }

    contract = parsed["scope_contract"]
    print(f"  ✅ Depth: {contract['depth_level']} | Domain: {contract['scope_anchor']['primary_domain']}")

    return {**state, "classifier_output": parsed, "scope_contract": contract, "status": "INTENT_CLASSIFIED"}


def generate_brd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating BRD...")
    contract = state.get("scope_contract", {})

    prompt = f"""
{BRD_SYSTEM}
{_feedback_block(state)}

SCOPE CONTRACT ENFORCEMENT:
{json.dumps(contract, indent=2)}

REQUIREMENT:
{state['requirement']}
"""
    brd = _llm_json(prompt)
    if not brd: brd = {"title": state["requirement"][:80], "functional_requirements": []}

    verdict = critique(brd, "brd", contract, state["requirement"])
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join([v["fix_instruction"] for v in verdict.get("violations", [])])
        return generate_brd(state)

    print(f"  ✅ BRD generated: {brd.get('title', 'Untitled')}")
    return {**state, "brd": brd, "status": "BRD_GENERATED"}


def generate_prd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating PRD...")
    contract = state.get("scope_contract", {})

    # Shantanu's Optimization: Compress BRD to YAML
    brd_context = _extract_yaml_context(state.get("brd", {}), ["title", "business_objectives", "kpis"])

    prompt = f"""
{PRD_SYSTEM}
{_feedback_block(state)}

SCOPE CONTRACT ENFORCEMENT:
{json.dumps(contract, indent=2)}

BRD CONTEXT (YAML):
{brd_context}

REQUIREMENT: {state['requirement']}
"""
    prd = _llm_json(prompt)
    if not prd: prd = {"title": state["requirement"][:80], "functional_requirements": []}

    verdict = critique(prd, "prd", contract, state["requirement"], prior_artifacts={"brd": state.get("brd")})
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join([v["fix_instruction"] for v in verdict.get("violations", [])])
        return generate_prd(state)

    print(f"  ✅ PRD generated: {len(prd.get('functional_requirements', []))} FRs")
    return {**state, "prd": prd, "status": "PRD_GENERATED"}


def generate_adr(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating ADR...")
    contract = state.get("scope_contract", {})

    # Shantanu's Optimization: Compress PRD to YAML
    prd_context = _extract_yaml_context(state.get("prd", {}), ["title", "non_functional_requirements", "technical_requirements"])

    prompt = f"""
{ADR_SYSTEM}
{_feedback_block(state)}

SCOPE CONTRACT ENFORCEMENT:
{json.dumps(contract, indent=2)}

PRD CONTEXT (YAML):
{prd_context}

REQUIREMENT: {state['requirement']}
"""
    adr = _llm_json(prompt)
    if not adr: adr = {"decisions": []}

    verdict = critique(adr, "adr", contract, state["requirement"])
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join([v["fix_instruction"] for v in verdict.get("violations", [])])
        return generate_adr(state)

    print(f"  ✅ ADR generated: {len(adr.get('decisions', []))} decisions")
    return {**state, "adr": adr, "status": "ADR_GENERATED"}


def generate_architecture(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating architecture design...")
    contract = state.get("scope_contract", {})
    prd = state.get("prd", {})

    # Compress PRD and ADR
    prd_context = _extract_yaml_context(prd, ["title", "functional_requirements", "non_functional_requirements"])
    adr_context = _extract_yaml_context(state.get("adr", {}), ["decisions"])

    prompt = f"""
{ARCHITECTURE_SYSTEM}
{_feedback_block(state)}

SCOPE CONTRACT ENFORCEMENT:
{json.dumps(contract, indent=2)}

PRD CONTEXT (YAML):
{prd_context}

ADR CONTEXT (YAML):
{adr_context}
"""
    arch = _llm_json(prompt)
    if not arch: arch = {"nodes": [], "edges": []}

    # Shantanu's Beautiful Mermaid Logic
    mermaid_lines = ["graph TD"]
    for node in arch.get("nodes", []):
        nid = re.sub(r'[^A-Z0-9_]', '_', node.get("id", "").upper())
        name = node.get("name", "").replace('"', "'")[:40]
        ntype = node.get("type", "service")

        if ntype == "database": mermaid_lines.append(f'    {nid}[("{name}")]')
        elif ntype == "external": mermaid_lines.append(f'    {nid}(["{name}"])')
        elif ntype == "queue": mermaid_lines.append(f'    {nid}>"{name}"]')
        elif ntype == "cache": mermaid_lines.append(f'    {nid}[/"{name}"/]')
        else: mermaid_lines.append(f'    {nid}["{name}"]')

    for edge in arch.get("edges", []):
        src = re.sub(r'[^A-Z0-9_]', '_', edge.get("source", "").upper())
        tgt = re.sub(r'[^A-Z0-9_]', '_', edge.get("target", "").upper())
        proto = edge.get("protocol", "").replace('|', '/')[:15]
        if proto: mermaid_lines.append(f'    {src} -->|{proto}| {tgt}')
        else: mermaid_lines.append(f'    {src} --> {tgt}')

    mermaid_lines.extend([
        "    classDef serviceStyle fill:#1a4a8a,stroke:#2E86DE,color:#fff",
        "    classDef dbStyle fill:#1a6b3a,stroke:#27AE60,color:#fff",
        "    classDef extStyle fill:#7d3c98,stroke:#9b59b6,color:#fff"
    ])
    for node in arch.get("nodes", []):
        nid = re.sub(r'[^A-Z0-9_]', '_', node.get("id", "").upper())
        ntype = node.get("type", "service")
        if ntype == "database": mermaid_lines.append(f"    class {nid} dbStyle")
        elif ntype == "external": mermaid_lines.append(f"    class {nid} extStyle")
        else: mermaid_lines.append(f"    class {nid} serviceStyle")

    arch["mermaid"] = "\n".join(mermaid_lines)

    # Validate
    verdict = critique(arch, "architecture", contract, state["requirement"])
    if verdict.get("verdict") == "REGENERATE":
        state["human_feedback"] = "\n".join([v["fix_instruction"] for v in verdict.get("violations", [])])
        return generate_architecture(state)

    print(f"  ✅ Architecture: {len(arch.get('nodes', []))} nodes")
    return {**state, "architecture": arch, "status": "ARCHITECTURE_GENERATED"}


def human_approval_gate(state: DiscoveryState) -> DiscoveryState:
    interrupt({
        "type": "DISCOVERY_REVIEW",
        "brd": state.get("brd"),
        "prd": state.get("prd"),
        "adr": state.get("adr"),
        "architecture": state.get("architecture")
    })
    return state


def process_approval(state: DiscoveryState) -> DiscoveryState:
    if state.get("approved"):
        return {**state, "status": "APPROVED_FOR_PLANNING"}
    return {**state, "status": "REJECTED"}


def build_discovery_graph():
    g = StateGraph(DiscoveryState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("generate_brd", generate_brd)
    g.add_node("generate_prd", generate_prd)
    g.add_node("generate_adr", generate_adr)
    g.add_node("generate_architecture", generate_architecture)
    g.add_node("human_approval_gate", human_approval_gate)
    g.add_node("process_approval", process_approval)

    g.set_entry_point("classify_intent")
    g.add_edge("classify_intent", "generate_brd")
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



    g.add_node("classify_intent", classify_intent)   # NEW
    g.add_edge("generate_brd", "generate_prd")
    g.add_edge("generate_prd", "generate_adr")
    g.add_edge("generate_adr", "generate_architecture")
    g.add_edge("generate_architecture", "human_approval_gate")
    g.add_edge("human_approval_gate", "process_approval")
    g.add_edge("process_approval", END)

    return g.compile(checkpointer=MemorySaver())