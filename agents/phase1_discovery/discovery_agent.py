import os
import sys
import json
import re
import yaml


from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from knowledge_layer.repo_summary import build_repo_summary
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

# CLAUDE FIX: signature updated with label: str = "json"
def safe_parse_json(raw: str, label: str = "json") -> dict:
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

def _unwrap_artifact(artifact: dict) -> dict:
    """
    The ASP system prompts return {"type":..., "title":..., "body":{...}, "unit_counts":..., ...}.
    Downstream code (UI, exporters, critic) expects the body's fields at the TOP LEVEL.
    This flattens body.* up while preserving title and metadata.
    """
    if not isinstance(artifact, dict) or not artifact:
        return artifact or {}
    body = artifact.get("body")
    if not isinstance(body, dict):
        return artifact  # already flat (fallback or non-ASP shape)
    flat = {**body}                                  # body fields become top-level
    if artifact.get("title") and "title" not in flat:
        flat["title"] = artifact["title"]
    # Preserve ASP metadata for the ExpansionDecisionEngine downstream
    for meta_key in ("type", "category", "confidence", "marginal_benefit",
                     "unit_counts", "evidence_links", "traces_to"):
        if meta_key in artifact:
            flat[meta_key] = artifact[meta_key]
    return flat
# -----------------------------------------
# Nodes
# -----------------------------------------

def classify_intent(state: DiscoveryState) -> DiscoveryState:
    """Phase 0.5 — Produces Adaptive Scope Profile (ASP) using REAL repo_summary"""
    print("[Phase 0.5] Classifying intent + producing Adaptive Scope Profile...")
    
    requirement = state["requirement"]
    selected_repos = state.get("selected_repos", [])
    is_new = state.get("is_new_project", True)

    # ── REAL REPO SUMMARY (NO MORE MOCKS!) ──
    candidate_repo_name = selected_repos[0]["name"] if selected_repos else None
    if is_new:
        repo_summary = {
            "exists": False,
            "matched_repo": candidate_repo_name,
            "match_score": 0.0,
            "languages": [],
            "symbol_overlap": 0.0,
            "tests_present": False,
            "top_symbols": []
        }
    else:
        # Query the actual knowledge layer (Qdrant & Postgres)!
        repo_summary = build_repo_summary(requirement, candidate_repo_name=candidate_repo_name)

    user_msg = f"REQUIREMENT:\n{requirement}\n\nREPO_SUMMARY:\n{json.dumps(repo_summary, indent=2)}"

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

    parsed = safe_parse_json(response.choices[0].message.content, "classifier")

    if parsed.get("ambiguity_detected") or not parsed.get("asp"):
        print("  ⚠️ Ambiguity detected. Using fallback ASP.")
        parsed["asp"] = {
            "depth_level": 3,
            "policy_mode": "managed",
            "allow_unbounded": False,
            "anchors": {
                "primary_domain": state["requirement"][:50],
                "core_capabilities": [state["requirement"][:100]],
                "user_types": ["primary user"],
                "explicit_integrations": [],
                "explicit_compliance": [],
                "production_intent": False
            },
            "unit_budgets": {"FR":10, "NFR":4, "ADR":5, "ARCH_NODE":8, "JIRA":15, "CODE_FILE":20, "RISK":4, "KPI":3, "SPRINT":2},
            "forbidden_elements": ["microservices", "kafka", "kubernetes", "multi_region", "service_mesh"],
            "mandatory_elements": []
        }

    contract = parsed["asp"]

    # Inject repo_summary into the ASP so downstream agents (codegen)
    # can find asp.repo_summary.matched_repo and fetch existing code.
    contract["repo_summary"] = repo_summary
    contract["_is_new_project"] = is_new
    contract.setdefault("build_mode",
                        "modify_existing" if (repo_summary.get("exists") and
                                              repo_summary.get("symbol_overlap", 0) > 0.3)
                        else "greenfield")

    print(f"  ✅ Depth: {contract.get('depth_level')} | Policy: {contract.get('policy_mode')} "
          f"| build_mode: {contract['build_mode']}")

    return {
        **state,
        "classifier_output": parsed,
        "scope_contract": contract,
        "status": "INTENT_CLASSIFIED",
    }
# CLAUDE FIX: Moved Fallback functions out to Module Level
def _fallback_asp(requirement: str, depth: int = 3) -> dict:
    return {
        "depth_level": depth,
        "policy_mode": "managed",
        "allow_unbounded": False,
        "anchors": {
            "primary_domain": requirement[:50],
            "core_capabilities": [requirement[:100]],
            "user_types": ["primary user"],
            "explicit_integrations": [],
            "explicit_compliance": [],
            "production_intent": False
        },
        "unit_budgets": {"FR":10, "NFR":4, "ADR":5, "ARCH_NODE":8, "JIRA":15, "CODE_FILE":20, "RISK":4, "KPI":3, "SPRINT":2},
        "forbidden_elements": ["microservices", "kafka", "kubernetes", "multi_region", "service_mesh"],
        "mandatory_elements": []
    }

def _make_fallback_brd(requirement: str, contract: dict) -> dict:
    return {"title": requirement[:80], "functional_requirements": []}


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
    brd = _unwrap_artifact(_llm_json(prompt))
    if not brd or not brd.get("title"):
        brd = {"title": state["requirement"][:80], "functional_requirements": []}

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
    prd = _unwrap_artifact(_llm_json(prompt))
    if not prd or not prd.get("functional_requirements"):
        prd = {"title": state["requirement"][:80], "functional_requirements": []}

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
    adr = _unwrap_artifact(_llm_json(prompt))
    if not adr or "decisions" not in adr:
        adr = {"decisions": []}

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
    arch = _unwrap_artifact(_llm_json(prompt))
    if not arch:
        arch = {"system_name": state["requirement"][:50], "nodes": [], "edges": []}

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
    # Static interrupt configured at compile-time (interrupt_before=...)
    # This node body just passes state through.
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
    
    g.add_edge("generate_brd", "generate_prd")
    g.add_edge("generate_prd", "generate_adr")
    g.add_edge("generate_adr", "generate_architecture")
    g.add_edge("generate_architecture", "human_approval_gate")
    g.add_edge("human_approval_gate", "process_approval")
    

    g.add_edge("process_approval", END)

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_approval_gate"],
    )
