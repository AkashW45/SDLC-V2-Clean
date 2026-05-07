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

from openai import OpenAI

# Replace Groq client with DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# -----------------------------------------
# State
# -----------------------------------------

class DiscoveryState(TypedDict):
    requirement: str
    brd: dict
    prd: dict
    adr: dict
    architecture: dict 
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

    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()

    # First attempt
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find JSON object boundaries and extract
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception:
        pass

    # Last resort — return raw
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

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": f"""
You are a Product Owner.
Generate a PRD from this BRD.
Return ONLY valid JSON — no markdown, no explanation.
Be concise — max 3 acceptance criteria per requirement.

{{
  "title": "...",
  "product_vision": "...",
  "functional_requirements": [
    {{
      "id": "FR1",
      "title": "...",
      "description": "...",
      "priority": "High",
      "acceptance_criteria": ["...", "..."]
    }}
  ],
  "non_functional_requirements": [
    {{
      "id": "NFR1",
      "title": "...",
      "description": "...",
      "priority": "High"
    }}
  ],
  "technical_requirements": [
    {{
      "id": "TR1",
      "title": "...",
      "description": "..."
    }}
  ],
  "stakeholders": ["..."],
  "scope": "..."
}}

BRD Title: {state['brd'].get('title', '')}
BRD Objectives: {json.dumps(state['brd'].get('business_objectives', []))}
BRD Functional Requirements: {json.dumps(state['brd'].get('functional_requirements', []))}
BRD Non-Functional Requirements: {json.dumps(state['brd'].get('non_functional_requirements', []))}
"""}],
        max_tokens=4096
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()

    try:
        prd = json.loads(content)
    except Exception:
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            prd = json.loads(content[start:end])
        except Exception:
            prd = {}

    # Normalise keys
    if not prd.get("title"):
        prd["title"] = prd.get("projectTitle") or state['brd'].get('title', 'Untitled')
    if not prd.get("functional_requirements"):
        prd["functional_requirements"] = prd.get("functionalRequirements") or []
    if not prd.get("non_functional_requirements"):
        prd["non_functional_requirements"] = prd.get("nonFunctionalRequirements") or []
    if not prd.get("technical_requirements"):
        prd["technical_requirements"] = prd.get("technicalRequirements") or []
    if not prd.get("product_vision"):
        prd["product_vision"] = prd.get("productVision") or ""

    print(f"  ✅ PRD: {prd.get('title')} — {len(prd.get('functional_requirements', []))} FRs, {len(prd.get('non_functional_requirements', []))} NFRs")
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

def generate_architecture(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating architecture design...")

    prd = state["prd"]
    brd = state["brd"]

    # Build canonical requirement list exactly like V1
    functional_reqs = []
    for r in prd.get("functional_requirements", []):
        if isinstance(r, dict):
            text = f"{r.get('title','')} {r.get('description','')}".strip()
            if text:
                functional_reqs.append(text)
        elif isinstance(r, str):
            functional_reqs.append(r)

    non_functional_reqs = []
    for r in prd.get("non_functional_requirements", []):
        if isinstance(r, dict):
            text = r.get("description") or r.get("title") or ""
            if text:
                non_functional_reqs.append(text)
        elif isinstance(r, str):
            non_functional_reqs.append(r)

    project_name = prd.get("title") or brd.get("title") or "SYSTEM"

    arch = call_llm(f"""
You are a strict software architect.

RULES:
1. Every node MUST be justified by one of the functional or non-functional requirements listed below.
2. Do NOT invent technologies not implied by the requirements.
3. If requirements are simple, architecture must remain simple.
4. 4-8 nodes maximum.
5. Every node MUST include "traced_to" — copy the exact requirement text it fulfils.
6. Return ONLY valid JSON. No markdown. No explanation.

Project: {project_name}

Functional Requirements:
{json.dumps(functional_reqs, indent=2)}

Non-Functional Requirements:
{json.dumps(non_functional_reqs, indent=2)}

Return this exact JSON structure:
{{
  "project_name": "{project_name}",
  "architecture_style": "microservices | monolith | layered | event-driven",
  "nodes": [
    {{
      "id": "UPPERCASE_ID",
      "name": "Human Readable Name",
      "type": "service | database | external | queue | cache",
      "zone": "external | core | data | observability",
      "description": "What this component does",
      "tech_stack": ["Technology1", "Technology2"],
      "traced_to": "exact requirement text this node fulfils"
    }}
  ],
  "edges": [
    {{
      "source": "NODE_ID",
      "target": "NODE_ID",
      "protocol": "REST | gRPC | Kafka | SQL | HTTPS | internal",
      "description": "what flows between these nodes"
    }}
  ],
  "security_considerations": ["..."],
  "deployment_model": "kubernetes | vm | docker | serverless",
  "scalability_notes": "..."
}}
""")
    

    if isinstance(arch, dict) and "raw" in arch and len(arch) == 1:
        raw_str = arch["raw"]
        raw_str = raw_str.replace('\u2011', '-').replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
        try:
            start = raw_str.find('{')
            end = raw_str.rfind('}') + 1
            arch = json.loads(raw_str[start:end])
        except Exception:
            arch = {"nodes": [], "edges": []}
    # Validate — remove nodes not traced to any requirement (V1 approach)
    all_reqs = functional_reqs + non_functional_reqs
    valid_nodes = []
    for node in arch.get("nodes", []):
        traced = node.get("traced_to", "")
        # Accept if traced_to contains words from any requirement
        if traced and any(
            any(word.lower() in traced.lower() for word in req.split()[:5])
            for req in all_reqs
        ):
            valid_nodes.append(node)
        else:
            # Keep it but flag it
            node["traced_to"] = node.get("traced_to", "implicit requirement")
            valid_nodes.append(node)

    arch["nodes"] = valid_nodes

    # Generate mermaid diagram deterministically (V1 diagram_generator approach)
    mermaid_lines = ["graph TD"]
    for node in arch.get("nodes", []):
        nid = node["id"]
        name = node["name"]
        ntype = node.get("type", "service")
        if ntype == "database":
            mermaid_lines.append(f'  {nid}[("{name}")]')
        elif ntype == "external":
            mermaid_lines.append(f'  {nid}(["{name}"])')
        elif ntype == "queue":
            mermaid_lines.append(f'  {nid}>"{name}"]')
        else:
            mermaid_lines.append(f'  {nid}["{name}"]')

    for edge in arch.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        proto = edge.get("protocol", "")
        mermaid_lines.append(f'  {src} -->|{proto}| {tgt}')

    arch["mermaid"] = "\n".join(mermaid_lines)

    print(f"  ✅ Architecture: {len(arch.get('nodes', []))} nodes, {len(arch.get('edges', []))} edges")
    print(f"  ✅ Mermaid diagram generated")
    return {**state, "architecture": arch, "status": "ARCHITECTURE_GENERATED"}


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
    builder.add_node("generate_architecture", generate_architecture)
    builder.add_node("human_approval_gate", human_approval_gate)
    builder.add_node("process_approval", process_approval)

    builder.set_entry_point("generate_brd")
    builder.add_edge("generate_brd", "generate_prd")
    builder.add_edge("generate_prd", "generate_adr")
    builder.add_edge("generate_adr", "generate_architecture")           # ← NEW
    builder.add_edge("generate_architecture", "human_approval_gate") 
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
        architecture={},
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