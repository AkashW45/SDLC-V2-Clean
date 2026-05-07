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
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class DiscoveryState(TypedDict):
    requirement: str
    brd: Dict[str, Any]
    prd: Dict[str, Any]
    adr: Dict[str, Any]
    architecture: Dict[str, Any]
    human_feedback: str
    approved: bool
    status: str


def _llm_json(prompt: str, max_tokens: int = 4096) -> dict:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
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
            return {}


def _feedback_block(state) -> str:
    fb = state.get("human_feedback", "")
    if fb:
        return f"\n\nIMPORTANT — USER FEEDBACK on previous attempt (you MUST address all of this):\n{fb}\n"
    return ""


def generate_brd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating BRD...")

    brd = _llm_json(f"""
You are a Senior Business Analyst.
Generate a comprehensive BRD as JSON.
{_feedback_block(state)}
Return ONLY valid JSON:
{{
  "title": "Project title",
  "executive_summary": "2-3 paragraph summary",
  "business_objectives": ["objective 1", "objective 2", "..."],
  "in_scope": ["item 1", "item 2", "..."],
  "out_of_scope": ["item 1", "..."],
  "stakeholders": ["stakeholder 1", "..."],
  "functional_requirements": [
    {{"id": "FR1", "title": "...", "description": "...", "priority": "High"}}
  ],
  "non_functional_requirements": [
    {{"id": "NFR1", "title": "...", "description": "...", "priority": "High"}}
  ],
  "assumptions": ["assumption 1", "..."],
  "risks": ["risk 1", "..."],
  "success_criteria": ["criterion 1", "..."]
}}

Requirement:
{state['requirement']}
""")

    if not brd.get("title"):
        brd["title"] = "Untitled Project"

    print(f"  ✅ BRD generated: {brd.get('title')}")
    return {**state, "brd": brd, "status": "BRD_GENERATED"}


def generate_prd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating PRD...")

    brd = state["brd"]
    prd = _llm_json(f"""
You are a Senior Product Owner.
Generate a detailed PRD from this BRD.
{_feedback_block(state)}
Return ONLY valid JSON:
{{
  "title": "...",
  "product_vision": "one paragraph",
  "functional_requirements": [
    {{"id": "FR1", "title": "...", "description": "...", "priority": "High",
      "acceptance_criteria": ["AC1", "AC2", "AC3"]}}
  ],
  "non_functional_requirements": [
    {{"id": "NFR1", "title": "...", "description": "...", "priority": "High"}}
  ],
  "technical_requirements": [
    {{"id": "TR1", "title": "...", "description": "..."}}
  ],
  "stakeholders": ["..."],
  "scope": "..."
}}

Minimum 5 FRs, 4 NFRs, 3 TRs.

BRD Title: {brd.get('title','')}
BRD Objectives: {json.dumps(brd.get('business_objectives', []))}
BRD Functional Requirements: {json.dumps(brd.get('functional_requirements', []))}
BRD Non-Functional Requirements: {json.dumps(brd.get('non_functional_requirements', []))}
""")

    if not prd.get("title"):
        prd["title"] = brd.get("title", "Untitled")
    for k in ["functional_requirements", "non_functional_requirements", "technical_requirements"]:
        if not prd.get(k):
            prd[k] = []

    print(f"  ✅ PRD: {prd['title']} — {len(prd['functional_requirements'])} FRs, {len(prd['non_functional_requirements'])} NFRs")
    return {**state, "prd": prd, "status": "PRD_GENERATED"}


def generate_adr(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating ADR...")

    prd = state["prd"]
    adr = _llm_json(f"""
You are a Senior Software Architect.
Generate Architecture Decision Records (ADRs) for this product.
{_feedback_block(state)}
Return ONLY valid JSON:
{{
  "decisions": [
    {{
      "id": "ADR-001",
      "title": "Decision title",
      "context": "Why this decision was needed",
      "decision": "What was decided",
      "consequences": ["consequence 1", "consequence 2"],
      "status": "Accepted"
    }}
  ]
}}

Minimum 5 decisions covering: tech stack, data store, deployment platform,
authentication, monitoring/observability.

Project: {prd.get('title','')}
Functional Requirements: {json.dumps(prd.get('functional_requirements', [])[:5])}
Non-Functional Requirements: {json.dumps(prd.get('non_functional_requirements', [])[:5])}
""")

    if not adr.get("decisions"):
        adr["decisions"] = []

    print(f"  ✅ ADR generated: {len(adr['decisions'])} decisions")
    return {**state, "adr": adr, "status": "ADR_GENERATED"}


def generate_architecture(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating architecture design...")

    prd = state["prd"]

    # Build canonical requirement list (V1 pattern)
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

    project_name = prd.get("title", "SYSTEM")

    arch = _llm_json(f"""
You are a strict software architect.
{_feedback_block(state)}
RULES:
1. Every node MUST be justified by a functional or non-functional requirement
2. Do NOT invent technologies not implied by requirements
3. 4-8 nodes maximum
4. Every node MUST include "traced_to" — the exact requirement text it fulfils
5. Return ONLY valid JSON

Project: {project_name}

Functional Requirements:
{json.dumps(functional_reqs, indent=2)}

Non-Functional Requirements:
{json.dumps(non_functional_reqs, indent=2)}

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

    # Generate mermaid deterministically (V1 pattern)
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
            mermaid_lines.append(f'  {nid}{{{{"{name}"}}}}')
        else:
            mermaid_lines.append(f'  {nid}["{name}"]')

    for edge in arch.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        proto = edge.get("protocol", "")
        mermaid_lines.append(f'  {src} -->|{proto}| {tgt}')

    arch["mermaid"] = "\n".join(mermaid_lines)

    print(f"  ✅ Architecture: {len(arch['nodes'])} nodes, {len(arch['edges'])} edges")
    return {**state, "architecture": arch, "status": "ARCHITECTURE_GENERATED"}


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
    g.add_node("generate_brd", generate_brd)
    g.add_node("generate_prd", generate_prd)
    g.add_node("generate_adr", generate_adr)
    g.add_node("generate_architecture", generate_architecture)
    g.add_node("human_approval_gate", human_approval_gate)
    g.add_node("process_approval", process_approval)

    g.set_entry_point("generate_brd")
    g.add_edge("generate_brd", "generate_prd")
    g.add_edge("generate_prd", "generate_adr")
    g.add_edge("generate_adr", "generate_architecture")
    g.add_edge("generate_architecture", "human_approval_gate")
    g.add_edge("human_approval_gate", "process_approval")
    g.add_edge("process_approval", END)

    return g.compile(checkpointer=MemorySaver())