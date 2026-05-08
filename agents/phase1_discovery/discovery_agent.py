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


def _llm_json(prompt: str, max_tokens: int = 16384) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
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

def _feedback_block(state) -> str:
    fb = state.get("human_feedback", "")
    if fb:
        return f"\n\nIMPORTANT — USER FEEDBACK on previous attempt (you MUST address all of this):\n{fb}\n"
    return ""


def generate_brd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating BRD...")

    brd = _llm_json(f"""
You are a Senior Business Analyst writing a BRD for executive review.
The BRD must read like a McKinsey document — specific, quantified, no fluff.
{_feedback_block(state)}
Return ONLY valid JSON with this exact structure:

{{
  "title": "Project title",
  "executive_summary": "3-paragraph executive summary covering: (1) the problem and its business impact in numbers, (2) the proposed solution and how it addresses the problem, (3) expected outcome and ROI",
  "business_context": "1-2 paragraphs on industry context, market drivers, competitive pressure",
  "business_objectives": [
    "Quantified objective with target metric (e.g., 'Reduce manual processing time by 40%')",
    "Another quantified objective"
  ],
  "in_scope": ["Specific deliverable 1", "..."],
  "out_of_scope": ["Specific exclusion 1", "..."],
  "stakeholders": [
    {{"role": "Sponsor", "name_or_team": "...", "responsibility": "..."}},
    {{"role": "Product Owner", "name_or_team": "...", "responsibility": "..."}}
  ],
  "raci_matrix": [
    {{"activity": "Requirements Sign-off", "responsible": "Business Analyst", "accountable": "Product Owner", "consulted": "Engineering Lead", "informed": "Executive Sponsor"}},
    {{"activity": "Architecture Review", "responsible": "Solution Architect", "accountable": "CTO", "consulted": "Security Team", "informed": "Product Owner"}},
    {{"activity": "Production Deployment", "responsible": "DevOps", "accountable": "Engineering Manager", "consulted": "QA Lead", "informed": "All Stakeholders"}}
  ],
  "functional_requirements": [
    {{"id": "FR1", "title": "...", "description": "Detailed paragraph", "priority": "Critical|High|Medium|Low", "business_value": "How it addresses business goal"}}
  ],
  "non_functional_requirements": [
    {{"id": "NFR1", "title": "...", "description": "Quantified target (e.g., '99.9% uptime', '<200ms p95 latency')", "priority": "High", "metric": "specific KPI"}}
  ],
  "kpis": [
    {{"name": "KPI name", "target": "specific number", "measurement_method": "how measured", "frequency": "daily/weekly/monthly"}}
  ],
  "risk_matrix": [
    {{"id": "R1", "risk": "Risk description", "likelihood": "High|Medium|Low", "impact": "High|Medium|Low", "mitigation": "specific mitigation strategy", "owner": "team responsible"}}
  ],
  "assumptions": ["..."],
  "dependencies": ["External system X must be available", "..."],
  "success_criteria": ["Specific measurable outcome 1", "..."],
  "timeline_estimate": "X weeks/months with major milestones",
  "budget_considerations": "infrastructure, licensing, team capacity"
}}

Minimum requirements:
- 4+ functional requirements with business value
- 5+ NFRs with quantified targets
- 4+ KPIs with specific numbers
- 6+ risks with likelihood/impact/mitigation
- 5+ RACI activities
- Stakeholders with at least 5 distinct roles

REQUIREMENT:
{state['requirement']}
""")

    if not brd.get("title"):
        brd["title"] = "Untitled Project"

    print(f"  ✅ BRD generated: {brd.get('title')} ({len(brd.get('functional_requirements',[]))} FRs, {len(brd.get('risk_matrix',[]))} risks)")
    return {**state, "brd": brd, "status": "BRD_GENERATED"}

def generate_prd(state: DiscoveryState) -> DiscoveryState:
    print("\n[Phase 1] Generating PRD...")

    brd = state["brd"]
    prd = _llm_json(f"""
You are a Senior Product Owner writing a PRD for engineering execution.
This must read like a Stripe/Linear PRD — specific user stories, not vague generalities.
{_feedback_block(state)}
Return ONLY valid JSON:

{{
  "title": "...",
  "executive_summary": "2-paragraph summary for product leadership",
  "product_vision": "Clear vision statement — who, what, why",
  "target_users": [
    {{"persona": "User type", "needs": "what they need", "pain_points": "current problems"}}
  ],
  "user_journeys": [
    {{"journey": "End-to-end flow name", "steps": ["step 1", "step 2"]}}
  ],
  "functional_requirements": [
    {{
      "id": "FR1",
      "title": "...",
      "user_story": "As a [persona] I want [goal] so that [benefit]",
      "description": "Detailed description",
      "priority": "P0|P1|P2",
      "acceptance_criteria": ["Given...When...Then..."],
      "edge_cases": ["edge case 1", "edge case 2"],
      "dependencies": ["depends on FR2"]
    }}
  ],
  "non_functional_requirements": [
    {{"id": "NFR1", "title": "...", "description": "Quantified", "priority": "High", "verification_method": "load test / monitoring / etc"}}
  ],
  "technical_requirements": [
    {{"id": "TR1", "title": "...", "description": "Specific technical constraint", "rationale": "why this constraint exists"}}
  ],
  "success_metrics": [
    {{"metric": "...", "baseline": "current value", "target": "desired value", "timeline": "when measured"}}
  ],
  "release_phases": [
    {{"phase": "MVP", "timeline": "Week 1-4", "scope": ["FR1", "FR2"]}},
    {{"phase": "Phase 2", "timeline": "Week 5-8", "scope": ["FR3"]}}
  ],
  "open_questions": ["question that needs decision"]
}}

Minimum:
- 6+ FRs with user stories AND edge cases
- 4+ NFRs with verification methods
- 4+ TRs with rationale
- 3+ user personas
- 4+ success metrics with baseline AND target
- 3+ release phases

BRD Context:
Title: {brd.get('title','')}
Summary: {brd.get('executive_summary','')[:500]}
Objectives: {json.dumps(brd.get('business_objectives', []))}
KPIs: {json.dumps(brd.get('kpis', []))}
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
You are a Senior Software Architect writing ADRs in MADR format.
Each ADR must include alternatives considered with trade-offs — not just the chosen path.
{_feedback_block(state)}
Return ONLY valid JSON:

{{
  "decisions": [
    {{
      "id": "ADR-001",
      "title": "Specific architectural decision",
      "status": "Accepted|Proposed|Deprecated",
      "date": "2026-05-08",
      "context": "Why this decision was needed — 2-3 sentences with specific drivers",
      "decision": "What was decided — be specific (technology, pattern, approach)",
      "rationale": "Why this option over others — paragraph",
      "alternatives_considered": [
        {{"option": "Alt 1", "pros": ["pro 1"], "cons": ["con 1"], "rejected_because": "specific reason"}}
      ],
      "consequences": {{
        "positive": ["positive consequence 1", "..."],
        "negative": ["negative consequence 1", "..."],
        "neutral": ["neutral implication"]
      }},
      "compliance_notes": "any regulatory/security implications",
      "supersedes": "ADR-XXX or null"
    }}
  ]
}}

Minimum 7 decisions covering:
1. Tech stack (language/framework)
2. Data store (type and rationale)
3. Deployment platform (cloud/on-prem/hybrid)
4. Authentication & authorization approach
5. Observability strategy (logging/monitoring/tracing)
6. API design (REST/GraphQL/gRPC)
7. Async processing (queue/streaming/batch)

Each MUST have at least 2 alternatives_considered with detailed pros/cons.

PRD Context:
Project: {prd.get('title','')}
NFRs: {json.dumps(prd.get('non_functional_requirements', [])[:6])}
Tech Requirements: {json.dumps(prd.get('technical_requirements', [])[:6])}
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