"""
Phase 4 — Code Generation Agent
Reads approved impact report and generates targeted code changes
for only the affected files identified in Phase 3.
Uses AST context from Knowledge Layer.
"""

import os
import ast
import json
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from agents.prompts.system_prompts import CODEGEN_SYSTEM
from agents.critic.critic_agent import critique
from agents.context_packet_builder import build_context_packet as build_context_packet_rag, inject_into_user_message
load_dotenv()
from agents.repo_workspace import ensure_repo_cloned, read_file as ws_read_file, get_repo_local_path

from api.persistence import audit

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# -----------------------------------------
# State
# -----------------------------------------

class CodegenState(TypedDict):
    requirement: str
    scope_contract: dict
    impact_report: dict
    adr: dict                  # Architecture Decision Record for tech stack enforcement
    existing_code: dict        # file_path -> current content
    generated_changes: list    # list of {file_path, content, change_summary}
    
   
    validation_errors: list
    status: str
    workspace_path: str
    thread_id: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str, max_tokens: int = 8192) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def get_postgres():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )


def get_symbols_for_file(repo_name: str, file_path: str) -> list:
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol_name, symbol_type, line_number, signature, docstring
        FROM symbols
        WHERE repo_name = %s AND file_path = %s
        ORDER BY line_number
    """, (repo_name, file_path))
    symbols = []
    for row in cur.fetchall():
        symbols.append({
            "name": row[0],
            "type": row[1],
            "line": row[2],
            "signature": row[3],
            "docstring": row[4]
        })
    cur.close()
    conn.close()
    return symbols


def read_file_from_repo(repo_path: str, file_path: str) -> str:
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  [!] Could not read {full_path}: {e}")
        return ""


def validate_python(code: str, file_path: str) -> list:
    """Validate Python syntax. Skip non-Python files (HTML, MD, JS, JSON, YAML, etc.)."""
    errors = []
    if not file_path.endswith(".py"):
        return errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError at line {e.lineno}: {e.msg}")
    return errors


def extract_file_outline(code: str) -> str:
    """Extract class and function names to provide file structure without full content."""
    outline = "[File Outline]\n"
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                outline += f"  class {node.name}\n"
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        outline += f"    def {item.name}(...)\n"
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                outline += f"  def {node.name}(...)\n"
    except SyntaxError:
        outline = "[Could not parse file outline]\n"
    return outline


def extract_relevant_context(code: str, affected_symbols: list) -> str:
    """
    Context Windowing: For files > 100 lines, extract only relevant functions/classes
    being modified plus a 20-line buffer. Otherwise return full code.
    """
    lines = code.split("\n")

    # If file is small, return everything
    if len(lines) <= 100:
        return code

    # Extract line ranges for affected symbols
    affected_ranges = []
    for sym in affected_symbols:
        sym_line = sym.get("line", 0) if isinstance(sym, dict) else 0
        if sym_line > 0:
            start = max(0, sym_line - 20)
            end = min(len(lines), sym_line + 20)
            affected_ranges.append((start, end))

    if not affected_ranges:
        return "\n".join(lines[:100])

    # Merge overlapping ranges
    affected_ranges.sort()
    merged_ranges = [affected_ranges[0]]
    for start, end in affected_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        if start <= last_end + 10:
            merged_ranges[-1] = (last_start, max(last_end, end))
        else:
            merged_ranges.append((start, end))

    # Extract windowed sections
    windowed_code = ""
    for start, end in merged_ranges:
        windowed_code += "\n".join(lines[start:end]) + "\n... (code omitted) ...\n"

    return windowed_code

def eval_first_check(state: CodegenState) -> CodegenState:
    """Check for goldenset.yaml before proceeding with code generation."""
    workspace_path = state.get('workspace_path', '')
    goldenset_path = os.path.join(workspace_path, 'goldenset.yaml')

    if os.path.exists(goldenset_path):
        audit(state.get('thread_id', 'unknown'), 'phase4', 'INFO', 'system', {
            'message': 'Eval-First Check Passed: goldenset.yaml found.'
        })
        print("  [Phase 4] Eval-First Check: PASSED (goldenset.yaml found)")
    else:
        audit(state.get('thread_id', 'unknown'), 'phase4', 'WARNING', 'system', {
            'message': 'Eval-First Check Failed: goldenset.yaml missing. Proceeding in SOFT MODE.'
        })
        print("  [Phase 4] Eval-First Check: WARNING (goldenset.yaml missing — soft mode)")

    return state

def load_existing_code(state: CodegenState) -> CodegenState:
    """
    Read current content of all affected files for brownfield codegen.
    For new projects (no affected files), skip entirely.
    For existing projects, auto-clone the target repo if needed, then read files
    with GitHub API fallback.
    """
    print("\n[Phase 4] Loading existing code...")

    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files", [])

    if not affected_files:
        print("  [Phase 4] No existing files — NEW PROJECT, generating fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    # Group affected files by repo so we clone each repo at most once
    repos_seen = set()
    existing_code = {}

    for af in affected_files:
        repo_name = af.get("repo_name", "")
        file_path = af.get("file_path", "")
        if not repo_name or not file_path:
            continue

        if repo_name not in repos_seen:
            # Best-effort clone before first file read for that repo
            ensure_repo_cloned(repo_name)
            repos_seen.add(repo_name)

        content = ws_read_file(repo_name, file_path)
        if content:
            # Key by "<repo>:<path>" so codegen knows which repo each file lives in
            existing_code[f"{repo_name}:{file_path}"] = content
            print(f"  ✅ Loaded: {repo_name}/{file_path} ({len(content)} chars)")
        else:
            print(f"  ⚠️  Could not read: {repo_name}/{file_path} (both local + API failed)")

    if not existing_code:
        # Affected files declared but none readable — fall back to greenfield path
        # rather than crashing. The LLM will get nothing useful otherwise.
        print("  ⚠️  Phase 3 declared affected files but none could be read — treating as fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    return {**state, "existing_code": existing_code, "status": "CODE_LOADED"}

def generate_fresh_project(state: CodegenState) -> CodegenState:
        print("  [Phase 4] Generating FRESH Polyglot project scaffold...")

        requirement = state["requirement"]
        adr = state.get("adr", {})
        impact = state.get("impact_report", {})
        architecture = impact.get("architecture", {})

        # Extract ADR and Architecture context
        adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
        arch_context = ""
        if architecture.get("nodes"):
            arch_summary = [f"- {n.get('name','')} ({n.get('type','service')}): {n.get('description','')}" for n in architecture.get("nodes",[])]
            arch_context = "\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(arch_summary)

        prompt = f"""
    You are a Senior Polyglot Software Architect scaffolding a brand new multi-service project.

    ADR (Agreed Tech Stack):
    {adr_text}
    {arch_context}

    REQUIREMENT:
    {requirement}

    INSTRUCTIONS:
    1. Read the ADR and Architecture to determine the EXACT programming languages and frameworks required.
    2. Generate a complete, production-ready starter project scaffold for ALL requested nodes.
    3. Include standard configuration files appropriate for the chosen stack (e.g., package.json, tsconfig.json, pom.xml, or requirements.txt).
    4. Provide the core application entry points, routes/controllers, models, and a README.md.
    5. Prefix file paths with the service/repo name to keep them organized (e.g., 'backend/main.py' or 'frontend/src/App.tsx').
    6. CRITICAL: DO NOT default to Python unless explicitly specified in the ADR or Architecture.

    Return ONLY valid JSON in this exact format:
    {{
      "files": [
        {{
          "file_path": "path/to/file.ext",
          "content": "complete file content as a string",
          "change_summary": "what this file does",
          "new_symbols_added": ["ClassName", "function_name"],
          "existing_symbols_modified": []
        }}
      ]
    }}
    """

        response = call_llm(prompt, max_tokens=4000)
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

        try:
            data = json.loads(response)
            generated_changes = data.get("files", [])
        except Exception as e:
            print(f"  [Phase 4] JSON parse failed: {e}")
            generated_changes = []

        errors = []
        for change in generated_changes:
            errors.extend(validate_python(change.get("content", ""), change.get("file_path", "")))

        print(f"  [Phase 4] Generated {len(generated_changes)} fresh files")
        return {
            **state,
            "generated_changes": generated_changes,
            "validation_errors": errors,
            "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
        }









# -----------------------------------------
# Nodes
# -----------------------------------------




def build_context_packet(state: CodegenState) -> dict:
    """
    Build minimal focused context packet for LLM.
    Only sends relevant symbols and file content — not entire codebase.
    """
    impact = state["impact_report"]
    affected_files = impact.get("affected_files", [])
    affected_symbols = impact.get("affected_symbols", [])

    context = {
        "requirement": state["requirement"],
        "risk_level": impact["risk_assessment"]["risk_level"],
        "breaking_changes": impact["risk_assessment"].get("breaking_changes", []),
        "files": []
    }

    for af in affected_files:
        file_path = af["file_path"]
        repo_name = af["repo_name"]

        # Get symbols from Knowledge Layer
        symbols = get_symbols_for_file(repo_name, file_path)

        # Get current content
        # existing_code is keyed by "<repo>:<path>" — fall back to bare path for safety
        current_content = (
            state["existing_code"].get(f"{repo_name}:{file_path}")
            or state["existing_code"].get(file_path, "")
        )

        context["files"].append({
            "file_path": file_path,
            "repo_name": repo_name,
            "current_content": current_content,
            "existing_symbols": symbols,
            "matched_symbols": af.get("matched_symbols", [])
        })

    return context


def generate_code_changes(state: CodegenState) -> CodegenState:
    print("\n[Phase 4] Generating Diff-Based Polyglot Code Changes...")
    
    if state.get("status") == "NEW_PROJECT_NO_CODE":
        return generate_fresh_project(state)
    
    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files", [])
    context = build_context_packet(state)
    asp = state.get("scope_contract", {}) 
    requirement = state.get("requirement", "")
    
    # ── CLAUDE'S RAG CONTEXT BUILDER ──
    packet = build_context_packet_rag(
        requirement=requirement,
        asp=asp,
        top_k=8, max_files=4
    )

    adr = state.get("adr", {})
    adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
    generated_changes = []
    all_errors = []
    workspace_path = state.get('workspace_path', '')

    for file_info in context["files"]:
        file_path = file_info["file_path"]
        current_content = file_info["current_content"]
        print(f"\n  Processing: {file_path}")

        file_outline = extract_file_outline(current_content)
        windowed_content = extract_relevant_context(current_content, file_info["existing_symbols"])
        content_for_llm = current_content if len(current_content.split("\n")) <= 100 else windowed_content

        base_prompt = f"""
You are an Expert Polyglot Software Engineer modifying an existing codebase.

ADR (Agreed Tech Stack): {adr_text}
REQUIREMENT: {requirement}
RISK LEVEL: {context['risk_level']}

FILE TO MODIFY: {file_path}

{file_outline}

RELEVANT FILE CONTENT (May be truncated for length):
```
{content_for_llm}
```
INSTRUCTIONS:
1. Read the ADR and file extension to determine the programming language.
2. Modify this file to implement the requirement. Keep existing functionality working.
3. Return ONLY the exact blocks of code that need to be changed.
4. The 'search_block' MUST match the existing file content exactly character-by-character.
5. EVERY file change MUST include a 'traces_to' field mapping back to the ASP.

Return ONLY valid JSON in this exact format:
{{
  "file_path": "{file_path}",
  "changes": [
    {{
      "search_block": "Exact existing code snippet to replace",
      "replace_block": "The new code to insert"
    }}
  ],
  "change_summary": "what was changed and why",
  "traces_to": ["capability phrase from ASP"]
}}
"""
        # Inject Qdrant chunks!
        prompt = inject_into_user_message(base_prompt, packet)

        max_retries = 3
        success = False

        for attempt in range(1, max_retries + 1):
            response = call_llm(prompt, max_tokens=4000)
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

            try:
                change_data = json.loads(response)
                modified_content = current_content
                diff_errors = []
                
                if change_data.get("changes"):
                    for diff in change_data["changes"]:
                        search_block = diff.get("search_block", "")
                        replace_block = diff.get("replace_block", "")
                        
                        if not search_block:
                            continue
                        if search_block not in modified_content:
                            diff_errors.append(f"{file_path}: 'search_block' not found in current content.")
                            continue
                        modified_content = modified_content.replace(search_block, replace_block, 1)

                    if diff_errors:
                        errors = diff_errors
                    else:
                        change_data["content"] = modified_content
                        errors = validate_python(modified_content, file_path)
                else:
                    errors = [f"{file_path}: No 'changes' array provided by LLM."]

                if errors:
                    print(f"  ⚠️  Attempt {attempt}/{max_retries} — errors: {errors}")
                    if attempt < max_retries:
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{json.dumps(errors)}\nFix the 'search_block' to exactly match."
                    continue

                if workspace_path:
                    full_path = os.path.join(workspace_path, change_data["file_path"])
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)

                generated_changes.append(change_data)
                print(f"  ✅ Generated patch for: {file_path}")
                success = True
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
                if attempt < max_retries:
                    prompt += f"\n\nJSON ERROR: {e}\nReturn ONLY valid JSON."

        if not success:
            all_errors.append(f"{file_path}: Failed after {max_retries} attempts")

    return {**state, "generated_changes": generated_changes, "validation_errors": all_errors, "status": "CODE_GENERATED" if not all_errors else "CODE_GENERATION_FAILED"}

# To avoid name clashing in Codegen, rename the imported builder at the top:
# from agents.context_packet_builder import build_context_packet as build_context_packet_rag


def validate_changes(state: CodegenState) -> CodegenState:
    """Validate all generated changes."""
    print("\n[Phase 4] Validating generated changes...")

    errors = list(state.get("validation_errors", []))
    changes = state.get("generated_changes", [])

    for change in changes:
        file_path = change.get("file_path", "")
        content = change.get("content", "")

        # Python syntax check
        if file_path.endswith(".py"):
            syntax_errors = validate_python(content, file_path)
            errors.extend(syntax_errors)

        # Check content is not empty, but allow empty Python package markers
        if not content.strip() and not file_path.endswith("__init__.py"):
            errors.append(f"{file_path}: Generated content is empty")

    if errors:
        print(f"  ❌ Validation failed: {len(errors)} errors")
        for e in errors:
            print(f"     - {e}")
        return {**state, "validation_errors": errors, "status": "VALIDATION_FAILED"}

    print(f"  ✅ All {len(changes)} files validated successfully")
    return {**state, "validation_errors": [], "status": "VALIDATED"}


def route_after_validation(state: CodegenState) -> str:
    if state["status"] == "VALIDATED":
        return "pass"
    return "fail"


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_codegen_graph():
    builder = StateGraph(CodegenState)

    builder.add_node("eval_first_check", eval_first_check)
    builder.add_node("load_existing_code", load_existing_code)
    builder.add_node("generate_fresh_project", generate_fresh_project)
    builder.add_node("generate_code_changes", generate_code_changes)
    builder.add_node("validate_changes", validate_changes)

    # Single linear flow into the router
    builder.set_entry_point("eval_first_check")
    builder.add_edge("eval_first_check", "load_existing_code")

    # load_existing_code routes to ONE of two generators — no extra edges
    def route_after_load(state: CodegenState) -> str:
        return "fresh" if state.get("status") == "NEW_PROJECT_NO_CODE" else "existing"

    builder.add_conditional_edges(
        "load_existing_code",
        route_after_load,
        {"fresh": "generate_fresh_project", "existing": "generate_code_changes"},
    )

    # Both generators converge on validation
    builder.add_edge("generate_fresh_project", "validate_changes")
    builder.add_edge("generate_code_changes", "validate_changes")

    # Validation terminates the graph
    builder.add_conditional_edges(
        "validate_changes",
        route_after_validation,
        {"pass": END, "fail": END},
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -----------------------------------------
# Run
# -----------------------------------------

def run_codegen(requirement: str, impact_report: dict, workspace_path: str, thread_id: str = "thread-codegen", adr: dict = None, scope_contract: dict = None):
    graph = build_codegen_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = CodegenState(
        requirement=requirement,
        scope_contract=scope_contract or {},
        impact_report=impact_report,
        adr=adr or {},
        existing_code={},
        generated_changes=[],
        validation_errors=[],
        status="STARTED",
        workspace_path=workspace_path,
        thread_id=thread_id
    )

    print("\n" + "="*50)
    print("--- Starting Phase 4 — Code Generation ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\n{'='*50}")
    print(f"Phase 4 Complete — {result['status']}")
    print(f"Files changed: {len(result['generated_changes'])}")
    for change in result['generated_changes']:
        print(f"  - {change['file_path']}: {change.get('change_summary', '')[:60]}")
    print(f"{'='*50}")

    return result


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":
    # Mock impact report from Phase 3
    mock_impact_report = {
        "requirement": "Add leave balance tracker",
        "affected_repos": ["leave-mgmt-backend"],
        "affected_files": [
            {
                "repo_name": "leave-mgmt-backend",
                "file_path": "app\\models.py",
                "relevance_score": 0.9,
                "matched_symbols": [
                    {"name": "LeaveRequest", "type": "class", "score": 0.9}
                ]
            },
            {
                "repo_name": "leave-mgmt-backend",
                "file_path": "app\\routes.py",
                "relevance_score": 0.85,
                "matched_symbols": [
                    {"name": "approve_leave", "type": "function", "score": 0.85}
                ]
            }
        ],
        "affected_symbols": [
            {"name": "LeaveRequest", "type": "class", "line": 3},
            {"name": "approve_leave", "type": "function", "line": 26}
        ],
        "risk_assessment": {
            "risk_level": "medium",
            "breaking_changes": [
                "LeaveRequest will gain a balance field",
                "approve_leave will decrement balance"
            ],
            "recommendation": "proceed_with_caution"
        }
    }

    # 1. Add the dummy scope contract here
    mock_scope_contract = {
        "depth_level": 3,
        "strict_mode": True,
        "project_context": "Leave Management System backend update"
    }

    requirement = "Add leave balance tracker. Each employee gets 20 days per year. Balance decreases when leave is approved."

    #Mock data 2->new project with no existing code
    # mock_impact_report = {
    #     "requirement": "Scaffold a brand new FastAPI backend for a Leave Management System",
    #     "affected_repos": [],
    #     "affected_files": [],
    #     "affected_symbols": [],
    #     "risk_assessment": {
    #        "risk_level": "low",
    #       "breaking_changes": [],
    #       "recommendation": "proceed"
    #        }
    #     }
    # requirement = "Scaffold a brand new FastAPI backend for a Leave Management System. Create the main.py entry point, a models.py file with a LeaveRequest model, and a basic requirements.txt."

    result = run_codegen(requirement, mock_impact_report, "/tmp/mock_workspace", "test-codegen-1")

    # Show generated content for first file
    if result['generated_changes']:
        first = result['generated_changes'][0]
        print(f"\nSample output for {first['file_path']}:")
        print(first['content'][:500])