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
import psycopg2

load_dotenv()

from core.llm_gateway import gateway
from api.persistence import audit


# -----------------------------------------
# State
# -----------------------------------------

class CodegenState(TypedDict):
    requirement: str
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

def call_llm(prompt: str, **kwargs) -> str:
    return gateway.generate(
        prompt=prompt,
        model="deepseek-chat",
        temperature=0.2,
        stream=False,
        tag="phase4_codegen",
        **kwargs
    ).strip()


def get_postgres():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )


def get_symbols_for_file(repo_name: str, file_path: str) -> list:
    """Get indexed symbols for a file from PostgreSQL."""
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
    """Read current file content from local repo."""
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  [!] Could not read {full_path}: {e}")
        return ""


def validate_python(code: str, file_path: str) -> list:
    """Validate Python syntax."""
    errors = []
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


def generate_fresh_project(state: CodegenState) -> CodegenState:
    """Generate complete project scaffold for a new requirement."""
    print("  [Phase 4] Generating FRESH project scaffold...")

    requirement = state["requirement"]
    adr = state.get("adr", {})
    
    # Extract tech stack from ADR
    adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
    tech_stack = ""
    if adr and isinstance(adr, dict):
        tech_stack = adr.get("technology_stack", "Python/FastAPI (default)")
        language = adr.get("programming_language", "Python")
        framework = adr.get("primary_framework", "FastAPI")

    prompt = f"""
You are an Expert Software Engineer specializing in the tech stack defined in the provided ADR and Architecture Decision Record.

ADR:
{adr_text}

CRITICAL: First, read the ADR to determine the agreed-upon programming language and framework. Generate starter code ONLY in the tech stack specified in the ADR. File extensions and project structure MUST match the specified language:
- If TypeScript/Node.js: Use .ts extensions, package.json, express/nestjs patterns
- If Python: Use .py extensions, requirements.txt, FastAPI/Django patterns
- If C#/.NET: Use .cs extensions, .csproj, ASP.NET patterns
- Otherwise: Follow the ADR specifications exactly

DO NOT default to Python unless explicitly specified in the ADR.

REQUIREMENT:
{requirement}

Generate a complete starter project scaffold with files appropriate for the tech stack in the ADR.

Return ONLY valid JSON:
{{
  "files": [
    {{
      "file_path": "path/to/file.ext (with correct extension for tech stack)",
      "content": "complete file content",
      "change_summary": "what this file does",
      "new_symbols_added": ["symbol1"],
      "existing_symbols_modified": []
    }}
  ]
}}
"""

    response = call_llm(prompt, max_tokens=4096)
    if response.startswith("```"):
        response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

    try:
        data = json.loads(response)
        generated_changes = data.get("files", [])
    except Exception as e:
        print(f"  [Phase 4] JSON parse failed: {e}")
        generated_changes = []

    # Validate syntax (language-agnostic for now, but could extend)
    errors = []
    for change in generated_changes:
        # Only validate Python files if that's the ADR stack
        if change.get("file_path", "").endswith(".py"):
            errors.extend(validate_python(change.get("content", ""), change["file_path"]))

    print(f"  [Phase 4] Generated {len(generated_changes)} fresh files")
    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": errors,
        "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
    }



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


# -----------------------------------------
# Nodes
# -----------------------------------------

def load_existing_code(state: CodegenState) -> CodegenState:
    """Read current content of all affected files. For new projects, skip."""
    print("\n[Phase 4] Loading existing code...")

    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files", [])

    # If no affected files (new project) — skip loading
    if not affected_files:
        print("  [Phase 4] No existing files — NEW PROJECT, generating fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    repo_path = os.getenv("REPO_PATH", r"C:\Users\user\leave-mgmt-backend")
    existing_code = {}

    for af in affected_files:
        file_path = af["file_path"]
        file_path_normalized = file_path.replace("\\", os.sep).replace("/", os.sep)
        content = read_file_from_repo(repo_path, file_path_normalized)
        if content:
            existing_code[file_path] = content
            print(f"  ✅ Loaded: {file_path} ({len(content)} chars)")

    return {**state, "existing_code": existing_code, "status": "CODE_LOADED"}


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
        current_content = state["existing_code"].get(file_path, "")

        context["files"].append({
            "file_path": file_path,
            "repo_name": repo_name,
            "current_content": current_content,
            "existing_symbols": symbols,
            "matched_symbols": af.get("matched_symbols", [])
        })

    return context


def generate_code_changes(state: CodegenState) -> CodegenState:
    """
    Generate targeted code changes for each affected file.
    Retries up to 3 times if syntax errors found.
    """
    print("\n[Phase 4] Generating code changes...")
    # NEW PROJECT mode — generate fresh files from architecture
    if state.get("status") == "NEW_PROJECT_NO_CODE":
        return generate_fresh_project(state)

    context = build_context_packet(state)
    adr = state.get("adr", {})
    adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
    generated_changes = []
    all_errors = []

    for file_info in context["files"]:
        file_path = file_info["file_path"]
        current_content = file_info["current_content"]
        print(f"\n  Processing: {file_path}")

        # Context Windowing: extract relevant sections for large files
        file_outline = extract_file_outline(current_content)
        windowed_content = extract_relevant_context(current_content, file_info["existing_symbols"])
        content_for_llm = current_content if len(current_content.split("\n")) <= 100 else windowed_content

        prompt = f"""
You are an Expert Software Engineer specializing in the tech stack defined in the provided ADR.

ADR:
{adr_text}

CRITICAL: The ADR defines the agreed-upon tech stack. Use the same programming language, framework, and conventions as specified in the ADR. Generate file extensions and code patterns that match the ADR. DO NOT default to Python unless the ADR specifies Python.

REQUIREMENT:
{context['requirement']}

RISK LEVEL: {context['risk_level']}

BREAKING CHANGES TO BE AWARE OF:
{json.dumps(context['breaking_changes'], indent=2)}

FILE TO MODIFY: {file_path}

{file_outline}

EXISTING SYMBOLS IN THIS FILE (from AST index):
{json.dumps(file_info['existing_symbols'], indent=2)}

RELEVANT FILE CONTENT:
```
{content_for_llm}
```

INSTRUCTIONS:
1. Modify this file to implement the requirement
2. Do NOT remove or rename existing functions/classes
3. Add new functions/classes as needed
4. Keep all existing functionality working
5. Add TODO comments for acceptance criteria
6. CRITICAL: Return syntactically valid code in the language specified by the ADR
7. Instead of returning the entire file, return ONLY the exact blocks of code that need to be changed.
8. The search_block must match the existing file content exactly.

Return ONLY valid JSON:
{{
  "file_path": "{file_path}",
  "changes": [
    {{
      "search_block": "Exact existing code snippet to replace",
      "replace_block": "The new code to insert"
    }}
  ],
  "change_summary": "what was changed and why",
  "new_symbols_added": ["symbol1"],
  "existing_symbols_modified": ["symbol1"]
}}
"""

        # Retry loop — up to 3 attempts
        max_retries = 3
        success = False

        for attempt in range(1, max_retries + 1):
            response = call_llm(prompt, max_tokens=3000)

            if response.startswith("```"):
                response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

            try:
                change = json.loads(response)
                content = change.get("content", "")

                if change.get("changes"):
                    content = current_content
                    diff_errors = []
                    for diff in change["changes"]:
                        search_block = diff.get("search_block", "")
                        replace_block = diff.get("replace_block", "")
                        if not search_block:
                            diff_errors.append(f"{file_path}: missing search_block in diff")
                            continue
                        if search_block not in content:
                            diff_errors.append(f"{file_path}: search_block not found in current content")
                            continue
                        content = content.replace(search_block, replace_block, 1)

                    if diff_errors:
                        errors = diff_errors
                    else:
                        change["content"] = content
                        errors = validate_python(content, file_path)
                else:
                    errors = validate_python(content, file_path)

                if errors:
                    print(f"  ⚠️  Attempt {attempt}/{max_retries} — syntax errors: {errors}")
                    if attempt < max_retries:
                        # Add error context to prompt for retry
                        prompt += f"""

PREVIOUS ATTEMPT FAILED WITH SYNTAX ERRORS:
{json.dumps(errors)}

Fix these syntax errors and return corrected JSON.
Pay special attention to:
- String escaping inside JSON
- Proper indentation
- No unterminated strings or brackets
"""
                    continue

                workspace_path = state.get('workspace_path', '')
                if workspace_path:
                    full_path = os.path.join(workspace_path, change["file_path"])
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"  💾 Saved to disk: {full_path}")

                generated_changes.append(change)
                print(f"  ✅ Generated (attempt {attempt}): {change.get('change_summary', '')[:80]}")
                print(f"     New symbols: {change.get('new_symbols_added', [])}")
                print(f"     Modified: {change.get('existing_symbols_modified', [])}")
                success = True
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
                if attempt < max_retries:
                    prompt += f"\n\nPREVIOUS ATTEMPT HAD JSON ERROR: {e}\nReturn ONLY valid JSON, no extra text."

        if not success:
            error = f"{file_path}: Failed after {max_retries} attempts"
            print(f"  ❌ {error}")
            all_errors.append(error)

    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": all_errors,
        "status": "CODE_GENERATED" if not all_errors else "CODE_GENERATION_FAILED"
    }

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
    builder.add_node("generate_code_changes", generate_code_changes)
    builder.add_node("validate_changes", validate_changes)

    builder.set_entry_point("eval_first_check")
    builder.add_edge("eval_first_check", "load_existing_code")
    builder.add_edge("load_existing_code", "generate_code_changes")
    builder.add_edge("generate_code_changes", "validate_changes")

    builder.add_conditional_edges(
        "validate_changes",
        route_after_validation,
        {
            "pass": END,
            "fail": END
        }
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -----------------------------------------
# Run
# -----------------------------------------

def run_codegen(requirement: str, impact_report: dict, workspace_path: str, thread_id: str = "thread-codegen", adr: dict = None):
    graph = build_codegen_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = CodegenState(
        requirement=requirement,
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

    requirement = "Add leave balance tracker. Each employee gets 20 days per year. Balance decreases when leave is approved."

    result = run_codegen(requirement, mock_impact_report, "/tmp/mock_workspace", "test-codegen-1")

    # Show generated content for first file
    if result['generated_changes']:
        first = result['generated_changes'][0]
        print(f"\nSample output for {first['file_path']}:")
        print(first['content'][:500])