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
from groq import Groq
from dotenv import load_dotenv
import psycopg2

load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# -----------------------------------------
# State
# -----------------------------------------

class CodegenState(TypedDict):
    requirement: str
    scope_contract: dict      # NEW
    impact_report: dict
    existing_code: dict        # file_path -> current content
    generated_changes: list    # list of {file_path, content, change_summary}
    validation_errors: list
    status: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
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


# Phase 4 Codegen — Critical fixes for fresh project generation
# Replace the generate_fresh_project function ENTIRELY in agents/phase4_codegen/codegen_agent.py
def generate_fresh_project(state: CodegenState) -> CodegenState:
    print("  [Phase 4] Generating FRESH project scaffold...")

    requirement = state["requirement"]
    impact = state.get("impact_report", {})
    architecture = impact.get("architecture", {})

    arch_context = ""
    if architecture.get("nodes"):
        arch_summary = []
        for node in architecture.get("nodes", []):
            arch_summary.append(f"- {node.get('name','')} ({node.get('type','service')}): {node.get('description','')}")
            if node.get('tech_stack'):
                arch_summary.append(f"  Tech: {', '.join(node['tech_stack'])}")
        arch_context = "\n\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(arch_summary)

    user_msg = json.dumps({
        "scope_contract": state.get("scope_contract", {}),
        "requirement": requirement,
        "architecture_context": arch_context
    }, indent=2)

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": (
                "You are a senior backend engineer scaffolding a brand new Python FastAPI project. "
                "Return ONLY valid JSON with a 'files' array containing objects with "
                "file_path, content, change_summary, new_symbols_added, existing_symbols_modified."
            )},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=8000
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()

    generated_changes = []
    try:
        data = json.loads(raw)
        generated_changes = data.get("files", [])
    except json.JSONDecodeError:
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            data = json.loads(raw[start:end])
            generated_changes = data.get("files", [])
        except Exception as e:
            print(f"  [Phase 4] Fresh project JSON parse failed: {e}")

    errors = []
    for change in generated_changes:
        if change.get("file_path", "").endswith(".py"):
            errs = validate_python(change.get("content", ""), change["file_path"])
            errors.extend(errs)

    print(f"  [Phase 4] Generated {len(generated_changes)} fresh files (errors: {len(errors)})")
    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": errors,
        "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
    }
# def generate_fresh_project(state: CodegenState) -> CodegenState:
#     """Generate complete project scaffold for a new requirement."""
#     print("  [Phase 4] Generating FRESH project scaffold...")

#     requirement = state["requirement"]
#     impact = state.get("impact_report", {})
#     architecture = impact.get("architecture", {})

#     # Build architecture context for grounded scaffolding
#     arch_context = ""
#     if architecture.get("nodes"):
#         arch_summary = []
#         for node in architecture.get("nodes", []):
#             arch_summary.append(f"- {node.get('name','')} ({node.get('type','service')}): {node.get('description','')}")
#             if node.get('tech_stack'):
#                 arch_summary.append(f"  Tech: {', '.join(node['tech_stack'])}")
#         arch_context = "\n\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(arch_summary)

#     prompt = f"""
# You are a senior backend engineer scaffolding a brand new Python FastAPI project.

# REQUIREMENT:
# {requirement}
# {arch_context}

# Generate a complete, working starter project with these files. Each MUST be production-quality:

# 1. main.py — FastAPI app entry with health endpoint
# 2. app/__init__.py — empty
# 3. app/models.py — Pydantic models matching the requirement
# 4. app/routes.py — API routes implementing the requirement (NOT just stubs)
# 5. app/services.py — business logic functions
# 6. app/database.py — SQLAlchemy setup (sqlite for now)
# 7. requirements.txt — fastapi, uvicorn, pydantic, sqlalchemy
# 8. README.md — how to run, what it does
# 9. .gitignore — standard Python
# 10. tests/__init__.py — empty
# 11. tests/test_main.py — basic health test

# Each file MUST be syntactically valid Python and immediately runnable with:
#   pip install -r requirements.txt && uvicorn main:app --reload

# Return ONLY valid JSON:
# {{
#   "files": [
#     {{
#       "file_path": "main.py",
#       "content": "complete file content as a string",
#       "change_summary": "what this file does",
#       "new_symbols_added": ["app", "main"],
#       "existing_symbols_modified": []
#     }}
#   ]
# }}
# """

#     # call_llm returns string (not parsed JSON) — handle that
#     response = call_llm(prompt)

#     if response.startswith("```"):
#         response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

#     # Try multiple JSON parse strategies
#     generated_changes = []
#     try:
#         data = json.loads(response)
#         generated_changes = data.get("files", [])
#     except json.JSONDecodeError:
#         # Try to find JSON object boundary
#         try:
#             start = response.find('{')
#             end = response.rfind('}') + 1
#             if start >= 0 and end > start:
#                 data = json.loads(response[start:end])
#                 generated_changes = data.get("files", [])
#         except Exception as e:
#             print(f"  [Phase 4] Fresh project JSON parse failed: {e}")
#             print(f"  [Phase 4] Response head: {response[:300]}")
#             generated_changes = []

#     # Validate Python syntax
#     errors = []
#     for change in generated_changes:
#         if change.get("file_path", "").endswith(".py"):
#             errs = validate_python(change.get("content", ""), change["file_path"])
#             errors.extend(errs)

#     print(f"  [Phase 4] Generated {len(generated_changes)} fresh files (errors: {len(errors)})")
#     return {
#         **state,
#         "generated_changes": generated_changes,
#         "validation_errors": errors,
#         "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
#     }
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
    print("\n[Phase 4] Generating code changes...")

    if state.get("status") == "NEW_PROJECT_NO_CODE":
        return generate_fresh_project(state)

    context = build_context_packet(state)
    generated_changes = []
    all_errors = []

    for file_info in context["files"]:
        file_path = file_info["file_path"]
        print(f"\n  Processing: {file_path}")

        user_msg = json.dumps({
            "scope_contract": state.get("scope_contract", {}),
            "requirement": context["requirement"],
            "risk_level": context["risk_level"],
            "breaking_changes": context["breaking_changes"],
            "file_path": file_info["file_path"],
            "existing_symbols": file_info["existing_symbols"],
            "current_file_content": file_info["current_content"]
        }, indent=2)

        max_retries = 3
        success = False

        for attempt in range(1, max_retries + 1):
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": (
                        "You are a senior software engineer modifying an existing Python file. "
                        "Return ONLY valid JSON with file_path, content (complete updated file), "
                        "change_summary, new_symbols_added, existing_symbols_modified."
                    )},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=3000
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()

            try:
                change = json.loads(raw)
                content = change.get("content", "")
                errors = validate_python(content, file_path)

                if errors:
                    print(f"  ⚠️  Attempt {attempt}/{max_retries} — syntax errors: {errors}")
                    if attempt < max_retries:
                        user_msg += f"\n\nPREVIOUS ATTEMPT FAILED WITH SYNTAX ERRORS:\n{json.dumps(errors)}"
                    continue

                generated_changes.append(change)
                print(f"  ✅ Generated: {change.get('change_summary', '')[:80]}")
                success = True
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
                if attempt < max_retries:
                    user_msg += f"\n\nPREVIOUS ATTEMPT HAD JSON ERROR: {e}"

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


# def generate_code_changes(state: CodegenState) -> CodegenState:
#     """
#     Generate targeted code changes for each affected file.
#     Retries up to 3 times if syntax errors found.
#     """
#     print("\n[Phase 4] Generating code changes...")
#     # NEW PROJECT mode — generate fresh files from architecture
#     if state.get("status") == "NEW_PROJECT_NO_CODE":
#         return generate_fresh_project(state)

#     context = build_context_packet(state)
#     generated_changes = []
#     all_errors = []

#     for file_info in context["files"]:
#         file_path = file_info["file_path"]
#         print(f"\n  Processing: {file_path}")

#         prompt = f"""
# You are a senior software engineer modifying an existing Python file.

# REQUIREMENT:
# {context['requirement']}

# RISK LEVEL: {context['risk_level']}

# BREAKING CHANGES TO BE AWARE OF:
# {json.dumps(context['breaking_changes'], indent=2)}

# FILE TO MODIFY: {file_path}

# EXISTING SYMBOLS IN THIS FILE (from AST index):
# {json.dumps(file_info['existing_symbols'], indent=2)}

# CURRENT FILE CONTENT:
# ```python
# {file_info['current_content']}
# ```

# INSTRUCTIONS:
# 1. Modify this file to implement the requirement
# 2. Do NOT remove or rename existing functions/classes
# 3. Add new functions/classes as needed
# 4. Keep all existing functionality working
# 5. Add TODO comments for acceptance criteria
# 6. CRITICAL: Return syntactically valid Python only
# 7. Return the COMPLETE updated file content

# Return ONLY valid JSON:
# {{
#   "file_path": "{file_path}",
#   "content": "complete updated file content here",
#   "change_summary": "what was changed and why",
#   "new_symbols_added": ["symbol1"],
#   "existing_symbols_modified": ["symbol1"]
# }}
# """

#         # Retry loop — up to 3 attempts
#         max_retries = 3
#         success = False

#         for attempt in range(1, max_retries + 1):
#             response = call_llm(prompt, max_tokens=3000)

#             if response.startswith("```"):
#                 response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

#             try:
#                 change = json.loads(response)
#                 content = change.get("content", "")
#                 errors = validate_python(content, file_path)

#                 if errors:
#                     print(f"  ⚠️  Attempt {attempt}/{max_retries} — syntax errors: {errors}")
#                     if attempt < max_retries:
#                         # Add error context to prompt for retry
#                         prompt += f"""

# PREVIOUS ATTEMPT FAILED WITH SYNTAX ERRORS:
# {json.dumps(errors)}

# Fix these syntax errors and return corrected JSON.
# Pay special attention to:
# - String escaping inside JSON
# - Proper indentation
# - No unterminated strings or brackets
# """
#                     continue

#                 generated_changes.append(change)
#                 print(f"  ✅ Generated (attempt {attempt}): {change.get('change_summary', '')[:80]}")
#                 print(f"     New symbols: {change.get('new_symbols_added', [])}")
#                 print(f"     Modified: {change.get('existing_symbols_modified', [])}")
#                 success = True
#                 break

#             except json.JSONDecodeError as e:
#                 print(f"  ⚠️  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
#                 if attempt < max_retries:
#                     prompt += f"\n\nPREVIOUS ATTEMPT HAD JSON ERROR: {e}\nReturn ONLY valid JSON, no extra text."

#         if not success:
#             error = f"{file_path}: Failed after {max_retries} attempts"
#             print(f"  ❌ {error}")
#             all_errors.append(error)

#     return {
#         **state,
#         "generated_changes": generated_changes,
#         "validation_errors": all_errors,
#         "status": "CODE_GENERATED" if not all_errors else "CODE_GENERATION_FAILED"
#     }


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

        # Empty __init__.py files are intentional and valid Python
        if file_path.endswith('__init__.py'):
           continue  # skip — empty init files are correct

        if not content or not content.strip():
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

    builder.add_node("load_existing_code", load_existing_code)
    builder.add_node("generate_code_changes", generate_code_changes)
    builder.add_node("validate_changes", validate_changes)

    builder.set_entry_point("load_existing_code")
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

def run_codegen(requirement: str, impact_report: dict, thread_id: str = "thread-codegen"):
    graph = build_codegen_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = CodegenState(
        requirement=requirement,
        impact_report=impact_report,
        existing_code={},
        generated_changes=[],
        validation_errors=[],
        status="STARTED"
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

    result = run_codegen(requirement, mock_impact_report, "test-codegen-1")

    # Show generated content for first file
    if result['generated_changes']:
        first = result['generated_changes'][0]
        print(f"\nSample output for {first['file_path']}:")
        print(first['content'][:500])