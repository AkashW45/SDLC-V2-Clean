#
"""
Phase 4 — Code Generation Agent
Reads approved impact report and generates targeted code changes
for only the affected files identified in Phase 3.
Uses AST context from Knowledge Layer.
Supports Polyglot generation (Python, TS, Java, etc.)
"""

import os
import ast
import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from groq import Groq
from dotenv import load_dotenv
import psycopg2
from agents.prompts.system_prompts import CODEGEN_SYSTEM
from agents.critic.critic_agent import critique

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
    scope_contract: dict
    scope_contract: dict
    impact_report: dict
    existing_code: dict      # file_path -> current content
    generated_changes: list  # list of {file_path, content, change_summary, ...}
    existing_code: dict
    generated_changes: list
    validation_errors: list
    status: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str) -> str:
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
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol_name, symbol_type, line_number, signature, docstring
        FROM symbols
        WHERE repo_name = %s AND file_path = %s
        ORDER BY line_number
    """, (repo_name, file_path))
    symbols =[]
    for row in cur.fetchall():
        symbols.append({
            "name": row[0],
            "type": row[1],
            "line": row[2],
            "signature": row[3],
            "docstring": row[4]
        })
    symbols = [
        {"name": r[0], "type": r[1], "line": r[2], "signature": r[3], "docstring": r[4]}
        for r in cur.fetchall()
    ]
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
    """Validate Python syntax. Gracefully skips non-Python files."""
    errors =[]

    # Safe bypass for Polyglot (JS/TS/Java/C#)
    if not file_path.endswith(".py"):
        return errors

    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError at line {e.lineno}: {e.msg}")
    return errors


# -----------------------------------------
# Nodes
# -----------------------------------------

def load_existing_code(state: CodegenState) -> CodegenState:
    """Read current content of all affected files. For new projects, skip."""
    print("\n[Phase 4] Loading existing code...")

    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files",[])

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
def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
    return raw


def _unwrap_files(data: dict) -> list:
    """CODEGEN_SYSTEM always returns {"files": [...]}. Unwrap that envelope."""
    if "files" in data and isinstance(data["files"], list):
        return data["files"]
    return [data]


# -----------------------------------------
# Nodes
# -----------------------------------------

def load_existing_code(state: CodegenState) -> CodegenState:
    print("\n[Phase 4] Loading existing code...")

    affected_files = state.get("impact_report", {}).get("affected_files", [])
    if not affected_files:
        print("  [Phase 4] No existing files — NEW PROJECT, generating fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    repo_path = os.getenv("REPO_PATH", "")
    existing_code = {}
    for af in affected_files:
        file_path = af["file_path"]
        normalized = file_path.replace("\\", os.sep).replace("/", os.sep)
        content = read_file_from_repo(repo_path, normalized)
        if content:
            existing_code[file_path] = content
            print(f"  ✅ Loaded: {file_path} ({len(content)} chars)")

    return {**state, "existing_code": existing_code, "status": "EXISTING_PROJECT"}


def generate_fresh_project(state: CodegenState) -> CodegenState:
    print("  [Phase 4] Generating FRESH project scaffold...")

    architecture = state.get("impact_report", {}).get("architecture", {})
    arch_context = ""
    if architecture.get("nodes"):
        lines = []
        for node in architecture["nodes"]:
            lines.append(f"- {node.get('name','')} ({node.get('type','service')}): {node.get('description','')}")
            if node.get("tech_stack"):
                lines.append(f"  Tech: {', '.join(node['tech_stack'])}")
        arch_context = "\n\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(lines)

    user_msg = json.dumps({
        "scope_contract": state.get("scope_contract", {}),
        "requirement": state["requirement"],
        "architecture_context": arch_context
    }, indent=2)

    raw = _strip_fences(call_llm(CODEGEN_SYSTEM + "\n\nUser request:\n" + user_msg, max_tokens=4000))

    generated_changes = []
    try:
        generated_changes = _unwrap_files(json.loads(raw))
    except json.JSONDecodeError:
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            generated_changes = _unwrap_files(json.loads(raw[start:end]))
        except Exception as e:
            print(f"  [Phase 4] Fresh project JSON parse failed: {e}")

    errors = []
    for change in generated_changes:
        if change.get("file_path", "").endswith(".py"):
            errors.extend(validate_python(change.get("content", ""), change["file_path"]))

    print(f"  [Phase 4] Generated {len(generated_changes)} fresh files (errors: {len(errors)})")
    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": errors,
        "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
    }


def build_context_packet(state: CodegenState) -> dict:
    """Build minimal focused context packet for LLM."""
    impact = state["impact_report"]
    affected_files = impact.get("affected_files",[])

    context = {
        "requirement": state["requirement"],
        "risk_level": impact["risk_assessment"]["risk_level"],
        "breaking_changes": impact["risk_assessment"].get("breaking_changes", []),
        "files":[]
    }
    for af in impact.get("affected_files", []):
        file_path = af["file_path"]
        repo_name = af["repo_name"]

        symbols = get_symbols_for_file(repo_name, file_path)
        current_content = state["existing_code"].get(file_path, "")

        context["files"].append({
            "file_path": file_path,
            "repo_name": af["repo_name"],
            "current_content": state["existing_code"].get(file_path, ""),
            "existing_symbols": get_symbols_for_file(af["repo_name"], file_path),
            "matched_symbols": af.get("matched_symbols", [])
            "repo_name": repo_name,
            "current_content": current_content,
            "existing_symbols": symbols,
            "matched_symbols": af.get("matched_symbols",[])
        })
    return context


def generate_fresh_project(state: CodegenState) -> CodegenState:
    """Generate complete project scaffold dynamically based on architecture."""
    print("  [Phase 4] Generating FRESH dynamic polyglot project scaffold...")

    requirement = state["requirement"]
    impact = state.get("impact_report", {})
    architecture = impact.get("architecture", {})

    arch_context = ""
    if architecture.get("nodes"):
        arch_summary =[]
        for node in architecture.get("nodes",[]):
            arch_summary.append(f"- Node: {node.get('name','')} ({node.get('type','service')})")
            arch_summary.append(f"  Description: {node.get('description','')}")
            if node.get('tech_stack'):
                arch_summary.append(f"  Tech: {', '.join(node['tech_stack'])}")
        arch_context = "\n\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(arch_summary)

    prompt = f"""
You are a Senior Polyglot Software Architect and Principal Engineer scaffolding a brand new multi-service project.

REQUIREMENT:
{requirement}
{arch_context}

INSTRUCTIONS:
1. Examine the ARCHITECTURE TO IMPLEMENT. Deduce the exact languages and frameworks required (e.g., React/TypeScript frontend, Node.js backend, Go, Java Spring Boot, Python FastAPI).
2. Generate a complete, working starter project scaffold for ALL requested nodes.
3. Include standard configuration files for the chosen stacks (e.g., package.json, pom.xml, requirements.txt, tsconfig.json, Dockerfile).
4. Provide the core application files, routes/controllers, models, and a README.md for each service.
5. Prefix file paths with the service/repo name to keep them organized (e.g., 'frontend/src/App.tsx', 'backend/main.py').

Return ONLY valid JSON in this exact format:
{{
  "files": [
    {{
      "file_path": "backend/main.py",
      "content": "complete file content as a string",
      "change_summary": "what this file does",
      "new_symbols_added": ["app", "main"],
      "existing_symbols_modified":[]
    }}
  ]
}}
"""
    response = call_llm(prompt)

    if response.startswith("```"):
        response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

    generated_changes =[]
    try:
        data = json.loads(response)
        generated_changes = data.get("files",[])
    except json.JSONDecodeError:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                generated_changes = data.get("files",[])
        except Exception as e:
            print(f"  [Phase 4] Fresh project JSON parse failed: {e}")
            generated_changes = []

    errors =[]
    for change in generated_changes:
        errs = validate_python(change.get("content", ""), change.get("file_path", ""))
        errors.extend(errs)

    print(f"  [Phase 4] Generated {len(generated_changes)} fresh files across multiple languages (errors: {len(errors)})")
    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": errors,
        "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
    }


def generate_code_changes(state: CodegenState) -> CodegenState:
    """Generate targeted code changes for each affected file."""
    print("\n[Phase 4] Generating code changes...")

    if state.get("status") == "NEW_PROJECT_NO_CODE":
        return generate_fresh_project(state)

    context = build_context_packet(state)
    generated_changes = []
    all_errors =[]

    for file_info in context["files"]:
        file_path = file_info["file_path"]
        print(f"\n  Processing: {file_path}")

        user_msg = json.dumps({
            "scope_contract": state.get("scope_contract", {}),
            "requirement": context["requirement"],
            "risk_level": context["risk_level"],
            "breaking_changes": context["breaking_changes"],
            "file_path": file_path,
            "existing_symbols": file_info["existing_symbols"],
            "current_file_content": file_info["current_content"]
        }, indent=2)

        success = False
        for attempt in range(1, 4):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": (
                        "You are a senior software engineer modifying an existing source code file. "
                        "Determine the language from the file extension. "
                        "Return ONLY valid JSON with file_path, content (complete updated file), "
                        "change_summary, new_symbols_added, existing_symbols_modified."
                    )},
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=3000
            )
            raw = _strip_fences(response.choices[0].message.content.strip())

            try:
                data = json.loads(raw)
                # CODEGEN_SYSTEM returns {"files": [...]}; unwrap and take first entry
                # for the single-file-per-request existing-project path.
                files = _unwrap_files(data)
                change = files[0] if files else data
                content = change.get("content", "")

                errors = validate_python(content, file_path)

                if errors:
                    print(f"  ⚠️  Attempt {attempt}/3 — syntax errors: {errors}")
                    if attempt < 3:
                        user_msg += f"\n\nPREVIOUS ATTEMPT FAILED WITH SYNTAX ERRORS:\n{json.dumps(errors)}"
                    continue

                generated_changes.append(change)
                print(f"  ✅ Generated: {change.get('change_summary', '')[:80]}")
                success = True
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/3 — JSON parse error: {e}")
                if attempt < 3:
                    user_msg += f"\n\nPREVIOUS ATTEMPT HAD JSON ERROR: {e}"

        if not success:
            error = f"{file_path}: Failed after 3 attempts"
            print(f"  ❌ {error}")
            all_errors.append(error)

    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": all_errors,
        "status": "CODE_GENERATED" if not all_errors else "CODE_GENERATION_FAILED"
    }


def validate_changes(state: CodegenState) -> CodegenState:
    print("\n[Phase 4] Validating generated changes...")

    errors = list(state.get("validation_errors",[]))
    changes = state.get("generated_changes",[])

    for change in changes:
        file_path = change.get("file_path", "")
        content = change.get("content", "")

        syntax_errors = validate_python(content, file_path)
        errors.extend(syntax_errors)

        if file_path.endswith('__init__.py'):
           continue
        # Empty __init__.py files are intentionally valid Python package markers
        if file_path.endswith("__init__.py"):
            continue

        if file_path.endswith(".py"):
            errors.extend(validate_python(content, file_path))

        if not content or not content.strip():
            errors.append(f"{file_path}: Generated content is empty")

    if errors:
        print(f"  ❌ Validation failed: {len(errors)} errors")
        return {**state, "validation_errors": errors, "status": "VALIDATION_FAILED"}

    print(f"  ✅ All {len(changes)} files validated successfully")
    return {**state, "validation_errors":[], "status": "VALIDATED"}


def run_critic_check(state: CodegenState) -> CodegenState:
    print("\n[Phase 4] Running critic validation...")
    scope_contract = state.get("scope_contract", {})
    if not scope_contract:
        print("  [!] No scope_contract — skipping critic")
        return state

    result = critique(
        artifact={"files": state["generated_changes"]},
        artifact_type="code",
        scope_contract=scope_contract,
        original_requirement=state["requirement"]
    )
    verdict = result.get("verdict", "ACCEPT")
    violations = result.get("violations", [])
    print(f"  Critic verdict: {verdict} | violations: {len(violations)}")
    for v in violations:
        print(f"    [{v.get('severity', '?')}] {v.get('problem', '')}")

    # Only hard-fail on REGENERATE; COMPRESS/EXPAND are warnings
    if verdict == "REGENERATE":
        return {
            **state,
            "status": "CRITIC_REJECTED",
            "validation_errors": [v["problem"] for v in violations]
        }
    return state


def route_after_validation(state: CodegenState) -> str:
    if state["status"] == "VALIDATED":
        return "pass"
    return "fail"

def build_codegen_graph():
    builder = StateGraph(CodegenState)
    builder.add_node("load_existing_code", load_existing_code)
    builder.add_node("generate_fresh_project", generate_fresh_project)
    builder.add_node("generate_code_changes", generate_code_changes)
    builder.add_node("validate_changes", validate_changes)
    builder.add_node("run_critic_check", run_critic_check)

    builder.set_entry_point("load_existing_code")

    def route_after_load(state: CodegenState) -> str:
        return "fresh" if state["status"] == "NEW_PROJECT_NO_CODE" else "existing"

    builder.add_conditional_edges(
        "load_existing_code",
        route_after_load,
        {"fresh": "generate_fresh_project", "existing": "generate_code_changes"}
    )

    builder.add_edge("generate_fresh_project", "validate_changes")
    builder.add_edge("generate_code_changes", "validate_changes")
    builder.add_conditional_edges("validate_changes", route_after_validation, {"pass": END, "fail": END})

    builder.add_conditional_edges(
        "validate_changes",
        route_after_validation,
        {"pass": "run_critic_check", "fail": END}
    )

    builder.add_edge("run_critic_check", END)

    return builder.compile(checkpointer=MemorySaver())

def run_codegen(requirement: str, impact_report: dict,
                scope_contract: dict = {},
                thread_id: str = "thread-codegen"):
    graph = build_codegen_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = CodegenState(
        requirement=requirement,
        impact_report=impact_report,
        existing_code={},
        generated_changes=[],
        validation_errors=[],
        scope_contract=scope_contract,
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
    print(f"Files changed: {len(result.get('generated_changes',[]))}")
    for change in result.get('generated_changes',[]):
        print(f"  - {change.get('file_path', 'unknown')}: {change.get('change_summary', '')[:60]}")
    print(f"Files changed: {len(result['generated_changes'])}")
    for change in result["generated_changes"]:
        print(f"  - {change['file_path']}: {change.get('change_summary', '')[:60]}")
    print(f"{'='*50}")

    return result

# -----------------------------------------
# Test Block
# -----------------------------------------
if __name__ == "__main__":
    # Mock impact report from Phase 3
    #Mock data 1->changging existing code
    mock_impact_report = {
        "requirement": "Add leave balance tracker",
        "affected_repos": ["leave-mgmt-backend"],
        "affected_files":[
            {
                "repo_name": "leave-mgmt-backend",
                "file_path": "app/models.py",
                "relevance_score": 0.9,
                "matched_symbols": [
                    {"name": "LeaveRequest", "type": "class", "score": 0.9}
                ]
                "matched_symbols": [{"name": "LeaveRequest", "type": "class", "score": 0.9}]
            },
            {
                "repo_name": "leave-mgmt-backend",
                "file_path": "app\\routes.py",
                "relevance_score": 0.85,
                "matched_symbols": [{"name": "approve_leave", "type": "function", "score": 0.85}]
            }
        ],
        "affected_symbols": [
            {"name": "LeaveRequest", "type": "class", "line": 3}
        ],
        "risk_assessment": {
            "risk_level": "medium",
            "breaking_changes": ["LeaveRequest will gain a balance field"],
            "recommendation": "proceed_with_caution"
        }
    }

    requirement = "Add leave balance tracker. Each employee gets 20 days per year."
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


    # Show generated content for first file
    if result.get('generated_changes'):
        first = result['generated_changes'][0]
        print(f"\nSample output for {first.get('file_path')}:")
        print(first.get('content', '')[:500])

    # 2. Update the function call to pass it in!
    result = run_codegen(
        requirement=requirement,
        impact_report=mock_impact_report,
        scope_contract=mock_scope_contract,  # <-- ADDED THIS
        thread_id="test-codegen-1"
    )

    if result["generated_changes"]:
        first = result["generated_changes"][0]
        print(f"\nSample output for {first['file_path']}:")
        print(first["content"][:500])
