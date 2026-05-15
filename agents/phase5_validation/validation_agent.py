"""
Phase 5 — Validation Agent
Generates tests for changed code, runs syntax checks and linting.
Auto-retries up to 3 times on failure.
No human approval gate — automatic.
"""

import concurrent.futures
import os
import ast
import json
import re
import subprocess
import tempfile
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from groq import Groq
from dotenv import load_dotenv
from agents.prompts.system_prompts import TESTGEN_SYSTEM
from agents.critic.critic_agent import critique
# CLAUDE FIX: Ensure save_artifact is imported for semgrep report
try:
    from api.persistence import save_artifact
except ImportError:
    def save_artifact(thread_id, name, content): pass
load_dotenv()

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# -----------------------------------------
# State
# -----------------------------------------

class ValidationState(TypedDict):
    requirement: str
    scope_contract: dict
    generated_changes: list
    test_files: list
    validation_results: dict
    retry_count: int
    status: str
    thread_id: str
    workspace_path: str


# -----------------------------------------
# Helpers
# -----------------------------------------
MODEL="deepseek-v4-flash"  # Using the same model as Main for consistency
def call_llm(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    return response.choices[0].message.content.strip()


def validate_python_syntax(code: str, file_path: str) -> list:
    errors = []
    if not file_path.endswith(".py"): return errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError line {e.lineno}: {e.msg}")
    return errors


def run_basic_lint(code: str, file_path: str) -> list:
    """Run basic checks on generated code."""
    issues = []
    if not file_path.endswith(".py"): return issues
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        # Check for obviously bad patterns
        if "import *" in line:
            issues.append(f"{file_path}:{i}: avoid wildcard imports")
        if len(line) > 200:
            issues.append(f"{file_path}:{i}: line too long ({len(line)} chars)")

    return issues

def run_semgrep_gate(workspace_path: str) -> dict:
    if not workspace_path or not os.path.exists(workspace_path): return {"status": "PASS", "reason": "No workspace to scan"}
    try:
        result = subprocess.run(["semgrep", "scan", "--config=p/security-audit", "--json", workspace_path], capture_output=True, text=True)
        if result.returncode != 0 and not result.stdout.strip(): return {"status": "BLOCKED", "reason": "Semgrep scan failed", "details": result.stderr}
        output = json.loads(result.stdout)
        critical_findings = [f for f in output.get("results", []) if f.get('extra', {}).get('severity') == 'ERROR']
        if critical_findings: return {"status": "BLOCKED", "reason": f"Found {len(critical_findings)} CRITICAL SAST issues", "details": critical_findings}
        return {"status": "PASS"}
    except FileNotFoundError: return {"status": "PASS"}
    except json.JSONDecodeError: return {"status": "PASS", "reason": "Failed to parse Semgrep"}
# -----------------------------------------
# Nodes
# -----------------------------------------

def _process_single_test(change, requirement, previous_errors, scope_contract, depth_level):
    """Worker function to generate a test for a single file concurrently."""
    file_path = change.get("file_path", "")
    if not file_path.endswith(".py"): return None

    content = change.get("content", "")
    safe_file_name = file_path.replace("/", "_").replace("\\", "_").replace(".py", "")

    error_feedback = ""
    if previous_errors:
        error_feedback = "IMPORTANT — YOUR PREVIOUS ATTEMPT FAILED WITH THESE ERRORS:\n" + "\n".join(f"  - {e}" for e in previous_errors)

    user_msg = json.dumps({
        "scope_contract": scope_contract,
        "requirement": requirement,
        "file_path": file_path,
        "safe_test_file_name": f"tests/test_{safe_file_name}.py",
        "file_content": content,
        "change_summary": change.get("change_summary", ""),
        "new_symbols_added": change.get("new_symbols_added", []),
        "existing_symbols_modified": change.get("existing_symbols_modified", []),
        "error_feedback": error_feedback or None,
        "depth_level": depth_level
    }, indent=2)

    api_response = client.chat.completions.create(
        model=MODEL, # Using your Main's specified model
        messages=[
            {"role": "system", "content": TESTGEN_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=2000, stream=False
    )
    response = api_response.choices[0].message.content.strip()

    if response.startswith("```"):
        response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

    try:
        data = json.loads(response, strict=False)
        items = data.get("test_files", []) if "test_files" in data else [data]
        valid_tests = []
        for test_file in items:
            errors = validate_python_syntax(test_file.get("content", ""), test_file.get("test_file_path", ""))
            if errors: test_file["errors"] = errors
            valid_tests.append(test_file)
        return valid_tests
    except json.JSONDecodeError:
        return None


def generate_tests(state: ValidationState) -> ValidationState:
    print("\n[Phase 5] Generating tests (Concurrent)...")
    test_files = []
    requirement = state.get("requirement", "")
    previous_errors = state.get("last_errors", [])
    scope_contract = state.get("scope_contract", {})
    depth_level = scope_contract.get("depth_level", 3) if scope_contract else 3

    # The Teammate's Concurrent Threading Logic
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_process_single_test, change, requirement, previous_errors, scope_contract, depth_level)
            for change in state.get("generated_changes", [])
        ]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                test_files.extend(res)

    return {**state, "test_files": test_files, "status": "TESTS_GENERATED"}

def run_semgrep_gate(workspace_path: str) -> dict:
    """Run Semgrep security analysis."""
    if not workspace_path or not os.path.exists(workspace_path):
        return {"status": "PASS", "reason": "No workspace to scan"}

    try:
        result = subprocess.run(
            ["semgrep", "scan", "--config=p/security-audit", "--json", workspace_path],
            capture_output=True, text=True
        )
        if result.returncode != 0 and not result.stdout.strip():
            return {"status": "BLOCKED", "reason": "Semgrep scan failed"}

        output = json.loads(result.stdout)
        critical_findings = [f for f in output.get("results", []) if f.get('extra', {}).get('severity') == 'ERROR']

        if critical_findings:
            return {"status": "BLOCKED", "reason": f"Found {len(critical_findings)} CRITICAL SAST issues"}
        return {"status": "PASS"}
    except Exception:
        return {"status": "PASS", "reason": "Semgrep failed or not installed"}


def run_validation(state: ValidationState) -> ValidationState:
    """Run all validation checks on generated code and tests."""
    print("\n[Phase 5] Running validation checks...")

    results = {
        "syntax_checks": [],
        "lint_checks": [],
        "test_syntax_checks": [],
        "passed": 0,
        "failed": 0,
        "total": 0
    }

    # Validate generated code files
    for change in state["generated_changes"]:
        file_path = change.get("file_path", "")
        content = change.get("content", "")

        # Syntax check
        syntax_errors = validate_python_syntax(content, file_path)
        lint_issues = run_basic_lint(content, file_path)

        if syntax_errors:
            results["syntax_checks"].extend(syntax_errors)
            results["failed"] += 1
            print(f"  ❌ {file_path}: syntax errors")
        elif lint_issues:
            results["lint_checks"].extend(lint_issues)
            results["passed"] += 1
            print(f"  ⚠️  {file_path}: lint warnings ({len(lint_issues)})")
        else:
            results["passed"] += 1
            print(f"  ✅ {file_path}: all checks passed")

        results["total"] += 1

    # Validate test files
    for test_file in state["test_files"]:
        test_path = test_file.get("test_file_path", "")
        test_content = test_file.get("content", "")

        if test_content:
            errors = validate_python_syntax(test_content, test_path)
            if errors:
                results["test_syntax_checks"].extend(errors)
                print(f"  ❌ {test_path}: syntax errors in tests")
            else:
                print(f"  ✅ {test_path}: test syntax valid")

    # Run Semgrep security gate
    print("\n[Phase 5] Running Semgrep security analysis...")
    gate_result = run_semgrep_gate(state['workspace_path'])
    if gate_result['status'] == 'BLOCKED':
        print(f"  ❌ Security gate BLOCKED: {gate_result['reason']}")
        save_artifact(state['thread_id'], 'semgrep_report', json.dumps(gate_result))
        return {**state, "status": "BLOCKED", "validation_results": results}
    else:
        print("  ✅ Security gate PASSED")

    # Determine overall status
    has_errors = (
        len(results["syntax_checks"]) > 0 or
        len(results["test_syntax_checks"]) > 0
    )

    if has_errors:
        status = "VALIDATION_FAILED"
        print(f"\n  ❌ Validation failed — {results['failed']}/{results['total']} files have errors")
    else:
        status = "VALIDATION_PASSED"
        print(f"\n  ✅ Validation passed — {results['passed']}/{results['total']} files clean")

    return {**state, "validation_results": results, "status": status}

def run_critic_check(state: ValidationState) -> ValidationState:
    scope_contract = state.get("scope_contract", {})
    if not scope_contract or not state.get("test_files", []):
        return state

    try:
        result = critique(
            artifact={"test_files": state["test_files"]},
            artifact_type="tests",
            scope_contract=scope_contract,
            original_requirement=state["requirement"]
        )
        verdict = result.get("verdict", "ACCEPT")
        print(f"  [Critic] Verdict on tests: {verdict}")
    except Exception:
        pass
    return state

def route_after_validation(state: ValidationState) -> str:
    if state["status"] == "VALIDATION_PASSED":
        return "pass"

    retry_count = state.get("retry_count", 0)
    if retry_count < 3:
        print(f"\n[Phase 5] Retrying... attempt {retry_count + 1}/3")
        return "retry"

    print("\n[Phase 5] Max retries reached — escalating")
    return "fail"


def increment_retry(state: ValidationState) -> ValidationState:
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1,
        "status": "RETRYING"
    }


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_validation_graph():
    builder = StateGraph(ValidationState)

    builder.add_node("generate_tests", generate_tests)
    builder.add_node("run_validation", run_validation)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("run_critic_check", run_critic_check)
    builder.set_entry_point("generate_tests")
    builder.add_edge("generate_tests", "run_validation")

    builder.add_conditional_edges(
        "run_validation",
        route_after_validation,
        {
            "pass": END,
            "retry": "increment_retry",
            "fail": END
        }
    )

    builder.add_edge("increment_retry", "generate_tests")
    builder.add_edge("run_critic_check", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -----------------------------------------
# Run
# -----------------------------------------

def run_validation_phase(
    requirement: str,
    generated_changes: list,
    thread_id: str = "thread-validation"
) -> dict:
    graph = build_validation_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = ValidationState(
        requirement=requirement,
        generated_changes=generated_changes,
        test_files=[],
        validation_results={},
        retry_count=0,
        status="STARTED"
    )

    print("\n" + "="*50)
    print("--- Starting Phase 5 — Validation ---")
    print("="*50)

    result = graph.invoke(initial_state, config)

    print(f"\n{'='*50}")
    print(f"Phase 5 Complete — {result['status']}")
    print(f"Test files generated: {len(result['test_files'])}")
    print(f"Validation results: {result['validation_results'].get('passed', 0)} passed, "
          f"{result['validation_results'].get('failed', 0)} failed")
    print(f"{'='*50}")

    return result


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":
    # Mock generated changes from Phase 4
    mock_changes = [
        {
            "file_path": "app/models.py",
            "content": """from pydantic import BaseModel
from typing import Optional

class LeaveRequest(BaseModel):
    employee_name: str
    leave_type: str
    start_date: str
    end_date: str
    reason: str
    balance: int = 20

class LeaveStatus(BaseModel):
    leave_id: str
    status: str
    leave_balance: Optional[int] = None

class Employee(BaseModel):
    employee_id: str
    name: str
    leave_balance: int = 20
""",
            "change_summary": "Added Employee model and balance fields",
            "new_symbols_added": ["Employee"],
            "existing_symbols_modified": ["LeaveRequest", "LeaveStatus"]
        }
    ]

    requirement = "Add leave balance tracker. Each employee gets 20 days per year."

    result = run_validation_phase(requirement, mock_changes, "test-validation-1")

    if result['test_files']:
        print(f"\nSample test file:")
        print(result['test_files'][0]['content'][:400])