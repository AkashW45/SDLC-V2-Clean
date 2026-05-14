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
import shutil
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

from core.llm_gateway import gateway
from api.persistence import save_artifact


# -----------------------------------------
# State
# -----------------------------------------

class ValidationState(TypedDict):
    requirement: str
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

def call_llm(prompt: str, **kwargs) -> str:
    return gateway.generate(
        prompt=prompt,
        model="deepseek-chat",
        temperature=0.2,
        stream=False,
        tag="phase5_validation",
        **kwargs
    ).strip()


def validate_python_syntax(code: str, file_path: str) -> list:
    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError line {e.lineno}: {e.msg}")
    return errors


def run_basic_lint(code: str, file_path: str) -> list:
    """Run basic checks on generated code."""
    issues = []

    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        # Check for obviously bad patterns
        if "import *" in line:
            issues.append(f"{file_path}:{i}: avoid wildcard imports")
        if len(line) > 200:
            issues.append(f"{file_path}:{i}: line too long ({len(line)} chars)")

    return issues


def _process_single_test(change: dict, requirement: str, workspace_path: str):
    file_path = change.get("file_path", "")
    if not file_path.endswith(".py"):
        return None

    content = change.get("content", "")
    change_summary = change.get("change_summary", "")
    new_symbols = change.get("new_symbols_added", [])
    modified_symbols = change.get("existing_symbols_modified", [])

    print(f"\n  Generating tests for: {file_path}")

    safe_file_name = file_path.replace('/', '_').replace('\\', '_').replace('.py', '')

    prompt = f"""
You are a senior QA engineer writing pytest tests.

FILE BEING TESTED: {file_path}

FILE CONTENT:
```python
{content}
```

CHANGES MADE:
{change_summary}

NEW SYMBOLS ADDED: {json.dumps(new_symbols)}
MODIFIED SYMBOLS: {json.dumps(modified_symbols)}

REQUIREMENT: {requirement}

Write comprehensive pytest tests that:
1. Test each new function/class added
2. Test modified functions still work correctly
3. Test edge cases (zero balance, negative values etc)
4. Use pytest fixtures where appropriate
5. No external dependencies — use mocks if needed
6. Do NOT return the full file as a single string.
7. Return ONLY valid JSON using the exact schema below.

Return ONLY valid JSON:
{{
  "test_file_path": "tests/test_{safe_file_name}.py",
  "imports": "import pytest\nfrom app.models import ...",
  "test_functions": [
    {{
      "name": "test_example",
      "code": "def test_example():\n    assert True"
    }}
  ],
  "test_count": 1,
  "tests_cover": ["function1"]
}}
"""

    response = call_llm(prompt, max_tokens=8192)

    if response.startswith("```"):
        response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

    try:
        test_file = json.loads(response)

        imports = test_file.get("imports", "")
        functions_code = "\n\n".join([
            fn.get("code", "") for fn in test_file.get("test_functions", [])
        ])
        test_file["content"] = f"{imports}\n\n{functions_code}"

        if workspace_path:
            full_path = os.path.join(workspace_path, test_file["test_file_path"])
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(test_file["content"])
            print(f"  💾 Saved test to disk: {full_path}")

        errors = validate_python_syntax(
            test_file.get("content", ""),
            test_file.get("test_file_path", "")
        )

        if errors:
            print(f"  ⚠️  Test syntax errors: {errors}")
            test_file["syntax_errors"] = errors
        else:
            print(f"  ✅ Tests generated: {test_file.get('test_count', 0)} tests")
            print(f"     Covers: {test_file.get('tests_cover', [])}")

        return test_file

    except json.JSONDecodeError as e:
        print(f"  ❌ Failed to parse test response: {e}")
        return None


def run_semgrep_gate(workspace_path: str) -> dict:
    result = subprocess.run(
        ["semgrep", "scan", "--config=p/security-audit", "--json", workspace_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        # Semgrep failed, assume blocked
        return {"status": "BLOCKED", "reason": "Semgrep scan failed", "details": result.stderr}

    try:
        output = json.loads(result.stdout)
        findings = output.get("results", [])
        critical_findings = [
            issue for issue in findings
            if issue.get('extra', {}).get('severity') == 'ERROR'
        ]
        if critical_findings:
            return {
                "status": "BLOCKED",
                "reason": f"Semgrep found {len(critical_findings)} CRITICAL SAST issues",
                "details": critical_findings
            }
        else:
            return {"status": "PASS"}
    except json.JSONDecodeError:
        return {"status": "BLOCKED", "reason": "Failed to parse Semgrep output", "details": result.stdout}


# -----------------------------------------
# Nodes
# -----------------------------------------

def generate_tests(state: ValidationState) -> ValidationState:
    """Generate pytest tests for all changed files."""
    print("\n[Phase 5] Generating tests...")

    test_files = []
    workspace_path = state.get('workspace_path', '')
    requirement = state.get("requirement", "")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_single_test, change, requirement, workspace_path) for change in state["generated_changes"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                test_files.append(result)

    return {**state, "test_files": test_files, "status": "TESTS_GENERATED"}


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

        syntax_errors = []
        lint_issues = []

        if file_path.endswith(".py"):
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


def route_after_validation(state: ValidationState) -> str:
    if state["status"] == "VALIDATION_PASSED":
        return "pass"
    if state["status"] == "BLOCKED":
        return "fail"

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

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -----------------------------------------
# Run
# -----------------------------------------

def run_validation_phase(
    requirement: str,
    generated_changes: list,
    workspace_path: str,
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
        status="STARTED",
        thread_id=thread_id,
        workspace_path=workspace_path
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

    result = run_validation_phase(requirement, mock_changes, "/tmp/mock_workspace", "test-validation-1")

    if result['test_files']:
        print(f"\nSample test file:")
        print(result['test_files'][0]['content'][:400])