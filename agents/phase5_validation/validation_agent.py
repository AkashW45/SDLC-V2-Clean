"""
Phase 5 — Validation Agent
Generates tests for changed code, runs syntax checks and linting.
Auto-retries up to 3 times on failure.
No human approval gate — automatic.
"""

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
from groq import Groq
from dotenv import load_dotenv

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
    scope_contract: dict      # NEW
    generated_changes: list
    test_files: list
    validation_results: dict
    retry_count: int
    status: str


# -----------------------------------------
# Helpers
# -----------------------------------------

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


# -----------------------------------------
# Nodes
# -----------------------------------------

def generate_tests(state: ValidationState) -> ValidationState:
    print("\n[Phase 5] Generating tests...")

    test_files = []

    for change in state["generated_changes"]:
        file_path = change.get("file_path", "")
        content = change.get("content", "")
        change_summary = change.get("change_summary", "")
        new_symbols = change.get("new_symbols_added", [])
        modified_symbols = change.get("existing_symbols_modified", [])

        print(f"\n  Generating tests for: {file_path}")

        user_msg = json.dumps({
            "scope_contract": state.get("scope_contract", {}),
            "file_path": file_path,
            "content": content,
            "change_summary": change_summary,
            "new_symbols": new_symbols,
            "modified_symbols": modified_symbols,
            "requirement": state["requirement"]
        }, indent=2)

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": (
                    "You are a senior QA engineer writing pytest tests. "
                    "Return ONLY valid JSON with test_file_path, content, test_count, tests_cover."
                )},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=2000
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()

        try:
            test_file = json.loads(raw)
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
            test_files.append(test_file)
        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse test response: {e}")

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