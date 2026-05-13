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
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from groq import Groq
from dotenv import load_dotenv
from agents.prompts.system_prompts import TESTGEN_SYSTEM
from agents.critic.critic_agent import critique

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
    last_errors: list
    status: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str, max_tokens: int = 2000) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content.strip()


def validate_python_syntax(code: str, file_path: str) -> list:
    errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError at line {e.lineno}: {e.msg}")
    return errors


def run_basic_lint(code: str, file_path: str) -> list:
    issues = []
    for i, line in enumerate(code.split("\n"), 1):
        if "import *" in line:
            issues.append(f"{file_path}:{i}: avoid wildcard imports")
        if len(line) > 200:
            issues.append(f"{file_path}:{i}: line too long ({len(line)} chars)")
    return issues


# -----------------------------------------
# Nodes
# -----------------------------------------

def generate_tests(state: ValidationState) -> ValidationState:
    """
    Generate pytest tests for all changed Python files.
    On retry, includes the previous errors in the prompt so the AI
    knows exactly what it did wrong and can fix it.
    """
    print("\n[Phase 5] Generating tests...")

    previous_errors = state.get("last_errors", [])
    scope_contract = state.get("scope_contract", {})
    depth_level = scope_contract.get("depth_level", 3) if scope_contract else 3

    test_files = []

    for change in state["generated_changes"]:
        file_path = change.get("file_path", "")
        # Only generate tests for Python source files
        if not file_path.endswith(".py"):
            continue

        content = change.get("content", "")
        print(f"\n  Generating tests for: {file_path}")

        # Pre-compute safe filename outside the f-string to avoid backslash SyntaxError
        safe_file_name = file_path.replace("/", "_").replace("\\", "_").replace(".py", "")

        error_feedback = ""
        if previous_errors:
            error_feedback = (
                "IMPORTANT — YOUR PREVIOUS ATTEMPT FAILED WITH THESE ERRORS.\n"
                "You MUST fix all of them in this new attempt:\n"
                + "\n".join(f"  - {e}" for e in previous_errors)
                + "\n\nCommon causes:\n"
                "- Missing import statements at the top of the test file\n"
                "- Referencing a class or function that doesn't exist in the source file\n"
                "- Invalid Python syntax (unclosed brackets, wrong indentation)\n"
                "- Testing for ValidationError or ValueError on a field that has no validator "
                "or constraint in the source — if the model accepts any value for a field, "
                "do NOT write a test expecting it to raise an error"
            )

        user_msg = json.dumps({
            "scope_contract": scope_contract,
            "requirement": state["requirement"],
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
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": TESTGEN_SYSTEM},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=2000,
            stream=False
        )
        response = api_response.choices[0].message.content.strip()

        if response.startswith("```"):
            response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

        try:
            data = json.loads(response, strict=False)
            # TESTGEN_SYSTEM returns {"test_files": [...]}; unwrap the envelope
            items = data.get("test_files", []) if "test_files" in data else [data]

            for test_file in items:
                print(f"  ✅ Tests: {test_file.get('test_count', 0)} | "
                      f"type: {test_file.get('test_type', 'unit')} | "
                      f"tickets: {test_file.get('validates_tickets', [])}")

                errors = validate_python_syntax(
                    test_file.get("content", ""),
                    test_file.get("test_file_path", "")
                )
                if errors:
                    print(f"  ⚠️  Test syntax errors detected: {errors}")
                    test_file["errors"] = errors
                else:
                    print(f"  ✅ Tests generated: {test_file.get('test_count', 0)} tests")
                    print(f"     Covers: {test_file.get('tests_cover', [])}")
                test_files.append(test_file)

        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse test response: {e}")

    return {**state, "test_files": test_files, "status": "TESTS_GENERATED"}


def run_validation(state: ValidationState) -> ValidationState:
    print("\n[Phase 5] Running validation checks...")

    results = {
        "syntax_checks": [],
        "lint_checks": [],
        "test_syntax_checks": [],
        "passed": 0,
        "failed": 0,
        "total": 0
    }

    # Validate generated source files
    for change in state["generated_changes"]:
        file_path = change.get("file_path", "")
        content = change.get("content", "")

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

    # Actually run pytest in a temp directory — syntax-valid tests must also pass
    pytest_passed = True
    if not results["syntax_checks"] and not results["test_syntax_checks"] and state["test_files"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source files
            for change in state["generated_changes"]:
                fp = change.get("file_path", "")
                dest = os.path.join(tmpdir, fp)
                os.makedirs(os.path.dirname(dest), exist_ok=True) if os.path.dirname(dest) else None
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(change.get("content", ""))

            # Write test files
            for tf in state["test_files"]:
                tp = tf.get("test_file_path", "")
                tc = tf.get("content", "")
                if tc:
                    dest = os.path.join(tmpdir, tp)
                    os.makedirs(os.path.dirname(dest), exist_ok=True) if os.path.dirname(dest) else None
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(tc)

            # Ensure all package directories have __init__.py
            dirs_needing_init = set()
            for change in state["generated_changes"]:
                d = os.path.dirname(change.get("file_path", ""))
                while d:
                    dirs_needing_init.add(d)
                    d = os.path.dirname(d)
            for tf in state["test_files"]:
                d = os.path.dirname(tf.get("test_file_path", ""))
                while d:
                    dirs_needing_init.add(d)
                    d = os.path.dirname(d)
            for d in dirs_needing_init:
                init_path = os.path.join(tmpdir, d, "__init__.py")
                os.makedirs(os.path.dirname(init_path), exist_ok=True)
                if not os.path.exists(init_path):
                    open(init_path, "w").close()

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir
            proc = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                env=env
            )
            pytest_passed = proc.returncode == 0
            results["pytest_output"] = proc.stdout + proc.stderr
            if pytest_passed:
                print(f"  ✅ pytest passed")
            else:
                print(f"  ❌ pytest failed:\n{proc.stdout[-500:]}")

    if not state["test_files"]:
        print("  ❌ No test files generated (likely due to JSON parsing errors).")
        results["failed"] += 1
        results["total"] += 1

    has_errors = (
        bool(results["syntax_checks"])
        or bool(results["test_syntax_checks"])
        or not pytest_passed
        or not state["test_files"]
    )

    if has_errors:
        status = "VALIDATION_FAILED"
        print(f"\n  ❌ Validation failed — {results['failed']}/{results['total']} files have errors")
    else:
        status = "VALIDATION_PASSED"
        print(f"\n  ✅ Validation passed — {results['passed']}/{results['total']} files clean")

    all_errors = results["syntax_checks"] + results["test_syntax_checks"]
    if not pytest_passed and "pytest_output" in results:
        all_errors.append(f"pytest failed:\n{results['pytest_output'][:1000]}")

    return {**state, "validation_results": results, "last_errors": all_errors, "status": status}


def run_critic_check(state: ValidationState) -> ValidationState:
    scope_contract = state.get("scope_contract", {})
    if not scope_contract or not state["test_files"]:
        return state

    result = critique(
        artifact={"test_files": state["test_files"]},
        artifact_type="tests",
        scope_contract=scope_contract,
        original_requirement=state["requirement"]
    )
    verdict = result.get("verdict", "ACCEPT")
    violations = result.get("violations", [])
    print(f"  Critic verdict on tests: {verdict} | violations: {len(violations)}")
    for v in violations:
        print(f"    [{v.get('severity', '?')}] {v.get('problem', '')}")
    # Log-only for test quality; don't fail the phase on critic warnings
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
    return {**state, "retry_count": state.get("retry_count", 0) + 1, "status": "RETRYING"}


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
            "pass": "run_critic_check",
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
    scope_contract: dict = {},
    thread_id: str = "thread-validation"
) -> dict:
    graph = build_validation_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = ValidationState(
        requirement=requirement,
        generated_changes=generated_changes,
        scope_contract=scope_contract,
        test_files=[],
        validation_results={},
        retry_count=0,
        last_errors=[],
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

    mock_scope_contract = {
        "depth_level": 3,
        "strict_mode": True,
        "project_context": "Leave Management System backend update"
    }

    requirement = "Add leave balance tracker. Each employee gets 20 days per year."
    result = run_validation_phase(
        requirement="Add leave balance tracker. Each employee gets 20 days per year.",
        generated_changes=mock_changes, # Using whatever mock_changes are already defined there
        scope_contract=mock_scope_contract, # <-- Make sure this is passed in!
        thread_id="test-validation-1"
    )
    if result["test_files"]:
        print(f"\nSample test file:")
        print(result["test_files"][0]["content"][:400])
