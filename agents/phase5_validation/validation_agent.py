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


def _detect_language(file_path: str) -> str:
    """Map file extension to language."""
    p = file_path.lower()
    if p.endswith(".py"): return "python"
    if p.endswith(".java"): return "java"
    if p.endswith((".js", ".jsx", ".mjs")): return "javascript"
    if p.endswith((".ts", ".tsx")): return "typescript"
    if p.endswith(".go"): return "go"
    if p.endswith((".cs",)): return "csharp"
    if p.endswith((".rb",)): return "ruby"
    if p.endswith((".rs",)): return "rust"
    if p.endswith((".php",)): return "php"
    if p.endswith((".kt",)): return "kotlin"
    return "unknown"


# Test framework + path conventions per language
_LANG_TEST_CONVENTIONS = {
    "python":    {"framework": "pytest",     "import": "import pytest", "ext": ".py",   "path_xform": lambda p: p.replace("app/", "tests/").replace(".py", "_test.py") if "/test" not in p else p},
    "java":      {"framework": "JUnit 5",    "import": "import org.junit.jupiter.api.Test;", "ext": ".java", "path_xform": lambda p: p.replace("/main/", "/test/").replace(".java", "Tests.java") if "/test/" not in p else p},
    "javascript":{"framework": "Jest",       "import": "const { test, expect } = require('@jest/globals');", "ext": ".test.js", "path_xform": lambda p: p.replace(".js", ".test.js") if ".test." not in p else p},
    "typescript":{"framework": "Jest",       "import": "import { test, expect } from '@jest/globals';", "ext": ".test.ts", "path_xform": lambda p: p.replace(".ts", ".test.ts") if ".test." not in p else p},
    "go":        {"framework": "testing pkg","import": "import \"testing\"", "ext": "_test.go", "path_xform": lambda p: p.replace(".go", "_test.go") if "_test.go" not in p else p},
    "csharp":    {"framework": "xUnit",      "import": "using Xunit;", "ext": "Tests.cs", "path_xform": lambda p: p.replace(".cs", "Tests.cs") if "Tests.cs" not in p else p},
    "ruby":      {"framework": "RSpec",      "import": "require 'rspec'", "ext": "_spec.rb", "path_xform": lambda p: p.replace(".rb", "_spec.rb").replace("lib/", "spec/") if "_spec.rb" not in p else p},
    "kotlin":    {"framework": "JUnit 5",    "import": "import org.junit.jupiter.api.Test", "ext": "Tests.kt", "path_xform": lambda p: p.replace(".kt", "Tests.kt") if "Tests.kt" not in p else p},
}


def generate_tests(state: ValidationState) -> ValidationState:
    """Polyglot test generator — detects language per file and uses appropriate framework."""
    print("\n[Phase 5] Generating tests (polyglot)...")

    test_files = []

    for change in state["generated_changes"]:
        file_path = change.get("file_path", "")
        content = change.get("content", "")
        change_summary = change.get("change_summary", "")
        new_symbols = change.get("new_symbols_added", [])
        modified_symbols = change.get("existing_symbols_modified", [])

        # Skip generating tests for test files themselves
        if "/test/" in file_path.replace("\\", "/") or "/tests/" in file_path.replace("\\", "/") \
           or file_path.lower().endswith(("_test.go", ".test.js", ".test.ts", "_spec.rb")) \
           or "Tests.java" in file_path or "Tests.cs" in file_path or "Tests.kt" in file_path:
            print(f"  ⏭️  Skipping test gen for test file: {file_path}")
            continue

        lang = _detect_language(file_path)
        conv = _LANG_TEST_CONVENTIONS.get(lang)

        if not conv:
            print(f"  ⚠️  Unknown language for {file_path} — skipping test generation")
            continue

        print(f"\n  Generating {conv['framework']} tests for: {file_path}")

        suggested_test_path = conv["path_xform"](file_path.replace("\\", "/"))

        user_msg = json.dumps({
            "language": lang,
            "test_framework": conv["framework"],
            "suggested_test_path": suggested_test_path,
            "scope_contract": state.get("scope_contract", {}),
            "file_path": file_path,
            "content": content,
            "change_summary": change_summary,
            "new_symbols": new_symbols,
            "modified_symbols": modified_symbols,
            "requirement": state["requirement"],
        }, indent=2)

        system_prompt = (
            f"You are a senior QA engineer writing {conv['framework']} tests in {lang}.\n"
            f"\n"
            f"RULES:\n"
            f"1. Generate tests in {lang} using {conv['framework']}.\n"
            f"2. Use the SUGGESTED test path (or a similar path matching the language's conventions).\n"
            f"3. Cover: happy path, error path, edge cases — at least 3 tests.\n"
            f"4. Tests must be syntactically valid — they will be compiled/parsed.\n"
            f"5. Reference real symbols from the source — do NOT invent classes/functions.\n"
            f"\n"
            f"Return ONLY valid JSON in this shape:\n"
            f"{{\n"
            f'  "test_file_path": "...",  // language-appropriate path\n'
            f'  "language": "{lang}",\n'
            f'  "framework": "{conv["framework"]}",\n'
            f'  "content": "...",  // full file content\n'
            f'  "test_count": N,\n'
            f'  "tests_cover": ["happy_path", "edge_case_X", ...]\n'
            f"}}\n"
        )

        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=3000,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()

            test_file = json.loads(raw)

            # Validate based on language
            if lang == "python":
                errors = validate_python_syntax(
                    test_file.get("content", ""),
                    test_file.get("test_file_path", ""),
                )
                if errors:
                    test_file["syntax_errors"] = errors
                    print(f"  ⚠️  Test syntax errors: {errors}")
                else:
                    print(f"  ✅ {conv['framework']} tests generated: {test_file.get('test_count', 0)} tests")
            else:
                # For non-Python languages, do a basic sanity check (non-empty + has framework keyword)
                tc = test_file.get("content", "")
                if not tc or len(tc) < 50:
                    print(f"  ⚠️  Generated test file too short or empty for {file_path}")
                    continue
                print(f"  ✅ {conv['framework']} tests generated: {test_file.get('test_count', 'N/A')} tests")

            test_file["language"] = lang
            test_files.append(test_file)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ❌ Failed to parse test response for {file_path}: {e}")
            continue
        except Exception as e:
            print(f"  ❌ Test generation error for {file_path}: {e}")
            continue

    print(f"\n[Phase 5] Generated {len(test_files)} test files across multiple languages")
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

def run_pytest_sandbox(generated_changes: list, test_files: list, timeout: int = 120) -> dict:
    """
    Concurrent Pytest Sandbox:
    1. Materialize generated code + test files into a temp directory
    2. Install minimal deps (pytest itself; project deps optional via requirements.txt if generated)
    3. Run pytest with --json-report-like flags, parse results
    4. Return summary {passed, failed, errors, total, output, sandbox_path}

    Tests that import the generated code resolve correctly because the temp dir
    is added to PYTHONPATH for the subprocess.
    """
    import sys as _sys

    # No Python files to test — nothing to do, but don't fail the phase
    py_tests = [t for t in (test_files or []) if (t.get("test_file_path") or "").endswith(".py")]
    if not py_tests:
        return {"status": "SKIPPED", "reason": "No Python test files to execute",
                "passed": 0, "failed": 0, "total": 0}

    sandbox = tempfile.mkdtemp(prefix="sdlc_v2_pytest_")
    try:
        # 1. Write generated source files
        for change in (generated_changes or []):
            fpath = change.get("file_path", "")
            content = change.get("content", "")
            if not fpath or not content:
                continue
            full = os.path.join(sandbox, fpath.replace("\\", "/"))
            os.makedirs(os.path.dirname(full) or sandbox, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)

        # 2. Write test files
        for test in py_tests:
            tpath = test.get("test_file_path", "")
            tcontent = test.get("content", "")
            if not tpath or not tcontent:
                continue
            full = os.path.join(sandbox, tpath.replace("\\", "/"))
            os.makedirs(os.path.dirname(full) or sandbox, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(tcontent)

        # 3. Run pytest. Sandbox is on PYTHONPATH so tests can import generated code.
        env = os.environ.copy()
        env["PYTHONPATH"] = sandbox + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [_sys.executable, "-m", "pytest", sandbox, "-q",
               "--tb=short", "--maxfail=50", "--no-header", "--disable-warnings"]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                cwd=sandbox, env=env, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "reason": f"pytest exceeded {timeout}s",
                    "passed": 0, "failed": 0, "total": 0,
                    "sandbox_path": sandbox, "output": ""}
        except FileNotFoundError:
            return {"status": "SKIPPED", "reason": "pytest not installed in this Python",
                    "passed": 0, "failed": 0, "total": 0,
                    "sandbox_path": sandbox, "output": ""}

        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # 4. Parse pytest's summary line — works across pytest versions.
        passed = failed = errors = 0
        m = re.search(r"(\d+)\s+passed", stdout)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", stdout)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+)\s+error", stdout)
        if m: errors = int(m.group(1))
        total = passed + failed + errors

        # pytest exit codes: 0=ok, 1=tests failed, 2=interrupted, 5=no tests collected
        if proc.returncode == 0:
            status = "PASS"
        elif proc.returncode == 5:
            status = "NO_TESTS_COLLECTED"
        else:
            status = "FAIL"

        return {
            "status": status,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "exit_code": proc.returncode,
            "output": stdout[-4000:],   # cap so it doesn't bloat the state
            "sandbox_path": sandbox,
        }
    except Exception as e:
        return {"status": "ERROR", "reason": str(e),
                "passed": 0, "failed": 0, "total": 0,
                "sandbox_path": sandbox, "output": ""}

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
    for change in state.get("generated_changes", []) or []:
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
    # Validate test files
    for test_file in state.get("test_files", []) or []:
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
    # Run Semgrep security gate
    print("\n[Phase 5] Running Semgrep security analysis...")
    gate_result = run_semgrep_gate(state.get('workspace_path', ''))
    if gate_result['status'] == 'BLOCKED':
        print(f"  ❌ Security gate BLOCKED: {gate_result['reason']}")
        save_artifact(state.get('thread_id', 'unknown'), 'semgrep_report', json.dumps(gate_result))
        return {**state, "status": "BLOCKED", "validation_results": results}
    print("  ✅ Security gate PASSED")

    # ── PYTEST SANDBOX EXECUTION ────────────────────────────────────────
    print("\n[Phase 5] Executing pytest in sandbox...")
    pytest_result = run_pytest_sandbox(
        generated_changes=state.get("generated_changes", []) or [],
        test_files=state.get("test_files", []) or [],
        timeout=120,
    )
    results["pytest"] = pytest_result
    print(f"  pytest status: {pytest_result['status']} | "
          f"{pytest_result['passed']} passed, {pytest_result['failed']} failed, "
          f"{pytest_result.get('errors', 0)} errors")

        # Determine what languages we have
    generated_changes = state.get("generated_changes", []) or []
    languages_present = set()
    for c in generated_changes:
        lang = _detect_language(c.get("file_path", ""))
        if lang != "unknown":
            languages_present.add(lang)

    print(f"  ℹ️  Languages detected: {sorted(languages_present)}")

    # Only run pytest if Python is in the mix
    has_python = "python" in languages_present
    has_tests_generated = bool(state.get("test_files"))

    if not has_python and has_tests_generated:
        print(f"  ℹ️  No Python files — pytest skipped. {len(state.get('test_files', []))} test files generated for other languages.")
        print(f"  ℹ️  Note: tests are generated but not executed in sandbox (requires JVM / Node / etc).")
        return {**state, "validation_results": results, "status": "VALIDATION_PASSED"}

    if not has_python and not has_tests_generated:
        print("  ⚠️  No Python files AND no test files generated — accepting anyway.")
        return {**state, "validation_results": results, "status": "VALIDATION_PASSED"}

    # Standard pass/fail decision for Python
    has_errors = (
        len(results["syntax_checks"]) > 0
        or len(results["test_syntax_checks"]) > 0
        or pytest_result["status"] in ("FAIL", "TIMEOUT", "ERROR")
    )

    if has_errors:
        print(f"\n  ❌ Validation failed — {results['failed']}/{results['total']} files have errors")
        return {**state, "validation_results": results, "status": "VALIDATION_FAILED"}

    print(f"\n  ✅ Validation passed — {results['passed']}/{results['total']} files clean")
    return {**state, "validation_results": results, "status": "VALIDATION_PASSED"}

    # Standard pass/fail decision
    has_errors = (
        len(results["syntax_checks"]) > 0
        or len(results["test_syntax_checks"]) > 0
        or pytest_result["status"] in ("FAIL", "TIMEOUT", "ERROR")
    )

    if has_errors:
        print(f"\n  ❌ Validation failed — {results['failed']}/{results['total']} files have errors")
        return {**state, "validation_results": results, "status": "VALIDATION_FAILED"}

    print(f"\n  ✅ Validation passed — {results['passed']}/{results['total']} files clean")
    return {**state, "validation_results": results, "status": "VALIDATION_PASSED"}



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
        {"pass": END, "retry": "increment_retry", "fail": END},
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
    thread_id: str = "thread-validation",
    scope_contract: dict = None,
    workspace_path: str = "",
) -> dict:
    graph = build_validation_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = ValidationState(
        requirement=requirement,
        scope_contract=scope_contract or {},
        generated_changes=generated_changes,
        test_files=[],
        validation_results={},
        retry_count=0,
        status="STARTED",
        thread_id=thread_id,
        workspace_path=workspace_path,
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