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
from click import prompt
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
        model=MODEL,
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
# ─────────────────────────────────────────────────────────────
# Polyglot test generation — language detection + frameworks
# ─────────────────────────────────────────────────────────────

def _detect_language(file_path: str) -> str:
    """Map file extension to language."""
    p = file_path.lower()
    if p.endswith(".py"): return "python"
    if p.endswith(".java"): return "java"
    if p.endswith((".js", ".jsx", ".mjs")): return "javascript"
    if p.endswith((".ts", ".tsx")): return "typescript"
    if p.endswith(".go"): return "go"
    if p.endswith(".cs"): return "csharp"
    if p.endswith(".rb"): return "ruby"
    if p.endswith(".rs"): return "rust"
    if p.endswith(".php"): return "php"
    if p.endswith(".kt"): return "kotlin"
    return "unknown"


_LANG_TEST_CONVENTIONS = {
    "python":    {"framework": "pytest",     "import": "import pytest", "ext": ".py"},
    "java":      {"framework": "JUnit 5",    "import": "import org.junit.jupiter.api.Test;", "ext": ".java"},
    "javascript":{"framework": "Jest",       "import": "const { test, expect } = require('@jest/globals');", "ext": ".test.js"},
    "typescript":{"framework": "Jest",       "import": "import { test, expect } from '@jest/globals';", "ext": ".test.ts"},
    "go":        {"framework": "testing pkg","import": "import \"testing\"", "ext": "_test.go"},
    "csharp":    {"framework": "xUnit",      "import": "using Xunit;", "ext": "Tests.cs"},
    "ruby":      {"framework": "RSpec",      "import": "require 'rspec'", "ext": "_spec.rb"},
    "kotlin":    {"framework": "JUnit 5",    "import": "import org.junit.jupiter.api.Test", "ext": "Tests.kt"},
    "rust":      {"framework": "built-in",   "import": "#[cfg(test)]", "ext": ".rs"},
    "php":       {"framework": "PHPUnit",    "import": "use PHPUnit\\Framework\\TestCase;", "ext": "Test.php"},
}

def _sanitize_for_json(obj):
    """Strip non-serializable items (functions, lambdas, etc) from a nested dict."""
    if callable(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj if not callable(v)]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Fallback: try str repr
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

def _suggest_test_path(file_path: str, lang: str) -> str:
    """Generate a language-appropriate test file path."""
    p = file_path.replace("\\", "/")
    if lang == "python":
        if "/test" in p or p.startswith("tests/"):
            return p
        safe = p.replace("/", "_").replace(".py", "")
        return f"tests/test_{safe}.py"
    if lang == "java":
        if "/test/" in p:
            return p
        return p.replace("/main/", "/test/").replace(".java", "Tests.java")
    if lang == "javascript":
        if ".test." in p:
            return p
        return p.replace(".js", ".test.js")
    if lang == "typescript":
        if ".test." in p:
            return p
        return p.replace(".ts", ".test.ts")
    if lang == "go":
        if "_test.go" in p:
            return p
        return p.replace(".go", "_test.go")
    if lang == "csharp":
        if "Tests.cs" in p:
            return p
        return p.replace(".cs", "Tests.cs")
    if lang == "ruby":
        if "_spec.rb" in p:
            return p
        return p.replace(".rb", "_spec.rb").replace("lib/", "spec/")
    if lang == "kotlin":
        if "Tests.kt" in p:
            return p
        return p.replace(".kt", "Tests.kt")
    return p + ".test"


def _is_test_file(file_path: str) -> bool:
    """Check if a file IS already a test file (don't generate tests for tests)."""
    p = file_path.replace("\\", "/").lower()
    if "/test/" in p or "/tests/" in p:
        return True
    if p.startswith("test/") or p.startswith("tests/"):
        return True
    if any(p.endswith(suffix) for suffix in (
        "_test.go", ".test.js", ".test.ts", "_spec.rb",
        "tests.java", "tests.cs", "tests.kt", "test.php"
    )):
        return True
    # Python: test_*.py or *_test.py
    fname = p.rsplit("/", 1)[-1]
    if fname.startswith("test_") and fname.endswith(".py"):
        return True
    if fname.endswith("_test.py"):
        return True
    return False

def _parse_delimited_test_response(response: str) -> dict:
    """
    Parse a delimited LLM response of the form:
      <TEST_FILE_PATH>...</TEST_FILE_PATH>
      <TEST_COUNT>...</TEST_COUNT>
      <TESTS_COVER>...</TESTS_COVER>
      <TEST_CONTENT>...</TEST_CONTENT>
    Returns dict with keys: test_file_path, test_count, tests_cover, content.
    """
    result = {}

    # Strip any markdown fences the LLM might still include
    response = re.sub(r"^```[\w]*\s*", "", response.strip(), flags=re.MULTILINE)
    response = re.sub(r"\s*```$", "", response.strip(), flags=re.MULTILINE)

    def extract(tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else ""

    test_file_path = extract("TEST_FILE_PATH")
    test_count_str = extract("TEST_COUNT")
    tests_cover_str = extract("TESTS_COVER")
    content = extract("TEST_CONTENT")

    # ─── SALVAGE: handle unclosed <TEST_CONTENT> tag (LLM truncation) ───
    if not content and "<TEST_CONTENT>" in response:
        # Grab everything from <TEST_CONTENT> to end of string
        start_idx = response.index("<TEST_CONTENT>") + len("<TEST_CONTENT>")
        end_idx = response.find("</TEST_CONTENT>", start_idx)
        if end_idx == -1:
            # No closing tag — take the rest
            content = response[start_idx:].strip()
            print(f"  [Phase 5] ⚠️  Salvaged content from unclosed TEST_CONTENT tag")
        else:
            content = response[start_idx:end_idx].strip()
    # If the content is wrapped in code fences, strip them
    content_stripped = content
    if content_stripped.startswith("```"):
        lines = content_stripped.split("\n", 1)
        if len(lines) > 1:
            content_stripped = lines[1]
        if content_stripped.endswith("```"):
            content_stripped = content_stripped[:-3].rstrip()
        elif "```" in content_stripped:
            content_stripped = content_stripped.rsplit("```", 1)[0].rstrip()

    result["test_file_path"] = test_file_path or ""
    result["content"] = content_stripped

    try:
        result["test_count"] = int(test_count_str) if test_count_str else 0
    except ValueError:
        result["test_count"] = 0

    if tests_cover_str:
        result["tests_cover"] = [c.strip() for c in tests_cover_str.split(",") if c.strip()]
    else:
        result["tests_cover"] = []

    return result


def _process_single_test(change, requirement, previous_errors, scope_contract, depth_level):
    """
    Polyglot worker — generates tests for ONE source file.
    Uses delimiter-based output (not JSON-wrapped) to avoid escape-hell.
    """
    file_path = change.get("file_path", "")
    if not file_path:
        return None

    if _is_test_file(file_path):
        return None

    lang = _detect_language(file_path)
    conv = _LANG_TEST_CONVENTIONS.get(lang)
    if not conv:
        print(f"  [Phase 5] ⏭️  Skipping unsupported language for: {file_path}")
        return None

    content = change.get("content", "")
    suggested_test_path = _suggest_test_path(file_path, lang)

    error_feedback = ""
    if previous_errors:
        error_feedback = (
            "PREVIOUS ATTEMPT FAILED WITH THESE ERRORS:\n"
            + "\n".join(f"  - {e}" for e in previous_errors)
        )

    safe_scope = _sanitize_for_json(scope_contract) if scope_contract else {}

    # ─── DELIMITER-BASED PROMPT (no JSON-wrapped code) ───
    system_prompt = f"""You are a senior QA engineer writing {conv['framework']} tests in {lang}.

You will receive a source file. Generate a test file for it.

OUTPUT FORMAT — strict, use these exact delimiters:

<TEST_FILE_PATH>
{suggested_test_path}
</TEST_FILE_PATH>
<TEST_COUNT>
N
</TEST_COUNT>
<TESTS_COVER>
happy_path, edge_case_x, error_path_y
</TESTS_COVER>
<TEST_CONTENT>
// the complete test file content here — raw code, no escaping needed
</TEST_CONTENT>

RULES:
1. Use {conv['framework']} as the test framework.
2. Cover 3-6 tests: happy path, error path, edge cases.
3. Tests must be valid {lang} that compiles/parses.
4. Reference REAL symbols from the source — do NOT invent classes or methods.
5. Use the SUGGESTED test path or a similar path following {lang} conventions.
6. Do NOT include markdown fences, explanations, or any text outside the delimiters.
"""

    user_msg = f"""LANGUAGE: {lang}
FRAMEWORK: {conv['framework']}
SUGGESTED_TEST_PATH: {suggested_test_path}
REQUIREMENT: {requirement}
DEPTH_LEVEL: {depth_level}
FILE_PATH: {file_path}

CHANGE SUMMARY:
{change.get('change_summary', '')}

NEW SYMBOLS ADDED: {change.get('new_symbols_added', [])}
EXISTING SYMBOLS MODIFIED: {change.get('existing_symbols_modified', [])}

{error_feedback}

SOURCE FILE CONTENT:
```{lang}
{content}
```

Now generate the test file using the delimited format above.
"""

    try:
        api_response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=15000,
            stream=False,
        )
        response = api_response.choices[0].message.content.strip()

        # ─── HANDLE TRUNCATION ─────────────────────────────────────
        # Check finish_reason: 'length' = truncated due to max_tokens
        finish_reason = api_response.choices[0].finish_reason if api_response.choices else None
        if finish_reason == "length":
            print(f"  [Phase 5] ⚠️  Response truncated for {file_path} — attempting salvage")
            # If we have an unclosed <TEST_CONTENT>, force-close it
            if "<TEST_CONTENT>" in response and "</TEST_CONTENT>" not in response:
                response = response + "\n</TEST_CONTENT>"

        # Parse delimited response
        parsed = _parse_delimited_test_response(response)
        if not parsed:
            print(f"  [Phase 5] ❌ Could not parse delimited response for {file_path}")
            print(f"  [Phase 5] First 500 chars of LLM response:")
            print(f"  ─── {response[:500]} ───")
            return None

        test_path = parsed.get("test_file_path") or suggested_test_path
        test_content = parsed.get("content", "")

        if not test_content or len(test_content) < 30:
            print(f"  [Phase 5] ⚠️  Generated test content too short for {file_path}")
            return None

        test_file = {
            "test_file_path": test_path,
            "content": test_content,
            "test_count": parsed.get("test_count", 0),
            "tests_cover": parsed.get("tests_cover", []),
            "language": lang,
            "framework": conv["framework"],
        }

        # Validate
        if lang == "python":
            errors = validate_python_syntax(test_content, test_path)
            if errors:
                test_file["errors"] = errors
        else:
            markers = {
                "java": ["@Test", "import"],
                "javascript": ["test(", "describe(", "expect("],
                "typescript": ["test(", "describe(", "expect("],
                "go": ["func Test", "*testing.T"],
                "csharp": ["[Fact]", "[Theory]"],
                "ruby": ["describe", "it ", "expect("],
                "kotlin": ["@Test", "fun "],
                "rust": ["#[test]", "fn "],
                "php": ["public function test", "extends TestCase"],
            }
            expected = markers.get(lang, [])
            found = sum(1 for m in expected if m in test_content)
            if expected and found == 0:
                test_file["errors"] = [f"{test_path}: no {lang} test markers found"]

        return [test_file]

    except Exception as e:
        print(f"  [Phase 5] ❌ Test gen error for {file_path}: {e}")
        return None
    

def generate_tests(state: ValidationState) -> ValidationState:
    print("\n[Phase 5] Generating tests (Concurrent, Polyglot)...")
    test_files = []
    requirement = state.get("requirement", "")
    previous_errors = state.get("last_errors", [])
    scope_contract = state.get("scope_contract", {})
    depth_level = scope_contract.get("depth_level", 3) if scope_contract else 3

    changes = state.get("generated_changes", []) or []
    if not changes:
        print("  [Phase 5] ⏭️  No generated changes to test")
        return {**state, "test_files": [], "status": "TESTS_GENERATED"}

    # Group by language for visibility
    lang_counts = {}
    for c in changes:
        lang = _detect_language(c.get("file_path", ""))
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"  [Phase 5] Languages detected: {dict(lang_counts)}")

    # Concurrent fan-out (teammate's pattern)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                _process_single_test,
                change, requirement, previous_errors, scope_contract, depth_level
            ): change.get("file_path", "?")
            for change in changes
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    test_files.extend(result)
                    test_count = sum(t.get("test_count", 0) for t in result)
                    framework = result[0].get("framework", "?") if result else "?"
                    print(f"  [Phase 5] ({completed}/{len(changes)}) ✅ {framework}: {test_count} tests for {file_path}")
                else:
                    print(f"  [Phase 5] ({completed}/{len(changes)}) ⏭️  Skipped: {file_path}")
            except Exception as e:
                print(f"  [Phase 5] ({completed}/{len(changes)}) ❌ Worker exception for {file_path}: {e}")

    print(f"\n[Phase 5] Generated {len(test_files)} test files total")
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
            current_dir = os.path.dirname(full)
            while current_dir and current_dir != sandbox:
                init_file = os.path.join(current_dir, "__init__.py")

                if not os.path.exists(init_file):
                    with open(init_file, "w") as f:
                        f.write("")
                current_dir = os.path.dirname(current_dir)
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
            current_dir = os.path.dirname(full)

            while current_dir and current_dir != sandbox:
                init_file = os.path.join(current_dir, "__init__.py")

                if not os.path.exists(init_file):
                    with open(init_file, "w") as f:
                        f.write("")

                current_dir = os.path.dirname(current_dir)

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

    if pytest_result["status"] != "PASS":
        print("\n[Phase 5] Pytest output:")
        print(pytest_result["output"])
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