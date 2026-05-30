"""
Polyglot sandbox runners for Phase 5 validation.

Each runner executes generated code in an isolated Docker container with
the right toolchain (JDK, Node, .NET, etc.) and parses test output into
a uniform ValidationResult.

Add new languages by creating a new Runner class and registering it in
LANGUAGE_RUNNERS.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    """Uniform result shape regardless of language."""
    language: str
    status: str                          # PASS | FAIL | ERROR | SKIPPED
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration_ms: int = 0
    test_output: str = ""                # raw output for debugging
    failure_messages: list = field(default_factory=list)
    notes: str = ""                       # human-readable summary


# ─────────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────────

def detect_language(generated_files: list) -> str:
    """
    Inspect file paths and content hints to determine project language.
    Returns one of: python, node, java, dotnet, unknown
    """
    if not generated_files:
        return "unknown"

    paths = [
        (f.get("file_path", "") if isinstance(f, dict) else str(f))
        for f in generated_files
    ]
    path_text = " ".join(paths).lower()

    # Strong signals (file presence)
    if any(p.endswith("pom.xml") or p.endswith("build.gradle") or p.endswith("build.gradle.kts") for p in paths):
        return "java"
    if any(p.endswith("package.json") for p in paths):
        return "node"
    if any(p.endswith(".csproj") or p.endswith(".sln") or p.endswith("Program.cs") for p in paths):
        return "dotnet"
    if any(p.endswith("requirements.txt") or p.endswith("pyproject.toml") or p.endswith("setup.py") for p in paths):
        return "python"

    # Extension counts as tiebreaker
    ext_counts = {".py": 0, ".js": 0, ".ts": 0, ".java": 0, ".cs": 0}
    for p in paths:
        for ext in ext_counts:
            if p.endswith(ext):
                ext_counts[ext] += 1

    if ext_counts[".java"] > 0:
        return "java"
    if ext_counts[".cs"] > 0:
        return "dotnet"
    if ext_counts[".ts"] > 0 or ext_counts[".js"] > 0:
        return "node"
    if ext_counts[".py"] > 0:
        return "python"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Docker Sandbox Helper
# ─────────────────────────────────────────────────────────────────────────

class DockerSandbox:
    """
    Runs a command in a Docker container with the project files mounted.
    Handles timeouts, cleanup, output capture.
    """

    def __init__(self, image: str, timeout_seconds: int = 300):
        self.image = image
        self.timeout = timeout_seconds

    def _docker_available(self) -> bool:
        """Quick check that docker is reachable."""
        try:
            r = subprocess.run(
                ["docker", "ps"], capture_output=True, timeout=5, text=True
            )
            return r.returncode == 0
        except Exception:
            return False

    def _image_exists(self) -> bool:
        """Check if the image is already pulled/built locally."""
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True, timeout=10, text=True,
            )
            return r.returncode == 0
        except Exception:
            return False

    def run(self, code_dir: str, command: list) -> dict:
        """
        Mount code_dir at /workspace inside a fresh container and run command.
        Returns dict: {returncode, stdout, stderr, duration_ms, error}
        """
        if not self._docker_available():
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Docker not running — start Docker Desktop",
                "duration_ms": 0,
                "error": "DOCKER_UNAVAILABLE",
            }

        if not self._image_exists():
            print(f"  [sandbox] Pulling {self.image} (first run, may take 1-3 min)...")
            pull = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True, text=True, timeout=600,
            )
            if pull.returncode != 0:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Image pull failed: {pull.stderr[:500]}",
                    "duration_ms": 0,
                    "error": "IMAGE_PULL_FAILED",
                }

        # Normalize Windows paths to Docker-friendly format
        # On Windows Git Bash: /c/Users/... → C:/Users/... for docker
        mount_path = code_dir
        if os.name == "nt" or shutil.which("docker") and "Microsoft" in (os.uname().release if hasattr(os, 'uname') else ""):
            # Try to convert /c/Users/... to C:\Users\... if needed
            mount_path = str(Path(code_dir).absolute())

        start = time.perf_counter()
        try:
            full_cmd = [
                "docker", "run", "--rm",
                "-v", f"{mount_path}:/workspace",
                "-w", "/workspace",
                #"--network", "none",      # no network — security + speed
                self.image,
            ] + command

            print(f"  [sandbox] $ {' '.join(command)}")
            result = subprocess.run(
                full_cmd,
                capture_output=True, text=True, timeout=self.timeout,
            )
            duration_ms = int((time.perf_counter() - start) * 1000)
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": duration_ms,
                "error": None,
            }
        except subprocess.TimeoutExpired:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Sandbox timeout after {self.timeout}s",
                "duration_ms": duration_ms,
                "error": "TIMEOUT",
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "error": "EXCEPTION",
            }


# ─────────────────────────────────────────────────────────────────────────
# Per-Language Runners
# ─────────────────────────────────────────────────────────────────────────

class PythonRunner:
    """Python: pytest in a python:3.11-slim container."""

    image = "python:3.11-slim"

    def run(self, code_dir: str) -> ValidationResult:
        sandbox = DockerSandbox(self.image, timeout_seconds=180)

        # Install deps if requirements.txt present, then run pytest
        # We use a shell wrapper to do both in one container run
        shell_cmd = (
            "set -e; "
            "if [ -f requirements.txt ]; then "
            "  pip install --quiet --no-cache-dir -r requirements.txt 2>&1 | tail -5 || true; "
            "fi; "
            "pip install --quiet pytest 2>&1 | tail -1; "
            "pytest -q --tb=short --no-header 2>&1 || true"
        )
        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])

        output = result["stdout"] + "\n" + result["stderr"]
        passed = output.count(" passed")
        failed = output.count(" failed")
        errors = output.count(" error")

        # Parse pytest summary line "X passed, Y failed in Zs"
        passed_n = failed_n = error_n = 0
        for line in output.split("\n"):
            if "passed" in line or "failed" in line or "error" in line:
                import re
                m = re.search(r"(\d+) passed", line)
                if m: passed_n = max(passed_n, int(m.group(1)))
                m = re.search(r"(\d+) failed", line)
                if m: failed_n = max(failed_n, int(m.group(1)))
                m = re.search(r"(\d+) error", line)
                if m: error_n = max(error_n, int(m.group(1)))

        status = "PASS" if (failed_n == 0 and error_n == 0 and passed_n > 0) else (
            "FAIL" if failed_n > 0 else "ERROR"
        )
        if result.get("error") == "DOCKER_UNAVAILABLE":
            status = "SKIPPED"

        return ValidationResult(
            language="python",
            status=status,
            passed=passed_n,
            failed=failed_n,
            errors=error_n,
            duration_ms=result["duration_ms"],
            test_output=output[-3000:],
            notes=f"pytest in {self.image}",
        )


class NodeRunner:
    """Node/JS/TS/Angular/React: npm test in a node:20-slim container."""

    image = "node:20-slim"

    def run(self, code_dir: str) -> ValidationResult:
        sandbox = DockerSandbox(self.image, timeout_seconds=300)

        # Install deps and run whatever test script package.json defines
        shell_cmd = (
            "set -e; "
            "if [ ! -f package.json ]; then echo 'NO_PACKAGE_JSON'; exit 0; fi; "
            "npm install --silent --no-audit --no-fund --prefer-offline 2>&1 | tail -10; "
            "npm test --silent 2>&1 || true"
        )
        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])

        output = result["stdout"] + "\n" + result["stderr"]

        if "NO_PACKAGE_JSON" in output:
            return ValidationResult(
                language="node", status="SKIPPED",
                duration_ms=result["duration_ms"],
                notes="No package.json found",
            )

        # Jest format: "Tests: X passed, Y failed, Z total"
        # Mocha format: "X passing", "Y failing"
        import re
        passed_n = failed_n = 0
        m = re.search(r"Tests:\s+(\d+)\s+failed.*?(\d+)\s+passed", output)
        if m:
            failed_n = int(m.group(1))
            passed_n = int(m.group(2))
        else:
            m = re.search(r"Tests:\s+(\d+)\s+passed", output)
            if m: passed_n = int(m.group(1))
            m = re.search(r"(\d+)\s+passing", output)
            if m: passed_n = max(passed_n, int(m.group(1)))
            m = re.search(r"(\d+)\s+failing", output)
            if m: failed_n = max(failed_n, int(m.group(1)))

        status = "PASS" if (failed_n == 0 and passed_n > 0) else (
            "FAIL" if failed_n > 0 else "ERROR"
        )
        if result.get("error") == "DOCKER_UNAVAILABLE":
            status = "SKIPPED"

        return ValidationResult(
            language="node",
            status=status,
            passed=passed_n,
            failed=failed_n,
            duration_ms=result["duration_ms"],
            test_output=output[-3000:],
            notes=f"npm test in {self.image}",
        )


class JavaRunner:
    """Java: mvn test (or gradle) in a maven:3.9-eclipse-temurin-17 container."""

    image = "maven:3.9-eclipse-temurin-17"

    def run(self, code_dir: str) -> ValidationResult:
        sandbox = DockerSandbox(self.image, timeout_seconds=600)

        # Prefer Maven when pom.xml exists. Only fall back to Gradle if there's
        # NO pom.xml but there IS a build.gradle (or .kts). This handles repos
        # that may have leftover gradle wrappers from build tool migrations.
        has_pom = os.path.exists(os.path.join(code_dir, "pom.xml"))
        has_gradle = os.path.exists(os.path.join(code_dir, "build.gradle")) or \
                    os.path.exists(os.path.join(code_dir, "build.gradle.kts"))
        is_gradle = has_gradle and not has_pom

        if is_gradle:
            shell_cmd = "chmod +x ./gradlew 2>/dev/null; (./gradlew test --no-daemon 2>&1 || gradle test --no-daemon 2>&1) || true"
        else:
            # Maven offline mode if dependencies are cached, otherwise online
            shell_cmd = "mvn test --batch-mode --no-transfer-progress 2>&1 || true"

        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])
        output = result["stdout"] + "\n" + result["stderr"]

        # Maven Surefire: "Tests run: X, Failures: Y, Errors: Z, Skipped: W"
        # Gradle similar
        import re
        passed_n = failed_n = error_n = 0
        m = re.search(r"Tests run:\s+(\d+),\s+Failures:\s+(\d+),\s+Errors:\s+(\d+)", output)
        if m:
            total = int(m.group(1))
            failed_n = int(m.group(2))
            error_n = int(m.group(3))
            passed_n = total - failed_n - error_n
        else:
            # Gradle format: "X tests completed, Y failed"
            m = re.search(r"(\d+)\s+tests? completed,\s+(\d+)\s+failed", output)
            if m:
                total = int(m.group(1))
                failed_n = int(m.group(2))
                passed_n = total - failed_n

        status = "PASS" if (failed_n == 0 and error_n == 0 and passed_n > 0) else (
            "FAIL" if failed_n > 0 else "ERROR"
        )
        if result.get("error") == "DOCKER_UNAVAILABLE":
            status = "SKIPPED"

        return ValidationResult(
            language="java",
            status=status,
            passed=passed_n,
            failed=failed_n,
            errors=error_n,
            duration_ms=result["duration_ms"],
            test_output=output[-3000:],
            notes=f"{'gradle' if is_gradle else 'maven'} test in {self.image}",
        )


class DotnetRunner:
    """C# / .NET: dotnet test in a mcr.microsoft.com/dotnet/sdk:8.0 container."""

    image = "mcr.microsoft.com/dotnet/sdk:8.0"

    def run(self, code_dir: str) -> ValidationResult:
        sandbox = DockerSandbox(self.image, timeout_seconds=600)

        shell_cmd = (
            "set -e; "
            "dotnet restore --nologo 2>&1 | tail -5; "
            "dotnet test --no-restore --nologo --verbosity minimal 2>&1 || true"
        )
        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])
        output = result["stdout"] + "\n" + result["stderr"]

        # dotnet test format: "Passed!  - Failed: X, Passed: Y, Skipped: Z, Total: W"
        import re
        passed_n = failed_n = 0
        m = re.search(r"Failed:\s+(\d+),\s+Passed:\s+(\d+)", output)
        if m:
            failed_n = int(m.group(1))
            passed_n = int(m.group(2))

        status = "PASS" if (failed_n == 0 and passed_n > 0) else (
            "FAIL" if failed_n > 0 else "ERROR"
        )
        if result.get("error") == "DOCKER_UNAVAILABLE":
            status = "SKIPPED"

        return ValidationResult(
            language="dotnet",
            status=status,
            passed=passed_n,
            failed=failed_n,
            duration_ms=result["duration_ms"],
            test_output=output[-3000:],
            notes=f"dotnet test in {self.image}",
        )


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────

LANGUAGE_RUNNERS = {
    "python": PythonRunner,
    "node":   NodeRunner,
    "java":   JavaRunner,
    "dotnet": DotnetRunner,
}


def run_sandbox_validation(
    generated_files: list,
    language_hint: str = None,
    target_repo: str = None,
) -> ValidationResult:
    """
    Main entry point. Picks the right Docker runner and executes tests.

    Strategy:
    - For languages that need a complete project structure to build (Java/Maven,
      .NET/MSBuild), we COPY the cloned repo from ./repos/<target_repo>/ to a
      temp dir, then overlay the generated/patched files on top. This way
      `mvn test` / `dotnet test` see a real project with pom.xml, src tree,
      and dependencies intact.
    - For Python/Node, single-file changes work fine in isolation — write only
      the generated files to a fresh temp dir (legacy behavior).

    `language_hint` overrides detection.
    `target_repo` is required for Java/.NET full-repo overlay.
    """
    import shutil

    if not generated_files:
        return ValidationResult(
            language="unknown", status="SKIPPED",
            notes="No files to validate",
        )

    language = language_hint or detect_language(generated_files)

    if language == "unknown":
        return ValidationResult(
            language="unknown", status="SKIPPED",
            notes="Could not detect project language",
        )

    runner_cls = LANGUAGE_RUNNERS.get(language)
    if not runner_cls:
        return ValidationResult(
            language=language, status="SKIPPED",
            notes=f"No runner registered for language={language}",
        )

    # Full-repo overlay required for build systems that need complete project trees
    needs_full_repo = language in ("java", "dotnet")

    repo_src = None
    if needs_full_repo and target_repo:
        # Look for the cloned repo on disk
        candidates = [
            os.path.join(os.getcwd(), "repos", target_repo),
            os.path.join(os.path.dirname(os.getcwd()), "repos", target_repo),
        ]
        for c in candidates:
            if os.path.isdir(c):
                repo_src = c
                break

    with tempfile.TemporaryDirectory(prefix=f"sdlc_sandbox_{language}_") as code_dir:
        # Step 1: if full-repo needed and we have a clone, copy it
        if repo_src:
            print(f"  [sandbox] Copying full repo from {repo_src} → temp dir...")
            for item in os.listdir(repo_src):
                if item in (".git", "node_modules", "__pycache__", ".venv", "target", "bin", "obj"):
                    continue
                src = os.path.join(repo_src, item)
                dst = os.path.join(code_dir, item)
                try:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                            "node_modules", "__pycache__", ".venv", "target", "bin", "obj"
                        ))
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(f"  [sandbox] ⚠️  Skipping {item}: {e}")
        elif needs_full_repo:
            print(f"  [sandbox] ⚠️  {language} needs full repo but no clone found "
                  f"at ./repos/{target_repo}/ — falling back to patches-only "
                  f"(build may fail with 'no POM/csproj')")

        # Step 2: overlay the generated/patched files (this is the change being tested)
        files_written = 0
        for f in generated_files:
            if not isinstance(f, dict):
                continue
            file_path = f.get("file_path") or f.get("test_file_path")
            content = f.get("content", "")
            if not file_path or not content:
                continue
            full_path = os.path.join(code_dir, file_path)
            os.makedirs(os.path.dirname(full_path) or code_dir, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as out:
                out.write(content)
            files_written += 1

        if files_written == 0:
            return ValidationResult(
                language=language, status="SKIPPED",
                notes="No files written to sandbox",
            )

        runner = runner_cls()
        overlay_kind = "full repo + patches" if repo_src else "patches only"
        print(f"  [sandbox] Running {language} validation in {runner.image} ({overlay_kind}, {files_written} files patched)...")
        return runner.run(code_dir)