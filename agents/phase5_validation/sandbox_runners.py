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

def detect_java_build_system(generated_files: list, target_repo: str = None) -> str:
    """
    Determine whether a Java project uses Maven or Gradle by checking:
    1. The generated_files for pom.xml / build.gradle hints
    2. The actual cloned repo at ./repos/<target_repo>/ if available

    Returns: 'maven' | 'gradle' | 'maven' (default fallback)
    """
    # Check generated files first
    paths = [
        (f.get("file_path") or f.get("test_file_path") or "").lower()
        for f in (generated_files or []) if isinstance(f, dict)
    ]
    for p in paths:
        if p.endswith("pom.xml") or "/pom.xml" in p:
            return "maven"
        if p.endswith("build.gradle") or p.endswith("build.gradle.kts"):
            return "gradle"

    # Check actual repo on disk
    if target_repo:
        for base in [
            os.path.join(os.getcwd(), "repos", target_repo),
            os.path.join(os.path.dirname(os.getcwd()), "repos", target_repo),
        ]:
            if os.path.isdir(base):
                if os.path.exists(os.path.join(base, "pom.xml")):
                    return "maven"
                if os.path.exists(os.path.join(base, "build.gradle")) or \
                   os.path.exists(os.path.join(base, "build.gradle.kts")):
                    return "gradle"

    # Default to Maven (more common in enterprise Java)
    return "maven"

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
            print(f"  [sandbox] Image {self.image} not local — attempting pull...")
            try:
                pull = subprocess.run(
                    ["docker", "pull", self.image],
                    capture_output=True, text=True, timeout=600,
                )
                if pull.returncode != 0:
                    print(f"  [sandbox] ⚠️  Pull failed: {pull.stderr[:300]}")
                    return {
                        "returncode": -1,
                        "stdout": "",
                        "stderr": f"Image '{self.image}' unavailable on Docker Hub or pull failed. "
                                f"Possible cause: image name doesn't exist (try a different JDK version), "
                                f"or network/auth issue.\n{pull.stderr[:500]}",
                        "duration_ms": 0,
                        "error": "IMAGE_PULL_FAILED",
                    }
                print(f"  [sandbox] ✅ Pulled {self.image}")
            except subprocess.TimeoutExpired:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Image pull for '{self.image}' timed out after 10 minutes",
                    "duration_ms": 0,
                    "error": "IMAGE_PULL_TIMEOUT",
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


class MavenRunner:
    """
    Java/Maven: dynamically picks the right maven+JDK image based on the
    project's pom.xml java.version / maven.compiler.source setting.
    """

    image = "maven:3.9-eclipse-temurin-17"  # default, overridden per-run

    def __init__(self, target_repo: str = None):
        self.target_repo = target_repo

    def run(self, code_dir: str) -> ValidationResult:
        # Detect required JDK from pom.xml
        jdk_version = detect_jdk_version(
            target_repo=self.target_repo,
            code_dir=code_dir,
            build_tool="maven",
        )
        runtime_image = maven_image_for_jdk(jdk_version)
        print(f"  [sandbox] Using Maven image for JDK {jdk_version}: {runtime_image}")

        sandbox = DockerSandbox(runtime_image, timeout_seconds=600)

        shell_cmd = (
            "if [ ! -f pom.xml ]; then "
            "  echo 'NO_POM_XML: No pom.xml found in workspace.'; "
            "  exit 0; "
            "fi; "
            "mvn test --batch-mode --no-transfer-progress 2>&1 || true"
        )

        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])
        output = result["stdout"] + "\n" + result["stderr"]

        if "NO_POM_XML" in output:
            return ValidationResult(
                language="java", status="SKIPPED",
                duration_ms=result["duration_ms"],
                notes="No pom.xml in workspace",
            )

        import re
        passed_n = failed_n = error_n = 0
        m = re.search(r"Tests run:\s+(\d+),\s+Failures:\s+(\d+),\s+Errors:\s+(\d+)", output)
        if m:
            total = int(m.group(1))
            failed_n = int(m.group(2))
            error_n = int(m.group(3))
            passed_n = total - failed_n - error_n

        # BUILD SUCCESSFUL with no test summary → at least compilation passed
        if "BUILD SUCCESS" in output and passed_n == 0 and failed_n == 0:
            passed_n = 1
        if "BUILD FAILURE" in output:
            failed_n = max(failed_n, 1)

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
            notes=f"maven test in {runtime_image} (jdk={jdk_version})",
        )

def detect_gradle_version(target_repo: str = None, code_dir: str = None) -> str:
    """
    Detect which Gradle major version a repo uses by reading its wrapper config.

    Returns major version string: "4" | "5" | "6" | "7" | "8". Defaults to "8".
    """
    import re

    candidates = []
    if code_dir:
        candidates.append(os.path.join(code_dir, "gradle", "wrapper", "gradle-wrapper.properties"))
    if target_repo:
        for base in [
            os.path.join(os.getcwd(), "repos", target_repo),
            os.path.join(os.path.dirname(os.getcwd()), "repos", target_repo),
        ]:
            candidates.append(os.path.join(base, "gradle", "wrapper", "gradle-wrapper.properties"))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(r"gradle-(\d+)\.\d+", content)
            if m:
                major = m.group(1)
                print(f"  [sandbox] Detected Gradle version {major} from wrapper.properties")
                return major
        except Exception as e:
            print(f"  [sandbox] Could not read {path}: {e}")
            continue

    print(f"  [sandbox] No gradle-wrapper.properties found — defaulting to Gradle 8")
    return "8"


def detect_jdk_version(target_repo: str = None, code_dir: str = None, build_tool: str = "maven") -> str:
    """
    Detect required JDK version from project config files.

    For Maven: reads pom.xml for
      <maven.compiler.source>11</maven.compiler.source>  OR
      <maven.compiler.target>11</maven.compiler.target>  OR
      <java.version>11</java.version>  OR
      <source>11</source> inside <configuration>

    For Gradle: reads build.gradle for
      sourceCompatibility = '11'  OR
      sourceCompatibility = JavaVersion.VERSION_11  OR
      targetCompatibility = '11'  OR
      languageVersion = JavaLanguageVersion.of(11)

    Returns: "8" | "11" | "17" | "21". Defaults: 17 for Maven, matches Gradle major for Gradle.
    """
    import re

    candidates = []
    config_file = "pom.xml" if build_tool == "maven" else "build.gradle"

    if code_dir:
        candidates.append(os.path.join(code_dir, config_file))
        # Also check .kts variant for gradle
        if build_tool == "gradle":
            candidates.append(os.path.join(code_dir, "build.gradle.kts"))

    if target_repo:
        for base in [
            os.path.join(os.getcwd(), "repos", target_repo),
            os.path.join(os.path.dirname(os.getcwd()), "repos", target_repo),
        ]:
            candidates.append(os.path.join(base, config_file))
            if build_tool == "gradle":
                candidates.append(os.path.join(base, "build.gradle.kts"))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if build_tool == "maven":
                # Try multiple Maven JDK indicators
                patterns = [
                    r"<maven\.compiler\.source>(\d+)(?:\.\d+)?</maven\.compiler\.source>",
                    r"<maven\.compiler\.target>(\d+)(?:\.\d+)?</maven\.compiler\.target>",
                    r"<java\.version>(\d+)(?:\.\d+)?</java\.version>",
                    r"<source>(\d+)(?:\.\d+)?</source>",
                    r"<target>(\d+)(?:\.\d+)?</target>",
                ]
            else:
                # Gradle patterns
                patterns = [
                    r"sourceCompatibility\s*=\s*['\"](\d+)(?:\.\d+)?['\"]",
                    r"sourceCompatibility\s*=\s*JavaVersion\.VERSION_(\d+)",
                    r"targetCompatibility\s*=\s*['\"](\d+)(?:\.\d+)?['\"]",
                    r"targetCompatibility\s*=\s*JavaVersion\.VERSION_(\d+)",
                    r"languageVersion\s*=\s*JavaLanguageVersion\.of\((\d+)\)",
                    r"jvmTarget\s*=\s*['\"](\d+)['\"]",  # Kotlin DSL
                ]

            for pat in patterns:
                m = re.search(pat, content)
                if m:
                    version = m.group(1)
                    # Normalize "1.8" notation (treat as 8)
                    if version == "1":
                        version = "8"
                    # Snap to nearest supported LTS
                    supported = {"8", "11", "17", "21"}
                    if version in supported:
                        print(f"  [sandbox] Detected JDK version {version} from {os.path.basename(path)}")
                        return version
                    # Map known minor variants
                    if version in ("9", "10"):
                        return "11"  # snap up to nearest LTS
                    if version in ("12", "13", "14", "15", "16"):
                        return "17"
                    if version in ("18", "19", "20"):
                        return "21"
        except Exception as e:
            print(f"  [sandbox] Could not read {path}: {e}")
            continue

    # Sensible defaults if no version found
    default = "17"
    print(f"  [sandbox] No JDK version found in config — defaulting to {default}")
    return default


# (gradle_version, jdk_version) -> docker image
# Covers Gradle 4-8 × JDK 8/11/17/21 combinations that actually exist on Docker Hub.
# When an exact combo doesn't exist, falls back via gradle_image_for_combo logic.
GRADLE_VERSION_IMAGES = {
    ("4", "8"):   "gradle:4-jdk8",
    ("5", "8"):   "gradle:5-jdk8",
    ("5", "11"):  "gradle:5-jdk11",
    ("6", "8"):   "gradle:6-jdk8",
    ("6", "11"):  "gradle:6-jdk11",
    ("7", "11"):  "gradle:7-jdk11",
    ("7", "17"):  "gradle:7-jdk17",
    ("8", "17"):  "gradle:8-jdk17",
    ("8", "21"):  "gradle:8-jdk21",
}


# Maven generally supports any JDK with a matching image. Single Maven version (3.9)
# pairs cleanly with all common JDKs via Eclipse Temurin.
MAVEN_VERSION_IMAGES = {
    "8":   "maven:3.9-eclipse-temurin-8",
    "11":  "maven:3.9-eclipse-temurin-11",
    "17":  "maven:3.9-eclipse-temurin-17",
    "21":  "maven:3.9-eclipse-temurin-21",
}


def gradle_image_for_combo(gradle_version: str, jdk_version: str) -> str:
    """Pick the right gradle image for a (gradle, jdk) version combo.

    Falls back to nearest compatible combo if exact match doesn't exist.
    """
    # Exact match first
    if (gradle_version, jdk_version) in GRADLE_VERSION_IMAGES:
        return GRADLE_VERSION_IMAGES[(gradle_version, jdk_version)]

    # Try lower JDK for same Gradle version
    fallback_order = {
        "21": ["21", "17", "11", "8"],
        "17": ["17", "11", "8"],
        "11": ["11", "8", "17"],
        "8":  ["8", "11"],
    }
    for jdk in fallback_order.get(jdk_version, ["17", "11", "8"]):
        if (gradle_version, jdk) in GRADLE_VERSION_IMAGES:
            print(f"  [sandbox] Exact gradle:{gradle_version}-jdk{jdk_version} not mapped — "
                  f"using gradle:{gradle_version}-jdk{jdk}")
            return GRADLE_VERSION_IMAGES[(gradle_version, jdk)]

    # Last resort
    return "gradle:8-jdk17"


def maven_image_for_jdk(jdk_version: str) -> str:
    """Pick the right Maven image for a given JDK version."""
    return MAVEN_VERSION_IMAGES.get(jdk_version, "maven:3.9-eclipse-temurin-17")

class GradleRunner:
    """
    Java/Gradle: dynamically picks the right gradle+JDK image combo based on
    the repo's gradle-wrapper.properties version AND build.gradle JDK setting.
    """

    image = "gradle:8-jdk17"  # default, overridden per-run

    def __init__(self, target_repo: str = None):
        self.target_repo = target_repo

    def run(self, code_dir: str) -> ValidationResult:
        gradle_major = detect_gradle_version(target_repo=self.target_repo, code_dir=code_dir)
        jdk_version = detect_jdk_version(
            target_repo=self.target_repo,
            code_dir=code_dir,
            build_tool="gradle",
        )
        runtime_image = gradle_image_for_combo(gradle_major, jdk_version)
        print(f"  [sandbox] Using Gradle image for Gradle {gradle_major} + JDK {jdk_version}: {runtime_image}")

        sandbox = DockerSandbox(runtime_image, timeout_seconds=600)

        shell_cmd = (
            "if [ ! -f build.gradle ] && [ ! -f build.gradle.kts ]; then "
            "  echo 'NO_BUILD_GRADLE: No build.gradle in workspace.'; "
            "  exit 0; "
            "fi; "
            "gradle test --no-daemon --console=plain --warning-mode=summary 2>&1 || true"
        )

        result = sandbox.run(code_dir, ["bash", "-c", shell_cmd])
        output = result["stdout"] + "\n" + result["stderr"]

        if "NO_BUILD_GRADLE" in output:
            return ValidationResult(
                language="java", status="SKIPPED",
                duration_ms=result["duration_ms"],
                notes="No build.gradle in workspace",
            )

        import re
        passed_n = failed_n = error_n = 0

        m = re.search(r"(\d+)\s+tests?\s+completed[,]?\s+(\d+)\s+failed", output)
        if m:
            total = int(m.group(1))
            failed_n = int(m.group(2))
            passed_n = total - failed_n

        if passed_n == 0 and failed_n == 0:
            passed_n = output.count("PASSED")
            failed_n = output.count("FAILED")

        if "BUILD SUCCESSFUL" in output and passed_n == 0 and failed_n == 0:
            passed_n = 1
        if "BUILD FAILED" in output:
            failed_n = max(failed_n, 1)

        status = "PASS" if (failed_n == 0 and passed_n > 0) else (
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
            notes=f"gradle test in {runtime_image} (gradle={gradle_major}, jdk={jdk_version})",
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

# For Java we have two runners — Maven and Gradle. The dispatcher in
# run_sandbox_validation picks the right one based on detect_java_build_system().
# All other languages map directly.
LANGUAGE_RUNNERS = {
    "python":       PythonRunner,
    "node":         NodeRunner,
    "java":         MavenRunner,   # default — overridden for Gradle in dispatcher
    "java_maven":   MavenRunner,
    "java_gradle":  GradleRunner,
    "dotnet":       DotnetRunner,
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

    # Java: differentiate Maven vs Gradle
    runner_key = language
    if language == "java":
        build_system = detect_java_build_system(generated_files, target_repo)
        runner_key = f"java_{build_system}"
        print(f"  [sandbox] Java build system: {build_system}")

    runner_cls = LANGUAGE_RUNNERS.get(runner_key) or LANGUAGE_RUNNERS.get(language)
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

        if runner_key in ("java_maven", "java_gradle"):
            runner = runner_cls(target_repo=target_repo)
        else:
            runner = runner_cls()
        overlay_kind = "full repo + patches" if repo_src else "patches only"
        print(f"  [sandbox] Running {language} validation in {runner.image} ({overlay_kind}, {files_written} files patched)...")
        return runner.run(code_dir)