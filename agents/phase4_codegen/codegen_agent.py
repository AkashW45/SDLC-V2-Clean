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
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from agents.prompts.system_prompts import CODEGEN_SYSTEM
from agents.critic.critic_agent import critique
from agents.context_packet_builder import build_context_packet as build_context_packet_rag, inject_into_user_message
load_dotenv()
from agents.repo_workspace import ensure_repo_cloned, read_file as ws_read_file, get_repo_local_path

from api.persistence import audit

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# -----------------------------------------
# State
# -----------------------------------------

class CodegenState(TypedDict, total=False):
    requirement: str
    scope_contract: dict
    impact_report: dict
    adr: dict                  # Architecture Decision Record for tech stack enforcement
    existing_code: dict        # file_path -> current content
    generated_changes: list    # list of {file_path, content, change_summary}

    # CI/CD artifact metadata — populated by the new generate_cicd node
    cicd_decision: dict        # what the LLM decided about deploy needs
    cicd_warnings: list        # validation warnings (no-test sanity checks)

    validation_errors: list
    status: str
    workspace_path: str
    thread_id: str


# -----------------------------------------
# Helpers
# -----------------------------------------

def call_llm(prompt: str, max_tokens: int = 8192) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=max_tokens
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
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"  [!] Could not read {full_path}: {e}")
        return ""


def validate_python(code: str, file_path: str) -> list:
    """Validate Python syntax. Skip non-Python files (HTML, MD, JS, JSON, YAML, etc.)."""
    errors = []
    if not file_path.endswith(".py"):
        return errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"{file_path}: SyntaxError at line {e.lineno}: {e.msg}")
    return errors


def extract_file_outline(code: str) -> str:
    """Extract class and function names to provide file structure without full content."""
    outline = "[File Outline]\n"
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                outline += f"  class {node.name}\n"
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        outline += f"    def {item.name}(...)\n"
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                outline += f"  def {node.name}(...)\n"
    except SyntaxError:
        outline = "[Could not parse file outline]\n"
    return outline

import re

def _normalize_for_matching(text: str) -> str:
    """Normalize whitespace + line endings for fuzzy matching."""
    # Convert tabs to 4 spaces, normalize line endings, collapse trailing whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.expandtabs(4)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def apply_patch(original_content: str, search_block: str, replace_block: str,
                file_path: str = "") -> str:
    """
    Apply a search/replace patch with progressive fuzzy matching.
    Raises ValueError only if all fallback strategies fail.
    """
    # Strategy 1: exact match (fast path)
    if search_block in original_content:
        return original_content.replace(search_block, replace_block, 1)

    # Strategy 2: normalized whitespace match
    norm_original = _normalize_for_matching(original_content)
    norm_search = _normalize_for_matching(search_block)

    if norm_search in norm_original:
        # Find position in normalized, map back to original
        # Simpler approach: replace in normalized version
        norm_result = norm_original.replace(norm_search, _normalize_for_matching(replace_block), 1)
        # Caveat: this loses original indentation; acceptable since file gets re-formatted
        return norm_result

    # Strategy 3: line-by-line match (ignore indentation entirely)
    search_lines = [line.strip() for line in search_block.strip().split("\n") if line.strip()]
    original_lines = original_content.split("\n")

    if len(search_lines) >= 1:
        # Find a window in original_lines where each non-empty line of search matches
        for i in range(len(original_lines) - len(search_lines) + 1):
            window = [line.strip() for line in original_lines[i:i + len(search_lines)] if line.strip()]
            if window == search_lines:
                # Found it — replace this window
                indent = ""
                if original_lines[i]:
                    indent = original_lines[i][:len(original_lines[i]) - len(original_lines[i].lstrip())]
                replace_lines = replace_block.split("\n")
                replace_lines = [indent + line for line in replace_lines]
                new_lines = original_lines[:i] + replace_lines + original_lines[i + len(search_lines):]
                return "\n".join(new_lines)

    # Strategy 4: regex match (ignore exact whitespace between tokens)
    try:
        pattern = re.escape(search_block.strip())
        # Replace escaped whitespace with flexible whitespace
        pattern = re.sub(r"(\\[\s\n\r\t]+)", r"\\s+", pattern)
        match = re.search(pattern, original_content)
        if match:
            return original_content[:match.start()] + replace_block + original_content[match.end():]
    except re.error:
        pass

    # All strategies failed — raise with diagnostic info
    snippet = search_block[:200].replace("\n", " ")
    raise ValueError(
        f"{file_path}: 'search_block' not found even after fuzzy matching. "
        f"Search starts with: '{snippet}...'"
    )

def extract_relevant_context(code: str, affected_symbols: list) -> str:
    """
    Context Windowing: For files > 100 lines, extract only relevant functions/classes
    being modified plus a 20-line buffer. Otherwise return full code.
    """
    lines = code.split("\n")

    # If file is small, return everything
    if len(lines) <= 100:
        return code

    # Extract line ranges for affected symbols
    affected_ranges = []
    for sym in affected_symbols:
        sym_line = sym.get("line", 0) if isinstance(sym, dict) else 0
        if sym_line > 0:
            start = max(0, sym_line - 20)
            end = min(len(lines), sym_line + 20)
            affected_ranges.append((start, end))

    if not affected_ranges:
        return "\n".join(lines[:100])

    # Merge overlapping ranges
    affected_ranges.sort()
    merged_ranges = [affected_ranges[0]]
    for start, end in affected_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        if start <= last_end + 10:
            merged_ranges[-1] = (last_start, max(last_end, end))
        else:
            merged_ranges.append((start, end))

    # Extract windowed sections
    windowed_code = ""
    for start, end in merged_ranges:
        windowed_code += "\n".join(lines[start:end]) + "\n... (code omitted) ...\n"

    return windowed_code

def eval_first_check(state: CodegenState) -> CodegenState:
    """Check for goldenset.yaml before proceeding with code generation."""
    workspace_path = state.get('workspace_path', '')
    goldenset_path = os.path.join(workspace_path, 'goldenset.yaml')

    if os.path.exists(goldenset_path):
        audit(state.get('thread_id', 'unknown'), 'phase4', 'INFO', 'system', {
            'message': 'Eval-First Check Passed: goldenset.yaml found.'
        })
        print("  [Phase 4] Eval-First Check: PASSED (goldenset.yaml found)")
    else:
        audit(state.get('thread_id', 'unknown'), 'phase4', 'WARNING', 'system', {
            'message': 'Eval-First Check Failed: goldenset.yaml missing. Proceeding in SOFT MODE.'
        })
        print("  [Phase 4] Eval-First Check: WARNING (goldenset.yaml missing — soft mode)")

    return state

def load_existing_code(state: CodegenState) -> CodegenState:
    """
    Read current content of all affected files for brownfield codegen.
    For new projects (no affected files), skip entirely.
    For existing projects, auto-clone the target repo if needed, then read files
    with GitHub API fallback.
    """
    print("\n[Phase 4] Loading existing code...")

    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files", [])

    if not affected_files:
        print("  [Phase 4] No existing files — NEW PROJECT, generating fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    # Group affected files by repo so we clone each repo at most once
    repos_seen = set()
    existing_code = {}

    for af in affected_files:
        repo_name = af.get("repo_name", "")
        file_path = af.get("file_path", "")
        if not repo_name or not file_path:
            continue

        if repo_name not in repos_seen:
            # Best-effort clone before first file read for that repo
            ensure_repo_cloned(repo_name)
            repos_seen.add(repo_name)

        content = ws_read_file(repo_name, file_path)
        if content:
            # Key by "<repo>:<path>" so codegen knows which repo each file lives in
            existing_code[f"{repo_name}:{file_path}"] = content
            print(f"  ✅ Loaded: {repo_name}/{file_path} ({len(content)} chars)")
        else:
            print(f"  ⚠️  Could not read: {repo_name}/{file_path} (both local + API failed)")

    if not existing_code:
        # Affected files declared but none readable — fall back to greenfield path
        # rather than crashing. The LLM will get nothing useful otherwise.
        print("  ⚠️  Phase 3 declared affected files but none could be read — treating as fresh")
        return {**state, "existing_code": {}, "status": "NEW_PROJECT_NO_CODE"}

    return {**state, "existing_code": existing_code, "status": "CODE_LOADED"}

def generate_fresh_project(state: CodegenState) -> CodegenState:
    print("  [Phase 4] Generating FRESH Polyglot project scaffold...")

    requirement = state["requirement"]
    adr = state.get("adr", {})
    impact = state.get("impact_report", {})
    architecture = impact.get("architecture", {})

    # Extract ADR and Architecture context
    adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
    arch_context = ""
    if architecture.get("nodes"):
        arch_summary = [f"- {n.get('name','')} ({n.get('type','service')}): {n.get('description','')}" for n in architecture.get("nodes",[])]
        arch_context = "\nARCHITECTURE TO IMPLEMENT:\n" + "\n".join(arch_summary)

    prompt = f"""
    You are a Senior Polyglot Software Architect scaffolding a brand new multi-service project.

    ADR (Agreed Tech Stack):
    {adr_text}
    {arch_context}

    REQUIREMENT:
    {requirement}

    INSTRUCTIONS:
    1. Read the ADR and Architecture to determine the EXACT programming languages and frameworks required.
    2. Generate a complete, production-ready starter project scaffold for ALL requested nodes.
    3. Include standard configuration files appropriate for the chosen stack (e.g., package.json, tsconfig.json, pom.xml, or requirements.txt).
    4. Provide the core application entry points, routes/controllers, models, and a README.md.
    5. Prefix file paths with the service/repo name to keep them organized (e.g., 'backend/main.py' or 'frontend/src/App.tsx').
    6. CRITICAL: DO NOT default to Python unless explicitly specified in the ADR or Architecture.

    Return ONLY valid JSON in this exact format:
    {{
      "files": [
        {{
          "file_path": "path/to/file.ext",
          "content": "complete file content as a string",
          "change_summary": "what this file does",
          "new_symbols_added": ["ClassName", "function_name"],
          "existing_symbols_modified": []
        }}
      ]
    }}
    """

    response = call_llm(prompt, max_tokens=4000)
    if response.startswith("```"):
        response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

    try:
        data = json.loads(response)
        generated_changes = data.get("files", [])
    except Exception as e:
        print(f"  [Phase 4] JSON parse failed: {e}")
        generated_changes = []

    errors = []
    for change in generated_changes:
        errors.extend(validate_python(change.get("content", ""), change.get("file_path", "")))

    print(f"  [Phase 4] Generated {len(generated_changes)} fresh files")
    return {
        **state,
        "generated_changes": generated_changes,
        "validation_errors": errors,
        "status": "CODE_GENERATED" if not errors else "CODE_GENERATION_FAILED"
    }









# -----------------------------------------
# Nodes
# -----------------------------------------




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
        # existing_code is keyed by "<repo>:<path>" — fall back to bare path for safety
        current_content = (
                state["existing_code"].get(f"{repo_name}:{file_path}")
                or state["existing_code"].get(file_path, "")
        )

        context["files"].append({
            "file_path": file_path,
            "repo_name": repo_name,
            "current_content": current_content,
            "existing_symbols": symbols,
            "matched_symbols": af.get("matched_symbols", [])
        })

    return context


def generate_code_changes(state: CodegenState) -> CodegenState:
    print("\n[Phase 4] Generating Diff-Based Polyglot Code Changes...")

    if state.get("status") == "NEW_PROJECT_NO_CODE":
        return generate_fresh_project(state)

    impact = state.get("impact_report", {})
    affected_files = impact.get("affected_files", [])
    context = build_context_packet(state)
    asp = state.get("scope_contract", {})
    requirement = state.get("requirement", "")

    # ── CLAUDE'S RAG CONTEXT BUILDER ──
    packet = build_context_packet_rag(
        requirement=requirement,
        asp=asp,
        top_k=8, max_files=4
    )

    adr = state.get("adr", {})
    adr_text = json.dumps(adr, indent=2) if adr else "ADR not provided."
    generated_changes = []
    all_errors = []
    workspace_path = state.get('workspace_path', '')
    # Build the brownfield RAG context once for the whole batch
    packet = build_context_packet_rag(
        requirement=requirement,
        asp=asp,
        top_k=8, max_files=4,
        selected_repos=state.get("selected_repos", []),
    )
    print(f"  [Phase 4] RAG packet size: {len(packet)} chars")

    for file_info in context["files"]:
        file_path = file_info["file_path"]
        current_content = file_info["current_content"]
        print(f"\n  Processing: {file_path}")

        file_outline = extract_file_outline(current_content)
        # >>> FIX 1: send FULL content with line numbers so LLM copies exactly
        numbered = "\n".join(
            f"{i+1:4d}| {line}" for i, line in enumerate(current_content.split("\n"))
        )
        windowed_content = extract_relevant_context(current_content, file_info["existing_symbols"])
        content_for_llm = current_content if len(current_content.split("\n")) <= 100 else windowed_content

        brownfield_ctx_block = f"\n\n{packet}\n\n" if packet else ""

        base_prompt = f"""
You are an Expert Polyglot Software Engineer modifying an existing codebase.

ADR (Agreed Tech Stack): {adr_text}
REQUIREMENT: {requirement}
RISK LEVEL: {context['risk_level']}
{brownfield_ctx_block}

FILE TO MODIFY: {file_path}

{file_outline}

RELEVANT FILE CONTENT (May be truncated for length):
```
{content_for_llm}
```
INSTRUCTIONS:
1. Read the ADR and file extension to determine the programming language.
2. Modify this file to implement the requirement. Keep existing functionality working.
3. Return ONLY the exact blocks of code that need to be changed.
4. The 'search_block' MUST match the existing file content exactly character-by-character.
5. EVERY file change MUST include a 'traces_to' field mapping back to the ASP.

Return ONLY valid JSON in this exact format:
{{
  "file_path": "{file_path}",
  "changes": [
    {{
      "search_block": "Exact existing code snippet to replace",
      "replace_block": "The new code to insert"
    }}
  ],
  "change_summary": "what was changed and why",
  "traces_to": ["capability phrase from ASP"]
}}
"""
        # Inject Qdrant chunks!
        prompt = inject_into_user_message(base_prompt, packet)

        max_retries = 3
        success = False

        for attempt in range(1, max_retries + 1):
            response = call_llm(prompt, max_tokens=4000)
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?", "", response).strip().strip("```").strip()

            try:
                change_data = json.loads(response)
                modified_content = current_content
                diff_errors = []

                if change_data.get("changes"):
                    for diff in change_data["changes"]:
                        search_block = diff.get("search_block", "")
                        replace_block = diff.get("replace_block", "")

                        if not search_block:
                            continue
                        try:
                            modified_content = apply_patch(
                                modified_content,
                                search_block,
                                replace_block,
                                file_path=file_path
                            )
                        except ValueError as e:
                            diff_errors.append(str(e))
                            continue

                    if diff_errors:
                        errors = diff_errors
                    else:
                        change_data["content"] = modified_content
                        errors = validate_python(modified_content, file_path)
                else:
                    errors = [f"{file_path}: No 'changes' array provided by LLM."]

                if errors:
                    print(f"  ⚠️  Attempt {attempt}/{max_retries} — errors: {errors}")
                    if attempt < max_retries:
                        # When retry is triggered by 'search_block not found':
                        retry_feedback = """
CRITICAL FEEDBACK FROM PREVIOUS ATTEMPT:

The patch's `search_block` did NOT match the actual file content. The file content
you saw in the EXISTING CODE CONTEXT block IS the ground truth. Do not paraphrase,
do not reformat indentation, do not change quotes. Copy the EXACT text from the
context as your search_block — character-for-character.

If you cannot find an exact verbatim string in the existing code that matches what
you want to modify, then this file should be CREATED FRESH (no search_block, just
a full new file body) rather than patched.

For files you cannot patch reliably, RETURN:
{
  "file_path": "path/to/file.java",
  "action": "skip_with_reason",
  "reason": "Could not find verbatim match in existing code"
}

Do NOT invent code that isn't there.
"""
                        prompt += f"\n\n{retry_feedback}"
                        # ─────────────────────────────────────────────
                    continue
                if workspace_path:
                    full_path = os.path.join(workspace_path, change_data["file_path"])
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)

                generated_changes.append(change_data)
                print(f"  ✅ Generated patch for: {file_path}")
                success = True
                break

            except json.JSONDecodeError as e:
                print(f"  ⚠️  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
                if attempt < max_retries:
                    prompt += f"\n\nJSON ERROR: {e}\nReturn ONLY valid JSON."

        if not success:
            all_errors.append(f"{file_path}: Failed after {max_retries} attempts")

    return {**state, "generated_changes": generated_changes, "validation_errors": all_errors, "status": "CODE_GENERATED" if not all_errors else "CODE_GENERATION_FAILED"}

# To avoid name clashing in Codegen, rename the imported builder at the top:
# from agents.context_packet_builder import build_context_packet as build_context_packet_rag


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

        # Check content is not empty, but allow empty Python package markers
        if not content.strip() and not file_path.endswith("__init__.py"):
            errors.append(f"{file_path}: Generated content is empty")

    total = len(changes) + len(errors)
    success_count = len(changes)
    success_rate = success_count / max(total, 1)

    # Threshold: tolerate up to 30% failures, but never accept 0 successes
    MIN_SUCCESS_RATE = 0.70

    if success_count == 0:
        # Nothing generated — hard fail
        return {
            **state,
            "status": "VALIDATION_FAILED",
            "errors": errors,
            "generated_changes": [],
            "failed_files_count": len(errors),
        }

    if success_rate < MIN_SUCCESS_RATE:
        # Too many failures — soft fail with partial output
        print(f"[Phase 4] ⚠️  Partial success: {success_count}/{total} files ({success_rate:.0%}) — below {MIN_SUCCESS_RATE:.0%} threshold")
        return {
            **state,
            "status": "PARTIAL_SUCCESS_BELOW_THRESHOLD",
            "errors": errors,
            "generated_changes": changes,
            "warning": f"Generated {success_count} of {total} files. Below threshold but partial output available for review.",
            "failed_files_count": len(errors),
        }

    # ≥70% success — accept and continue, but expose failures for visibility
    if errors:
        print(f"[Phase 4] ✅ Partial success accepted: {success_count}/{total} files ({success_rate:.0%})")
        return {
            **state,
            "status": "CODE_GENERATED_WITH_WARNINGS",
            "errors": errors,  # not failures, warnings now
            "generated_changes": changes,
            "warning": f"Generated {success_count} of {total} files successfully. {len(errors)} files failed and were skipped.",
            "failed_files_count": len(errors),
        }

    # All succeeded
    return {**state, "status": "CODE_GENERATED", "generated_changes": changes}


def route_after_validation(state: CodegenState) -> str:
    """Route after validate_changes.

    Accept any of the success-shaped statuses so the graph terminates the
    right way. The original version only checked for the literal "VALIDATED"
    which validate_changes never produces.
    """
    if state.get("status") in ("CODE_GENERATED", "CODE_GENERATED_WITH_WARNINGS"):
        return "pass"
    return "fail"


# -----------------------------------------
# CI/CD Artifact Generation (Phase 4 → 7 bridge)
# -----------------------------------------

def generate_cicd_node(state: CodegenState) -> CodegenState:
    """Generate Dockerfile, .deploy.yaml, and GitHub Actions CI/CD workflow.

    Runs after main code generation but before validation, so the new files
    flow through the same validation step as the rest of generated_changes.

    Skips entirely if code generation produced nothing (no point creating
    deploy infra for an empty pipeline).

    Brownfield safety: we look at what's *actually on disk* in the cloned
    repo, NOT just at the files Phase 3 flagged as affected. Phase 3's
    `affected_files` only lists files relevant to the requirement (e.g.
    `app/models.py`) — it never includes infra files like `Dockerfile`.
    If we only consulted that list, we'd miss existing infra and clobber
    it. This function scans the repo root directly.
    """
    from agents.phase4_codegen.cicd_generator import generate_cicd_artifacts
    from agents.repo_workspace import get_repo_local_path

    changes = state.get("generated_changes", [])
    if not changes:
        print("\n[Phase 4] No code generated — skipping CI/CD artifact generation")
        return state

    print("\n[Phase 4] Generating CI/CD + deployment artifacts...")

    # ── Detect whether this is a brownfield run by checking impact_report ──
    # If Phase 3 listed affected files, the repo already exists. If the status
    # came in as NEW_PROJECT_NO_CODE, it's greenfield. We need both pieces:
    #   - is_brownfield: whether to apply the "be conservative" policy
    #   - existing_files: what's actually present on disk so we don't clobber
    impact = state.get("impact_report") or {}
    affected_files = impact.get("affected_files", [])
    is_brownfield = bool(affected_files) and state.get("status") != "NEW_PROJECT_NO_CODE"

    existing_files: set = set()
    if is_brownfield:
        # Collect distinct repo names from affected_files and scan each one
        # on disk. We only look at the *top-level* + a few well-known infra
        # paths because deploy artifacts always live near the repo root.
        repo_names = {af.get("repo_name") for af in affected_files if af.get("repo_name")}
        for repo_name in repo_names:
            try:
                repo_path = get_repo_local_path(repo_name)
            except Exception as e:
                print(f"  [cicd] cannot resolve local path for {repo_name}: {e}")
                continue
            if not repo_path or not os.path.isdir(repo_path):
                print(f"  [cicd] repo {repo_name} not cloned locally — "
                      f"cannot detect existing infra files")
                continue

            # Top-level files (Dockerfile, .deploy.yaml, .dockerignore, etc.)
            for entry in os.listdir(repo_path):
                full = os.path.join(repo_path, entry)
                if os.path.isfile(full):
                    existing_files.add(entry)

            # Known infra subdirectories — we look one level deep
            for sub in (".github/workflows",):
                sub_path = os.path.join(repo_path, sub)
                if os.path.isdir(sub_path):
                    for entry in os.listdir(sub_path):
                        if os.path.isfile(os.path.join(sub_path, entry)):
                            existing_files.add(f"{sub}/{entry}")

        # Echo what we found so the run log makes the decision auditable
        infra_present = sorted(
            f for f in existing_files
            if f in ("Dockerfile", ".deploy.yaml", ".dockerignore")
            or f.startswith(".github/workflows/")
        )
        if infra_present:
            print(f"  [cicd] brownfield infra detected on disk: {infra_present}")
        else:
            print(f"  [cicd] brownfield repo but no infra files found "
                  f"(Dockerfile/.deploy.yaml/workflow) — LLM will decide")
    else:
        print("  [cicd] greenfield project — generating full deployment stack")

    result = generate_cicd_artifacts(
        requirement=state.get("requirement", ""),
        generated_changes=changes,
        adr=state.get("adr"),
        architecture=(state.get("impact_report") or {}).get("architecture"),
        existing_files_in_repo=existing_files,
        is_brownfield=is_brownfield,
    )

    new_files = result.get("new_files", [])
    if new_files:
        # Append, don't replace — keep all the regular files Phase 4 generated.
        combined = list(changes) + new_files
        return {
            **state,
            "generated_changes": combined,
            "cicd_decision": result.get("decision"),
        }

    return {**state, "cicd_decision": result.get("decision")}


def validate_cicd_node(state: CodegenState) -> CodegenState:
    """Sanity-check CI/CD artifacts WITHOUT running them.

    Per directive: no tests run. We only check that:
      - YAML files parse
      - Dockerfile has the required directives (FROM, EXPOSE if a port was
        promised, CMD or ENTRYPOINT)
      - GitHub Actions workflow has the structural fields (`on`, `jobs`)
      - .deploy.yaml has the keys Phase 7 will read (deploy_target, service_name)

    Failures here are recorded as warnings — they don't fail the pipeline,
    because deployment artifacts being suboptimal shouldn't kill an
    otherwise-successful codegen.
    """
    print("\n[Phase 4] Validating CI/CD artifacts (no tests run)...")
    changes = state.get("generated_changes", [])
    warnings: list = list(state.get("cicd_warnings", []))

    for c in changes:
        path = c.get("file_path", "")
        content = c.get("content", "")

        # Dockerfile checks
        if path == "Dockerfile" or path.endswith("/Dockerfile"):
            if "FROM " not in content:
                warnings.append(f"{path}: missing FROM directive")
            if "CMD " not in content and "ENTRYPOINT " not in content:
                warnings.append(f"{path}: missing CMD or ENTRYPOINT")

        # YAML files (deploy.yaml + workflow) — parse only
        elif path == ".deploy.yaml" or path.endswith(".deploy.yaml") \
                or (path.startswith(".github/workflows/") and (path.endswith(".yml") or path.endswith(".yaml"))):
            try:
                import yaml
                parsed = yaml.safe_load(content)
                if not isinstance(parsed, dict):
                    warnings.append(f"{path}: YAML did not parse to a mapping")
                else:
                    if path.endswith(".deploy.yaml"):
                        for required_key in ("deploy_target", "service_name"):
                            if required_key not in parsed:
                                warnings.append(
                                    f"{path}: missing required key '{required_key}'"
                                )
                    elif path.startswith(".github/workflows/"):
                        for required_key in ("on", "jobs"):
                            if required_key not in parsed and True not in parsed:
                                # GitHub Actions weirdly parses `on:` → bool True
                                # under some yaml libs; allow both
                                warnings.append(
                                    f"{path}: missing required key '{required_key}'"
                                )
            except Exception as e:
                warnings.append(f"{path}: YAML parse error — {e}")

    if warnings:
        print(f"  ⚠ {len(warnings)} CI/CD validation warning(s):")
        for w in warnings[:10]:
            print(f"    - {w}")
        if len(warnings) > 10:
            print(f"    (+ {len(warnings) - 10} more)")
    else:
        print("  ✅ All CI/CD artifacts validated")

    return {**state, "cicd_warnings": warnings}


# -----------------------------------------
# Build Graph
# -----------------------------------------

def build_codegen_graph():
    builder = StateGraph(CodegenState)

    builder.add_node("eval_first_check", eval_first_check)
    builder.add_node("load_existing_code", load_existing_code)
    builder.add_node("generate_fresh_project", generate_fresh_project)
    builder.add_node("generate_code_changes", generate_code_changes)
    builder.add_node("generate_cicd", generate_cicd_node)         # ← NEW
    builder.add_node("validate_cicd", validate_cicd_node)         # ← NEW
    builder.add_node("validate_changes", validate_changes)

    # Single linear flow into the router
    builder.set_entry_point("eval_first_check")
    builder.add_edge("eval_first_check", "load_existing_code")

    # load_existing_code routes to ONE of two generators — no extra edges
    def route_after_load(state: CodegenState) -> str:
        return "fresh" if state.get("status") == "NEW_PROJECT_NO_CODE" else "existing"

    builder.add_conditional_edges(
        "load_existing_code",
        route_after_load,
        {"fresh": "generate_fresh_project", "existing": "generate_code_changes"},
    )

    # Both generators flow into CI/CD artifact generation
    builder.add_edge("generate_fresh_project", "generate_cicd")
    builder.add_edge("generate_code_changes", "generate_cicd")

    # CI/CD generation → CI/CD validation (no tests, just sanity checks) → main validation
    builder.add_edge("generate_cicd", "validate_cicd")
    builder.add_edge("validate_cicd", "validate_changes")

    # Validation terminates the graph
    builder.add_conditional_edges(
        "validate_changes",
        route_after_validation,
        {"pass": END, "fail": END},
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# -----------------------------------------
# Run
# -----------------------------------------

def run_codegen(requirement: str, impact_report: dict, workspace_path: str, thread_id: str = "thread-codegen", adr: dict = None, scope_contract: dict = None):
    graph = build_codegen_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = CodegenState(
        requirement=requirement,
        scope_contract=scope_contract or {},
        impact_report=impact_report,
        adr=adr or {},
        existing_code={},
        generated_changes=[],
        validation_errors=[],
        status="STARTED",
        workspace_path=workspace_path,
        thread_id=thread_id
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

    result = run_codegen(requirement, mock_impact_report, "/tmp/mock_workspace", "test-codegen-1")

    # Show generated content for first file
    if result['generated_changes']:
        first = result['generated_changes'][0]
        print(f"\nSample output for {first['file_path']}:")
        print(first['content'][:500])