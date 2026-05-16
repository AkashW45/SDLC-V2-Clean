import os
import json
import time
import re
import requests
from dotenv import load_dotenv

load_dotenv()
import os
import time
from openai import OpenAI

# Initialize the client at the top of your file
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-v4-flash" # or deepseek-v4-flash depending on your needs

def story_to_jira_context(story: dict, epic: dict = None, project: str = "SDLC") -> dict:
    """
    Convert an SDLC-V2 sprint story to the jira_context shape the generator expects.
    Fills missing fields with empty defaults so the generator never crashes.
    """
    return {
        "key": story.get("story_id") or story.get("id") or "STORY-?",
        "summary": story.get("title") or story.get("summary") or "",
        "description": story.get("description") or "",
        "issue_type": "Story",
        "priority": story.get("priority") or "Medium",
        "status": "To Do",
        "project": project,
        "assignee": "",
        "fix_versions": [],
        "labels": story.get("labels") or [],
        "components": [],
        "environment": "",
        "flagged": False,
        "sprint": {"name": "", "goal": "", "state": ""},
        "parent": {
            "key": epic.get("epic_id", "") if epic else "",
            "summary": epic.get("title", "") if epic else "",
            "status": "",
        },
        "linked_tickets": [],
        "subtasks": [],
        "comments": [],
        "attachments": [],
        "changelog": [],
        # Bake the acceptance criteria into the description so the
        # anti-hallucination word-set picks them up
        "_acceptance_criteria": story.get("acceptance_criteria", []),
    }





# ─────────────────────────────────────────────────────────
# Build context block — everything from Jira, nothing added
# ─────────────────────────────────────────────────────────
def format_context_for_prompt(ctx: dict) -> str:
    lines = []

    lines.append(f"TICKET:    {ctx['key']}")
    lines.append(f"TYPE:      {ctx['issue_type']}")
    lines.append(f"PRIORITY:  {ctx['priority']}")
    lines.append(f"STATUS:    {ctx['status']}")
    lines.append(f"PROJECT:   {ctx['project']}")

    if ctx.get("assignee") and ctx["assignee"] != "Unassigned":
        lines.append(f"ASSIGNEE:  {ctx['assignee']}")
    if ctx.get("flagged"):
        lines.append("FLAGGED: This ticket is flagged — treat as elevated risk")
    if ctx.get("fix_versions"):
        lines.append(f"FIX VERSION: {', '.join(ctx['fix_versions'])}")
    if ctx.get("labels"):
        lines.append(f"LABELS: {', '.join(ctx['labels'])}")
    if ctx.get("components"):
        lines.append(f"COMPONENTS: {', '.join(ctx['components'])}")
    if ctx.get("environment"):
        lines.append(f"\nENVIRONMENT:\n{ctx['environment']}")

    lines.append(f"\nSUMMARY:\n{ctx['summary']}")

    if ctx.get("description"):
        lines.append(f"\nDESCRIPTION:\n{ctx['description']}")
    ac = ctx.get("_acceptance_criteria", [])
    if ac:
        lines.append("\nACCEPTANCE CRITERIA:")
        for i, a in enumerate(ac, 1):
            lines.append(f"  {i}. {a}")    

    sprint = ctx.get("sprint", {})
    if sprint.get("name"):
        lines.append(f"\nSPRINT: {sprint['name']} [{sprint.get('state','').upper()}]")
        if sprint.get("goal"):
            lines.append(f"SPRINT GOAL: {sprint['goal']}")

    parent = ctx.get("parent", {})
    if parent.get("summary"):
        lines.append(f"\nPARENT: [{parent['key']}] {parent['summary']} ({parent.get('status','')})")

    linked = ctx.get("linked_tickets", [])
    if linked:
        lines.append("\nLINKED TICKETS:")
        for lt in linked:
            lines.append(f"  [{lt['key']}] {lt['summary']} — {lt['link_type']} | {lt['status']}")

    subtasks = ctx.get("subtasks", [])
    if subtasks:
        lines.append("\nSUBTASKS:")
        for s in subtasks:
            lines.append(f"  [{s['key']}] {s['summary']} ({s['status']})")

    comments = ctx.get("comments", [])
    if comments:
        lines.append("\nTEAM DISCUSSION:")
        for c in comments[:5]:
            lines.append(f"  [{c['author']}] {c['date'][:10]}: {c['body'][:400]}")

    attachments = ctx.get("attachments", [])
    if attachments:
        lines.append("\nATTACHMENTS:")
        for a in attachments:
            lines.append(f"  - {a['filename']} by {a['author']}")

    changes = ctx.get("changelog", [])
    if changes:
        lines.append("\nRECENT CHANGES:")
        for ch in changes[:4]:
            lines.append(f"  {ch['author']} changed [{ch['field']}]: {ch['from'] or 'empty'} → {ch['to'] or 'empty'}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Extract all meaningful words from context
# Used to validate test cases aren't hallucinated
# ─────────────────────────────────────────────────────────
def _extract_context_vocabulary(ctx: dict) -> set:
    """
    Build a set of meaningful words from the Jira context.
    Any test case content should be derivable from these words.
    """
    text_parts = [
        ctx.get("summary", ""),
        ctx.get("description", ""),
        ctx.get("environment", ""),
        " ".join(ctx.get("labels", [])),
        " ".join(ctx.get("components", [])),
        " ".join(ctx.get("fix_versions", [])),
        ctx.get("issue_type", ""),
        ctx.get("priority", ""),
        ctx.get("project", ""),
        ctx.get("key", ""),
    ]

    for c in ctx.get("comments", []):
        text_parts.append(c.get("body", ""))
    for lt in ctx.get("linked_tickets", []):
        text_parts.append(lt.get("summary", ""))
    for s in ctx.get("subtasks", []):
        text_parts.append(s.get("summary", ""))

    parent = ctx.get("parent", {})
    text_parts.append(parent.get("summary", ""))

    sprint = ctx.get("sprint", {})
    text_parts.append(sprint.get("name", ""))
    text_parts.append(sprint.get("goal", ""))

    full_text = " ".join(text_parts).lower()
    # Extract words of 3+ chars, ignore stopwords
    stopwords = {
        "the","and","for","are","was","has","had","not","but","with",
        "from","this","that","they","have","will","been","when","what",
        "your","also","into","its","our","can","all","any","each","more"
    }
    words = set(re.findall(r'\b[a-z]{3,}\b', full_text)) - stopwords
    return words


# ─────────────────────────────────────────────────────────
# Validate a single test case against context
# Returns (is_valid, reason)
# ─────────────────────────────────────────────────────────
def _validate_test_case(tc: dict, vocab: set, ctx: dict) -> tuple:

    title    = (tc.get("title") or "").lower()
    steps    = " ".join(tc.get("steps") or []).lower()
    expected = (tc.get("expected_result") or "").lower()
    combined = f"{title} {steps} {expected}"

    # Rule 1 — must have all required fields
    for field in ["id", "title", "steps", "expected_result", "category", "priority"]:
        if not tc.get(field):
            return False, f"Missing required field: {field}"

    # Rule 2 — steps must be a list with at least 2 items
    if not isinstance(tc.get("steps"), list) or len(tc["steps"]) < 2:
        return False, "Steps must be a list with at least 2 items"

    # Rule 3 — expected result must be specific (not generic filler)
    generic_phrases = [
        "it should work", "works correctly", "as expected",
        "successfully", "no errors", "properly", "correctly"
    ]
    if all(phrase in expected for phrase in generic_phrases[:2]) and len(expected) < 30:
        return False, "Expected result is too generic"

    # Rule 4 — title and content must contain words from Jira context
    # At least 2 meaningful words from context must appear in the test case
    combined_words = set(re.findall(r'\b[a-z]{3,}\b', combined))
    overlap = combined_words & vocab
    if len(overlap) < 2:
        return False, f"Test case content not traceable to Jira context (overlap: {overlap})"

    # Rule 5 — priority must be valid
    if tc.get("priority") not in ("P1", "P2", "P3"):
        return False, f"Invalid priority: {tc.get('priority')}"

    # Rule 6 — category must be valid
    valid_cats = {"Functional","Edge Case","Negative","Integration","Regression","Reproduction"}
    if tc.get("category") not in valid_cats:
        return False, f"Invalid category: {tc.get('category')}"

    return True, "OK"


# ─────────────────────────────────────────────────────────
# Validate entire test suite — strip hallucinated cases
# ─────────────────────────────────────────────────────────
def _validate_suite(result: dict, ctx: dict) -> dict:
    vocab = _extract_context_vocabulary(ctx)
    suite = result.get("test_suite", [])

    valid_cases   = []
    removed_cases = []

    for tc in suite:
        is_valid, reason = _validate_test_case(tc, vocab, ctx)
        if is_valid:
            valid_cases.append(tc)
        else:
            removed_cases.append({"id": tc.get("id","?"), "reason": reason})

    if removed_cases:
        existing_notes = result.get("qa_notes", "")
        removed_info   = ", ".join([f"{r['id']} ({r['reason']})" for r in removed_cases])
        result["qa_notes"] = (
            f"{existing_notes}\n"
            f"⚠️ {len(removed_cases)} hallucinated test case(s) removed: {removed_info}"
        ).strip()

    result["test_suite"] = valid_cases
    return result


def get_type_specific_rules(issue_type: str) -> str:
    issue_type = (issue_type or "").lower()
    if issue_type == "bug":
        return """
ISSUE TYPE IS BUG — use the reproduction steps from the description:
- TC-001 must reproduce the exact bug using the steps written in the description
- TC-002 must verify the fix resolves the issue
- TC-003 must be a regression test — adjacent functionality still works
- Reference the specific environment (OS/browser) if mentioned
"""
    elif issue_type in ("story", "user story"):
        return """
ISSUE TYPE IS USER STORY — map directly to acceptance criteria in description:
- One test case per acceptance criteria point — use the exact AC wording
- Include the happy path as described
- Include at least 2 negative paths based on the inputs mentioned
- Do not invent acceptance criteria not in the description
"""
    elif issue_type == "epic":
        return """
ISSUE TYPE IS EPIC — cover the scope described:
- Generate integration test cases covering capabilities mentioned in description
- Reference only the features and flows explicitly listed
"""
    else:
        return """
ISSUE TYPE IS TASK — verify only what is described:
- Test the specific deliverable described in summary and description
- Include integration test only if external systems are explicitly mentioned
"""


def build_prompt(context_block: str, issue_type: str) -> str:
    type_rules = get_type_specific_rules(issue_type)

    return f"""You are a senior QA engineer generating test cases from a Jira ticket.

STRICT ANTI-HALLUCINATION RULES — these are absolute:
1. Use ONLY information present in the JIRA CONTEXT below — nothing else
2. Every test case title, step, and expected result must be directly traceable to the context
3. Do NOT invent features, fields, endpoints, or behaviours not mentioned in the context
4. Do NOT use generic phrases like "it should work" or "system works correctly"
5. If description mentions a specific URL, field name, or error — use those exact terms
6. If context is sparse — write fewer but accurate test cases, not more invented ones
7. Never reference technology, frameworks, or systems not mentioned in the context

OUTPUT RULES:
- Return ONLY valid JSON — no markdown, no code fences, no text before or after
- Generate between 6 and 12 test cases
- Always include at least 2 Negative and 1 Regression test case

JSON SCHEMA:
{{
  "ticket_key": "exact ticket key from context",
  "summary": "exact summary from context",
  "issue_type": "exact issue type from context",
  "risk_level": "Low | Medium | High | Critical",
  "test_suite": [
    {{
      "id": "TC-001",
      "category": "Functional | Edge Case | Negative | Integration | Regression | Reproduction",
      "title": "Specific title using exact terms from context",
      "preconditions": "Exact state required — only mention systems/data in context",
      "steps": [
        "Step 1: specific action using terms from context",
        "Step 2: specific action",
        "Step 3: verify specific outcome from context"
      ],
      "expected_result": "Specific measurable outcome using exact terms from context",
      "priority": "P1 | P2 | P3",
      "tags": ["from-context-labels-or-components"]
    }}
  ],
  "coverage_summary": {{
    "total_cases": 0,
    "functional": 0,
    "edge_cases": 0,
    "negative": 0,
    "integration": 0,
    "regression": 0,
    "reproduction": 0
  }},
  "qa_notes": "Risks and dependencies — only from context"
}}

PRIORITY RULES:
- P1 = must pass before release
- P2 = important
- P3 = nice to have

{type_rules}

JIRA CONTEXT — use only what is below, nothing else:
{context_block}
"""


# ─────────────────────────────────────────────────────────
# Parse LLM response safely
# ─────────────────────────────────────────────────────────
def _parse_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = [l for l in content.split("\n") if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    start = content.find("{")
    end   = content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(content[start:end])


# ─────────────────────────────────────────────────────────
# Recount coverage from actual suite — never trust LLM count
# ─────────────────────────────────────────────────────────
def _fix_coverage_summary(result: dict) -> dict:
    suite   = result.get("test_suite", [])
    cat_map = {
        "functional": 0, "edge_cases": 0, "negative": 0,
        "integration": 0, "regression": 0, "reproduction": 0,
    }
    for t in suite:
        cat = (t.get("category") or "").lower().replace(" ", "_")
        if cat in cat_map:
            cat_map[cat] += 1
        elif "edge" in cat:
            cat_map["edge_cases"] += 1
        elif "negative" in cat:
            cat_map["negative"] += 1
        else:
            cat_map["functional"] += 1

    result["coverage_summary"] = {
        "total_cases":  len(suite),
        **cat_map
    }
    return result


# ─────────────────────────────────────────────────────────
# Fallback — built purely from Jira context fields
# No invented content — uses only what we know from ticket
# ─────────────────────────────────────────────────────────
def _make_fallback(ctx: dict) -> dict:
    key        = ctx.get("key", "TC")
    summary    = ctx.get("summary", "feature")
    issue_type = ctx.get("issue_type", "Task")
    env        = ctx.get("environment", "the target environment")
    version    = (ctx.get("fix_versions") or ["the fix version"])[0]
    component  = (ctx.get("components") or ["the component"])[0]
    labels     = ctx.get("labels", [])
    tags       = labels[:2] if labels else [issue_type.lower()]

    suite = [
        {
            "id": f"{key}-TC-001",
            "category": "Functional",
            "title": f"Verify {summary} completes successfully",
            "preconditions": f"System deployed to {env}, version {version} active",
            "steps": [
                f"Step 1: Navigate to {component}",
                f"Step 2: Perform the action described in {key}: {summary[:60]}",
                "Step 3: Verify the operation completes without errors"
            ],
            "expected_result": f"{summary[:80]} completes and system remains stable",
            "priority": "P1",
            "tags": tags
        },
        {
            "id": f"{key}-TC-002",
            "category": "Negative",
            "title": f"Verify {summary} handles invalid input gracefully",
            "preconditions": f"System deployed to {env}",
            "steps": [
                f"Step 1: Navigate to {component}",
                "Step 2: Submit invalid or empty input",
                "Step 3: Verify error handling"
            ],
            "expected_result": "Appropriate error message displayed, no system crash or data corruption",
            "priority": "P1",
            "tags": ["negative"] + tags
        },
        {
            "id": f"{key}-TC-003",
            "category": "Negative",
            "title": f"Verify {summary} handles missing required data",
            "preconditions": f"System deployed to {env}",
            "steps": [
                f"Step 1: Navigate to {component}",
                "Step 2: Leave required fields empty and submit",
                "Step 3: Verify validation response"
            ],
            "expected_result": "Validation error shown for missing required fields, no partial save",
            "priority": "P2",
            "tags": ["negative", "validation"] + tags
        },
        {
            "id": f"{key}-TC-004",
            "category": "Edge Case",
            "title": f"Verify {summary} handles boundary and edge conditions",
            "preconditions": f"System deployed to {env}, version {version}",
            "steps": [
                f"Step 1: Navigate to {component}",
                "Step 2: Enter boundary value inputs (min/max/special characters)",
                "Step 3: Verify system handles edge values correctly"
            ],
            "expected_result": "System handles boundary values without errors or unexpected behaviour",
            "priority": "P2",
            "tags": ["edge-case"] + tags
        },
        {
            "id": f"{key}-TC-005",
            "category": "Regression",
            "title": f"Verify existing {component} functionality not broken by {key} changes",
            "preconditions": f"Version {version} deployed to {env}",
            "steps": [
                f"Step 1: Test existing {component} functionality unrelated to {key}",
                "Step 2: Verify all previously working features still work",
                "Step 3: Check application logs for unexpected errors"
            ],
            "expected_result": f"All existing {component} functionality works as before — no regression",
            "priority": "P1",
            "tags": ["regression"] + tags
        },
        {
            "id": f"{key}-TC-006",
            "category": "Functional",
            "title": f"Verify {key} deployment in {env} environment",
            "preconditions": f"Version {version} deployed to {env}",
            "steps": [
                f"Step 1: Confirm version {version} is active in {env}",
                f"Step 2: Execute smoke test for {summary[:60]}",
                "Step 3: Verify no errors in deployment logs"
            ],
            "expected_result": f"Version {version} running in {env} with no deployment errors",
            "priority": "P2",
            "tags": ["smoke-test"] + tags
        },
    ]

    return {
        "ticket_key":  key,
        "summary":     summary,
        "issue_type":  issue_type,
        "risk_level":  "Medium",
        "test_suite":  suite,
        "coverage_summary": {
            "total_cases": 6, "functional": 2, "edge_cases": 1,
            "negative": 2, "integration": 0, "regression": 1, "reproduction": 0,
        },
        "qa_notes": (
            f"⚠️ Fallback test cases — generated from {key} context only "
            f"({summary[:60]}). AI generation failed. Review and enhance manually."
        )
    }


# ─────────────────────────────────────────────────────────
# Main entry point — retry + validate + fallback
# ─────────────────────────────────────────────────────────
def generate_test_cases(jira_context: dict) -> dict:

    context_block = format_context_for_prompt(jira_context)
    issue_type    = jira_context.get("issue_type", "Task")
    prompt        = build_prompt(context_block, issue_type)

    last_error = None

    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 * attempt)

              # ── DEEPSEEK SDK CALL (Replaces raw requests.post) ──
            response = client.chat.completions.create(
                model=MODEL, # e.g., "deepseek-chat" or "deepseek-v4-flash"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # zero temperature = no creativity = no hallucination
            )

            # Extract the text content from the SDK response object
            content = response.choices[0].message.content
            # ────────────────────────────────────────────────────
            result  = _parse_response(content)

            suite = result.get("test_suite", [])
            if not suite:
                raise ValueError("LLM returned empty test_suite")

            # Strip hallucinated test cases
            result = _validate_suite(result, jira_context)

            # If too many stripped — retry
            if len(result.get("test_suite", [])) < 3:
                raise ValueError(
                    f"Too many test cases failed validation "
                    f"({len(suite)} generated, {len(result.get('test_suite',[]))} valid)"
                )

            # Recount coverage from actual valid suite
            result = _fix_coverage_summary(result)
            return result

        except Exception as e:
            last_error = e
            continue

    # All attempts failed — use fallback built from Jira context only
    fallback = _make_fallback(jira_context)
    fallback["qa_notes"] += (
        f"\nGeneration error: {str(last_error)[:120]}"
    )
    return fallback