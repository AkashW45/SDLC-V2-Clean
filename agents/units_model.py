"""
agents/units_model.py  (FINAL — emits unit_counts for the decide() engine)

The decide() engine wants `unit_counts` — a dict like {"FR": 3, "CODE_FILE": 2, "ENDPOINT": 5}.
It applies depth-aware weights itself. So this module's job is just to COUNT, not to weight.

attach_units(artifact, asp) does, in order:
  1. extract unit_counts from the artifact (by type)
  2. attach artifact["unit_counts"]
  3. ALSO compute a flat units_estimate (for display / quick sorting) — the engine
     will recompute properly with depth weights, but this gives a sane preview number
  4. attach artifact["estimated_dev_hours"] for the dashboard

Public API:
  extract_unit_counts(artifact) -> dict
  attach_units(artifact, asp=None) -> artifact   (mutates + returns)
  units_to_hours(units) -> float
"""
import math

# Flat preview weights — ONLY for the quick units_estimate preview.
# The real depth-aware weighting happens inside expansion_engine.decide().
_PREVIEW_WEIGHTS = {
    "FR": 1.2, "NFR": 1.0, "CODE_FILE": 1.0, "ENDPOINT": 4.0,
    "DB_TABLE": 10.0, "MIGRATION": 8.0, "INTEGRATION": 20.0,
    "DEPENDENCY": 4.0, "TEST_CASE": 0.5, "INFRA_SERVICE": 15.0,
    "NODE": 3.0, "JIRA": 1.1, "ADR": 1.5, "SPRINT": 6.0,
}
_UNITS_TO_HOURS = 2.0


def units_to_hours(units: int) -> float:
    return round(units * _UNITS_TO_HOURS, 1)


# ────────────────────────────────────────────────────────────────────
# Per-type unit_counts extractors
# ────────────────────────────────────────────────────────────────────
def extract_unit_counts(artifact: dict) -> dict:
    """Extract a {UNIT_TYPE: count} dict from any artifact type."""
    atype = (artifact.get("type") or artifact.get("artifact_type") or "").upper()
    body = artifact.get("body", {})
    if isinstance(body, str):
        body = {}

    if atype in ("CODE", "PATCH"):
        return _count_code(artifact)
    elif atype in ("ARCH", "ARCHITECTURE"):
        return _count_architecture(body)
    elif atype in ("SPRINT", "SPRINT_PLAN"):
        return _count_sprint(body)
    elif atype == "TESTS":
        return _count_tests(artifact)
    elif atype == "BRD":
        return _count_brd(body)
    elif atype == "PRD":
        return _count_prd(body)
    elif atype == "ADR":
        return _count_adr(body)
    elif atype in ("DEPLOY", "DEPLOY_PLAN"):
        return _count_deploy(body)
    elif atype == "DOC":
        return {"CODE_FILE": 1}
    else:
        return {"CODE_FILE": 1}


def _count_code(artifact: dict) -> dict:
    files = artifact.get("files", [])
    if not files and isinstance(artifact.get("body"), dict):
        files = artifact["body"].get("files", [])

    code_files = 0
    endpoints = 0
    dependencies = 0
    db_tables = 0
    migrations = 0

    for f in files:
        path = (f.get("path") or f.get("file_path") or "").lower()
        content = f.get("content", "")
        code_files += 1

        # endpoints — FastAPI / Flask / Express route patterns
        endpoints += content.count("@app.") + content.count("@router.")
        endpoints += content.count(".route(") + content.count("app.get(") + content.count("app.post(")

        # dependencies — requirements.txt / package.json
        if "requirements.txt" in path:
            dependencies += len([
                l for l in content.splitlines()
                if l.strip() and not l.strip().startswith("#")
            ])
        elif "package.json" in path:
            dependencies += content.count('": "')  # rough dep count

        # db tables / models — SQLAlchemy models, CREATE TABLE
        db_tables += content.count("class ") if "models" in path else 0
        db_tables += content.upper().count("CREATE TABLE")

        # migrations
        if "migration" in path or "alembic" in path:
            migrations += 1

    counts = {"CODE_FILE": code_files}
    if endpoints:
        counts["ENDPOINT"] = endpoints
    if dependencies:
        counts["DEPENDENCY"] = min(dependencies, 40)  # cap noise
    if db_tables:
        counts["DB_TABLE"] = db_tables
    if migrations:
        counts["MIGRATION"] = migrations
    return counts


def _count_architecture(body: dict) -> dict:
    nodes = body.get("nodes", [])
    counts = {"NODE": len(nodes)}
    db = sum(1 for n in nodes if (n.get("type") or "").lower() == "database")
    integ = sum(1 for n in nodes if (n.get("type") or "").lower() == "external")
    infra = sum(1 for n in nodes if (n.get("type") or "").lower()
                in ("service", "gateway", "queue", "cache"))
    if db:
        counts["DB_TABLE"] = db
    if integ:
        counts["INTEGRATION"] = integ
    if infra:
        counts["INFRA_SERVICE"] = infra
    return counts


def _count_sprint(body: dict) -> dict:
    tickets = body.get("jira_tickets", [])
    sprints = body.get("sprint_plan", {}).get("sprints", [])
    counts = {"JIRA": len(tickets)}
    if sprints:
        counts["SPRINT"] = len(sprints)
    return counts


def _count_tests(artifact: dict) -> dict:
    test_files = artifact.get("test_files", [])
    if not test_files and isinstance(artifact.get("body"), dict):
        test_files = artifact["body"].get("test_files", [])
    total_tests = sum(tf.get("test_count", 1) for tf in test_files)
    return {"CODE_FILE": len(test_files), "TEST_CASE": total_tests}


def _count_brd(body: dict) -> dict:
    counts = {}
    frs = len(body.get("functional_requirements", []))
    nfrs = len(body.get("non_functional_requirements", []))
    if frs:
        counts["FR"] = frs
    if nfrs:
        counts["NFR"] = nfrs
    if not counts:
        counts["FR"] = len(body.get("top_features", [])) or 1
    return counts


def _count_prd(body: dict) -> dict:
    frs = len(body.get("functional_requirements", []))
    return {"FR": frs or 1}


def _count_adr(body: dict) -> dict:
    decisions = len(body.get("decisions", []))
    return {"ADR": decisions or 1}


def _count_deploy(body: dict) -> dict:
    sequence = body.get("deploy_sequence", [])
    migrations = sum(1 for s in sequence if (s.get("type") or "").lower() == "migration")
    counts = {}
    if migrations:
        counts["MIGRATION"] = migrations
    services = len(sequence) - migrations
    if services:
        counts["INFRA_SERVICE"] = services
    if not counts:
        counts["INFRA_SERVICE"] = 1
    return counts


# ────────────────────────────────────────────────────────────────────
# Main — attach unit_counts + preview estimate + dev-hours to artifact
# ────────────────────────────────────────────────────────────────────
def _preview_units(unit_counts: dict, compliance_needed: bool = False) -> int:
    """Flat preview estimate. The engine recomputes the real depth-aware number."""
    total = 0.0
    for k, count in unit_counts.items():
        weight = _PREVIEW_WEIGHTS.get(k.upper(), 1.0)
        total += count * weight
    if compliance_needed:
        total *= 1.5
    return max(1, math.ceil(total))


def attach_units(artifact: dict, asp: dict = None) -> dict:
    """
    Mutates and returns the artifact with:
      artifact["unit_counts"]        — {UNIT_TYPE: count}  (consumed by decide())
      artifact["units_estimate"]     — flat preview int    (display / pre-sort only)
      artifact["estimated_dev_hours"]— float               (dashboard)
      artifact["compliance_needed"]  — bool

    decide() will RECOMPUTE units_estimate using depth-aware weights — this is just a preview.
    """
    unit_counts = extract_unit_counts(artifact)
    artifact["unit_counts"] = unit_counts

    compliance_needed = False
    if asp:
        compliance = asp.get("anchors", {}).get("explicit_compliance", [])
        compliance_needed = bool(compliance)
    artifact["compliance_needed"] = compliance_needed

    preview = _preview_units(unit_counts, compliance_needed)
    artifact["units_estimate"] = preview
    artifact["estimated_dev_hours"] = units_to_hours(preview)
    return artifact
