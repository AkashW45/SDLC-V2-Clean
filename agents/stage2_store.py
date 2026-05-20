"""
agents/stage2_store.py  (FINAL — aligned with decide() engine)

Persistence helpers for Stage 2 — ASP, artifacts, decisions, unit weights, PR registry.
All other Stage 2 modules import from here. No business logic — pure DB access.

KEY CHANGE vs first version:
  load_unit_weights() now returns the DEPTH-KEYED format the decide() engine expects:
    { "FR": {1: 2.0, 2: 1.5, 3: 1.2, 4: 1.0, 5: 0.9}, "CODE_FILE": {...}, ... }
"""
import os
import json
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
from dotenv import load_dotenv

from core.db_clients import PooledConn

load_dotenv()


def _conn():
    # Returns a pooled connection that supports the same .cursor() / .commit()
    # / .close() API as a raw psycopg2 connection. .close() returns it to
    # the pool instead of dropping the socket.
    return PooledConn()


def _now():
    return datetime.now(timezone.utc).isoformat()


# ────────────────────────────────────────────────────────────────────
# ASP
# ────────────────────────────────────────────────────────────────────
def save_asp(asp: dict, thread_id: str, created_by: str = "classifier") -> str:
    asp_id = asp.get("asp_id") or str(uuid.uuid4())
    asp["asp_id"] = asp_id
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO asp (asp_id, thread_id, payload, allow_unbounded, policy_mode, depth_level, created_by, created_at, version)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (asp_id) DO UPDATE SET
            payload = EXCLUDED.payload,
                                           allow_unbounded = EXCLUDED.allow_unbounded,
                                           policy_mode = EXCLUDED.policy_mode,
                                           depth_level = EXCLUDED.depth_level,
                                           version = asp.version + 1""",
        (
            asp_id, thread_id, json.dumps(asp),
            asp.get("allow_unbounded", False),
            asp.get("policy_mode", "managed"),
            asp.get("depth_level"),
            created_by, _now(), asp.get("version", 1),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return asp_id


def get_asp(asp_id: str) -> dict:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT payload FROM asp WHERE asp_id = %s", (asp_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["payload"] if row else {}


def get_asp_by_thread(thread_id: str) -> dict:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT payload FROM asp WHERE thread_id = %s ORDER BY created_at DESC LIMIT 1",
        (thread_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["payload"] if row else {}


# ────────────────────────────────────────────────────────────────────
# Artifacts
# ────────────────────────────────────────────────────────────────────
def save_artifact(artifact: dict, asp_id: str, thread_id: str,
                  created_by: str = "generator") -> str:
    artifact_id = artifact.get("artifact_id") or str(uuid.uuid4())
    artifact["artifact_id"] = artifact_id
    artifact["asp_id"] = asp_id
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO artifacts (artifact_id, asp_id, thread_id, type, category, payload, units_estimate, confidence, status, created_by, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (artifact_id) DO UPDATE SET
            payload = EXCLUDED.payload,
                                                units_estimate = EXCLUDED.units_estimate,
                                                confidence = EXCLUDED.confidence,
                                                status = EXCLUDED.status""",
        (
            artifact_id, asp_id, thread_id,
            artifact.get("type") or artifact.get("artifact_type", "DOC"),
            artifact.get("category", "mvp"),
            json.dumps(artifact),
            artifact.get("units_estimate", 0),
            artifact.get("confidence", 0),
            artifact.get("status", "candidate"),
            created_by, _now(),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return artifact_id


def update_artifact_status(artifact_id: str, status: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("UPDATE artifacts SET status = %s WHERE artifact_id = %s", (status, artifact_id))
    conn.commit()
    cur.close()
    conn.close()


def get_artifacts_by_asp(asp_id: str, status: str = None) -> list:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    if status:
        cur.execute(
            "SELECT payload, status FROM artifacts WHERE asp_id = %s AND status = %s ORDER BY created_at",
            (asp_id, status),
        )
    else:
        cur.execute(
            "SELECT payload, status FROM artifacts WHERE asp_id = %s ORDER BY created_at",
            (asp_id,),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        art = r["payload"]
        art["status"] = r["status"]
        result.append(art)
    return result


def get_artifacts_by_thread(thread_id: str) -> list:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT payload, status FROM artifacts WHERE thread_id = %s ORDER BY created_at",
        (thread_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        art = r["payload"]
        art["status"] = r["status"]
        result.append(art)
    return result


# ────────────────────────────────────────────────────────────────────
# Decisions
# ────────────────────────────────────────────────────────────────────
def save_decision(artifact_id: str, thread_id: str, decision: dict,
                  decided_by: str) -> str:
    decision_id = str(uuid.uuid4())
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO artifact_decisions (decision_id, artifact_id, thread_id, decision, decided_by, decided_at)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (decision_id, artifact_id, thread_id, json.dumps(decision), decided_by, _now()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return decision_id


def get_decisions_by_artifact(artifact_id: str) -> list:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT decision, decided_by, decided_at FROM artifact_decisions WHERE artifact_id = %s ORDER BY decided_at",
        (artifact_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"decision": r["decision"], "decided_by": r["decided_by"], "decided_at": str(r["decided_at"])}
        for r in rows
    ]


# ────────────────────────────────────────────────────────────────────
# Unit weights — DEPTH-KEYED format for the decide() engine
# Table units_weights stores rows: (name, depth_level, weight)
# load_unit_weights() returns: { "FR": {1:2.0, 2:1.5, ...}, ... }
# ────────────────────────────────────────────────────────────────────
def _default_unit_weights() -> dict:
    """
    Depth-aware default weights.
    Logic: at LOW depth, each unit is 'heavier' (you want fewer, simpler things).
           at HIGH depth, each unit is 'lighter' (many units are expected & normal).
    This makes the SAME artifact cost more budget at depth 1 than at depth 5,
    which naturally lets bigger projects have more sprints/files/nodes.
    """
    return {
        "FR":          {1: 2.0, 2: 1.6, 3: 1.2, 4: 1.0, 5: 0.8},
        "NFR":         {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8},
        "CODE_FILE":   {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.6},
        "ENDPOINT":    {1: 6.0, 2: 5.0, 3: 4.0, 4: 3.5, 5: 3.0},
        "DB_TABLE":    {1: 14.0, 2: 12.0, 3: 10.0, 4: 9.0, 5: 8.0},
        "MIGRATION":   {1: 10.0, 2: 9.0, 3: 8.0, 4: 7.0, 5: 6.0},
        "INTEGRATION": {1: 25.0, 2: 22.0, 3: 20.0, 4: 18.0, 5: 16.0},
        "DEPENDENCY":  {1: 5.0, 2: 4.5, 3: 4.0, 4: 3.5, 5: 3.0},
        "TEST_CASE":   {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3},
        "INFRA_SERVICE": {1: 18.0, 2: 16.0, 3: 15.0, 4: 13.0, 5: 12.0},
        "NODE":        {1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0},
        "JIRA":        {1: 1.5, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.9},
        "ADR":         {1: 2.0, 2: 1.8, 3: 1.5, 4: 1.3, 5: 1.2},
        "SPRINT":      {1: 8.0, 2: 7.0, 3: 6.0, 4: 5.5, 5: 5.0},
    }


def load_unit_weights() -> dict:
    """
    Returns depth-keyed unit weights: { UNIT_TYPE: { depth: weight } }.
    Reads from units_weights table if rows exist, otherwise returns defaults.
    Table schema expected: units_weights(name TEXT, depth_level INT, weight FLOAT).
    """
    defaults = _default_unit_weights()
    try:
        conn = _conn()
        cur = conn.cursor()
        # Detect whether the table has the depth_level column
        cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'units_weights'
                    """)
        cols = {r[0] for r in cur.fetchall()}

        if "depth_level" in cols:
            cur.execute("SELECT name, depth_level, weight FROM units_weights")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            if rows:
                result = {}
                for name, depth_level, weight in rows:
                    result.setdefault(name.upper(), {})[int(depth_level)] = float(weight)
                # Merge defaults for any missing unit types
                for k, v in defaults.items():
                    if k not in result:
                        result[k] = v
                return result
        else:
            # Old flat schema (name, weight) — promote each flat weight to all depths
            cur.execute("SELECT name, weight FROM units_weights")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            if rows:
                result = {}
                for name, weight in rows:
                    result[name.upper()] = {d: float(weight) for d in range(1, 6)}
                for k, v in defaults.items():
                    if k not in result:
                        result[k] = v
                return result
    except Exception as e:
        print(f"[stage2_store] load_unit_weights failed, using defaults: {e}")
    return defaults


def set_unit_weight(name: str, depth_level: int, weight: float):
    """Set a single (name, depth_level) weight. Requires depth-aware schema."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO units_weights (name, depth_level, weight) VALUES (%s, %s, %s)
            ON CONFLICT (name, depth_level) DO UPDATE SET weight = EXCLUDED.weight""",
        (name.upper(), depth_level, weight),
    )
    conn.commit()
    cur.close()
    conn.close()


# ────────────────────────────────────────────────────────────────────
# Audit
# ────────────────────────────────────────────────────────────────────
def audit(thread_id: str, actor: str, action: str, data: dict = None):
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO audit_log (id, thread_id, actor, action, data, created_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (str(uuid.uuid4()), thread_id, actor, action, json.dumps(data or {}), _now()),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[stage2_store] audit failed: {e}")


# ────────────────────────────────────────────────────────────────────
# PR registry — idempotency
# ────────────────────────────────────────────────────────────────────
def get_pr_by_request_id(unique_request_id: str) -> dict:
    conn = _conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM pr_registry WHERE unique_request_id = %s", (unique_request_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def save_pr(unique_request_id: str, artifact_id: str, thread_id: str,
            repo: str, branch: str, pr_url: str, pr_number: int):
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO pr_registry (unique_request_id, artifact_id, thread_id, repo, branch, pr_url, pr_number, status, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s)
               ON CONFLICT (unique_request_id) DO NOTHING""",
        (unique_request_id, artifact_id, thread_id, repo, branch, pr_url, pr_number, _now()),
    )
    conn.commit()
    cur.close()
    conn.close()
