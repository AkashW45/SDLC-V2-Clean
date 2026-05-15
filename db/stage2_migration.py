"""
db/stage2_migration.py  (FINAL — depth-aware units_weights)

Creates Stage 2 tables: asp, artifacts, artifact_decisions, units_weights, pr_registry, audit_log.

KEY CHANGE: units_weights is now (name, depth_level, weight) — composite PK —
so the decide() engine can use depth-aware weighting.

Run once:  python db/stage2_migration.py
Idempotent — safe to re-run.
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DDL = """
-- ── Adaptive Scope Profile ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS asp (
    asp_id           UUID PRIMARY KEY,
    thread_id        TEXT,
    payload          JSONB NOT NULL,
    allow_unbounded  BOOLEAN NOT NULL DEFAULT FALSE,
    policy_mode      TEXT NOT NULL DEFAULT 'managed',
    depth_level      INT,
    created_by       TEXT,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT now(),
    version          INT DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_asp_thread ON asp(thread_id);

-- ── Artifacts ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id    UUID PRIMARY KEY,
    asp_id         UUID REFERENCES asp(asp_id),
    thread_id      TEXT,
    type           TEXT NOT NULL,
    category       TEXT NOT NULL DEFAULT 'mvp',
    payload        JSONB NOT NULL,
    units_estimate INT DEFAULT 0,
    confidence     INT DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'candidate',
    created_by     TEXT,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_asp ON artifacts(asp_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(thread_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);

-- ── Artifact decisions ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS artifact_decisions (
    decision_id   UUID PRIMARY KEY,
    artifact_id   UUID REFERENCES artifacts(artifact_id),
    thread_id     TEXT,
    decision      JSONB NOT NULL,
    decided_by    TEXT,
    decided_at    TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_decisions_artifact ON artifact_decisions(artifact_id);

-- ── Unit weights — DEPTH-AWARE (name, depth_level, weight) ─────────
CREATE TABLE IF NOT EXISTS units_weights (
    name         TEXT NOT NULL,
    depth_level  INT  NOT NULL,
    weight       FLOAT NOT NULL,
    PRIMARY KEY (name, depth_level)
);

-- ── Audit log ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          UUID PRIMARY KEY,
    thread_id   TEXT,
    actor       TEXT,
    action      TEXT,
    data        JSONB,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- ── PR registry — idempotent PR creation ───────────────────────────
CREATE TABLE IF NOT EXISTS pr_registry (
    unique_request_id  TEXT PRIMARY KEY,
    artifact_id        UUID,
    thread_id          TEXT,
    repo               TEXT,
    branch             TEXT,
    pr_url             TEXT,
    pr_number          INT,
    status             TEXT DEFAULT 'open',
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""

# Depth-aware default weights — (name, depth_level): weight
# At LOW depth each unit is heavier (fewer, simpler things expected).
# At HIGH depth each unit is lighter (many units are normal).
DEFAULT_WEIGHTS = {
    "FR":            {1: 2.0, 2: 1.6, 3: 1.2, 4: 1.0, 5: 0.8},
    "NFR":           {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8},
    "CODE_FILE":     {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.6},
    "ENDPOINT":      {1: 6.0, 2: 5.0, 3: 4.0, 4: 3.5, 5: 3.0},
    "DB_TABLE":      {1: 14.0, 2: 12.0, 3: 10.0, 4: 9.0, 5: 8.0},
    "MIGRATION":     {1: 10.0, 2: 9.0, 3: 8.0, 4: 7.0, 5: 6.0},
    "INTEGRATION":   {1: 25.0, 2: 22.0, 3: 20.0, 4: 18.0, 5: 16.0},
    "DEPENDENCY":    {1: 5.0, 2: 4.5, 3: 4.0, 4: 3.5, 5: 3.0},
    "TEST_CASE":     {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3},
    "INFRA_SERVICE": {1: 18.0, 2: 16.0, 3: 15.0, 4: 13.0, 5: 12.0},
    "NODE":          {1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0},
    "JIRA":          {1: 1.5, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.9},
    "ADR":           {1: 2.0, 2: 1.8, 3: 1.5, 4: 1.3, 5: 1.2},
    "SPRINT":        {1: 8.0, 2: 7.0, 3: 6.0, 4: 5.5, 5: 5.0},
}


def _conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
    )


def migrate():
    conn = _conn()
    cur = conn.cursor()
    print("[Migration] Creating Stage 2 tables...")
    cur.execute(DDL)

    print("[Migration] Inserting depth-aware default unit weights...")
    for name, depth_map in DEFAULT_WEIGHTS.items():
        for depth_level, weight in depth_map.items():
            cur.execute(
                """INSERT INTO units_weights (name, depth_level, weight)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (name, depth_level) DO NOTHING""",
                (name, depth_level, weight),
            )

    conn.commit()
    cur.close()
    conn.close()
    print("[Migration] Stage 2 schema ready (depth-aware units_weights).")


if __name__ == "__main__":
    migrate()
