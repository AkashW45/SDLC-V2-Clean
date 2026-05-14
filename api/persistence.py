import psycopg2
from psycopg_pool import ConnectionPool
import json
from datetime import datetime
import os

def _conn():
    # Get the host (Docker compose sets this to "sdlc_postgres" for the API container)
    db_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    
    # If we are inside Docker, use the internal port 5432. Otherwise, use 5437.
    db_port = "5432" if db_host == "sdlc_postgres" else os.getenv("POSTGRES_PORT", "5437")
    
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres_password"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )

def setup_db():
    conn = _conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pipelines (
            thread_id VARCHAR(50) PRIMARY KEY,
            requirement TEXT NOT NULL,
            status VARCHAR(50),
            phase VARCHAR(20),
            sub_stage VARCHAR(150),
            current_state JSONB,
            pr_urls JSONB,
            error TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            thread_id VARCHAR(50),
            phase VARCHAR(20),
            event VARCHAR(100),
            actor VARCHAR(100),
            details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS artifact_registry (
            artifact_id SERIAL PRIMARY KEY,
            thread_id VARCHAR(255),
            "key" VARCHAR(255),
            version INT,
            status VARCHAR(50),
            producing_phase VARCHAR(50),
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(thread_id, "key", version)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS replay_jobs (
            job_id SERIAL PRIMARY KEY,
            thread_id VARCHAR(255),
            target_artifact VARCHAR(255),
            status VARCHAR(50),
            diff_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_thread ON audit_log(thread_id, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pipelines_updated ON pipelines(updated_at DESC)")

    conn.commit()
    cur.close()
    conn.close()
    print("[Persistence] Tables ready")


def init_persistence_tables():
    return setup_db()


def enforce_audit_retention() -> int:
    """
    Enforce 90-day retention policy for audit logs.
    Note: Because we revoked DELETE from PUBLIC, this specific cleanup job must run under a privileged admin role via a cron job, not the standard app role.
    """
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM audit_log WHERE created_at < NOW() - INTERVAL '90 days';
    """)
    deleted_count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return deleted_count


def save_pipeline(thread_id: str, entry: dict, safe_state: dict):
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO pipelines
              (thread_id, requirement, status, phase, sub_stage, current_state, pr_urls, error, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (thread_id) DO UPDATE SET
              status = EXCLUDED.status,
              phase = EXCLUDED.phase,
              sub_stage = EXCLUDED.sub_stage,
              current_state = EXCLUDED.current_state,
              pr_urls = EXCLUDED.pr_urls,
              error = EXCLUDED.error,
              updated_at = NOW()
        """, (
            thread_id,
            entry.get("requirement", ""),
            entry.get("status", ""),
            entry.get("phase", ""),
            entry.get("sub_stage", ""),
            json.dumps(safe_state),
            json.dumps(entry.get("pr_urls", [])),
            entry.get("error") or ""
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Persistence] save_pipeline failed: {e}")


def load_all_pipelines() -> dict:
    out = {}
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT thread_id, requirement, status, phase, sub_stage,
                   current_state, pr_urls, error, updated_at
            FROM pipelines
            ORDER BY updated_at DESC
            LIMIT 50
        """)
        for row in cur.fetchall():
            tid = row[0]
            out[tid] = {
                "thread_id": tid,
                "requirement": row[1],
                "status": row[2] or "",
                "phase": row[3] or "",
                "sub_stage": row[4] or "",
                "current_state": row[5] or {},
                "pr_urls": row[6] or [],
                "error": row[7] or "",
                "updated_at": row[8].isoformat() if row[8] else ""
            }
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Persistence] load_all_pipelines failed: {e}")
    return out


def audit(thread_id: str, phase: str, event: str, actor: str = "system", details: dict = None):
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log (thread_id, phase, event, actor, details)
            VALUES (%s, %s, %s, %s, %s)
        """, (thread_id, phase, event, actor, json.dumps(details or {})))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Audit] failed: {e}")


def get_audit_log(thread_id: str) -> list:
    out = []
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT phase, event, actor, details, created_at
            FROM audit_log
            WHERE thread_id = %s
            ORDER BY created_at ASC
        """, (thread_id,))
        for row in cur.fetchall():
            out.append({
                "phase": row[0],
                "event": row[1],
                "actor": row[2],
                "details": row[3] or {},
                "at": row[4].isoformat() if row[4] else ""
            })
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Audit] read failed: {e}")
    return out


def save_artifact(thread_id: str, key: str, phase: str, content: str) -> int:
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) FROM artifact_registry WHERE thread_id = %s AND \"key\" = %s",
            (thread_id, key)
        )
        max_version = cur.fetchone()[0] or 0
        new_version = max_version + 1
        cur.execute(
            "INSERT INTO artifact_registry (thread_id, \"key\", version, status, producing_phase, content)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (thread_id, key, new_version, "ACTIVE", phase, content)
        )
        conn.commit()
        cur.close()
        conn.close()
        return new_version
    except Exception as e:
        print(f"[Persistence] save_artifact failed: {e}")
        return 0


def create_replay_job(thread_id: str, target_artifact: str) -> int:
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO replay_jobs (thread_id, target_artifact, status, diff_summary) VALUES (%s, %s, %s, %s) RETURNING job_id",
            (thread_id, target_artifact, "PENDING", "")
        )
        job_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return job_id
    except Exception as e:
        print(f"[Persistence] create_replay_job failed: {e}")
        return 0


def update_replay_job(job_id: int, status: str, diff_summary: str):
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE replay_jobs SET status = %s, diff_summary = %s WHERE job_id = %s",
            (status, diff_summary, job_id)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Persistence] update_replay_job failed: {e}")


def get_artifact(thread_id: str, key: str, version: int = None) -> dict:
    out = {}
    try:
        conn = _conn()
        cur = conn.cursor()
        if version is None:
            cur.execute(
                "SELECT thread_id, \"key\", version, status, producing_phase, content, created_at"
                " FROM artifact_registry"
                " WHERE thread_id = %s AND \"key\" = %s"
                " ORDER BY version DESC LIMIT 1",
                (thread_id, key)
            )
        else:
            cur.execute(
                "SELECT thread_id, \"key\", version, status, producing_phase, content, created_at"
                " FROM artifact_registry"
                " WHERE thread_id = %s AND \"key\" = %s AND version = %s",
                (thread_id, key, version)
            )
        row = cur.fetchone()
        if row:
            out = {
                "thread_id": row[0],
                "key": row[1],
                "version": row[2],
                "status": row[3],
                "producing_phase": row[4],
                "content": row[5],
                "created_at": row[6].isoformat() if row[6] else ""
            }
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Persistence] get_artifact failed: {e}")
    return out