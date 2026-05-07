import psycopg2
import json
from datetime import datetime
import os

def _conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )


def init_persistence_tables():
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
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_thread ON audit_log(thread_id, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pipelines_updated ON pipelines(updated_at DESC)")

    conn.commit()
    cur.close()
    conn.close()
    print("[Persistence] Tables ready")


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