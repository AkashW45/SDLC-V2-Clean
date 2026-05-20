"""
Production-grade indexer queue with parallelism control and retries.

Why this exists: when you sync hundreds of repos, doing it inline blocks the
API for hours. Doing it in raw background tasks gives no retry, no visibility,
no concurrency control. This module is a small but real worker pool.

Job lifecycle:
    QUEUED → RUNNING → SUCCESS | FAILED | RETRYING

State is held in-memory (job_store dict) and mirrored to PostgreSQL so it
survives server restarts. Pending jobs auto-resume on startup.
"""

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import psycopg2
import os

logger = logging.getLogger(__name__)

# Worker pool size — tune based on CPU + IO available.
# Indexing is IO-heavy (git clone, postgres writes, embedding model) so >cores is fine.
MAX_WORKERS = int(os.getenv("INDEXER_WORKERS", "4"))
MAX_RETRIES = int(os.getenv("INDEXER_MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE = 5  # seconds; 5, 10, 20...


@dataclass
class IndexJob:
    job_id: str
    repo_name: str
    repo_url: str
    branch: str = "main"
    status: str = "QUEUED"         # QUEUED | RUNNING | SUCCESS | FAILED | RETRYING
    attempts: int = 0
    last_error: str = ""
    force: bool = False           # NEW
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "branch": self.branch,
            "status": self.status,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": (self.finished_at - self.started_at)
            if self.started_at and self.finished_at else None,
        }


class IndexerQueue:
    def __init__(self, max_workers: int = MAX_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="indexer")
        self._jobs: dict[str, IndexJob] = {}
        self._lock = threading.Lock()
        self._ensure_table()
        self._resume_pending_on_startup()

    # ── persistence ─────────────────────────────────────────────────
    def _conn(self):
        from core.db_clients import PooledConn
        return PooledConn()

    def _ensure_table(self):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute("""
                        CREATE TABLE IF NOT EXISTS indexer_jobs (
                                                                    job_id VARCHAR(64) PRIMARY KEY,
                            repo_name VARCHAR(255) NOT NULL,
                            repo_url TEXT,
                            branch VARCHAR(100),
                            status VARCHAR(20),
                            attempts INTEGER DEFAULT 0,
                            last_error TEXT,
                            created_at DOUBLE PRECISION,
                            started_at DOUBLE PRECISION,
                            finished_at DOUBLE PRECISION
                            );
                        """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_indexer_jobs_status ON indexer_jobs(status);")
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"[IndexerQueue] table init failed: {e}")

    def _persist(self, job: IndexJob):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute("""
                        INSERT INTO indexer_jobs (job_id, repo_name, repo_url, branch, status,
                                                  attempts, last_error, created_at, started_at, finished_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (job_id) DO UPDATE
                                                        SET status = EXCLUDED.status,
                                                        attempts = EXCLUDED.attempts,
                                                        last_error = EXCLUDED.last_error,
                                                        started_at = EXCLUDED.started_at,
                                                        finished_at = EXCLUDED.finished_at
                        """, (job.job_id, job.repo_name, job.repo_url, job.branch, job.status,
                              job.attempts, job.last_error[:2000], job.created_at,
                              job.started_at, job.finished_at))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"[IndexerQueue] persist failed: {e}")

    def _resume_pending_on_startup(self):
        try:
            conn = self._conn()
            cur = conn.cursor()
            cur.execute("""
                        SELECT job_id, repo_name, repo_url, branch, status, attempts, last_error,
                               created_at, started_at, finished_at
                        FROM indexer_jobs
                        WHERE status IN ('QUEUED', 'RUNNING', 'RETRYING')
                        """)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            for r in rows:
                job = IndexJob(job_id=r[0], repo_name=r[1], repo_url=r[2] or "",
                               branch=r[3] or "main", status="QUEUED",
                               attempts=r[5] or 0, last_error=r[6] or "",
                               created_at=r[7] or time.time())
                self._jobs[job.job_id] = job
                self._executor.submit(self._run, job)
            if rows:
                logger.info(f"[IndexerQueue] resumed {len(rows)} pending job(s) on startup")
        except Exception as e:
            logger.warning(f"[IndexerQueue] resume skipped: {e}")

    # ── public API ──────────────────────────────────────────────────
    def enqueue(self, repo_name: str, repo_url: str, branch: str = "main",force: bool = False) -> str:
        """Schedule a repo for cloning + indexing.  force=True ignores SHA cache."""
        job_id = f"idx-{uuid.uuid4().hex[:12]}"
        job = IndexJob(job_id=job_id, repo_name=repo_name,
                       repo_url=repo_url, branch=branch)
        # Stash force flag on the job
        job.force = force
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        self._executor.submit(self._run, job)
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self, status: Optional[str] = None) -> list:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return [j.to_dict() for j in sorted(jobs, key=lambda x: -x.created_at)]

    def summary(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        counts = {"QUEUED": 0, "RUNNING": 0, "SUCCESS": 0, "FAILED": 0, "RETRYING": 0}
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        return {"total": len(jobs), "by_status": counts}

    # ── worker ──────────────────────────────────────────────────────
    def _run(self, job: IndexJob):
        from agents.repo_workspace import ensure_repo_cloned
        from knowledge_layer.indexer import index_repo

        while job.attempts < MAX_RETRIES:
            job.attempts += 1
            job.status = "RUNNING"
            job.started_at = time.time()
            self._persist(job)
            logger.info(f"[IndexerQueue] {job.job_id} {job.repo_name} attempt {job.attempts}")

            try:
                local = ensure_repo_cloned(job.repo_name,
                                           repo_url=job.repo_url,
                                           branch=job.branch)
                if not local:
                    raise RuntimeError(f"clone failed for {job.repo_name}")

                result = index_repo(local, job.repo_name, force=job.force)

                job.status = "SUCCESS"
                # Surface skip info in last_error field for visibility
                if result and result.get("status") == "SKIPPED_NO_CHANGES":
                    job.last_error = f"SKIPPED — no changes since last index (sha={result['sha'][:8]})"
                else:
                    job.last_error = ""
                job.finished_at = time.time()
                job.last_error = ""
                self._persist(job)
                logger.info(f"[IndexerQueue] {job.job_id} SUCCESS in "
                            f"{job.finished_at - job.started_at:.1f}s")
                return
            except Exception as e:
                tb = traceback.format_exc()
                job.last_error = f"{type(e).__name__}: {e}\n{tb[-500:]}"
                logger.warning(f"[IndexerQueue] {job.job_id} attempt {job.attempts} failed: {e}")
                if job.attempts < MAX_RETRIES:
                    job.status = "RETRYING"
                    self._persist(job)
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** (job.attempts - 1)))
                else:
                    job.status = "FAILED"
                    job.finished_at = time.time()
                    self._persist(job)
                    logger.error(f"[IndexerQueue] {job.job_id} FAILED after {MAX_RETRIES} attempts")


# Singleton — initialized lazily so tests / scripts don't trigger startup
_queue: Optional[IndexerQueue] = None
_init_lock = threading.Lock()


def get_queue() -> IndexerQueue:
    global _queue
    if _queue is None:
        with _init_lock:
            if _queue is None:
                _queue = IndexerQueue()
    return _queue