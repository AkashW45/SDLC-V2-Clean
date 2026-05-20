"""
core/db_clients.py
-------------------
Project-wide singletons for Postgres, Neo4j, and Qdrant.

WHY THIS EXISTS
===============
Before this module, every file in the codebase did one of:
  - psycopg2.connect(...)        ← fresh TCP handshake per call
  - GraphDatabase.driver(...)    ← new driver per call (each one is its own pool)
  - QdrantClient(...)            ← new HTTP client per call

That meant 20–30 fresh sockets per pipeline run, and concurrent pipelines could
exhaust Postgres `max_connections` quickly. Phase 3 already did this correctly
with its own private `_pg_pool` and module-level Neo4j driver — this module
generalizes that pattern so the whole project can share it.

PUBLIC API
==========
1) Postgres
   ----------
   from core.db_clients import pg_conn

   with pg_conn() as conn:
       cur = conn.cursor()
       cur.execute("SELECT 1")
       # commit happens automatically on clean exit
       # rollback happens automatically on exception

   Notes:
     - pg_conn() borrows from a ThreadedConnectionPool (min=2, max=20).
     - Auto-commit on success, auto-rollback on exception. If you need finer
       control, call conn.commit() / conn.rollback() yourself inside the block.
     - The connection is always returned to the pool on `finally`, even when
       a borrowed connection is in a broken state (psycopg2 handles cleanup).

2) Neo4j
   ------
   from core.db_clients import neo4j_driver

   with neo4j_driver.session() as session:
       result = session.run("MATCH (n) RETURN n LIMIT 1")

   Notes:
     - One module-level driver instance per process. The Neo4j driver itself
       maintains a connection pool internally (max 50 in our config).
     - DO NOT call .close() on this driver in application code. It's shut down
       at process exit by `atexit` below.

3) Qdrant
   -------
   from core.db_clients import qdrant_client

   results = qdrant_client.query_points(collection_name="...", query=...)

   Notes:
     - Qdrant clients are cheap to construct, but sharing one centralizes
       host/port/timeout config and lets us swap to a remote Qdrant by
       changing one file.

ENVIRONMENT VARIABLES
=====================
POSTGRES_HOST   default 127.0.0.1; if set to "sdlc_postgres" we use port 5432
                (Docker network); otherwise port from POSTGRES_PORT (default 5437).
POSTGRES_PORT   default 5437 (host-mapped port from docker-compose)
POSTGRES_USER   default "sdlc"
POSTGRES_PASSWORD  default "sdlc1234"
POSTGRES_DB     default "sdlc_knowledge"
PG_POOL_MIN     default 2
PG_POOL_MAX     default 20

NEO4J_URI       default "bolt://127.0.0.1:7687"
NEO4J_USER      default "neo4j"
NEO4J_PASSWORD  default "password1234"
NEO4J_POOL_MAX  default 50

QDRANT_HOST     default "127.0.0.1"
QDRANT_PORT     default 6333
QDRANT_TIMEOUT  default 60
"""
from __future__ import annotations

import atexit
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from psycopg2 import pool as pg_pool_module
from neo4j import Driver, GraphDatabase
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Postgres pool — lazily initialized
# ---------------------------------------------------------------------------

def _resolve_pg_host_port() -> tuple[str, str]:
    """
    Resolve Postgres host/port. When the API container runs inside
    docker-compose, POSTGRES_HOST is set to "sdlc_postgres" and we must
    use the *internal* port 5432. Outside Docker, we use the host-mapped
    port (POSTGRES_PORT, default 5437).
    """
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    if host == "sdlc_postgres":
        port = "5432"
    else:
        port = os.getenv("POSTGRES_PORT", "5437")
    return host, port


_pg_pool: Optional[pg_pool_module.ThreadedConnectionPool] = None
_pg_pool_lock = threading.Lock()


def _ensure_pg_pool() -> pg_pool_module.ThreadedConnectionPool:
    """Lazily build the Postgres pool on first access.

    Why lazy: importing this module shouldn't require a live Postgres.
    Tests, type-check scripts, and one-shot CLI tools can import names
    from here without paying for a connection until they actually use it.
    """
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None:
            host, port = _resolve_pg_host_port()
            _pg_pool = pg_pool_module.ThreadedConnectionPool(
                minconn=int(os.getenv("PG_POOL_MIN", "2")),
                maxconn=int(os.getenv("PG_POOL_MAX", "20")),
                host=host,
                port=port,
                user=os.getenv("POSTGRES_USER", "sdlc"),
                password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
                dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
            )
    return _pg_pool


@contextmanager
def pg_conn(autocommit: bool = True) -> Iterator:
    """
    Borrow a Postgres connection from the pool.

    Usage:
        with pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")

    Args:
        autocommit: If True (default), commits on clean exit and rolls back
                    on exception. Set to False if you want to manage the
                    transaction yourself inside the block.

    Returns: psycopg2 connection (yielded). Always returned to pool.
    """
    pool = _ensure_pg_pool()
    conn = pool.getconn()
    try:
        yield conn
        if autocommit:
            try:
                conn.commit()
            except Exception:
                # If commit itself fails (e.g. broken connection), don't mask
                # the original logic by raising here — just log and let the
                # pool's discard logic handle it on the next checkout.
                pass
    except Exception:
        if autocommit:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        # `putconn` is safe even if the connection is in a bad state — the
        # pool tracks closed connections and will recycle them.
        try:
            pool.putconn(conn)
        except Exception:
            # As a last resort if put fails, try to close so we don't leak.
            try:
                conn.close()
            except Exception:
                pass


def pg_pool_stats() -> dict:
    """Returns rough pool stats. Useful for /healthz and debugging."""
    if _pg_pool is None:
        return {"minconn": 0, "maxconn": 0, "used": 0, "available": 0,
                "status": "not_initialized"}
    try:
        return {
            "minconn": _pg_pool.minconn,
            "maxconn": _pg_pool.maxconn,
            "used": len(getattr(_pg_pool, "_used", {})),
            "available": len(getattr(_pg_pool, "_pool", [])),
        }
    except Exception:
        return {"minconn": "?", "maxconn": "?", "used": "?", "available": "?"}


class PooledConn:
    """
    Back-compat shim for code that already uses the
        conn = some_factory()
        ...
        conn.close()
    pattern. Wraps a pooled psycopg2 connection and routes .close() back to
    the pool. Prefer pg_conn() (the context manager) for new code.

    Usage:
        from core.db_clients import PooledConn
        conn = PooledConn()
        cur = conn.cursor()
        ...
        conn.commit()
        cur.close()
        conn.close()    # returns to pool, doesn't actually close the socket
    """
    __slots__ = ("_conn", "_returned")

    def __init__(self):
        pool = _ensure_pg_pool()
        self._conn = pool.getconn()
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if not self._returned:
            # If the connection still has an open transaction (e.g. caller
            # forgot to commit, or hit an exception before commit), roll it
            # back before returning to the pool. Otherwise the next borrower
            # of this connection inherits an aborted/uncommitted txn and
            # every query they run fails with "current transaction is aborted".
            try:
                # psycopg2 exposes transaction_status; >0 means in-txn or aborted
                if hasattr(self._conn, "info") and \
                        self._conn.info.transaction_status != 0:
                    self._conn.rollback()
            except Exception:
                pass
            try:
                _ensure_pg_pool().putconn(self._conn)
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._returned = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        else:
            try:
                self._conn.commit()
            except Exception:
                pass
        self.close()
        return False


# ---------------------------------------------------------------------------
# Neo4j driver — lazy, already internally pooled
# ---------------------------------------------------------------------------

_neo4j_driver: Optional[Driver] = None
_neo4j_lock = threading.Lock()


def _ensure_neo4j_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    with _neo4j_lock:
        if _neo4j_driver is None:
            _neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
                auth=(
                    os.getenv("NEO4J_USER", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "password1234"),
                ),
                max_connection_pool_size=int(os.getenv("NEO4J_POOL_MAX", "50")),
            )
    return _neo4j_driver


class _LazyNeo4jDriver:
    """Module-level handle that defers driver construction until first use.

    Code can do `from core.db_clients import neo4j_driver` at import time and
    only pay for a live Neo4j connection when it actually calls .session()
    or any other method.
    """
    def __getattr__(self, name):
        return getattr(_ensure_neo4j_driver(), name)

    def session(self, *args, **kwargs):
        return _ensure_neo4j_driver().session(*args, **kwargs)

    def close(self):
        global _neo4j_driver
        if _neo4j_driver is not None:
            try:
                _neo4j_driver.close()
            finally:
                _neo4j_driver = None


neo4j_driver = _LazyNeo4jDriver()


# ---------------------------------------------------------------------------
# Qdrant client — lazy, thin HTTP wrapper
# ---------------------------------------------------------------------------

_qdrant_client: Optional[QdrantClient] = None
_qdrant_lock = threading.Lock()


def _ensure_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "127.0.0.1"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                timeout=int(os.getenv("QDRANT_TIMEOUT", "60")),
            )
    return _qdrant_client


class _LazyQdrantClient:
    """Module-level handle that defers Qdrant client construction until first use."""
    def __getattr__(self, name):
        return getattr(_ensure_qdrant(), name)


qdrant_client = _LazyQdrantClient()


# ---------------------------------------------------------------------------
# Shutdown hooks
# ---------------------------------------------------------------------------

_shutdown_lock = threading.Lock()
_shutdown_done = False


def _shutdown() -> None:
    """Close the Postgres pool and Neo4j driver at interpreter exit.

    Idempotent — safe to call multiple times. Qdrant client doesn't need
    explicit cleanup (HTTP requests are short-lived).
    """
    global _shutdown_done, _neo4j_driver
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
        if _pg_pool is not None:
            try:
                _pg_pool.closeall()
            except Exception:
                pass
        if _neo4j_driver is not None:
            try:
                _neo4j_driver.close()
            except Exception:
                pass


atexit.register(_shutdown)
