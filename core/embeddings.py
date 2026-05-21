"""
core/embeddings.py — process-wide shared sentence-transformer embedder.

WHY THIS EXISTS
---------------
Before this module, `SentenceTransformer("all-MiniLM-L6-v2")` was instantiated
in eight different places (phase0 selector, phase3 impact analyzer, evidence
resolver, repo summary, project registry, indexer, context engine, setup).
Several were at module top-level, so importing them built a *separate* ~90 MB
model copy each time — wasted memory and, worse, the model-load + first-call
cost landed on whichever request triggered it first (typically the dashboard's
`/pipeline/preview-routing` call, making it appear to "hang" for seconds).

This module collapses all of that into ONE lazily-built, thread-safe singleton.
Every caller now shares the same in-memory model. Combined with the FastAPI
startup warmup (see api/main.py) and Docker build-time model caching (see
Dockerfile), the per-request encode cost drops to milliseconds.

USAGE
-----
    from core.embeddings import get_embedder, encode

    vec = encode("some text")          # -> list[float], length 384
    model = get_embedder()             # the shared SentenceTransformer
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

# The single model name used everywhere in the project. Centralised so a model
# swap is a one-line change instead of a grep-and-replace across eight files.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder = None
_lock = threading.Lock()


def get_embedder():
    """Return the process-wide SentenceTransformer, building it once.

    Thread-safe: uses double-checked locking so concurrent first-callers
    (e.g. two requests racing at startup) don't each build their own model.
    """
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:  # re-check inside the lock
                from sentence_transformers import SentenceTransformer
                logger.info(f"[embeddings] loading model '{MODEL_NAME}' (one-time)...")
                _embedder = SentenceTransformer(MODEL_NAME)
                logger.info("[embeddings] model loaded and cached process-wide.")
    return _embedder


def encode(text, **kwargs):
    """Encode text to a plain Python list (JSON-serialisable).

    Mirrors the `.encode(...).tolist()` pattern the call sites already used,
    so they can switch to `encode(text)` with identical behaviour.
    """
    return get_embedder().encode(text, **kwargs).tolist()


def warmup() -> bool:
    """Force the model to load now (paying download + init cost up front).

    Call this from a FastAPI startup hook so the cost is paid at boot rather
    than on the first user request. Returns True on success, False if the
    model could not be loaded (logged, never raised — warmup must not crash
    server startup).
    """
    try:
        get_embedder().encode("warmup")
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[embeddings] warmup failed (will lazy-load later): {e}")
        return False


def is_loaded() -> bool:
    """True if the model has already been built (useful for health checks)."""
    return _embedder is not None
