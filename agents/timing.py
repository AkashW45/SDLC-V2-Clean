"""
Lightweight per-phase / per-step timing instrumentation.
Records to Postgres and the in-memory pipeline_store so dashboard can render it.
"""
import time
from contextlib import contextmanager
from typing import Optional
import os

# Lazy import to avoid circular dep
_db_writer = None

def _get_writer():
    global _db_writer
    if _db_writer is None:
        try:
            from agents.persistence import save_timing_event
            _db_writer = save_timing_event
        except Exception:
            _db_writer = lambda *a, **k: None  # no-op fallback
    return _db_writer


@contextmanager
def timed(thread_id: str, phase: str, step: str = None, store: dict = None):
    """
    Context manager that records elapsed time for a phase or step.

    Usage:
        with timed(thread_id, "phase_4", "llm_call") as t:
            response = call_llm(prompt)
        # t.elapsed_ms is now available
    """
    label = f"{phase}.{step}" if step else phase
    start = time.perf_counter()

    class Timer:
        elapsed_ms = 0
        elapsed_s = 0.0

    t = Timer()
    try:
        yield t
    finally:
        elapsed_s = time.perf_counter() - start
        t.elapsed_s = elapsed_s
        t.elapsed_ms = int(elapsed_s * 1000)
        print(f"[timing] {thread_id} {label}: {t.elapsed_ms} ms ({elapsed_s:.2f}s)")

        # Persist to pipeline_store so dashboard sees it
        if store is not None and thread_id in store:
            timings = store[thread_id].setdefault("timings", {})
            timings.setdefault(phase, {})
            if step:
                timings[phase][step] = t.elapsed_ms
            else:
                timings[phase]["_total"] = t.elapsed_ms

        # Persist to DB
        try:
            _get_writer()(thread_id, phase, step, t.elapsed_ms)
        except Exception as e:
            print(f"[timing] DB write failed (non-fatal): {e}")


def phase_summary(timings: dict) -> str:
    """Format a human-readable summary of phase timings."""
    if not timings:
        return "No timings recorded."
    lines = ["Phase Timing Summary:"]
    grand_total = 0
    for phase in sorted(timings.keys()):
        steps = timings[phase]
        total = steps.get("_total", sum(v for k, v in steps.items() if k != "_total"))
        grand_total += total
        lines.append(f"  {phase}: {total/1000:.2f}s")
        for step, ms in sorted(steps.items()):
            if step == "_total":
                continue
            lines.append(f"    └─ {step}: {ms/1000:.2f}s")
    lines.append(f"  TOTAL: {grand_total/1000:.2f}s")
    return "\n".join(lines)