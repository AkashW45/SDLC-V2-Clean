"""
knowledge-layer/repo_summary.py
Produces a repo_summary JSON object the Intent Analyzer (Phase 0) embeds in the ASP.

repo_summary tells the classifier:
  - does a matching repo already exist?
  - what languages?
  - how much symbol overlap with the requirement?
  - are tests present?
  - what are the top symbols?

build_repo_summary(requirement, candidate_repo_name=None) -> dict
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Lazy singletons
_embedder = None
_qdrant = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(url="http://127.0.0.1:6333", timeout=10)
    return _qdrant


def _pg_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
    )


def _empty_summary() -> dict:
    return {
        "exists": False,
        "matched_repo": None,
        "match_score": 0.0,
        "languages": [],
        "symbol_overlap": 0.0,
        "tests_present": False,
        "top_symbols": [],
    }


def build_repo_summary(requirement: str, candidate_repo_name: str = None) -> dict:
    """
    Build a repo_summary for the ASP.

    If candidate_repo_name is given, summarize that specific repo.
    Otherwise, search project_embeddings for the best semantic match.
    """
    try:
        # ── Step 1: find the matching repo ──
        if candidate_repo_name:
            matched_repo = candidate_repo_name
            match_score = 1.0
        else:
            matched_repo, match_score = _find_best_repo_match(requirement)

        if not matched_repo:
            return _empty_summary()

        # ── Step 2: gather repo metadata from repo_maps ──
        languages, file_count, last_indexed = _get_repo_metadata(matched_repo)
        if file_count == 0:
            # Repo known to registry but never indexed
            return {
                **_empty_summary(),
                "exists": True,
                "matched_repo": matched_repo,
                "match_score": round(match_score, 2),
                "note": "Repo registered but not indexed",
            }

        # ── Step 3: top symbols from symbols table ──
        top_symbols = _get_top_symbols(matched_repo)

        # ── Step 4: symbol overlap with requirement ──
        symbol_overlap = _compute_symbol_overlap(requirement, top_symbols)

        # ── Step 5: tests present? ──
        tests_present = _detect_tests(matched_repo)

        return {
            "exists": True,
            "matched_repo": matched_repo,
            "match_score": round(match_score, 2),
            "languages": languages,
            "file_count": file_count,
            "symbol_overlap": round(symbol_overlap, 2),
            "tests_present": tests_present,
            "top_symbols": top_symbols[:15],
            "last_indexed": str(last_indexed) if last_indexed else None,
        }

    except Exception as e:
        print(f"[repo_summary] build failed: {e}")
        return _empty_summary()


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _find_best_repo_match(requirement: str):
    """Search project_embeddings for the best semantic match. Returns (repo_name, score)."""
    try:
        q = _get_qdrant()
        vec = _get_embedder().encode(requirement).tolist()
        hits = q.search(
            collection_name="project_embeddings",
            query_vector=vec,
            limit=1,
            with_payload=True,
        )
        if hits:
            top = hits[0]
            payload = top.payload or {}
            # project_embeddings payload has 'repos' list — take the first repo
            repos = payload.get("repos", [])
            if repos:
                first = repos[0]
                repo_name = first if isinstance(first, str) else first.get("name", "")
                return repo_name, top.score
        return None, 0.0
    except Exception as e:
        print(f"[repo_summary] repo match search failed: {e}")
        return None, 0.0


def _get_repo_metadata(repo_name: str):
    """Returns (languages, file_count, last_indexed)."""
    try:
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT language, file_count, last_indexed FROM repo_maps WHERE repo_name = %s",
            (repo_name,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            lang = row[0]
            languages = [lang] if isinstance(lang, str) else (lang or [])
            return languages, row[1] or 0, row[2]
        return [], 0, None
    except Exception as e:
        print(f"[repo_summary] metadata fetch failed: {e}")
        return [], 0, None


def _get_top_symbols(repo_name: str):
    """Top symbols by frequency / importance from the symbols table."""
    try:
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT symbol_name FROM symbols
               WHERE repo_name = %s
               ORDER BY symbol_name
               LIMIT 50""",
            (repo_name,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[repo_summary] symbol fetch failed: {e}")
        return []


def _compute_symbol_overlap(requirement: str, symbols: list) -> float:
    """
    Fraction of repo symbols whose names appear (loosely) in the requirement.
    High overlap → the requirement is probably about this repo.
    """
    if not symbols:
        return 0.0
    import re
    req_words = set(re.findall(r"[a-z]{3,}", requirement.lower()))
    if not req_words:
        return 0.0

    matches = 0
    for sym in symbols:
        # Split camelCase / snake_case symbol into words
        sym_words = set(re.findall(r"[a-z]{3,}", re.sub(r"([A-Z])", r" \1", sym).lower()))
        if sym_words & req_words:
            matches += 1
    return matches / len(symbols)


def _detect_tests(repo_name: str) -> bool:
    """Check if the repo has test files indexed."""
    try:
        conn = _pg_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT COUNT(*) FROM symbols
               WHERE repo_name = %s
               AND (file_path ILIKE %s OR file_path ILIKE %s OR symbol_name ILIKE %s)""",
            (repo_name, "%test%", "%spec%", "test_%"),
        )
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception as e:
        print(f"[repo_summary] test detection failed: {e}")
        return False