"""
agents/evidence_resolver.py
Validates that an artifact's evidence_links actually resolve to real sources.

Evidence link types:
  - "requirement": exact quoted phrase from user_input — resolved if phrase appears in ASP
  - "doc":         knowledge-layer doc id (qdrant point id) — resolved via Qdrant lookup
  - "commit":      repo + sha + path + lines — resolved via GitHub API
  - "symbol":      a symbol name — resolved via Neo4j / Postgres symbols table

resolve_evidence(artifact, asp) -> artifact with each evidence_link annotated:
    evidence_link["resolved"] = bool
    evidence_link["evidence_score"] = float 0..1
    evidence_link["resolved_snippet"] = str

EVIDENCE_THRESHOLD default 0.7 — below this, the link is considered unresolved.
"""
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

EVIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_THRESHOLD", "0.7"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Lazy-loaded singletons
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


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ────────────────────────────────────────────────────────────────────
# Per-type resolvers
# ────────────────────────────────────────────────────────────────────
def _resolve_requirement(link: dict, asp: dict) -> dict:
    """Requirement-type evidence: the quoted snippet must appear in user_input or anchors."""
    snippet = (link.get("snippet") or "").lower().strip()
    if not snippet:
        link["resolved"] = False
        link["evidence_score"] = 0.0
        link["resolved_snippet"] = ""
        return link

    haystack = asp.get("user_input", "").lower()
    anchors = asp.get("anchors", {})
    for cap in anchors.get("core_capabilities", []):
        haystack += " " + str(cap).lower()

    # Exact substring → score 1.0
    if snippet in haystack:
        link["resolved"] = True
        link["evidence_score"] = 1.0
        link["resolved_snippet"] = snippet
        return link

    # Fuzzy — word overlap ratio
    snippet_words = set(re.findall(r"[a-z]{3,}", snippet))
    hay_words = set(re.findall(r"[a-z]{3,}", haystack))
    if snippet_words:
        overlap = len(snippet_words & hay_words) / len(snippet_words)
    else:
        overlap = 0.0

    link["evidence_score"] = round(overlap, 2)
    link["resolved"] = overlap >= EVIDENCE_THRESHOLD
    link["resolved_snippet"] = snippet if link["resolved"] else ""
    return link


def _resolve_doc(link: dict, asp: dict) -> dict:
    """Doc-type evidence: doc_id must exist in Qdrant; compute similarity to requirement."""
    doc_id = link.get("doc_id")
    if not doc_id:
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link

    try:
        q = _get_qdrant()
        # Search all known collections for the point
        collections = ["code_embeddings", "project_embeddings",
                       "contract_embeddings", "repo_map_embeddings"]
        found = None
        for coll in collections:
            try:
                points = q.retrieve(collection_name=coll, ids=[doc_id], with_payload=True, with_vectors=True)
                if points:
                    found = points[0]
                    break
            except Exception:
                continue

        if not found:
            link["resolved"] = False
            link["evidence_score"] = 0.0
            link["resolved_snippet"] = ""
            return link

        # Similarity between doc vector and requirement embedding
        req_vec = _get_embedder().encode(asp.get("user_input", "")).tolist()
        doc_vec = found.vector
        score = _cosine(req_vec, doc_vec)

        link["evidence_score"] = round(score, 2)
        link["resolved"] = score >= EVIDENCE_THRESHOLD
        payload = found.payload or {}
        link["resolved_snippet"] = (payload.get("snippet") or payload.get("summary") or "")[:300]
        return link
    except Exception as e:
        print(f"[EvidenceResolver] doc resolution failed: {e}")
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link


def _resolve_commit(link: dict, asp: dict) -> dict:
    """Commit-type evidence: verify repo + sha + path exist via GitHub API."""
    repo = link.get("repo")
    sha = link.get("sha")
    path = link.get("path")
    if not (repo and sha and path):
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link

    try:
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={sha}"
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            import base64
            data = resp.json()
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
            start = link.get("start_line", 1) - 1
            end = link.get("end_line", start + 5)
            lines = content.splitlines()
            snippet = "\n".join(lines[max(0, start):end])
            link["resolved"] = True
            link["evidence_score"] = 1.0
            link["resolved_snippet"] = snippet[:500]
        else:
            link["resolved"] = False
            link["evidence_score"] = 0.0
            link["resolved_snippet"] = ""
        return link
    except Exception as e:
        print(f"[EvidenceResolver] commit resolution failed: {e}")
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link


def _resolve_symbol(link: dict, asp: dict) -> dict:
    """Symbol-type evidence: verify symbol exists in Postgres symbols table."""
    symbol = link.get("symbol") or link.get("snippet")
    if not symbol:
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=os.getenv("POSTGRES_PORT", "5433"),
            user=os.getenv("POSTGRES_USER", "sdlc"),
            password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
            dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol_name, file_path FROM symbols WHERE symbol_name ILIKE %s LIMIT 1",
            (f"%{symbol}%",),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            link["resolved"] = True
            link["evidence_score"] = 1.0
            link["resolved_snippet"] = f"{row[0]} in {row[1]}"
        else:
            link["resolved"] = False
            link["evidence_score"] = 0.0
            link["resolved_snippet"] = ""
        return link
    except Exception as e:
        print(f"[EvidenceResolver] symbol resolution failed: {e}")
        link["resolved"] = False
        link["evidence_score"] = 0.0
        return link


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────
def resolve_evidence(artifact: dict, asp: dict) -> dict:
    """
    Resolve every evidence_link in an artifact.
    Adds artifact['evidence_resolved'] (bool) and artifact['evidence_score'] (avg float).
    """
    links = artifact.get("evidence_links", [])

    if not links:
        # No evidence at all — for requirement-type artifacts that may be OK at depth 1,
        # but generally this means the artifact is unsupported.
        artifact["evidence_resolved"] = False
        artifact["evidence_score"] = 0.0
        artifact["evidence_note"] = "No evidence_links provided"
        return artifact

    resolved_links = []
    scores = []
    for link in links:
        ltype = (link.get("type") or "").lower()
        if ltype == "requirement":
            link = _resolve_requirement(link, asp)
        elif ltype == "doc":
            link = _resolve_doc(link, asp)
        elif ltype == "commit":
            link = _resolve_commit(link, asp)
        elif ltype == "symbol":
            link = _resolve_symbol(link, asp)
        else:
            link["resolved"] = False
            link["evidence_score"] = 0.0
            link["resolved_snippet"] = ""
        resolved_links.append(link)
        scores.append(link.get("evidence_score", 0.0))

    artifact["evidence_links"] = resolved_links
    avg_score = sum(scores) / len(scores) if scores else 0.0
    # Artifact is resolved if MAJORITY of links resolve and avg score clears threshold
    resolved_count = sum(1 for l in resolved_links if l.get("resolved"))
    artifact["evidence_resolved"] = (
        resolved_count >= len(resolved_links) / 2 and avg_score >= EVIDENCE_THRESHOLD
    )
    artifact["evidence_score"] = round(avg_score, 2)
    artifact["evidence_note"] = f"{resolved_count}/{len(resolved_links)} links resolved"
    return artifact


def resolve_batch(artifacts: list, asp: dict) -> list:
    """Resolve evidence for a list of artifacts."""
    return [resolve_evidence(a, asp) for a in artifacts]