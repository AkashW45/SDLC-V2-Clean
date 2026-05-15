"""
agents/context_packet_builder.py  (FINAL)

Closes the brownfield gap. Before codegen runs, this module fetches the actual
relevant code from the existing codebase and formats it as a context packet that
gets injected into the codegen LLM call.

Without this: codegen sees only `repo_summary` metadata (top_symbols list) and
guesses at file structure when emitting PATCH artifacts.

With this: codegen sees the ACTUAL function bodies / class definitions / route
handlers most relevant to the requirement, fetched via:
  1. Embed the requirement (Qdrant code_embeddings vector search)
  2. Top-K relevant code chunks
  3. Enrich each chunk with its Postgres symbols row (file_path, line numbers, kind)
  4. Optionally fetch the full file content for the highest-scoring chunks
  5. Format as a deterministic "RELEVANT EXISTING CODE" block

For GREENFIELD: returns empty packet — there is no existing code to ground against.

Public API:
  build_context_packet(requirement, asp, top_k=8, max_files=4) -> str
  inject_into_user_message(user_message, packet) -> str
"""
import os
import logging
from typing import Dict, Any, List

import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ContextPacketBuilder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)


# ─────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────
def _qdrant() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))


def _pg():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
    )


def _openai_for_embeddings() -> OpenAI:
    """Uses the embedding endpoint — keep cheap & fast (text-embedding-3-small)."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),  # None = openai.com default
    )


def _embed(text: str) -> List[float]:
    """Embed a single query string. Returns [] on failure (graceful degrade)."""
    try:
        client = _openai_for_embeddings()
        resp = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text[:8000],  # cap query length
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.warning(f"_embed failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Step 1: vector search in code_embeddings
# ─────────────────────────────────────────────────────────────────────
def _search_code_chunks(query_vec: List[float], repo_name: str,
                        top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Search Qdrant code_embeddings for the most relevant chunks in a given repo.
    Falls back to no-filter search if filter syntax fails.
    """
    if not query_vec:
        return []
    try:
        qd = _qdrant()
        hits = qd.search(
            collection_name="code_embeddings",
            query_vector=query_vec,
            limit=top_k * 2,  # over-fetch so filter on repo still gives us top_k
            with_payload=True,
        )
    except Exception as e:
        logger.warning(f"qdrant search failed: {e}")
        return []

    chunks = []
    for h in hits:
        payload = h.payload or {}
        # Filter by repo (case-insensitive substring match — handles full URLs and short names)
        chunk_repo = (payload.get("repo") or payload.get("repo_name") or "").lower()
        if repo_name and repo_name.lower() not in chunk_repo and chunk_repo not in repo_name.lower():
            continue
        chunks.append({
            "score": float(h.score),
            "file_path": payload.get("file_path") or payload.get("path", ""),
            "symbol": payload.get("symbol") or payload.get("name", ""),
            "language": payload.get("language", ""),
            "snippet": payload.get("content") or payload.get("code", "") or payload.get("snippet", ""),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
        })
        if len(chunks) >= top_k:
            break
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Step 2: enrich chunks with Postgres symbols data
# ─────────────────────────────────────────────────────────────────────
def _enrich_with_symbols(chunks: List[Dict[str, Any]], repo_name: str) -> List[Dict[str, Any]]:
    """
    For each chunk, look up its row in the symbols table to get authoritative
    file_path, line numbers, kind (function/class/route). Gracefully skip if the
    symbols table doesn't exist or row not found.
    """
    if not chunks:
        return chunks
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for c in chunks:
            symbol = c.get("symbol") or ""
            fp = c.get("file_path") or ""
            if not symbol and not fp:
                continue
            try:
                cur.execute(
                    """SELECT file_path, name, kind, start_line, end_line, signature
                       FROM symbols
                       WHERE repo ILIKE %s
                         AND (name = %s OR file_path = %s)
                       LIMIT 1""",
                    (f"%{repo_name}%", symbol, fp),
                )
                row = cur.fetchone()
                if row:
                    c["file_path"] = row.get("file_path") or c["file_path"]
                    c["kind"] = row.get("kind")
                    c["start_line"] = row.get("start_line") or c.get("start_line")
                    c["end_line"] = row.get("end_line") or c.get("end_line")
                    c["signature"] = row.get("signature")
            except Exception as e:
                logger.debug(f"symbols lookup skipped for {symbol}: {e}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"_enrich_with_symbols failed: {e}")
    return chunks


# ─────────────────────────────────────────────────────────────────────
# Step 3: fetch full file content for the highest-scoring chunks
# ─────────────────────────────────────────────────────────────────────
def _fetch_full_files(chunks: List[Dict[str, Any]], repo_name: str,
                      max_files: int = 4) -> Dict[str, str]:
    """
    For the top distinct files in the chunk list, fetch their full content from
    the indexed_files table (if it exists) or from GitHub API as a fallback.
    Returns {file_path: content}. Capped to max_files to stay within context.
    """
    seen_paths = []
    for c in chunks:
        fp = c.get("file_path")
        if fp and fp not in seen_paths:
            seen_paths.append(fp)
        if len(seen_paths) >= max_files:
            break

    file_contents: Dict[str, str] = {}
    if not seen_paths:
        return file_contents

    # Try indexed_files table first
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Check whether the table exists
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'indexed_files' LIMIT 1
        """)
        if cur.fetchone():
            for fp in seen_paths:
                try:
                    cur.execute(
                        """SELECT content FROM indexed_files
                           WHERE repo ILIKE %s AND file_path = %s LIMIT 1""",
                        (f"%{repo_name}%", fp),
                    )
                    row = cur.fetchone()
                    if row and row.get("content"):
                        file_contents[fp] = row["content"]
                except Exception as e:
                    logger.debug(f"indexed_files lookup failed for {fp}: {e}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.debug(f"_fetch_full_files indexed_files step skipped: {e}")

    # Fallback: GitHub API for any file not yet found
    missing = [fp for fp in seen_paths if fp not in file_contents]
    if missing:
        try:
            import requests, base64
            token = os.getenv("GITHUB_TOKEN")
            owner = os.getenv("GITHUB_OWNER", "AkashW45")
            headers = {"Accept": "application/vnd.github+json"}
            if token:
                headers["Authorization"] = f"token {token}"
            for fp in missing:
                try:
                    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{fp}"
                    r = requests.get(url, headers=headers, timeout=8)
                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, dict) and data.get("encoding") == "base64":
                            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                            file_contents[fp] = content
                except Exception as e:
                    logger.debug(f"github fetch failed for {fp}: {e}")
        except Exception as e:
            logger.debug(f"github fetch step skipped: {e}")

    return file_contents


# ─────────────────────────────────────────────────────────────────────
# Step 4: format the packet
# ─────────────────────────────────────────────────────────────────────
def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n" + text[-half:]


def _format_packet(chunks: List[Dict[str, Any]], file_contents: Dict[str, str],
                   repo_name: str, requirement: str) -> str:
    """Build the deterministic 'RELEVANT EXISTING CODE' block."""
    lines = []
    lines.append("=" * 70)
    lines.append("RELEVANT EXISTING CODE — fetched from your codebase")
    lines.append("=" * 70)
    lines.append(f"Repository: {repo_name}")
    lines.append(f"Requirement: {requirement[:200]}")
    lines.append("")
    lines.append("This is the actual code from the existing system, most relevant")
    lines.append("to the requirement above. When you emit PATCH artifacts:")
    lines.append("  - reference these exact file paths")
    lines.append("  - preserve existing function signatures unless the change requires updating them")
    lines.append("  - do not regenerate code that already exists and works")
    lines.append("  - only emit the DELTA needed for the requirement")
    lines.append("")

    if chunks:
        lines.append("─" * 70)
        lines.append("TOP RELEVANT SYMBOLS (vector-search ranked):")
        lines.append("─" * 70)
        for i, c in enumerate(chunks[:8], start=1):
            lines.append(
                f"[{i}] {c.get('file_path', '?')} :: {c.get('symbol', '?')} "
                f"({c.get('kind', 'symbol')}, lines {c.get('start_line', '?')}-{c.get('end_line', '?')}, "
                f"score={c.get('score', 0):.3f})"
            )
            if c.get("signature"):
                lines.append(f"    signature: {c['signature']}")
            snippet = c.get("snippet") or ""
            if snippet:
                lines.append("    snippet:")
                for sl in _truncate(snippet, 600).splitlines():
                    lines.append(f"      {sl}")
        lines.append("")

    if file_contents:
        lines.append("─" * 70)
        lines.append("FULL FILE CONTENTS (top files for full context):")
        lines.append("─" * 70)
        # Total budget across files — keep packet under ~12k chars total
        per_file = max(800, 10000 // max(1, len(file_contents)))
        for fp, content in file_contents.items():
            lines.append("")
            lines.append(f"### FILE: {fp}")
            lines.append("```")
            lines.append(_truncate(content, per_file))
            lines.append("```")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF EXISTING CODE CONTEXT")
    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def build_context_packet(requirement: str, asp: Dict[str, Any],
                         top_k: int = 8, max_files: int = 4) -> str:
    """
    Build the existing-code context packet for codegen.

    GREENFIELD: returns "" (no existing code to fetch).
    BROWNFIELD: returns a formatted block with top-K relevant chunks + top-N full files.

    Failure-tolerant: any step (Qdrant down, Postgres down, embedding fails) degrades
    gracefully and returns either a partial packet or "".
    """
    build_mode = (asp.get("build_mode") or "greenfield").lower()
    if build_mode != "modify_existing":
        logger.info("[ContextPacket] greenfield — no existing code to fetch")
        return ""

    repo_summary = asp.get("repo_summary", {}) or {}
    repo_name = (
        repo_summary.get("matched_repo")
        or repo_summary.get("name")
        or repo_summary.get("repo_name")
        or ""
    )
    if not repo_name:
        logger.warning("[ContextPacket] build_mode=modify_existing but no repo_summary.matched_repo — returning empty")
        return ""

    logger.info(f"[ContextPacket] building for repo={repo_name}, top_k={top_k}, max_files={max_files}")

    # Step 1: embed the requirement
    query_vec = _embed(requirement)
    if not query_vec:
        logger.warning("[ContextPacket] embedding failed — returning empty packet")
        return ""

    # Step 2: vector search
    chunks = _search_code_chunks(query_vec, repo_name, top_k=top_k)
    if not chunks:
        logger.warning(f"[ContextPacket] no chunks found in code_embeddings for {repo_name}")
        # Still useful to tell the LLM the repo exists even without chunks
        return (
            f"RELEVANT EXISTING CODE — repo '{repo_name}' is indexed but no specific "
            f"code chunks matched this requirement closely. Emit PATCH artifacts that "
            f"reference real file paths from repo_summary.top_symbols. Do NOT invent "
            f"file paths."
        )

    # Step 3: enrich with symbols metadata
    chunks = _enrich_with_symbols(chunks, repo_name)

    # Step 4: fetch full files for top hits
    file_contents = _fetch_full_files(chunks, repo_name, max_files=max_files)

    # Step 5: format
    packet = _format_packet(chunks, file_contents, repo_name, requirement)
    logger.info(
        f"[ContextPacket] built: {len(chunks)} chunks, {len(file_contents)} full files, "
        f"{len(packet)} chars"
    )
    return packet


def inject_into_user_message(user_message: str, packet: str) -> str:
    """
    Prepend the context packet to the codegen user message.
    If packet is empty (greenfield), returns user_message unchanged.
    """
    if not packet:
        return user_message
    return packet + "\n\n" + user_message
