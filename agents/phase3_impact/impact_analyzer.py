"""
Phase 3 — Impact Analysis
Queries Qdrant (semantic search) + Neo4j (dependency traversal) + PostgreSQL (symbol lookup)
to produce a structured impact report before any code generation begins.

Optimizations vs original:
  - Module-level singletons for all clients (Qdrant, Neo4j driver, Groq)
  - PostgreSQL ThreadedConnectionPool + context-manager borrow/return
  - Embedding encoded once per unique query via lru_cache
  - Batched Neo4j queries: single UNWIND round-trip for all file paths
  - Parallel execution of Steps 2 & 3 (Neo4j + Postgres) via ThreadPoolExecutor
  - Early exit when semantic search returns no hits
  - top-level `import re` (was imported inside try block)
  - set-comprehension for affected_repos dedup
"""

import os
import re
import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import psycopg2
from psycopg2 import pool as pg_pool
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# -----------------------------------------
# Module-level Singletons
# (created once at import time, reused across every call)
# -----------------------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
    timeout=60,
)

_neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password1234"),
    ),
)

_pg_pool = pg_pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5433"),
    user=os.getenv("POSTGRES_USER", "sdlc"),
    password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
    dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge"),
)


@contextmanager
def _borrow_pg():
    """Borrow a Postgres connection from the pool and return it on exit."""
    conn = _pg_pool.getconn()
    try:
        yield conn
    finally:
        _pg_pool.putconn(conn)


# -----------------------------------------
# Step 1 — Semantic Search (Qdrant)
# -----------------------------------------

@lru_cache(maxsize=256)
def _encode(query: str) -> tuple:
    """Encode query text to an embedding vector.

    Result is cached so repeated or identical requirements skip the
    (relatively expensive) transformer inference.
    Returns a tuple so it is hashable for lru_cache.
    """
    return tuple(embedder.encode(query).tolist())


def semantic_search(query: str, top_k: int = 10) -> list:
    """Find the most relevant files and symbols for the change request."""
    query_vector = list(_encode(query))

    results = qdrant_client.query_points(
        collection_name="code_embeddings",
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "repo_name": r.payload.get("repo_name"),
            "file_path": r.payload.get("file_path"),
            "symbol_name": r.payload.get("symbol_name"),
            "symbol_type": r.payload.get("symbol_type"),
            "score": round(r.score, 4),
        }
        for r in results.points
    ]


# -----------------------------------------
# Step 2 — Dependency Traversal (Neo4j) — Batched
# -----------------------------------------

def get_dependents_batch(file_paths: list) -> dict:
    """Return all dependents for every path in one Neo4j round-trip.

    Returns: {file_path: [{"file_path": str, "repo_name": str}]}
    """
    with _neo4j_driver.session() as session:
        result = session.run(
            """
            UNWIND $paths AS p
            MATCH (f:File {path: p})<-[:IMPORTS]-(dep)
            RETURN p AS source, dep.path AS dep_path, dep.repo_name AS dep_repo
            """,
            paths=file_paths,
        )
        grouped: dict = {}
        for record in result:
            grouped.setdefault(record["source"], []).append(
                {"file_path": record["dep_path"], "repo_name": record["dep_repo"]}
            )
    return grouped


def get_affected_symbols_batch(file_paths: list) -> dict:
    """Return all defined symbols for every path in one Neo4j round-trip.

    Returns: {file_path: [{"name": str, "type": str, "line": int}]}
    """
    with _neo4j_driver.session() as session:
        result = session.run(
            """
            UNWIND $paths AS p
            MATCH (f:File {path: p})-[:DEFINES]->(s:Symbol)
            RETURN p AS file_path, s.name AS name, s.type AS type, s.line AS line
            """,
            paths=file_paths,
        )
        grouped: dict = {}
        for record in result:
            grouped.setdefault(record["file_path"], []).append(
                {"name": record["name"], "type": record["type"], "line": record["line"]}
            )
    return grouped


# Kept for external compatibility — wraps the batch variants
def get_dependents(file_path: str) -> list:
    return get_dependents_batch([file_path]).get(file_path, [])


def get_affected_symbols(file_path: str) -> list:
    return get_affected_symbols_batch([file_path]).get(file_path, [])


def get_dependencies(file_path: str) -> list:
    """Find all files this file imports."""
    with _neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (f:File {path: $file_path})-[:IMPORTS]->(dep)
            RETURN dep.name AS name
            """,
            file_path=file_path,
        )
        return [record["name"] for record in result]


# -----------------------------------------
# Step 3 — Protocol Contract Check (Postgres)
# -----------------------------------------

def check_protocol_contracts(repo_name: str) -> list:
    """Find all protocol contracts for a repo (uses pooled connection)."""
    with _borrow_pg() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT contract_type, file_path, contract_name, parsed_content
            FROM protocol_contracts
            WHERE repo_name = %s
            """,
            (repo_name,),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "contract_type": row[0],
            "file_path": row[1],
            "contract_name": row[2],
            "parsed_content": row[3],
        }
        for row in rows
    ]


def _fetch_contracts_parallel(repo_names: list) -> list:
    """Fetch contracts for all repos concurrently."""
    all_contracts: list = []
    if not repo_names:
        return all_contracts

    with ThreadPoolExecutor(max_workers=min(len(repo_names), 8)) as executor:
        futures = {
            executor.submit(check_protocol_contracts, repo): repo
            for repo in repo_names
        }
        for future in as_completed(futures):
            repo = futures[future]
            contracts = future.result()
            if contracts:
                print(
                    f"  {repo} has {len(contracts)} contracts: "
                    f"{[c['contract_name'] for c in contracts]}"
                )
            else:
                print(f"  {repo} — no protocol contracts indexed")
            all_contracts.extend(contracts)

    return all_contracts


# -----------------------------------------
# Step 4 — Risk Assessment (LLM)
# -----------------------------------------

def assess_risk(
        requirement: str,
        affected_files: list,
        affected_symbols: list,
        contracts: list,
) -> dict:
    """Use LLM to assess the risk level of the change."""

    prompt = f"""
You are a senior software architect assessing the risk of a code change.

Requirement:
{requirement}

Affected Files:
{json.dumps(affected_files, indent=2)}

Affected Symbols (classes/functions that will change):
{json.dumps(affected_symbols, indent=2)}

Protocol Contracts Affected:
{json.dumps(contracts, indent=2)}

Assess the risk and return ONLY valid JSON:
{{
  "risk_level": "low|medium|high",
  "risk_reasons": ["reason 1", "reason 2"],
  "breaking_changes": ["change 1", "change 2"],
  "recommendation": "proceed|proceed_with_caution|requires_architect_review"
}}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    try:
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
        return json.loads(content)
    except Exception:
        return {
            "risk_level": "medium",
            "risk_reasons": ["Could not parse risk assessment"],
            "breaking_changes": [],
            "recommendation": "proceed_with_caution",
        }


# -----------------------------------------
# Main Impact Analysis
# -----------------------------------------

def run_impact_analysis(requirement: str) -> dict:
    """
    Full Phase 3 impact analysis pipeline.
    Returns a structured impact report.

    Execution order
    ───────────────
    1. Semantic search  (Qdrant)       — sequential, must finish first
    2. Dependency graph (Neo4j)        ─┐ run in parallel via
    3. Protocol contracts (Postgres)   ─┘ ThreadPoolExecutor
    4. Risk assessment  (Groq LLM)     — sequential, needs 2 & 3
    """

    print(f"\n{'='*50}")
    print("Phase 3 — Impact Analysis")
    print(f"Requirement: {requirement[:100]}...")
    print(f"{'='*50}")

    # ------------------------------------------------------------------
    # Step 1 — Semantic Search
    # ------------------------------------------------------------------
    print("\n[Step 1] Semantic search across indexed repos...")
    hits = semantic_search(requirement, top_k=10)
    print(f"  Found {len(hits)} relevant symbols")

    if not hits:
        print("  No affected files found — aborting analysis.")
        return {
            "requirement": requirement,
            "status": "NO_AFFECTED_FILES",
            "affected_repos": [],
            "affected_files": [],
            "affected_symbols": [],
            "dependents": {},
            "protocol_contracts": [],
            "risk_assessment": {},
        }

    # Deduplicate affected files (preserve highest-score hit per file)
    affected_files: dict = {}
    for hit in hits:
        key = f"{hit['repo_name']}:{hit['file_path']}"
        if key not in affected_files:
            affected_files[key] = {
                "repo_name": hit["repo_name"],
                "file_path": hit["file_path"],
                "relevance_score": hit["score"],
                "matched_symbols": [],
            }
        affected_files[key]["matched_symbols"].append(
            {"name": hit["symbol_name"], "type": hit["symbol_type"], "score": hit["score"]}
        )

    affected_files_list = list(affected_files.values())
    file_paths = [f["file_path"] for f in affected_files_list]
    affected_repos = list({f["repo_name"] for f in affected_files_list})

    print(f"  Affected files: {file_paths}")

    # ------------------------------------------------------------------
    # Steps 2 & 3 — Neo4j batched queries + Postgres contract checks
    # run concurrently; neither depends on the other's output
    # ------------------------------------------------------------------
    print("\n[Step 2] Traversing dependency graph (batched)...")
    print("[Step 3] Checking protocol contracts (parallel)...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_symbols    = executor.submit(get_affected_symbols_batch, file_paths)
        fut_dependents = executor.submit(get_dependents_batch, file_paths)
        fut_contracts  = executor.submit(_fetch_contracts_parallel, affected_repos)

        symbols_by_file: dict = fut_symbols.result()     # {file_path: [symbols]}
        dependents_map: dict  = fut_dependents.result()  # {file_path: [dependents]}
        all_contracts: list   = fut_contracts.result()

    # Flatten symbols list
    all_affected_symbols = [s for syms in symbols_by_file.values() for s in syms]

    # Log dependents
    for fp, deps in dependents_map.items():
        print(f"  {fp} is depended on by: {[d['file_path'] for d in deps]}")

    # ------------------------------------------------------------------
    # Step 4 — Risk Assessment (LLM)
    # ------------------------------------------------------------------
    print("\n[Step 4] Assessing risk...")
    risk = assess_risk(requirement, affected_files_list, all_affected_symbols, all_contracts)
    print(f"  Risk level: {risk['risk_level']}")
    print(f"  Recommendation: {risk['recommendation']}")

    impact_report = {
        "requirement": requirement,
        "affected_repos": affected_repos,
        "affected_files": affected_files_list,
        "affected_symbols": all_affected_symbols,
        "dependents": dependents_map,
        "protocol_contracts": all_contracts,
        "risk_assessment": risk,
        "status": "PENDING_APPROVAL",
    }

    print(f"\n{'='*50}")
    print("✅ Impact Analysis Complete")
    print(f"   Repos affected: {affected_repos}")
    print(f"   Files affected: {len(affected_files_list)}")
    print(f"   Symbols affected: {len(all_affected_symbols)}")
    print(f"   Risk: {risk['risk_level']}")
    print(f"{'='*50}\n")

    return impact_report


# -----------------------------------------
# CLI Test
# -----------------------------------------

if __name__ == "__main__":
    requirement = (
        "Add leave balance tracker. Each employee gets 20 days per year. "
        "Balance decreases when leave is approved."
    )

    report = run_impact_analysis(requirement)

    print("\nFull Impact Report:")
    print(json.dumps(report, indent=2))
