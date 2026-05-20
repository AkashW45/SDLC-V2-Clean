"""
Phase 3 — Impact Analysis
Unified Version: Blends Context Precision (PRD/ADR injection) with High-Performance
Concurrency (Neo4j batching, PG pooling, and ThreadPools).
"""

import os
import re
import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from core.db_clients import (
    pg_conn as _borrow_pg,
    neo4j_driver as _neo4j_driver,
    qdrant_client,
)

load_dotenv()

# -----------------------------------------
# Module-level Singletons
# -----------------------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# qdrant_client, _neo4j_driver, and _borrow_pg are imported above from
# core.db_clients — a single project-wide pool/driver/client shared across
# all agents. Phase 3 previously kept its own private pools; removing them
# here cuts ~10 redundant Postgres connections and one extra Neo4j driver
# per process.


# -----------------------------------------
# Step 1 — Semantic Search (Qdrant)
# -----------------------------------------

@lru_cache(maxsize=256)
def _encode(query: str) -> tuple:
    """Encode query text to an embedding vector with LRU Cache."""
    return tuple(embedder.encode(query).tolist())


def semantic_search(query: str, top_k: int = 3, repo_names: list = None) -> list:
    """
    Find the most relevant files. If repo_names is provided, results are scoped
    to those repos only — preventing impact analysis from bleeding into projects
    the user didn't select.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchAny

    query_vector = list(_encode(query))

    qdrant_filter = None
    if repo_names:
        # Filter to repos the user actually selected
        qdrant_filter = Filter(
            must=[FieldCondition(key="repo_name", match=MatchAny(any=repo_names))]
        )

    results = qdrant_client.query_points(
        collection_name="code_embeddings",
        query=query_vector,
        query_filter=qdrant_filter,
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
# Step 2 — Dependency Traversal (Neo4j Batched)
# -----------------------------------------

def get_dependents_batch(file_paths: list) -> dict:
    """Return all dependents for every path in one Neo4j round-trip."""
    if not file_paths:
        return {}
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
    """Return all defined symbols for every path in one Neo4j round-trip."""
    if not file_paths:
        return {}
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


# Wrappers for external compatibility
def get_dependents(file_path: str) -> list:
    return get_dependents_batch([file_path]).get(file_path, [])

def get_affected_symbols(file_path: str) -> list:
    return get_affected_symbols_batch([file_path]).get(file_path, [])


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
        futures = {executor.submit(check_protocol_contracts, repo): repo for repo in repo_names}
        for future in as_completed(futures):
            repo = futures[future]
            contracts = future.result()
            if contracts:
                print(f"  {repo} has {len(contracts)} contracts: {[c['contract_name'] for c in contracts]}")
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
        functional_requirements: list = None,
        adr: dict = None
) -> dict:
    """Use DeepSeek to assess risk level, injecting PRD/ADR for strict Context Precision."""

    functional_reqs_str = ""
    adr_text = "ADR not provided."
    if adr and isinstance(adr, dict):
        adr_text = json.dumps(adr, indent=2)
    if functional_requirements:
        req_summary = "\n".join([f"- {req.get('title', 'Untitled')}" for req in functional_requirements[:5]])
        functional_reqs_str = f"\n\nFunctional Requirements Summary:\n{req_summary}"

    prompt = f"""
You are a senior software architect assessing the risk of a code change.

ADR:
{adr_text}

CRITICAL: First, read the ADR to determine the agreed-upon programming language, framework, and tech stack. You MUST assess risk and breaking changes in strict alignment with this tech stack.

Requirement:
{requirement}{functional_reqs_str}

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

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    content = response.choices[0].message.content.strip()

    try:
        if content.startswith("```"):
            content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
        return json.loads(content)
    except Exception as e:
        print(f"  [Risk Assessment] JSON parse failed: {e}")
        return {
            "risk_level": "medium",
            "risk_reasons": ["Could not parse risk assessment output"],
            "breaking_changes": [],
            "recommendation": "proceed_with_caution"
        }


# -----------------------------------------
# Main Impact Analysis Runner
# -----------------------------------------

def run_impact_analysis(requirement: str, prd: dict = None, adr: dict = None,
                        selected_repos: list = None) -> dict:
    """
    Execution order:
    1. Semantic search (Qdrant) — sequential
    2. Dependency graph (Neo4j batched) ─┐ Concurrent execution
    3. Protocol contracts (Postgres)    ─┘
    4. Risk assessment (LLM w/ Context Precision)
    """

    print(f"\n{'='*50}")
    print("Phase 3 — Impact Analysis")
    print(f"Requirement: {requirement[:100]}...")
    print(f"{'='*50}")

    functional_reqs = []
    if prd and isinstance(prd, dict):
        functional_reqs = prd.get("functional_requirements", [])

    # 1. Semantic Search
    print("\n[Step 1] Semantic search across indexed repos...")
    # Scope semantic search to the selected repos so we don't bleed into other projects
    selected_repo_names = [
        r if isinstance(r, str) else r.get("name")
        for r in (selected_repos or [])
        if (r if isinstance(r, str) else r.get("name"))
    ]
    hits = semantic_search(requirement, top_k=8, repo_names=selected_repo_names or None)
    print(f"  Scoped to repos: {selected_repo_names or 'ALL'}")
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
            "risk_assessment": {"risk_level": "low", "breaking_changes": [], "recommendation": "proceed"},
        }

    # Deduplicate
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
    affected_repo_names = list({f["repo_name"] for f in affected_files_list})

    # Build a name→metadata lookup from Phase 0's selected_repos so we can
    # enrich affected_repos with git URLs, type, language, etc. in one place.
    # Downstream phases (Phase 7) can then use impact_report["affected_repos"]
    # directly without needing a separate selected_repos lookup.
    selected_repos_by_name = {}
    for r in (selected_repos or []):
        if isinstance(r, dict) and r.get("name"):
            selected_repos_by_name[r["name"]] = r

    affected_repos = []
    for name in affected_repo_names:
        meta = selected_repos_by_name.get(name)
        if meta:
            # Merge Phase 0 metadata (name, url, type, language, etc.) and mark impacted
            affected_repos.append({**meta, "impacted": True})
        else:
            # No Phase 0 metadata — store name only for backward compatibility
            affected_repos.append({"name": name, "impacted": True})

    print(f"  Affected files: {file_paths}")

    # 2 & 3. Neo4j and Postgres (Concurrent Execution)
    print("\n[Step 2] Traversing dependency graph (batched)...")
    print("[Step 3] Checking protocol contracts (parallel)...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_symbols    = executor.submit(get_affected_symbols_batch, file_paths)
        fut_dependents = executor.submit(get_dependents_batch, file_paths)
        fut_contracts  = executor.submit(_fetch_contracts_parallel, affected_repo_names)

        symbols_by_file: dict = fut_symbols.result()
        dependents_map: dict  = fut_dependents.result()
        all_contracts: list   = fut_contracts.result()

    all_affected_symbols = [s for syms in symbols_by_file.values() for s in syms]

    for fp, deps in dependents_map.items():
        print(f"  {fp} is depended on by: {[d['file_path'] for d in deps]}")

    # 4. Risk Assessment
    print("\n[Step 4] Assessing risk (Context Precision)...")
    risk = assess_risk(
        requirement,
        affected_files_list,
        all_affected_symbols,
        all_contracts,
        functional_requirements=functional_reqs,
        adr=adr
    )

    print(f"  Risk level: {risk.get('risk_level')}")
    print(f"  Recommendation: {risk.get('recommendation')}")

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
    print(f"   Repos affected: {[r['name'] for r in affected_repos]}")
    print(f"   Files affected: {len(affected_files_list)}")
    print(f"   Symbols affected: {len(all_affected_symbols)}")
    print(f"   Risk: {risk.get('risk_level')}")
    print(f"{'='*50}\n")

    return impact_report

# -----------------------------------------
# CLI Test
# -----------------------------------------
if __name__ == "__main__":
    requirement = "Add leave balance tracker. Each employee gets 20 days per year."
    report = run_impact_analysis(requirement)
    print("\nFull Impact Report:")
    print(json.dumps(report, indent=2))