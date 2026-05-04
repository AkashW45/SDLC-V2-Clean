"""
Phase 3 — Impact Analysis
Queries Qdrant (semantic search) + Neo4j (dependency traversal) + PostgreSQL (symbol lookup)
to produce a structured impact report before any code generation begins.
"""

import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import psycopg2
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# -----------------------------------------
# Connections
# -----------------------------------------

def get_postgres():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )

def get_qdrant():
    return QdrantClient(
        url="http://127.0.0.1:6333",
        timeout=60
    )

def get_neo4j():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password1234")
        )
    )

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# Step 1 — Semantic Search (Qdrant)
# -----------------------------------------

def semantic_search(query: str, top_k: int = 10) -> list:
    """Find most relevant files and symbols for the change request."""
    qdrant = get_qdrant()

    query_vector = embedder.encode(query).tolist()

    results = qdrant.query_points(
        collection_name="code_embeddings",
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    hits = []
    for r in results.points:
        hits.append({
            "repo_name": r.payload.get("repo_name"),
            "file_path": r.payload.get("file_path"),
            "symbol_name": r.payload.get("symbol_name"),
            "symbol_type": r.payload.get("symbol_type"),
            "score": round(r.score, 4)
        })

    return hits


# -----------------------------------------
# Step 2 — Dependency Traversal (Neo4j)
# -----------------------------------------

def get_dependents(file_path: str) -> list:
    """Find all files that import or depend on this file."""
    driver = get_neo4j()
    dependents = []

    with driver.session() as session:
        result = session.run("""
            MATCH (f:File {path: $file_path})<-[:IMPORTS]-(dependent)
            RETURN dependent.path as path, dependent.repo_name as repo
        """, file_path=file_path)

        for record in result:
            dependents.append({
                "file_path": record["path"],
                "repo_name": record["repo"]
            })

    driver.close()
    return dependents


def get_dependencies(file_path: str) -> list:
    """Find all files this file imports."""
    driver = get_neo4j()
    dependencies = []

    with driver.session() as session:
        result = session.run("""
            MATCH (f:File {path: $file_path})-[:IMPORTS]->(dep)
            RETURN dep.name as name
        """, file_path=file_path)

        for record in result:
            dependencies.append(record["name"])

    driver.close()
    return dependencies


def get_affected_symbols(file_path: str) -> list:
    """Get all symbols defined in a file."""
    driver = get_neo4j()
    symbols = []

    with driver.session() as session:
        result = session.run("""
            MATCH (f:File {path: $file_path})-[:DEFINES]->(s:Symbol)
            RETURN s.name as name, s.type as type, s.line as line
        """, file_path=file_path)

        for record in result:
            symbols.append({
                "name": record["name"],
                "type": record["type"],
                "line": record["line"]
            })

    driver.close()
    return symbols


# -----------------------------------------
# Step 3 — Protocol Contract Check
# -----------------------------------------

def check_protocol_contracts(repo_name: str) -> list:
    """Find all protocol contracts for affected repos."""
    conn = get_postgres()
    cur = conn.cursor()

    cur.execute("""
        SELECT contract_type, file_path, contract_name, parsed_content
        FROM protocol_contracts
        WHERE repo_name = %s
    """, (repo_name,))

    contracts = []
    for row in cur.fetchall():
        contracts.append({
            "contract_type": row[0],
            "file_path": row[1],
            "contract_name": row[2],
            "parsed_content": row[3]
        })

    cur.close()
    conn.close()
    return contracts


# -----------------------------------------
# Step 4 — Risk Assessment (LLM)
# -----------------------------------------

def assess_risk(
    requirement: str,
    affected_files: list,
    affected_symbols: list,
    contracts: list
) -> dict:
    """Use LLM to assess risk level of the change."""

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
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()

    try:
        # Clean markdown fences if present
        if content.startswith("```"):
            import re
            content = re.sub(r"```(?:json)?", "", content).strip().strip("```").strip()
        return json.loads(content)
    except Exception:
        return {
            "risk_level": "medium",
            "risk_reasons": ["Could not parse risk assessment"],
            "breaking_changes": [],
            "recommendation": "proceed_with_caution"
        }


# -----------------------------------------
# Main Impact Analysis
# -----------------------------------------

def run_impact_analysis(requirement: str) -> dict:
    """
    Full Phase 3 impact analysis pipeline.
    Returns structured impact report.
    """

    print(f"\n{'='*50}")
    print("Phase 3 — Impact Analysis")
    print(f"Requirement: {requirement[:100]}...")
    print(f"{'='*50}")

    # Step 1 — Semantic Search
    print("\n[Step 1] Semantic search across indexed repos...")
    hits = semantic_search(requirement, top_k=10)
    print(f"  Found {len(hits)} relevant symbols")

    # Deduplicate affected files
    affected_files = {}
    for hit in hits:
        key = f"{hit['repo_name']}:{hit['file_path']}"
        if key not in affected_files:
            affected_files[key] = {
                "repo_name": hit["repo_name"],
                "file_path": hit["file_path"],
                "relevance_score": hit["score"],
                "matched_symbols": []
            }
        affected_files[key]["matched_symbols"].append({
            "name": hit["symbol_name"],
            "type": hit["symbol_type"],
            "score": hit["score"]
        })

    affected_files_list = list(affected_files.values())
    print(f"  Affected files: {[f['file_path'] for f in affected_files_list]}")

    # Step 2 — Dependency Traversal
    print("\n[Step 2] Traversing dependency graph...")
    all_affected_symbols = []
    dependents_map = {}

    for af in affected_files_list:
        symbols = get_affected_symbols(af["file_path"])
        all_affected_symbols.extend(symbols)

        dependents = get_dependents(af["file_path"])
        if dependents:
            dependents_map[af["file_path"]] = dependents
            print(f"  {af['file_path']} is depended on by: {[d['file_path'] for d in dependents]}")

    # Step 3 — Protocol Contract Check
    print("\n[Step 3] Checking protocol contracts...")
    all_contracts = []
    affected_repos = list(set(f["repo_name"] for f in affected_files_list))

    for repo in affected_repos:
        contracts = check_protocol_contracts(repo)
        all_contracts.extend(contracts)
        if contracts:
            print(f"  {repo} has {len(contracts)} contracts: {[c['contract_name'] for c in contracts]}")
        else:
            print(f"  {repo} — no protocol contracts indexed")

    # Step 4 — Risk Assessment
    print("\n[Step 4] Assessing risk...")
    risk = assess_risk(
        requirement,
        affected_files_list,
        all_affected_symbols,
        all_contracts
    )
    print(f"  Risk level: {risk['risk_level']}")
    print(f"  Recommendation: {risk['recommendation']}")

    # Build final impact report
    impact_report = {
        "requirement": requirement,
        "affected_repos": affected_repos,
        "affected_files": affected_files_list,
        "affected_symbols": all_affected_symbols,
        "dependents": dependents_map,
        "protocol_contracts": all_contracts,
        "risk_assessment": risk,
        "status": "PENDING_APPROVAL"
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
    requirement = "Add leave balance tracker. Each employee gets 20 days per year. Balance decreases when leave is approved."

    report = run_impact_analysis(requirement)

    print("\nFull Impact Report:")
    print(json.dumps(report, indent=2))