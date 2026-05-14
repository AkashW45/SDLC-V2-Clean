"""
Project Registry — manages metadata about projects (groupings of repos).
"""

import os
import json
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_postgres():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )


def get_qdrant():
    return QdrantClient(url="http://127.0.0.1:6333", timeout=60)


def register_project(project_id: str, project_name: str, description: str,
                     domain: str, tech_stack: list, repos: list, owner_team: str):
    """Add or update a project in the registry."""
    conn = get_postgres()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO projects (project_id, project_name, description, domain, tech_stack, repos, owner_team)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (project_id) DO UPDATE
        SET project_name = EXCLUDED.project_name,
            description = EXCLUDED.description,
            domain = EXCLUDED.domain,
            tech_stack = EXCLUDED.tech_stack,
            repos = EXCLUDED.repos,
            owner_team = EXCLUDED.owner_team
    """, (
        project_id,
        project_name,
        description,
        domain,
        json.dumps(tech_stack),
        json.dumps(repos),
        owner_team
    ))
    conn.commit()
    cur.close()
    conn.close()

    # Create embedding from project description + name + domain
    text_for_embedding = f"{project_name}. {description}. Domain: {domain}. Repos: {', '.join(repos)}"
    embedding = get_embedder().encode(text_for_embedding).tolist()

    # Store in Qdrant
    qdrant = get_qdrant()
    qdrant.upsert(
        collection_name="project_embeddings",
        points=[PointStruct(
            id=hash(project_id) % (2**63),
            vector=embedding,
            payload={
                "project_id": project_id,
                "project_name": project_name,
                "description": description,
                "domain": domain,
                "repos": repos
            }
        )]
    )

    print(f"✅ Registered: {project_name} ({len(repos)} repos)")


def search_projects(requirement: str, top_k: int = 3) -> list:
    """Find top-N candidate projects for a requirement."""
    embedder = get_embedder()
    query_vector = embedder.encode(requirement).tolist()

    qdrant = get_qdrant()
    results = qdrant.query_points(
        collection_name="project_embeddings",
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    candidates = []
    for r in results.points:
        candidates.append({
            "project_id": r.payload["project_id"],
            "project_name": r.payload["project_name"],
            "description": r.payload["description"],
            "domain": r.payload["domain"],
            "repos": r.payload["repos"],
            "score": round(r.score, 4)
        })

    return candidates


def list_all_projects() -> list:
    """List all registered projects."""
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute("""
        SELECT project_id, project_name, description, domain, tech_stack, repos, owner_team
        FROM projects ORDER BY project_name
    """)
    projects = []
    for r in cur.fetchall():
        projects.append({
            "project_id": r[0],
            "project_name": r[1],
            "description": r[2],
            "domain": r[3],
            "tech_stack": r[4],
            "repos": r[5],
            "owner_team": r[6]
        })
    cur.close()
    conn.close()
    return projects


# ── CLI for registering demo projects ────────────────────────────────────────
if __name__ == "__main__":
    # Register your existing demo project
    register_project(
        project_id="leave-mgmt",
        project_name="Leave Management System",
        description="Employee leave request, approval, and balance tracking system. "
                    "Handles annual leave entitlement, sick leave, casual leave, "
                    "manager approval workflow, and year-end balance reset.",
        domain="HR",
        tech_stack=["Python", "FastAPI", "Pydantic"],
        repos=["leave-mgmt-backend", "leave-mgmt-frontend", "leave-mgmt-batch"],
        owner_team="HR Platform Team"
    )

    # Add more demo projects to make selection meaningful
    register_project(
        project_id="wealth-mgmt",
        project_name="Wealth Management Platform",
        description="Portfolio management and investment advisory platform for "
                    "high net worth clients. Includes portfolio analytics, "
                    "trade execution, risk assessment, and client reporting.",
        domain="Banking",
        tech_stack=["Java", "Spring Boot", "React"],
        repos=["wealth-frontend", "wealth-backend", "wealth-analytics", "wealth-shared-lib"],
        owner_team="Wealth Engineering"
    )

    register_project(
        project_id="payment-gateway",
        project_name="Secure Payment Gateway",
        description="PCI-DSS compliant credit card payment processing system. "
                    "Handles authorization, capture, refund, tokenization, "
                    "3D Secure authentication, and fraud detection.",
        domain="Payments",
        tech_stack=["Java", "Spring", "Kafka"],
        repos=["payment-api", "payment-tokenizer", "payment-fraud-engine", "payment-batch"],
        owner_team="Payments Platform"
    )

    register_project(
        project_id="loan-origination",
        project_name="Loan Origination System",
        description="End-to-end loan application processing including credit check, "
                    "underwriting, approval workflow, document management, "
                    "and disbursement.",
        domain="Lending",
        tech_stack=["Python", "Django", "PostgreSQL"],
        repos=["loan-frontend", "loan-backend", "loan-credit-check", "loan-document-mgmt"],
        owner_team="Lending Platform"
    )

    print("\n✅ All projects registered. Test search:")

    # Test
    test_query = "Add leave balance tracker"
    candidates = search_projects(test_query, top_k=3)
    print(f"\nQuery: '{test_query}'")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c['project_name']} (score: {c['score']})")
        print(f"     Repos: {', '.join(c['repos'])}")