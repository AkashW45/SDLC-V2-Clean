"""
Project Registry — production-grade.

Manages the list of projects that Phase 0 can route requirements to.

Three usage modes:
  1. As a library: register_project(...), search_projects(...), list_all_projects()
  2. As a CLI:
       python knowledge_layer/project_registry.py register --config projects.yaml
       python knowledge_layer/project_registry.py register --project-id X --name "..." --repos "r1,r2"
       python knowledge_layer/project_registry.py list
       python knowledge_layer/project_registry.py delete --project-id X
       python knowledge_layer/project_registry.py search --query "..."
  3. Through the API: POST /knowledge/projects/register

No hardcoded demo projects. The CLI/config drives everything.
"""

import argparse
import json
import os
import sys

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
# SentenceTransformer now comes from the shared singleton (core/embeddings.py).

load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Connections (lazy)
# ─────────────────────────────────────────────────────────────────────
_embedder = None


def get_embedder():
    # Shared process-wide singleton (see core/embeddings.py).
    from core.embeddings import get_embedder as _shared
    return _shared()


def get_postgres():
    # Pooled connection — .close() returns it to the pool.
    from core.db_clients import PooledConn
    return PooledConn()


def get_qdrant():
    # Process-wide Qdrant client singleton.
    from core.db_clients import qdrant_client
    return qdrant_client


# ─────────────────────────────────────────────────────────────────────
# Core CRUD
# ─────────────────────────────────────────────────────────────────────
def register_project(
        project_id: str,
        project_name: str,
        description: str,
        domain: str = "",
        tech_stack: list = None,
        repos: list = None,
        owner_team: str = "",
) -> dict:
    """
    Idempotent register/update of a project in both PostgreSQL and Qdrant.
    Returns the registered project dict.
    """
    tech_stack = tech_stack or []
    repos = repos or []

    if not project_id or not project_name:
        raise ValueError("project_id and project_name are required")

    # ── PostgreSQL ──
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (project_id, project_name, description, domain, tech_stack, repos, owner_team)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE
                                            SET project_name = EXCLUDED.project_name,
                                            description  = EXCLUDED.description,
                                            domain       = EXCLUDED.domain,
                                            tech_stack   = EXCLUDED.tech_stack,
                                            repos        = EXCLUDED.repos,
                                            owner_team   = EXCLUDED.owner_team
        """,
        (project_id, project_name, description, domain,
         json.dumps(tech_stack), json.dumps(repos), owner_team),
    )
    conn.commit()
    cur.close()
    conn.close()

    # ── Qdrant ──
    text_for_embedding = (
        f"{project_name}. {description}. Domain: {domain}. "
        f"Tech: {', '.join(tech_stack)}. Repos: {', '.join(repos)}"
    )
    embedding = get_embedder().encode(text_for_embedding).tolist()

    qdrant = get_qdrant()
    qdrant.upsert(
        collection_name="project_embeddings",
        points=[PointStruct(
            id=abs(hash(project_id)) % (2 ** 63),
            vector=embedding,
            payload={
                "project_id": project_id,
                "project_name": project_name,
                "name": project_name,
                "description": description,
                "domain": domain,
                "tech_stack": tech_stack,
                "repos": repos,
                "owner_team": owner_team,
            },
        )],
    )

    print(f"✅ Registered: {project_name} ({project_id}) — {len(repos)} repo(s)")
    return {
        "project_id": project_id,
        "project_name": project_name,
        "description": description,
        "domain": domain,
        "tech_stack": tech_stack,
        "repos": repos,
        "owner_team": owner_team,
    }


def delete_project(project_id: str) -> bool:
    """Remove a project from both PostgreSQL and Qdrant."""
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()

    try:
        get_qdrant().delete(
            collection_name="project_embeddings",
            points_selector=[abs(hash(project_id)) % (2 ** 63)],
        )
    except Exception as e:
        print(f"[registry] Qdrant delete warning: {e}")

    if deleted:
        print(f"🗑  Deleted project: {project_id}")
        return True
    print(f"⚠️  No project found: {project_id}")
    return False


def search_projects(requirement: str, top_k: int = 3) -> list:
    """Semantic search over project_embeddings."""
    query_vector = get_embedder().encode(requirement).tolist()
    results = get_qdrant().query_points(
        collection_name="project_embeddings",
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    out = []
    for r in results.points:
        p = r.payload or {}
        out.append({
            "project_id": p.get("project_id", ""),
            "project_name": p.get("project_name", ""),
            "description": p.get("description", ""),
            "domain": p.get("domain", ""),
            "repos": p.get("repos", []),
            "score": round(r.score, 4),
        })
    return out


def list_all_projects() -> list:
    """List all registered projects (from PostgreSQL — the source of truth)."""
    conn = get_postgres()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT project_id, project_name, description, domain, tech_stack, repos, owner_team
        FROM projects
        ORDER BY project_name
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "project_id": r[0],
            "project_name": r[1],
            "description": r[2],
            "domain": r[3],
            "tech_stack": r[4],
            "repos": r[5],
            "owner_team": r[6],
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────
# Bulk loader — clone + index + register in one flow
# ─────────────────────────────────────────────────────────────────────
def register_from_config(config_path: str, clone: bool = True, index: bool = True) -> list:
    """
    Read projects.yaml (or .json) and register each project.
    Optionally clones repos into WORKSPACE_ROOT and runs the indexer.

    Schema:
        projects:
          - project_id: flask-contacts
            project_name: Flask Contacts API
            description: REST API for managing personal contacts.
            domain: CRM
            tech_stack: [Python, Flask, SQLAlchemy]
            owner_team: Platform Team
            repos:
              - name: flask-contacts-api
                url: https://github.com/owner/flask-contacts-api.git
                type: backend
                branch: main
    """
    config = _load_config(config_path)
    projects = config.get("projects") or []
    if not projects:
        print(f"⚠️  No projects found in {config_path}")
        return []

    # Lazy import so the registry doesn't hard-depend on the workspace module
    if clone:
        try:
            from agents.repo_workspace import ensure_repo_cloned, get_repo_local_path
        except ImportError:
            print("[registry] agents.repo_workspace missing — cloning disabled")
            clone = False

    if index:
        try:
            from knowledge_layer.indexer import index_repo
        except ImportError:
            print("[registry] knowledge_layer.indexer missing — indexing disabled")
            index = False

    results = []
    for proj in projects:
        repo_names = []
        for repo in proj.get("repos", []) or []:
            repo_name = repo.get("name") if isinstance(repo, dict) else repo
            repo_url = repo.get("url") if isinstance(repo, dict) else None
            branch = (repo.get("branch") if isinstance(repo, dict) else None) or "main"
            if not repo_name:
                continue
            repo_names.append(repo_name)

            if clone:
                local = ensure_repo_cloned(repo_name, repo_url=repo_url, branch=branch)
                if not local:
                    print(f"  ⚠️  Skipping index — clone failed for {repo_name}")
                    continue
                if index:
                    try:
                        index_repo(local, repo_name)
                    except Exception as e:
                        print(f"  ⚠️  Index error for {repo_name}: {e}")

        results.append(register_project(
            project_id=proj["project_id"],
            project_name=proj["project_name"],
            description=proj.get("description", ""),
            domain=proj.get("domain", ""),
            tech_stack=proj.get("tech_stack", []) or [],
            repos=repo_names,
            owner_team=proj.get("owner_team", ""),
        ))

    return results


def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML required for YAML configs: pip install pyyaml")
        return yaml.safe_load(text) or {}
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Project Registry CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # register
    p_reg = sub.add_parser("register", help="Register one or many projects")
    p_reg.add_argument("--config", help="Path to projects.yaml or projects.json")
    p_reg.add_argument("--project-id")
    p_reg.add_argument("--name", help="project_name")
    p_reg.add_argument("--description", default="")
    p_reg.add_argument("--domain", default="")
    p_reg.add_argument("--tech-stack", default="", help="Comma-separated, e.g. 'Python,Flask'")
    p_reg.add_argument("--repos", default="", help="Comma-separated repo names")
    p_reg.add_argument("--owner-team", default="")
    p_reg.add_argument("--no-clone", action="store_true", help="Skip cloning repos")
    p_reg.add_argument("--no-index", action="store_true", help="Skip indexing repos")

    # list
    sub.add_parser("list", help="List all registered projects")

    # delete
    p_del = sub.add_parser("delete", help="Delete a project from the registry")
    p_del.add_argument("--project-id", required=True)

    # search
    p_search = sub.add_parser("search", help="Search projects by requirement")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "register":
        if args.config:
            register_from_config(args.config,
                                 clone=not args.no_clone,
                                 index=not args.no_index)
        elif args.project_id and args.name:
            register_project(
                project_id=args.project_id,
                project_name=args.name,
                description=args.description,
                domain=args.domain,
                tech_stack=[s.strip() for s in args.tech_stack.split(",") if s.strip()],
                repos=[s.strip() for s in args.repos.split(",") if s.strip()],
                owner_team=args.owner_team,
            )
        else:
            print("ERROR: provide --config OR (--project-id and --name)", file=sys.stderr)
            sys.exit(2)

    elif args.cmd == "list":
        projects = list_all_projects()
        if not projects:
            print("No projects registered.")
        else:
            print(f"{len(projects)} project(s):")
            for p in projects:
                print(f"  • {p['project_id']:30}  {p['project_name']}  "
                      f"({len(p.get('repos') or [])} repos)")

    elif args.cmd == "delete":
        delete_project(args.project_id)

    elif args.cmd == "search":
        results = search_projects(args.query, top_k=args.top_k)
        if not results:
            print("No matches.")
        for i, c in enumerate(results, 1):
            print(f"  {i}. {c['project_name']:30}  score={c['score']}  "
                  f"repos={','.join(c['repos'])}")


if __name__ == "__main__":
    _cli()