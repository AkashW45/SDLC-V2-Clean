import os
import subprocess
import requests
import json
import psycopg2
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
# SentenceTransformer is provided via the shared singleton (core/embeddings.py).
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")

if not GITHUB_TOKEN or not GITHUB_OWNER:
    print("❌ ERROR: GITHUB_TOKEN and GITHUB_REPO_OWNER must be set in .env")
    exit(1)

# ---------------------------------------------------------
# The 12-Project Enterprise Portfolio (Depths 1 to 5)
# ---------------------------------------------------------
PORTFOLIO =[
    # ================= DEPTH 1 (Toys / Static) =================
    {
        "project_id": "static-landing-page",
        "project_name": "Corporate Landing Page",
        "description": "Simple HTML5 static marketing page. No backend, no database.",
        "domain": "Marketing",
        "tech_stack": ["HTML", "CSS", "JavaScript"],
        "owner_team": "Marketing",
        "repos": [{"name": "html5-boilerplate", "source": "https://github.com/h5bp/html5-boilerplate.git", "type": "frontend"}]
    },
    {
        "project_id": "python-cli-tool",
        "project_name": "DevOps CLI Utilities",
        "description": "A simple Python package for local devops scripts. Runs in memory.",
        "domain": "DevOps",
        "tech_stack": ["Python"],
        "owner_team": "Platform",
        "repos": [{"name": "sample-python-cli", "source": "https://github.com/pypa/sampleproject.git", "type": "service"}]
    },

    # ================= DEPTH 2 (Basic / Local Data) =================
    {
        "project_id": "vanilla-todo-app",
        "project_name": "Task Manager (Local)",
        "description": "Basic ToDo app storing data in browser localStorage. No integrations.",
        "domain": "Productivity",
        "tech_stack": ["JavaScript"],
        "owner_team": "Internal Tools",
        "repos": [{"name": "vanillajs-todo", "source": "https://github.com/tastejs/todomvc.git", "type": "frontend"}]
    },
    {
        "project_id": "basic-flask-api",
        "project_name": "Contacts REST API",
        "description": "Simple REST API using SQLite for storing contacts.",
        "domain": "CRM",
        "tech_stack": ["Python", "Flask"],
        "owner_team": "Sales",
        "repos": [{"name": "flask-contacts-api", "source": "https://github.com/miguelgrinberg/flasky.git", "type": "backend"}]
    },

    # ================= DEPTH 3 (Auth / Multi-Role) =================
    {
        "project_id": "hr-leave-mgmt",
        "project_name": "Leave Management System",
        "description": "Tracks employee vacation and sick days. Requires JWT auth and manager roles.",
        "domain": "HR",
        "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
        "owner_team": "HR Tech",
        "repos": [{"name": "leave-mgmt-core", "source": "https://github.com/tiangolo/full-stack-fastapi-template.git", "type": "backend"}]
    },
    {
        "project_id": "legacy-petclinic",
        "project_name": "Spring PetClinic Monolith",
        "description": "Standard monolithic web app with Spring MVC, handling veterinarians, owners, and pet records.",
        "domain": "Healthcare",
        "tech_stack": ["Java", "Spring Boot", "MySQL"],
        "owner_team": "Legacy Support",
        "repos": [{"name": "spring-petclinic-monolith", "source": "https://github.com/spring-projects/spring-petclinic.git", "type": "backend"}]
    },

    # ================= DEPTH 4 (Production SaaS / Multi-Repo) =================
    {
        "project_id": "social-blog-saas",
        "project_name": "Conduit Social Publishing",
        "description": "Production-ready social blogging platform with comments, favorites, and user profiles. Distinct frontend and backend.",
        "domain": "Social Media",
        "tech_stack": ["Python", "Django", "React", "Redux"],
        "owner_team": "Product Squad Alpha",
        "repos":[
            {"name": "conduit-django-api", "source": "https://github.com/gothinkster/django-realworld-example-app.git", "type": "backend"},
            {"name": "conduit-react-ui", "source": "https://github.com/gothinkster/react-redux-realworld-example-app.git", "type": "frontend"}
        ]
    },
    {
        "project_id": "realtime-chat-node",
        "project_name": "Customer Support Chat",
        "description": "Realtime customer support chat backend with event-driven architecture.",
        "domain": "Customer Success",
        "tech_stack": ["TypeScript", "NestJS", "WebSockets"],
        "owner_team": "Support Tech",
        "repos": [{"name": "nestjs-chat-backend", "source": "https://github.com/lujakob/nestjs-realworld-example-app.git", "type": "backend"}]
    },
    {
        "project_id": "inventory-tracker",
        "project_name": "Global Inventory Tracker",
        "description": "High-throughput inventory tracking system for global warehouses.",
        "domain": "Logistics",
        "tech_stack": ["Go", "Gin"],
        "owner_team": "Supply Chain",
        "repos": [{"name": "golang-inventory-api", "source": "https://github.com/gothinkster/golang-gin-realworld-example-app.git", "type": "backend"}]
    },

    # ================= DEPTH 5 (Enterprise / Compliance / RPC) =================
    {
        "project_id": "secure-banking-ledger",
        "project_name": "Core Banking Ledger",
        "description": "Highly secure financial transaction ledger. PCI-DSS compliant. Strict audit trails and role-based access.",
        "domain": "Finance (PCI-DSS)",
        "tech_stack": ["C#", ".NET Core"],
        "owner_team": "FinTech Core",
        "repos": [{"name": "csharp-ledger-api", "source": "https://github.com/gothinkster/aspnetcore-realworld-example-app.git", "type": "backend"}]
    },
    {
        "project_id": "telehealth-portal",
        "project_name": "Telehealth Patient Portal",
        "description": "Patient portal handling medical records and prescriptions. Strict HIPAA compliance required. SOC2 audited.",
        "domain": "Healthcare (HIPAA)",
        "tech_stack": ["Java", "Spring Boot", "React"],
        "owner_team": "Health Engineering",
        "repos": [{"name": "java-telehealth-api", "source": "https://github.com/gothinkster/spring-boot-realworld-example-app.git", "type": "backend"}]
    },
    {
        "project_id": "gcp-microservices",
        "project_name": "Cloud Boutique Microservices",
        "description": "Massive scale, multi-region distributed e-commerce architecture. Communicates entirely over gRPC using protobuf contracts.",
        "domain": "Retail (Multi-Region)",
        "tech_stack": ["Go", "Python", "C#", "gRPC", "Kubernetes"],
        "owner_team": "Platform Architecture",
        "repos": [{"name": "boutique-microservices", "source": "https://github.com/GoogleCloudPlatform/microservices-demo.git", "type": "service"}]
    }
]

# ---------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------
print("Loading AI Embedder...")
from core.embeddings import get_embedder
embedder = get_embedder()

def get_postgres():
    # Pooled connection — .close() returns it to the pool.
    from core.db_clients import PooledConn
    return PooledConn()

def create_github_repo(repo_name: str) -> str:
    url = f"https://api.github.com/user/repos"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    check = requests.get(f"https://api.github.com/repos/{GITHUB_OWNER}/{repo_name}", headers=headers)
    if check.status_code == 200:
        return check.json()["clone_url"]
    resp = requests.post(url, headers=headers, json={"name": repo_name, "private": False})
    if resp.status_code == 201:
        return resp.json()["clone_url"]
    raise Exception(f"GitHub Error: {resp.text}")

def run_cmd(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_setup():
    base_dir = Path(os.getcwd()) / "repos"
    base_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("🚀 PROVISIONING 12-PROJECT ENTERPRISE PORTFOLIO")
    print("="*70)

    for proj in PORTFOLIO:
        print(f"\n[Project] {proj['project_name']} | Domain: {proj['domain']}")
        registered_repos =[]

        for repo in proj["repos"]:
            repo_name = repo["name"]
            source_url = repo["source"]
            local_path = base_dir / repo_name

            my_github_url = create_github_repo(repo_name)

            if not local_path.exists():
                print(f"  📥 Cloning {repo_name}...")
                run_cmd(["git", "clone", source_url, str(local_path)])
                print(f"  🚀 Pushing to {GITHUB_OWNER}/{repo_name}...")
                run_cmd(["git", "remote", "remove", "origin"], cwd=str(local_path))
                auth_url = my_github_url.replace("https://", f"https://{GITHUB_TOKEN}@")
                run_cmd(["git", "remote", "add", "origin", auth_url], cwd=str(local_path))
                try:
                    run_cmd(["git", "push", "-u", "origin", "master"], cwd=str(local_path))
                except:
                    run_cmd(["git", "push", "-u", "origin", "main"], cwd=str(local_path))
            else:
                print(f"  📂 {repo_name} already exists locally.")

            registered_repos.append({"name": repo_name, "type": repo["type"], "url": my_github_url, "exists": True})

            print(f"  🧠 Indexing AST & Protocols for {repo_name}...")
            try:
                subprocess.run(["python", "knowledge-layer/indexer.py", "--repo-path", str(local_path), "--repo-name", repo_name], check=True)
            except Exception as e:
                print(f"  ⚠️ Indexing warned/failed, but continuing... {e}")

        # Register Project
        conn = get_postgres()
        cur = conn.cursor()
        cur.execute("""
                    INSERT INTO projects (project_id, project_name, description, domain, tech_stack, repos, owner_team)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_id) DO UPDATE
                                                        SET project_name = EXCLUDED.project_name, description = EXCLUDED.description,
                                                        domain = EXCLUDED.domain, tech_stack = EXCLUDED.tech_stack, repos = EXCLUDED.repos
                    """, (
                        proj["project_id"], proj["project_name"], proj["description"],
                        proj["domain"], json.dumps(proj["tech_stack"]), json.dumps(registered_repos), proj["owner_team"]
                    ))
        conn.commit()
        conn.close()

        text_for_embedding = f"{proj['project_name']}. {proj['description']}. Domain: {proj['domain']}."
        from core.db_clients import qdrant_client as qdrant
        qdrant.upsert(collection_name="project_embeddings", points=[PointStruct(
            id=hash(proj["project_id"]) % (2**63),
            vector=embedder.encode(text_for_embedding).tolist(),
            payload={"project_id": proj["project_id"], "project_name": proj["project_name"], "description": proj["description"], "repos": registered_repos}
        )])
        print(f"  ✅ Registered Project in DB & Qdrant.")

    print("\n" + "="*70)
    print("🎉 PORTFOLIO PROVISIONED SUCCESSFULLY!")
    print("Your platform is now loaded with 12 enterprise projects spanning Depths 1 to 5.")
    print("="*70)

if __name__ == "__main__":
    run_setup()