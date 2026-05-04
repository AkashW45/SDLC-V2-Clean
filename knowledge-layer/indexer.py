"""
Knowledge Layer — Repo Indexer
Indexes a repo's AST symbols, dependency graph, embeddings, and protocol contracts
into PostgreSQL, Neo4j, and Qdrant.

Usage:
    python knowledge-layer/indexer.py --repo-path /path/to/repo --repo-name my-repo
"""

import os
import ast
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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

# Load embedding model once
print("[Indexer] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[Indexer] Embedding model ready")


# -----------------------------------------
# File Discovery
# -----------------------------------------

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript"
}

PROTOCOL_EXTENSIONS = {
    ".proto": "grpc",
    ".yaml": "openapi_or_asyncapi",
    ".yml": "openapi_or_asyncapi",
    ".json": "openapi_or_schema",
    ".avsc": "avro"
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", ".idea", ".vscode", "target", "bin", "obj"
}


def discover_files(repo_path: str) -> tuple:
    """Returns (source_files, protocol_files)"""
    source_files = []
    protocol_files = []
    repo = Path(repo_path)

    for path in repo.rglob("*"):
        # Skip hidden and build dirs
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue

        if path.is_file():
            ext = path.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                source_files.append(path)
            elif ext in PROTOCOL_EXTENSIONS:
                protocol_files.append(path)

    return source_files, protocol_files


# -----------------------------------------
# Python AST Parser
# -----------------------------------------

def parse_python_file(file_path: Path) -> list:
    """Extract symbols from a Python file using AST."""
    symbols = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class
                docstring = ast.get_docstring(node) or ""
                symbols.append({
                    "name": node.name,
                    "type": "class",
                    "line": node.lineno,
                    "signature": f"class {node.name}",
                    "docstring": docstring,
                    "content": f"class {node.name}: {docstring}"
                })

                # Extract methods inside class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [a.arg for a in item.args.args]
                        method_doc = ast.get_docstring(item) or ""
                        symbols.append({
                            "name": f"{node.name}.{item.name}",
                            "type": "method",
                            "line": item.lineno,
                            "signature": f"def {item.name}({', '.join(args)})",
                            "docstring": method_doc,
                            "content": f"{node.name}.{item.name}({', '.join(args)}): {method_doc}"
                        })

            elif isinstance(node, ast.FunctionDef):
                # Top-level functions only
                args = [a.arg for a in node.args.args]
                docstring = ast.get_docstring(node) or ""

                # Detect decorators (FastAPI routes etc)
                decorators = []
                for dec in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(dec))
                    except Exception:
                        pass

                symbols.append({
                    "name": node.name,
                    "type": "function",
                    "line": node.lineno,
                    "signature": f"def {node.name}({', '.join(args)})",
                    "docstring": docstring,
                    "decorators": decorators,
                    "content": f"def {node.name}({', '.join(args)}): {docstring}"
                })

    except SyntaxError as e:
        print(f"  [!] Syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"  [!] Error parsing {file_path}: {e}")

    return symbols


def extract_imports(file_path: Path) -> list:
    """Extract import dependencies from a Python file."""
    imports = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)

    except Exception:
        pass

    return imports


# -----------------------------------------
# Protocol Contract Parser
# -----------------------------------------

def parse_protocol_contract(file_path: Path) -> dict:
    """Parse a protocol contract file and extract key info."""
    ext = file_path.suffix.lower()
    content = ""

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    result = {
        "file_path": str(file_path),
        "contract_type": PROTOCOL_EXTENSIONS.get(ext, "unknown"),
        "contract_name": file_path.stem,
        "raw_content": content[:5000],  # cap at 5000 chars
        "parsed_content": {}
    }

    # Detect OpenAPI
    if ext in (".yaml", ".yml", ".json"):
        try:
            import yaml
            data = yaml.safe_load(content) if ext in (".yaml", ".yml") else json.loads(content)
            if isinstance(data, dict):
                if "openapi" in data or "swagger" in data:
                    result["contract_type"] = "openapi"
                    result["parsed_content"] = {
                        "title": data.get("info", {}).get("title", ""),
                        "version": data.get("info", {}).get("version", ""),
                        "paths": list(data.get("paths", {}).keys())
                    }
                elif "asyncapi" in data:
                    result["contract_type"] = "asyncapi"
                    result["parsed_content"] = {
                        "title": data.get("info", {}).get("title", ""),
                        "channels": list(data.get("channels", {}).keys())
                    }
        except Exception:
            pass

    # Detect proto
    if ext == ".proto":
        result["contract_type"] = "grpc"
        # Extract service and message names
        services = []
        messages = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("service "):
                services.append(line.split("{")[0].replace("service ", "").strip())
            elif line.startswith("message "):
                messages.append(line.split("{")[0].replace("message ", "").strip())
        result["parsed_content"] = {
            "services": services,
            "messages": messages
        }

    return result


# -----------------------------------------
# PostgreSQL Indexing
# -----------------------------------------

def index_symbols_postgres(conn, repo_name: str, file_path: str, symbols: list, language: str):
    cur = conn.cursor()

    # Delete existing symbols for this file
    cur.execute(
        "DELETE FROM symbols WHERE repo_name = %s AND file_path = %s",
        (repo_name, file_path)
    )

    for sym in symbols:
        cur.execute("""
            INSERT INTO symbols 
            (repo_name, file_path, symbol_name, symbol_type, language, line_number, signature, docstring)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            repo_name,
            file_path,
            sym["name"],
            sym["type"],
            language,
            sym.get("line", 0),
            sym.get("signature", ""),
            sym.get("docstring", "")
        ))

    conn.commit()
    cur.close()


def index_contract_postgres(conn, repo_name: str, contract: dict):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO protocol_contracts
        (repo_name, contract_type, file_path, contract_name, raw_content, parsed_content)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        repo_name,
        contract["contract_type"],
        contract["file_path"],
        contract["contract_name"],
        contract["raw_content"],
        json.dumps(contract["parsed_content"])
    ))

    conn.commit()
    cur.close()


def update_repo_map_postgres(conn, repo_name: str, repo_path: str, language: str, file_count: int):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO repo_maps (repo_name, repo_path, language, file_count, last_indexed)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (repo_name) DO UPDATE
        SET repo_path = EXCLUDED.repo_path,
            language = EXCLUDED.language,
            file_count = EXCLUDED.file_count,
            last_indexed = NOW()
    """, (repo_name, repo_path, language, file_count))

    conn.commit()
    cur.close()


# -----------------------------------------
# Qdrant Indexing
# -----------------------------------------

def index_embeddings_qdrant(qdrant, repo_name: str, file_path: str, symbols: list):
    if not symbols:
        return

    points = []
    for sym in symbols:
        text = sym.get("content", sym["name"])
        if not text.strip():
            continue

        embedding = embedder.encode(text).tolist()

        # Create deterministic ID from repo+file+symbol
        uid = hashlib.md5(
            f"{repo_name}:{file_path}:{sym['name']}".encode()
        ).hexdigest()
        point_id = int(uid[:8], 16)  # use first 8 hex chars as int

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "repo_name": repo_name,
                "file_path": file_path,
                "symbol_name": sym["name"],
                "symbol_type": sym["type"],
                "signature": sym.get("signature", ""),
                "line": sym.get("line", 0)
            }
        ))

    if points:
        qdrant.upsert(
            collection_name="code_embeddings",
            points=points
        )


def index_contract_qdrant(qdrant, repo_name: str, contract: dict):
    text = f"{contract['contract_type']} {contract['contract_name']} {contract['raw_content'][:500]}"
    embedding = embedder.encode(text).tolist()

    uid = hashlib.md5(
        f"{repo_name}:{contract['file_path']}".encode()
    ).hexdigest()
    point_id = int(uid[:8], 16)

    qdrant.upsert(
        collection_name="contract_embeddings",
        points=[PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "repo_name": repo_name,
                "file_path": contract["file_path"],
                "contract_type": contract["contract_type"],
                "contract_name": contract["contract_name"],
                "parsed_content": contract["parsed_content"]
            }
        )]
    )


# -----------------------------------------
# Neo4j Indexing
# -----------------------------------------

def index_dependency_graph(driver, repo_name: str, file_path: str, imports: list, symbols: list):
    with driver.session() as session:

        # Create repo node
        session.run("""
            MERGE (r:Repo {name: $repo_name})
            SET r.last_indexed = datetime()
        """, repo_name=repo_name)

        # Create file node
        session.run("""
            MERGE (f:File {path: $file_path})
            SET f.repo_name = $repo_name,
                f.last_indexed = datetime()
            WITH f
            MATCH (r:Repo {name: $repo_name})
            MERGE (r)-[:CONTAINS]->(f)
        """, file_path=file_path, repo_name=repo_name)

        # Create symbol nodes and link to file
        for sym in symbols:
            session.run("""
                MERGE (s:Symbol {id: $symbol_id})
                SET s.name = $name,
                    s.type = $type,
                    s.repo_name = $repo_name,
                    s.file_path = $file_path,
                    s.line = $line
                WITH s
                MATCH (f:File {path: $file_path})
                MERGE (f)-[:DEFINES]->(s)
            """,
                symbol_id=f"{repo_name}:{file_path}:{sym['name']}",
                name=sym["name"],
                type=sym["type"],
                repo_name=repo_name,
                file_path=file_path,
                line=sym.get("line", 0)
            )

        # Create import relationships
        for imp in imports:
            session.run("""
                MATCH (f:File {path: $file_path})
                MERGE (dep:Module {name: $import_name})
                MERGE (f)-[:IMPORTS]->(dep)
            """, file_path=file_path, import_name=imp)


# -----------------------------------------
# Main Indexer
# -----------------------------------------

def index_repo(repo_path: str, repo_name: str):
    print(f"\n{'='*50}")
    print(f"Indexing repo: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"{'='*50}")

    # Connect to all stores
    conn = get_postgres()
    qdrant = get_qdrant()
    neo4j_driver = get_neo4j()

    # Discover files
    source_files, protocol_files = discover_files(repo_path)
    print(f"\n[Discovery] Found {len(source_files)} source files")
    print(f"[Discovery] Found {len(protocol_files)} protocol contract files")

    # Detect primary language
    lang_counts = {}
    for f in source_files:
        lang = SUPPORTED_EXTENSIONS.get(f.suffix.lower(), "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    primary_language = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"
    print(f"[Discovery] Primary language: {primary_language}")

    # Index source files
    print(f"\n[Indexing] Processing source files...")
    indexed_count = 0

    for file_path in source_files:
        rel_path = str(file_path.relative_to(repo_path))
        lang = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown")

        # Parse symbols
        symbols = []
        imports = []

        if lang == "python":
            symbols = parse_python_file(file_path)
            imports = extract_imports(file_path)
        # Future: add java, csharp, typescript parsers here

        if symbols:
            # Index in PostgreSQL
            index_symbols_postgres(conn, repo_name, rel_path, symbols, lang)

            # Index embeddings in Qdrant
            index_embeddings_qdrant(qdrant, repo_name, rel_path, symbols)

            # Index dependency graph in Neo4j
            index_dependency_graph(neo4j_driver, repo_name, rel_path, imports, symbols)

            indexed_count += 1
            print(f"  ✅ {rel_path} — {len(symbols)} symbols")
        else:
            print(f"  ⚪ {rel_path} — no symbols found")

    # Index protocol contracts
    print(f"\n[Indexing] Processing protocol contracts...")
    contract_count = 0

    for file_path in protocol_files:
        rel_path = str(file_path.relative_to(repo_path))
        contract = parse_protocol_contract(file_path)

        if contract:
            index_contract_postgres(conn, repo_name, contract)
            index_contract_qdrant(qdrant, repo_name, contract)
            contract_count += 1
            print(f"  ✅ {rel_path} — {contract['contract_type']}")

    # Update repo map
    update_repo_map_postgres(
        conn, repo_name, repo_path, primary_language,
        len(source_files) + len(protocol_files)
    )

    # Close connections
    conn.close()
    neo4j_driver.close()

    print(f"\n{'='*50}")
    print(f"✅ Indexing complete for {repo_name}")
    print(f"   Source files indexed: {indexed_count}/{len(source_files)}")
    print(f"   Protocol contracts indexed: {contract_count}")
    print(f"{'='*50}\n")


# -----------------------------------------
# CLI Entry Point
# -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDLC Knowledge Layer Indexer")
    parser.add_argument("--repo-path", required=True, help="Path to the repo to index")
    parser.add_argument("--repo-name", required=True, help="Name of the repo")
    args = parser.parse_args()

    index_repo(args.repo_path, args.repo_name)