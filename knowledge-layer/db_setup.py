"""
Knowledge Layer — Database Setup
Run once to create all tables in PostgreSQL and collections in Qdrant.
"""

import os
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------
# PostgreSQL Setup
# -----------------------------------------

def setup_postgres():
    print("[PostgreSQL] Connecting...")
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        user=os.getenv("POSTGRES_USER", "sdlc"),
        password=os.getenv("POSTGRES_PASSWORD", "sdlc1234"),
        dbname=os.getenv("POSTGRES_DB", "sdlc_knowledge")
    )
    cur = conn.cursor()

    print("[PostgreSQL] Creating tables...")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            symbol_name VARCHAR(255) NOT NULL,
            symbol_type VARCHAR(50) NOT NULL,
            language VARCHAR(50) NOT NULL,
            line_number INTEGER,
            signature TEXT,
            docstring TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS protocol_contracts (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255) NOT NULL,
            contract_type VARCHAR(50) NOT NULL,
            file_path TEXT NOT NULL,
            contract_name VARCHAR(255),
            raw_content TEXT,
            parsed_content JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS repo_maps (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255) UNIQUE NOT NULL,
            repo_path TEXT,
            language VARCHAR(50),
            summary TEXT,
            file_count INTEGER,
            last_indexed TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_coverage (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            test_file_path TEXT,
            coverage_percent FLOAT,
            last_updated TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS change_history (
            id SERIAL PRIMARY KEY,
            repo_name VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            commit_hash VARCHAR(100),
            commit_message TEXT,
            author VARCHAR(255),
            changed_at TIMESTAMP,
            diff_summary TEXT
        );
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_repo ON symbols(repo_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_contracts_repo ON protocol_contracts(repo_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_contracts_type ON protocol_contracts(contract_type);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_repo_maps_name ON repo_maps(repo_name);")

    conn.commit()
    cur.close()
    conn.close()
    print("[PostgreSQL] ✅ All tables created successfully")


# -----------------------------------------
# Qdrant Setup
# -----------------------------------------

def setup_qdrant():
    print("[Qdrant] Connecting...")
    client = QdrantClient(
        url="http://127.0.0.1:6333",
        timeout=60
    )

    print("[Qdrant] Creating collections...")

    # Code embeddings — for semantic search across files and functions
    if not client.collection_exists("code_embeddings"):
        client.create_collection(
            collection_name="code_embeddings",
            vectors_config=VectorParams(
                size=384,  # sentence-transformers/all-MiniLM-L6-v2
                distance=Distance.COSINE
            )
        )
        print("[Qdrant] ✅ Created collection: code_embeddings")
    else:
        print("[Qdrant] ℹ️  Collection already exists: code_embeddings")

    # Protocol contract embeddings — for semantic search across APIs/protos
    if not client.collection_exists("contract_embeddings"):
        client.create_collection(
            collection_name="contract_embeddings",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print("[Qdrant] ✅ Created collection: contract_embeddings")
    else:
        print("[Qdrant] ℹ️  Collection already exists: contract_embeddings")

    # Repo map embeddings — one per repo
    if not client.collection_exists("repo_map_embeddings"):
        client.create_collection(
            collection_name="repo_map_embeddings",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print("[Qdrant] ✅ Created collection: repo_map_embeddings")
    else:
        print("[Qdrant] ℹ️  Collection already exists: repo_map_embeddings")

    print("[Qdrant] ✅ All collections ready")


# -----------------------------------------
# Neo4j Setup
# -----------------------------------------

def setup_neo4j():
    print("[Neo4j] Connecting...")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password1234")
        )
    )

    print("[Neo4j] Creating constraints and indexes...")

    with driver.session() as session:

        # Repo node
        session.run("""
            CREATE CONSTRAINT repo_name_unique IF NOT EXISTS
            FOR (r:Repo) REQUIRE r.name IS UNIQUE
        """)

        # File node
        session.run("""
            CREATE CONSTRAINT file_path_unique IF NOT EXISTS
            FOR (f:File) REQUIRE f.path IS UNIQUE
        """)

        # Symbol node
        session.run("""
            CREATE INDEX symbol_name_index IF NOT EXISTS
            FOR (s:Symbol) ON (s.name)
        """)

        # Contract node
        session.run("""
            CREATE CONSTRAINT contract_unique IF NOT EXISTS
            FOR (c:Contract) REQUIRE c.id IS UNIQUE
        """)

        print("[Neo4j] ✅ Constraints and indexes created")

    driver.close()
    print("[Neo4j] ✅ Setup complete")


# -----------------------------------------
# Run All
# -----------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("SDLC Knowledge Layer — Database Setup")
    print("=" * 50)

    setup_postgres()
    setup_qdrant()
    setup_neo4j()

    print("=" * 50)
    print("✅ All databases ready for indexing")
    print("=" * 50)