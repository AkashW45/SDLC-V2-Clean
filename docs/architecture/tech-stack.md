# Tech Stack

## Fixed Core (Do Not Change)

| Component | Technology | Version |
|-----------|-----------|---------|
| Agent Orchestration | LangGraph | latest |
| LLM | DeepSeek V4 Pro | `deepseek-v4-pro` (1M context, thinking mode) |
| LLM Client SDK | OpenAI Python SDK | latest |
| Vector DB | Qdrant | latest |
| Graph DB | Neo4j | 5.18 |
| Relational DB | PostgreSQL | 16 |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 (384-dim) |
| Backend API | FastAPI | latest |
| Dashboard | Vanilla HTML + Mermaid.js 10 | — |
| Workflow Integration | n8n self-hosted | latest |
| Version Control | GitHub | via REST API |
| Project Management | Jira Cloud | via REST API v3 |
| Document Export | openpyxl + markdown | latest |

## DeepSeek V4 Pro Configuration

```python
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[...],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=8192
)
```

Used in: Discovery, Planning, Impact Risk Assessment, Codegen, Validation, Deployment

## Languages Supported

| Language | Status | Indexer | Codegen |
|----------|--------|---------|---------|
| Python | ✅ Production | AST module | DeepSeek V4 Pro |
| Java | 🟡 Planned | Tree-sitter | OpenRewrite codemods |
| C# | 🟡 Planned | Tree-sitter | Roslyn CodeFix |
| TypeScript | 🟡 Planned | Tree-sitter | ts-morph |

## Communication Protocols Indexed

| Protocol | File Types | Parser |
|----------|-----------|--------|
| REST | OpenAPI/Swagger YAML/JSON | yaml + json |
| gRPC | .proto | regex extractor |
| Async messaging | AsyncAPI YAML | yaml |
| Avro | .avsc | json |

## Persistence Schema

| Table | Purpose |
|-------|---------|
| `projects` | Project registry (id, name, description, repos, tech stack) |
| `repo_maps` | Indexed repo metadata |
| `symbols` | AST-extracted symbols (classes, functions, methods) |
| `protocol_contracts` | API contracts (OpenAPI, proto, AsyncAPI) |
| `pipelines` | Active + historical pipeline state |
| `audit_log` | Every pipeline event with actor + timestamp |
| `change_history` | Git commit history per file |
| `test_coverage` | Coverage metrics per file |

## Qdrant Collections

| Collection | Vector | Payload |
|-----------|--------|---------|
| `project_embeddings` | 384-dim | project_id, name, description, repos |
| `code_embeddings` | 384-dim | repo_name, file_path, symbol_name, type |
| `contract_embeddings` | 384-dim | repo_name, file_path, contract_type |
| `repo_map_embeddings` | 384-dim | repo_name, summary |

## Environment Variables

```env
DEEPSEEK_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
GITHUB_REPO_OWNER=AkashW45

JIRA_EMAIL=...@wissen.com
JIRA_API_TOKEN=...
JIRA_BASE_URL=wissen-team-tqg48t7b.atlassian.net
JIRA_PROJECT_KEY=DEV

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_USER=sdlc
POSTGRES_PASSWORD=sdlc1234
POSTGRES_DB=sdlc_knowledge

QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password1234

REPO_PATH=C:\path\to\local\repo  # for existing project codegen
```