import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SDLC Automation Platform V2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0",
        "services": {
            "qdrant": "localhost:6333",
            "neo4j": "localhost:7474",
            "postgres": "localhost:5433"
        }
    }