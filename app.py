"""
GraphRAG.POC HTTP API server.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /search/stream   — NDJSON stream of thought_step + answer events
    GET  /health          — liveness check

Comparison mode is handled client-side: the frontend fires two simultaneous
POST /search/stream calls with mode="graphrag" and mode="vector".
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
import chromadb

from approaches.graphragapproach import GraphRAGApproach
from approaches.queryrouter import QueryRouter

load_dotenv()

from logging_config import setup_logging
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__)

CHUNK_INDEX = os.environ.get("AZURE_SEARCH_CHUNK_INDEX", "procurement-chunks")
ENTITY_INDEX = os.environ.get("AZURE_SEARCH_ENTITY_INDEX", "procurement-entities")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "./chromadb_data")
CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
EMB_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
EMB_DIMENSIONS = int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1024"))
MAX_HOPS = int(os.environ.get("GRAPHRAG_MAX_HOPS", "3"))
MAX_ENTITIES_PER_HOP = int(os.environ.get("GRAPHRAG_MAX_ENTITIES_PER_HOP", "10"))
ENTITY_TOP_K = int(os.environ.get("GRAPHRAG_ENTITY_TOP_K", "5"))
CHUNK_TOP_K = int(os.environ.get("GRAPHRAG_CHUNK_TOP_K", "15"))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    chunk_col = chroma_client.get_or_create_collection(
        name=CHUNK_INDEX, metadata={"hnsw:space": "cosine"}
    )
    entity_col = chroma_client.get_or_create_collection(
        name=ENTITY_INDEX, metadata={"hnsw:space": "cosine"}
    )
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    _state["approach"] = GraphRAGApproach(
        openai_client=openai_client,
        chat_deployment=CHAT_DEPLOYMENT,
        emb_deployment=EMB_DEPLOYMENT,
        emb_dimensions=EMB_DIMENSIONS,
        chunk_col=chunk_col,
        entity_col=entity_col,
        max_hops=MAX_HOPS,
        max_entities_per_hop=MAX_ENTITIES_PER_HOP,
        entity_top_k=ENTITY_TOP_K,
        chunk_top_k=CHUNK_TOP_K,
    )
    _state["router"] = QueryRouter()
    yield


app = FastAPI(title="GraphRAG POC API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    mode: str = "auto"  # "auto" | "graphrag" | "vector"


@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    approach: GraphRAGApproach = _state["approach"]
    router: QueryRouter = _state["router"]

    effective_mode = req.mode
    if effective_mode == "auto":
        effective_mode = "graphrag" if router.should_use_graphrag(req.query) else "vector"

    logger.info("Query received | mode=%s effective=%s | %r", req.mode, effective_mode, req.query)

    async def generate():
        event_count = 0
        try:
            async for event in approach.run_streaming(req.query, mode=effective_mode):
                yield json.dumps(event).encode() + b"\n"
                event_count += 1
        except Exception:
            logger.exception("Streaming failed for query %r", req.query)
            yield json.dumps({"type": "error", "message": "Internal server error"}).encode() + b"\n"
        finally:
            logger.info("Query complete | mode=%s | %d events | %r", effective_mode, event_count, req.query)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


@app.get("/graph")
async def get_graph():
    entity_col: chromadb.Collection = _state["approach"].entity_col
    results = entity_col.get(include=["metadatas"])

    node_ids: set[str] = set(results["ids"])
    nodes = []
    edges = []

    for eid, meta in zip(results["ids"], results["metadatas"]):
        nodes.append({
            "id": eid,
            "name": meta.get("entity_name", eid),
            "type": meta.get("entity_type", "unknown"),
            "mention_count": meta.get("mention_count", 1),
            "source_files": json.loads(meta.get("source_files", "[]")),
        })
        try:
            related = json.loads(meta.get("related_entities", "[]"))
        except Exception:
            related = []
        for r in related:
            target = r.get("entity_id", "")
            if target and target in node_ids:
                edges.append({
                    "source": eid,
                    "target": target,
                    "relationship_type": r.get("relationship_type", "related_to"),
                })

    return {"nodes": nodes, "edges": edges}


@app.get("/health")
async def health():
    return {"status": "ok"}
