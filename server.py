"""
FastAPI Server for Tirumala Multi-Agent AI Assistant
Wraps the LangGraph pipeline in async HTTP endpoints.

Run:
    uvicorn server:app --reload --port 8000
"""

import _env_setup  # noqa: F401

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from query.agents.pipeline import run_query
from query.agents.knowledge_rag_agent import check_ollama_health

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: warmup check
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    if check_ollama_health():
        logger.info("✅ Ollama is reachable.")
    else:
        logger.warning("⚠️  Ollama is not running. Answers may use raw context.")
    yield


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Tirumala AI Assistant API",
    description="Multi-agent RAG pipeline for Tirumala TTD queries",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    chat_history: list[dict] = Field(default_factory=list, description="Previous conversation turns")


class QueryResponse(BaseModel):
    query_original: str
    query_english: str
    language: str
    agent_route: str
    answer: str
    sources: list = []
    domains: list = []
    verification: dict = {}
    suggestions: list = []
    response_time_s: float


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    ollama_ok = await asyncio.to_thread(check_ollama_health)
    return {
        "status": "healthy",
        "ollama": "connected" if ollama_ok else "disconnected",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Submit a query to the multi-agent RAG pipeline.
    Runs the synchronous pipeline in a thread pool to avoid blocking.
    """
    start = time.time()

    try:
        result = await asyncio.to_thread(
            run_query, req.query, req.chat_history
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - start

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return QueryResponse(
        query_original=result.get("query_original", ""),
        query_english=result.get("query_english", ""),
        language=result.get("language", "en"),
        agent_route=result.get("agent_route", "unknown"),
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        domains=result.get("domains", []),
        verification=result.get("verification", {}),
        suggestions=result.get("suggestions", []),
        response_time_s=round(elapsed, 2),
    )
