"""
Embedding & Qdrant Vector Store Module (LangChain)
Generates embeddings using the configured multilingual model via LangChain HuggingFaceEmbeddings
and stores them in a local Qdrant vectorstore.
Also supports Hybrid Search (Qdrant + BM25) and reranking.
"""

import json
import logging
import numpy as np
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ocr.config import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DEVICE, 
    VECTOR_STORE_DIR, 
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    QDRANT_PORT
)

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global models
# ──────────────────────────────────────────────
_embed_model = None
_reranker = None
_bm25 = None
_bm25_corpus = None

METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────

def get_embedding_model():
    """Lazy-load embedding model."""
    global _embed_model

    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}")

        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            },
        )

    return _embed_model


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────

def get_reranker():
    """Lazy-load reranker."""
    global _reranker

    if _reranker is None:
        logger.info("Loading reranker model...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")

    return _reranker


# ──────────────────────────────────────────────
# Build Qdrant + BM25
# ──────────────────────────────────────────────

def build_qdrant_index(documents: list[dict]):

    global _bm25, _bm25_corpus

    if not documents:
        logger.error("No documents available for indexing.")
        return None, []

    embeddings = get_embedding_model()

    metadata = [doc["metadata"] for doc in documents]

    lc_docs = [
        Document(
            page_content=doc["text"],
            metadata={**doc["metadata"], "text": doc["text"]},
        )
        for doc in documents
    ]

    logger.info(f"Building Qdrant index (Docker) at {QDRANT_URL}:{QDRANT_PORT}...")
    
    # In server mode, we initialize the client and pass it to from_documents
    client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
    
    vectorstore = QdrantVectorStore.from_documents(
        lc_docs,
        embeddings,
        url=QDRANT_URL,
        port=QDRANT_PORT,
        collection_name=QDRANT_COLLECTION_NAME,
        force_recreate=True,
    )

    logger.info("Qdrant index built successfully")

    # ─────────────────────────
    # Build BM25 index
    # ─────────────────────────

    texts = [doc["text"] for doc in documents]

    tokenized = [t.split() for t in texts]

    _bm25 = BM25Okapi(tokenized)
    _bm25_corpus = documents

    logger.info("BM25 index built")

    return vectorstore, metadata


# ──────────────────────────────────────────────
# Save Qdrant Metadata
# ──────────────────────────────────────────────

def save_qdrant_index(vectorstore, metadata):

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Saved Qdrant database to disk")
    logger.info(f"Saved metadata ({len(metadata)} entries)")


# ──────────────────────────────────────────────
# Load Qdrant
# ──────────────────────────────────────────────

def load_qdrant_index():
    
    embeddings = get_embedding_model()

    logger.info(f"Connecting to Qdrant Docker at {QDRANT_URL}:{QDRANT_PORT}")
    client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Loaded Qdrant index")

    return vectorstore, metadata