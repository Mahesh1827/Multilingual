"""
Embedding & FAISS Vector Store Module (LangChain)
Generates embeddings using BGE via LangChain HuggingFaceEmbeddings
and stores them in a LangChain FAISS vectorstore.
Also supports Hybrid Search (FAISS + BM25) and reranking.
"""

import json
import logging
import numpy as np
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ocr.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, VECTOR_STORE_DIR

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

FAISS_STORE_DIR = str(VECTOR_STORE_DIR)
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
# Build FAISS + BM25
# ──────────────────────────────────────────────

def build_faiss_index(documents: list[dict]):

    global _bm25, _bm25_corpus

    if not documents:
        logger.error("No documents available for FAISS indexing.")
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

    logger.info(f"Building FAISS index for {len(lc_docs)} chunks...")
    vectorstore = FAISS.from_documents(lc_docs, embeddings)

    logger.info(f"FAISS index built: {vectorstore.index.ntotal} vectors")

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
# Save FAISS
# ──────────────────────────────────────────────

def save_faiss_index(vectorstore, metadata):

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(FAISS_STORE_DIR)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved FAISS index ({vectorstore.index.ntotal} vectors)")
    logger.info(f"Saved metadata ({len(metadata)} entries)")


# ──────────────────────────────────────────────
# Load FAISS
# ──────────────────────────────────────────────

def load_faiss_index():

    index_file = Path(FAISS_STORE_DIR) / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found at: {FAISS_STORE_DIR}")

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Loaded FAISS index: {vectorstore.index.ntotal} vectors")

    return vectorstore, metadata


# ──────────────────────────────────────────────
# Hybrid Search + Reranking
# ──────────────────────────────────────────────

def search_faiss(query: str, top_k: int = 5):

    vectorstore, _ = load_faiss_index()

    # BGE query instruction
    instruction = "Represent this sentence for searching relevant passages: "
    query_embed = instruction + query

    # ─────────────────────────
    # FAISS semantic search
    # ─────────────────────────

    faiss_results = vectorstore.similarity_search_with_score(query_embed, k=10)

    docs = [doc for doc, _ in faiss_results]

    # ─────────────────────────
    # BM25 keyword search
    # ─────────────────────────

    if _bm25 is not None:

        tokenized_query = query.split()

        scores = _bm25.get_scores(tokenized_query)

        top_idx = np.argsort(scores)[::-1][:5]

        for idx in top_idx:
            doc = _bm25_corpus[idx]

            docs.append(
                Document(
                    page_content=doc["text"],
                    metadata=doc["metadata"],
                )
            )

    # remove duplicates
    seen = set()
    unique_docs = []

    for d in docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    # ─────────────────────────
    # Reranking
    # ─────────────────────────

    reranker = get_reranker()

    pairs = [(query, d.page_content) for d in unique_docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(unique_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc, score in ranked[:top_k]:

        results.append({
            "score": float(score),
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    return results