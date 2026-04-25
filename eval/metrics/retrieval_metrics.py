"""
Retrieval Metrics — Hybrid Semantic + Lexical
----------------------------------------------
Precision@K, Recall@K, Hit Rate, MRR, NDCG@K

Uses a HYBRID approach for determining chunk relevance:
  1. Semantic similarity (primary) — works across all languages
  2. Lexical token overlap (secondary) — Unicode-aware, works for Indic scripts

A chunk is considered relevant if EITHER method scores above its threshold.
"""

import math
import re
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "are", "was", "were", "be", "has", "have", "had",
    "this", "that", "at", "by", "from", "as", "not", "but", "so", "if",
    "i", "you", "he", "she", "we", "they", "its", "can", "will", "do",
}

LEXICAL_RELEVANCE_THRESHOLD = 0.12   # Min token overlap to consider relevant (lexical)
SEMANTIC_RELEVANCE_THRESHOLD = 0.45  # Min semantic similarity to consider relevant


def _tokenize(text: str) -> set[str]:
    """
    Extract meaningful tokens from text — supports both Latin and Unicode scripts.
    Uses Unicode word boundaries to handle Indic scripts (Telugu, Hindi, Tamil, Kannada).
    """
    # Extract all word-like tokens (Unicode-aware: letters, digits, 3+ chars)
    tokens = set(re.findall(r'\b\w{3,}\b', text.lower(), re.UNICODE))
    return tokens - _STOPWORDS


# ──────────────────────────────────────────────
# Semantic similarity for chunk relevance
# ──────────────────────────────────────────────
_semantic_model = None


def _get_semantic_model():
    """Lazy-load sentence-transformer for semantic chunk relevance."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer("intfloat/multilingual-e5-large")
            logger.info("📊 [Retrieval Metrics] Semantic model loaded.")
        except Exception as e:
            logger.warning("📊 [Retrieval Metrics] Semantic model unavailable: %s", e)
    return _semantic_model


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity using sentence-transformers."""
    model = _get_semantic_model()
    if model is None:
        return 0.0
    try:
        import numpy as np
        embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
        return float(max(0.0, min(1.0, np.dot(embeddings[0], embeddings[1]))))
    except Exception:
        return 0.0


def _chunk_is_relevant(
    chunk_text: str,
    expected_answer: str,
    keywords: list[str] | None = None,
) -> bool:
    """
    Determine if a retrieved chunk is relevant to the expected answer.

    Uses HYBRID approach:
      1. Semantic similarity (primary) — works across all languages
      2. Lexical token overlap (secondary) — catches exact keyword matches

    A chunk is relevant if EITHER method scores above its threshold.
    """
    if not chunk_text.strip() or not expected_answer.strip():
        return False

    # ── Method 1: Semantic similarity ──
    sem_score = _semantic_similarity(chunk_text[:500], expected_answer)
    if sem_score >= SEMANTIC_RELEVANCE_THRESHOLD:
        return True

    # ── Method 2: Lexical token overlap (Unicode-aware) ──
    expected_tokens = _tokenize(expected_answer)
    if keywords:
        expected_tokens |= {k.lower() for k in keywords if len(k) >= 3}

    chunk_tokens = _tokenize(chunk_text)

    if not expected_tokens or not chunk_tokens:
        return sem_score >= (SEMANTIC_RELEVANCE_THRESHOLD * 0.8)  # Relaxed fallback

    overlap = len(chunk_tokens & expected_tokens) / len(expected_tokens)
    return overlap >= LEXICAL_RELEVANCE_THRESHOLD


def _get_relevance_labels(
    chunks: list[dict],
    expected_answer: str,
    keywords: list[str] | None = None,
) -> list[bool]:
    """Return a list of booleans indicating relevance for each chunk."""
    return [
        _chunk_is_relevant(
            chunk.get("text", ""),
            expected_answer,
            keywords,
        )
        for chunk in chunks
    ]


# ──────────────────────────────────────────────
# Retrieval Metrics
# ──────────────────────────────────────────────

def precision_at_k(relevance_labels: list[bool], k: int | None = None) -> float:
    """
    Precision@K: fraction of top-K retrieved chunks that are relevant.

    Args:
        relevance_labels: Ordered list of booleans (True = relevant).
        k: Number of top results to consider. Defaults to len(labels).
    """
    if not relevance_labels:
        return 0.0
    k = k or len(relevance_labels)
    top_k = relevance_labels[:k]
    return sum(top_k) / len(top_k)


def recall_at_k(relevance_labels: list[bool], total_relevant: int, k: int | None = None) -> float:
    """
    Recall@K: fraction of all relevant chunks that appear in top-K.

    Args:
        relevance_labels: Ordered list of booleans.
        total_relevant: Total number of relevant chunks in the entire corpus.
        k: Number of top results to consider.
    """
    if total_relevant == 0:
        return 0.0
    k = k or len(relevance_labels)
    retrieved_relevant = sum(relevance_labels[:k])
    return retrieved_relevant / total_relevant


def hit_rate(relevance_labels: list[bool]) -> float:
    """
    Hit Rate: 1.0 if at least one relevant chunk was retrieved, else 0.0.
    """
    return 1.0 if any(relevance_labels) else 0.0


def mean_reciprocal_rank(relevance_labels: list[bool]) -> float:
    """
    MRR: 1 / (position of first relevant result).
    Returns 0.0 if no relevant result found.
    """
    for i, is_rel in enumerate(relevance_labels):
        if is_rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevance_labels: list[bool], k: int | None = None) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Uses binary relevance (1 or 0).
    """
    if not relevance_labels:
        return 0.0

    k = k or len(relevance_labels)
    labels = relevance_labels[:k]

    # DCG
    dcg = sum(
        (1.0 if rel else 0.0) / math.log2(i + 2)
        for i, rel in enumerate(labels)
    )

    # Ideal DCG: all relevant items first
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum(
        (1.0 if rel else 0.0) / math.log2(i + 2)
        for i, rel in enumerate(ideal_labels)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_retrieval_metrics(
    chunks: list[dict],
    expected_answer: str,
    keywords: list[str] | None = None,
    k: int = 5,
) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Args:
        chunks: Retrieved chunks (each with 'text' key).
        expected_answer: The golden expected answer.
        keywords: Optional extra keywords to determine relevance.
        k: Top-K cutoff.

    Returns:
        Dict with precision_at_k, recall_at_k, hit_rate, mrr, ndcg_at_k.
    """
    labels = _get_relevance_labels(chunks, expected_answer, keywords)
    total_relevant = max(sum(labels), 1)  # At least 1 to avoid div/0 in recall

    return {
        "precision_at_k": round(precision_at_k(labels, k), 4),
        "recall_at_k":    round(recall_at_k(labels, total_relevant, k), 4),
        "hit_rate":        round(hit_rate(labels), 4),
        "mrr":             round(mean_reciprocal_rank(labels), 4),
        "ndcg_at_k":       round(ndcg_at_k(labels, k), 4),
        "relevant_count":  sum(labels),
        "retrieved_count": len(chunks),
    }
