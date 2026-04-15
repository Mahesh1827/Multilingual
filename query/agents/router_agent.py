"""
Router Agent — Embedding-based Domain Routing
Maps a user query to one or more domain agents using cosine similarity
against pre-computed domain description embeddings.

Replaces the old keyword-scoring approach which misrouted ambiguous queries.
"""

import logging
import threading
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

_embedding_lock = threading.Lock()

# ──────────────────────────────────────────────
# Domain descriptions (one sentence each, designed for embedding)
# ──────────────────────────────────────────────
_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "about_tirumala": (
        "General information about Tirumala hills, geography, seven hills, "
        "Saptagiri, Akasa Ganga, climate, altitude, population, and the "
        "spiritual significance of the region"
    ),
    "devotional_practice": (
        "Hindu devotional practices at Tirumala including pujas, archana, "
        "bhajans, fasting, vratam, offerings, prasadam distribution, "
        "sevas, aarti ceremonies, and worship rituals"
    ),
    "festival_events": (
        "Festivals and annual events at Tirumala TTD including Brahmotsavam, "
        "Rathotsavam, Vaikunta Ekadasi, Kalyanotsavam, Teppotsavam, "
        "and temple celebrations and processions"
    ),
    "knowledge_scripture": (
        "Vedic scriptures, Upanishads, Puranas, Bhagavata, Vishnu Sahasranama, "
        "Venkateswara Suprabhatam, stotras, mantras, and theological teachings "
        "related to Lord Venkateswara"
    ),
    "temple_history": (
        "History and heritage of Sri Venkateswara Temple including ancient origins, "
        "legends, mythology, dynastic contributions from Pallava Chola Vijayanagara, "
        "inscriptions, gopuram, vimana, and temple architecture"
    ),
    "pilgrimage_seva": (
        "Practical pilgrimage guidance including darshan ticket booking, virtual queue, "
        "special darshan types, seva booking procedures, TTD accommodation and lodges, "
        "laddu prasadam, dress code, entry rules, and how to visit Tirumala"
    ),
}

# Similarity threshold — domains scoring below this are excluded
_SIMILARITY_THRESHOLD = 0.35

# ──────────────────────────────────────────────
# Embedding cache (computed once on first call)
# ──────────────────────────────────────────────
_domain_embeddings: dict[str, np.ndarray] | None = None
_domain_keys: list[str] = []

def _ensure_domain_embeddings():
    """Lazy-load and cache domain description embeddings (thread-safe)."""
    global _domain_embeddings, _domain_keys
    if _domain_embeddings is not None:
        return

    with _embedding_lock:
        # Double-check after acquiring lock
        if _domain_embeddings is not None:
            return

        try:
            from ocr.vector_store import get_embedding_model
            model = get_embedding_model()

            _domain_keys = list(_DOMAIN_DESCRIPTIONS.keys())
            descriptions = [_DOMAIN_DESCRIPTIONS[k] for k in _domain_keys]

            raw = model.embed_documents(descriptions)
            _domain_embeddings = {
                k: np.array(emb) for k, emb in zip(_domain_keys, raw)
            }
            logger.info(f"Router: pre-computed embeddings for {len(_domain_keys)} domains.")
        except Exception as e:
            logger.error(f"Router: failed to load embeddings — falling back to all domains. {e}")
            _domain_embeddings = {}


def route_query(query_text: str) -> tuple[str, list[str]]:
    """
    Route a query to domain agent(s) using embedding similarity.

    Returns:
        ("knowledge_rag", [list of matched domain keys sorted by relevance])
    """
    _ensure_domain_embeddings()

    if not _domain_embeddings:
        # Fallback: no embeddings available — route to all domains
        logger.warning("Router → Embedding unavailable. Routing to all domains.")
        return "knowledge_rag", []

    try:
        from ocr.vector_store import get_embedding_model
        model = get_embedding_model()

        query_emb = np.array(model.embed_query(query_text))

        # Cosine similarity against each domain
        scores: dict[str, float] = {}
        for domain_key, domain_emb in _domain_embeddings.items():
            sim = float(np.dot(query_emb, domain_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(domain_emb) + 1e-9
            ))
            scores[domain_key] = sim

        # Filter and rank
        matched = {k: v for k, v in scores.items() if v >= _SIMILARITY_THRESHOLD}

        if matched:
            ranked = sorted(matched.keys(), key=lambda d: matched[d], reverse=True)
            logger.info(
                f"Router → Embedding match: {ranked} "
                f"(scores: {{{', '.join(f'{k}: {scores[k]:.3f}' for k in ranked)}}})"
            )
            return "knowledge_rag", ranked

        # Nothing above threshold — return top 2 by score instead of all domains
        ranked_all = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        top_n = ranked_all[:2]
        logger.info(
            f"Router → No domain above threshold ({_SIMILARITY_THRESHOLD}). "
            f"Using top-2: {top_n} "
            f"(scores: {{{', '.join(f'{k}: {scores[k]:.3f}' for k in top_n)}}})"
        )
        return "knowledge_rag", top_n

    except Exception as e:
        logger.error(f"Router → Embedding query failed: {e}. Routing to all domains.")
        return "knowledge_rag", []


def get_primary_domain(query_text: str) -> str:
    """Return the single best matching domain for a query."""
    _, domains = route_query(query_text)
    return domains[0] if domains else "about_tirumala"
