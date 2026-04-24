"""
FAQ / Cache Agent — Semantic Matching
--------------------------------------
Multilingual FAQ matching with embedding-based semantic similarity.

Strategy (cascading):
  1. Instant replies for greetings and out-of-scope questions
  2. Semantic embedding match (BGE cosine similarity) against cached Q&A
  3. Fuzzy string fallback (SequenceMatcher) for ASR/spelling errors
  4. If nothing matches → signal the pipeline to proceed to RAG

Cross-Language Support:
  - All FAQ matching is done on the ENGLISH version of the query
  - Incoming queries in Telugu/Hindi/Tamil/Kannada are already translated
    to English by the pipeline before reaching this agent
  - This means "darshan timings enti?" (Telugu) and "darshan ka samay?"
    (Hindi) both match "What are the darshan timings?" in the cache

Confidence Tiers:
  HIGH   (≥ 0.85): Return cached answer directly
  MEDIUM (≥ 0.70): Return best match with gentle clarification
  LOW    (< 0.70): No match — proceed to RAG

Enhancements:
  - Embedding cache: question embeddings stored alongside Q&A in JSON
  - Hit counter: tracks how often each cached answer is served
  - last_accessed: separate from creation timestamp for freshness
  - Frequency-weighted eviction: evicts lowest hit_count entries first
  - get_cache_stats() for monitoring
"""

import json
import logging
import re
import numpy as np
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Cache file locations
# ──────────────────────────────────────────────
_CACHE_FILE      = Path(__file__).resolve().parents[2] / ".cache" / "faq_cache.json"
_EMBEDDINGS_FILE = Path(__file__).resolve().parents[2] / ".cache" / "faq_embeddings.json"

# Import centralized config (with safe fallbacks for standalone testing)
try:
    from query.config import FAQ_CACHE_TTL_DAYS, FAQ_CACHE_MAX_SIZE, TTD_HELPLINE, TTD_WEBSITE, ASSISTANT_NAME
    _CACHE_TTL_DAYS  = FAQ_CACHE_TTL_DAYS
    _CACHE_MAX_SIZE  = FAQ_CACHE_MAX_SIZE
except ImportError:
    _CACHE_TTL_DAYS  = 30
    _CACHE_MAX_SIZE  = 5000
    TTD_HELPLINE     = "1800-425-1333"
    TTD_WEBSITE      = "https://tirupatibalaji.ap.gov.in"
    ASSISTANT_NAME   = "Govinda"

# ──────────────────────────────────────────────
# Similarity thresholds
# ──────────────────────────────────────────────
_SEMANTIC_HIGH   = 0.85   # Direct match — return immediately
_SEMANTIC_MEDIUM = 0.78   # Likely match — return with soft clarification (raised from 0.70 to reduce false positives)
_FUZZY_THRESHOLD = 0.80   # SequenceMatcher fallback threshold

# ──────────────────────────────────────────────
# Greetings / meta — always handled instantly
# ──────────────────────────────────────────────
_GREETINGS = {"hi", "hello", "hey", "namaste"}
_THANKS    = {"thank you", "thanks", "thank u", "dhanyavad"}

# Social / small-talk phrases that must NEVER reach the semantic cache
_SOCIAL_QUERIES = {
    "how are you", "how r u", "how are u", "how r you",
    "how do you do", "how's it going", "how is it going",
    "what's up", "whats up", "wassup",
    "good morning", "good afternoon", "good evening", "good night",
    "are you there", "you there",
}

_META = {
    "who are you":     "I'm an AI assistant for Tirumala Tirupati Devasthanams (TTD), happy to help with all your Tirumala questions!",
    "what can you do": "I can help with darshan timings, sevas, festivals, temple history, pilgrimage guidance, and much more. What would you like to know?",
    "bye":             "Goodbye! Safe travels and Jai Venkateswara!",
    "goodbye":         "Goodbye! Safe travels and Jai Venkateswara!",
}

_SCOPE_KEYWORDS = {
    "tirumala", "tirupati", "ttd", "venkateswara", "balaji", "srinivasa",
    "temple", "darshan", "prasad", "prasadam", "laddu", "pilgrimage",
    "seva", "brahmotsavam", "festival", "puja", "saptagiri", "hills",
    "akasa ganga", "shrine", "vishnu", "devotion", "worship", "scripture",
    "veda", "history", "ticket", "booking", "accommodation", "food",
    "travel", "reach", "weather", "timing", "open", "close", "cost",
    "price", "fee", "kiosk", "chandragiri", "alipiri", "triumala", "tirumla",
    "chairman", "naidu", "board", "eo", "officer", "trustee", "appointment",
    "dress", "code", "rules", "guidelines", "phone", "mobile", "camera",
    "luggage", "locker", "footwear", "shoes", "banned", "prohibited",
    "allowed", "restricted", "queue", "waiting", "time",
}

_LOCATIONS = {
    "tirumala", "tirupati", "tirupathi", "visakhapatnam", "vizag",
    "chennai", "bangalore", "sri city", "sricity", "ahobilam",
    "mahanandi", "srisailam", "nellore", "vijayawada"
}

# 🚨 Master list of failure phrases to NEVER cache
_FAILURE_PHRASES = [
    "does not contain",
    "i can only answer",
    "i can only help with tirumala",
    "not have enough information",
    "not supported by the available",
    "could not find current information",
    "information not mentioned in context",
    "i am sorry",
    "i'm sorry",
    "web search failed",
    "cannot answer",
    "i'm not sure about that",
    "answer is empty or contains only stopwords",
    "contains only stopwords",
    "don't have specific information",
    "please contact ttd directly",
    "encountered an issue",
    "temporary issue processing",
    "i don't have that",
    "specific detail right now",
    "not fully supported by",
]

# Noise words to strip from queries before matching
_NOISE_WORDS = {
    "in", "the", "context", "of", "tirumala", "please", "tell", "me",
    "about", "can", "you", "what", "is", "a", "an",
}


# ──────────────────────────────────────────────
# Query normalization
# ──────────────────────────────────────────────

def _normalize_query(q: str) -> str:
    """
    Normalize a query for better matching:
    - lowercase
    - strip parenthetical context hints added by pipeline
    - remove noise words
    - collapse whitespace
    """
    q = q.lower().strip()
    # Remove the "(in the context of Tirumala)" hint added by _prepare_query
    q = re.sub(r"\(in the context of tirumala\)", "", q)
    # Remove punctuation except apostrophes
    q = re.sub(r"[^\w\s']", " ", q)
    # Remove noise words (but keep if it's a very short query)
    words = q.split()
    if len(words) > 3:
        words = [w for w in words if w not in _NOISE_WORDS]
    return " ".join(words).strip()


# ──────────────────────────────────────────────
# Embedding engine (lazy-loaded, reuses the existing BGE model)
# ──────────────────────────────────────────────
_embed_model = None

def _get_embed_model():
    """Lazy-load the same BGE embedding model used by the RAG vectorstore."""
    global _embed_model
    if _embed_model is None:
        try:
            from ocr.vector_store import get_embedding_model
            _embed_model = get_embedding_model()
            logger.info("📋 [FAQ] Embedding model loaded for semantic matching.")
        except Exception as e:
            logger.warning("📋 [FAQ] Embedding model unavailable: %s. Using fuzzy fallback.", e)
    return _embed_model


def _embed_text(text: str) -> list[float] | None:
    """Embed a single text string. Returns vector or None on failure."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        return model.embed_query(text)
    except Exception as e:
        logger.warning("📋 [FAQ] Embedding failed: %s", e)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _load_cache() -> list[dict]:
    """
    Load cache from disk. Embeddings are stored in a separate file to keep
    faq_cache.json human-readable. On first run after upgrade, existing
    inline embeddings are automatically migrated to the separate file.
    """
    if not _CACHE_FILE.exists():
        return []
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))

        # ── Load embeddings from separate file ──────────────────────────────
        embeddings: dict[str, list] = {}
        if _EMBEDDINGS_FILE.exists():
            try:
                embeddings = json.loads(_EMBEDDINGS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        # ── Migration: extract inline embeddings (old format) ───────────────
        migrated = False
        for entry in data:
            if "embedding" in entry:
                q = entry.get("question", "")
                if q and q not in embeddings:
                    embeddings[q] = entry["embedding"]
                del entry["embedding"]
                migrated = True

        if migrated:
            _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _EMBEDDINGS_FILE.write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")
            logger.info("🗂️ [FAQ Cache] Migrated inline embeddings → faq_embeddings.json")

        # ── Merge embeddings back in-memory for this session ────────────────
        for entry in data:
            q = entry.get("question", "")
            if q in embeddings:
                entry["embedding"] = embeddings[q]

        original_len = len(data)

        # ── SELF-HEALING: Filter bad/expired entries ─────────────────────────
        valid_data = []
        now = datetime.now()
        cutoff = now - timedelta(days=_CACHE_TTL_DAYS)

        for entry in data:
            ans_lower = entry.get("answer", "").lower()
            if any(phrase in ans_lower for phrase in _FAILURE_PHRASES):
                continue  # Remove bad answers

            # TTL eviction
            ts_str = entry.get("timestamp")
            if ts_str:
                try:
                    if datetime.fromisoformat(ts_str) < cutoff:
                        continue  # Expired
                except (ValueError, TypeError):
                    pass
            else:
                entry["timestamp"] = now.isoformat()

            if "hit_count" not in entry:
                entry["hit_count"] = 0
            if "last_accessed" not in entry:
                entry["last_accessed"] = entry.get("timestamp", now.isoformat())

            valid_data.append(entry)

        removed = original_len - len(valid_data)
        if removed > 0:
            logger.info(f"🧹 [FAQ Cache] Evicted {removed} entries (bad answers + TTL expired).")
            _save_cache(valid_data)

        return valid_data
    except Exception:
        return []


def _save_cache(cache: list[dict]) -> None:
    """Save cache to disk. Embeddings are written to a separate file."""
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    embeddings: dict[str, list] = {}
    clean_cache = []
    for entry in cache:
        q = entry.get("question", "")
        emb = entry.get("embedding")
        if emb is not None and q:
            embeddings[q] = emb
        # Strip embedding from the main JSON entry
        clean_entry = {k: v for k, v in entry.items() if k != "embedding"}
        clean_cache.append(clean_entry)

    _CACHE_FILE.write_text(
        json.dumps(clean_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if embeddings:
        _EMBEDDINGS_FILE.write_text(
            json.dumps(embeddings, ensure_ascii=False), encoding="utf-8"
        )


def _extract_year(text: str) -> str | None:
    """Extracts a 4-digit year from text."""
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group(1) if match else None


# ──────────────────────────────────────────────
# Semantic + Fuzzy lookup
# ──────────────────────────────────────────────

def _fuzzy_similarity(a: str, b: str) -> float:
    """Character-level fuzzy similarity with temporal and location penalties."""
    a_clean = _normalize_query(a)
    b_clean = _normalize_query(b)

    # TEMPORAL PENALTY: Years must match exactly
    year_a = _extract_year(a_clean)
    year_b = _extract_year(b_clean)
    if year_a != year_b:
        return 0.0

    # EXTREME PENALTY: "Maximum" vs "Current" should not match
    extremes = {"maximum", "minimum", "highest", "lowest", "top", "bottom"}
    extreme_a = [ex for ex in extremes if ex in a_clean]
    extreme_b = [ex for ex in extremes if ex in b_clean]
    if extreme_a != extreme_b:
        return 0.0

    # LOCATION MISMATCH PENALTY
    locs_a = {loc for loc in _LOCATIONS if loc in a_clean}
    locs_b = {loc for loc in _LOCATIONS if loc in b_clean}
    if locs_a != locs_b:
        return 0.0

    return SequenceMatcher(None, a_clean, b_clean).ratio()


def lookup_cache(question: str, language: str = "en") -> tuple[str | None, float]:
    """
    Find the best matching cached answer using semantic + fuzzy similarity.
    Translates the cached answer to the user's requested language at retrieval time.

    Args:
        question: English version of the user's question (used for embedding match).
        language: ISO 639-1 code of the user's language for the response.

    Returns:
        (answer_in_user_language, confidence) — answer is None if no match found.
    """
    cache = _load_cache()
    if not cache:
        return None, 0.0

    normalized_q = _normalize_query(question)
    query_embedding = _embed_text(normalized_q)

    best_score = 0.0
    best_answer = None
    best_idx = -1
    best_method = "none"

    for i, entry in enumerate(cache):
        cached_q = entry.get("question", "")

        # ── Strategy 1: Semantic embedding similarity ──
        if query_embedding is not None:
            cached_embedding = entry.get("embedding")
            if cached_embedding is not None and len(cached_embedding) == len(query_embedding):
                sem_score = _cosine_similarity(query_embedding, cached_embedding)

                # Apply temporal penalty to semantic scores too
                year_q = _extract_year(normalized_q)
                year_c = _extract_year(cached_q.lower())
                if year_q != year_c and (year_q or year_c):
                    sem_score *= 0.3  # Heavy penalty for year mismatch

                if sem_score > best_score:
                    best_score = sem_score
                    best_answer = entry["answer"]
                    best_idx = i
                    best_method = "semantic"

        # ── Strategy 2: Fuzzy string fallback ──
        fuzzy_score = _fuzzy_similarity(question, cached_q)
        if fuzzy_score > best_score:
            best_score = fuzzy_score
            best_answer = entry["answer"]
            best_idx = i
            best_method = "fuzzy"

    # ── Confidence gating ──
    # Before accepting, verify the INTENT of the query matches the cached question.
    # This prevents "dress code to visit Tirumala" from matching "where is Tirumala"
    # just because both mention "Tirumala".
    if best_score >= _SEMANTIC_MEDIUM and best_idx >= 0:
        # Extract intent-carrying words (strip common noise + location names)
        _intent_noise = _NOISE_WORDS | _LOCATIONS | {"how", "where", "when", "do", "does", "did", "are", "was", "were", "to", "for", "and", "or", "it", "its", "i", "my", "we", "our", "visit", "go", "know", "want", "need"}
        q_intent_words = {w for w in normalized_q.split() if w not in _intent_noise and len(w) > 2}
        cached_q_norm = _normalize_query(cache[best_idx].get("question", ""))
        c_intent_words = {w for w in cached_q_norm.split() if w not in _intent_noise and len(w) > 2}

        # If both queries have intent words but ZERO overlap, the match is a false positive
        if q_intent_words and c_intent_words and not (q_intent_words & c_intent_words):
            # Only reject if score is below HIGH threshold (very high scores are likely genuine)
            if best_score < _SEMANTIC_HIGH:
                logger.info(
                    "📋 [FAQ Cache] REJECTED false-positive: score=%.3f but intent mismatch. "
                    "query_intent=%s vs cached_intent=%s",
                    best_score, q_intent_words, c_intent_words,
                )
                return None, best_score
        # Increment hit counter
        cache[best_idx]["hit_count"] = cache[best_idx].get("hit_count", 0) + 1
        cache[best_idx]["last_accessed"] = datetime.now().isoformat()
        _save_cache(cache)

        hit_count = cache[best_idx]["hit_count"]
        logger.info(
            "📋 [FAQ Cache] Hit! method=%s, score=%.3f, hit_count=%d, q=%r",
            best_method, best_score, hit_count, cache[best_idx]["question"][:60],
        )

        # ── Translate cached English answer to user's language at retrieval time ──
        cached_lang = cache[best_idx].get("language", "en")
        if language != "en" and cached_lang != language:
            try:
                from query.voice_pipeline import IndicTranslator
                tr = IndicTranslator()
                translated = tr.english_to_indic(best_answer, language)
                if translated and translated.strip() != best_answer.strip():
                    logger.info("📋 [FAQ Cache] Translated cached answer [en→%s].", language)
                    return translated, best_score
            except Exception as e:
                logger.warning("📋 [FAQ Cache] Translation of cached answer failed: %s", e)

        return best_answer, best_score

    if best_score > 0:
        logger.info(
            "📋 [FAQ Cache] Near-miss: method=%s, score=%.3f (threshold=%.2f)",
            best_method, best_score, _SEMANTIC_MEDIUM,
        )

    return None, best_score


def save_to_cache(question: str, answer: str, language: str = "en") -> None:
    """
    Save a Q&A pair to the cache.

    IMPORTANT: Always store the ENGLISH answer. If the answer is in another
    language, translate it to English before storing so that any user language
    can retrieve and translate it correctly at lookup time.
    """
    # ── Ensure we store an English answer ──
    if language != "en":
        try:
            from query.voice_pipeline import IndicTranslator
            tr = IndicTranslator()
            en_answer = tr.english_to_indic(answer, "en")  # wrong direction, fix:
            # Translate from user's language → English
            en_answer = tr.indic_to_english(answer, language)
            if en_answer and en_answer.strip() and not getattr(tr, "_last_translation_failed", False):
                logger.info("📋 [FAQ Cache] Storing English version of %s answer.", language)
                answer = en_answer
                language = "en"
        except Exception as e:
            logger.warning("📋 [FAQ Cache] Could not translate answer to English for caching: %s", e)
            # Store as-is with the detected language tag so retrieval can handle it

    # SAFEGUARD: Do not cache failure or fallback messages
    ans_lower = answer.lower()
    if any(phrase in ans_lower for phrase in _FAILURE_PHRASES):
        logger.info("📋 [FAQ Cache] BLOCKED: Refusing to cache a fallback/failure response.")
        return

    # SAFEGUARD: Do not cache non-English text when language tag is "en".
    # This prevents romanized Hindi/Telugu answers from polluting the cache
    # and false-matching unrelated queries.
    if language == "en":
        non_ascii = sum(1 for c in answer if ord(c) > 127)
        if len(answer) > 0 and non_ascii / len(answer) > 0.15:
            logger.info("📋 [FAQ Cache] BLOCKED: Answer tagged 'en' but contains %.0f%% non-ASCII chars.", non_ascii / len(answer) * 100)
            return
        # Also block romanized Indic (Latin-script Hindi/Telugu) if question contains pipeline context hint
        if "(in the context of tirumala)" in question.lower():
            _romanized_markers = {"aapko", "kaise", "hoga", "chahiye", "karna", "bata", "wahan", "milengi", "uplabdh"}
            answer_words = set(re.findall(r'\b[a-z]+\b', ans_lower))
            if len(answer_words & _romanized_markers) >= 3:
                logger.info("📋 [FAQ Cache] BLOCKED: Answer appears to be romanized Indic text.")
                return

    cache = _load_cache()
    now_iso = datetime.now().isoformat()

    # Normalize and embed the question
    normalized_q = _normalize_query(question)
    embedding = _embed_text(normalized_q)
    # Tag the stored language so retrieval knows what language is in the cache
    _entry_language = language

    # Check if a very similar question already exists (update it)
    if embedding is not None:
        for entry in cache:
            cached_emb = entry.get("embedding")
            if cached_emb is not None:
                sim = _cosine_similarity(embedding, cached_emb)
                if sim >= _SEMANTIC_HIGH:
                    entry["answer"] = answer  # refresh answer
                    entry["timestamp"] = now_iso
                    entry["last_accessed"] = now_iso
                    # Don't reset hit_count — it accumulates
                    _save_cache(cache)
                    logger.info("📋 [FAQ Cache] Updated existing entry (semantic match=%.3f).", sim)
                    return
    else:
        # Fallback: fuzzy check
        for entry in cache:
            if _fuzzy_similarity(question, entry["question"]) >= _FUZZY_THRESHOLD:
                entry["answer"] = answer
                entry["timestamp"] = now_iso
                entry["last_accessed"] = now_iso
                _save_cache(cache)
                logger.info("📋 [FAQ Cache] Updated existing entry (fuzzy match).")
                return

    # New entry
    new_entry = {
        "question": question.strip(),
        "answer": answer,
        "language": _entry_language,      # language the answer is stored in
        "timestamp": now_iso,
        "last_accessed": now_iso,
        "hit_count": 0,
    }
    if embedding is not None:
        new_entry["embedding"] = embedding

    cache.append(new_entry)

    # Frequency-weighted eviction: evict entries with lowest hit_count first,
    # but protect entries created in the last 24 hours (grace period).
    if len(cache) > _CACHE_MAX_SIZE:
        now = datetime.now()
        grace_cutoff = (now - timedelta(hours=24)).isoformat()

        # Partition: protected (recent) vs evictable
        protected = [e for e in cache if e.get("timestamp", "") >= grace_cutoff]
        evictable = [e for e in cache if e.get("timestamp", "") < grace_cutoff]

        # Sort evictable by hit_count (ascending), then last_accessed (oldest first)
        evictable.sort(key=lambda e: (e.get("hit_count", 0), e.get("last_accessed", "")))

        need_to_remove = len(cache) - _CACHE_MAX_SIZE
        evictable = evictable[min(need_to_remove, len(evictable)):]
        cache = evictable + protected
        logger.info(f"📋 [FAQ Cache] Eviction: removed {need_to_remove} low-value entries (protected {len(protected)} recent).")

    _save_cache(cache)
    logger.info(f"📋 [FAQ Cache] Saved new entry. Cache size: {len(cache)}")


# ──────────────────────────────────────────────
# Cache Stats (for monitoring / debugging)
# ──────────────────────────────────────────────

def get_cache_stats() -> dict:
    """Return statistics about the FAQ cache."""
    cache = _load_cache()
    if not cache:
        return {"size": 0, "total_hits": 0, "top_questions": [], "embedded_count": 0}

    total_hits = sum(e.get("hit_count", 0) for e in cache)
    embedded_count = sum(1 for e in cache if e.get("embedding"))

    top = sorted(cache, key=lambda e: e.get("hit_count", 0), reverse=True)[:5]
    top_questions = [
        {"question": e["question"][:80], "hit_count": e.get("hit_count", 0)}
        for e in top
    ]

    return {
        "size": len(cache),
        "total_hits": total_hits,
        "avg_hits": round(total_hits / len(cache), 1) if cache else 0,
        "embedded_count": embedded_count,
        "top_questions": top_questions,
    }


def get_cache_health() -> dict:
    """Return production health metrics for the FAQ cache.

    Metrics:
        staleness_ratio: fraction of entries older than half the TTL
        embedding_coverage: fraction of entries with embeddings
        eviction_pressure: how close to max capacity (0.0 = empty, 1.0 = full)
        health_grade: A/B/C/D overall assessment
    """
    cache = _load_cache()
    if not cache:
        return {
            "health_grade": "A",
            "staleness_ratio": 0.0,
            "embedding_coverage": 1.0,
            "eviction_pressure": 0.0,
            "size": 0,
        }

    now = datetime.now()
    half_ttl = timedelta(days=_CACHE_TTL_DAYS / 2)
    stale_count = 0
    embedded_count = 0

    for entry in cache:
        ts_str = entry.get("timestamp")
        if ts_str:
            try:
                if datetime.fromisoformat(ts_str) < (now - half_ttl):
                    stale_count += 1
            except (ValueError, TypeError):
                stale_count += 1
        if entry.get("embedding"):
            embedded_count += 1

    staleness = stale_count / len(cache)
    coverage = embedded_count / len(cache)
    pressure = len(cache) / _CACHE_MAX_SIZE

    # Grade: A (healthy), B (watch), C (degraded), D (critical)
    if staleness < 0.2 and coverage > 0.8 and pressure < 0.7:
        grade = "A"
    elif staleness < 0.4 and coverage > 0.5:
        grade = "B"
    elif staleness < 0.6 or coverage < 0.3:
        grade = "C"
    else:
        grade = "D"

    return {
        "health_grade": grade,
        "staleness_ratio": round(staleness, 3),
        "embedding_coverage": round(coverage, 3),
        "eviction_pressure": round(pressure, 3),
        "size": len(cache),
    }


# ──────────────────────────────────────────────
# FAQ Agent logic
# ──────────────────────────────────────────────

def faq_agent(query: str, language: str = "en") -> str | None:
    """
    Returns:
        str  — an instant answer (greeting / meta / cached / out-of-scope)
        None — query is Tirumala-related but not cached → go to RAG

    Args:
        query:    English version of the user's question.
        language: ISO 639-1 code the response should be in.
    """
    logger.info("📋 [FAQ Agent] Checking cache and greetings...")
    q = query.lower().strip()

    # Time-based greeting
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        time_greeting = "Good morning"
    elif 12 <= hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    # 1. Greetings
    if any(word in q.split() for word in _GREETINGS):
        return (
            f"Jai Balaji 🙏 {time_greeting}! Welcome, devotee.\n"
            "I am Govinda, your Tirumala assistant. How can I serve you today?\n"
            "Feel free to ask me about darshan, sevas, festivals, temple history, "
            "accommodation, or travel to Tirumala."
        )

    # 1b. Social / small-talk ("how are you", "what's up", etc.)
    #     Must be checked BEFORE the semantic cache — these have zero Tirumala
    #     relevance and would otherwise false-match temple answers at ~0.70 similarity.
    q_stripped = re.sub(r"[^\w\s]", "", q).strip()  # strip punctuation for matching
    if q_stripped in _SOCIAL_QUERIES or any(q_stripped.startswith(s) for s in _SOCIAL_QUERIES):
        return (
            f"{time_greeting}! I am Govinda, your Tirumala AI assistant. 🙏\n"
            "I'm doing great and ready to help you!\n"
            "You can ask me about darshan timings, sevas, festivals, temple history, "
            "accommodation, or travel to Tirumala. What would you like to know?"
        )

    # 2. Thanks
    if any(phrase in q for phrase in _THANKS):
        return (
            "You're most welcome! Jai Balaji 🙏\n"
            "Is there anything else I can help you with about Tirumala?"
        )

    # 3. Meta questions
    for phrase, response in _META.items():
        if phrase in q:
            return response

    # 4. Semantic cache lookup (pass language so answer is returned in user's language)
    cached_answer, confidence = lookup_cache(query, language=language)
    if cached_answer is not None:
        if confidence >= _SEMANTIC_HIGH:
            # HIGH confidence — return directly (feels natural, no "from FAQ")
            return cached_answer
        else:
            # MEDIUM confidence — return with a gentle nudge
            return (
                f"{cached_answer}\n\n"
                "(If this doesn't fully answer your question, feel free to rephrase it!)"
            )

    # 5. Out-of-scope filter
    has_scope = any(kw in q for kw in _SCOPE_KEYWORDS)
    if not has_scope:
        logger.info(f"[FAQ Agent] Out of scope: '{query}'")
        return (
            "I can only help with Tirumala-related information. "
            "Feel free to ask me about darshan timings, temple history, sevas, festivals, "
            "accommodation, or pilgrimage guidance! Jai Balaji 🙏"
        )

    # 6. In-scope but not cached → let RAG handle it
    return None



# ──────────────────────────────────────────────
# LangGraph node
# ──────────────────────────────────────────────

def faq_node(state: dict) -> dict:
    query    = state.get("query_english", "")
    language = state.get("language", "en")
    answer   = faq_agent(query, language=language)
    if answer is not None:
        return {"final_answer": answer, "cache_hit": True}
    return {"cache_hit": False}