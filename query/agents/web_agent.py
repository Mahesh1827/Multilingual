"""
Web Search Agent
Uses Tavily API (primary) with DuckDuckGo fallback for reliable web search.
Tavily returns pre-extracted content — no need for parallel URL scraping.

Enhancements:
  - Content relevance filtering (removes non-Tirumala noise)
  - TTD source boosting (official sites ranked higher)
  - Rate-limit handling with exponential backoff
  - Cleaner summarization prompt
"""

import logging
import os
import re
import time
import numpy as np

from query.agents.knowledge_rag_agent import reason
from ocr.vector_store import get_embedding_model

logger = logging.getLogger(__name__)

_FALLBACK_MESSAGE = (
    "I wasn't able to find live information for that right now. "
    "You can check the official TTD website (tirupatibalaji.ap.gov.in) "
    "or call 1800-425-1333 for the latest updates. "
    "Feel free to ask me about darshan timings, temple history, or pilgrimage services! "
    "Jai Balaji \U0001f64f"
)

# ──────────────────────────────────────────────
# TTD Source Boosting — official domains get priority
# ──────────────────────────────────────────────
_TRUSTED_DOMAINS = [
    "ttdevasthanams.ap.gov.in",
    "tirumala.org",
    "ttd.ap.gov.in",
    "tirumalainfo.com",
    "wikipedia.org",
]

_TIRUMALA_KEYWORDS = {
    "tirumala", "tirupati", "ttd", "venkateswara", "balaji", "srinivasa",
    "temple", "darshan", "prasad", "laddu", "seva", "brahmotsavam",
    "pilgrimage", "saptagiri", "devasthanam", "hundi", "gopuram",
}


def _is_trusted_source(url: str) -> bool:
    """Check if a URL is from a trusted TTD-related source."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in _TRUSTED_DOMAINS)


def _relevance_score(text: str) -> float:
    """Score how relevant a text chunk is to Tirumala/TTD (0.0 - 1.0)."""
    if not text:
        return 0.0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    matches = words & _TIRUMALA_KEYWORDS
    # Normalize by keyword count (more matches = more relevant)
    return min(len(matches) / 3.0, 1.0)


# ──────────────────────────────────────────────
# Tavily Search (primary) with rate-limit handling
# ──────────────────────────────────────────────

def _search_tavily(query: str, max_results: int = 5) -> dict:
    """Search using the Tavily API. Returns pre-extracted content."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("🔑 [Web Agent] TAVILY_API_KEY not set. Skipping Tavily.")
        return {"text": "", "source_urls": [], "chunks": []}

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
            )

            chunks = []
            source_urls = []

            # Tavily provides a pre-synthesized answer
            if response.get("answer"):
                chunks.append(response["answer"])

            # Also grab individual result content
            for result in response.get("results", []):
                content = result.get("content", "")
                url = result.get("url", "")
                if content:
                    # Boost trusted sources by placing them first
                    if url and _is_trusted_source(url):
                        chunks.insert(0, content)
                    else:
                        chunks.append(content)
                if url:
                    source_urls.append(url)

            combined = "\n\n".join(chunks[:5])
            logger.info(f"🔍 [Tavily] Got {len(chunks)} passages from {len(source_urls)} sources.")
            return {"text": combined, "source_urls": source_urls, "chunks": chunks[:5]}

        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                wait_time = 2 ** attempt
                logger.warning(f"🔍 [Tavily] Rate limited. Retrying in {wait_time}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
                continue
            logger.error(f"🔍 [Tavily] Error: {e}")
            return {"text": "", "source_urls": [], "chunks": []}

    logger.error("🔍 [Tavily] All retries exhausted.")
    return {"text": "", "source_urls": [], "chunks": []}


# ──────────────────────────────────────────────
# DuckDuckGo Fallback
# ──────────────────────────────────────────────

def _search_ddg(query: str, max_results: int = 5) -> dict:
    """Fallback search using DuckDuckGo (no API key needed)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return {"text": "", "source_urls": [], "chunks": []}

        chunks = [res.get("body", "") for res in results if res.get("body")]
        urls = [res.get("href", "") for res in results if res.get("href")]

        combined = "\n\n".join(chunks)
        logger.info(f"🦆 [DDG Fallback] Got {len(chunks)} snippets.")
        return {"text": combined, "source_urls": urls, "chunks": chunks}

    except Exception as e:
        logger.error(f"🦆 [DDG Fallback] Error: {e}")
        return {"text": "", "source_urls": [], "chunks": []}


# ──────────────────────────────────────────────
# Unified search: Tavily → DDG fallback
# ──────────────────────────────────────────────

def search_web_for_content(query: str, max_results: int = 5) -> dict:
    """Try Tavily first, fall back to DuckDuckGo."""
    result = _search_tavily(query, max_results)
    if result.get("text"):
        return result

    logger.info("🔄 [Web Agent] Tavily returned no results. Falling back to DuckDuckGo.")
    return _search_ddg(query, max_results)


# ──────────────────────────────────────────────
# Content Relevance Filtering
# ──────────────────────────────────────────────

def _filter_irrelevant_content(chunks: list[str], min_relevance: float = 0.1) -> list[str]:
    """Remove chunks that have low Tirumala/TTD relevance."""
    if not chunks:
        return chunks

    filtered = []
    for chunk in chunks:
        score = _relevance_score(chunk)
        if score >= min_relevance:
            filtered.append(chunk)
        else:
            logger.debug(f"🗑️ [Web Agent] Filtered irrelevant chunk (relevance={score:.2f})")

    if not filtered and chunks:
        # If everything was filtered, keep the best one
        best = max(chunks, key=_relevance_score)
        filtered = [best]
        logger.info("🗑️ [Web Agent] All chunks filtered — keeping best candidate.")

    return filtered


# ──────────────────────────────────────────────
# Semantic re-ranking of web chunks
# ──────────────────────────────────────────────

def _semantic_rerank_web(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Re-rank web chunks by embedding similarity to query."""
    if not chunks or len(chunks) <= top_k:
        return chunks

    try:
        model = get_embedding_model()
        query_emb = np.array(model.embed_query(query))
        chunk_embs = np.array(model.embed_documents(chunks))
        scores = np.dot(chunk_embs, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        logger.error(f"🧠 [Web RAG] Reranking failed: {e}")
        return chunks[:top_k]


# ──────────────────────────────────────────────
# LangGraph Node
# ──────────────────────────────────────────────

def web_node(state: dict) -> dict:
    """Web Agent with strict persona, relevance filtering, and source boosting."""
    query_en = state.get("query_english", "")
    language = state.get("language", "en")

    q_lower = query_en.lower()
    search_query = query_en

    # Ensure Tirumala/TTD context is in the search
    if "tirumala" not in q_lower and "tirupati" not in q_lower and "ttd" not in q_lower:
        search_query = f"Tirumala TTD {query_en}"

    # For chairman/officer/board queries, add "history list"
    admin_keywords = ["chairman", "president", "eo ", "executive officer", "board", "trustee"]
    if any(kw in q_lower for kw in admin_keywords):
        search_query = f"TTD {query_en} history complete list"

    # For historical questions, push toward historical documents
    historical_markers = ["first", "earliest", "original", "founded", "1930", "1940", "1950", "1960", "1970", "1980"]
    if any(marker in q_lower for marker in historical_markers):
        search_query = f"{search_query} history since 1933"

    # For year-specific queries, quote the year for exact match
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", q_lower)
    if year_match:
        search_query = f'"{year_match.group(1)}" {search_query}'

    logger.info(f"[Web Agent] Final search query: {search_query}")
    try:
        result = search_web_for_content(search_query, max_results=5)
        web_text = result.get("text", "")

        if not web_text:
            return {"final_answer": _FALLBACK_MESSAGE, "agent_route": "web_search"}

        # Filter irrelevant content first
        raw_chunks = result.get("chunks", [])
        relevant_chunks = _filter_irrelevant_content(raw_chunks)

        # HALLUCINATION GUARD: If all chunks were filtered as irrelevant,
        # return the fallback instead of passing empty context to the LLM
        if not relevant_chunks:
            logger.warning("🛡️ [Web Agent] All web results filtered as irrelevant. Returning fallback.")
            return {"final_answer": _FALLBACK_MESSAGE, "agent_route": "web_search"}

        # Then semantic rerank the relevant passages
        best_chunks = _semantic_rerank_web(query_en, relevant_chunks, top_k=3)

        web_chunks = [
            {"text": chunk, "metadata": {"source": "web", "page": "web"}}
            for chunk in best_chunks
        ]

        # Build language directive for the web agent
        lang_names = {
            "te": "Telugu", "hi": "Hindi", "ta": "Tamil",
            "kn": "Kannada", "en": "English",
        }
        lang_name = lang_names.get(language, "English")
        lang_directive = (
            f"LANGUAGE RULE: The user's requested language is {lang_name}. "
            f"You MUST write your entire response strictly using the NATIVE official alphabet/script of the {lang_name} language "
            f"(e.g., Devanagari script for Hindi, Telugu script for Telugu). "
            f"Under NO circumstances should you use Latin/English characters (Romanization/Hinglish) to write {lang_name} words, "
            f"even if the user's original message used English letters. Do NOT answer in English."
            if language != "en" else
            "LANGUAGE RULE: Respond in English."
        )

        original_text = state.get("user_input", query_en)

        # Strict Govinda Persona Prompt — with context-type awareness
        answer = reason(
            query_en,
            web_chunks,
            language,
            original_text=original_text,
            system_prompt=(
                "You are Govinda, a warm, devotion-driven virtual assistant for "
                "Tirumala Tirupati Devasthanams (TTD), answering from live web search results. "
                "You love helping pilgrims and devotees.\n\n"
                f"{lang_directive}\n\n"
                "CONTEXT TYPE AWARENESS (CRITICAL):\n"
                "- Web results may contain operational info (timings, prices, bookings) "
                "or devotional/historical content (legends, scripture).\n"
                "- NEVER use historical content to answer booking/timing questions\n"
                "- NEVER mix mythology with factual operational details\n\n"
                "DOMAIN RESTRICTION (STRICT):\n"
                "- Answer ONLY questions about Tirumala (darshan, accommodation, laddu, history, sevas, festivals, travel, facilities).\n"
                "- If the question is unrelated to Tirumala, say: 'I can only help with Tirumala-related information. Jai Balaji 🙏'\n\n"
                "KNOWLEDGE RULES (ANTI-HALLUCINATION):\n"
                "1. Answer ONLY using facts explicitly stated in the provided web results. Never add your own knowledge.\n"
                "2. Do NOT estimate timings, prices, or availability.\n"
                "3. Keep answers concise and clear — 3 to 5 sentences unless the user asks for a list.\n"
                "4. Start naturally: 'Sure, ...', 'According to the latest information, ...', etc.\n"
                "5. TEMPORAL RULE (CRITICAL): If the user asks about a specific time period (e.g. '1930s', 'first chairman'), "
                "only use information from web results that explicitly mentions that exact period. "
                "If results only contain current info, say: 'I couldn't find specific information for that time period. "
                "You may want to check the official TTD records.'\n"
                "6. Never assume or connect information across different time periods.\n"
                "7. If the results contain a list, present the relevant entries clearly.\n"
                "8. Never add meta-commentary about your rules or how you work."
            ),
            chat_history=state.get("chat_history", [])
        )

        # Append source citations if available
        source_urls = result.get("source_urls", [])
        if source_urls and answer:
            top_urls = source_urls[:3]
            citation_lines = ["\nSources:"]
            for i, url in enumerate(top_urls, 1):
                citation_lines.append(f"  {i}. {url}")
            answer = answer.rstrip() + "\n" + "\n".join(citation_lines)

        return {"final_answer": answer, "agent_route": "web_search", "context_chunks": web_chunks}
    except Exception as e:
        logger.error(f"🌐 [Web Agent] Error: {e}")
        return {"final_answer": _FALLBACK_MESSAGE, "agent_route": "web_search"}