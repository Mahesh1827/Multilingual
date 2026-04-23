"""
Master Pipeline Orchestrator — LangGraph
Cascade: FAQ Cache → Knowledge RAG → Web Search

Routing Rules:
  - Live / real-time query      → skip FAQ + RAG → direct Web
  - Temporal (specific year)    → skip FAQ → try RAG → fallback Web
  - Conversational (greeting)   → FAQ handles instantly
  - Factual query               → FAQ → RAG → fallback Web

Enhancements:
  - LLM-powered intent classification (with keyword fast-path)
  - @safe_agent_call error handling on every agent node
  - Contextual follow-up suggestions generator
"""

import csv
import logging
import re
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START

from query.agents.faq_agent import faq_node, save_to_cache
from query.agents.web_agent import web_node
from query.agents.knowledge_rag_agent import rag_node
from query.agents.error_handling import safe_agent_call
from query.config import PIPELINE_TIMEOUT_S, MAX_QUERY_LENGTH, APP_VERSION

logger = logging.getLogger(__name__)
_CSV_LOG_FILE = Path(__file__).resolve().parents[2] / ".cache" / "query_log.csv"

# ─────────────────────────────────────────────────────────────
# Shared state schema
# ─────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    # Input
    user_input:     str
    query_text:     str
    language:       str
    query_english:  str
    chat_history:   list[dict]

    # Validation (set by _validate_query_node)
    validation_status:  str    # "VALID" | "NEEDS_CORRECTION" | "INVALID" | ""
    validation_reason:  str    # short explanation when not VALID

    # Intent flags (set by _analyze_query_intent)
    is_live:        bool   # real-time data needed → Web
    is_temporal:    bool   # specific year → skip FAQ, try RAG first
    intent:         str    # "factual" | "real_time" | "conversational"

    # Routing signals
    cache_hit:          bool
    rag_found:          bool
    needs_web_fallback: bool
    web_retry_count:    int

    # Agent outputs
    agent_route:    str
    final_answer:   str
    context_chunks: list
    domains:        list
    verification:   dict
    suggestions:    list    # follow-up suggestions
    error:          Optional[str]
    timing_ms:      dict    # per-node timing metrics


# ─────────────────────────────────────────────────────────────
# Pre-processing nodes
# ─────────────────────────────────────────────────────────────

# Strips leading greeting words Whisper adds when translating Indic audio
# (e.g. "Namaste how many hills..." → "how many hills...").
_GREETING_PREFIX = re.compile(
    r"^(namaste|hi|hello|hey|good morning|good evening)[!.,\s]+",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# Input Validation Node — catches bad ASR before agents run
# ─────────────────────────────────────────────────────────────

@safe_agent_call("validation_agent", fallback_state={"validation_status": "VALID", "validation_reason": ""})
def _validate_query_node(state: QueryState) -> dict:
    """
    Validates the transcribed English query.
    - VALID:            proceed normally
    - NEEDS_CORRECTION: replace query_english with LLM-corrected version
    - INVALID:          build a dynamic clarification message with native-language
                        suggestions of likely intent, then short-circuit to response
    """
    from query.agents.validation_agent import validate_and_correct, build_clarification_message

    q      = state["query_english"]
    result = validate_and_correct(q)
    status = result["status"]

    if status == "NEEDS_CORRECTION" and result["corrected_query"]:
        logger.info(
            "🔧 [Validation] Corrected: %r → %r",
            q[:60], result["corrected_query"][:60],
        )
        # Still show the user what was understood, in case the correction is wrong
        suggestions = result.get("suggestions", [])
        correction_note = ""
        if suggestions:
            correction_note = (
                f"\n(I understood your question as: \"{result['corrected_query']}\" "
                "— let me know if that's not right.)"
            )
        return {
            "query_english":      result["corrected_query"],
            "validation_status":  status,
            "validation_reason":  result["reason"],
            "_correction_note":   correction_note,
        }

    if status == "INVALID":
        logger.warning("⚠️ [Validation] Invalid query: %r — %s", q[:60], result["reason"])
        clarification = build_clarification_message(
            reason=result.get("reason", ""),
            suggestions=result.get("suggestions", []),
        )
        return {
            "validation_status": status,
            "validation_reason": result["reason"],
            "final_answer":      clarification,
            "agent_route":       "validation_error",
        }

    return {"validation_status": "VALID", "validation_reason": ""}


def _after_validation(state: QueryState) -> str:
    """Short-circuit to response on invalid query; proceed normally otherwise."""
    if state.get("validation_status") == "INVALID":
        logger.info("🔀 [Router] Invalid query → short-circuit to response")
        return "response_aggregator"
    return "analyze_intent"


def _prepare_query(state: QueryState) -> dict:
    query = state["user_input"].strip()

    # Strip leading greeting words that Whisper often prepends when translating
    # Indic audio (e.g. "Namaste how many hills..." → "how many hills...").
    # We keep the original query for display; only sanitise the routed version.
    query_english = _GREETING_PREFIX.sub("", query).strip() if query else query

    # Implicit Contextualization:
    context_words = ["tirumala", "tirupati", "venkateswara", "ttd", "balaji", "temple", "god", "swamy"]
    if query_english and not any(word in query_english.lower() for word in context_words):
        query_english = f"{query_english} (in the context of Tirumala)"
        logger.info(f"Contextualized implicit query to: {query_english}")

    # Preserve language already set in state (e.g. from voice pipeline / STT)
    lang = state.get("language") or "en"

    return {
        "query_text":    query,
        "query_english": query_english,
        "language":      lang,
    }


# ─────────────────────────────────────────────────────────────
# Enhanced Intent Classification (keyword fast-path + LLM)
# ─────────────────────────────────────────────────────────────

# Keyword patterns for fast-path (no LLM call needed)
_LIVE_PATTERNS = [
    # Forward order: temporal marker before weather/temperature
    r"\b(current|today|now|live|right now|at the moment|latest)\b.*\b(weather|temp(?:erature)?|status|forecast|update)\b",
    # Reverse order: temperature/weather before temporal marker (e.g. "What is the temperature now?")
    r"\b(weather|temp(?:erature)?)\b.*\b(current|today|now|live|right now|at the moment|latest)\b",
    # Standalone temperature — always a real-time query in Tirumala context
    r"\btemp(?:erature)?\b",
    r"\b(weather forecast|current weather|today.*weather|weather today)\b",
]
_GREETING_PATTERNS = [
    # Must be a standalone greeting — ends immediately after optional punctuation.
    # "Namaste how many hills..." will NOT match; "Namaste" alone WILL match.
    r"^(hi|hello|hey|namaste|good morning|good evening|howdy)[!.,\s]*$",
    r"^(thank you|thanks|thank u|dhanyavad|bye|goodbye)[!.,\s]*$",
    r"^(who are you|what can you do|help)[!?,\s]*$",
]

# Factual fast-path: question patterns that are clearly informational
_FACTUAL_PATTERNS = [
    r"^(what|how|where|when|who|which|why|tell me|explain|list|name)\b",
    r"\b(history|timings?|cost|price|fee|rules?|dress code|height|altitude|distance)\b",
    r"\b(book(?:ing)?|ticket|accommodation|lodge|seva|darshan|prasadam|laddu)\b",
    r"\b(festival|brahmotsavam|vaikunta|scripture|stotra|mantra|legend)\b",
]

def _classify_intent_fast(query: str) -> Optional[str]:
    """Fast keyword-based intent classifier. Returns None if ambiguous."""
    q = query.lower().strip()

    # Check greetings / conversational
    for pat in _GREETING_PATTERNS:
        if re.search(pat, q):
            return "conversational"

    # Check real-time / live
    for pat in _LIVE_PATTERNS:
        if re.search(pat, q):
            return "real_time"

    # Check factual — most Tirumala queries are factual
    for pat in _FACTUAL_PATTERNS:
        if re.search(pat, q):
            return "factual"

    return None  # ambiguous → needs deeper classification


def _classify_intent_llm(query: str) -> str:
    """
    LLM-based intent classification for ambiguous queries.
    Uses the same LLM already loaded for reasoning (no extra model).
    Falls back to 'factual' on any error.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a query intent classifier. Classify the user query into exactly ONE category.\n"
             "Categories:\n"
             "- factual: Questions about facts, history, information, places, procedures, timings, prices\n"
             "- real_time: Questions needing live/current data (weather, news, today's status, current events)\n"
             "- conversational: Greetings, thanks, meta-questions about the bot itself\n\n"
             "Reply with ONLY the category name, nothing else."),
            ("human", "{query}"),
        ])

        chain = prompt | _get_llm() | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()

        # Validate the result
        if result in ("factual", "real_time", "conversational"):
            return result

        # If LLM returned something unexpected, default to factual
        logger.warning(f"LLM classifier returned unexpected: '{result}' — defaulting to factual")
        return "factual"

    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e} — defaulting to factual")
        return "factual"


def _analyze_query_intent(state: QueryState) -> dict:
    """Classifies the query intent using fast-path keywords, then LLM fallback."""
    q = state["query_english"]

    # 1. Fast-path: obvious cases
    intent = _classify_intent_fast(q)

    # 2. LLM fallback for ambiguous queries
    if intent is None:
        intent = _classify_intent_llm(q)

    # Derive legacy flags from intent
    is_live = intent == "real_time"

    # Temporal detection (specific year) — always via regex
    is_temporal = bool(re.search(r"\b(20\d{2}|19\d{2})\b", q.lower()))

    logger.info(f"🔍 [Pipeline] Intent → {intent}, live={is_live}, temporal={is_temporal}")
    return {"intent": intent, "is_live": is_live, "is_temporal": is_temporal}


# ─────────────────────────────────────────────────────────────
# Agent nodes (wrapped with error handling)
# ─────────────────────────────────────────────────────────────

@safe_agent_call("faq_agent", fallback_state={"cache_hit": False})
def _faq_node(state: QueryState) -> dict:
    """FAQ cache lookup. Bypassed for live and temporal queries."""
    if state.get("is_live") or state.get("is_temporal"):
        logger.info("📋 [Pipeline] Live/Temporal query → bypassing FAQ cache")
        return {"cache_hit": False}

    result = faq_node({"query_english": state["query_english"], "language": state["language"]})
    if result.get("cache_hit"):
        logger.info("📋 [Pipeline] FAQ cache hit")
        return {**result, "agent_route": "faq_cache"}
    return {**result}


@safe_agent_call("rag_agent", fallback_state={"needs_web_fallback": True, "rag_found": False})
def _rag_node(state: QueryState) -> dict:
    """RAG agent. Always attempted unless query is live."""
    result = rag_node({
        "query_english": state["query_english"],
        "language":      state["language"],
        "chat_history":   state.get("chat_history", [])
    })
    chunks        = result.get("context_chunks", [])
    answer        = result.get("final_answer", "")
    needs_fallback = result.get("needs_web_fallback", False)

    rag_found = bool(chunks) and bool(answer) and not needs_fallback
    logger.info(f"📘 [Pipeline] RAG found={rag_found}, needs_fallback={needs_fallback}")

    return {
        **result,
        "rag_found":          rag_found,
        "needs_web_fallback": needs_fallback,
        "agent_route":        "rag",
    }


@safe_agent_call("web_agent", fallback_state={"agent_route": "web_search"})
def _web_node(state: QueryState) -> dict:
    """Web search agent. Returns the web answer directly. Supports retries."""
    retry_count = state.get("web_retry_count", 0)

    query_en = state["query_english"]
    if retry_count > 0:
        logger.info(f"🔄 [Pipeline] Web Search Retry #{retry_count} — broadening query...")
        query_en = re.sub(r"\b(20\d{2}|19\d{2})\b", "", query_en).strip()

    result = web_node({
        "query_english": query_en,
        "language":      state["language"],
        "chat_history":   state.get("chat_history", [])
    })
    new_answer = result.get("final_answer", "")

    if not new_answer or "I am currently unable to search the live web" in new_answer:
        rag_answer = state.get("final_answer", "")
        if rag_answer:
            logger.info("🌐 [Pipeline] Web failed; keeping RAG answer.")
            return {"final_answer": rag_answer, "agent_route": "rag"}
        return {**result, "agent_route": "web_search"}

    logger.info("🌐 [Pipeline] Web search successful.")
    return {**result, "agent_route": "web_search", "web_retry_count": retry_count + 1}


# ─────────────────────────────────────────────────────────────
# Post-processing nodes
# ─────────────────────────────────────────────────────────────

def _response_aggregator(state: QueryState) -> dict:
    """Formats the final answer."""
    answer = state.get("final_answer", "").strip()
    logger.info("✅ [Aggregator] Final answer ready.")
    final = format_answer(answer)
    return {"final_answer": final}


# LLM artifacts that sometimes appear at the end of answers and must be stripped
_ANSWER_ARTIFACTS = re.compile(
    r'\s*Answer is empty or contains only stopwords\.?\s*$',
    re.IGNORECASE,
)


def _clean_answer(answer: str) -> str:
    """Strip known LLM artifacts from the final answer."""
    return _ANSWER_ARTIFACTS.sub('', answer).strip()


def format_answer(answer: str) -> str:
    return f"\n\n{_clean_answer(answer)}\n"


def _timed_node(node_name: str, fn):
    """Wrap a node function to record execution time in state['timing_ms']."""
    def wrapper(state):
        t0 = time.time()
        result = fn(state)
        elapsed_ms = round((time.time() - t0) * 1000)
        timing = state.get("timing_ms", {})
        timing[node_name] = elapsed_ms
        result["timing_ms"] = timing
        logger.info(f"⏱️ [{node_name}] completed in {elapsed_ms}ms")
        return result
    wrapper.__name__ = fn.__name__
    return wrapper


def _cache_saver(state: QueryState) -> dict:
    """Save the final answer to the FAQ cache and CSV log."""
    question = state.get("query_english", "")
    answer   = state.get("final_answer", "")
    route    = state.get("agent_route", "unknown")

    if not (question and answer):
        return {}

    # Never cache FAQ hits (already there) or live/temporal responses
    if route == "faq_cache":
        return {}
    if state.get("is_live") or state.get("is_temporal"):
        logger.info("💾 [Cache Saver] Skipping cache for live/temporal query.")
        _log_to_csv(question, answer, route)
        return {}

    # Never cache ungrounded web search answers
    verification = state.get("verification", {})
    if route == "web_search":
        is_grounded = verification.get("is_grounded", False)
        confidence = verification.get("confidence", 0.0)
        if not is_grounded or confidence < 0.5:
            logger.info(f"💾 [Cache Saver] Skipping cache for ungrounded web answer (grounded={is_grounded}, conf={confidence}).")
            _log_to_csv(question, answer, route)
            return {}

    save_to_cache(question, answer, language=state.get("language", "en"))
    _log_to_csv(question, answer, route)
    return {}


def _log_to_csv(question: str, answer: str, route: str):
    _CSV_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = _CSV_LOG_FILE.exists()
    headers = ["Timestamp", "Question", "Answer", "Agent Route"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question.strip(),
        answer.strip(),
        route
    ]
    try:
        with open(_CSV_LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        logger.info(f"📊 [CSV Log] Activity saved to {_CSV_LOG_FILE.name}")
    except Exception as e:
        logger.error(f"❌ [CSV Log] Error: {e}")


# ─────────────────────────────────────────────────────────────
# Suggestions Generator
# ─────────────────────────────────────────────────────────────

def _suggestions_node(state: QueryState) -> dict:
    """Generate 2-3 contextual follow-up questions based on the Q&A."""
    answer = state.get("final_answer", "").strip()
    query  = state.get("query_text", "").strip()
    route  = state.get("agent_route", "")

    # Skip suggestions for greetings / validation errors / empty answers
    if route in ("faq_cache", "validation_error") or not answer or len(answer) < 30:
        return {"suggestions": []}

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a suggestion engine for a Tirumala Temple AI Assistant.\n"
             "Given a question and answer about Tirumala/TTD, suggest exactly 3 brief follow-up questions "
             "the user might want to ask next.\n\n"
             "Rules:\n"
             "- Each suggestion must be a single short question (max 10 words)\n"
             "- Suggestions must be related to Tirumala, TTD, or the temple\n"
             "- Output ONLY the 3 questions, one per line, numbered 1. 2. 3.\n"
             "- No extra text or explanation"),
            ("human",
             "Question: {question}\nAnswer: {answer}\n\nSuggest 3 follow-up questions:"),
        ])

        chain = prompt | _get_llm() | StrOutputParser()
        raw = chain.invoke({"question": query, "answer": answer[:500]})

        # Parse numbered lines
        suggestions = []
        for line in raw.strip().split("\n"):
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned and len(cleaned) > 5:
                suggestions.append(cleaned)
            if len(suggestions) >= 3:
                break

        logger.info(f"💡 [Suggestions] Generated {len(suggestions)} follow-up suggestions")
        return {"suggestions": suggestions}

    except Exception as e:
        logger.warning(f"💡 [Suggestions] Generation failed: {e}")
        return {"suggestions": []}


# ─────────────────────────────────────────────────────────────
# Conditional edges
# ─────────────────────────────────────────────────────────────

def _after_intent(state: QueryState) -> str:
    """Route based on intent BEFORE hitting FAQ."""
    intent = state.get("intent", "factual")

    if intent == "real_time" or state.get("is_live"):
        logger.info("🔀 [Router] Real-time query → direct Web Search")
        return "web_search_agent"
    return "faq_agent"


def _after_faq(state: QueryState) -> str:
    """FAQ hit → done. Miss → RAG."""
    if state.get("cache_hit"):
        return "response_aggregator"
    return "rag_agent"


def _after_rag(state: QueryState) -> str:
    """RAG succeeded → done. Failed → Web."""
    if state.get("needs_web_fallback"):
        logger.info("🔀 [Router] RAG failed → Web Search fallback")
        return "web_search_agent"
    return "cache_saver"


def _after_web(state: QueryState) -> str:
    """Self-correction loop: if web answer is empty or failed, retry once."""
    ans = state.get("final_answer", "").lower()
    retry_count = state.get("web_retry_count", 0)

    # 🚨 Check if it's an Out-of-Scope rejection (matches both old and new friendly phrases)
    _rejection_phrases = [
        "i can only answer questions related to tirumala",
        "i can only help with questions about tirumala",
        "i can only help with tirumala",
        "i can only help with tirumala-related information",
    ]
    failure_indicators = [
        "unable to search the live web",
        "i can only answer questions related to tirumala",
        "no information for your query",
        "is not specified in the given text",
        "i am currently unable to search"
    ]

    if any(phrase in ans for phrase in _rejection_phrases):
        return "cache_saver"

    if (not ans or any(p in ans for p in failure_indicators)) and retry_count < 2:
        logger.warning(f"🔄 [Router] Web result suboptimal. Retrying... (Attempt {retry_count})")
        return "web_search_agent"

    return "cache_saver"


# ─────────────────────────────────────────────────────────────
# Build + compile the graph
# ─────────────────────────────────────────────────────────────

def _build_graph():
    graph = StateGraph(QueryState)

    # Pre-processing
    graph.add_node("prepare_query",        _prepare_query)
    graph.add_node("validate_query",       _validate_query_node)
    graph.add_node("analyze_intent",       _analyze_query_intent)

    # Agents (wrapped with timing)
    graph.add_node("faq_agent",            _timed_node("faq", _faq_node))
    graph.add_node("rag_agent",            _timed_node("rag", _rag_node))
    graph.add_node("web_search_agent",     _timed_node("web", _web_node))

    # Post-processing
    graph.add_node("cache_saver",          _cache_saver)
    graph.add_node("response_aggregator",  _response_aggregator)
    graph.add_node("suggestions",          _suggestions_node)

    # ── Linear pre-processing ──
    graph.add_edge(START,                  "prepare_query")
    graph.add_edge("prepare_query",        "validate_query")

    # ── Validation gate ──
    graph.add_conditional_edges("validate_query", _after_validation, {
        "response_aggregator": "response_aggregator",
        "analyze_intent":      "analyze_intent",
    })

    # ── Intent-based routing ──
    graph.add_conditional_edges("analyze_intent", _after_intent, {
        "web_search_agent": "web_search_agent",
        "faq_agent":        "faq_agent",
    })

    # ── Cascade: FAQ → RAG → Web ──
    graph.add_conditional_edges("faq_agent", _after_faq, {
        "response_aggregator": "response_aggregator",
        "rag_agent":           "rag_agent",
    })

    graph.add_conditional_edges("rag_agent", _after_rag, {
        "cache_saver":      "cache_saver",
        "web_search_agent": "web_search_agent",
    })

    graph.add_conditional_edges("web_search_agent", _after_web, {
        "web_search_agent": "web_search_agent",
        "cache_saver":      "cache_saver",
    })

    graph.add_edge("cache_saver",         "response_aggregator")
    graph.add_edge("response_aggregator", "suggestions")
    graph.add_edge("suggestions",         END)

    return graph.compile()


_pipeline = _build_graph()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_query(user_input: str, chat_history: list[dict] = None, language: str = "en") -> dict:
    logger.info("=" * 60)
    logger.info("PIPELINE START  [FAQ → RAG → Web]")
    logger.info("=" * 60)

    if not user_input.strip():
        return {"error": "Query is empty."}

    if len(user_input) > MAX_QUERY_LENGTH:
        return {"error": f"Query is too long. Please keep your question under {MAX_QUERY_LENGTH} characters."}

    initial_state: QueryState = {
        "user_input":         user_input,
        "query_text":         "",
        "language":           language or "en",   # carries detected lang from STT
        "query_english":      "",
        "chat_history":       chat_history or [],
        "validation_status":  "",
        "validation_reason":  "",
        "is_live":            False,
        "is_temporal":        False,
        "intent":             "factual",
        "cache_hit":          False,
        "rag_found":          False,
        "needs_web_fallback": False,
        "web_retry_count":    0,
        "agent_route":        "",
        "final_answer":       "",
        "context_chunks":     [],
        "domains":            [],
        "verification":       {},
        "suggestions":        [],
        "error":              None,
        "timing_ms":          {},
    }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_pipeline.invoke, initial_state)
            result = future.result(timeout=PIPELINE_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        logger.error(f"❌ [Pipeline] Timed out after {PIPELINE_TIMEOUT_S}s")
        return {
            "error": (
                "The request took too long to process. "
                "Please try again with a simpler question. Jai Balaji 🙏"
            )
        }
    except Exception as e:
        logger.error(f"❌ [Pipeline] Unexpected error: {e}")
        return {"error": f"Pipeline error: {e}"}

    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE  route={result.get('agent_route', '?')}")
    logger.info("=" * 60)

    if result.get("error"):
        return {"error": result["error"]}

    # Collect timing metrics
    timing = result.get("timing_ms", {})
    total_ms = sum(timing.values()) if timing else 0

    return {
        "query_original": result.get("query_text", ""),
        "query_english":  result.get("query_english", ""),
        "language":       result.get("language", "en"),
        "agent_route":    result.get("agent_route", "unknown"),
        "answer":         result.get("final_answer", ""),
        "sources":        result.get("context_chunks", []),
        "domains":        result.get("domains", []),
        "verification":   result.get("verification", {}),
        "suggestions":    result.get("suggestions", []),
        "timing_ms":      timing,
        "total_time_ms":  total_ms,
        "version":        APP_VERSION,
    }
