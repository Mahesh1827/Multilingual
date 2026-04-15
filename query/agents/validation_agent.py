"""
Input Validation and Correction Agent
======================================
Catches bad ASR transcriptions before they reach RAG / Web Search.
Works for all 5 supported languages: Telugu, Hindi, Tamil, Kannada, English.

Three outcomes:
  VALID            — query is clear; proceed normally
  NEEDS_CORRECTION — garbled/romanized Indic phonetics; corrected English provided
  INVALID          — gibberish or completely unrecognizable; ask user to clarify
                     with smart native-language suggestions of likely intent
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Fast-path vocabulary: recognized English + Tirumala domain words
# If ≥25 % of query words are in this set → skip LLM, mark VALID
# ─────────────────────────────────────────────────────────────

_KNOWN_WORDS = frozenset([
    # Common English function / question words
    "what", "is", "the", "how", "many", "where", "when", "who", "why", "which",
    "are", "was", "were", "will", "can", "could", "does", "do", "has", "have",
    "had", "be", "been", "a", "an", "i", "me", "my", "we", "our",
    "in", "on", "at", "of", "to", "for", "from", "about", "by", "with", "or",
    "and", "but", "if", "not", "no", "yes", "please", "tell", "give", "show",
    "explain", "list", "name", "names", "much", "long", "far", "any", "all",
    "number", "total", "type", "types", "mean", "means", "called",
    # Tirumala / TTD domain
    "tirumala", "tirupati", "venkateswara", "venkatachalam", "seshachalam",
    "vedachalam", "garudachalam", "anjanachalam", "narayanachalam",
    "ttd", "balaji", "temple", "darshan", "seva", "prasadam", "laddu", "laddoo",
    "brahmotsavam", "hills", "seven", "current", "temperature", "weather",
    "today", "now", "time", "timings", "timing", "fee", "cost", "price", "book",
    "distance", "height", "altitude", "history", "famous", "located", "location",
    "reach", "route", "bus", "train", "helicopter", "entry",
    "context",  # added by _prepare_query
])

# ─────────────────────────────────────────────────────────────
# LLM validation + suggestion prompt
# ─────────────────────────────────────────────────────────────

_VALIDATION_PROMPT = """\
You are a query validator for a Tirumala Temple AI assistant.
The query below was produced by speech recognition and may contain ASR errors.
Supported languages: Telugu, Hindi, Tamil, Kannada, English.

Query: "{query}"

Decide ONE of:
• VALID            — query is clear, meaningful, and answerable
• NEEDS_CORRECTION — garbled phonetic/romanized Indic text that can be recovered;
                     provide the corrected English version in corrected_query
• INVALID          — gibberish or completely unrecognizable

For INVALID and NEEDS_CORRECTION: also provide 2-3 suggestions of what the user
likely intended to ask. Each suggestion must have:
  - "native": the question in the detected language's script (Telugu/Hindi/Tamil/Kannada/English)
  - "english": the English meaning

Respond ONLY with valid JSON (no markdown, no extra text):

For VALID:
{{"status":"VALID","corrected_query":"","reason":"","suggestions":[]}}

For NEEDS_CORRECTION:
{{"status":"NEEDS_CORRECTION","corrected_query":"<corrected English>","reason":"<brief>",
  "suggestions":[{{"native":"<native script>","english":"<English meaning>"}}]}}

For INVALID:
{{"status":"INVALID","corrected_query":"","reason":"<brief>",
  "suggestions":[
    {{"native":"<likely question 1 in native script>","english":"<English>"}},
    {{"native":"<likely question 2 in native script>","english":"<English>"}},
    {{"native":"<likely question 3 in native script>","english":"<English>"}}
  ]}}

Examples of good suggestions for a garbled Telugu query:
  {{"native":"ప్రస్తుత ఉష్ణోగ్రత ఎంత?","english":"What is the current temperature?"}}
  {{"native":"దర్శన సమయం ఎంత?","english":"What are the darshan timings?"}}
  {{"native":"తిరుమల ఏడు కొండలు పేర్లు?","english":"Names of seven hills of Tirumala?"}}"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def _is_likely_valid(query: str) -> bool:
    """
    Heuristic fast-path: True when ≥25 % of alpha-words are in _KNOWN_WORDS.
    Avoids an LLM call for clearly valid English / domain queries.
    """
    words = re.findall(r'\b[a-z]+\b', query.lower())
    if not words:
        return False
    if len(words) <= 2:
        return True  # single/two-word queries are almost always fine
    known = sum(1 for w in words if w in _KNOWN_WORDS)
    return (known / len(words)) >= 0.25


def validate_and_correct(query: str) -> dict:
    """
    Validate a speech-transcribed English query.

    Returns:
        {
            "status":          "VALID" | "NEEDS_CORRECTION" | "INVALID",
            "corrected_query": str,          # filled for NEEDS_CORRECTION
            "reason":          str,          # short explanation
            "suggestions":     list[dict],   # [{"native": ..., "english": ...}, ...]
        }
    """
    _empty = {"status": "INVALID", "corrected_query": "", "reason": "Empty query", "suggestions": []}

    if not query or not query.strip():
        return _empty

    # Fast path — clearly valid → skip expensive LLM call
    if _is_likely_valid(query):
        return {"status": "VALID", "corrected_query": "", "reason": "", "suggestions": []}

    # Slow path — suspicious query → ask LLM
    logger.info("🔍 [Validation] Suspicious query, running LLM check: %r", query[:80])
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([("human", _VALIDATION_PROMPT)])
        chain  = prompt | _get_llm() | StrOutputParser()
        raw    = chain.invoke({"query": query}).strip()

        # Extract the JSON object (handles extra text from talkative LLMs)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data        = json.loads(m.group())
            status      = data.get("status", "VALID").upper()
            if status not in ("VALID", "INVALID", "NEEDS_CORRECTION"):
                status = "VALID"

            # Validate suggestion format
            raw_suggestions = data.get("suggestions", [])
            suggestions = []
            for s in raw_suggestions:
                if isinstance(s, dict) and s.get("native") and s.get("english"):
                    suggestions.append({
                        "native":  s["native"].strip(),
                        "english": s["english"].strip(),
                    })

            return {
                "status":          status,
                "corrected_query": (data.get("corrected_query") or "").strip(),
                "reason":          (data.get("reason")          or "").strip(),
                "suggestions":     suggestions,
            }

    except Exception as e:
        logger.warning("Validation LLM call failed: %s — defaulting to VALID", e)

    # On any error, let the query through — fail-open is safer than blocking
    return {"status": "VALID", "corrected_query": "", "reason": "", "suggestions": []}


def build_clarification_message(reason: str, suggestions: list[dict]) -> str:
    """
    Build a human-friendly clarification message with native-language suggestions.

    Example output:
        I couldn't clearly understand your question.

        Did you mean:
          • "ప్రస్తుత ఉష్ణోగ్రత ఎంత?" (What is the current temperature?)
          • "దర్శన సమయం ఎంత?" (What are the darshan timings?)
          • "తిరుమల ఏడు కొండలు పేర్లు?" (Names of the seven hills?)

        Please try rephrasing your question or speaking more clearly.
    """
    lines = ["I couldn't clearly understand your question."]

    if reason:
        lines.append(f"It seems: {reason}.")

    if suggestions:
        lines.append("\nDid you mean:")
        for s in suggestions:
            lines.append(f'  \u2022 "{s["native"]}" ({s["english"]})')

    lines.append(
        "\nPlease rephrase your question or speak more clearly.\n\n"
        "Example questions you can ask:\n"
        "  \u2022 What are the darshan timings at Tirumala?\n"
        "  \u2022 What is the current temperature?\n"
        "  \u2022 How much does the laddu prasadam cost?\n"
        "  \u2022 How do I reach Tirumala from Tirupati?"
    )
    return "\n".join(lines)
