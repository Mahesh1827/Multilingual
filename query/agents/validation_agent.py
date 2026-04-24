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
    # Tirumala / TTD domain — expanded
    "tirumala", "tirupati", "venkateswara", "venkatachalam", "seshachalam",
    "vedachalam", "garudachalam", "anjanachalam", "narayanachalam",
    "ttd", "balaji", "temple", "darshan", "seva", "prasadam", "laddu", "laddoo",
    "brahmotsavam", "hills", "seven", "current", "temperature", "weather",
    "today", "now", "time", "timings", "timing", "fee", "cost", "price", "book",
    "distance", "height", "altitude", "history", "famous", "located", "location",
    "reach", "route", "bus", "train", "helicopter", "entry",
    "context",  # added by _prepare_query
    # Additional TTD domain terms
    "annadanam", "suprabhatam", "vaikunta", "ekadasi", "kalyanotsavam",
    "hundi", "cottages", "srivari", "govinda", "saptagiri", "alipiri",
    "tiruchanur", "padmavathi", "srinivasa", "mangapuram", "devasthanam",
    "accommodation", "lodge", "guest", "house", "room", "queue", "virtual",
    "special", "free", "paid", "online", "offline", "token", "counter",
    "dress", "code", "rules", "mobile", "phone", "camera", "allowed",
    "food", "canteen", "meals", "breakfast", "lunch", "dinner",
    "donation", "offering", "calendar", "festival", "annual", "daily",
    "morning", "evening", "night", "open", "close", "closed",
    "walk", "steps", "footpath", "ghat", "road", "parking",
])

_LANG_NAMES = {
    "en": "English", "te": "Telugu", "hi": "Hindi",
    "ta": "Tamil", "kn": "Kannada",
}

_VALIDATION_PROMPT = """\
You are a multilingual query validator for a Tirumala Temple AI assistant.
The query below was produced by speech recognition and may contain ASR errors.
Supported languages: Telugu, Hindi, Tamil, Kannada, English.

Source language: {language_name}
English query (may be auto-translated): "{query}"
Original text: "{original_text}"

IMPORTANT — Speech recognition frequently mishears words in ALL languages:

English ASR errors:
  - Any word that sounds similar to a Tirumala-related term should be corrected
  - e.g. "Kirmala"/"Tirmala"→"Tirumala", "darsan"→"darshan", etc.

Telugu (తెలుగు) ASR errors:
  - Vowel length confusion: short↔long (ఎ↔ఏ, ఉ↔ఊ, ఒ↔ఓ)
  - Aspirated↔unaspirated: భ↔బ, ధ↔ద, ఘ↔గ
  - Numbers: ఇర్వే→ఇరవై, తమ్ముదు→తొమ్మిది, సబ్బిలు→సభ్యులు

Hindi (हिंदी) ASR errors:
  - Nasalization errors: ं/ँ often wrong
  - Aspirated↔unaspirated: ख↔क, घ↔ग, थ↔त, ध↔द
  - मन्दिर→मंदिर, दर्सन→दर्शन

Tamil (தமிழ்) ASR errors:
  - கோவில்↔கோயில், உறுப்பினர்↔உறுப்பினர்கள்
  - Vowel length confusion in all positions

Kannada (ಕನ್ನಡ) ASR errors:
  - ದೇವಸ್ತಾನ→ದೇವಸ್ಥಾನ, ದರ್ಶಣ→ದರ್ಶನ
  - Aspirated↔unaspirated consonants

RULES:
1. Use BOTH the English query AND the original text to understand intent
2. If the original text is in native script, use it to resolve ambiguities
3. If ANY word is a misheard/misspelled version of a known term, return NEEDS_CORRECTION
4. The corrected_query must ALWAYS be in correct English

Decide ONE of:
• VALID            — query is completely clear, all words correct, answerable as-is
• NEEDS_CORRECTION — contains misheard/misspelled words; provide corrected English
• INVALID          — gibberish or completely unrecognizable

Respond ONLY with valid JSON (no markdown, no extra text):

For VALID:
{{"status":"VALID","corrected_query":"","reason":"","suggestions":[]}}

For NEEDS_CORRECTION:
{{"status":"NEEDS_CORRECTION","corrected_query":"<corrected English>","reason":"<brief>",
  "suggestions":[{{"native":"<question in {language_name} script>","english":"<English meaning>"}}]}}

For INVALID:
{{"status":"INVALID","corrected_query":"","reason":"<brief>",
  "suggestions":[
    {{"native":"<likely question 1 in {language_name}>","english":"<English>"}},
    {{"native":"<likely question 2 in {language_name}>","english":"<English>"}},
    {{"native":"<likely question 3 in {language_name}>","english":"<English>"}}
  ]}}"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def _is_likely_valid(query: str) -> bool:
    """
    Heuristic fast-path: decides whether to skip the LLM validation call.

    Returns False (= send to LLM) when ANY word is unrecognized and ≥4 chars.
    This catches ASR garbling like "Kirmala" (meant "Tirumala") — the LLM
    can semantically understand it from context without hardcoded dictionaries.
    """
    words = re.findall(r'\b[a-z]+\b', query.lower())
    if not words:
        return False
    if len(words) <= 2:
        return True  # single/two-word queries are almost always fine

    # If ALL words are recognized → clearly valid, skip LLM
    unknown_words = [w for w in words if w not in _KNOWN_WORDS and len(w) >= 4]
    if not unknown_words:
        return True

    # Has unrecognized words — let LLM understand and correct them
    logger.info(
        "🔍 [Validation] Unrecognized words detected: %s — routing to LLM for correction",
        unknown_words,
    )
    return False


def validate_and_correct(query: str, language: str = "en", original_text: str = "") -> dict:
    """
    Validate a speech-transcribed query in any supported language.

    Uses the LLM to intelligently understand and correct ASR errors
    in English, Telugu, Hindi, Tamil, and Kannada — no hardcoded
    dictionaries needed.

    Args:
        query:         English version of the user's question.
        language:      ISO 639-1 code of the source language.
        original_text: The original text before translation (native script).

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

    if not original_text:
        original_text = query

    language_name = _LANG_NAMES.get(language, "English")

    # Fast path — only for English queries where ALL words are recognized
    # Non-English queries always go to LLM (the LLM understands native scripts)
    if language == "en" and _is_likely_valid(query):
        return {"status": "VALID", "corrected_query": "", "reason": "", "suggestions": []}

    # For non-English: if the translated English looks clean AND original is clean, skip LLM
    if language != "en" and _is_likely_valid(query):
        # But if original text has non-ASCII (native script), let LLM verify
        non_ascii_ratio = sum(1 for c in original_text if ord(c) > 127) / max(len(original_text), 1)
        if non_ascii_ratio < 0.1:
            # Romanized input that looks valid — skip LLM
            return {"status": "VALID", "corrected_query": "", "reason": "", "suggestions": []}
        # Native script — let LLM check for ASR errors using both texts

    # LLM path — intelligent correction for any language
    logger.info(
        "🔍 [Validation] LLM check [%s]: query=%r, original=%r",
        language_name, query[:60], original_text[:60],
    )
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([("human", _VALIDATION_PROMPT)])
        chain  = prompt | _get_llm() | StrOutputParser()
        raw    = chain.invoke({
            "query":         query,
            "original_text": original_text,
            "language_name": language_name,
        }).strip()

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


def build_clarification_message(reason: str, suggestions: list[dict], language: str = "en") -> str:
    """
    Build a warm, friendly clarification message when we couldn't understand the query.
    Includes native-language suggestions and an invitation to try again.
    """
    # Warm, devotee-friendly opening
    lines = [
        "Namaskaram! I'm sorry, I couldn't quite catch what you said. "
        "Could you please repeat your question a little more clearly?"
    ]

    if suggestions:
        lines.append("\nPerhaps you were asking one of these:")
        for i, s in enumerate(suggestions, 1):
            lines.append(f'  {i}. "{s["native"]}" — {s["english"]}')

    lines.append(
        "\nYou can also try:"
        "\n  • Speaking a bit slower and closer to the mic"
        "\n  • Typing your question instead (press Ctrl+C to switch to text mode)"
        "\n  • Asking in a different language — I understand English, Telugu, Hindi, Tamil, and Kannada"
    )

    lines.append(
        "\nSome things I can help with:"
        "\n  • Darshan timings and ticket booking"
        "\n  • Accommodation and travel to Tirumala"
        "\n  • Temple history, sevas, and festivals"
        "\n  • Dress code, rules, and guidelines"
    )

    lines.append("\nJai Balaji! Please try again.")
    return "\n".join(lines)


# Short, spoken-friendly version for TTS (long text sounds bad when spoken)
def build_spoken_clarification(suggestions: list[dict] = None) -> str:
    """Build a short version suitable for TTS (spoken aloud)."""
    msg = "I'm sorry, I couldn't understand that clearly. Could you please repeat your question?"
    if suggestions and len(suggestions) > 0:
        msg += f" Did you perhaps mean: {suggestions[0]['english']}?"
    return msg
