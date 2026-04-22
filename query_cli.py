"""
Interactive CLI for the Tirumala Online Query Pipeline
Supports text input or microphone input mode.
"""

import _env_setup # noqa: F401 — sets T-drive caches, DLL paths, imports torch first

import logging
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path

# Suppress loud logs for interactive mode
logging.getLogger("query").setLevel(logging.WARNING)
logging.getLogger("ocr").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Structured logging via loguru (intercepts all standard logging)
try:
    from query.logging_config import setup_logging
    setup_logging()
except ImportError:
    pass

from query.agents.pipeline import run_query
from query.agents.knowledge_rag_agent import check_ollama_health


# ─────────────────────────────────────────────
# Language support (text mode)
# ─────────────────────────────────────────────

_SUPPORTED_LANGS: dict[str, str] = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
}

_UNSUPPORTED_LANG_MSG = (
    "Sorry, I currently support only English, Telugu, Hindi, Tamil, and Kannada. "
    "Please ask your question in one of these languages."
)

_text_translator = None   # lazy-loaded IndicTranslator


def _get_translator():
    global _text_translator
    if _text_translator is None:
        from query.voice_pipeline import IndicTranslator
        _text_translator = IndicTranslator()
    return _text_translator


def _detect_romanized_indic(text: str) -> str | None:
    """
    Detect romanized (Latin-script) Hindi, Telugu, Tamil, or Kannada
    using vocabulary and phonetic patterns. Returns ISO code or None.
    """
    t = text.lower().strip()
    words = set(re.findall(r'\b[a-z]+\b', t))
    if not words:
        return None

    # ── Telugu romanized markers ──
    _te_words = {
        "undi", "ento", "ela", "ekkada", "eppudu", "enduku", "enni", "anni",
        "cheppandi", "evaru", "emiti", "naku", "meeru", "manamu",
        "tirumala", "darshnam", "kodallu", "konda", "gudi", "devudu",
        "chesukovali", "kosam", "kaadu", "avunu", "ledu", "vundi",
        "cheyali", "kavali", "undi", "undhi", "povali", "vellali",
        "vastundi", "untundi", "cheppu", "anni", "ivi", "avi",
    }
    _te_suffixes = ("undi", "andi", "alu", "ulu", "llo", "ndi", "aru", "ali", "ochi")

    # ── Hindi romanized markers ──
    _hi_words = {
        "kya", "kaise", "kahan", "kab", "kyun", "kitna", "kitne", "kaun",
        "hai", "hain", "tha", "thi", "hoga", "karna", "karo", "batao",
        "mujhe", "mein", "aap", "hum", "yeh", "woh", "waha", "yahan",
        "aur", "lekin", "nahi", "nahin", "ji", "accha", "bahut", "thoda",
        "ka", "ki", "ke", "ko", "se", "pe", "par", "tak",
        "chahiye", "sakta", "sakti", "sakte", "dijiye", "bataiye",
        "mandir", "darshan", "kaise", "karein", "milega", "milta",
    }
    _hi_suffixes = ("iye", "ein", "ega", "ogi", "ata", "ati", "enge")

    # ── Tamil romanized markers ──
    _ta_words = {
        "enna", "eppadi", "enga", "eppo", "yaaru", "ethanai", "evan",
        "irukku", "irukkirathu", "illa", "aam", "irukkum", "varum",
        "enaku", "ungalukku", "naan", "neenga", "avan", "aval",
        "kovil", "koil", "darshan", "eppadi", "poga", "poganum",
        "sollunga", "theriyuma", "anga", "inga", "romba", "nalla",
        "pannanum", "vaanga", "paar", "paarunga", "konjam",
    }
    _ta_suffixes = ("kku", "ngu", "ngal", "thu", "rukku", "anga", "unga")

    # ── Kannada romanized markers ──
    _kn_words = {
        "enu", "hege", "elli", "yavaga", "yaaru", "eshtu", "yake",
        "ide", "ide", "illa", "houdu", "alla", "untu", "irutte",
        "nanage", "neenu", "neevu", "avanu", "avalu", "avaru",
        "devaalaya", "devalaya", "darshana", "hogi", "barutta",
        "helu", "nodri", "gottaa", "alli", "illi", "tumba", "swalpa",
        "maadu", "beku", "aagutta", "bartaare", "kelsa",
    }
    _kn_suffixes = ("alli", "inda", "ige", "anu", "avu", "iru", "atte")

    # Score each language
    scores = {}
    for lang, lang_words, lang_suffixes in [
        ("te", _te_words, _te_suffixes),
        ("hi", _hi_words, _hi_suffixes),
        ("ta", _ta_words, _ta_suffixes),
        ("kn", _kn_words, _kn_suffixes),
    ]:
        # Exact word matches
        word_hits = len(words & lang_words)
        # Suffix matches (partial phonetic patterns)
        suffix_hits = sum(1 for w in words if any(w.endswith(s) for s in lang_suffixes))
        scores[lang] = word_hits * 2 + suffix_hits  # weight exact matches higher

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]
    runner_up = sorted(scores.values(), reverse=True)[1]

    # SHORT QUERY (1-2 words): accept a single strong keyword hit (score >= 1)
    # without requiring dominance — a single "hai" or "enti" is unambiguous.
    words = text.lower().split()
    if len(words) <= 2 and best_score >= 1:
        return best_lang

    # LONGER QUERY: require at least 2 points AND be clearly dominant
    if best_score >= 2 and best_score > runner_up:
        return best_lang

    return None


# ── Explicit language override parsing ──

_LANG_OVERRIDE_PATTERNS = {
    "en": re.compile(r"\b(tell|answer|reply|respond|speak)\s+(me\s+)?(in\s+)?english\b", re.IGNORECASE),
    "te": re.compile(r"\b(tell|answer|reply|respond|speak)\s+(me\s+)?(in\s+)?telugu\b", re.IGNORECASE),
    "hi": re.compile(r"\b(tell|answer|reply|respond|speak)\s+(me\s+)?(in\s+)?hindi\b", re.IGNORECASE),
    "ta": re.compile(r"\b(tell|answer|reply|respond|speak)\s+(me\s+)?(in\s+)?tamil\b", re.IGNORECASE),
    "kn": re.compile(r"\b(tell|answer|reply|respond|speak)\s+(me\s+)?(in\s+)?kannada\b", re.IGNORECASE),
}


def _detect_language_override(text: str) -> str | None:
    """
    Check if the user explicitly requested a response language.
    Returns ISO code if found, else None.
    """
    for lang, pat in _LANG_OVERRIDE_PATTERNS.items():
        if pat.search(text):
            return lang
    return None


def _strip_language_override(text: str) -> str:
    """Remove the 'tell in <language>' phrase from the query so it doesn't confuse the pipeline."""
    for pat in _LANG_OVERRIDE_PATTERNS.values():
        text = pat.sub("", text)
    return text.strip()


def _detect_text_language(text: str) -> str:
    """
    Detect the language of typed/pasted text.
    Returns ISO 639-1 code: "en", "te", "hi", "ta", "kn", or "unknown".

    Detection cascade (fastest → most expensive):
      1. Unicode script-range analysis  — 100% reliable for native-script chars
      2. Lingua (statistical)           — good for native-script + mixed text
      3. Romanized Indic detector       — vocabulary/phonetic (Latin-script Indic)
      4. ASCII ratio fallback           — mostly ASCII → English
    """
    if not text or not text.strip():
        return "unknown"

    # Step 1: Unicode script ranges (instant, 100% accurate for native chars)
    try:
        from query.STT import _detect_script_language
        script_lang = _detect_script_language(text)
        if script_lang:
            logger.debug("Lang detect [script]: %s → %s", text[:30], script_lang)
            return script_lang
    except Exception:
        pass

    # Step 2: Lingua statistical detector (good for 3+ word queries)
    try:
        from query.STT import _detect_language_lingua
        lingua_lang = _detect_language_lingua(text)
        if lingua_lang:
            logger.debug("Lang detect [lingua]: %s → %s", text[:30], lingua_lang)
            return lingua_lang
    except Exception:
        pass

    # Step 3: Romanized Indic (Latin-script Hindi/Telugu/Tamil/Kannada)
    romanized = _detect_romanized_indic(text)
    if romanized:
        logger.debug("Lang detect [romanized]: %s → %s", text[:30], romanized)
        return romanized

    # Step 4: ASCII ratio fallback — mostly ASCII → English
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) < 0.10:
        return "en"

    return "unknown"


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _get_username() -> str:
    import os
    return os.environ.get("USERNAME", os.environ.get("USER", "user"))


# ─────────────────────────────────────────────
# Auto-start Ollama if not running
# ─────────────────────────────────────────────

def _ensure_ollama_running() -> bool:
    """Start Ollama if it is not already running."""
    
    if check_ollama_health():
        return True

    ollama_exe = shutil.which("ollama")

    if ollama_exe is None:
        candidates = [
            Path(r"C:\Users") / _get_username() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            Path(r"C:\Program Files\Ollama\ollama.exe"),
        ]

        for c in candidates:
            if c.exists():
                ollama_exe = str(c)
                break

    if ollama_exe is None:
        return False

    print("🔄 Starting Ollama server...")

    subprocess.Popen(
        [ollama_exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    for _ in range(15):
        time.sleep(1)

        if check_ollama_health():
            print("✅ Ollama server started.")
            return True

    return False


# ─────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────

def _show_result(result: dict, response_lang: str = "en"):

    print("\n" + "=" * 70)

    print("Answer:")
    print(result["answer"])

    print("-" * 70)

    route = result.get("agent_route", "unknown")
    print(f"Agent: {route.replace('_', ' ').title()}")

    domains = result.get("domains", [])
    if domains:
        print(f"Domains: {', '.join(domains)}")

    vf = result.get("verification", {})
    if vf:
        print(f"Grounded: {vf.get('is_grounded', 'N/A')} (confidence: {vf.get('confidence', 0.0):.2f})")

    sources = result["sources"]

    if sources:

        print("\nSources:")

        for i, src in enumerate(sources[:5], 1):

            meta = src["metadata"]

            print(
                f"   [{i}] {meta.get('source_file', 'web')} "
                f"(Page {meta.get('page', '?')})"
            )

    # Display follow-up suggestions
    suggestions = result.get("suggestions", [])
    if suggestions:
        print("\n💡 Try asking:")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")

    # Display English translation at the absolute bottom if needed
    if response_lang not in ("en", "unknown") and result.get("answer"):
        try:
            translator = _get_translator()
            english_translation = translator.indic_to_english(result["answer"], response_lang)
            if english_translation and english_translation.strip() != result["answer"].strip():
                print("\n[English Translation]:")
                print(english_translation)
        except Exception:
            pass

    print("=" * 70)


# ─────────────────────────────────────────────
# Conversational Memory / Query Reformulation
# ─────────────────────────────────────────────

def _is_followup(text: str) -> bool:
    """Return True if the user is referring back to previous context.

    Reduces false positives by requiring either:
      - A leading pronoun/demonstrative (sentence starts with reference word)
      - Multiple reference words in the query
      - Very short query with a reference word (likely a clarification)
    """
    t = text.strip().lower()
    words = re.findall(r'\b\w+\b', t)
    if not words:
        return False

    reference_words = {"it", "that", "this", "these", "those", "they", "them"}
    found = [w for w in words if w in reference_words]

    # Leading pronoun/demonstrative → almost certainly a follow-up
    if words[0] in reference_words:
        return True

    # Multiple reference words → likely a follow-up
    if len(found) >= 2:
        return True

    # Very short query (≤5 words) with a reference word → likely a clarification
    if found and len(words) <= 5:
        return True

    return False


# Phrases that indicate a failed/fallback answer — never include in follow-up context
_FAILED_ANSWER_PHRASES = [
    "i am sorry",
    "i'm sorry",
    "unable to",
    "could not find",
    "don't have that",
    "does not contain",
    "web search failed",
    "encountered an error",
]


def _build_query_with_context(user_input: str, history: list) -> str:
    """Prepend the last Q+A as context when a follow-up is detected."""
    if not history or not _is_followup(user_input):
        return user_input

    last = history[-1]

    # Don't inject context from failed/error answers — it adds noise
    ans_lower = last["a"].lower()
    if any(phrase in ans_lower for phrase in _FAILED_ANSWER_PHRASES):
        return user_input

    return (
        f"Previous question: {last['q']}\n"
        f"Previous answer summary: {last['a'][:600]}\n\n"
        f"Follow-up question (answer using the above context if relevant): {user_input}"
    )


def _build_pipeline_history(history: list) -> list[dict]:
    """Convert internal Q+A history to the pipeline's chat_history format."""
    pipeline_history = []
    for turn in history:
        pipeline_history.append({"role": "user", "content": turn["q"]})
        pipeline_history.append({"role": "assistant", "content": turn["a"]})
    return pipeline_history

# ─────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────

def main():

    print("=" * 70)
    print(" Tirumala Multi-Agent AI Assistant")
    print("=" * 70)

    from query.config import USE_GROQ, GROQ_MODEL

    if USE_GROQ:
        print(f"LLM: Groq API ({GROQ_MODEL})")
    elif _ensure_ollama_running():
        print("Ollama is running.")
    else:
        print("Warning: Ollama not running. Answers may use raw context.")

    print("\nSelect Interaction Mode:")
    print("  [1] Text Chat (Default)")
    print("  [2] Voice Interface")
    mode = input("\nEnter choice [1 or 2]: ").strip()

    if mode == "2":
        from query.voice_pipeline import VoicePipeline

        print("\n  🌐 Voice language will be auto-detected.")
        lang_hint = "auto"
        
        voice_history = []
        
        def cli_voice_rag_fn(english_query: str, detected_lang: str = "en") -> str:
            print("\nThinking...")
            start = time.time()
            try:
                # Reformulate query with memory BEFORE sending to pipeline
                final_query = _build_query_with_context(english_query, voice_history)
                pipeline_history = _build_pipeline_history(voice_history)

                # Pass detected_lang so LLM responds directly in user's language
                result = run_query(final_query, chat_history=pipeline_history, language=detected_lang)
                elapsed = time.time() - start
                print(f"Response time: {elapsed:.2f}s")

                if isinstance(result, dict) and "answer" in result:
                    answer_text = result["answer"]
                    
                    if detected_lang not in ("en", "unknown") and answer_text:
                        translator = _get_translator()
                        translated = translator.english_to_indic(answer_text, detected_lang)
                        if translated:
                            answer_text = translated
                            result["answer"] = answer_text

                    voice_history.append({"q": english_query, "a": answer_text})
                    if len(voice_history) > 3:
                        voice_history.pop(0)
                    
                    _show_result(result, detected_lang)
                    return answer_text
                return str(result)
            except Exception as e:
                print(f"Pipeline Error: {e}")
                return "I am sorry, I encountered an error."

        print("\nLoading Voice Pipeline... This might take a few seconds.")
        pipeline = VoicePipeline(
            rag_fn=cli_voice_rag_fn,
            wake_word=None,
            language_hint=lang_hint,
        )

        try:
            pipeline.run()
        except KeyboardInterrupt:
            pipeline.stop()
            print("\nVoice pipeline stopped.")
        return

    print("\n--- TEXT MODE ACTIVATED ---")

    # ── Conversation history (last 3 turns) ──
    _history: list = []
    _MAX_HISTORY = 3

    while True:
        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting.")
            break

        # ── Explicit language override ("tell in Hindi", etc.) ──
        response_lang_override = _detect_language_override(user_input)
        clean_input = _strip_language_override(user_input) if response_lang_override else user_input

        # ── Language detection ──────────────────────────────────
        detected_lang = _detect_text_language(clean_input)

        # Rule 4: unknown / unsupported → fall back to English gracefully
        if detected_lang not in _SUPPORTED_LANGS:
            detected_lang = "en"

        lang_name = _SUPPORTED_LANGS.get(detected_lang, "")
        if lang_name and detected_lang != "en":
            print(f"[Language detected: {lang_name}]")

        # Decide which language the response should be in:
        # - Explicit override takes priority ("tell in Hindi")
        # - Otherwise, respond in the detected language
        response_lang = response_lang_override or detected_lang

        if response_lang_override:
            override_name = _SUPPORTED_LANGS.get(response_lang_override, response_lang_override)
            print(f"[Response language override: {override_name}]")

        # ── Translate non-English input to English for retrieval ─────────
        translation_failed = False
        if detected_lang not in ("en", "unknown"):
            translator = _get_translator()
            english_input = translator.indic_to_english(clean_input, detected_lang)
            translation_failed = getattr(translator, "_last_translation_failed", False)

            if translation_failed:
                # Translation service unavailable — warn user and proceed with
                # original text. The LLM lang_hint will try to parse it natively.
                lang_name_str = _SUPPORTED_LANGS.get(detected_lang, detected_lang)
                print(
                    f"[⚠ Translation service unavailable. Attempting to process "
                    f"{lang_name_str} query directly...]"
                )
                english_input = clean_input  # pass native text, lang_hint handles it
            elif english_input and english_input.strip() != clean_input.strip():
                print(f"[Translated to English for retrieval]: {english_input}")
        else:
            english_input = clean_input

        # ── Reformulate with conversational context ─────────────
        print("Thinking...")
        start = time.time()

        final_query = _build_query_with_context(english_input, _history)
        pipeline_history = _build_pipeline_history(_history)

        try:
            # Pass response_lang so LLM answers directly in user's language
            result = run_query(final_query, chat_history=pipeline_history, language=response_lang)
        except Exception as e:
            print(f"Pipeline Error: {e}")
            continue

        elapsed = time.time() - start
        print(f"Response time: {elapsed:.2f}s")

        if "error" in result:
            print(f"Error: {result['error']}")
            continue

        # ── Display answer ────────────────────────────────────────
        # Explicit output translation phase
        answer = result.get("answer", "")

        if response_lang not in ("en", "unknown") and answer:
            print(f"Translating answer to {_SUPPORTED_LANGS.get(response_lang, response_lang)}...")
            translator = _get_translator()
            translated = translator.english_to_indic(answer, response_lang)
            if translated:
                answer = translated
                result["answer"] = answer

        # Store English query + answer for follow-up context tracking
        if answer:
            _history.append({"q": english_input, "a": answer})
            if len(_history) > _MAX_HISTORY:
                _history.pop(0)

        _show_result(result, response_lang)


# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()