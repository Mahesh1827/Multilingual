"""
Online Query Pipeline — Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ──────────────────────────────────────────────
# LLM Provider
# ──────────────────────────────────────────────
# Priority: Groq API (cloud, fast) → Ollama (local, offline)
# Set GROQ_API_KEY in .env to use Groq; otherwise falls back to Ollama.
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL",      "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:14b")
OLLAMA_TIMEOUT  = 120  # seconds

# Which provider to use (auto-detected from API key)
USE_GROQ = bool(GROQ_API_KEY)

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K_RETRIEVE = 6          # chunks per domain query
TOP_K_FINAL    = 5          # top chunks passed to LLM after re-rank

# ──────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────
LANG_DETECT_CONFIDENCE = 0.70   # below this → treat as English
DEFAULT_LANGUAGE       = "en"
SUPPORTED_VOICE_LANGUAGES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
}

# ──────────────────────────────────────────────
# Verifier
# ──────────────────────────────────────────────
VERIFIER_OVERLAP_THRESHOLD = 0.10   # min keyword overlap ratio to consider grounded

# ──────────────────────────────────────────────
# Whisper ASR (optional voice input)
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large

# ──────────────────────────────────────────────
# Domain Routing Keywords
# ──────────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "about_tirumala": [
        "tirumala", "tirupati", "venkateswara", "srinivasa", "balaji",
        "seven hills", "saptagiri", "akasa ganga", "location", "geography",
        "altitude", "climate", "population",
    ],
    "devotional_practice": [
        "prayer", "worship", "devotion", "ritual", "puja", "archana",
        "bhajan", "kirtan", "meditation", "fast", "fasting", "vow",
        "vratam", "offering", "prasad", "prasadam", "seva", "aarti",
    ],
    "festival_events": [
        "festival", "brahmotsavam", "utsav", "celebration", "event",
        "procession", "rathotsavam", "celestial", "annual", "kalyanotsavam",
        "vaikunta ekadasi", "teppotsavam",
    ],
    "knowledge_scripture": [
        "scripture", "veda", "upanishad", "purana", "bhagavata",
        "stotra", "mantra", "vishnu sahasranama", "suprabhatam",
        "theology", "philosophy", "dharma", "moksha",
    ],
    "temple_history": [
        "history", "origin", "legend", "mythology", "ancient",
        "inscription", "dynasty", "pallava", "chola", "vijayanagara",
        "agam", "temple construction", "gopuram", "vimana", "architecture",
    ],
    "pilgrimage_seva": [
        "darshan", "ticket", "booking", "queue", "accommodation", "lodge",
        "prasadam", "laddu", "special darshan", "seva booking", "pilgrim",
        "yatri", "ttd", "how to", "when to visit", "dress code", "rules",
        "entry pass", "virtual queue",
    ],
}
# ──────────────────────────────────────────────
# Web Search
# ──────────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ──────────────────────────────────────────────
# Production Settings
# ──────────────────────────────────────────────
APP_VERSION            = "2.0.0"
ASSISTANT_NAME         = "Govinda"
PIPELINE_TIMEOUT_S     = 60        # max seconds for full pipeline run
FAQ_CACHE_TTL_DAYS     = 30          # cache expiry
FAQ_CACHE_MAX_SIZE     = 10000        # max cached Q&A pairs
MAX_QUERY_LENGTH       = 2000        # reject queries longer than this
TTD_HELPLINE           = "1800-425-4141"
TTD_WEBSITE            = "https://tirupatibalaji.ap.gov.in"

# Govinda persona fallback string — must be deterministic for downstream matching
GOVINDA_FALLBACK_ANSWER = (
    "I don't have specific information on this right now. "
    f"Please contact TTD directly at {TTD_HELPLINE} or visit {TTD_WEBSITE} "
    "for the latest details. Jai Balaji 🙏"
)
