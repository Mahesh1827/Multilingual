"""
Structured Evaluation Logger — Enhanced
-----------------------------------------
Generates [LOG] metadata blocks for each query response.
Aligned with the production RAG evaluation specification.

Log fields:
  query_language:     ISO 639-1 code
  query_theme:        1-3 word topic tag
  query_type:         factual | procedural | comparative | conversational | out-of-scope
  retrieval_hit:      true | false
  sources_used:       comma-separated source IDs or "none"
  top_score:          highest relevance score among retrieved chunks
  confidence:         High | Medium | Low
  fallback_triggered: true | false
  judge_flags:        comma-separated criteria below threshold, or "none"
"""

import logging
import re

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Theme extraction
# ──────────────────────────────────────────────

_THEME_PATTERNS: list[tuple[str, list[str]]] = [
    ("darshan booking",    ["darshan", "ticket", "booking", "book", "queue"]),
    ("accommodation",      ["accommodation", "lodge", "cottage", "guest house", "stay", "room"]),
    ("laddu prasadam",     ["laddu", "prasadam", "prasad"]),
    ("temple history",     ["history", "origin", "ancient", "legend", "mythology"]),
    ("festival events",    ["festival", "brahmotsavam", "vaikunta", "ekadasi", "utsav", "celebration"]),
    ("seva details",       ["seva", "puja", "archana", "kalyanotsavam", "suprabhatam"]),
    ("dress code",         ["dress", "code", "attire", "clothing", "wear"]),
    ("travel directions",  ["reach", "travel", "route", "bus", "train", "airport", "distance"]),
    ("scripture",          ["scripture", "veda", "stotra", "mantra", "sahasranama"]),
    ("temple info",        ["temple", "tirumala", "tirupati", "seven hills", "altitude", "location"]),
    ("weather",            ["weather", "temperature", "climate", "forecast", "rain"]),
    ("timings",            ["timing", "time", "open", "close", "hours"]),
    ("greeting",           ["hello", "hi", "namaste", "hey", "good morning"]),
    ("out of scope",       ["capital", "president", "cricket", "movie", "politics"]),
]


def detect_query_theme(question: str) -> str:
    """Detect the query theme/topic from the question text."""
    q_lower = question.lower()
    for theme, keywords in _THEME_PATTERNS:
        if any(kw in q_lower for kw in keywords):
            return theme
    return "general"


def extract_source_ids(chunks: list[dict]) -> list[str]:
    """Extract source identifiers from retrieved chunks."""
    return [f"SRC-{i:03d}" for i in range(1, len(chunks) + 1)]


def get_top_score(chunks: list[dict]) -> float:
    """Get the highest relevance/rerank score among chunks."""
    if not chunks:
        return 0.0
    scores = []
    for chunk in chunks:
        # Try rerank_score first, then raw score
        s = chunk.get("rerank_score", chunk.get("score", 0))
        if isinstance(s, (int, float)):
            scores.append(float(s))
    return round(max(scores), 4) if scores else 0.0


def generate_log_block(
    query_language: str,
    question: str,
    chunks: list[dict],
    confidence: str,
    agent_route: str,
    query_type: str = "factual",
    judge_flags: list[str] | None = None,
) -> str:
    """
    Generate a structured [LOG] metadata block.

    Returns:
        Formatted string with [LOG]...[/LOG] block.
    """
    theme = detect_query_theme(question)
    source_ids = extract_source_ids(chunks)
    top_score = get_top_score(chunks)
    retrieval_hit = len(chunks) > 0
    fallback = agent_route in ("web_search", "error")
    flags_str = ", ".join(judge_flags) if judge_flags else "none"

    log_block = (
        "[LOG]\n"
        f"query_language: {query_language}\n"
        f"query_theme: {theme}\n"
        f"query_type: {query_type}\n"
        f"retrieval_hit: {'true' if retrieval_hit else 'false'}\n"
        f"sources_used: {', '.join(source_ids) if source_ids else 'none'}\n"
        f"top_score: {top_score}\n"
        f"confidence: {confidence}\n"
        f"fallback_triggered: {'true' if fallback else 'false'}\n"
        f"judge_flags: {flags_str}\n"
        "[/LOG]"
    )
    return log_block


def parse_log_block(log_text: str) -> dict | None:
    """Parse a [LOG]...[/LOG] block back into a dict."""
    match = re.search(r"\[LOG\](.*?)\[/LOG\]", log_text, re.DOTALL)
    if not match:
        return None

    result = {}
    for line in match.group(1).strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result
