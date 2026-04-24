"""
Structured Evaluation Logger
-----------------------------
Generates [LOG] metadata blocks for each query response.
Used by the evaluation harness and can be integrated into the live pipeline.

Metadata tracks:
  - query_language: detected language code
  - query_theme: 1-3 word topic tag
  - retrieval_hit: whether relevant chunks were found
  - sources_used: comma-separated source IDs
  - confidence: High / Medium / Low
  - fallback_triggered: whether web fallback was needed
"""

import logging
import re

logger = logging.getLogger(__name__)

# Theme extraction keywords → topic tags
_THEME_PATTERNS: list[tuple[str, list[str]]] = [
    ("darshan booking",    ["darshan", "ticket", "booking", "book", "queue"]),
    ("accommodation",      ["accommodation", "lodge", "cottage", "guest house", "stay", "room"]),
    ("laddu prasadam",     ["laddu", "prasadam", "prasad"]),
    ("temple history",     ["history", "origin", "ancient", "legend", "mythology"]),
    ("festival events",    ["festival", "brahmotsavam", "vaikunta", "ekadasi", "utsav", "celebration"]),
    ("seva details",       ["seva", "puja", "archana", "kalyanotsavam", "suprabhatam"]),
    ("dress code",         ["dress", "code", "attire", "clothing", "wear"]),
    ("travel directions",  ["reach", "travel", "route", "bus", "train", "airport", "distance"]),
    ("scripture",          ["scripture", "veda", "stotra", "mantra", "sahasranama", "suprabhatam"]),
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
    ids = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source_file = meta.get("source_file", "unknown")
        page = meta.get("page", "?")
        src_id = f"SRC-{i:03d}"
        ids.append(src_id)
    return ids


def generate_log_block(
    query_language: str,
    question: str,
    chunks: list[dict],
    confidence: str,
    agent_route: str,
) -> str:
    """
    Generate a structured [LOG] metadata block.

    Returns:
        Formatted string with [LOG]...[/LOG] block.
    """
    theme = detect_query_theme(question)
    source_ids = extract_source_ids(chunks)
    retrieval_hit = len(chunks) > 0
    fallback = agent_route in ("web_search", "error")

    log_block = (
        "[LOG]\n"
        f"query_language: {query_language}\n"
        f"query_theme: {theme}\n"
        f"retrieval_hit: {'true' if retrieval_hit else 'false'}\n"
        f"sources_used: {', '.join(source_ids) if source_ids else 'none'}\n"
        f"confidence: {confidence}\n"
        f"fallback_triggered: {'true' if fallback else 'false'}\n"
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
