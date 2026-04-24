"""
Generation Metrics
------------------
Faithfulness, Answer Relevancy, Answer Correctness,
Language Match, Completeness, Conciseness.

These metrics evaluate the quality of generated answers
against retrieved sources and golden expected answers.
"""

import re

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "are", "was", "were", "be", "has", "have", "had",
    "this", "that", "at", "by", "from", "as", "not", "but", "so", "if",
    "i", "you", "he", "she", "we", "they", "its", "can", "will", "do",
}


def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase tokens (3+ chars) minus stopwords."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower())) - _STOPWORDS


def _f1_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute F1-style overlap between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Unicode script ranges for language detection
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "hi"),   # Devanagari → Hindi
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C80, 0x0CFF, "kn"),   # Kannada
]


def _detect_response_language(text: str) -> str:
    """Detect the primary language of a response using Unicode script blocks."""
    counts: dict[str, int] = {}
    ascii_count = 0

    for ch in text:
        cp = ord(ch)
        if cp <= 127:
            ascii_count += 1
            continue
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return "en"  # All ASCII → English

    best_lang = max(counts, key=counts.__getitem__)
    return best_lang


# ──────────────────────────────────────────────
# Generation Metrics
# ──────────────────────────────────────────────

def faithfulness_score(answer: str, context_text: str) -> float:
    """
    Faithfulness: Is the answer grounded in the retrieved context?

    Measured as F1 overlap between answer tokens and context tokens.
    Higher = more faithful to sources.
    """
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_text)
    return round(_f1_overlap(answer_tokens, context_tokens), 4)


def answer_relevancy_score(answer: str, question: str) -> float:
    """
    Answer Relevancy: Does the answer directly address the user's question?

    Measured as F1 overlap between answer tokens and question tokens.
    """
    answer_tokens = _tokenize(answer)
    question_tokens = _tokenize(question)
    return round(_f1_overlap(answer_tokens, question_tokens), 4)


def answer_correctness_score(answer: str, expected_answer: str) -> float:
    """
    Answer Correctness: Is the answer factually correct?

    Measured as F1 overlap between answer tokens and expected (golden) answer tokens.
    """
    answer_tokens = _tokenize(answer)
    expected_tokens = _tokenize(expected_answer)
    return round(_f1_overlap(answer_tokens, expected_tokens), 4)


def language_match_score(answer: str, expected_lang: str) -> float:
    """
    Language Match: Is the response in the correct language?

    Returns 1.0 if detected language matches expected, 0.0 otherwise.
    For English queries, we check ASCII dominance.
    For Indic queries, we check if the correct script is present.
    """
    if not answer.strip():
        return 0.0

    detected = _detect_response_language(answer)

    # English pipeline always responds in English (translation is external)
    # So for now, if expected is any language, check if answer is English
    # (since the RAG agent always responds in English)
    if expected_lang == "en":
        return 1.0 if detected == "en" else 0.0

    # For non-English queries, the pipeline responds in English and
    # translation happens externally. So English answers are valid.
    # If we detect native script matching the expected lang, that's also valid.
    if detected == expected_lang or detected == "en":
        return 1.0

    return 0.0


def completeness_score(answer: str, expected_answer: str, sub_questions: list[str] | None = None) -> float:
    """
    Completeness: Are all parts of the question addressed?

    For single questions: checks if key concepts from expected answer appear.
    For multi-part questions: checks each sub-question has representation.
    """
    if not answer.strip():
        return 0.0

    answer_tokens = _tokenize(answer)

    if sub_questions:
        # Check each sub-question is addressed
        addressed = 0
        for sq in sub_questions:
            sq_tokens = _tokenize(sq)
            if sq_tokens:
                overlap = len(answer_tokens & sq_tokens) / len(sq_tokens)
                if overlap >= 0.15:
                    addressed += 1
        return round(addressed / len(sub_questions), 4) if sub_questions else 1.0

    # Single question: check key concept coverage from expected answer
    expected_tokens = _tokenize(expected_answer)
    if not expected_tokens:
        return 1.0

    # Extract "key" tokens (longer, more specific words)
    key_tokens = {t for t in expected_tokens if len(t) >= 4}
    if not key_tokens:
        key_tokens = expected_tokens

    covered = len(answer_tokens & key_tokens)
    return round(min(covered / max(len(key_tokens), 1), 1.0), 4)


def conciseness_score(answer: str, expected_answer: str) -> float:
    """
    Conciseness: Is the answer appropriately sized?

    Penalizes answers that are >3x the expected answer length.
    Returns 1.0 for well-sized answers, decreasing toward 0.0 for bloated ones.
    """
    if not answer.strip():
        return 0.0

    answer_len = len(answer.split())
    expected_len = max(len(expected_answer.split()), 5)  # Min 5 words baseline

    ratio = answer_len / expected_len

    if ratio <= 2.0:
        return 1.0  # Good — within 2x expected length
    elif ratio <= 3.0:
        return round(1.0 - (ratio - 2.0) * 0.5, 4)  # Linear decay 1.0 → 0.5
    elif ratio <= 5.0:
        return round(0.5 - (ratio - 3.0) * 0.2, 4)  # Further decay 0.5 → 0.1
    else:
        return 0.1  # Extremely verbose


def compute_all_generation_metrics(
    answer: str,
    question: str,
    expected_answer: str,
    context_text: str,
    expected_lang: str = "en",
    sub_questions: list[str] | None = None,
) -> dict:
    """
    Compute all generation metrics for a single query.

    Returns:
        Dict with faithfulness, relevancy, correctness, language_match,
        completeness, conciseness.
    """
    return {
        "faithfulness":   faithfulness_score(answer, context_text),
        "relevancy":      answer_relevancy_score(answer, question),
        "correctness":    answer_correctness_score(answer, expected_answer),
        "language_match": language_match_score(answer, expected_lang),
        "completeness":   completeness_score(answer, expected_answer, sub_questions),
        "conciseness":    conciseness_score(answer, expected_answer),
    }
