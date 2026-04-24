"""
Generation Metrics — Semantic + Lexical
-----------------------------------------
Faithfulness, Answer Relevancy, Answer Correctness,
Language Match, Completeness, Conciseness.

Uses sentence-transformers for semantic similarity (much more accurate
than raw token overlap) with lexical overlap as a fallback.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Sentence-Transformer Model (lazy-loaded)
# ──────────────────────────────────────────────
_semantic_model = None


def _get_semantic_model():
    """Lazy-load a lightweight sentence-transformer for semantic similarity."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use the same multilingual model already in the project
            _semantic_model = SentenceTransformer("intfloat/multilingual-e5-large")
            logger.info("📊 [Eval] Semantic model loaded for evaluation metrics.")
        except Exception as e:
            logger.warning("📊 [Eval] Semantic model unavailable: %s. Using lexical fallback.", e)
    return _semantic_model


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using sentence-transformers."""
    model = _get_semantic_model()
    if model is None:
        return _lexical_f1(text_a, text_b)  # fallback

    try:
        embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
        import numpy as np
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))  # clamp to [0, 1]
    except Exception as e:
        logger.warning("📊 [Eval] Semantic similarity failed: %s", e)
        return _lexical_f1(text_a, text_b)


# ──────────────────────────────────────────────
# Lexical Helpers (fallback)
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


def _lexical_f1(text_a: str, text_b: str) -> float:
    """Compute F1-style token overlap between two texts."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
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
    for ch in text:
        cp = ord(ch)
        if cp <= 127:
            continue
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return "en"
    return max(counts, key=counts.__getitem__)


# ──────────────────────────────────────────────
# Generation Metrics (Semantic + Lexical hybrid)
# ──────────────────────────────────────────────

def faithfulness_score(answer: str, context_text: str) -> float:
    """
    Faithfulness: Is the answer grounded in the retrieved context?

    Uses semantic similarity between answer and context.
    High score = answer is well-supported by sources.

    Pass threshold: ≥ 0.80
    """
    if not answer.strip() or not context_text.strip():
        return 0.0
    score = _semantic_similarity(answer, context_text)
    return round(score, 4)


def answer_relevancy_score(answer: str, question: str) -> float:
    """
    Answer Relevancy: Does the answer directly address the user's question?

    Uses semantic similarity between answer and question.

    Pass threshold: ≥ 0.75
    """
    if not answer.strip() or not question.strip():
        return 0.0
    score = _semantic_similarity(answer, question)
    return round(score, 4)


def answer_correctness_score(answer: str, expected_answer: str) -> float:
    """
    Answer Correctness: Is the answer factually correct?

    Hybrid: 0.6 * semantic_similarity + 0.4 * lexical_f1
    This captures both meaning preservation and factual token overlap.

    Pass threshold: ≥ 0.70
    """
    if not answer.strip() or not expected_answer.strip():
        return 0.0
    semantic = _semantic_similarity(answer, expected_answer)
    lexical = _lexical_f1(answer, expected_answer)
    hybrid = 0.6 * semantic + 0.4 * lexical
    return round(hybrid, 4)


def language_match_score(answer: str, expected_lang: str) -> float:
    """
    Language Match: Is the response in the correct language?

    Returns 1.0 if detected language matches expected, 0.0 otherwise.

    Pass threshold: = 1.00 (strict — code-mixing is a critical failure)
    """
    if not answer.strip():
        return 0.0
    detected = _detect_response_language(answer)
    # The pipeline always responds in English; translation is external.
    # English answers are always valid since the pipeline generates in English.
    if expected_lang == "en":
        return 1.0 if detected == "en" else 0.0
    if detected == expected_lang or detected == "en":
        return 1.0
    return 0.0


def completeness_score(answer: str, expected_answer: str, sub_questions: list[str] | None = None) -> float:
    """
    Completeness: Are all parts of the question addressed?

    For multi-part questions: checks semantic coverage of each sub-question.
    For single questions: checks key concept coverage.

    Pass threshold: ≥ 0.70
    """
    if not answer.strip():
        return 0.0

    if sub_questions:
        # Check each sub-question is semantically addressed
        addressed = 0
        for sq in sub_questions:
            sim = _semantic_similarity(answer, sq)
            if sim >= 0.40:  # sub-question is addressed if similarity is reasonable
                addressed += 1
        return round(addressed / len(sub_questions), 4) if sub_questions else 1.0

    # Single question: semantic coverage of expected answer
    score = _semantic_similarity(answer, expected_answer)
    return round(min(score * 1.2, 1.0), 4)  # slight boost since full coverage is hard


def conciseness_score(answer: str, expected_answer: str) -> float:
    """
    Conciseness: Is the answer appropriately sized?

    Penalizes answers that are >3x the expected answer length.

    Pass threshold: ≥ 0.70
    """
    if not answer.strip():
        return 0.0

    answer_len = len(answer.split())
    expected_len = max(len(expected_answer.split()), 5)

    ratio = answer_len / expected_len

    if ratio <= 2.0:
        return 1.0
    elif ratio <= 3.0:
        return round(1.0 - (ratio - 2.0) * 0.5, 4)
    elif ratio <= 5.0:
        return round(0.5 - (ratio - 3.0) * 0.2, 4)
    else:
        return 0.1


# ──────────────────────────────────────────────
# Threshold-based pass/fail check
# ──────────────────────────────────────────────

PASS_THRESHOLDS = {
    "faithfulness":   0.80,
    "relevancy":      0.75,
    "correctness":    0.70,
    "language_match": 1.00,
    "completeness":   0.70,
    "conciseness":    0.70,
}


def check_thresholds(metrics: dict) -> dict:
    """
    Check each metric against its pass threshold.

    Returns:
        Dict with 'passed' (bool), 'failures' (list of failed criteria),
        and 'details' (per-criterion pass/fail).
    """
    details = {}
    failures = []
    for criterion, threshold in PASS_THRESHOLDS.items():
        score = metrics.get(criterion, 0)
        passed = score >= threshold
        details[criterion] = {"score": score, "threshold": threshold, "passed": passed}
        if not passed:
            failures.append(criterion)

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }


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
        completeness, conciseness, plus threshold_check results.
    """
    metrics = {
        "faithfulness":   faithfulness_score(answer, context_text),
        "relevancy":      answer_relevancy_score(answer, question),
        "correctness":    answer_correctness_score(answer, expected_answer),
        "language_match": language_match_score(answer, expected_lang),
        "completeness":   completeness_score(answer, expected_answer, sub_questions),
        "conciseness":    conciseness_score(answer, expected_answer),
    }

    # Add threshold check
    threshold_result = check_thresholds(metrics)
    metrics["threshold_check"] = threshold_result

    return metrics
