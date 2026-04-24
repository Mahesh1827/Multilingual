"""
LLM-Based Evaluation Judge
----------------------------
Uses the project's LLM (Groq/Ollama) to score answers on 6 criteria
with strict pass thresholds aligned to the production RAG eval spec.

Criteria & Thresholds:
  Faithfulness    ≥ 0.80 — every claim traceable to a SOURCE
  Relevancy       ≥ 0.75 — answer directly addresses user intent
  Correctness     ≥ 0.70 — no factual errors or unsupported inferences
  Language Match  = 1.00 — response language = query language
  Completeness    ≥ 0.70 — all sub-questions addressed
  Conciseness     ≥ 0.70 — no padding, repetition, or unnecessary hedging

Usage:
    from eval.judge import llm_judge_score, judge_confidence_label
    scores = llm_judge_score(question, answer, context, expected_answer, language)
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Pass thresholds (aligned with generation_metrics.py)
# ──────────────────────────────────────────────

JUDGE_THRESHOLDS = {
    "faithfulness":   0.80,
    "relevancy":      0.75,
    "correctness":    0.70,
    "language_match": 1.00,
    "completeness":   0.70,
    "conciseness":    0.70,
}


# ──────────────────────────────────────────────
# LLM Judge Prompt
# ──────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are a strict evaluation judge for a multilingual RAG (Retrieval-Augmented Generation) system.

You will receive:
- A user QUESTION
- The system's ANSWER
- The CONTEXT (retrieved source chunks used to generate the answer)
- The EXPECTED answer (ground truth)
- The expected LANGUAGE of the response

Score the ANSWER on each criterion using a 0.0–1.0 scale (two decimal places).

CRITERIA:

1. **faithfulness** (0.0–1.0): Is every claim in the answer traceable to the provided context?
   - 1.0: All claims directly supported by context
   - 0.5: Most claims supported, some minor additions
   - 0.0: Major claims not in context (hallucination)

2. **relevancy** (0.0–1.0): Does the answer directly address the user's actual question?
   - 1.0: Perfectly addresses the question intent
   - 0.5: Partially addresses, some tangential info
   - 0.0: Does not address the question at all

3. **correctness** (0.0–1.0): Is the answer factually correct compared to the expected answer?
   - 1.0: Fully correct, matches expected answer
   - 0.5: Partially correct, some inaccuracies
   - 0.0: Incorrect or contradicts expected answer

4. **language_match** (0.0 or 1.0): Is the response in the correct language with no code-mixing?
   - 1.0: Correct language, natural and fluent
   - 0.0: Wrong language or significant code-mixing

5. **completeness** (0.0–1.0): Are all parts/sub-questions of the query addressed?
   - 1.0: All aspects fully covered
   - 0.5: Main point addressed, some gaps
   - 0.0: Major aspects missing

6. **conciseness** (0.0–1.0): Is the answer appropriately concise without padding or repetition?
   - 1.0: Perfect length, no padding
   - 0.5: Slightly verbose but acceptable
   - 0.0: Extremely verbose, lots of repetition

OUTPUT FORMAT (strict JSON, no extra text):
{{
  "faithfulness": 0.00,
  "relevancy": 0.00,
  "correctness": 0.00,
  "language_match": 0.00,
  "completeness": 0.00,
  "conciseness": 0.00,
  "overall_score": 0.00,
  "reasoning": "1-2 sentence explanation"
}}
"""


def llm_judge_score(
    question: str,
    answer: str,
    context_text: str,
    expected_answer: str,
    expected_lang: str = "en",
) -> dict | None:
    """
    Use the project's LLM to judge an answer on 6 criteria.

    Returns:
        Dict with scores per criterion + overall_score + reasoning +
        threshold_check (pass/fail per criterion) + judge_flags (failed criteria).
        None if LLM call fails.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from query.agents.knowledge_rag_agent import _get_llm

        prompt = ChatPromptTemplate.from_messages([
            ("system", _JUDGE_PROMPT),
            ("human",
             "QUESTION: {question}\n\n"
             "ANSWER: {answer}\n\n"
             "CONTEXT: {context}\n\n"
             "EXPECTED ANSWER: {expected}\n\n"
             "EXPECTED LANGUAGE: {language}\n\n"
             "Score this answer (JSON only):"),
        ])

        chain = prompt | _get_llm() | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer[:1500],
            "context": context_text[:2000],
            "expected": expected_answer,
            "language": expected_lang,
        })

        # Parse JSON from LLM response
        json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = json.loads(raw.strip())

        # Validate and clamp scores
        required_keys = {"faithfulness", "relevancy", "correctness", "language_match", "completeness", "conciseness"}
        if not required_keys.issubset(scores.keys()):
            logger.warning("LLM judge missing keys: %s", required_keys - scores.keys())
            return None

        for key in required_keys:
            scores[key] = max(0.0, min(1.0, float(scores[key])))
            scores[key] = round(scores[key], 2)

        # Compute overall if not provided
        if "overall_score" not in scores:
            scores["overall_score"] = round(
                sum(scores[k] for k in required_keys) / len(required_keys), 2
            )

        # ── Threshold-based pass/fail ──
        failures = []
        threshold_details = {}
        for criterion, threshold in JUDGE_THRESHOLDS.items():
            score = scores.get(criterion, 0)
            passed = score >= threshold
            threshold_details[criterion] = {
                "score": score,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                failures.append(criterion)

        scores["threshold_check"] = {
            "passed": len(failures) == 0,
            "failures": failures,
            "details": threshold_details,
        }
        scores["judge_flags"] = failures if failures else ["none"]

        # ── Confidence derivation ──
        overall = scores["overall_score"]
        if overall >= 0.75 and len(failures) <= 1:
            scores["confidence"] = "High"
        elif overall >= 0.50:
            scores["confidence"] = "Medium"
        else:
            scores["confidence"] = "Low"

        logger.info(
            "🧑‍⚖️ [Judge] Overall=%.2f Confidence=%s Flags=%s",
            overall, scores["confidence"], failures or "none",
        )
        return scores

    except Exception as e:
        logger.warning("🧑‍⚖️ [Judge] LLM evaluation failed: %s", e)
        return None


def judge_confidence_label(metrics: dict) -> str:
    """
    Derive a confidence label from computed generation metrics (non-LLM).

    Args:
        metrics: Dict from compute_all_generation_metrics().

    Returns:
        "High", "Medium", or "Low"
    """
    faith = metrics.get("faithfulness", 0)
    correct = metrics.get("correctness", 0)
    relevancy = metrics.get("relevancy", 0)

    avg = (faith + correct + relevancy) / 3

    if avg >= 0.70:
        return "High"
    elif avg >= 0.45:
        return "Medium"
    else:
        return "Low"


def classify_query_type(question: str) -> str:
    """
    Classify the query type for logging.

    Returns: factual | procedural | comparative | conversational | out-of-scope
    """
    q = question.lower().strip()

    # Conversational
    _conversational = [
        "hello", "hi", "hey", "namaste", "thank", "bye", "goodbye",
        "who are you", "what can you do", "how are you",
    ]
    if any(q.startswith(c) or q == c for c in _conversational):
        return "conversational"

    # Procedural
    _procedural = ["how to", "how do", "how can", "steps to", "process", "book", "booking", "register"]
    if any(p in q for p in _procedural):
        return "procedural"

    # Comparative
    _comparative = ["compare", "difference between", "vs", "versus", "better", "which is"]
    if any(c in q for c in _comparative):
        return "comparative"

    # Out-of-scope (no Tirumala keywords)
    _scope = [
        "tirumala", "tirupati", "venkateswara", "ttd", "balaji", "temple",
        "darshan", "seva", "prasad", "laddu", "festival", "brahmotsavam",
    ]
    if not any(s in q for s in _scope):
        # Check if it's a general knowledge question
        _general = ["capital", "president", "cricket", "movie", "politics", "weather"]
        if any(g in q for g in _general):
            return "out-of-scope"

    return "factual"
