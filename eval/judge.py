"""
LLM-Based Pairwise Judge
-------------------------
Uses the project's existing LLM (Groq/Ollama) to evaluate answers
on 5 criteria: Faithfulness, Relevancy, Language Match, Completeness, Conciseness.

Each criterion is scored 1-5. The judge also provides a confidence level
(High/Medium/Low) based on overall scoring.

Usage:
    from eval.judge import llm_judge_score
    scores = llm_judge_score(question, answer, context, expected_answer, language)
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are a strict evaluation judge for a multilingual RAG (Retrieval-Augmented Generation) system.

You will receive:
- A user QUESTION
- The system's ANSWER
- The CONTEXT (retrieved source chunks used to generate the answer)
- The EXPECTED answer (ground truth)
- The expected LANGUAGE of the response

Score the ANSWER on each criterion using a 1-5 scale:
  1 = Very Poor, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent

CRITERIA:

1. **Faithfulness** (1-5): Is every claim in the answer traceable to the provided context?
   - 5: All claims directly supported by context
   - 3: Most claims supported, minor additions
   - 1: Major claims not in context (hallucination)

2. **Relevancy** (1-5): Does the answer directly address the user's actual question?
   - 5: Perfectly addresses the question
   - 3: Partially addresses, some tangential info
   - 1: Does not address the question at all

3. **Language_Match** (1-5): Is the response in the correct language with no code-mixing?
   - 5: Correct language, natural and fluent
   - 3: Mostly correct but some code-mixing
   - 1: Wrong language entirely

4. **Completeness** (1-5): Are all parts/sub-questions of the query addressed?
   - 5: All aspects fully covered
   - 3: Main point addressed, some gaps
   - 1: Major aspects missing

5. **Conciseness** (1-5): Is the answer appropriately concise without padding or repetition?
   - 5: Perfect length, no padding
   - 3: Slightly verbose but acceptable
   - 1: Extremely verbose, lots of repetition

OUTPUT FORMAT (strict JSON, no extra text):
{{
  "faithfulness": <1-5>,
  "relevancy": <1-5>,
  "language_match": <1-5>,
  "completeness": <1-5>,
  "conciseness": <1-5>,
  "overall_score": <1-5>,
  "reasoning": "<1-2 sentence explanation>"
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
    Use the project's LLM to judge an answer on 5 criteria.

    Returns:
        Dict with scores per criterion + overall_score + reasoning.
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
            "answer": answer[:1500],        # Truncate to avoid token limits
            "context": context_text[:2000],
            "expected": expected_answer,
            "language": expected_lang,
        })

        # Parse JSON from LLM response
        # Try to find JSON block in the response
        json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = json.loads(raw.strip())

        # Validate expected keys
        required_keys = {"faithfulness", "relevancy", "language_match", "completeness", "conciseness"}
        if not required_keys.issubset(scores.keys()):
            logger.warning("LLM judge missing keys: %s", required_keys - scores.keys())
            return None

        # Clamp scores to 1-5
        for key in required_keys:
            scores[key] = max(1, min(5, int(scores[key])))

        # Compute overall if not provided
        if "overall_score" not in scores:
            scores["overall_score"] = round(
                sum(scores[k] for k in required_keys) / len(required_keys), 1
            )

        # Derive confidence
        overall = scores["overall_score"]
        if overall >= 4.0:
            scores["confidence"] = "High"
        elif overall >= 2.5:
            scores["confidence"] = "Medium"
        else:
            scores["confidence"] = "Low"

        logger.info("🧑‍⚖️ [Judge] Scores: %s", {k: scores[k] for k in required_keys})
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

    if avg >= 0.5:
        return "High"
    elif avg >= 0.25:
        return "Medium"
    else:
        return "Low"
