"""
RAGAS Evaluation Harness
Runs golden dataset questions through the pipeline and evaluates:
  - Answer faithfulness (is the answer supported by retrieved context?)
  - Answer relevance (does the answer address the question?)

Usage:
    python -m eval.run_eval
"""

import _env_setup  # noqa: F401

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from query.agents.pipeline import run_query

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    """Run all golden dataset questions through the pipeline and collect results."""
    dataset = load_golden_dataset()
    print(f"\n{'='*70}")
    print(f"  RAGAS Evaluation — {len(dataset)} questions")
    print(f"{'='*70}\n")

    results = []
    total_time = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected_answer"]
        domain = item["domain"]
        test_lang = item.get("language", "en")

        label = f"[{test_lang.upper()}] " if test_lang != "en" else ""
        print(f"[{i}/{len(dataset)}] {label}{question[:80]}")

        start = time.time()
        try:
            result = run_query(question)
            elapsed = time.time() - start
            total_time += elapsed

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            route = result.get("agent_route", "unknown")
            verification = result.get("verification", {})

            # Simple faithfulness score: overlap between answer and context
            context_text = " ".join(s.get("text", "") for s in sources if isinstance(s, dict))
            faithfulness = _compute_overlap(answer, context_text) if context_text else 0.0

            # Simple relevance score: overlap between answer and expected answer
            relevance = _compute_overlap(answer, expected)

            entry = {
                "question": question,
                "expected_answer": expected,
                "actual_answer": answer,
                "domain": domain,
                "agent_route": route,
                "faithfulness": round(faithfulness, 3),
                "relevance": round(relevance, 3),
                "is_grounded": verification.get("is_grounded", False),
                "response_time_s": round(elapsed, 2),
            }

            status = "✅" if relevance > 0.3 else "⚠️"
            print(f"  {status} route={route} faith={faithfulness:.2f} rel={relevance:.2f} ({elapsed:.1f}s)\n")

        except Exception as e:
            elapsed = time.time() - start
            entry = {
                "question": question,
                "expected_answer": expected,
                "actual_answer": f"ERROR: {e}",
                "domain": domain,
                "agent_route": "error",
                "faithfulness": 0.0,
                "relevance": 0.0,
                "is_grounded": False,
                "response_time_s": round(elapsed, 2),
            }
            print(f"  ❌ ERROR: {e}\n")

        results.append(entry)

    # Compute aggregate scores
    avg_faith = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    avg_rel = sum(r["relevance"] for r in results) / len(results) if results else 0
    grounded_pct = sum(1 for r in results if r["is_grounded"]) / len(results) * 100 if results else 0
    avg_time = total_time / len(results) if results else 0

    summary = {
        "total_questions": len(results),
        "avg_faithfulness": round(avg_faith, 3),
        "avg_relevance": round(avg_rel, 3),
        "grounded_pct": round(grounded_pct, 1),
        "avg_response_time_s": round(avg_time, 2),
        "total_time_s": round(total_time, 1),
    }

    output = {"summary": summary, "results": results}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Questions:       {summary['total_questions']}")
    print(f"  Avg Faithfulness: {summary['avg_faithfulness']:.3f}")
    print(f"  Avg Relevance:    {summary['avg_relevance']:.3f}")
    print(f"  Grounded:         {summary['grounded_pct']:.1f}%")
    print(f"  Avg Response:     {summary['avg_response_time_s']:.2f}s")
    print(f"  Total Time:       {summary['total_time_s']:.1f}s")
    print(f"\n  Results saved to: {RESULTS_PATH}")
    print(f"{'='*70}\n")

    return output


def _compute_overlap(text_a: str, text_b: str) -> float:
    """Compute token overlap ratio between two texts (F1-style, bidirectional)."""
    if not text_a or not text_b:
        return 0.0

    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "of", "to", "and", "or",
        "for", "with", "are", "was", "were", "be", "has", "have", "had",
        "this", "that", "at", "by", "from", "as", "not", "but", "so",
    }

    import re
    tokens_a = set(re.findall(r"\b[a-z]{3,}\b", text_a.lower())) - stopwords
    tokens_b = set(re.findall(r"\b[a-z]{3,}\b", text_b.lower())) - stopwords

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1 score


if __name__ == "__main__":
    run_evaluation()
