"""
Multilingual RAG Evaluation Harness
====================================
Production-grade evaluation framework for the Tirumala AI Assistant.

Metrics Computed:
  Retrieval: Precision@K, Recall@K, Hit Rate, MRR, NDCG@K
  Generation: Faithfulness, Relevancy, Correctness, Language Match, Completeness, Conciseness
  LLM Judge: (optional) 5-criterion scoring via Groq/Ollama

Usage:
    python -m eval.run_eval                 # Standard evaluation
    python -m eval.run_eval --llm-judge     # Include LLM judge scoring
    python -m eval.run_eval --quick         # First 10 questions only
"""

import _env_setup  # noqa: F401

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from query.agents.pipeline import run_query

from eval.metrics.retrieval_metrics import compute_all_retrieval_metrics
from eval.metrics.generation_metrics import compute_all_generation_metrics
from eval.judge import llm_judge_score, judge_confidence_label, classify_query_type
from eval.eval_logger import generate_log_block, detect_query_theme

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_EVAL_DIR            = Path(__file__).parent
GOLDEN_DATASET_PATH  = _EVAL_DIR / "golden_dataset.json"
RESULTS_PATH         = _EVAL_DIR / "results.json"
REPORT_PATH          = _EVAL_DIR / "report.md"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Per-query evaluation
# ──────────────────────────────────────────────

def evaluate_single(
    item: dict,
    index: int,
    total: int,
    use_llm_judge: bool = False,
) -> dict:
    """Run a single golden question through the pipeline and compute all metrics."""
    question       = item["question"]
    expected       = item["expected_answer"]
    domain         = item["domain"]
    test_lang      = item.get("language", "en")
    keywords       = item.get("relevant_keywords", [])
    sub_questions  = item.get("sub_questions")

    label = f"[{test_lang.upper()}] " if test_lang != "en" else ""
    print(f"  [{index}/{total}] {label}{question[:80]}")

    start = time.time()
    try:
        result = run_query(question, language=test_lang)
        elapsed = time.time() - start

        answer   = result.get("answer", "")
        sources  = result.get("sources", [])
        route    = result.get("agent_route", "unknown")
        verif    = result.get("verification", {})

        # Build context text from sources
        context_text = " ".join(
            s.get("text", "") for s in sources if isinstance(s, dict)
        )

        # FAQ cache special handling: the cached answer IS the source
        # so if context is empty but we have an answer, use answer as context
        if not context_text.strip() and answer.strip():
            context_text = answer

        # ── Retrieval metrics ──
        retrieval = compute_all_retrieval_metrics(
            chunks=sources if isinstance(sources, list) else [],
            expected_answer=expected,
            keywords=keywords,
            k=5,
        )

        # ── Generation metrics ──
        generation = compute_all_generation_metrics(
            answer=answer,
            question=question,
            expected_answer=expected,
            context_text=context_text,
            expected_lang=test_lang,
            sub_questions=sub_questions,
        )

        # ── Confidence ──
        confidence = judge_confidence_label(generation)

        # ── Query classification ──
        query_type = classify_query_type(question)

        # ── Threshold check ──
        threshold_check = generation.get("threshold_check", {})
        judge_flags = threshold_check.get("failures", [])

        # ── LLM Judge (optional) ──
        judge_scores = None
        if use_llm_judge:
            judge_scores = llm_judge_score(
                question=question,
                answer=answer,
                context_text=context_text[:2000],
                expected_answer=expected,
                expected_lang=test_lang,
            )
            # Override judge_flags with LLM judge flags if available
            if judge_scores and judge_scores.get("judge_flags"):
                judge_flags = judge_scores["judge_flags"]

        # ── Structured log ──
        log_block = generate_log_block(
            query_language=test_lang,
            question=question,
            chunks=sources if isinstance(sources, list) else [],
            confidence=confidence,
            agent_route=route,
            query_type=query_type,
            judge_flags=judge_flags if judge_flags else None,
        )

        entry = {
            "question":         question,
            "expected_answer":  expected,
            "actual_answer":    answer,
            "domain":           domain,
            "language":         test_lang,
            "agent_route":      route,
            "query_type":       query_type,
            "is_grounded":      verif.get("is_grounded", False),
            "response_time_s":  round(elapsed, 2),
            "retrieval":        retrieval,
            "generation":       generation,
            "confidence":       confidence,
            "query_theme":      detect_query_theme(question),
            "threshold_passed": threshold_check.get("passed", False),
            "judge_flags":      judge_flags if judge_flags else ["none"],
            "log_block":        log_block,
        }

        if judge_scores:
            entry["llm_judge"] = judge_scores

        # Status indicator — use threshold pass/fail
        t_pass = threshold_check.get("passed", False)
        faith = generation.get("faithfulness", 0)
        correct = generation.get("correctness", 0)
        flags_str = ", ".join(judge_flags) if judge_flags else "none"
        status = "✅" if t_pass else "⚠️"
        print(f"    {status} route={route} faith={faith:.2f} correct={correct:.2f} flags=[{flags_str}] ({elapsed:.1f}s)")

    except Exception as e:
        elapsed = time.time() - start
        entry = {
            "question":         question,
            "expected_answer":  expected,
            "actual_answer":    f"ERROR: {e}",
            "domain":           domain,
            "language":         test_lang,
            "agent_route":      "error",
            "is_grounded":      False,
            "response_time_s":  round(elapsed, 2),
            "retrieval":        {},
            "generation":       {},
            "confidence":       "Low",
            "query_theme":      detect_query_theme(question),
            "log_block":        "",
        }
        print(f"    ❌ ERROR: {e}")

    return entry


# ──────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────

def _safe_avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate_results(results: list[dict]) -> dict:
    """Compute aggregate metrics across all results."""
    n = len(results)
    if n == 0:
        return {}

    # Retrieval aggregates
    ret_keys = ["precision_at_k", "recall_at_k", "hit_rate", "mrr", "ndcg_at_k"]
    retrieval_agg = {}
    for key in ret_keys:
        vals = [r["retrieval"].get(key, 0) for r in results if r.get("retrieval")]
        retrieval_agg[f"avg_{key}"] = _safe_avg(vals)

    # Generation aggregates
    gen_keys = ["faithfulness", "relevancy", "correctness", "language_match", "completeness", "conciseness"]
    generation_agg = {}
    for key in gen_keys:
        vals = [r["generation"].get(key, 0) for r in results if r.get("generation")]
        generation_agg[f"avg_{key}"] = _safe_avg(vals)

    # Other aggregates
    grounded_pct = round(sum(1 for r in results if r.get("is_grounded")) / n * 100, 1)
    avg_time = _safe_avg([r["response_time_s"] for r in results])
    total_time = round(sum(r["response_time_s"] for r in results), 1)

    # Confidence distribution
    conf_dist = {"High": 0, "Medium": 0, "Low": 0}
    for r in results:
        c = r.get("confidence", "Low")
        conf_dist[c] = conf_dist.get(c, 0) + 1

    # Threshold pass rate
    threshold_pass_count = sum(1 for r in results if r.get("threshold_passed", False))
    threshold_pass_rate = round(threshold_pass_count / n * 100, 1)

    # Most common judge flags (failure patterns)
    flag_counts: dict[str, int] = {}
    for r in results:
        for flag in r.get("judge_flags", []):
            if flag != "none":
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    top_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Query type distribution
    type_dist: dict[str, int] = {}
    for r in results:
        qt = r.get("query_type", "factual")
        type_dist[qt] = type_dist.get(qt, 0) + 1

    # LLM judge aggregates (if present)
    judge_agg = {}
    judge_results = [r.get("llm_judge") for r in results if r.get("llm_judge")]
    if judge_results:
        for key in ["faithfulness", "relevancy", "correctness", "language_match", "completeness", "conciseness", "overall_score"]:
            vals = [j.get(key, 0) for j in judge_results]
            judge_agg[f"avg_{key}"] = _safe_avg(vals)

    return {
        "total_questions":      n,
        "retrieval":            retrieval_agg,
        "generation":           generation_agg,
        "grounded_pct":         grounded_pct,
        "threshold_pass_rate":  threshold_pass_rate,
        "top_failure_flags":    top_flags,
        "query_type_dist":      type_dist,
        "avg_response_time_s":  avg_time,
        "total_time_s":         total_time,
        "confidence_dist":      conf_dist,
        "llm_judge":            judge_agg if judge_agg else None,
    }


def aggregate_by_group(results: list[dict], group_key: str) -> dict:
    """Aggregate metrics grouped by a key (domain, language)."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = r.get(group_key, "unknown")
        groups.setdefault(key, []).append(r)

    return {key: aggregate_results(items) for key, items in sorted(groups.items())}


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def generate_report(summary: dict, by_domain: dict, by_lang: dict, results: list[dict]) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# Multilingual RAG Evaluation Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Questions:** {summary['total_questions']}")
    lines.append(f"**Total Time:** {summary['total_time_s']}s")
    lines.append(f"**Avg Response Time:** {summary['avg_response_time_s']}s")
    lines.append("")

    # Overall metrics table
    lines.append("## Overall Metrics\n")
    lines.append("### Retrieval Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for key, val in summary.get("retrieval", {}).items():
        lines.append(f"| {key.replace('avg_', '').replace('_', ' ').title()} | {val:.4f} |")

    lines.append("\n### Generation Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for key, val in summary.get("generation", {}).items():
        lines.append(f"| {key.replace('avg_', '').replace('_', ' ').title()} | {val:.4f} |")

    lines.append(f"\n**Grounded:** {summary['grounded_pct']}%")
    lines.append(f"**Threshold Pass Rate:** {summary.get('threshold_pass_rate', 0)}%")

    # Top failure flags
    top_flags = summary.get("top_failure_flags", [])
    if top_flags:
        lines.append("\n### Top Failure Patterns\n")
        lines.append("| Criterion | Failures |")
        lines.append("|-----------|----------|")
        for flag, count in top_flags:
            lines.append(f"| {flag.replace('_', ' ').title()} | {count} |")

    # Confidence distribution
    conf = summary.get("confidence_dist", {})
    lines.append("\n### Confidence Distribution\n")
    lines.append("| Level | Count |")
    lines.append("|-------|-------|")
    for level in ["High", "Medium", "Low"]:
        lines.append(f"| {level} | {conf.get(level, 0)} |")

    # Query type distribution
    type_dist = summary.get("query_type_dist", {})
    if type_dist:
        lines.append("\n### Query Type Distribution\n")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for qt, count in sorted(type_dist.items()):
            lines.append(f"| {qt} | {count} |")

    # LLM Judge (if present)
    if summary.get("llm_judge"):
        lines.append("\n### LLM Judge Scores (0.0-1.0 scale)\n")
        lines.append("| Criterion | Avg Score |")
        lines.append("|-----------|-----------|")
        for key, val in summary["llm_judge"].items():
            lines.append(f"| {key.replace('avg_', '').replace('_', ' ').title()} | {val:.1f} |")

    # Per-domain breakdown
    lines.append("\n---\n## Per-Domain Breakdown\n")
    for domain, stats in by_domain.items():
        lines.append(f"### {domain.replace('_', ' ').title()} ({stats['total_questions']} questions)\n")
        gen = stats.get("generation", {})
        lines.append(f"- Faithfulness: {gen.get('avg_faithfulness', 0):.4f}")
        lines.append(f"- Correctness: {gen.get('avg_correctness', 0):.4f}")
        lines.append(f"- Avg Response Time: {stats.get('avg_response_time_s', 0):.2f}s")
        lines.append("")

    # Per-language breakdown
    lines.append("\n---\n## Per-Language Breakdown\n")
    lang_names = {"en": "English", "te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada"}
    for lang, stats in by_lang.items():
        name = lang_names.get(lang, lang)
        lines.append(f"### {name} ({stats['total_questions']} questions)\n")
        gen = stats.get("generation", {})
        ret = stats.get("retrieval", {})
        lines.append(f"- Hit Rate: {ret.get('avg_hit_rate', 0):.4f}")
        lines.append(f"- Faithfulness: {gen.get('avg_faithfulness', 0):.4f}")
        lines.append(f"- Correctness: {gen.get('avg_correctness', 0):.4f}")
        lines.append(f"- Language Match: {gen.get('avg_language_match', 0):.4f}")
        lines.append("")

    # Per-query details (failures only)
    failures = [r for r in results if r.get("generation", {}).get("correctness", 0) < 0.2]
    if failures:
        lines.append("\n---\n## Low-Scoring Queries (Correctness < 0.2)\n")
        for r in failures:
            lines.append(f"- **Q:** {r['question'][:80]}")
            lines.append(f"  - Route: {r['agent_route']}, Correctness: {r.get('generation', {}).get('correctness', 0):.3f}")
            lines.append(f"  - Answer: {r['actual_answer'][:100]}...")
            lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────

def run_evaluation(use_llm_judge: bool = False, quick: bool = False):
    """Run the full evaluation harness."""
    dataset = load_golden_dataset()
    if quick:
        dataset = dataset[:10]

    n = len(dataset)
    print(f"\n{'='*70}")
    print(f"  Multilingual RAG Evaluation — {n} questions")
    if use_llm_judge:
        print(f"  LLM Judge: ENABLED")
    print(f"{'='*70}\n")

    results = []
    for i, item in enumerate(dataset, 1):
        entry = evaluate_single(item, i, n, use_llm_judge=use_llm_judge)
        results.append(entry)

    # ── Aggregate ──
    summary   = aggregate_results(results)
    by_domain = aggregate_by_group(results, "domain")
    by_lang   = aggregate_by_group(results, "language")

    # ── Save JSON results ──
    output = {
        "metadata": {
            "timestamp":      datetime.now().isoformat(),
            "total_questions": n,
            "llm_judge":       use_llm_judge,
        },
        "summary":    summary,
        "by_domain":  by_domain,
        "by_language": by_lang,
        "results":    results,
    }

    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Generate Markdown report ──
    report_md = generate_report(summary, by_domain, by_lang, results)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    # ── Console summary ──
    ret = summary.get("retrieval", {})
    gen = summary.get("generation", {})
    conf = summary.get("confidence_dist", {})

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Questions:          {summary['total_questions']}")
    print(f"  Avg Response Time:  {summary['avg_response_time_s']:.2f}s")
    print(f"  Total Time:         {summary['total_time_s']:.1f}s")
    print(f"")
    print(f"  ── Retrieval ──")
    print(f"  Precision@K:  {ret.get('avg_precision_at_k', 0):.4f}")
    print(f"  Recall@K:     {ret.get('avg_recall_at_k', 0):.4f}")
    print(f"  Hit Rate:     {ret.get('avg_hit_rate', 0):.4f}")
    print(f"  MRR:          {ret.get('avg_mrr', 0):.4f}")
    print(f"  NDCG@K:       {ret.get('avg_ndcg_at_k', 0):.4f}")
    print(f"")
    print(f"  ── Generation ──")
    print(f"  Faithfulness:    {gen.get('avg_faithfulness', 0):.4f}")
    print(f"  Relevancy:       {gen.get('avg_relevancy', 0):.4f}")
    print(f"  Correctness:     {gen.get('avg_correctness', 0):.4f}")
    print(f"  Language Match:  {gen.get('avg_language_match', 0):.4f}")
    print(f"  Completeness:    {gen.get('avg_completeness', 0):.4f}")
    print(f"  Conciseness:     {gen.get('avg_conciseness', 0):.4f}")
    print(f"  Grounded:        {summary['grounded_pct']:.1f}%")
    print(f"")
    print(f"  ── Thresholds ──")
    print(f"  Pass Rate:    {summary.get('threshold_pass_rate', 0):.1f}%")
    top_flags = summary.get("top_failure_flags", [])
    if top_flags:
        flags_str = ", ".join(f"{f}({c})" for f, c in top_flags[:3])
        print(f"  Top Failures: {flags_str}")
    print(f"")
    print(f"  ── Confidence ──")
    print(f"  High: {conf.get('High', 0)}  Medium: {conf.get('Medium', 0)}  Low: {conf.get('Low', 0)}")

    if summary.get("llm_judge"):
        judge = summary["llm_judge"]
        print(f"")
        print(f"  ── LLM Judge (0.0-1.0) ──")
        print(f"  Overall:  {judge.get('avg_overall_score', 0):.2f}")
        print(f"  Faith:    {judge.get('avg_faithfulness', 0):.2f}")
        print(f"  Correct:  {judge.get('avg_correctness', 0):.2f}")

    print(f"")
    print(f"  Results:  {RESULTS_PATH}")
    print(f"  Report:   {REPORT_PATH}")
    print(f"{'='*70}\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual RAG Evaluation Harness")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-based judge scoring")
    parser.add_argument("--quick", action="store_true", help="Run first 10 questions only")
    args = parser.parse_args()

    run_evaluation(use_llm_judge=args.llm_judge, quick=args.quick)
