"""
Evaluation Metrics Module
Retrieval + Generation metrics for multilingual RAG evaluation.
"""

from eval.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    compute_all_retrieval_metrics,
)

from eval.metrics.generation_metrics import (
    faithfulness_score,
    answer_relevancy_score,
    answer_correctness_score,
    language_match_score,
    completeness_score,
    conciseness_score,
    compute_all_generation_metrics,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_rate",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "compute_all_retrieval_metrics",
    "faithfulness_score",
    "answer_relevancy_score",
    "answer_correctness_score",
    "language_match_score",
    "completeness_score",
    "conciseness_score",
    "compute_all_generation_metrics",
]
