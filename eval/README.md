# Multilingual RAG Evaluation Framework

Production-grade evaluation harness for the Tirumala AI Assistant.

## Quick Start

```bash
# Standard evaluation (all 54 test cases)
python -m eval.run_eval

# Quick test (first 10 questions)
python -m eval.run_eval --quick

# With LLM judge scoring (slower, uses Groq/Ollama)
python -m eval.run_eval --llm-judge
```

## Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of retrieved chunks that are relevant |
| **Recall@K** | Fraction of relevant chunks that were retrieved |
| **Hit Rate** | Did at least one relevant chunk appear? (0 or 1) |
| **MRR** | Mean Reciprocal Rank — position of first relevant result |
| **NDCG@K** | Normalized Discounted Cumulative Gain — ranking quality |

### Generation Metrics
| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (F1 overlap) |
| **Relevancy** | Does the answer address the user's question? (F1 overlap) |
| **Correctness** | Is the answer factually correct vs expected? (F1 overlap) |
| **Language Match** | Is the response in the correct language? (0 or 1) |
| **Completeness** | Are all sub-questions addressed? (0-1 ratio) |
| **Conciseness** | Is the answer appropriately concise? (0-1, penalizes bloat) |

### LLM Judge (Optional)
When `--llm-judge` is enabled, each answer is also scored 1-5 by the LLM on:
Faithfulness, Relevancy, Language Match, Completeness, Conciseness.

## Output Files

| File | Contents |
|------|----------|
| `results.json` | Per-query detailed results with all metrics |
| `report.md` | Human-readable summary with per-domain and per-language breakdown |

## Golden Dataset

`golden_dataset.json` contains 54 test cases covering:
- **29** English factual questions (all 6 domains)
- **12** Native-script Indic questions (3 per language: te, hi, ta, kn)
- **8** Romanized Indic questions (2 per language)
- **1** Language override test case
- **2** Out-of-scope rejection tests
- **3** Conversational/greeting tests

### Adding New Test Cases

```json
{
  "question": "Your question here",
  "expected_answer": "The expected ground-truth answer",
  "domain": "pilgrimage_seva",
  "language": "te",
  "relevant_keywords": ["key", "terms", "for", "retrieval"],
  "sub_questions": ["optional", "for multi-part queries"],
  "note": "Human-readable description of this test case"
}
```

## Architecture

```
eval/
├── __init__.py              — Module exports
├── golden_dataset.json      — 54 multilingual test cases
├── metrics/
│   ├── __init__.py
│   ├── retrieval_metrics.py — Precision@K, Recall@K, Hit Rate, MRR, NDCG
│   └── generation_metrics.py — Faithfulness, Relevancy, Correctness, etc.
├── judge.py                 — LLM-based pairwise judge (5-criterion scoring)
├── eval_logger.py           — Structured [LOG] metadata block generator
├── run_eval.py              — Main evaluation harness (CLI)
├── results.json             — (generated) Detailed per-query results
├── report.md                — (generated) Human-readable summary
└── README.md                — This file
```
