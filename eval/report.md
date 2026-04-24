# Multilingual RAG Evaluation Report

**Generated:** 2026-04-24 15:02:20
**Total Questions:** 56
**Total Time:** 220.6s
**Avg Response Time:** 3.9398s

## Overall Metrics

### Retrieval Metrics

| Metric | Score |
|--------|-------|
| Precision At K | 0.4786 |
| Recall At K | 0.4821 |
| Hit Rate | 0.4821 |
| Mrr | 0.4732 |
| Ndcg At K | 0.4779 |

### Generation Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | 0.9799 |
| Relevancy | 0.8014 |
| Correctness | 0.5276 |
| Language Match | 0.9821 |
| Completeness | 0.9423 |
| Conciseness | 0.8187 |

**Grounded:** 1.8%
**Threshold Pass Rate:** 7.1%

### Top Failure Patterns

| Criterion | Failures |
|-----------|----------|
| Correctness | 51 |
| Conciseness | 14 |
| Relevancy | 6 |
| Faithfulness | 1 |
| Language Match | 1 |

### Confidence Distribution

| Level | Count |
|-------|-------|
| High | 55 |
| Medium | 0 |
| Low | 1 |

### Query Type Distribution

| Type | Count |
|------|-------|
| conversational | 3 |
| factual | 47 |
| out-of-scope | 2 |
| procedural | 4 |

---
## Per-Domain Breakdown

### About Tirumala (9 questions)

- Faithfulness: 1.0000
- Correctness: 0.5945
- Avg Response Time: 1.73s

### Conversational (3 questions)

- Faithfulness: 0.6667
- Correctness: 0.4824
- Avg Response Time: 41.97s

### Devotional Practice (6 questions)

- Faithfulness: 1.0000
- Correctness: 0.5238
- Avg Response Time: 0.29s

### Festival Events (5 questions)

- Faithfulness: 1.0000
- Correctness: 0.5534
- Avg Response Time: 0.21s

### Knowledge Scripture (4 questions)

- Faithfulness: 1.0000
- Correctness: 0.5650
- Avg Response Time: 0.12s

### Out Of Scope (2 questions)

- Faithfulness: 1.0000
- Correctness: 0.5576
- Avg Response Time: 0.12s

### Pilgrimage Seva (18 questions)

- Faithfulness: 0.9932
- Correctness: 0.5117
- Avg Response Time: 3.92s

### Temple History (9 questions)

- Faithfulness: 1.0000
- Correctness: 0.4727
- Avg Response Time: 0.56s


---
## Per-Language Breakdown

### English (35 questions)

- Hit Rate: 0.7714
- Faithfulness: 0.9679
- Correctness: 0.5705
- Language Match: 0.9714

### Hindi (6 questions)

- Hit Rate: 0.0000
- Faithfulness: 1.0000
- Correctness: 0.4512
- Language Match: 1.0000

### Kannada (5 questions)

- Hit Rate: 0.0000
- Faithfulness: 1.0000
- Correctness: 0.4643
- Language Match: 1.0000

### Tamil (5 questions)

- Hit Rate: 0.0000
- Faithfulness: 1.0000
- Correctness: 0.4480
- Language Match: 1.0000

### Telugu (5 questions)

- Hit Rate: 0.0000
- Faithfulness: 1.0000
- Correctness: 0.4623
- Language Match: 1.0000


---
## Low-Scoring Queries (Correctness < 0.2)

- **Q:** hello
  - Route: unknown, Correctness: 0.000
  - Answer: ...
