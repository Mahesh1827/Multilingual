"""
Quick end-to-end test: Query the Qdrant Docker database
Tests semantic retrieval across multiple languages.
"""
import _env_setup  # noqa: F401

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from ocr.vector_store import load_qdrant_index

print("=" * 60)
print("  Multilingual Semantic Search Test")
print("=" * 60)

# Load the Qdrant index from Docker
print("\n[1/4] Connecting to Qdrant Docker...")
vectorstore, metadata = load_qdrant_index()
print(f"  ✅ Connected! Metadata contains {len(metadata)} entries.")

# Test queries in multiple languages
test_queries = [
    ("English",  "What is the history of Tirumala temple?"),
    ("English",  "How to book darshan tickets at TTD?"),
    ("Telugu",   "తిరుమల ఆలయ చరిత్ర ఏమిటి?"),      # "What is Tirumala temple history?"
    ("Hindi",    "तिरुमाला मंदिर का इतिहास क्या है?"),  # "What is Tirumala temple history?"
]

for lang_name, query in test_queries:
    print(f"\n{'─' * 60}")
    print(f"  [{lang_name}] Query: {query}")
    print(f"{'─' * 60}")

    results = vectorstore.similarity_search_with_score(query, k=3)

    if not results:
        print("  ❌ No results found!")
        continue

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source_file", "?")
        page = doc.metadata.get("page", "?")
        doc_lang = doc.metadata.get("language", "?")
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  Result {i}: [score={score:.4f}] [{doc_lang}] {source} (p.{page})")
        print(f"    → {snippet}...")

print(f"\n{'=' * 60}")
print("  ✅ All tests complete!")
print(f"{'=' * 60}")
