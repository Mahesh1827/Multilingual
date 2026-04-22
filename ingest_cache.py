"""
Ingest Cache Script
===================
Reads all processed document chunks from the JSON cache and uploads 
them to the Qdrant Docker container. This avoids re-running the OCR.
"""

import logging
from pathlib import Path
import json
from tqdm import tqdm

from ocr.config import EXTRACTED_TEXT_DIR
from ocr.vector_store import build_qdrant_index, save_qdrant_index

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

def ingest_all_cached_docs():
    cache_dir = EXTRACTED_TEXT_DIR / "chunk_cache"
    
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return

    cache_files = list(cache_dir.glob("*.json"))
    logger.info(f"Found {len(cache_files)} cached document files.")

    all_chunks = []
    
    logger.info("Loading chunks from disk...")
    for cf in tqdm(cache_files, desc="Reading cache"):
        try:
            with open(cf, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to read {cf.name}: {e}")

    if not all_chunks:
        logger.error("No chunks found in cache.")
        return

    logger.info(f"Total chunks to index: {len(all_chunks)}")
    
    # This will trigger the build_qdrant_index function in vector_store.py
    # Since we updated config.py to QDRANT_MODE='server', it will push to Docker.
    vectorstore, metadata = build_qdrant_index(all_chunks)

    if vectorstore:
        save_qdrant_index(vectorstore, metadata)
        logger.info("\n✅ Migration to Qdrant Docker Complete!")
    else:
        logger.error("\n❌ Migration failed.")

if __name__ == "__main__":
    ingest_all_cached_docs()
