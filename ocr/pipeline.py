"""
Document Processing Pipeline — Main Orchestrator
PDF/DOCX → Text Layer Detection → OCR → Web Search
→ Clean → Chunk → Embed → Qdrant

Multilingual: scans Data_main/{English,Telugu,Hindi,Tamil}/
and assigns the correct language code to each document.
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

import json

from ocr.config import (
    DATA_DIR,
    LANGUAGE_FOLDERS,
    AGENT_FOLDERS,
    OCR_CONFIDENCE_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    PADDLEOCR_USE_GPU,
    EXTRACTED_TEXT_DIR,
)

from ocr.extractors import extract_from_file, pdf_page_to_image
from ocr.ocr_engine import run_paddle_ocr
from ocr.text_processing import prepare_chunks_with_metadata, prepare_table_chunks_with_metadata
from ocr.vector_store import build_qdrant_index, save_qdrant_index


# ──────────────────────────────────────────────
# Logging with tqdm conflict resolution
# ──────────────────────────────────────────────

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Use tqdm.write to ensure logs don't clash with progress bar
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

# Configure Root Logger
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

# Clear default handlers to avoid duplicates
for handler in _root_logger.handlers[:]:
    _root_logger.removeHandler(handler)

# Add TQDM-safe handler
_tqdm_handler = TqdmLoggingHandler()
_tqdm_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
_root_logger.addHandler(_tqdm_handler)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Document-Level Cache (resume after crash)
# ──────────────────────────────────────────────

CHUNK_CACHE_DIR = EXTRACTED_TEXT_DIR / "chunk_cache"


def _cache_path(file_path: Path) -> Path:
    """Return the JSON cache file path for a given source document."""
    safe_name = file_path.stem.replace(" ", "_") + "_" + str(abs(hash(str(file_path)))) + ".json"
    return CHUNK_CACHE_DIR / safe_name


def _save_chunk_cache(file_path: Path, chunks: list[dict]) -> None:
    """Save extracted chunks to disk so they can be reloaded on resume."""
    CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(file_path)
    # Convert Path objects to strings for JSON serialisation
    serialisable = [
        {"text": c["text"], "metadata": {k: str(v) if isinstance(v, Path) else v
                                          for k, v in c["metadata"].items()}}
        for c in chunks
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False)
    logger.info(f"Cache saved → {cache_file.name} ({len(chunks)} chunks)")


def _load_chunk_cache(file_path: Path) -> list[dict] | None:
    """Load chunks from disk cache, or return None if cache does not exist."""
    cache_file = _cache_path(file_path)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Cache HIT → {file_path.name} ({len(chunks)} chunks, skipping OCR)")
        return chunks
    return None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_language_info(file_path: Path) -> dict:
    """
    Determine the language info for a file based on its parent folder.
    Returns {"iso": "te", "paddle_lang": "te", "folder": "Telugu"} etc.
    Falls back to English if the folder is not recognized.
    """
    for folder_name, lang_info in LANGUAGE_FOLDERS.items():
        folder_path = DATA_DIR / folder_name
        try:
            file_path.relative_to(folder_path)
            return {**lang_info, "folder": folder_name}
        except ValueError:
            continue
    return {"iso": "en", "paddle_lang": "en", "folder": "unknown"}


def get_agent_category(file_path: Path) -> str:
    """Map a file to its agent category (based on language folder)."""
    for category, folder in AGENT_FOLDERS.items():
        try:
            file_path.relative_to(folder)
            return category
        except ValueError:
            continue
    return "unknown"


def collect_all_documents() -> list[Path]:
    """
    Collect all supported documents from all language subfolders in Data_main.
    """
    files = []

    for folder_name, lang_info in LANGUAGE_FOLDERS.items():
        folder_path = DATA_DIR / folder_name

        if not folder_path.exists():
            logger.warning(f"Language folder not found: {folder_path}")
            continue

        for ext in SUPPORTED_EXTENSIONS:
            found = [
                p for p in folder_path.rglob(f"*{ext}")
                if not p.name.startswith("~$")
            ]

            files.extend(found)

            if found:
                logger.info(
                    f"{folder_name} ({lang_info['iso']}): "
                    f"{len(found)} {ext} files"
                )

    return sorted(set(files))


# ──────────────────────────────────────────────
# Fast OCR Skip Detection
# ──────────────────────────────────────────────

def page_likely_has_text(image):

    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img, 50, 150)

    edge_ratio = edges.sum() / (img.shape[0] * img.shape[1])

    return edge_ratio > 0.015


# ──────────────────────────────────────────────
# File Processing
# ──────────────────────────────────────────────

def process_single_file(file_path: Path) -> list[dict]:

    agent_category = get_agent_category(file_path)
    lang_info = get_language_info(file_path)
    folder_language = lang_info["iso"]
    paddle_lang = lang_info["paddle_lang"]

    all_chunks = []

    total_pages = 0
    processed_pages = 0
    empty_pages = 0
    failed_pages = 0
    table_count = 0

    logger.info("=" * 60)
    logger.info(f"STARTING: {file_path.name}")
    logger.info(f"Folder Lang: {folder_language} | Category: {agent_category}")
    logger.info("-" * 60)

    page_results = extract_from_file(file_path)

    for page_data in page_results:

        total_pages += 1

        page_num = page_data["page"]
        text = page_data["text"]
        method = page_data["method"]

        logger.info(f"==> Processing Page {page_num}/{len(page_results)} (method: {method}) ...")

        if method == "needs_ocr":

            try:
                image = pdf_page_to_image(file_path, page_num - 1)

                if image is None:
                    failed_pages += 1
                    logger.warning(f"{file_path.name} | Page {page_num}: image extraction failed")
                    continue

                img = np.array(image)

                if img.size == 0:
                    failed_pages += 1
                    logger.warning(f"{file_path.name} | Page {page_num}: empty image detected")
                    continue

                # 🔥 GPU optimization: resize large images
                if img.shape[0] > 2000:
                    img = cv2.resize(img, None, fx=0.5, fy=0.5)

                if not page_likely_has_text(img):
                    empty_pages += 1
                    logger.info(f"{file_path.name} | Page {page_num}: blank page → skipping OCR")
                    continue

                # Use per-language PaddleOCR
                ocr_result = run_paddle_ocr(
                    img,
                    lang=paddle_lang,
                    use_gpu=PADDLEOCR_USE_GPU,
                )

                text = ocr_result.get("text", "")
                confidence = ocr_result.get("confidence", 0.0)

                method = "paddleocr"

                if not text.strip():
                    empty_pages += 1
                    logger.warning(f"{file_path.name} | Page {page_num}: OCR returned empty text")
                    continue

                if confidence < OCR_CONFIDENCE_THRESHOLD:

                    logger.info(
                        f"{file_path.name} | Page {page_num}: low OCR confidence ({confidence:.2f})"
                    )

                    if len(text.strip()) < 30:
                        logger.info(
                            f"{file_path.name} | Page {page_num}: low quality OCR → keeping raw text"
                        )
            except Exception as e:
                failed_pages += 1
                logger.error(f"{file_path.name} | Page {page_num}: OCR failed → {e}")
                continue

        # ── Table chunks (extracted separately, never split) ──
        tables = page_data.get("tables", [])
        if tables:
            table_chunks = prepare_table_chunks_with_metadata(
                tables=tables,
                source_file=file_path,
                agent_category=agent_category,
                folder_language=folder_language,
            )
            all_chunks.extend(table_chunks)
            table_count += len(table_chunks)

        if text.strip():

            processed_pages += 1

            chunks = prepare_chunks_with_metadata(
                text=text,
                source_file=file_path,
                page_num=page_num,
                agent_category=agent_category,
                extraction_method=method,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                folder_language=folder_language,
            )

            all_chunks.extend(chunks)

    logger.info("\n---------------------------------------")
    logger.info(f"Document: {file_path.name}")
    logger.info(f"Language         : {folder_language}")
    logger.info(f"Total pages      : {total_pages}")
    logger.info(f"Processed pages  : {processed_pages}")
    logger.info(f"Empty OCR pages  : {empty_pages}")
    logger.info(f"Failed pages     : {failed_pages}")
    logger.info(f"Table chunks     : {table_count}")
    logger.info(f"Total chunks     : {len(all_chunks)}")
    logger.info("---------------------------------------\n")

    return all_chunks


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def run_pipeline():

    logger.info("=" * 60)
    logger.info("TIRUMALA MULTILINGUAL DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 60)

    files = collect_all_documents()

    # Log language distribution
    lang_counts = {}
    for f in files:
        lang = get_language_info(f)["iso"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    logger.info(f"\nTotal documents: {len(files)}")
    for lang, count in sorted(lang_counts.items()):
        logger.info(f"  {lang}: {count} documents")
    logger.info("")

    if not files:
        logger.error("No documents found.")
        return

    all_documents = []

    cached_count = 0
    processed_count = 0

    for i, file_path in enumerate(tqdm(files, desc="Processing documents"), start=1):

        logger.info("\n================================================")
        logger.info(f"Document {i}/{len(files)} → {file_path.name}")
        logger.info("================================================")

        try:
            # ── Try loading from cache first ──
            cached = _load_chunk_cache(file_path)
            if cached is not None:
                all_documents.extend(cached)
                cached_count += 1
                continue

            # ── Full OCR/extraction ──
            chunks = process_single_file(file_path)
            all_documents.extend(chunks)
            processed_count += 1

            # ── Save to cache immediately after success ──
            if chunks:
                _save_chunk_cache(file_path, chunks)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    logger.info(f"\nDocuments loaded from cache : {cached_count}")
    logger.info(f"Documents freshly processed: {processed_count}")

    logger.info(f"\nTotal chunks extracted: {len(all_documents)}")

    # Log language distribution of chunks
    chunk_lang_counts = {}
    for doc in all_documents:
        lang = doc["metadata"].get("language", "unknown")
        chunk_lang_counts[lang] = chunk_lang_counts.get(lang, 0) + 1
    for lang, count in sorted(chunk_lang_counts.items()):
        logger.info(f"  {lang}: {count} chunks")

    if not all_documents:
        logger.error("No text extracted from documents.")
        return

    logger.info("\nBuilding Qdrant index with multilingual embeddings...")

    vectorstore, metadata = build_qdrant_index(all_documents)

    if vectorstore is not None:
        save_qdrant_index(vectorstore, metadata)

        logger.info("\n✅ Pipeline complete!")
        logger.info(f"Documents processed: {len(files)}")
        logger.info(f"Total chunks: {len(all_documents)}")
        logger.info(f"Languages: {list(chunk_lang_counts.keys())}")

    else:
        logger.error("Qdrant index build failed.")


if __name__ == "__main__":
    run_pipeline()