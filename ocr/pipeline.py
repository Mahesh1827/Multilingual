"""
Document Processing Pipeline — Main Orchestrator
PDF/DOCX → Text Layer Detection → OCR → Web Search
→ Clean → Chunk → Embed → FAISS
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

from ocr.config import (
    AGENT_FOLDERS,
    OCR_CONFIDENCE_THRESHOLD,
    PADDLEOCR_LANG,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)

from ocr.extractors import extract_from_file, pdf_page_to_image
from ocr.ocr_engine import run_paddle_ocr
from ocr.text_processing import prepare_chunks_with_metadata, prepare_table_chunks_with_metadata
from ocr.vector_store import build_faiss_index, save_faiss_index


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_agent_category(file_path: Path) -> str:
    for category, folder in AGENT_FOLDERS.items():
        try:
            file_path.relative_to(folder)
            return category
        except ValueError:
            continue
    return "unknown"


def collect_all_documents() -> list[Path]:
    files = []

    for category, folder in AGENT_FOLDERS.items():

        if not folder.exists():
            logger.warning(f"Folder not found: {folder}")
            continue

        for ext in SUPPORTED_EXTENSIONS:

            found = [
                p for p in folder.rglob(f"*{ext}")
                if not p.name.startswith("~$")
            ]

            files.extend(found)

            if found:
                logger.info(f"{category}: {len(found)} {ext} files")

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
    all_chunks = []

    total_pages = 0
    processed_pages = 0
    empty_pages = 0
    failed_pages = 0
    table_count = 0

    logger.info(f"\nProcessing: {file_path.name} [{agent_category}]")

    page_results = extract_from_file(file_path)

    for page_data in page_results:

        total_pages += 1

        page_num = page_data["page"]
        text = page_data["text"]
        method = page_data["method"]

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

                # 🔥 FORCE GPU USAGE
                from ocr.config import USE_GPU
                ocr_result = run_paddle_ocr(
                    img,
                    PADDLEOCR_LANG,
                    USE_GPU  # 🚀 FORCE GPU (ignore config)
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
            )

            all_chunks.extend(chunks)

    logger.info("\n---------------------------------------")
    logger.info(f"Document: {file_path.name}")
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
    logger.info("TIRUMALA DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 60)

    files = collect_all_documents()

    logger.info(f"\nTotal documents: {len(files)}\n")

    if not files:
        logger.error("No documents found.")
        return

    all_documents = []

    for i, file_path in enumerate(tqdm(files, desc="Processing documents"), start=1):

        logger.info("\n================================================")
        logger.info(f"Document {i}/{len(files)} → {file_path.name}")
        logger.info("================================================")

        try:
            chunks = process_single_file(file_path)
            all_documents.extend(chunks)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    logger.info(f"\nTotal chunks extracted: {len(all_documents)}")

    if not all_documents:
        logger.error("No text extracted from documents.")
        return

    logger.info("\nBuilding FAISS index...")

    vectorstore, metadata = build_faiss_index(all_documents)

    if vectorstore is not None:
        save_faiss_index(vectorstore, metadata)

        logger.info("\nPipeline complete!")
        logger.info(f"Vectors: {vectorstore.index.ntotal}")
        logger.info(f"Documents processed: {len(files)}")

    else:
        logger.error("FAISS index build failed.")


if __name__ == "__main__":
    run_pipeline()