"""
Multilingual AI Assistant for Tirumala
Document Processing Pipeline — Configuration
"""

import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
OUTPUT_DIR   = PROJECT_ROOT / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
IMAGES_DIR   = OUTPUT_DIR / "page_images"
EXTRACTED_TEXT_DIR = OUTPUT_DIR / "extracted_text"

for d in [OUTPUT_DIR, VECTOR_STORE_DIR, IMAGES_DIR, EXTRACTED_TEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Agent Document Folders
# ──────────────────────────────────────────────
AGENTS_DIR = DATA_DIR
AGENT_FOLDERS = {
    "about_tirumala": DATA_DIR,
}

# ──────────────────────────────────────────────
# OCR Settings
# ──────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.85
VLM_CONFIDENCE_THRESHOLD = 0.85

PADDLEOCR_LANG = "en"

# Auto-detect GPU availability
import torch as _torch
USE_GPU = _torch.cuda.is_available()

# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Use GPU for embeddings if available
EMBEDDING_DEVICE = "cuda" if USE_GPU else "cpu"


# ──────────────────────────────────────────────
# Supported Extensions
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}