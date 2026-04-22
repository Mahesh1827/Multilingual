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
DATA_DIR     = PROJECT_ROOT / "Data_main"
OUTPUT_DIR   = PROJECT_ROOT / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
IMAGES_DIR   = OUTPUT_DIR / "page_images"
EXTRACTED_TEXT_DIR = OUTPUT_DIR / "extracted_text"

for d in [OUTPUT_DIR, VECTOR_STORE_DIR, IMAGES_DIR, EXTRACTED_TEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Language Folder Mappings
# Each subfolder in Data_main maps to an ISO 639-1 code
# and a PaddleOCR language code.
# ──────────────────────────────────────────────
LANGUAGE_FOLDERS = {
    "English": {"iso": "en", "paddle_lang": "en"},
    "Telugu":  {"iso": "te", "paddle_lang": "te"},
    "Hindi":   {"iso": "hi", "paddle_lang": "hi"},
    "Tamil":   {"iso": "ta", "paddle_lang": "ta"},
}

# ──────────────────────────────────────────────
# Agent Document Folders
# Maps agent category keys → folder paths.
# Each language subfolder is treated as an agent category
# for domain routing.
# ──────────────────────────────────────────────
AGENT_FOLDERS = {}
for folder_name in LANGUAGE_FOLDERS:
    folder_path = DATA_DIR / folder_name
    if folder_path.exists():
        AGENT_FOLDERS[folder_name.lower()] = folder_path

# ──────────────────────────────────────────────
# OCR Settings
# ──────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.85
VLM_CONFIDENCE_THRESHOLD = 0.85

# Default PaddleOCR lang — overridden per-document at runtime
PADDLEOCR_LANG = "en"

# Auto-detect GPU availability (Safe for torch/embeddings now that PaddleOCR is isolated)
import torch as _torch
USE_GPU = _torch.cuda.is_available()

# Enabled GPU for PaddleOCR using stable combination (torch 2.5.1 / paddleocr 2.9.1)
PADDLEOCR_USE_GPU = True

# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# ──────────────────────────────────────────────
# Embedding Model — MULTILINGUAL
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Use GPU for embeddings if available
EMBEDDING_DEVICE = "cuda" if USE_GPU else "cpu"

# ──────────────────────────────────────────────
# Qdrant Vector Database (Docker Server)
# ──────────────────────────────────────────────
QDRANT_URL = "http://localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "tirumala_multilingual"

# ──────────────────────────────────────────────
# Supported Extensions
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}