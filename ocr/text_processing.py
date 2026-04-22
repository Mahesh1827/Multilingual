"""
Text Processing Module — Cleaning, Chunking, Metadata Tagging
Prepares extracted text for embedding and vector storage.

Uses sentence-aware splitting (nltk) to prevent mid-sentence cuts,
with configurable overlap between adjacent chunks.
Includes automatic language detection for multilingual metadata.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Language Detection
# ──────────────────────────────────────────────

def detect_language(text: str, fallback: str = "en") -> str:
    """
    Detect the language of a text string.
    Returns an ISO 639-1 code (e.g. 'en', 'te', 'hi', 'ta').
    Falls back to the provided default if detection fails.
    """
    if not text or len(text.strip()) < 20:
        return fallback
    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception:
        return fallback


# ──────────────────────────────────────────────
# Sentence tokenizer (lazy-loaded)
# ──────────────────────────────────────────────
_sent_tokenizer_ready = False

def _ensure_nltk():
    """Download punkt_tab tokenizer if not already present."""
    global _sent_tokenizer_ready
    if _sent_tokenizer_ready:
        return
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _sent_tokenizer_ready = True
    except Exception as e:
        logger.warning(f"NLTK setup failed: {e}. Falling back to regex sentence splitting.")
        _sent_tokenizer_ready = True  # don't retry


def _sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences using nltk, with regex fallback."""
    try:
        _ensure_nltk()
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Regex fallback: split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Clean extracted text by removing artifacts and normalizing whitespace."""
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page number patterns like "Page 1", "- 1 -", etc.
    text = re.sub(r"(?i)(?:page\s*\d+|\-\s*\d+\s*\-)", "", text)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Strip each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Remove leading/trailing whitespace
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[str]:
    """
    Split text into overlapping chunks using sentence-aware boundaries.
    
    1. Tokenize text into sentences
    2. Group sentences into chunks up to chunk_size characters
    3. Overlap by including trailing sentences from the previous chunk
    """
    sentences = _sentence_tokenize(text)
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for sent in sentences:
        sent_len = len(sent)
        
        # If single sentence exceeds chunk_size, add it as its own chunk
        if sent_len > chunk_size:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            chunks.append(sent)
            current_chunk_sentences = []
            current_length = 0
            continue
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + sent_len + 1 > chunk_size and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
            # Overlap: keep trailing sentences from current chunk that fit within overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s) + 1 <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_length = sum(len(s) for s in current_chunk_sentences) + max(len(current_chunk_sentences) - 1, 0)
        
        current_chunk_sentences.append(sent)
        current_length += sent_len + (1 if current_length > 0 else 0)
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return chunks


def prepare_chunks_with_metadata(
    text: str,
    source_file: Path,
    page_num: int,
    agent_category: str,
    extraction_method: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    folder_language: str = "en",
) -> list[dict]:
    """
    Clean, chunk, and attach metadata to text.
    Returns a list of dicts ready for embedding.
    
    Args:
        folder_language: ISO 639-1 code inferred from the folder name.
                         Used as the primary language tag; langdetect is
                         used as a secondary confirmation.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    chunks = chunk_text(cleaned, chunk_size, chunk_overlap)

    # Detect language from the first substantial chunk for confirmation
    detected_lang = detect_language(cleaned[:500], fallback=folder_language)
    # Trust folder language as primary, but log mismatches
    language = folder_language
    if detected_lang != folder_language and len(cleaned) > 100:
        logger.debug(
            f"Language mismatch: folder={folder_language}, detected={detected_lang} "
            f"in {source_file.name} p.{page_num}. Using folder language."
        )

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "source_file": source_file.name,
                "source_path": str(source_file),
                "page": page_num,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "agent_category": agent_category,
                "extraction_method": extraction_method,
                "language": language,
                "content_type": "text",
            },
        })

    logger.info(
        f"{source_file.name} p.{page_num}: {len(chunks)} chunks "
        f"(cleaned {len(text)} → {len(cleaned)} chars) [lang={language}, method={extraction_method}]"
    )
    return documents


def prepare_table_chunks_with_metadata(
    tables: list[dict],
    source_file: Path,
    agent_category: str,
    folder_language: str = "en",
) -> list[dict]:
    """
    Prepare table data as individual chunks (never split mid-table).
    Each table becomes a single chunk tagged with content_type: "table".
    
    Args:
        tables: list of table dicts from extractors.extract_tables_from_page()
        source_file: Path to the source document
        agent_category: agent category for routing
        folder_language: ISO 639-1 code inferred from the folder name
    
    Returns:
        list of dicts ready for embedding
    """
    documents = []
    for table in tables:
        text = table.get("text", "")
        if not text.strip():
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source_file": source_file.name,
                "source_path": str(source_file),
                "page": table.get("page", 0),
                "chunk_index": table.get("table_index", 0),
                "total_chunks": 1,
                "agent_category": agent_category,
                "extraction_method": "table_extraction",
                "content_type": "table",
                "table_headers": table.get("headers", []),
                "table_num_rows": table.get("num_rows", 0),
                "language": folder_language,
            },
        })

    if documents:
        logger.info(
            f"{source_file.name}: {len(documents)} table chunks prepared [lang={folder_language}]"
        )
    return documents