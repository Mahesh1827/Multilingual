"""
OCR Worker — PaddleOCR Subprocess Worker
=========================================
This file is designed to run as a STANDALONE SUBPROCESS — never imported
directly by the main pipeline. It has ZERO torch/PyTorch imports, which
is what allows PaddleOCR to use the GPU safely without the pybind11/CUDA
clash that occurs when both frameworks share the same Python process.

Usage (internal — called by ocr_engine.py bridge):
    python ocr_worker.py --lang en --gpu < image_bytes_stdin
    Outputs: JSON to stdout  {text, confidence, details}

Do NOT import this file. It is only ever launched via subprocess.
"""

import os
import sys
import json
import logging
import threading
import argparse
import base64
import io

# ──────────────────────────────────────────────
# ENV FIXES (must be set before PaddleOCR loads)
# ──────────────────────────────────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "False"

import numpy as np
import cv2
from PIL import Image

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Per-Language PaddleOCR Cache
# ──────────────────────────────────────────────

_paddle_ocr_cache: dict = {}
_lock = threading.Lock()


def get_paddle_ocr(lang: str = "en", use_gpu: bool = True):
    """
    Get or create a PaddleOCR instance for the given language.
    Runs on GPU (safe here — no PyTorch in this process).
    """
    cache_key = f"{lang}_{use_gpu}"
    if cache_key in _paddle_ocr_cache:
        return _paddle_ocr_cache[cache_key]

    with _lock:
        if cache_key in _paddle_ocr_cache:
            return _paddle_ocr_cache[cache_key]

        mode = "GPU" if use_gpu else "CPU"
        logger.info(f"Loading PaddleOCR (lang={lang}, {mode} mode)")

        from paddleocr import PaddleOCR

        _paddle_ocr_cache[cache_key] = PaddleOCR(
            use_angle_cls=False,
            lang=lang,
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_gpu=use_gpu,
            show_log=False,
        )

    return _paddle_ocr_cache[cache_key]


# ──────────────────────────────────────────────
# Core OCR Logic
# ──────────────────────────────────────────────

def run_paddle_ocr(image, lang: str = "en", use_gpu: bool = True) -> dict:
    """
    Run PaddleOCR on a numpy image array.
    Returns: {"text": str, "confidence": float, "details": list}
    """
    if image is None:
        return {"text": "", "confidence": 0.0, "details": []}

    try:
        img = np.array(image)
        if img.size == 0:
            raise ValueError("Empty image array")
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        return {"text": "", "confidence": 0.0, "details": []}

    # Convert to BGR for PaddleOCR v2
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]

    if h < 40 or w < 40:
        logger.warning("Image too small for OCR")
        return {"text": "", "confidence": 0.0, "details": []}

    # Resize very large images (GPU memory management)
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ocr = get_paddle_ocr(lang=lang, use_gpu=use_gpu)

    logger.info(f"Running PaddleOCR on {w}x{h} image (lang={lang}, gpu={use_gpu})")

    try:
        results = ocr.ocr(img, cls=False)
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e} — retrying at 75%")
        try:
            small = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            results = ocr.ocr(small, cls=False)
            logger.info("PaddleOCR retry succeeded")
        except Exception as e2:
            logger.error(f"PaddleOCR retry failed: {e2}")
            return {"text": "", "confidence": 0.0, "details": []}

    if not results or results == [None]:
        return {"text": "", "confidence": 0.0, "details": []}

    lines, confidences, details = [], [], []

    for page in results:
        if page is None:
            continue
        for item in page:
            bbox, (text, conf) = item
            if not text.strip():
                continue
            lines.append(text)
            confidences.append(conf)
            details.append({
                "text": text,
                "confidence": conf,
                "bbox": [list(pt) for pt in bbox],  # make bbox JSON-serializable
            })

    if not lines:
        return {"text": "", "confidence": 0.0, "details": []}

    avg_conf = sum(confidences) / len(confidences)
    logger.info(f"PaddleOCR [{lang}]: {len(lines)} lines, avg confidence: {avg_conf:.3f}")

    return {
        "text": "\n".join(lines),
        "confidence": avg_conf,
        "details": details,
    }


# ──────────────────────────────────────────────
# Subprocess Entry Point
# ──────────────────────────────────────────────

def main():
    """
    Entry point when run as a subprocess.

    Protocol:
      stdin  → base64-encoded PNG image bytes
      stdout → JSON result: {text, confidence, details}
      stderr → logs (forwarded by parent process)
    """
    parser = argparse.ArgumentParser(description="PaddleOCR Worker")
    parser.add_argument("--lang",    type=str,  default="en",   help="OCR language code")
    parser.add_argument("--gpu",     action="store_true",        help="Use GPU for PaddleOCR")
    args = parser.parse_args()

    # Redirect stdout → stderr during PaddleOCR loading so that
    # the GPU warnings PaddleOCR prints don't corrupt our JSON output.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        # Read base64-encoded image from stdin
        raw = sys.stdin.buffer.read()
        if not raw:
            sys.stdout = real_stdout
            print(json.dumps({"text": "", "confidence": 0.0, "details": [], "error": "No input"}))
            sys.exit(0)

        # Decode base64 → PIL Image → numpy
        img_bytes = base64.b64decode(raw)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        result = run_paddle_ocr(image, lang=args.lang, use_gpu=args.gpu)

        # Restore real stdout for the JSON result
        sys.stdout = real_stdout
        print(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()

    except Exception as e:
        sys.stdout = real_stdout
        logger.error(f"Worker fatal error: {e}")
        print(json.dumps({"text": "", "confidence": 0.0, "details": [], "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()