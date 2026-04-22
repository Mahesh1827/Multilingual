"""
OCR Module — Subprocess Bridge (PaddleOCR) + Qwen2.5-VL fallback (GPU)
=======================================================================
Optimized for RAG document processing (Multilingual)
Supports per-language PaddleOCR for Telugu, Hindi, Tamil, English.

Architecture (GPU-safe on Windows):
  ┌─────────────────────────────────────────────────────┐
  │  main_pipeline.py  (this process)                   │
  │  PyTorch  → Qwen2.5-VL  (GPU)                       │
  │  PyTorch  → multilingual-e5-large embeddings (GPU)  │
  │                                                     │
  │  run_paddle_ocr_subprocess()                        │
  │       │  stdin: base64 PNG                          │
  │       ▼                                             │
  │  [ocr_worker.py subprocess]                         │
  │       PaddlePaddle → PaddleOCR (GPU)                │
  │       │  stdout: JSON result                        │
  │       ▼                                             │
  │  Returns {"text", "confidence", "details"}          │
  └─────────────────────────────────────────────────────┘

Two frameworks, two processes — zero pybind11/CUDA clash.
"""

import os
import sys
import json
import base64
import subprocess
import logging
import io
import threading
from pathlib import Path

# ──────────────────────────────────────────────
# ENV FIXES
# ──────────────────────────────────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "False"

import torch

from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Worker script path — resolves relative to this file
# ──────────────────────────────────────────────
_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "ocr_worker.py")


# ──────────────────────────────────────────────
# Image Preprocessing (unchanged from original)
# ──────────────────────────────────────────────

def preprocess_image_for_ocr(image):

    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    if img is None or img.size == 0:
        raise ValueError("Invalid OCR image")

    img = img.astype("uint8")

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    # Resize large pages (GPU-friendly)
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Deskew (skip on very large images to prevent freezing)
    h, w = img.shape[:2]
    if h * w < 4_000_000:
        coords = np.column_stack(np.where(img > 10))

        if len(coords) > 0 and len(coords) < 500_000:
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return img


# ──────────────────────────────────────────────
# PaddleOCR Subprocess Bridge
# ──────────────────────────────────────────────

def _image_to_base64_png(image) -> bytes:
    """
    Convert a numpy array or PIL Image to base64-encoded PNG bytes.
    This is the payload sent to the OCR worker subprocess via stdin.
    """
    if isinstance(image, np.ndarray):
        # numpy BGR → PIL RGB
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        elif image.shape[2] == 4:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue())


def _build_clean_env() -> dict:
    """
    Build a clean environment for the OCR worker subprocess.
    Removes torch lib paths that could cause DLL conflicts with PaddlePaddle CUDA.
    """
    env = os.environ.copy()
    # Strip any torch lib/bin paths from PATH to prevent DLL conflicts
    path_parts = env.get("PATH", "").split(os.pathsep)
    clean_parts = [p for p in path_parts if "torch" not in p.lower()]
    env["PATH"] = os.pathsep.join(clean_parts)
    return env


def _run_worker(
    img_b64: bytes,
    lang: str,
    use_gpu: bool,
    timeout: int,
) -> tuple[int, str, str]:
    """
    Low-level: spawn ocr_worker.py and return (returncode, stdout, stderr).
    """
    cmd = [sys.executable, _WORKER_SCRIPT, "--lang", lang]
    if use_gpu:
        cmd.append("--gpu")

    mode = "GPU" if use_gpu else "CPU"
    logger.info(f"Spawning OCR worker (lang={lang}, gpu={use_gpu})")

    proc = subprocess.run(
        cmd,
        input=img_b64,
        capture_output=True,
        timeout=timeout,
        env=_build_clean_env(),
    )

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()

    # Forward worker stderr to our logger
    if stderr:
        for line in stderr.splitlines():
            logger.debug(f"[ocr_worker] {line}")

    return proc.returncode, stdout, stderr


def run_paddle_ocr_subprocess(
    image,
    lang: str = "en",
    use_gpu: bool = True,
    timeout: int = 120,
) -> dict:
    """
    Run PaddleOCR in an isolated subprocess (ocr_worker.py).

    Returns dict: {"text": str, "confidence": float, "details": list}

    Auto-fallback: If GPU inference crashes (exit code != 0), the call
    is automatically retried with CPU mode so document processing never
    stalls.

    Args:
        image:    numpy array (BGR or grayscale) or PIL Image
        lang:     PaddleOCR language code ("en", "te", "hi", "ta", etc.)
        use_gpu:  Whether PaddleOCR should use GPU (safe in subprocess)
        timeout:  Max seconds to wait for the worker (default: 120s)
    """
    if image is None:
        logger.error("OCR received None image")
        return {"text": "", "confidence": 0.0, "details": []}

    empty_result = {"text": "", "confidence": 0.0, "details": []}

    try:
        img_b64 = _image_to_base64_png(image)
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        return empty_result

    # ── Attempt 1: requested mode (GPU or CPU) ──
    try:
        rc, stdout, stderr = _run_worker(img_b64, lang, use_gpu, timeout)

        if rc != 0 and use_gpu:
            # GPU crashed — auto-fallback to CPU
            logger.warning(
                f"OCR GPU worker crashed (code {rc}). "
                f"Falling back to CPU mode …"
            )
            rc, stdout, stderr = _run_worker(img_b64, lang, False, timeout)

        if rc != 0:
            logger.error(f"OCR worker exited with code {rc}")
            return empty_result

        if not stdout:
            logger.warning("OCR worker produced no output")
            return empty_result

        # PaddleOCR v3 prints warnings to stdout before the JSON.
        # Extract only the last line that looks like JSON.
        json_line = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                json_line = line
                break

        if json_line is None:
            logger.warning("OCR worker produced no JSON output")
            return empty_result

        result = json.loads(json_line)

        if "error" in result:
            logger.error(f"OCR worker error: {result['error']}")
            return empty_result

        logger.info(
            f"PaddleOCR [{lang}] via subprocess: "
            f"{len(result.get('details', []))} lines, "
            f"confidence: {result.get('confidence', 0.0):.3f}"
        )
        return result

    except subprocess.TimeoutExpired:
        logger.error(f"OCR worker timed out after {timeout}s")
        return empty_result

    except json.JSONDecodeError as e:
        logger.error(f"OCR worker returned invalid JSON: {e}")
        return empty_result

    except Exception as e:
        logger.error(f"OCR subprocess failed: {e}")
        return empty_result


# ──────────────────────────────────────────────
# Alias — keeps pipeline.py import unchanged
# ──────────────────────────────────────────────

def run_paddle_ocr(image, lang: str = "en", use_gpu: bool = True) -> dict:
    """
    Public alias for run_paddle_ocr_subprocess().
    Maintains the same call signature as the original function so that
    pipeline.py requires zero changes.
    """
    return run_paddle_ocr_subprocess(image, lang=lang, use_gpu=use_gpu)


# ──────────────────────────────────────────────
# Qwen2.5-VL OCR Fallback (UNCHANGED)
# ──────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None
_vlm_available = None


def load_vlm():

    global _vlm_model, _vlm_processor, _vlm_available

    if _vlm_available is False:
        return None, None

    if _vlm_model is None:

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

            logger.info(f"Loading VLM: {model_name}")

            _vlm_processor = AutoProcessor.from_pretrained(model_name)

            _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            _vlm_available = True

        except Exception as e:
            logger.warning(f"VLM initialization failed: {e}")
            _vlm_available = False
            return None, None

    return _vlm_model, _vlm_processor


def run_vlm_ocr(image):

    model, processor = load_vlm()

    if model is None:
        return {"text": "", "confidence": 0.0}

    try:
        prompt = "Extract all text content from this document image."

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text_input],
            images=[image],
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=2048)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        confidence = 0.90 if len(text) > 50 else 0.60

        return {"text": text, "confidence": confidence}

    except Exception as e:
        logger.error(f"VLM OCR failed: {e}")
        return {"text": "", "confidence": 0.0}