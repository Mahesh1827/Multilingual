"""
OCR Module — PaddleOCR + Qwen2.5-VL fallback
Optimized for RAG document processing (GPU FIXED VERSION)
"""

import os
import threading
import logging
from pathlib import Path

# ──────────────────────────────────────────────
# ENV FIXES (must run early)
# ──────────────────────────────────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "False"

# 🔥 Import torch first (ensures CUDA runtime loads)
import torch

# 🔥 Ensure Paddle can see cuDNN from torch
torch_lib = Path(torch.__file__).parent / "lib"
if torch_lib.exists():
    os.environ["PATH"] = str(torch_lib) + ";" + os.environ["PATH"]

from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Thread lock for safe Paddle init
_lock = threading.Lock()

# ──────────────────────────────────────────────
# Image Preprocessing
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

    # Deskew
    coords = np.column_stack(np.where(img > 10))

    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = img.shape[:2]
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
# PaddleOCR Engine (GPU SAFE)
# ──────────────────────────────────────────────

_paddle_ocr = None

def get_paddle_ocr(lang="en"):

    global _paddle_ocr

    with _lock:  # 🔥 thread-safe init

        if _paddle_ocr is None:

            logger.info("🚀 Loading PaddleOCR (GPU enabled)")

            # 🔥 Lazy import (VERY IMPORTANT)
            from paddleocr import PaddleOCR

            _paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang=lang,
                use_gpu=True,   # 🚀 FORCE GPU
                show_log=False
            )

    return _paddle_ocr


# ──────────────────────────────────────────────
# PaddleOCR Execution
# ──────────────────────────────────────────────

def run_paddle_ocr(image, lang="en", use_gpu=True):

    if image is None:
        logger.error("OCR received None image")
        return {"text": "", "confidence": 0.0, "details": []}

    try:
        img = np.array(image)
        if img.size == 0:
            raise ValueError("Empty image")

    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        return {"text": "", "confidence": 0.0, "details": []}

    try:
        preprocessed = preprocess_image_for_ocr(image)

    except Exception as e:
        logger.error(f"OCR preprocessing failed: {e}")
        return {"text": "", "confidence": 0.0, "details": []}

    if preprocessed is None or preprocessed.size == 0:
        logger.error("Invalid image after preprocessing")
        return {"text": "", "confidence": 0.0, "details": []}

    h, w = preprocessed.shape[:2]

    if h < 40 or w < 40:
        logger.warning("Image too small for OCR")
        return {"text": "", "confidence": 0.0, "details": []}

    # 🔥 Always GPU
    ocr = get_paddle_ocr(lang)

    try:
        result = ocr.ocr(preprocessed, cls=False)

    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")

        try:
            small = cv2.resize(
                preprocessed,
                None,
                fx=0.75,
                fy=0.75,
                interpolation=cv2.INTER_AREA,
            )

            result = ocr.ocr(small, cls=False)
            logger.info("PaddleOCR retry succeeded")

        except Exception as e2:
            logger.error(f"PaddleOCR retry failed: {e2}")
            return {"text": "", "confidence": 0.0, "details": []}

    if not result:
        return {"text": "", "confidence": 0.0, "details": []}

    lines, confidences, details = [], [], []

    for line in result:
        for line_info in line:
            bbox, (text, conf) = line_info
            lines.append(text)
            confidences.append(conf)

            details.append({
                "text": text,
                "confidence": conf,
                "bbox": bbox
            })

    if not lines:
        return {"text": "", "confidence": 0.0, "details": []}

    avg_conf = sum(confidences) / len(confidences)

    logger.info(f"PaddleOCR: {len(lines)} lines, avg confidence: {avg_conf:.3f}")

    return {
        "text": "\n".join(lines),
        "confidence": avg_conf,
        "details": details
    }


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