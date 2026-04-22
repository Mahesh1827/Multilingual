"""
Environment bootstrap — MUST be imported first.

- Forces all caches to T drive
- Fixes PyTorch CUDA DLL loading (env-based, safe)
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# 🔴 ADD THIS AT THE VERY TOP (first lines of the file)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


# ─────────────────────────────────────────────
# Existing code continues below
# (your torch setup, DLL paths, etc.)
# ─────────────────────────────────────────────

# ──────────────────────────────────────────────
# 1. Redirect ALL caches to local .cache folder
# ──────────────────────────────────────────────
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "model_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

_CACHE_DIRS = {
    "HF_HOME":              str(_CACHE_ROOT / "huggingface"),
    "HF_HUB_CACHE":         str(_CACHE_ROOT / "huggingface" / "hub"),
    "TRANSFORMERS_CACHE":   str(_CACHE_ROOT / "huggingface" / "hub"),
    "HF_DATASETS_CACHE":    str(_CACHE_ROOT / "huggingface" / "datasets"),
    "WHISPER_CACHE":        str(_CACHE_ROOT / "whisper"),
    "PPOCR_HOME":           str(_CACHE_ROOT / "paddleocr"),
    "PADDLE_HOME":          str(_CACHE_ROOT / "paddle"),
    "TORCH_HOME":           str(_CACHE_ROOT / "torch"),
    "UV_CACHE_DIR":         str(_CACHE_ROOT.parent / "uv-cache"),
    "TEMP":                 str(_CACHE_ROOT.parent / "tmp"),
    "TMP":                  str(_CACHE_ROOT.parent / "tmp"),
}

for key, path in _CACHE_DIRS.items():
    os.environ[key] = path
    Path(path).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. DLL FIX (ONLY ENV — NO SYSTEM CUDA)
# ──────────────────────────────────────────────
if os.name == "nt":
    venv_base = Path(sys.prefix)

    torch_paths = [
        venv_base / "Lib/site-packages/torch/lib",
        venv_base / "Lib/site-packages/torch/bin",
    ]

    for p in torch_paths:
        if p.exists():
            os.add_dll_directory(str(p))

# ──────────────────────────────────────────────
# 3. DLL paths for torch — torch itself is NOT imported here.
#    ocr_engine.py imports paddle FIRST, then torch, to avoid the
#    pybind11 C++ type re-registration clash on Windows.
# ──────────────────────────────────────────────
import sys
from pathlib import Path as _Path
_venv_base = _Path(sys.prefix)
for _p in [_venv_base / "Lib/site-packages/torch/lib",
            _venv_base / "Lib/site-packages/torch/bin"]:
    if _p.exists():
        os.add_dll_directory(str(_p))

# torch DLL path is registered above via add_dll_directory.
# torch itself is imported later in ocr_engine.py AFTER paddle boots.
