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

# Optional debug (you can remove later)
print("🔍 ENV CHECK:")
print("HF_TOKEN:", "FOUND" if os.getenv("HF_TOKEN") else "NOT FOUND")
print("TAVILY_API_KEY:", "FOUND" if os.getenv("TAVILY_API_KEY") else "NOT FOUND")


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
# 3. Import torch FIRST (CRITICAL)
# ──────────────────────────────────────────────
import torch  # noqa: E402

# ──────────────────────────────────────────────
# 4. Force cuDNN + CUDA DLL visibility for Paddle
# ──────────────────────────────────────────────

import torch
import os
from pathlib import Path

# 🔥 Get torch CUDA/cuDNN DLL path
torch_lib = Path(torch.__file__).parent / "lib"

if torch_lib.exists() and os.name == "nt":
    # Windows only — add torch DLL path for Paddle CUDA visibility
    os.environ["PATH"] = str(torch_lib) + ";" + os.environ["PATH"]
    os.add_dll_directory(str(torch_lib))

print("✅ Torch DLL path added:", torch_lib)

# Optional debug (can remove later)
print("🔥 CUDA Available:", torch.cuda.is_available())
print("🔥 CUDA Version:", torch.version.cuda)