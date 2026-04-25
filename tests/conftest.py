"""
Shared fixtures for the Tirumala AI Assistant test suite.

Key design decisions:
  - Heavy ML modules (torch, paddle, transformers) are stubbed in sys.modules
    BEFORE any project code is imported, so CI runners need no GPU.
  - `client` is function-scoped so each test gets a fresh test client,
    making per-test patching work correctly.
  - Patch target is `server.run_query` / `server.check_ollama_health`
    because server.py uses `from ... import` (binds names in server's namespace).
"""

import os
import sys
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────────────────────
# Stub heavy modules BEFORE any project import
# ──────────────────────────────────────────────────────────────

def _torch_stub():
    t = MagicMock(name="torch")
    t.cuda.is_available.return_value = False
    t.version.cuda = "N/A"
    t.__file__ = "/stub/torch/__init__.py"
    return t


_HEAVY = [
    "torch", "torchvision", "torchaudio",
    "paddle", "paddlepaddle", "paddleocr",
    "transformers", "accelerate", "sentencepiece",
    "sentence_transformers",
    "cv2",
    "pdf2image",
]
for _m in _HEAVY:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock(name=_m)
sys.modules["torch"] = _torch_stub()

# ──────────────────────────────────────────────────────────────
# Inject test-safe env vars before dotenv runs
# ──────────────────────────────────────────────────────────────

os.environ.setdefault("HF_TOKEN",        "hf_test_ci")
os.environ.setdefault("TAVILY_API_KEY",  "tvly_test_ci")
os.environ.setdefault("GROQ_API_KEY",    "gsk_test_ci")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# ──────────────────────────────────────────────────────────────
# Default pipeline result used by most tests
# ──────────────────────────────────────────────────────────────

DEFAULT_RESULT = {
    "query_original": "What are the darshan timings?",
    "query_english":  "What are the darshan timings?",
    "language":       "en",
    "agent_route":    "rag",
    "answer":         "Darshan timings are from 2 AM to 11 PM.",
    "sources":        ["doc1.pdf", "doc2.pdf"],
    "domains":        ["pilgrimage_seva"],
    "verification":   {"grounded": True, "score": 0.85},
    "suggestions":    ["How to book tickets?", "What is the laddu price?"],
}


# ──────────────────────────────────────────────────────────────
# Function-scoped client — fresh per test, correct patch target
# ──────────────────────────────────────────────────────────────

import pytest  # noqa: E402  (after sys.modules stubs)


@pytest.fixture()
def client():
    """
    Return a Flask test client with run_query and check_ollama_health mocked.
    Function-scoped so per-test patches on server.run_query work correctly.
    """
    # Patch names in server's own namespace (where `from X import Y` bound them)
    with patch("server.run_query", return_value=DEFAULT_RESULT), \
         patch("server.check_ollama_health", return_value=True):
        from server import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c
