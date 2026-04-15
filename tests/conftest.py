"""
Shared fixtures for the Tirumala AI Assistant test suite.

Design principles:
  - ALL heavy model imports (torch, paddle, sentence-transformers) are mocked
    at the module level so tests run on the GitHub CI runner (CPU-only, no GPU).
  - The FastAPI TestClient is created once per session for speed.
  - Environment variables are set before any project module is imported.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────────────────────
# Stub out heavy imports BEFORE any project code is loaded.
# This prevents torch/paddle/paddleocr from being imported during
# collection, which would crash on a CPU-only CI runner.
# ──────────────────────────────────────────────────────────────

def _make_torch_stub():
    """Return a minimal torch stub that satisfies all import-time usages."""
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = False
    torch.version.cuda = "N/A"
    torch.__file__ = "/stub/torch/__init__.py"
    return torch


# Patch heavy modules before any project import
_HEAVY_MODULES = [
    "torch", "torchvision", "torchaudio",
    "paddlepaddle", "paddle", "paddleocr",
    "transformers", "accelerate", "sentencepiece",
    "sentence_transformers",
    "faiss",
    "cv2",
    "pdf2image",
]

for _mod in _HEAVY_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock(name=_mod)

# Provide a slightly more realistic torch stub
sys.modules["torch"] = _make_torch_stub()


# ──────────────────────────────────────────────────────────────
# Environment variables — set before any dotenv load
# ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="session")
def set_env_vars():
    """Inject test-safe env vars for the whole session."""
    env_overrides = {
        "HF_TOKEN": "hf_test_token_ci",
        "TAVILY_API_KEY": "tvly_test_key_ci",
        "GROQ_API_KEY": "gsk_test_key_ci",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen2.5:7b",
    }
    with patch.dict(os.environ, env_overrides):
        yield


# ──────────────────────────────────────────────────────────────
# FastAPI TestClient — session-scoped (one startup per test run)
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Return the FastAPI app with mocked pipeline dependencies."""
    # Mock the pipeline and health-check so server.py can be imported
    with patch("query.agents.pipeline.run_query") as mock_run, \
         patch("query.agents.knowledge_rag_agent.check_ollama_health", return_value=True):

        mock_run.return_value = {
            "query_original": "test query",
            "query_english": "test query",
            "language": "en",
            "agent_route": "rag",
            "answer": "This is a mock answer for testing.",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "domains": ["pilgrimage_seva"],
            "verification": {"grounded": True, "score": 0.85},
            "suggestions": ["Related question 1?", "Related question 2?"],
        }

        from server import app as fastapi_app
        return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """Return a synchronous TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
