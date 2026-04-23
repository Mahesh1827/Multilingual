"""
Flask Server for Tirumala Multi-Agent AI Assistant
Wraps the LangGraph pipeline in HTTP endpoints.

Run:
    flask run --port 8000
    OR
    gunicorn -w 1 -b 0.0.0.0:8000 server:app
"""

import _env_setup  # noqa: F401

import logging
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from query.agents.pipeline import run_query
from query.agents.knowledge_rag_agent import check_ollama_health
from query.config import APP_VERSION, MAX_QUERY_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# ──────────────────────────────────────────────
# Startup log (runs once at import time)
# ──────────────────────────────────────────────

def _warmup():
    """Log Ollama status on first import (non-blocking)."""
    try:
        if check_ollama_health():
            logger.info("✅ Ollama is reachable.")
        else:
            logger.warning("⚠️  Ollama is not running. Answers may use raw context.")
    except Exception:
        logger.warning("⚠️  Ollama check skipped (not available).")


_warmup()


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    ollama_ok = check_ollama_health()
    return jsonify({
        "status": "healthy",
        "version": APP_VERSION,
        "ollama": "connected" if ollama_ok else "disconnected",
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Production monitoring metrics."""
    try:
        from query.agents.faq_agent import get_cache_stats, get_cache_health
        from query.agents.error_handling import get_error_stats

        return jsonify({
            "version": APP_VERSION,
            "ollama": "connected" if check_ollama_health() else "disconnected",
            "cache_stats": get_cache_stats(),
            "cache_health": get_cache_health(),
            "error_stats": get_error_stats(),
        })
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"detail": str(e)}), 500


@app.route("/query", methods=["POST"])
def query_endpoint():
    """
    Submit a query to the multi-agent RAG pipeline.
    Expects JSON body with a 'query' field and optional 'chat_history'.
    """
    # ── Validate request ─────────────────────────────────────
    if not request.is_json:
        return jsonify({"detail": "Content-Type must be application/json"}), 422

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"detail": "Invalid JSON body"}), 422

    query = data.get("query", "")
    if not isinstance(query, str) or len(query.strip()) == 0:
        return jsonify({"detail": "query field is required and must be non-empty"}), 422

    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            "detail": f"Query too long. Maximum {MAX_QUERY_LENGTH} characters."
        }), 422

    chat_history = data.get("chat_history", [])
    language = data.get("language", "en")

    # ── Run pipeline ─────────────────────────────────────────
    start = time.time()

    try:
        result = run_query(query, chat_history, language=language)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"detail": str(e)}), 500

    elapsed = time.time() - start

    if "error" in result:
        return jsonify({"detail": result["error"]}), 400

    return jsonify({
        "query_original": result.get("query_original", ""),
        "query_english":  result.get("query_english", ""),
        "language":       result.get("language", "en"),
        "agent_route":    result.get("agent_route", "unknown"),
        "answer":         result.get("answer", ""),
        "sources":        result.get("sources", []),
        "domains":        result.get("domains", []),
        "verification":   result.get("verification", {}),
        "suggestions":    result.get("suggestions", []),
        "timing_ms":      result.get("timing_ms", {}),
        "response_time_s": round(elapsed, 2),
        "version":        APP_VERSION,
    })


# ──────────────────────────────────────────────
# Direct execution
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
