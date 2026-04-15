"""
Tests for POST /query endpoint and the RAG pipeline contract.

All LLM / vector-store calls are mocked via `server.run_query`
(the name bound in server.py's namespace after `from ... import run_query`).
"""

import pytest
from unittest.mock import patch

from tests.conftest import DEFAULT_RESULT


# ──────────────────────────────────────────────────────────────
# Helper: create a Flask test client with custom patches
# ──────────────────────────────────────────────────────────────

def _make_client(run_query_rv=None, run_query_se=None):
    """Return a Flask test client with server.run_query patched."""
    kw = {"return_value": run_query_rv} if run_query_se is None else {"side_effect": run_query_se}
    ctx = patch("server.run_query", **kw)
    ctx2 = patch("server.check_ollama_health", return_value=True)
    ctx.start()
    ctx2.start()
    from server import app
    app.config["TESTING"] = True
    c = app.test_client()
    return c, ctx, ctx2


def _cleanup(*ctxs):
    for c in ctxs:
        c.stop()


# ──────────────────────────────────────────────────────────────
# Happy-path
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointHappyPath:

    def test_query_returns_200(self, client):
        assert client.post("/query", json={"query": "Darshan timings?"}).status_code == 200

    def test_response_is_valid_json(self, client):
        assert isinstance(client.post("/query", json={"query": "Darshan timings?"}).get_json(), dict)

    def test_response_contains_required_fields(self, client):
        required = [
            "query_original", "query_english", "language",
            "agent_route", "answer", "sources", "domains",
            "verification", "suggestions", "response_time_s",
        ]
        data = client.post("/query", json={"query": "Darshan timings?"}).get_json()
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_answer_is_non_empty_string(self, client):
        data = client.post("/query", json={"query": "Darshan timings?"}).get_json()
        assert isinstance(data["answer"], str) and len(data["answer"]) > 0

    def test_language_is_valid_code(self, client):
        lang = client.post("/query", json={"query": "Darshan timings?"}).get_json()["language"]
        assert lang in ("en", "hi", "te", "ta", "kn")

    def test_sources_is_list(self, client):
        assert isinstance(client.post("/query", json={"query": "Darshan timings?"}).get_json()["sources"], list)

    def test_suggestions_is_list(self, client):
        assert isinstance(client.post("/query", json={"query": "Darshan timings?"}).get_json()["suggestions"], list)

    def test_response_time_is_numeric(self, client):
        rt = client.post("/query", json={"query": "Darshan timings?"}).get_json()["response_time_s"]
        assert isinstance(rt, (int, float)) and rt >= 0

    def test_with_chat_history(self, client):
        payload = {
            "query": "Laddu price?",
            "chat_history": [
                {"role": "user",      "content": "Tell me about Tirumala"},
                {"role": "assistant", "content": "Tirumala is a temple town..."},
            ],
        }
        assert client.post("/query", json=payload).status_code == 200


# ──────────────────────────────────────────────────────────────
# Multilingual queries
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointMultilingual:

    @pytest.mark.parametrize("query_text, lang_code", [
        ("What are the darshan timings?",        "en"),
        ("\u0c24\u0c3f\u0c30\u0c41\u0c2e\u0c32 \u0c26\u0c30\u0c4d\u0c36\u0c28 \u0c38\u0c2e\u0c2f\u0c02 \u0c0e\u0c02\u0c24", "te"),
        ("\u0924\u093f\u0930\u0941\u092e\u0932\u093e \u0926\u0930\u094d\u0936\u0928 \u0938\u092e\u092f \u0915\u094d\u092f\u093e \u0939\u0948", "hi"),
        ("\u0ba4\u0bbf\u0bb0\u0bc1\u0bae\u0bb2\u0bc8 \u0ba4\u0bb0\u0bbf\u0b9a\u0ba9 \u0ba8\u0bc7\u0bb0\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9", "ta"),
        ("\u0ca4\u0cbf\u0cb0\u0cc1\u0cae\u0cb2 \u0ca6\u0cb0\u0ccd\u0cb6\u0ca8 \u0cb8\u0cae\u0caf \u0c8e\u0cb7\u0ccd\u0c9f\u0cc1", "kn"),
    ])
    def test_multilingual_query_accepted(self, query_text, lang_code):
        result = dict(DEFAULT_RESULT, language=lang_code, query_original=query_text)
        c, ctx1, ctx2 = _make_client(run_query_rv=result)
        try:
            response = c.post("/query", json={"query": query_text})
            assert response.status_code == 200
            assert response.get_json()["language"] == lang_code
        finally:
            _cleanup(ctx1, ctx2)


# ──────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointErrorHandling:

    def test_empty_query_returns_422(self, client):
        assert client.post("/query", json={"query": ""}).status_code == 422

    def test_missing_query_field_returns_422(self, client):
        assert client.post("/query", json={}).status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        response = client.post(
            "/query",
            data="plain text",
            content_type="text/plain",
        )
        assert response.status_code in (415, 422)

    def test_pipeline_exception_returns_500(self):
        c, ctx1, ctx2 = _make_client(run_query_se=RuntimeError("Vector store not loaded"))
        try:
            response = c.post("/query", json={"query": "What is Tirumala?"})
            assert response.status_code == 500
        finally:
            _cleanup(ctx1, ctx2)

    def test_pipeline_error_response_has_detail_field(self):
        c, ctx1, ctx2 = _make_client(run_query_se=RuntimeError("crash"))
        try:
            response = c.post("/query", json={"query": "What is Tirumala?"})
            assert "detail" in response.get_json()
        finally:
            _cleanup(ctx1, ctx2)


# ──────────────────────────────────────────────────────────────
# Pipeline result contract
# ──────────────────────────────────────────────────────────────

class TestPipelineResultContract:

    def test_agent_route_forwarded_correctly(self):
        result = dict(DEFAULT_RESULT, agent_route="web_search")
        c, ctx1, ctx2 = _make_client(run_query_rv=result)
        try:
            response = c.post("/query", json={"query": "Latest TTD news?"})
            assert response.get_json()["agent_route"] == "web_search"
        finally:
            _cleanup(ctx1, ctx2)

    def test_sources_forwarded_from_pipeline(self):
        sources = ["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"]
        result = dict(DEFAULT_RESULT, sources=sources)
        c, ctx1, ctx2 = _make_client(run_query_rv=result)
        try:
            response = c.post("/query", json={"query": "Tirumala history?"})
            assert response.get_json()["sources"] == sources
        finally:
            _cleanup(ctx1, ctx2)

    def test_pipeline_returning_error_key_gives_400(self):
        error_result = {"error": "Query validation failed"}
        c, ctx1, ctx2 = _make_client(run_query_rv=error_result)
        try:
            response = c.post("/query", json={"query": "test"})
            assert response.status_code == 400
        finally:
            _cleanup(ctx1, ctx2)

    def test_verification_dict_forwarded(self):
        verification = {"grounded": True, "score": 0.95}
        result = dict(DEFAULT_RESULT, verification=verification)
        c, ctx1, ctx2 = _make_client(run_query_rv=result)
        try:
            response = c.post("/query", json={"query": "Tell me about Balaji?"})
            assert response.get_json()["verification"]["grounded"] is True
        finally:
            _cleanup(ctx1, ctx2)


# ──────────────────────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────────────────────

class TestCORSHeaders:

    def test_cors_header_present_on_query(self, client):
        response = client.post(
            "/query",
            json={"query": "What is Tirumala?"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert "access-control-allow-origin" in response.headers

    def test_options_preflight_allowed(self, client):
        response = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code in (200, 204)
