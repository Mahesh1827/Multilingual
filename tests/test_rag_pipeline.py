"""
Tests for the /query endpoint and the RAG pipeline contract.

The actual LLM / vector-store calls are mocked so these tests verify:
  - HTTP contract (request shape, response shape, status codes)
  - Error handling (empty query, pipeline failures)
  - Response field presence and type constraints
  - CORS headers

No real model inference happens here.
"""

import pytest
from unittest.mock import patch


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

VALID_PIPELINE_RESULT = {
    "query_original": "What are the darshan timings at Tirumala?",
    "query_english":  "What are the darshan timings at Tirumala?",
    "language":       "en",
    "agent_route":    "rag",
    "answer":         "Darshan timings at Tirumala are from 2 AM to 11 PM.",
    "sources":        ["TTD_dataset.pdf", "sri_venkateswara_swamy_darshan.pdf"],
    "domains":        ["pilgrimage_seva"],
    "verification":   {"grounded": True, "score": 0.91},
    "suggestions":    [
        "How to book a darshan ticket online?",
        "What is the special entry darshan fee?",
    ],
}


@pytest.fixture
def mock_pipeline():
    """Patch run_query to return a canned response."""
    with patch("query.agents.pipeline.run_query", return_value=VALID_PIPELINE_RESULT) as m:
        yield m


@pytest.fixture
def mock_pipeline_error():
    """Patch run_query to simulate a pipeline crash."""
    with patch("query.agents.pipeline.run_query", side_effect=RuntimeError("Vector store not loaded")) as m:
        yield m


# ──────────────────────────────────────────────────────────────
# Happy-path: valid query
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointHappyPath:
    """POST /query with well-formed requests."""

    def test_query_returns_200(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        assert response.status_code == 200

    def test_response_is_valid_json(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        data = response.json()
        assert isinstance(data, dict)

    def test_response_contains_required_fields(self, client, mock_pipeline):
        required_fields = [
            "query_original", "query_english", "language",
            "agent_route", "answer", "sources", "domains",
            "verification", "suggestions", "response_time_s",
        ]
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        data = response.json()
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_answer_is_non_empty_string(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        assert isinstance(response.json()["answer"], str)
        assert len(response.json()["answer"]) > 0

    def test_language_is_valid_code(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        lang = response.json()["language"]
        assert lang in ("en", "hi", "te", "ta", "kn"), f"Unexpected language: {lang}"

    def test_sources_is_list(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        assert isinstance(response.json()["sources"], list)

    def test_suggestions_is_list(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        assert isinstance(response.json()["suggestions"], list)

    def test_response_time_is_numeric(self, client, mock_pipeline):
        response = client.post("/query", json={"query": "What are the darshan timings?"})
        rt = response.json()["response_time_s"]
        assert isinstance(rt, (int, float))
        assert rt >= 0

    def test_with_chat_history(self, client, mock_pipeline):
        """Pipeline must accept a non-empty chat_history."""
        payload = {
            "query": "What is the laddu price?",
            "chat_history": [
                {"role": "user",      "content": "Tell me about Tirumala"},
                {"role": "assistant", "content": "Tirumala is a temple town..."},
            ],
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────
# Multi-language queries
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointMultilingual:
    """Verify the API accepts queries in all supported languages."""

    @pytest.mark.parametrize("query_text, lang_code", [
        ("What are the darshan timings?",          "en"),
        ("తిరుమల దర్శన సమయం ఎంత",                  "te"),
        ("तिरुमला दर्शन समय क्या है",               "hi"),
        ("திருமலை தரிசன நேரம் என்ன",               "ta"),
        ("ತಿರುಮಲ ದರ್ಶನ ಸಮಯ ಎಷ್ಟು",                 "kn"),
    ])
    def test_multilingual_query_accepted(self, client, query_text, lang_code):
        result = dict(VALID_PIPELINE_RESULT, language=lang_code, query_original=query_text)
        with patch("query.agents.pipeline.run_query", return_value=result):
            response = client.post("/query", json={"query": query_text})
        assert response.status_code == 200
        assert response.json()["language"] == lang_code


# ──────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────

class TestQueryEndpointErrorHandling:
    """Validate error responses for invalid / edge-case inputs."""

    def test_empty_query_string_returns_422(self, client):
        """Pydantic min_length=1 must reject empty strings."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        """The query field is required."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        """Non-JSON body must be rejected."""
        response = client.post(
            "/query",
            content="plain text body",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (415, 422)

    def test_pipeline_exception_returns_500(self, client, mock_pipeline_error):
        """Unhandled pipeline errors must surface as HTTP 500."""
        response = client.post("/query", json={"query": "What is Tirumala?"})
        assert response.status_code == 500

    def test_pipeline_error_response_has_detail_field(self, client, mock_pipeline_error):
        """500 response body must contain a 'detail' field."""
        response = client.post("/query", json={"query": "What is Tirumala?"})
        data = response.json()
        assert "detail" in data


# ──────────────────────────────────────────────────────────────
# Pipeline result contract
# ──────────────────────────────────────────────────────────────

class TestPipelineResultContract:
    """
    Validate that run_query output satisfies the QueryResponse schema.
    Tests the mapping layer between pipeline dict and API response model.
    """

    def test_agent_route_forwarded_correctly(self, client):
        result = dict(VALID_PIPELINE_RESULT, agent_route="web_search")
        with patch("query.agents.pipeline.run_query", return_value=result):
            response = client.post("/query", json={"query": "Latest TTD news?"})
        assert response.json()["agent_route"] == "web_search"

    def test_sources_forwarded_from_pipeline(self, client):
        sources = ["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"]
        result = dict(VALID_PIPELINE_RESULT, sources=sources)
        with patch("query.agents.pipeline.run_query", return_value=result):
            response = client.post("/query", json={"query": "Tirumala history?"})
        assert response.json()["sources"] == sources

    def test_pipeline_returning_error_key_gives_400(self, client):
        """If pipeline returns {'error': ...}, the API must return 400."""
        error_result = {"error": "Query validation failed: offensive content detected"}
        with patch("query.agents.pipeline.run_query", return_value=error_result):
            response = client.post("/query", json={"query": "test"})
        assert response.status_code == 400

    def test_verification_dict_forwarded(self, client):
        verification = {"grounded": True, "score": 0.95, "flags": []}
        result = dict(VALID_PIPELINE_RESULT, verification=verification)
        with patch("query.agents.pipeline.run_query", return_value=result):
            response = client.post("/query", json={"query": "Tell me about Balaji?"})
        assert response.json()["verification"]["grounded"] is True


# ──────────────────────────────────────────────────────────────
# CORS headers
# ──────────────────────────────────────────────────────────────

class TestCORSHeaders:
    """CORS must be configured — required for browser-based clients."""

    def test_cors_header_present_on_query(self, client, mock_pipeline):
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
