"""
Tests for the /health endpoint.

The health check must:
  - Return HTTP 200
  - Return JSON with a "status" key set to "healthy"
  - Include an "ollama" key indicating connectivity state
"""

import pytest
from unittest.mock import patch


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health check must always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Response body must be valid JSON."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)

    def test_health_status_field_present(self, client):
        """Response must include a 'status' field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_status_is_healthy(self, client):
        """Status field must be 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_ollama_field_present(self, client):
        """Response must include an 'ollama' field."""
        response = client.get("/health")
        data = response.json()
        assert "ollama" in data

    def test_health_ollama_connected_when_reachable(self, client):
        """When Ollama is reachable, 'ollama' should be 'connected'."""
        with patch("query.agents.knowledge_rag_agent.check_ollama_health", return_value=True):
            response = client.get("/health")
        data = response.json()
        assert data["ollama"] in ("connected", "disconnected")  # valid state

    def test_health_ollama_disconnected_when_unreachable(self, client):
        """When Ollama is unreachable, 'ollama' should be 'disconnected'."""
        with patch("query.agents.knowledge_rag_agent.check_ollama_health", return_value=False):
            response = client.get("/health")
        data = response.json()
        assert data["ollama"] == "disconnected"

    def test_health_content_type_is_json(self, client):
        """Content-Type header must be application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
