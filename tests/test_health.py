"""
Tests for GET /health endpoint.
"""

from unittest.mock import patch


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_returns_json(self, client):
        assert isinstance(client.get("/health").get_json(), dict)

    def test_health_status_field_present(self, client):
        assert "status" in client.get("/health").get_json()

    def test_health_status_is_healthy(self, client):
        assert client.get("/health").get_json()["status"] == "healthy"

    def test_health_ollama_field_present(self, client):
        assert "ollama" in client.get("/health").get_json()

    def test_health_ollama_connected_when_reachable(self, client):
        # server.check_ollama_health is already patched to True in conftest
        data = client.get("/health").get_json()
        assert data["ollama"] == "connected"

    def test_health_ollama_disconnected_when_unreachable(self, client):
        # Override the conftest patch specifically for this test
        with patch("server.check_ollama_health", return_value=False):
            response = client.get("/health")
        assert response.get_json()["ollama"] == "disconnected"

    def test_health_content_type_is_json(self, client):
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
