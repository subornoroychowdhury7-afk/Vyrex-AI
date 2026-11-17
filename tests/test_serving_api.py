"""
Unit and Integration tests for VYREX AI Production FastAPI Serving Layer.
"""

import pytest
from fastapi.testclient import TestClient
from vyrex_ai.serving.api import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "VYREX AI Production API"
        assert "Suborno Roy Chowdhury" in data["creator"]


def test_models_endpoint():
    with TestClient(app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "vyrex-ai-1.7b" in model_ids


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["model_name"] == "vyrex-ai-1.7b"


def test_chat_completions_endpoint():
    with TestClient(app) as client:
        payload = {
            "model": "vyrex-ai-1.7b",
            "messages": [
                {"role": "user", "content": "Who created you?"}
            ],
            "max_tokens": 40,
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "Suborno Roy Chowdhury" in data["choices"][0]["message"]["content"]
        assert data["usage"]["completion_tokens"] > 0

