from fastapi.testclient import TestClient
import pytest
from pathlib import Path

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_requires_fields():
    response = client.post("/predict", json={})
    assert response.status_code == 422
