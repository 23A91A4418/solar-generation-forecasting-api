"""
tests/test_api.py
-----------------
Unit tests for the FastAPI inference service.

Run:
    pytest tests/ -v
"""

import sys
import os
import types
import numpy as np
import pytest

# ── Stub the model so tests run without a real .pkl file ──────────────
class _FakeModel:
    def predict(self, X):
        return np.array([500.0] * len(X))

# Patch before importing the app
import importlib
import unittest.mock as mock

# Ensure the app module can be imported by patching joblib.load and os.path.exists
@pytest.fixture(scope="session", autouse=True)
def patch_model_loading():
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("joblib.load", return_value=_FakeModel()):
        # Import app after patching
        import app.main as _main
        _main._model = _FakeModel()
        yield _main


@pytest.fixture(scope="session")
def client(patch_model_loading):
    from fastapi.testclient import TestClient
    import app.main as _main
    return TestClient(_main.app)


# ── Health Endpoint ───────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body(client):
    response = client.get("/health")
    data = response.json()
    assert data == {"status": "ok"}


# ── Predict Endpoint — Valid Input ────────────────────────────────────

VALID_PAYLOAD = {
    "IRRADIATION": 0.75,
    "AMBIENT_TEMPERATURE": 28.0,
    "MODULE_TEMPERATURE": 42.0,
    "hour": 12.0,
    "month": 6,
    "day_of_week": 2,
    "sin_hour": 0.0,
    "cos_hour": -1.0,
    "irr_x_temp": 21.0,
}


def test_predict_valid_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_valid_response_schema(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert "predicted_power" in data
    assert isinstance(data["predicted_power"], (int, float))


def test_predict_power_is_non_negative(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["predicted_power"] >= 0.0


# ── Predict Endpoint — Invalid Input (422) ────────────────────────────

def test_predict_invalid_type_returns_422(client):
    """IRRADIATION is a string; Pydantic must reject with 422."""
    bad_payload = {**VALID_PAYLOAD, "IRRADIATION": "not-a-number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    """Missing required field must return 422."""
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "IRRADIATION"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_empty_body_returns_422(client):
    """Empty body must return 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_irradiation_out_of_range_422(client):
    """IRRADIATION > 2.0 should fail the ge/le validator → 422."""
    bad_payload = {**VALID_PAYLOAD, "IRRADIATION": 999.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
