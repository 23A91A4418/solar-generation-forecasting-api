"""
app/main.py
-----------
FastAPI inference service for solar power generation forecasting.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — predict AC_POWER from weather features

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# ── Model path (overridable via env var) ─────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/advanced_model.pkl")

# ── Global model placeholder ─────────────────────────────────────────
_model: Any = None

FEATURE_ORDER = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hour",
    "month",
    "day_of_week",
    "sin_hour",
    "cos_hour",
    "irr_x_temp",
]


# ── Startup / Shutdown ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    global _model
    if not os.path.exists(MODEL_PATH):
        print(
            f"[API] FATAL: Model file not found at '{MODEL_PATH}'. "
            "Run `python scripts/train.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    _model = joblib.load(MODEL_PATH)
    print(f"[API] Model loaded from '{MODEL_PATH}'")
    yield
    _model = None
    print("[API] Model unloaded.")


app = FastAPI(
    title="Solar Generation Forecasting API",
    description=(
        "Predicts AC power output (kW) of a solar panel array "
        "from meteorological features using a trained LightGBM model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """All features required by the trained model."""

    IRRADIATION: float = Field(
        ..., ge=0.0, le=2.0,
        description="Solar irradiation in kW/m² (0–2)"
    )
    AMBIENT_TEMPERATURE: float = Field(
        ..., ge=-10.0, le=60.0,
        description="Ambient air temperature in °C"
    )
    MODULE_TEMPERATURE: float = Field(
        ..., ge=-10.0, le=100.0,
        description="Solar module surface temperature in °C"
    )
    hour: float = Field(
        ..., ge=0.0, lt=24.0,
        description="Hour of day (0–23.99, fractional minutes allowed)"
    )
    month: int = Field(
        ..., ge=1, le=12,
        description="Calendar month (1=Jan … 12=Dec)"
    )
    day_of_week: int = Field(
        ..., ge=0, le=6,
        description="Day of week (0=Monday … 6=Sunday)"
    )
    sin_hour: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Sine encoding of hour: sin(2π·hour/24)"
    )
    cos_hour: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Cosine encoding of hour: cos(2π·hour/24)"
    )
    irr_x_temp: float = Field(
        ...,
        description="Interaction term: IRRADIATION × AMBIENT_TEMPERATURE"
    )

    model_config = {"json_schema_extra": {
        "example": {
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
    }}


class PredictionResponse(BaseModel):
    predicted_power: float = Field(
        ..., description="Predicted AC power output in kW"
    )


class HealthResponse(BaseModel):
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Operations"],
)
def health_check():
    """Returns 200 OK when the service is up and the model is loaded."""
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Solar AC Power",
    tags=["Inference"],
)
def predict(request: PredictionRequest):
    """
    Accepts a JSON payload with all required meteorological and engineered
    features and returns the predicted AC power output in kW.

    Invalid or missing fields will result in an HTTP 422 response.
    """
    features = np.array([[getattr(request, col) for col in FEATURE_ORDER]])
    raw_prediction = float(_model.predict(features)[0])
    # Clamp to non-negative (power cannot be negative)
    predicted_power = max(0.0, raw_prediction)
    return PredictionResponse(predicted_power=round(predicted_power, 4))
