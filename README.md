# Solar Generation Forecasting API

A containerized REST API that predicts solar panel AC power output from meteorological data. Built with FastAPI, LightGBM, and Docker.

## Architecture

```
ML Pipeline (offline)                 Inference Service (Docker)
──────────────────────────────        ──────────────────────────────
data/raw_data.csv
        │
scripts/preprocess.py ──────────▶ data/processed_data.csv
                                          │
                              scripts/train.py
                              ├── models/linear_regression_model.pkl
                              ├── models/advanced_model.pkl
                              └── reports/metrics.json
                                                            │
                                             ┌─────────────────────────┐
                                             │  Docker Container        │
HTTP POST /predict ──────────────────────▶  │  FastAPI + LightGBM      │
                                             │  GET  /health            │
                                             │  POST /predict           │
                                             └─────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 2. Clone & Install

```bash
git clone <repo-url>
cd solar-generation-forecasting-api
pip install -r requirements.txt
```

### 3. Generate Dataset

> If you have the real Kaggle dataset, place the merged CSV at `data/raw_data.csv`.
> Otherwise, generate realistic synthetic data:

```bash
python scripts/generate_data.py
```

### 4. Run the ML Pipeline

```bash
# Step 1: Preprocess
python scripts/preprocess.py

# Step 2: Train models
python scripts/train.py
```

After training, check `reports/metrics.json` for model performance.

### 5. Deploy with Docker (Single Command)

```bash
docker-compose up --build -d
```

Wait ~15 seconds for the health check to pass, then test:

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IRRADIATION": 0.75,
    "AMBIENT_TEMPERATURE": 28.0,
    "MODULE_TEMPERATURE": 42.0,
    "hour": 12.0,
    "month": 6,
    "day_of_week": 2,
    "sin_hour": 0.0,
    "cos_hour": -1.0,
    "irr_x_temp": 21.0
  }'
```

Expected response:
```json
{"predicted_power": 637.5421}
```

Stop the service:
```bash
docker-compose down
```

## API Reference

### `GET /health`

```json
HTTP 200
{"status": "ok"}
```

### `POST /predict`

**Request body** (all fields required, all float/int):

| Field | Type | Description |
|---|---|---|
| `IRRADIATION` | float | Solar irradiation kW/m² (0–2) |
| `AMBIENT_TEMPERATURE` | float | Air temperature °C |
| `MODULE_TEMPERATURE` | float | Panel surface temperature °C |
| `hour` | float | Hour of day (0–23.99) |
| `month` | int | Calendar month (1–12) |
| `day_of_week` | int | Day of week (0=Mon, 6=Sun) |
| `sin_hour` | float | `sin(2π·hour/24)` |
| `cos_hour` | float | `cos(2π·hour/24)` |
| `irr_x_temp` | float | `IRRADIATION × AMBIENT_TEMPERATURE` |

**Success response** (HTTP 200):
```json
{"predicted_power": 637.5421}
```

**Error response** (HTTP 422) — invalid type or missing field:
```json
{"detail": [{"type": "...", "loc": [...], "msg": "..."}]}
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Tests

```bash
pytest tests/ -v
```

All tests use a stubbed model — no pre-trained `.pkl` files required.

## Project Structure

```
solar-generation-forecasting-api/
├── app/
│   └── main.py                  # FastAPI service
├── data/
│   └── raw_data.csv             # Source data (gitignored if generated)
├── models/                      # Trained model artifacts (generated)
├── reports/
│   └── metrics.json             # Evaluation metrics (generated)
├── scripts/
│   ├── generate_data.py         # Synthetic data generator
│   ├── preprocess.py            # Data cleaning & feature engineering
│   └── train.py                 # Model training & serialization
├── tests/
│   ├── test_api.py              # API endpoint tests
│   └── test_preprocessing.py   # Feature engineering unit tests
├── .env.example                 # Environment variable documentation
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── VIDEO_SCRIPT.md
```

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | Host port for the API |
| `MODEL_PATH` | `models/advanced_model.pkl` | Path to model artifact |
| `LOG_LEVEL` | `info` | Uvicorn log level |

## Model Performance

After running `scripts/train.py`, `reports/metrics.json` will contain:

```json
{
    "linear_regression": {"rmse": ..., "r2": ..., "mae": ...},
    "advanced_model":    {"rmse": ..., "r2": ..., "mae": ...}
}
```

Requirements: `linear_regression.r2 > 0.5` and `advanced_model.r2 > linear_regression.r2`.

## Dataset

Uses [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) from Kaggle — `Plant_1_Generation_Data.csv` merged with `Plant_1_Weather_Sensor_Data.csv`. A synthetic data generator (`scripts/generate_data.py`) is provided for reproducibility without a Kaggle account.