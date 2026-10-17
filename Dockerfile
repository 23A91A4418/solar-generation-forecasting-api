# ── Build Stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies in a separate layer for optimal cache re-use
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime Stage ─────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Solar Forecasting API"
LABEL description="FastAPI service for solar power generation forecasting"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and pre-trained model artifacts
COPY app/ ./app/
COPY models/ ./models/

# Non-root user for security
RUN addgroup --system apigroup && adduser --system --ingroup apigroup apiuser
USER apiuser

# Expose API port
EXPOSE 8000

# Health-check so Docker/Compose can verify the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
