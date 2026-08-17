FROM python:3.12-slim


# =============================================================================
# Environment
# =============================================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/frontend \
    ARTIFACTS_ROOT=/app/artifacts


# =============================================================================
# Working directory
# =============================================================================

WORKDIR /app


# =============================================================================
# Python dependencies
# =============================================================================

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Frontend application
# =============================================================================

COPY frontend ./frontend


# =============================================================================
# Demo data
# =============================================================================

COPY data/interim/claims.parquet \
    ./data/interim/claims.parquet


# =============================================================================
# Frontend analytical artifacts
# =============================================================================

COPY artifacts/metadata \
    ./artifacts/metadata

COPY artifacts/explainability \
    ./artifacts/explainability


# =============================================================================
# Network
# =============================================================================

EXPOSE 8501


# =============================================================================
# Container health check
# =============================================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1


# =============================================================================
# Application startup
# =============================================================================

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]