FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=.:src

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src

COPY artifacts/models ./artifacts/models
COPY artifacts/preprocessors ./artifacts/preprocessors
COPY artifacts/metadata/health_fraud_model_metadata.json ./artifacts/metadata/health_fraud_model_metadata.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]