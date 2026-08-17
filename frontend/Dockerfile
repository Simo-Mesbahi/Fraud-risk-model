FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend

COPY data/interim/claims.parquet \
    ./data/interim/claims.parquet

COPY artifacts/explainability \
    ./artifacts/explainability

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]