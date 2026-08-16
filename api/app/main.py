from __future__ import annotations

from fastapi import FastAPI

from api.app.routers.healthcheck import (
    router as healthcheck_router,
)

from api.app.routers.predictions import (
    router as predictions_router,
)


app = FastAPI(
    title="Health Insurance Fraud Risk API",
    description=(
        "API for ranking health insurance claims "
        "according to predicted fraud risk. "
        "The model is intended to support human "
        "fraud investigation, not automatic claim rejection."
    ),
    version="1.0.0",
)


app.include_router(
    healthcheck_router
)

app.include_router(
    predictions_router
)


@app.get(
    "/",
    tags=["system"],
)
def root() -> dict[str, str]:
    return {
        "service": (
            "Health Insurance Fraud Risk API"
        ),
        "status": "running",
        "health": "/health",
        "model_info": "/model-info",
        "swagger": "/docs",
        "openapi": "/openapi.json",
    }
    