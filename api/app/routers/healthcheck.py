from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from api.app.dependencies import (
    get_prediction_service,
)

from api.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
)

from api.app.services.prediction_service import (
    PredictionService,
)


router = APIRouter(
    tags=["system"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def healthcheck(
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> HealthResponse:
    info = service.model_info()

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=info["model_name"],
        model_version=info["model_version"],
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info(
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> ModelInfoResponse:
    return ModelInfoResponse(
        **service.model_info()
    )