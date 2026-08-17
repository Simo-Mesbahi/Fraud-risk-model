from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    status,
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


# =============================================================================
# Router
# =============================================================================


router = APIRouter(
    tags=[
        "system",
    ],
)


# =============================================================================
# Dependencies
# =============================================================================


PredictionServiceDependency = Annotated[
    PredictionService,
    Depends(
        get_prediction_service
    ),
]


# =============================================================================
# Health
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check inference service health",
    description=(
        "Verify that the inference API is running and that "
        "the frozen fraud-risk model is available."
    ),
    response_description=(
        "Current inference-service health and deployed model identity."
    ),
    operation_id="healthcheck",
)
def healthcheck(
    service: PredictionServiceDependency,
) -> HealthResponse:
    """
    Return lightweight inference-service health information.

    Notes
    -----
    Resolving PredictionService also resolves the cached FraudScorer.
    Therefore a successful response confirms that the deployed model
    service is available to the API process.
    """

    info = (
        service.model_info()
    )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=str(
            info[
                "model_name"
            ]
        ),
        model_version=str(
            info[
                "model_version"
            ]
        ),
    )


# =============================================================================
# Model information
# =============================================================================


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get deployed model information",
    description=(
        "Return the frozen model contract, operational review policy, "
        "probability-scoring configuration and explainability capabilities "
        "exposed by the deployed fraud-risk model."
    ),
    response_description=(
        "Deployed model metadata and inference capabilities."
    ),
    operation_id="get_model_info",
)
def model_info(
    service: PredictionServiceDependency,
) -> ModelInfoResponse:
    """
    Return the public model contract exposed by the inference service.

    The response is validated by ModelInfoResponse before being returned,
    preventing accidental backend/frontend contract drift.
    """

    info = (
        service.model_info()
    )

    return ModelInfoResponse(
        **info
    )