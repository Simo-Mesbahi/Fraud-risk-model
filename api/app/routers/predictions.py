from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from api.app.dependencies import (
    get_prediction_service,
)

from api.app.schemas import (
    BatchClaimsRequest,
    BatchScoreResponse,
    ClaimScore,
    RankedClaimScore,
    SingleClaimRequest,
    SingleScoreResponse,
    TopReviewRequest,
    TopReviewResponse,
)

from api.app.services.prediction_service import (
    PredictionService,
)


router = APIRouter(
    prefix="",
    tags=["scoring"],
)


def _raise_validation_error(
    exc: Exception,
) -> None:
    raise HTTPException(
        status_code=422,
        detail=str(exc),
    ) from exc


@router.post(
    "/score",
    response_model=SingleScoreResponse,
)
def score_claim(
    request: SingleClaimRequest,
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> SingleScoreResponse:
    try:
        prediction = (
            service.score_single(
                request.claim
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        _raise_validation_error(
            exc
        )

    return SingleScoreResponse(
        prediction=ClaimScore(
            **prediction
        )
    )


@router.post(
    "/score-batch",
    response_model=BatchScoreResponse,
)
def score_batch(
    request: BatchClaimsRequest,
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> BatchScoreResponse:
    try:
        results = (
            service.score_batch(
                request.claims
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        _raise_validation_error(
            exc
        )

    predictions = [
        ClaimScore(
            **result
        )
        for result in results
    ]

    return BatchScoreResponse(
        count=len(predictions),
        predictions=predictions,
    )


@router.post(
    "/top-review",
    response_model=TopReviewResponse,
)
def top_review(
    request: TopReviewRequest,
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> TopReviewResponse:
    try:
        results = (
            service.top_review(
                claims=request.claims,
                review_fraction=(
                    request.review_fraction
                ),
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        _raise_validation_error(
            exc
        )

    predictions = [
        RankedClaimScore(
            **result
        )
        for result in results
    ]

    return TopReviewResponse(
        total_claims=len(
            request.claims
        ),
        selected_claims=len(
            predictions
        ),
        review_fraction=(
            request.review_fraction
        ),
        predictions=predictions,
    )