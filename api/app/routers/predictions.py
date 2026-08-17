from __future__ import annotations

from typing import (
    Annotated,
    NoReturn,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.app.dependencies import (
    get_prediction_service,
)

from api.app.schemas import (
    BatchClaimsRequest,
    BatchScoreResponse,
    ClaimExplanation,
    ClaimScore,
    ExplainClaimRequest,
    ExplainClaimResponse,
    RankedClaimScore,
    SingleClaimRequest,
    SingleScoreResponse,
    TopReviewRequest,
    TopReviewResponse,
)

from api.app.services.prediction_service import (
    PredictionService,
)


# =============================================================================
# Router
# =============================================================================


router = APIRouter(
    prefix="",
    tags=[
        "scoring",
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
# Error handling
# =============================================================================


VALIDATION_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
)


def _raise_validation_error(
    exc: Exception,
) -> NoReturn:
    """
    Convert model/input validation failures into HTTP 422 responses.

    This keeps business/data errors separate from genuine internal
    server failures.
    """

    detail = (
        str(
            exc
        ).strip()
        or exc.__class__.__name__
    )

    raise HTTPException(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        detail=detail,
    ) from exc


def _raise_inference_error(
    exc: Exception,
) -> NoReturn:
    """
    Convert unexpected inference failures into HTTP 500 responses.

    Runtime errors should not be misrepresented as invalid user input.
    """

    raise HTTPException(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        detail=(
            "The inference pipeline could not complete "
            "the requested operation."
        ),
    ) from exc


# =============================================================================
# Score one claim
# =============================================================================


@router.post(
    "/score",
    response_model=SingleScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score one insurance claim",
    description=(
        "Run one health-insurance claim through the frozen "
        "feature-engineering, preprocessing and fraud-risk model pipeline."
    ),
    response_description=(
        "Fraud-risk probability and deployed model identity."
    ),
    operation_id="score_claim",
)
def score_claim(
    request: SingleClaimRequest,
    service: PredictionServiceDependency,
) -> SingleScoreResponse:
    """
    Score one claim using the deployed frozen model.
    """

    try:

        prediction = (
            service.score_single(
                request.claim
            )
        )

    except VALIDATION_EXCEPTIONS as exc:

        _raise_validation_error(
            exc
        )

    except RuntimeError as exc:

        _raise_inference_error(
            exc
        )

    return SingleScoreResponse(
        prediction=ClaimScore(
            **prediction
        )
    )


# =============================================================================
# Explain one claim
# =============================================================================


@router.post(
    "/explain",
    response_model=ExplainClaimResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain one claim prediction",
    description=(
        "Generate a local TreeSHAP explanation for one claim using "
        "the exact same frozen feature-engineering, preprocessing and "
        "XGBoost model pipeline used for production scoring. "
        "SHAP values are returned in raw-margin/log-odds space and "
        "are numerically verified against the model probability."
    ),
    response_description=(
        "Local TreeSHAP decomposition, strongest risk drivers "
        "and numerical consistency checks."
    ),
    operation_id="explain_claim",
)
def explain_claim(
    request: ExplainClaimRequest,
    service: PredictionServiceDependency,
) -> ExplainClaimResponse:
    """
    Explain one deployed-model prediction.

    The backend verifies:

        base_value + sum(SHAP)
            ~= raw XGBoost margin

    and:

        sigmoid(reconstructed margin)
            ~= predict_proba positive-class probability.

    If either consistency check fails, the explanation is not returned
    as a successful response.
    """

    try:

        explanation = (
            service.explain_single(
                request.claim,
                top_k=request.top_k,
            )
        )

    except VALIDATION_EXCEPTIONS as exc:

        _raise_validation_error(
            exc
        )

    except RuntimeError as exc:

        _raise_inference_error(
            exc
        )

    return ExplainClaimResponse(
        explanation=ClaimExplanation(
            **explanation
        )
    )


# =============================================================================
# Score batch
# =============================================================================


@router.post(
    "/score-batch",
    response_model=BatchScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a portfolio of claims",
    description=(
        "Score multiple health-insurance claims through the deployed "
        "fraud-risk model while preserving submitted row order."
    ),
    response_description=(
        "Fraud-risk prediction for every submitted claim."
    ),
    operation_id="score_batch",
)
def score_batch(
    request: BatchClaimsRequest,
    service: PredictionServiceDependency,
) -> BatchScoreResponse:
    """
    Score a portfolio of claims.
    """

    try:

        results = (
            service.score_batch(
                request.claims
            )
        )

    except VALIDATION_EXCEPTIONS as exc:

        _raise_validation_error(
            exc
        )

    except RuntimeError as exc:

        _raise_inference_error(
            exc
        )

    predictions = [
        ClaimScore(
            **result
        )
        for result in results
    ]

    if (
        len(
            predictions
        )
        != len(
            request.claims
        )
    ):

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Inference result count does not match "
                "submitted claim count."
            ),
        )

    return BatchScoreResponse(
        count=len(
            predictions
        ),
        predictions=predictions,
    )


# =============================================================================
# Investigation prioritization
# =============================================================================


@router.post(
    "/top-review",
    response_model=TopReviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Build a prioritized review population",
    description=(
        "Score and rank a portfolio from highest to lowest model risk, "
        "then return the highest-risk fraction according to the supplied "
        "investigation-capacity policy."
    ),
    response_description=(
        "Highest-risk claims selected for investigator review."
    ),
    operation_id="top_review",
)
def top_review(
    request: TopReviewRequest,
    service: PredictionServiceDependency,
) -> TopReviewResponse:
    """
    Rank and select claims for human review.

    review_fraction represents operational investigation capacity.
    It is not an automatic fraud-decision threshold.
    """

    try:

        results = (
            service.top_review(
                claims=request.claims,
                review_fraction=(
                    request.review_fraction
                ),
            )
        )

    except VALIDATION_EXCEPTIONS as exc:

        _raise_validation_error(
            exc
        )

    except RuntimeError as exc:

        _raise_inference_error(
            exc
        )

    predictions = [
        RankedClaimScore(
            **result
        )
        for result in results
    ]

    if not predictions:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Investigation ranking returned "
                "an empty review population."
            ),
        )

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