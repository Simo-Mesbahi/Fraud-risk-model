from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
)


# =============================================================================
# Requests
# =============================================================================


class SingleClaimRequest(
    BaseModel,
):
    """
    One claim to score.

    The claim dictionary must contain the source variables
    required by the frozen model feature contract.
    """

    claim: dict[str, Any]

    model_config = {
        "extra": "forbid",
    }


class BatchClaimsRequest(
    BaseModel,
):
    """
    Multiple claims to score in one request.
    """

    claims: list[
        dict[str, Any]
    ] = Field(
        min_length=1,
        max_length=10_000,
    )

    model_config = {
        "extra": "forbid",
    }


class TopReviewRequest(
    BaseModel,
):
    """
    Rank claims and return only the highest-risk
    fraction for investigation.
    """

    claims: list[
        dict[str, Any]
    ] = Field(
        min_length=1,
        max_length=10_000,
    )

    review_fraction: float = Field(
        default=0.03,
        gt=0,
        le=1,
    )

    model_config = {
        "extra": "forbid",
    }


# =============================================================================
# Responses
# =============================================================================


class HealthResponse(
    BaseModel,
):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(
    BaseModel,
):
    model_name: str
    model_version: str
    target: str
    feature_count: int
    review_policy: dict[str, Any] | None


class ClaimScore(
    BaseModel,
):
    claim_id: str | None = None

    fraud_risk_score: float = Field(
        ge=0,
        le=1,
    )

    model_name: str
    model_version: str


class SingleScoreResponse(
    BaseModel,
):
    prediction: ClaimScore


class BatchScoreResponse(
    BaseModel,
):
    count: int
    predictions: list[
        ClaimScore
    ]


class RankedClaimScore(
    ClaimScore,
):
    risk_rank: int = Field(
        ge=1,
    )

    risk_percentile: float = Field(
        gt=0,
        le=1,
    )

    review_fraction: float = Field(
        gt=0,
        le=1,
    )

    selected_for_review: bool


class TopReviewResponse(
    BaseModel,
):
    total_claims: int
    selected_claims: int
    review_fraction: float

    predictions: list[
        RankedClaimScore
    ]