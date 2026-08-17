from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# =============================================================================
# Shared configuration
# =============================================================================


class APIModel(
    BaseModel,
):
    """
    Base model shared by all API contracts.

    Design goals
    ------------
    - Reject unexpected fields.
    - Reject NaN / Infinity values.
    - Keep OpenAPI contracts explicit.
    - Avoid silent schema drift between backend and frontend.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


# =============================================================================
# Type aliases
# =============================================================================


SHAPDirection = Literal[
    "increase",
    "decrease",
    "neutral",
]


ExplanationMethod = Literal[
    "TreeSHAP",
]


ExplanationSpace = Literal[
    "raw_margin_log_odds",
]


# =============================================================================
# Requests
# =============================================================================


class SingleClaimRequest(
    APIModel,
):
    """
    Request payload for scoring one claim.

    The claim object contains the source variables required by the
    frozen feature-engineering and inference pipeline.
    """

    claim: dict[
        str,
        Any,
    ] = Field(
        description=(
            "Complete source claim payload consumed by "
            "the frozen inference pipeline."
        ),
    )


class BatchClaimsRequest(
    APIModel,
):
    """
    Request payload for batch fraud-risk scoring.
    """

    claims: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "Claims to score. "
            "Maximum batch size: 10,000 claims."
        ),
    )


class TopReviewRequest(
    APIModel,
):
    """
    Request payload for portfolio ranking and review selection.

    review_fraction represents operational investigation capacity,
    not an individual fraud classification threshold.
    """

    claims: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "Portfolio of claims to rank."
        ),
    )

    review_fraction: float = Field(
        default=0.03,
        gt=0,
        le=1,
        description=(
            "Fraction of the highest-risk portfolio "
            "selected for investigator review."
        ),
        examples=[
            0.03
        ],
    )


class ExplainClaimRequest(
    APIModel,
):
    """
    Request payload for one local TreeSHAP explanation.

    The explanation is generated from exactly the same engineered
    and preprocessed representation used by production scoring.
    """

    claim: dict[
        str,
        Any,
    ] = Field(
        description=(
            "Complete claim payload to explain."
        ),
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description=(
            "Maximum number of strongest positive, "
            "negative and overall SHAP drivers returned."
        ),
        examples=[
            8
        ],
    )


# =============================================================================
# Health / runtime responses
# =============================================================================


class HealthResponse(
    APIModel,
):
    """
    Runtime health contract.
    """

    status: str = Field(
        description=(
            "Current inference-service status."
        ),
        examples=[
            "healthy"
        ],
    )

    model_loaded: bool = Field(
        description=(
            "Whether frozen model artifacts are loaded "
            "and available for inference."
        ),
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )


# =============================================================================
# Model information
# =============================================================================


class ExplainabilityInfo(
    APIModel,
):
    """
    Explainability capability exposed by the deployed model.
    """

    available: bool

    method: ExplanationMethod

    output_space: ExplanationSpace

    transformed_feature_count: int = Field(
        ge=1,
    )


class ModelInfoResponse(
    APIModel,
):
    """
    Frozen model and deployment contract.
    """

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    target: str = Field(
        min_length=1,
    )

    feature_count: int = Field(
        ge=1,
        description=(
            "Number of business/model-input features "
            "before preprocessing."
        ),
    )

    transformed_feature_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of features produced by the frozen "
            "preprocessing pipeline."
        ),
    )

    probability_method: str | None = Field(
        default=None,
        description=(
            "Method used to expose fraud-risk probabilities."
        ),
        examples=[
            "predict_proba"
        ],
    )

    explainability: ExplainabilityInfo | None = Field(
        default=None,
        description=(
            "Local model-explainability capability."
        ),
    )

    review_policy: dict[
        str,
        Any,
    ] | None = Field(
        default=None,
        description=(
            "Operational investigation-capacity policy."
        ),
    )


# =============================================================================
# Prediction responses
# =============================================================================


class ClaimScore(
    APIModel,
):
    """
    Common model-scoring result.
    """

    claim_id: str | None = Field(
        default=None,
    )

    fraud_risk_score: float = Field(
        ge=0,
        le=1,
        description=(
            "Predicted probability of the positive fraud class."
        ),
        examples=[
            0.7312
        ],
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )


class SingleScoreResponse(
    APIModel,
):
    """
    Response for POST /score.
    """

    prediction: ClaimScore


class BatchScoreResponse(
    APIModel,
):
    """
    Response for POST /score-batch.
    """

    count: int = Field(
        ge=1,
    )

    predictions: list[
        ClaimScore
    ] = Field(
        min_length=1,
    )


# =============================================================================
# Ranking responses
# =============================================================================


class RankedClaimScore(
    ClaimScore,
):
    """
    Claim score enriched with portfolio ranking information.
    """

    risk_rank: int = Field(
        ge=1,
        description=(
            "1-based portfolio rank, where rank 1 is "
            "the highest-risk claim."
        ),
    )

    risk_percentile: float = Field(
        gt=0,
        le=1,
        description=(
            "Relative portfolio risk percentile. "
            "Higher values represent higher relative risk."
        ),
    )

    review_fraction: float = Field(
        gt=0,
        le=1,
    )

    selected_for_review: bool = Field(
        description=(
            "Whether the claim belongs to the selected "
            "investigation-capacity population."
        ),
    )


class TopReviewResponse(
    APIModel,
):
    """
    Response for POST /top-review.
    """

    total_claims: int = Field(
        ge=1,
    )

    selected_claims: int = Field(
        ge=1,
    )

    review_fraction: float = Field(
        gt=0,
        le=1,
    )

    predictions: list[
        RankedClaimScore
    ] = Field(
        min_length=1,
    )


# =============================================================================
# SHAP contribution
# =============================================================================


class SHAPDriver(
    APIModel,
):
    """
    One local transformed-feature SHAP contribution.

    SHAP values are expressed in raw XGBoost margin/log-odds space.

    Positive values push the model toward higher fraud risk.
    Negative values push the model toward lower fraud risk.
    """

    feature: str = Field(
        min_length=1,
        description=(
            "Transformed model feature name."
        ),
        examples=[
            "claim_to_service_median_ratio"
        ],
    )

    feature_value: float = Field(
        description=(
            "Value of this feature after frozen preprocessing."
        ),
    )

    shap_value: float = Field(
        description=(
            "Signed local SHAP contribution in raw-margin space."
        ),
    )

    absolute_shap_value: float = Field(
        ge=0,
        description=(
            "Absolute magnitude of the SHAP contribution."
        ),
    )

    direction: SHAPDirection = Field(
        description=(
            "Whether the contribution increases or decreases "
            "the fraud-risk model margin."
        ),
    )


# =============================================================================
# SHAP consistency
# =============================================================================


class SHAPConsistency(
    APIModel,
):
    """
    Numerical verification of the local explanation.

    The explanation is considered valid when:

        base_value + sum(SHAP values)
            ≈ XGBoost raw margin

    and:

        sigmoid(reconstructed raw margin)
            ≈ predict_proba positive-class probability
    """

    shap_additivity_ok: bool

    probability_consistency_ok: bool

    raw_margin_absolute_error: float = Field(
        ge=0,
    )

    probability_absolute_error: float = Field(
        ge=0,
    )

    shap_tolerance: float = Field(
        gt=0,
    )

    probability_tolerance: float = Field(
        gt=0,
    )


# =============================================================================
# Local explanation
# =============================================================================


class ClaimExplanation(
    APIModel,
):
    """
    Complete local TreeSHAP explanation for one claim.
    """

    claim_id: str | None = Field(
        default=None,
    )

    fraud_risk_score: float = Field(
        ge=0,
        le=1,
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    explanation_method: ExplanationMethod

    explanation_space: ExplanationSpace

    transformed_feature_count: int = Field(
        ge=1,
    )

    base_value: float = Field(
        description=(
            "TreeSHAP expected model output in raw-margin space."
        ),
    )

    shap_sum: float = Field(
        description=(
            "Sum of all local SHAP contributions."
        ),
    )

    model_raw_margin: float = Field(
        description=(
            "Raw XGBoost output before logistic transformation."
        ),
    )

    reconstructed_raw_margin: float = Field(
        description=(
            "base_value + sum(SHAP contributions)."
        ),
    )

    reconstructed_probability: float = Field(
        ge=0,
        le=1,
        description=(
            "Probability reconstructed from the SHAP raw margin."
        ),
    )

    positive_drivers: list[
        SHAPDriver
    ] = Field(
        description=(
            "Strongest contributions increasing model fraud risk."
        ),
    )

    negative_drivers: list[
        SHAPDriver
    ] = Field(
        description=(
            "Strongest contributions decreasing model fraud risk."
        ),
    )

    strongest_drivers: list[
        SHAPDriver
    ] = Field(
        description=(
            "Strongest local contributions ranked by "
            "absolute SHAP magnitude."
        ),
    )

    all_contributions: list[
        SHAPDriver
    ] = Field(
        min_length=1,
        description=(
            "Complete transformed-feature SHAP decomposition."
        ),
    )

    consistency: SHAPConsistency


class ExplainClaimResponse(
    APIModel,
):
    """
    Response contract for POST /explain.
    """

    explanation: ClaimExplanation