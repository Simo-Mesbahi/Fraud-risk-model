from __future__ import annotations

import math

from typing import Any

import numpy as np
import pandas as pd
import pytest

from api.app.services.prediction_service import (
    PredictionService,
)

from health_fraud.models.predict import (
    FraudScorer,
)


# =============================================================================
# Construction
# =============================================================================


def test_prediction_service_requires_fraud_scorer() -> None:
    """
    PredictionService must reject arbitrary scorer objects.
    """

    with pytest.raises(
        TypeError,
        match="FraudScorer",
    ):
        PredictionService(
            scorer=object(),  # type: ignore[arg-type]
        )


def test_prediction_service_accepts_valid_scorer(
    fraud_scorer: FraudScorer,
) -> None:
    """
    A valid frozen FraudScorer must initialize the service.
    """

    service = PredictionService(
        scorer=fraud_scorer,
    )

    assert service.scorer is fraud_scorer


# =============================================================================
# Claim validation
# =============================================================================


@pytest.mark.parametrize(
    "claim",
    [
        None,
        [],
        "claim",
        42,
    ],
)
def test_validate_claim_rejects_non_mapping(
    claim: Any,
) -> None:
    """
    One claim must be represented by a dictionary.
    """

    with pytest.raises(
        TypeError,
        match="claim must be a dictionary",
    ):
        PredictionService._validate_claim(
            claim
        )


def test_validate_claim_rejects_empty_mapping() -> None:
    """
    Empty claims are not valid inference inputs.
    """

    with pytest.raises(
        ValueError,
        match="claim cannot be empty",
    ):
        PredictionService._validate_claim(
            {}
        )


def test_validate_claim_accepts_non_empty_mapping() -> None:
    """
    Any non-empty mapping passes service-level structural validation.
    """

    PredictionService._validate_claim(
        {
            "claim_id": "CLM_TEST",
        }
    )


# =============================================================================
# Batch validation
# =============================================================================


@pytest.mark.parametrize(
    "claims",
    [
        None,
        {},
        "claims",
        1,
    ],
)
def test_validate_claims_requires_list(
    claims: Any,
) -> None:
    """
    Batch inference requires a list container.
    """

    with pytest.raises(
        TypeError,
        match="claims must be a list",
    ):
        PredictionService._validate_claims(
            claims
        )


def test_validate_claims_rejects_empty_list() -> None:
    """
    Empty inference batches are invalid.
    """

    with pytest.raises(
        ValueError,
        match="At least one claim is required",
    ):
        PredictionService._validate_claims(
            []
        )


def test_validate_claims_reports_invalid_position() -> None:
    """
    Validation failures should preserve the failing batch index.
    """

    claims: list[Any] = [
        {
            "claim_id": "CLM_1",
        },
        {},
    ]

    with pytest.raises(
        ValueError,
        match="Invalid claim at index 1",
    ):
        PredictionService._validate_claims(
            claims  # type: ignore[arg-type]
        )


# =============================================================================
# DataFrame conversion
# =============================================================================


def test_to_dataframe_preserves_claim_count(
    claim_batch: list[dict[str, Any]],
) -> None:
    """
    DataFrame conversion must preserve batch cardinality.
    """

    frame = (
        PredictionService
        ._to_dataframe(
            claim_batch
        )
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert len(frame) == len(
        claim_batch
    )


def test_to_dataframe_preserves_claim_id(
    single_claim: dict[str, Any],
) -> None:
    """
    Business identifiers must survive service conversion.
    """

    frame = (
        PredictionService
        ._to_dataframe(
            [
                single_claim
            ]
        )
    )

    assert (
        frame.iloc[0][
            "claim_id"
        ]
        == single_claim[
            "claim_id"
        ]
    )


# =============================================================================
# JSON serialization
# =============================================================================


@pytest.mark.parametrize(
    (
        "value",
        "expected",
    ),
    [
        (
            float("nan"),
            None,
        ),
        (
            float("inf"),
            None,
        ),
        (
            float("-inf"),
            None,
        ),
        (
            np.float64(1.25),
            1.25,
        ),
        (
            np.int64(7),
            7,
        ),
    ],
)
def test_json_safe_numeric_values(
    value: Any,
    expected: Any,
) -> None:
    """
    Model output must be converted into strict JSON-safe scalars.
    """

    result = (
        PredictionService
        ._json_safe(
            value
        )
    )

    assert result == expected


def test_json_safe_timestamp() -> None:
    """
    pandas timestamps must become ISO-8601 strings.
    """

    timestamp = pd.Timestamp(
        "2026-01-15 12:30:00"
    )

    result = (
        PredictionService
        ._json_safe(
            timestamp
        )
    )

    assert result == (
        "2026-01-15T12:30:00"
    )


def test_json_safe_nested_structure() -> None:
    """
    Serialization must recurse through nested model output.
    """

    payload = {
        "score": np.float64(
            0.42
        ),
        "values": [
            np.int64(2),
            float("nan"),
        ],
    }

    result = (
        PredictionService
        ._json_safe(
            payload
        )
    )

    assert result == {
        "score": 0.42,
        "values": [
            2,
            None,
        ],
    }


def test_records_requires_dataframe() -> None:
    """
    _records must not silently accept arbitrary inputs.
    """

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        PredictionService._records(
            []  # type: ignore[arg-type]
        )


# =============================================================================
# Model contract
# =============================================================================


def test_model_info_contract(
    prediction_service: PredictionService,
) -> None:
    """
    The service must expose the deployed frozen model identity.
    """

    info = (
        prediction_service
        .model_info()
    )

    assert (
        info[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        info[
            "model_version"
        ]
        == "1.0.0"
    )

    assert (
        info[
            "target"
        ]
        == "is_fraud"
    )

    assert (
        info[
            "feature_count"
        ]
        == 57
    )

    assert (
        info[
            "transformed_feature_count"
        ]
        == 107
    )

    assert (
        info[
            "probability_method"
        ]
        == "predict_proba"
    )

    assert (
        info[
            "review_policy"
        ][
            "fraction"
        ]
        == pytest.approx(
            0.03
        )
    )


# =============================================================================
# Single scoring
# =============================================================================


def test_score_single_returns_valid_probability(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    Single-claim scoring must return a valid fraud-risk probability.
    """

    result = (
        prediction_service
        .score_single(
            single_claim
        )
    )

    score = float(
        result[
            "fraud_risk_score"
        ]
    )

    assert (
        0.0
        <= score
        <= 1.0
    )

    assert (
        result[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        result[
            "model_version"
        ]
        == "1.0.0"
    )


def test_score_single_preserves_claim_identity(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    The prediction must remain associated with the submitted claim.
    """

    result = (
        prediction_service
        .score_single(
            single_claim
        )
    )

    assert (
        result[
            "claim_id"
        ]
        == single_claim[
            "claim_id"
        ]
    )


def test_score_single_is_deterministic(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    Frozen inference must be deterministic for identical input.
    """

    first = (
        prediction_service
        .score_single(
            single_claim
        )
    )

    second = (
        prediction_service
        .score_single(
            single_claim
        )
    )

    assert (
        first[
            "fraud_risk_score"
        ]
        == pytest.approx(
            second[
                "fraud_risk_score"
            ],
            abs=1e-12,
        )
    )


# =============================================================================
# Batch scoring
# =============================================================================


def test_score_batch_preserves_cardinality(
    prediction_service: PredictionService,
    claim_batch: list[dict[str, Any]],
) -> None:
    """
    Every submitted claim must receive exactly one score.
    """

    results = (
        prediction_service
        .score_batch(
            claim_batch
        )
    )

    assert len(
        results
    ) == len(
        claim_batch
    )


def test_score_batch_preserves_order(
    prediction_service: PredictionService,
    claim_batch: list[dict[str, Any]],
) -> None:
    """
    Standard batch scoring must preserve source claim order.
    """

    results = (
        prediction_service
        .score_batch(
            claim_batch
        )
    )

    submitted_ids = [
        claim[
            "claim_id"
        ]
        for claim
        in claim_batch
    ]

    returned_ids = [
        result[
            "claim_id"
        ]
        for result
        in results
    ]

    assert returned_ids == submitted_ids


def test_batch_scores_are_probabilities(
    prediction_service: PredictionService,
    claim_batch: list[dict[str, Any]],
) -> None:
    """
    All portfolio scores must lie in [0, 1].
    """

    results = (
        prediction_service
        .score_batch(
            claim_batch
        )
    )

    assert all(
        0.0
        <= float(
            result[
                "fraud_risk_score"
            ]
        )
        <= 1.0
        for result
        in results
    )


# =============================================================================
# Review prioritization
# =============================================================================


def test_top_review_selects_three_percent(
    prediction_service: PredictionService,
    review_claim_batch: list[dict[str, Any]],
) -> None:
    """
    100 claims with a 3% review policy must select exactly three claims.
    """

    results = (
        prediction_service
        .top_review(
            claims=review_claim_batch,
            review_fraction=0.03,
        )
    )

    assert len(
        results
    ) == 3


def test_top_review_is_sorted_by_descending_risk(
    prediction_service: PredictionService,
    review_claim_batch: list[dict[str, Any]],
) -> None:
    """
    Investigation population must be ordered highest risk first.
    """

    results = (
        prediction_service
        .top_review(
            claims=review_claim_batch,
            review_fraction=0.03,
        )
    )

    scores = [
        float(
            result[
                "fraud_risk_score"
            ]
        )
        for result
        in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_top_review_has_contiguous_ranks(
    prediction_service: PredictionService,
    review_claim_batch: list[dict[str, Any]],
) -> None:
    """
    Returned investigation ranks must begin at one and remain contiguous.
    """

    results = (
        prediction_service
        .top_review(
            claims=review_claim_batch,
            review_fraction=0.03,
        )
    )

    ranks = [
        int(
            result[
                "risk_rank"
            ]
        )
        for result
        in results
    ]

    assert ranks == [
        1,
        2,
        3,
    ]


def test_top_review_marks_every_returned_claim_selected(
    prediction_service: PredictionService,
    review_claim_batch: list[dict[str, Any]],
) -> None:
    """
    /top-review semantics require every returned record to be selected.
    """

    results = (
        prediction_service
        .top_review(
            claims=review_claim_batch,
            review_fraction=0.03,
        )
    )

    assert all(
        result[
            "selected_for_review"
        ]
        is True
        for result
        in results
    )


@pytest.mark.parametrize(
    "review_fraction",
    [
        0.0,
        -0.01,
        1.01,
    ],
)
def test_top_review_rejects_invalid_fraction(
    prediction_service: PredictionService,
    claim_batch: list[dict[str, Any]],
    review_fraction: float,
) -> None:
    """
    Review capacity must remain within the supported probability interval.
    """

    with pytest.raises(
        (
            ValueError,
            TypeError,
        )
    ):
        prediction_service.top_review(
            claims=claim_batch,
            review_fraction=review_fraction,
        )


# =============================================================================
# Explainability
# =============================================================================


def test_explain_single_returns_treeshap(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    Local explanation must use the deployed TreeSHAP contract.
    """

    explanation = (
        prediction_service
        .explain_single(
            single_claim,
            top_k=8,
        )
    )

    assert (
        explanation[
            "explanation_method"
        ]
        == "TreeSHAP"
    )

    assert (
        explanation[
            "explanation_space"
        ]
        == "raw_margin_log_odds"
    )

    assert (
        explanation[
            "transformed_feature_count"
        ]
        == 107
    )


def test_explain_single_preserves_claim_id(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    Local explanation must belong to the submitted claim.
    """

    explanation = (
        prediction_service
        .explain_single(
            single_claim,
            top_k=8,
        )
    )

    assert (
        explanation[
            "claim_id"
        ]
        == single_claim[
            "claim_id"
        ]
    )


def test_explanation_contains_full_feature_vector(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    TreeSHAP must expose one contribution per transformed feature.
    """

    explanation = (
        prediction_service
        .explain_single(
            single_claim,
            top_k=8,
        )
    )

    contributions = (
        explanation[
            "all_contributions"
        ]
    )

    assert len(
        contributions
    ) == 107


def test_explanation_additivity_is_valid(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    SHAP reconstruction must satisfy the deployed numerical tolerance.
    """

    explanation = (
        prediction_service
        .explain_single(
            single_claim,
            top_k=8,
        )
    )

    consistency = (
        explanation[
            "consistency"
        ]
    )

    assert (
        consistency[
            "shap_additivity_ok"
        ]
        is True
    )

    assert (
        consistency[
            "probability_consistency_ok"
        ]
        is True
    )

    assert (
        float(
            consistency[
                "raw_margin_absolute_error"
            ]
        )
        <= float(
            consistency[
                "shap_tolerance"
            ]
        )
    )

    assert (
        float(
            consistency[
                "probability_absolute_error"
            ]
        )
        <= float(
            consistency[
                "probability_tolerance"
            ]
        )
    )


def test_score_and_explanation_probability_match(
    prediction_service: PredictionService,
    single_claim: dict[str, Any],
) -> None:
    """
    Prediction and TreeSHAP paths must describe the same deployed model output.
    """

    prediction = (
        prediction_service
        .score_single(
            single_claim
        )
    )

    explanation = (
        prediction_service
        .explain_single(
            single_claim,
            top_k=8,
        )
    )

    assert (
        float(
            explanation[
                "fraud_risk_score"
            ]
        )
        == pytest.approx(
            float(
                prediction[
                    "fraud_risk_score"
                ]
            ),
            abs=1e-12,
        )
    )