from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from api.app.services.prediction_service import (
    PredictionService,
)

from frontend.utils.data import (
    serialize_row,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def explanation_claim(
    demo_claims_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return one JSON-safe claim for explainability integration tests.
    """

    return serialize_row(
        demo_claims_frame.iloc[0]
    )


# =============================================================================
# Core explanation contract
# =============================================================================


def test_explanation_contract(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    TreeSHAP explanation must expose the complete deployed contract.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    assert explanation[
        "claim_id"
    ] == explanation_claim[
        "claim_id"
    ]

    assert explanation[
        "model_name"
    ] == "XGBoost"

    assert explanation[
        "model_version"
    ] == "1.0.0"

    assert explanation[
        "explanation_method"
    ] == "TreeSHAP"

    assert explanation[
        "explanation_space"
    ] == "raw_margin_log_odds"

    assert explanation[
        "transformed_feature_count"
    ] == 107


# =============================================================================
# Numerical reconstruction
# =============================================================================


def test_shap_reconstructs_raw_margin(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    base_value + SHAP sum must reconstruct the XGBoost raw margin.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    reconstructed = (
        float(
            explanation[
                "base_value"
            ]
        )
        + float(
            explanation[
                "shap_sum"
            ]
        )
    )

    model_margin = float(
        explanation[
            "model_raw_margin"
        ]
    )

    tolerance = float(
        explanation[
            "consistency"
        ][
            "shap_tolerance"
        ]
    )

    assert reconstructed == pytest.approx(
        model_margin,
        abs=tolerance,
    )


def test_shap_reconstructs_probability(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    Reconstructed SHAP probability must match deployed fraud probability.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    reconstructed = float(
        explanation[
            "reconstructed_probability"
        ]
    )

    model_probability = float(
        explanation[
            "fraud_risk_score"
        ]
    )

    tolerance = float(
        explanation[
            "consistency"
        ][
            "probability_tolerance"
        ]
    )

    assert reconstructed == pytest.approx(
        model_probability,
        abs=tolerance,
    )


# =============================================================================
# Contribution vector
# =============================================================================


def test_all_contributions_cover_transformed_space(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    One SHAP contribution must exist for every transformed feature.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
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

    feature_names = [
        item[
            "feature"
        ]
        for item
        in contributions
    ]

    assert len(
        feature_names
    ) == len(
        set(
            feature_names
        )
    )


def test_absolute_shap_values_are_consistent(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    absolute_shap_value must equal abs(shap_value).
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    for item in (
        explanation[
            "all_contributions"
        ]
    ):

        assert float(
            item[
                "absolute_shap_value"
            ]
        ) == pytest.approx(
            abs(
                float(
                    item[
                        "shap_value"
                    ]
                )
            ),
            abs=1e-12,
        )


# =============================================================================
# Driver ranking
# =============================================================================


def test_strongest_drivers_are_sorted_by_absolute_impact(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    strongest_drivers must be sorted by descending absolute SHAP impact.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    drivers = (
        explanation[
            "strongest_drivers"
        ]
    )

    impacts = [
        float(
            item[
                "absolute_shap_value"
            ]
        )
        for item
        in drivers
    ]

    assert impacts == sorted(
        impacts,
        reverse=True,
    )

    assert len(
        drivers
    ) <= 8


def test_positive_drivers_have_positive_shap(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    Positive drivers must only contain positive contributions.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    assert all(
        float(
            item[
                "shap_value"
            ]
        ) > 0
        for item
        in explanation[
            "positive_drivers"
        ]
    )


def test_negative_drivers_have_negative_shap(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    Negative drivers must only contain negative contributions.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    assert all(
        float(
            item[
                "shap_value"
            ]
        ) < 0
        for item
        in explanation[
            "negative_drivers"
        ]
    )


# =============================================================================
# top_k contract
# =============================================================================


@pytest.mark.parametrize(
    "top_k",
    [
        1,
        3,
        8,
        20,
    ],
)
def test_explanation_respects_top_k(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
    top_k: int,
) -> None:
    """
    top_k must constrain ranked driver subsets but not full attribution vector.
    """

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=top_k,
        )
    )

    assert len(
        explanation[
            "all_contributions"
        ]
    ) == 107

    assert len(
        explanation[
            "strongest_drivers"
        ]
    ) <= top_k

    assert len(
        explanation[
            "positive_drivers"
        ]
    ) <= top_k

    assert len(
        explanation[
            "negative_drivers"
        ]
    ) <= top_k


def test_explanation_rejects_invalid_top_k(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    top_k must remain strictly positive.
    """

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        prediction_service.explain_single(
            explanation_claim,
            top_k=0,
        )


# =============================================================================
# Score / explanation identity
# =============================================================================


def test_explanation_matches_standard_scoring(
    prediction_service: PredictionService,
    explanation_claim: dict[str, Any],
) -> None:
    """
    Scoring and explanation must represent the exact same model output.
    """

    prediction = (
        prediction_service
        .score_single(
            explanation_claim
        )
    )

    explanation = (
        prediction_service
        .explain_single(
            explanation_claim,
            top_k=8,
        )
    )

    assert float(
        explanation[
            "fraud_risk_score"
        ]
    ) == pytest.approx(
        float(
            prediction[
                "fraud_risk_score"
            ]
        ),
        abs=1e-12,
    )