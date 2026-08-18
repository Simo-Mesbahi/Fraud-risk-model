from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from health_fraud.models.predict import (
    FORBIDDEN_FEATURES,
    PreparedModelInput,
    FraudScorer,
)


# =============================================================================
# Artifact contract
# =============================================================================


def test_artifact_paths_exist(
    model_path: Path,
    preprocessor_path: Path,
    metadata_path: Path,
) -> None:
    """
    All frozen inference artifacts must exist.
    """

    assert model_path.exists()
    assert preprocessor_path.exists()
    assert metadata_path.exists()


def test_fraud_scorer_rejects_missing_artifacts(
    tmp_path: Path,
) -> None:
    """
    FraudScorer must fail fast when persisted artifacts are unavailable.
    """

    with pytest.raises(
        FileNotFoundError,
        match="Missing required model artifacts",
    ):
        FraudScorer(
            model_path=(
                tmp_path
                / "missing_model.joblib"
            ),
            preprocessor_path=(
                tmp_path
                / "missing_preprocessor.joblib"
            ),
            metadata_path=(
                tmp_path
                / "missing_metadata.json"
            ),
        )


# =============================================================================
# Frozen metadata contract
# =============================================================================


def test_frozen_model_identity(
    fraud_scorer: FraudScorer,
) -> None:
    """
    The loaded scorer must expose the expected production identity.
    """

    assert (
        fraud_scorer.model_name
        == "XGBoost"
    )

    assert (
        fraud_scorer.model_version
        == "1.0.0"
    )

    assert (
        fraud_scorer.target
        == "is_fraud"
    )


def test_frozen_feature_count(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Source feature contract must remain frozen at 57 features.
    """

    assert (
        fraud_scorer.feature_count
        == 57
    )

    assert (
        len(
            fraud_scorer.features
        )
        == 57
    )


def test_frozen_features_are_unique(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Duplicate source features would invalidate the preprocessing contract.
    """

    assert (
        len(
            set(
                fraud_scorer.features
            )
        )
        == len(
            fraud_scorer.features
        )
    )


def test_no_forbidden_leakage_features(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Synthetic target or leakage variables must never enter inference.
    """

    leakage = (
        FORBIDDEN_FEATURES
        .intersection(
            fraud_scorer.features
        )
    )

    assert leakage == set()


# =============================================================================
# Transformed feature contract
# =============================================================================


def test_transformed_feature_count(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Persisted preprocessor must expose exactly 107 transformed features.
    """

    assert (
        len(
            fraud_scorer
            .transformed_feature_names
        )
        == 107
    )


def test_transformed_feature_names_are_unique(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Transformed feature names must remain unique for SHAP attribution.
    """

    names = (
        fraud_scorer
        .transformed_feature_names
    )

    assert (
        len(
            names
        )
        == len(
            set(
                names
            )
        )
    )


def test_model_and_preprocessor_feature_width_match(
    fraud_scorer: FraudScorer,
) -> None:
    """
    XGBoost input width must match persisted preprocessing output width.
    """

    model_feature_count = getattr(
        fraud_scorer.model,
        "n_features_in_",
        None,
    )

    if model_feature_count is not None:
        assert int(
            model_feature_count
        ) == 107


# =============================================================================
# Generic dataframe validation
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        "frame",
        42,
    ],
)
def test_validate_dataframe_rejects_non_dataframe(
    value: Any,
) -> None:
    """
    Inference must only accept pandas DataFrames.
    """

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        FraudScorer._validate_dataframe(
            value
        )


def test_validate_dataframe_rejects_empty_frame() -> None:
    """
    Empty inference batches are invalid.
    """

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        FraudScorer._validate_dataframe(
            pd.DataFrame()
        )


def test_validate_input_rejects_missing_features(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Engineered input must contain the complete frozen feature contract.
    """

    frame = pd.DataFrame(
        {
            "claim_amount": [
                100.0
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required model features",
    ):
        fraud_scorer.validate_input(
            frame
        )


# =============================================================================
# Feature preparation
# =============================================================================


def test_prepare_model_input_returns_contract_object(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Model preparation must return the complete shared inference representation.
    """

    frame = (
        demo_claims_frame
        .head(5)
        .copy()
    )

    prepared = (
        fraud_scorer
        .prepare_model_input(
            frame
        )
    )

    assert isinstance(
        prepared,
        PreparedModelInput,
    )


def test_prepare_model_input_preserves_row_count(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Feature engineering and preprocessing must not alter portfolio cardinality.
    """

    frame = (
        demo_claims_frame
        .head(10)
        .copy()
    )

    prepared = (
        fraud_scorer
        .prepare_model_input(
            frame
        )
    )

    assert (
        len(
            prepared.engineered
        )
        == len(
            frame
        )
    )

    assert (
        prepared.transformed.shape[0]
        == len(
            frame
        )
    )


def test_prepare_model_input_uses_frozen_feature_order(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Model input columns must exactly follow metadata feature order.
    """

    frame = (
        demo_claims_frame
        .head(3)
        .copy()
    )

    prepared = (
        fraud_scorer
        .prepare_model_input(
            frame
        )
    )

    assert (
        prepared
        .model_input
        .columns
        .tolist()
        == fraud_scorer.features
    )


def test_prepare_model_input_has_expected_width(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Persisted preprocessing must output the expected 107-dimensional vector.
    """

    prepared = (
        fraud_scorer
        .prepare_model_input(
            demo_claims_frame
            .head(4)
            .copy()
        )
    )

    assert (
        prepared.transformed.shape[1]
        == 107
    )


# =============================================================================
# Probability scoring
# =============================================================================


def test_predict_proba_preserves_cardinality(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    One probability must be returned per submitted claim.
    """

    frame = (
        demo_claims_frame
        .head(20)
        .copy()
    )

    probabilities = (
        fraud_scorer
        .predict_proba(
            frame
        )
    )

    assert (
        len(
            probabilities
        )
        == len(
            frame
        )
    )


def test_predict_proba_returns_finite_probabilities(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Model output must be finite and remain inside [0, 1].
    """

    probabilities = (
        fraud_scorer
        .predict_proba(
            demo_claims_frame
            .head(25)
            .copy()
        )
    )

    assert np.all(
        np.isfinite(
            probabilities
        )
    )

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )


def test_predict_proba_is_deterministic(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Frozen inference must be deterministic for identical input.
    """

    frame = (
        demo_claims_frame
        .head(8)
        .copy()
    )

    first = (
        fraud_scorer
        .predict_proba(
            frame
        )
    )

    second = (
        fraud_scorer
        .predict_proba(
            frame
        )
    )

    np.testing.assert_allclose(
        first,
        second,
        rtol=0.0,
        atol=1e-12,
    )


# =============================================================================
# Standard scoring
# =============================================================================


def test_score_preserves_claim_identity(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Scored rows must remain associated with source claim IDs.
    """

    frame = (
        demo_claims_frame
        .head(10)
        .copy()
    )

    scored = (
        fraud_scorer
        .score(
            frame
        )
    )

    assert (
        scored[
            "claim_id"
        ]
        .tolist()
        == frame[
            "claim_id"
        ]
        .tolist()
    )


def test_score_contract_columns(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Standard score output must expose the deployed scoring contract.
    """

    scored = (
        fraud_scorer
        .score(
            demo_claims_frame
            .head(3)
            .copy()
        )
    )

    assert {
        "claim_id",
        "fraud_risk_score",
        "model_name",
        "model_version",
    }.issubset(
        scored.columns
    )


# =============================================================================
# Ranking
# =============================================================================


def test_rank_orders_descending_risk(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Rank must order claims highest fraud risk first.
    """

    ranked = (
        fraud_scorer
        .rank(
            demo_claims_frame
            .head(50)
            .copy()
        )
    )

    scores = (
        ranked[
            "fraud_risk_score"
        ]
        .tolist()
    )

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_rank_is_contiguous(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Risk rank must be exactly 1..N.
    """

    ranked = (
        fraud_scorer
        .rank(
            demo_claims_frame
            .head(20)
            .copy()
        )
    )

    assert (
        ranked[
            "risk_rank"
        ]
        .tolist()
        == list(
            range(
                1,
                21,
            )
        )
    )


def test_highest_risk_percentile_is_one(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    The ranking contract defines highest risk as percentile 1.0.
    """

    ranked = (
        fraud_scorer
        .rank(
            demo_claims_frame
            .head(20)
            .copy()
        )
    )

    assert (
        ranked.iloc[0][
            "risk_percentile"
        ]
        == pytest.approx(
            1.0
        )
    )


# =============================================================================
# Top-fraction selection
# =============================================================================


def test_select_top_fraction_uses_ceiling(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Review population uses ceil(N * fraction).
    """

    frame = (
        demo_claims_frame
        .head(100)
        .copy()
    )

    selected = (
        fraud_scorer
        .select_top_fraction(
            dataframe=frame,
            review_fraction=0.03,
        )
    )

    assert len(
        selected
    ) == 3


def test_select_top_fraction_always_selects_at_least_one(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    A non-empty portfolio must yield at least one review candidate.
    """

    selected = (
        fraud_scorer
        .select_top_fraction(
            dataframe=(
                demo_claims_frame
                .head(1)
                .copy()
            ),
            review_fraction=0.01,
        )
    )

    assert len(
        selected
    ) == 1


@pytest.mark.parametrize(
    "review_fraction",
    [
        0,
        -0.01,
        1.01,
    ],
)
def test_select_top_fraction_rejects_invalid_fraction(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
    review_fraction: float,
) -> None:
    """
    Investigation capacity must lie in (0, 1].
    """

    with pytest.raises(
        ValueError,
        match="interval",
    ):
        fraud_scorer.select_top_fraction(
            dataframe=(
                demo_claims_frame
                .head(10)
                .copy()
            ),
            review_fraction=review_fraction,
        )


def test_select_top_fraction_marks_selected_claims(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Returned queue entries must explicitly carry review-selection state.
    """

    selected = (
        fraud_scorer
        .select_top_fraction(
            dataframe=(
                demo_claims_frame
                .head(100)
                .copy()
            ),
            review_fraction=0.03,
        )
    )

    assert (
        selected[
            "selected_for_review"
        ]
        .eq(
            True
        )
        .all()
    )


# =============================================================================
# TreeSHAP
# =============================================================================


def test_explain_single_requires_exactly_one_claim(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    explain_single() must reject multi-row inputs.
    """

    with pytest.raises(
        ValueError,
        match="exactly one claim",
    ):
        fraud_scorer.explain_single(
            demo_claims_frame
            .head(2)
            .copy()
        )


def test_explain_single_returns_full_transformed_vector(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Local TreeSHAP must expose one contribution per transformed feature.
    """

    explanation = (
        fraud_scorer
        .explain_single(
            demo_claims_frame
            .head(1)
            .copy()
        )
    )

    assert (
        explanation[
            "transformed_feature_count"
        ]
        == 107
    )

    assert (
        len(
            explanation[
                "all_contributions"
            ]
        )
        == 107
    )


def test_explain_single_shap_additivity(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    TreeSHAP base value plus contributions must reconstruct model margin.
    """

    explanation = (
        fraud_scorer
        .explain_single(
            demo_claims_frame
            .head(1)
            .copy()
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


def test_explain_single_probability_matches_predict_proba(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    Explainability and scoring must consume the same inference representation.
    """

    frame = (
        demo_claims_frame
        .head(1)
        .copy()
    )

    probability = float(
        fraud_scorer
        .predict_proba(
            frame
        )[0]
    )

    explanation = (
        fraud_scorer
        .explain_single(
            frame
        )
    )

    assert (
        float(
            explanation[
                "fraud_risk_score"
            ]
        )
        == pytest.approx(
            probability,
            abs=1e-12,
        )
    )


def test_shap_driver_direction_matches_sign(
    fraud_scorer: FraudScorer,
    demo_claims_frame: pd.DataFrame,
) -> None:
    """
    SHAP direction labels must agree with the numerical contribution sign.
    """

    explanation = (
        fraud_scorer
        .explain_single(
            demo_claims_frame
            .head(1)
            .copy()
        )
    )

    for contribution in (
        explanation[
            "all_contributions"
        ]
    ):

        value = float(
            contribution[
                "shap_value"
            ]
        )

        direction = (
            contribution[
                "direction"
            ]
        )

        if value > 0:
            assert direction == "increase"

        elif value < 0:
            assert direction == "decrease"

        else:
            assert direction == "neutral"


# =============================================================================
# Model information
# =============================================================================


def test_model_info_reports_complete_runtime_contract(
    fraud_scorer: FraudScorer,
) -> None:
    """
    Runtime metadata must describe the exact deployed inference stack.
    """

    info = (
        fraud_scorer
        .model_info()
    )

    assert info[
        "model_name"
    ] == "XGBoost"

    assert info[
        "model_version"
    ] == "1.0.0"

    assert info[
        "target"
    ] == "is_fraud"

    assert info[
        "feature_count"
    ] == 57

    assert info[
        "transformed_feature_count"
    ] == 107

    assert info[
        "probability_method"
    ] == "predict_proba"

    assert (
        info[
            "explainability"
        ][
            "available"
        ]
        is True
    )

    assert (
        info[
            "explainability"
        ][
            "method"
        ]
        == "TreeSHAP"
    )

    assert (
        info[
            "explainability"
        ][
            "output_space"
        ]
        == "raw_margin_log_odds"
    )