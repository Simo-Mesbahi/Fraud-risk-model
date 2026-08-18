from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from api.app.main import app

from frontend.utils.data import (
    serialize_row,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(
    scope="module",
)
def client() -> Generator[
    TestClient,
    None,
    None,
]:
    """
    Return an in-process HTTP client for the FastAPI application.
    """

    with TestClient(
        app
    ) as test_client:

        yield test_client


@pytest.fixture
def api_claim(
    demo_claims_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return one strict JSON-compatible claim payload.
    """

    return serialize_row(
        demo_claims_frame.iloc[0]
    )


@pytest.fixture
def api_claims(
    demo_claims_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Return 100 strict JSON-compatible claim payloads.
    """

    return [
        serialize_row(
            row
        )
        for _, row
        in (
            demo_claims_frame
            .head(100)
            .iterrows()
        )
    ]


# =============================================================================
# Root
# =============================================================================


def test_root_endpoint(
    client: TestClient,
) -> None:
    """
    Root endpoint must expose application identity.
    """

    response = client.get(
        "/"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "service"
        ]
        == "Health Insurance Fraud Risk API"
    )

    assert (
        payload[
            "version"
        ]
        == "1.1.0"
    )


# =============================================================================
# Health
# =============================================================================


def test_health_endpoint(
    client: TestClient,
) -> None:
    """
    Health endpoint must confirm model readiness.
    """

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "status"
        ]
        == "ok"
    )

    assert (
        payload[
            "model_loaded"
        ]
        is True
    )

    assert (
        payload[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        payload[
            "model_version"
        ]
        == "1.0.0"
    )


# =============================================================================
# Model info
# =============================================================================


def test_model_info_endpoint(
    client: TestClient,
) -> None:
    """
    Runtime contract must expose exact deployed model capabilities.
    """

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        payload[
            "model_version"
        ]
        == "1.0.0"
    )

    assert (
        payload[
            "target"
        ]
        == "is_fraud"
    )

    assert (
        payload[
            "feature_count"
        ]
        == 57
    )

    assert (
        payload[
            "transformed_feature_count"
        ]
        == 107
    )

    assert (
        payload[
            "probability_method"
        ]
        == "predict_proba"
    )

    assert (
        payload[
            "explainability"
        ][
            "available"
        ]
        is True
    )

    assert (
        payload[
            "explainability"
        ][
            "method"
        ]
        == "TreeSHAP"
    )


# =============================================================================
# Single scoring
# =============================================================================


def test_score_endpoint(
    client: TestClient,
    api_claim: dict[str, Any],
) -> None:
    """
    /score must execute real single-claim inference.
    """

    response = client.post(
        "/score",
        json={
            "claim":
                api_claim,
        },
    )

    assert (
        response.status_code
        == 200
    )

    prediction = (
        response.json()[
            "prediction"
        ]
    )

    assert (
        prediction[
            "claim_id"
        ]
        == api_claim[
            "claim_id"
        ]
    )

    assert (
        0.0
        <= float(
            prediction[
                "fraud_risk_score"
            ]
        )
        <= 1.0
    )

    assert (
        prediction[
            "model_name"
        ]
        == "XGBoost"
    )


def test_score_endpoint_is_deterministic(
    client: TestClient,
    api_claim: dict[str, Any],
) -> None:
    """
    Repeated HTTP inference for identical input must remain deterministic.
    """

    first = client.post(
        "/score",
        json={
            "claim":
                api_claim,
        },
    )

    second = client.post(
        "/score",
        json={
            "claim":
                api_claim,
        },
    )

    assert (
        first.status_code
        == 200
    )

    assert (
        second.status_code
        == 200
    )

    first_score = (
        first.json()[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    second_score = (
        second.json()[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    assert float(
        first_score
    ) == pytest.approx(
        float(
            second_score
        ),
        abs=1e-12,
    )


# =============================================================================
# Batch scoring
# =============================================================================


def test_score_batch_endpoint(
    client: TestClient,
    api_claims: list[dict[str, Any]],
) -> None:
    """
    /score-batch must preserve portfolio cardinality.
    """

    batch = (
        api_claims[:20]
    )

    response = client.post(
        "/score-batch",
        json={
            "claims":
                batch,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "count"
        ]
        == 20
    )

    assert (
        len(
            payload[
                "predictions"
            ]
        )
        == 20
    )


def test_score_batch_preserves_claim_order(
    client: TestClient,
    api_claims: list[dict[str, Any]],
) -> None:
    """
    Standard batch endpoint must preserve submission order.
    """

    batch = (
        api_claims[:20]
    )

    response = client.post(
        "/score-batch",
        json={
            "claims":
                batch,
        },
    )

    assert (
        response.status_code
        == 200
    )

    predictions = (
        response.json()[
            "predictions"
        ]
    )

    submitted = [
        claim[
            "claim_id"
        ]
        for claim
        in batch
    ]

    returned = [
        prediction[
            "claim_id"
        ]
        for prediction
        in predictions
    ]

    assert (
        returned
        == submitted
    )


# =============================================================================
# Top review
# =============================================================================


def test_top_review_endpoint(
    client: TestClient,
    api_claims: list[dict[str, Any]],
) -> None:
    """
    3% operational capacity over 100 claims must select three.
    """

    response = client.post(
        "/top-review",
        json={
            "claims":
                api_claims,

            "review_fraction":
                0.03,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "total_claims"
        ]
        == 100
    )

    assert (
        payload[
            "selected_claims"
        ]
        == 3
    )

    assert (
        payload[
            "review_fraction"
        ]
        == pytest.approx(
            0.03
        )
    )

    assert (
        len(
            payload[
                "predictions"
            ]
        )
        == 3
    )


def test_top_review_is_ranked(
    client: TestClient,
    api_claims: list[dict[str, Any]],
) -> None:
    """
    Investigation queue must be ranked highest-risk first.
    """

    response = client.post(
        "/top-review",
        json={
            "claims":
                api_claims,

            "review_fraction":
                0.03,
        },
    )

    assert (
        response.status_code
        == 200
    )

    predictions = (
        response.json()[
            "predictions"
        ]
    )

    ranks = [
        item[
            "risk_rank"
        ]
        for item
        in predictions
    ]

    assert ranks == [
        1,
        2,
        3,
    ]

    scores = [
        float(
            item[
                "fraud_risk_score"
            ]
        )
        for item
        in predictions
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


# =============================================================================
# Explainability
# =============================================================================


def test_explain_endpoint(
    client: TestClient,
    api_claim: dict[str, Any],
) -> None:
    """
    /explain must expose a complete TreeSHAP explanation.
    """

    response = client.post(
        "/explain",
        json={
            "claim":
                api_claim,

            "top_k":
                8,
        },
    )

    assert (
        response.status_code
        == 200
    )

    explanation = (
        response.json()[
            "explanation"
        ]
    )

    assert (
        explanation[
            "claim_id"
        ]
        == api_claim[
            "claim_id"
        ]
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

    assert (
        len(
            explanation[
                "all_contributions"
            ]
        )
        == 107
    )


def test_explain_endpoint_consistency(
    client: TestClient,
    api_claim: dict[str, Any],
) -> None:
    """
    HTTP explanation must retain SHAP numerical consistency.
    """

    response = client.post(
        "/explain",
        json={
            "claim":
                api_claim,

            "top_k":
                8,
        },
    )

    assert (
        response.status_code
        == 200
    )

    consistency = (
        response.json()[
            "explanation"
        ][
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


# =============================================================================
# Endpoint consistency
# =============================================================================


def test_score_and_explain_return_same_probability(
    client: TestClient,
    api_claim: dict[str, Any],
) -> None:
    """
    /score and /explain must represent the same frozen model prediction.
    """

    score_response = client.post(
        "/score",
        json={
            "claim":
                api_claim,
        },
    )

    explain_response = client.post(
        "/explain",
        json={
            "claim":
                api_claim,

            "top_k":
                8,
        },
    )

    assert (
        score_response.status_code
        == 200
    )

    assert (
        explain_response.status_code
        == 200
    )

    score = (
        score_response.json()[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    explained_score = (
        explain_response.json()[
            "explanation"
        ][
            "fraud_risk_score"
        ]
    )

    assert float(
        score
    ) == pytest.approx(
        float(
            explained_score
        ),
        abs=1e-12,
    )


# =============================================================================
# Validation errors
# =============================================================================


def test_score_rejects_missing_claim_payload(
    client: TestClient,
) -> None:
    """
    Missing required request fields must fail through FastAPI validation.
    """

    response = client.post(
        "/score",
        json={},
    )

    assert (
        response.status_code
        == 422
    )


def test_batch_rejects_empty_claim_list(
    client: TestClient,
) -> None:
    """
    Empty portfolios must not reach model inference.
    """

    response = client.post(
        "/score-batch",
        json={
            "claims":
                [],
        },
    )

    assert (
        response.status_code
        == 422
    )


@pytest.mark.parametrize(
    "review_fraction",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_top_review_rejects_invalid_fraction(
    client: TestClient,
    api_claims: list[dict[str, Any]],
    review_fraction: float,
) -> None:
    """
    Invalid operational capacities must return validation errors.
    """

    response = client.post(
        "/top-review",
        json={
            "claims":
                api_claims[:10],

            "review_fraction":
                review_fraction,
        },
    )

    assert (
        response.status_code
        == 422
    )