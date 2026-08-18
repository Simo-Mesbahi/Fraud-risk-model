from __future__ import annotations

from collections.abc import Generator
from typing import Any

import httpx
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from api.app.main import app

from frontend.api_client import (
    FraudAPIClient,
)

from frontend.utils.data import (
    serialize_row,
)


# =============================================================================
# Requests-compatible response adapter
# =============================================================================


class RequestsCompatibleResponse:
    """
    Adapt an ``httpx.Response`` returned by FastAPI TestClient to the subset
    of the ``requests.Response`` contract consumed by ``FraudAPIClient``.

    FraudAPIClient expects:
    - ``ok``
    - ``status_code``
    - ``headers``
    - ``text``
    - ``json()``

    FastAPI TestClient returns ``httpx.Response`` objects, which do not expose
    the requests-specific ``ok`` property.
    """

    def __init__(
        self,
        response: httpx.Response,
    ) -> None:
        self._response = response

    @property
    def ok(
        self,
    ) -> bool:
        """
        Mirror ``requests.Response.ok`` semantics.

        HTTP responses with status codes below 400 are considered successful.
        """

        return (
            self.status_code
            < 400
        )

    @property
    def status_code(
        self,
    ) -> int:
        """
        Return the HTTP status code.
        """

        return int(
            self._response.status_code
        )

    @property
    def headers(
        self,
    ) -> httpx.Headers:
        """
        Return response headers.
        """

        return (
            self._response.headers
        )

    @property
    def text(
        self,
    ) -> str:
        """
        Return response body as decoded text.
        """

        return (
            self._response.text
        )

    def json(
        self,
    ) -> Any:
        """
        Decode and return the JSON response body.
        """

        return (
            self._response.json()
        )


# =============================================================================
# Test HTTP session adapter
# =============================================================================


class ClientSessionAdapter:
    """
    requests.Session-compatible adapter around FastAPI TestClient.

    ``FraudAPIClient`` dispatches all network communication through
    ``Session.request()``.

    This adapter preserves that interface while routing requests directly into
    the real FastAPI ASGI application.

    Advantages
    ----------
    - no external Uvicorn process;
    - no TCP socket dependency;
    - no Docker dependency;
    - deterministic integration testing;
    - real frontend client;
    - real FastAPI routes;
    - real PredictionService;
    - real XGBoost inference;
    - real TreeSHAP explainability.
    """

    def __init__(
        self,
        client: TestClient,
    ) -> None:
        self.client = client

        self.headers: dict[
            str,
            str,
        ] = {}

    def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> RequestsCompatibleResponse:
        """
        Execute a requests-style HTTP call against FastAPI TestClient.
        """

        normalized_method = (
            str(
                method
            )
            .strip()
            .upper()
        )

        if not normalized_method:
            raise ValueError(
                "HTTP method cannot be empty."
            )

        path = self._path(
            url
        )

        # FraudAPIClient supplies requests-style timeout configuration:
        #
        #     timeout=(connect_timeout, read_timeout)
        #
        # TestClient operates entirely in-process, so this argument must not
        # be forwarded to the underlying httpx test transport.
        kwargs.pop(
            "timeout",
            None,
        )

        response = (
            self.client.request(
                method=normalized_method,
                url=path,
                **kwargs,
            )
        )

        return RequestsCompatibleResponse(
            response
        )

    def close(
        self,
    ) -> None:
        """
        Mirror ``requests.Session.close()``.

        No resources are owned directly by this adapter because TestClient
        lifecycle management belongs to the fixture creating it.
        """

        return None

    @staticmethod
    def _path(
        url: str,
    ) -> str:
        """
        Convert an absolute HTTP URL into a TestClient-relative path.

        Examples
        --------
        http://testserver/health
            -> /health

        http://testserver/model-info
            -> /model-info

        /score
            -> /score
        """

        text = (
            str(
                url
            )
            .strip()
        )

        if not text:
            raise ValueError(
                "URL cannot be empty."
            )

        marker = "://"

        if marker not in text:

            if text.startswith(
                "/"
            ):
                return text

            return (
                "/"
                + text
            )

        remainder = (
            text.split(
                marker,
                1,
            )[1]
        )

        slash_index = (
            remainder.find(
                "/"
            )
        )

        if slash_index == -1:
            return "/"

        return remainder[
            slash_index:
        ]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(
    scope="module",
)
def fastapi_client() -> Generator[
    TestClient,
    None,
    None,
]:
    """
    Start the real FastAPI application in-process.

    Application lifespan startup and shutdown hooks are executed by the
    TestClient context manager.
    """

    with TestClient(
        app
    ) as client:
        yield client


@pytest.fixture
def frontend_client(
    fastapi_client: TestClient,
) -> Generator[
    FraudAPIClient,
    None,
    None,
]:
    """
    Return the real frontend API client wired to the real FastAPI application.
    """

    client = FraudAPIClient(
        base_url="http://testserver",
        connect_timeout=1.0,
        read_timeout=10.0,
        retry_total=0,
    )

    # FraudAPIClient creates a real requests.Session during initialization.
    # For this in-process integration test it is replaced by our deterministic
    # TestClient-backed session adapter.
    original_session = (
        client.session
    )

    client.session = (
        ClientSessionAdapter(
            fastapi_client
        )
    )

    # The original network-backed requests.Session is no longer needed.
    original_session.close()

    try:
        yield client

    finally:
        client.close()


@pytest.fixture
def frontend_claim(
    demo_claims_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return one strict JSON-compatible frontend claim.
    """

    return serialize_row(
        demo_claims_frame.iloc[0]
    )


@pytest.fixture
def frontend_claims(
    demo_claims_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Return a deterministic 100-claim JSON-compatible portfolio.
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
# Client construction contract
# =============================================================================


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "localhost:8000",
        "fraud-api:8000",
    ],
)
def test_client_rejects_invalid_base_url(
    url: str,
) -> None:
    """
    FraudAPIClient must require an explicit HTTP or HTTPS backend URL.
    """

    with pytest.raises(
        ValueError
    ):
        FraudAPIClient(
            base_url=url
        )


def test_client_normalizes_trailing_slash() -> None:
    """
    Base URL normalization must prevent duplicate path separators.
    """

    client = FraudAPIClient(
        base_url=(
            "http://localhost:8000/"
        )
    )

    try:
        assert (
            client.base_url
            == "http://localhost:8000"
        )

    finally:
        client.close()


# =============================================================================
# Health contract
# =============================================================================


def test_frontend_health_contract(
    frontend_client: FraudAPIClient,
) -> None:
    """
    Frontend must consume and validate the deployed health contract.
    """

    payload = (
        frontend_client
        .health()
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
# Model information contract
# =============================================================================


def test_frontend_model_info_contract(
    frontend_client: FraudAPIClient,
) -> None:
    """
    Frontend must receive the exact deployed model contract.
    """

    payload = (
        frontend_client
        .model_info()
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

    explainability = (
        payload[
            "explainability"
        ]
    )

    assert (
        explainability[
            "available"
        ]
        is True
    )

    assert (
        explainability[
            "method"
        ]
        == "TreeSHAP"
    )

    assert (
        explainability[
            "output_space"
        ]
        == "raw_margin_log_odds"
    )

    assert (
        explainability[
            "transformed_feature_count"
        ]
        == 107
    )


# =============================================================================
# Single scoring contract
# =============================================================================


def test_frontend_score_claim(
    frontend_client: FraudAPIClient,
    frontend_claim: dict[str, Any],
) -> None:
    """
    Streamlit client must receive one valid reconciled prediction.
    """

    response = (
        frontend_client
        .score_claim(
            frontend_claim
        )
    )

    prediction = (
        response[
            "prediction"
        ]
    )

    assert (
        prediction[
            "claim_id"
        ]
        == frontend_claim[
            "claim_id"
        ]
    )

    score = float(
        prediction[
            "fraud_risk_score"
        ]
    )

    assert (
        0.0
        <= score
        <= 1.0
    )

    assert (
        prediction[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        prediction[
            "model_version"
        ]
        == "1.0.0"
    )


def test_frontend_score_is_deterministic(
    frontend_client: FraudAPIClient,
    frontend_claim: dict[str, Any],
) -> None:
    """
    Identical frontend requests must return identical model probabilities.
    """

    first = (
        frontend_client
        .score_claim(
            frontend_claim
        )
    )

    second = (
        frontend_client
        .score_claim(
            frontend_claim
        )
    )

    first_score = float(
        first[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    second_score = float(
        second[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    assert first_score == pytest.approx(
        second_score,
        abs=1e-12,
    )


# =============================================================================
# Explainability contract
# =============================================================================


def test_frontend_explain_claim(
    frontend_client: FraudAPIClient,
    frontend_claim: dict[str, Any],
) -> None:
    """
    Frontend must consume a complete TreeSHAP explanation.
    """

    response = (
        frontend_client
        .explain_claim(
            frontend_claim,
            top_k=8,
        )
    )

    explanation = (
        response[
            "explanation"
        ]
    )

    assert (
        explanation[
            "claim_id"
        ]
        == frontend_claim[
            "claim_id"
        ]
    )

    assert (
        explanation[
            "model_name"
        ]
        == "XGBoost"
    )

    assert (
        explanation[
            "model_version"
        ]
        == "1.0.0"
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


def test_frontend_explanation_consistency(
    frontend_client: FraudAPIClient,
    frontend_claim: dict[str, Any],
) -> None:
    """
    Frontend must receive a numerically valid SHAP reconstruction.
    """

    response = (
        frontend_client
        .explain_claim(
            frontend_claim,
            top_k=8,
        )
    )

    consistency = (
        response[
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


# =============================================================================
# Score / explanation consistency
# =============================================================================


def test_frontend_score_and_explanation_match(
    frontend_client: FraudAPIClient,
    frontend_claim: dict[str, Any],
) -> None:
    """
    Scoring and explainability must represent the exact same frozen model.
    """

    score_response = (
        frontend_client
        .score_claim(
            frontend_claim
        )
    )

    explanation_response = (
        frontend_client
        .explain_claim(
            frontend_claim,
            top_k=8,
        )
    )

    score = float(
        score_response[
            "prediction"
        ][
            "fraud_risk_score"
        ]
    )

    explained_score = float(
        explanation_response[
            "explanation"
        ][
            "fraud_risk_score"
        ]
    )

    assert score == pytest.approx(
        explained_score,
        abs=1e-12,
    )


# =============================================================================
# Batch scoring contract
# =============================================================================


def test_frontend_batch_contract(
    frontend_client: FraudAPIClient,
    frontend_claims: list[dict[str, Any]],
) -> None:
    """
    Frontend batch scoring must preserve portfolio cardinality.
    """

    claims = (
        frontend_claims[:20]
    )

    response = (
        frontend_client
        .score_batch(
            claims
        )
    )

    assert (
        response[
            "count"
        ]
        == 20
    )

    predictions = (
        response[
            "predictions"
        ]
    )

    assert (
        len(
            predictions
        )
        == 20
    )

    assert all(
        (
            0.0
            <= float(
                prediction[
                    "fraud_risk_score"
                ]
            )
            <= 1.0
        )
        for prediction
        in predictions
    )


def test_frontend_batch_preserves_identity_and_order(
    frontend_client: FraudAPIClient,
    frontend_claims: list[dict[str, Any]],
) -> None:
    """
    Batch predictions must remain aligned with submitted claim identities.
    """

    claims = (
        frontend_claims[:20]
    )

    response = (
        frontend_client
        .score_batch(
            claims
        )
    )

    submitted_ids = [
        claim[
            "claim_id"
        ]
        for claim
        in claims
    ]

    returned_ids = [
        prediction[
            "claim_id"
        ]
        for prediction
        in response[
            "predictions"
        ]
    ]

    assert (
        returned_ids
        == submitted_ids
    )


# =============================================================================
# Investigation queue contract
# =============================================================================


def test_frontend_top_review_contract(
    frontend_client: FraudAPIClient,
    frontend_claims: list[dict[str, Any]],
) -> None:
    """
    Frontend must consume the operational top-fraction investigation queue.
    """

    response = (
        frontend_client
        .top_review(
            frontend_claims,
            0.03,
        )
    )

    assert (
        response[
            "total_claims"
        ]
        == 100
    )

    assert (
        response[
            "selected_claims"
        ]
        == 3
    )

    assert (
        response[
            "review_fraction"
        ]
        == pytest.approx(
            0.03
        )
    )

    predictions = (
        response[
            "predictions"
        ]
    )

    assert (
        len(
            predictions
        )
        == 3
    )

    assert [
        int(
            item[
                "risk_rank"
            ]
        )
        for item
        in predictions
    ] == [
        1,
        2,
        3,
    ]

    assert all(
        (
            item[
                "selected_for_review"
            ]
            is True
        )
        for item
        in predictions
    )


def test_frontend_top_review_sorted_descending(
    frontend_client: FraudAPIClient,
    frontend_claims: list[dict[str, Any]],
) -> None:
    """
    Investigation queue exposed to Streamlit must be highest-risk first.
    """

    response = (
        frontend_client
        .top_review(
            frontend_claims,
            0.03,
        )
    )

    scores = [
        float(
            item[
                "fraud_risk_score"
            ]
        )
        for item
        in response[
            "predictions"
        ]
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


# =============================================================================
# Input protection
# =============================================================================


def test_frontend_rejects_empty_batch(
    frontend_client: FraudAPIClient,
) -> None:
    """
    Empty portfolios must fail explicitly.
    """

    with pytest.raises(
        Exception
    ):
        frontend_client.score_batch(
            []
        )


@pytest.mark.parametrize(
    "fraction",
    [
        0.0,
        -0.01,
        1.01,
    ],
)
def test_frontend_rejects_invalid_review_fraction(
    frontend_client: FraudAPIClient,
    frontend_claims: list[dict[str, Any]],
    fraction: float,
) -> None:
    """
    Invalid investigation capacities must fail explicitly.
    """

    with pytest.raises(
        Exception
    ):
        frontend_client.top_review(
            frontend_claims[:10],
            fraction,
        )


# =============================================================================
# Runtime model identity
# =============================================================================


def test_frontend_runtime_model_identity_is_consistent(
    frontend_client: FraudAPIClient,
) -> None:
    """
    Health and model-info must identify the same deployed model.
    """

    health = (
        frontend_client
        .health()
    )

    model = (
        frontend_client
        .model_info()
    )

    assert (
        health[
            "model_name"
        ]
        == model[
            "model_name"
        ]
    )

    assert (
        health[
            "model_version"
        ]
        == model[
            "model_version"
        ]
    )