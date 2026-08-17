from __future__ import annotations

import math

from datetime import (
    date,
    datetime,
)

from typing import Any

import numpy as np
import pandas as pd
import requests

from requests.adapters import (
    HTTPAdapter,
)

from urllib3.util.retry import (
    Retry,
)


# =============================================================================
# Exceptions
# =============================================================================


class FraudAPIError(
    RuntimeError
):
    """
    Base exception for frontend-to-inference API communication.
    """


class FraudAPIUnavailableError(
    FraudAPIError
):
    """
    Raised when the inference API cannot be reached.
    """


class FraudAPITimeoutError(
    FraudAPIError
):
    """
    Raised when an inference request exceeds its configured timeout.
    """


class FraudAPIResponseError(
    FraudAPIError
):
    """
    Raised when the API returns an unsuccessful or malformed response.
    """


class FraudAPIContractError(
    FraudAPIError
):
    """
    Raised when a successful API response violates the expected
    frontend/backend contract.
    """


# =============================================================================
# Client
# =============================================================================


class FraudAPIClient:
    """
    Production-oriented HTTP client for Health Fraud Intelligence.

    Responsibilities
    ----------------
    - persistent HTTP session and connection pooling;
    - separate connect/read timeouts;
    - safe retries for idempotent requests only;
    - recursive JSON-safe payload normalization;
    - FastAPI/Pydantic error parsing;
    - typed frontend communication failures;
    - defensive response-contract validation;
    - claim identity reconciliation;
    - model identity consistency checks;
    - single-claim scoring;
    - portfolio batch scoring;
    - investigation prioritization;
    - local TreeSHAP explainability;
    - numerical SHAP consistency validation.

    Important
    ---------
    POST inference requests are deliberately not retried automatically.

    Although the current scoring API is computational, avoiding POST retries
    protects the frontend if the backend later introduces persistence,
    audit logging, workflow events or other side effects.
    """

    MAX_BATCH_SIZE = 10_000

    MAX_EXPLANATION_TOP_K = 50

    DEFAULT_EXPLANATION_TOP_K = 8

    SHAP_DIRECTION_VALUES = {
        "increase",
        "decrease",
        "neutral",
    }

    EXPECTED_EXPLANATION_METHOD = (
        "TreeSHAP"
    )

    EXPECTED_EXPLANATION_SPACE = (
        "raw_margin_log_odds"
    )

    FLOAT_COMPARISON_TOLERANCE = (
        1e-12
    )

    SHAP_DRIVER_TOLERANCE = (
        1e-8
    )

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 3.0,
        read_timeout: float = 60.0,
        retry_total: int = 2,
    ) -> None:

        normalized = (
            str(
                base_url
            )
            .strip()
            .rstrip("/")
        )

        if not normalized:

            raise ValueError(
                "base_url cannot be empty."
            )

        if not (
            normalized.startswith(
                "http://"
            )
            or normalized.startswith(
                "https://"
            )
        ):

            raise ValueError(
                (
                    "base_url must begin with "
                    "http:// or https://."
                )
            )

        connect_timeout = (
            self._positive_float(
                connect_timeout,
                field="connect_timeout",
            )
        )

        read_timeout = (
            self._positive_float(
                read_timeout,
                field="read_timeout",
            )
        )

        retry_total = (
            self._require_non_negative_int_input(
                retry_total,
                field="retry_total",
            )
        )

        self.base_url = normalized

        self.timeout = (
            connect_timeout,
            read_timeout,
        )

        self.session = (
            requests.Session()
        )

        self.session.headers.update(
            {
                "Accept":
                    "application/json",

                "Content-Type":
                    "application/json",

                "User-Agent":
                    (
                        "health-fraud-intelligence-"
                        "frontend/7.0"
                    ),
            }
        )

        self._configure_retries(
            retry_total
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(
        self,
    ) -> None:
        """
        Close pooled HTTP connections.
        """

        self.session.close()

    def __enter__(
        self,
    ) -> FraudAPIClient:

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()

    # =========================================================================
    # Retry configuration
    # =========================================================================

    def _configure_retries(
        self,
        retry_total: int,
    ) -> None:
        """
        Configure connection pooling and safe idempotent retries.

        Only GET/HEAD/OPTIONS requests may be replayed automatically.
        """

        retry = Retry(
            total=retry_total,

            connect=retry_total,

            # Never replay after response streaming has started.
            read=0,

            status=retry_total,

            backoff_factor=0.25,

            status_forcelist=(
                429,
                502,
                503,
                504,
            ),

            allowed_methods=frozenset(
                {
                    "GET",
                    "HEAD",
                    "OPTIONS",
                }
            ),

            respect_retry_after_header=True,

            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=20,
        )

        self.session.mount(
            "http://",
            adapter,
        )

        self.session.mount(
            "https://",
            adapter,
        )

    # =========================================================================
    # JSON normalization
    # =========================================================================

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:
        """
        Recursively normalize frontend values into JSON-safe objects.

        Handles common values originating from:
        - pandas DataFrames;
        - Parquet datasets;
        - CSV imports;
        - NumPy arrays/scalars;
        - Streamlit widgets;
        - datetime objects.

        Missing and non-finite numerical values are represented as null.
        """

        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):

            return {
                str(
                    key
                ):
                    cls._json_safe(
                        item
                    )

                for (
                    key,
                    item,
                ) in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):

            return [
                cls._json_safe(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            np.ndarray,
        ):

            return cls._json_safe(
                value.tolist()
            )

        if isinstance(
            value,
            np.generic,
        ):

            return cls._json_safe(
                value.item()
            )

        if isinstance(
            value,
            pd.Timestamp,
        ):

            if pd.isna(
                value
            ):
                return None

            return value.isoformat()

        if isinstance(
            value,
            pd.Timedelta,
        ):

            if pd.isna(
                value
            ):
                return None

            return value.isoformat()

        if isinstance(
            value,
            datetime,
        ):

            return value.isoformat()

        if isinstance(
            value,
            date,
        ):

            return value.isoformat()

        try:

            missing = pd.isna(
                value
            )

            if isinstance(
                missing,
                (
                    bool,
                    np.bool_,
                ),
            ) and bool(
                missing
            ):

                return None

        except (
            TypeError,
            ValueError,
        ):
            pass

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):

            numeric = float(
                value
            )

            if not math.isfinite(
                numeric
            ):

                return None

            return numeric

        return value

    # =========================================================================
    # Error parsing
    # =========================================================================

    @staticmethod
    def _extract_error_detail(
        response: requests.Response,
    ) -> str:
        """
        Extract human-readable FastAPI/Pydantic error information.
        """

        try:

            payload = (
                response.json()
            )

        except ValueError:

            text = (
                response.text
                .strip()
            )

            return (
                text
                or f"HTTP {response.status_code}"
            )

        if not isinstance(
            payload,
            dict,
        ):

            return str(
                payload
            )

        detail = payload.get(
            "detail"
        )

        # ---------------------------------------------------------------------
        # FastAPI / Pydantic validation errors
        # ---------------------------------------------------------------------

        if isinstance(
            detail,
            list,
        ):

            messages: list[
                str
            ] = []

            for item in detail:

                if not isinstance(
                    item,
                    dict,
                ):

                    messages.append(
                        str(
                            item
                        )
                    )

                    continue

                location = item.get(
                    "loc",
                    [],
                )

                message = item.get(
                    "msg",
                    "Validation error",
                )

                location_text = (
                    " → ".join(
                        str(
                            part
                        )
                        for part
                        in location
                    )
                )

                if location_text:

                    messages.append(
                        (
                            f"{location_text}: "
                            f"{message}"
                        )
                    )

                else:

                    messages.append(
                        str(
                            message
                        )
                    )

            formatted = (
                "; ".join(
                    messages
                )
            )

        elif detail is not None:

            formatted = str(
                detail
            )

        elif payload.get(
            "message"
        ) is not None:

            formatted = str(
                payload[
                    "message"
                ]
            )

        else:

            formatted = str(
                payload
            )

        request_id = (
            payload.get(
                "request_id"
            )
            or response.headers.get(
                "X-Request-ID"
            )
        )

        if request_id:

            formatted = (
                f"{formatted} "
                f"[request_id={request_id}]"
            )

        return formatted

    # =========================================================================
    # Core HTTP request
    # =========================================================================

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute one API request and validate its response envelope.

        JSON payloads are normalized before transmission to prevent
        pandas / NumPy serialization failures.
        """

        if not isinstance(
            method,
            str,
        ):

            raise TypeError(
                "method must be a string."
            )

        method = (
            method
            .strip()
            .upper()
        )

        if not method:

            raise ValueError(
                "method cannot be empty."
            )

        if not isinstance(
            path,
            str,
        ):

            raise TypeError(
                "path must be a string."
            )

        path = path.strip()

        if not path:

            raise ValueError(
                "path cannot be empty."
            )

        if not path.startswith(
            "/"
        ):

            path = (
                "/"
                + path
            )

        url = (
            f"{self.base_url}{path}"
        )

        if (
            "json"
            in kwargs
        ):

            kwargs[
                "json"
            ] = self._json_safe(
                kwargs[
                    "json"
                ]
            )

        try:

            response = (
                self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs,
                )
            )

        except requests.ConnectTimeout as exc:

            raise FraudAPITimeoutError(
                (
                    "Connection to the inference "
                    "API timed out."
                )
            ) from exc

        except requests.ReadTimeout as exc:

            raise FraudAPITimeoutError(
                (
                    "The inference API exceeded "
                    "the configured response timeout."
                )
            ) from exc

        except requests.ConnectionError as exc:

            raise FraudAPIUnavailableError(
                (
                    "The inference API is currently "
                    "unavailable."
                )
            ) from exc

        except requests.RequestException as exc:

            raise FraudAPIError(
                (
                    "Unexpected HTTP communication "
                    f"failure: {exc}"
                )
            ) from exc

        if not response.ok:

            detail = (
                self._extract_error_detail(
                    response
                )
            )

            raise FraudAPIResponseError(
                (
                    f"API request failed "
                    f"[{response.status_code}] "
                    f"{detail}"
                )
            )

        if (
            response.status_code
            == 204
        ):

            return {}

        try:

            payload = (
                response.json()
            )

        except ValueError as exc:

            raise FraudAPIResponseError(
                (
                    "Inference API returned "
                    "invalid JSON."
                )
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):

            raise FraudAPIResponseError(
                (
                    "Inference API returned an "
                    "unexpected response structure."
                )
            )

        return payload

    # =========================================================================
    # Generic contract helpers
    # =========================================================================

    @staticmethod
    def _require_mapping(
        payload: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """
        Require one dictionary field from an API response.
        """

        value = payload.get(
            key
        )

        if not isinstance(
            value,
            dict,
        ):

            raise FraudAPIContractError(
                (
                    f"API response field '{key}' "
                    "must be an object."
                )
            )

        return value

    @staticmethod
    def _require_list(
        payload: dict[str, Any],
        key: str,
    ) -> list[Any]:
        """
        Require one list field from an API response.
        """

        value = payload.get(
            key
        )

        if not isinstance(
            value,
            list,
        ):

            raise FraudAPIContractError(
                (
                    f"API response field '{key}' "
                    "must be a list."
                )
            )

        return value

    # =========================================================================
    # Health
    # =========================================================================

    def health(
        self,
    ) -> dict[str, Any]:
        """
        Retrieve and validate inference-service health.
        """

        payload = self._request(
            "GET",
            "/health",
        )

        required = {
            "status",
            "model_loaded",
            "model_name",
            "model_version",
        }

        missing = (
            required
            - set(
                payload
            )
        )

        if missing:

            raise FraudAPIContractError(
                (
                    "Health response is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        if not isinstance(
            payload[
                "model_loaded"
            ],
            bool,
        ):

            raise FraudAPIContractError(
                (
                    "Health field 'model_loaded' "
                    "must be boolean."
                )
            )

        self._require_non_empty_string(
            payload[
                "status"
            ],
            field="status",
        )

        self._require_non_empty_string(
            payload[
                "model_name"
            ],
            field="model_name",
        )

        self._require_non_empty_string(
            payload[
                "model_version"
            ],
            field="model_version",
        )

        return payload

    # =========================================================================
    # Model information
    # =========================================================================

    def model_info(
        self,
    ) -> dict[str, Any]:
        """
        Retrieve and validate the deployed model contract.
        """

        payload = self._request(
            "GET",
            "/model-info",
        )

        required = {
            "model_name",
            "model_version",
            "target",
            "feature_count",
        }

        missing = (
            required
            - set(
                payload
            )
        )

        if missing:

            raise FraudAPIContractError(
                (
                    "Model-info response is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        self._require_non_empty_string(
            payload[
                "model_name"
            ],
            field="model_name",
        )

        self._require_non_empty_string(
            payload[
                "model_version"
            ],
            field="model_version",
        )

        self._require_non_empty_string(
            payload[
                "target"
            ],
            field="target",
        )

        self._require_positive_int(
            payload[
                "feature_count"
            ],
            field="feature_count",
        )

        # ---------------------------------------------------------------------
        # Probability contract
        # ---------------------------------------------------------------------

        probability_method = (
            payload.get(
                "probability_method"
            )
        )

        if probability_method is not None:

            self._require_non_empty_string(
                probability_method,
                field="probability_method",
            )

        # ---------------------------------------------------------------------
        # Transformed feature contract
        # ---------------------------------------------------------------------

        transformed_feature_count = (
            payload.get(
                "transformed_feature_count"
            )
        )

        if (
            transformed_feature_count
            is not None
        ):

            transformed_feature_count = (
                self._require_positive_int(
                    transformed_feature_count,
                    field=(
                        "transformed_feature_count"
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # Review policy
        # ---------------------------------------------------------------------

        review_policy = (
            payload.get(
                "review_policy"
            )
        )

        if review_policy is not None:

            if not isinstance(
                review_policy,
                dict,
            ):

                raise FraudAPIContractError(
                    (
                        "model-info review_policy "
                        "must be an object."
                    )
                )

            policy_type = (
                review_policy.get(
                    "type"
                )
            )

            if policy_type is not None:

                self._require_non_empty_string(
                    policy_type,
                    field=(
                        "review_policy.type"
                    ),
                )

            fraction = (
                review_policy.get(
                    "fraction"
                )
            )

            if fraction is not None:

                fraction = (
                    self._finite_number(
                        fraction,
                        field=(
                            "review_policy.fraction"
                        ),
                    )
                )

                if not (
                    0
                    < fraction
                    <= 1
                ):

                    raise FraudAPIContractError(
                        (
                            "review_policy.fraction "
                            "must lie in (0, 1]."
                        )
                    )

        # ---------------------------------------------------------------------
        # Explainability capability
        # ---------------------------------------------------------------------

        explainability = (
            payload.get(
                "explainability"
            )
        )

        if explainability is not None:

            if not isinstance(
                explainability,
                dict,
            ):

                raise FraudAPIContractError(
                    (
                        "model-info explainability "
                        "must be an object."
                    )
                )

            available = (
                explainability.get(
                    "available"
                )
            )

            if not isinstance(
                available,
                bool,
            ):

                raise FraudAPIContractError(
                    (
                        "explainability.available "
                        "must be boolean."
                    )
                )

            if available:

                if (
                    explainability.get(
                        "method"
                    )
                    != self.EXPECTED_EXPLANATION_METHOD
                ):

                    raise FraudAPIContractError(
                        (
                            "Unexpected explainability "
                            "method."
                        )
                    )

                if (
                    explainability.get(
                        "output_space"
                    )
                    != self.EXPECTED_EXPLANATION_SPACE
                ):

                    raise FraudAPIContractError(
                        (
                            "Unexpected explainability "
                            "output space."
                        )
                    )

                explained_count = (
                    self._require_positive_int(
                        explainability.get(
                            "transformed_feature_count"
                        ),
                        field=(
                            "explainability."
                            "transformed_feature_count"
                        ),
                    )
                )

                if (
                    transformed_feature_count
                    is not None
                    and explained_count
                    != transformed_feature_count
                ):

                    raise FraudAPIContractError(
                        (
                            "Explainability transformed feature "
                            "count does not match model-info."
                        )
                    )

        return payload

    # =========================================================================
    # Single scoring
    # =========================================================================

    def score_claim(
        self,
        claim: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score one claim through the deployed inference pipeline.
        """

        self._validate_claim(
            claim
        )

        payload = self._request(
            "POST",
            "/score",
            json={
                "claim":
                    claim,
            },
        )

        prediction = self._require_mapping(
            payload,
            "prediction",
        )

        self._validate_prediction(
            prediction
        )

        self._validate_single_claim_identity(
            claim,
            prediction,
        )

        return payload

    # =========================================================================
    # Local TreeSHAP explainability
    # =========================================================================

    def explain_claim(
        self,
        claim: dict[str, Any],
        *,
        top_k: int = DEFAULT_EXPLANATION_TOP_K,
    ) -> dict[str, Any]:
        """
        Generate and validate one local TreeSHAP explanation.
        """

        self._validate_claim(
            claim
        )

        top_k = (
            self._require_int_in_range(
                top_k,
                field="top_k",
                minimum=1,
                maximum=(
                    self.MAX_EXPLANATION_TOP_K
                ),
            )
        )

        payload = self._request(
            "POST",
            "/explain",
            json={
                "claim":
                    claim,

                "top_k":
                    top_k,
            },
        )

        explanation = self._require_mapping(
            payload,
            "explanation",
        )

        self._validate_explanation(
            explanation
        )

        self._validate_single_claim_identity(
            claim,
            explanation,
        )

        return payload

    # =========================================================================
    # Batch scoring
    # =========================================================================

    def score_batch(
        self,
        claims: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Score a complete portfolio and validate response integrity.
        """

        self._validate_claim_batch(
            claims
        )

        payload = self._request(
            "POST",
            "/score-batch",
            json={
                "claims":
                    claims,
            },
        )

        predictions = self._require_list(
            payload,
            "predictions",
        )

        count = (
            self._require_non_negative_int(
                payload.get(
                    "count"
                ),
                field="count",
            )
        )

        if (
            count
            != len(
                predictions
            )
        ):

            raise FraudAPIContractError(
                (
                    "Batch response count does not match "
                    "the number of predictions."
                )
            )

        if (
            count
            != len(
                claims
            )
        ):

            raise FraudAPIContractError(
                (
                    "Batch response count does not match "
                    "the submitted portfolio size."
                )
            )

        normalized_predictions: list[
            dict[str, Any]
        ] = []

        for (
            index,
            prediction,
        ) in enumerate(
            predictions
        ):

            if not isinstance(
                prediction,
                dict,
            ):

                raise FraudAPIContractError(
                    (
                        f"Batch prediction at index "
                        f"{index} is not an object."
                    )
                )

            self._validate_prediction(
                prediction
            )

            normalized_predictions.append(
                prediction
            )

        self._validate_batch_identity(
            claims,
            normalized_predictions,
            allow_subset=False,
        )

        self._validate_model_consistency(
            normalized_predictions
        )

        return payload

    # =========================================================================
    # Investigation ranking
    # =========================================================================

    def top_review(
        self,
        claims: list[
            dict[str, Any]
        ],
        review_fraction: float,
    ) -> dict[str, Any]:
        """
        Rank claims and return the authoritative review population.

        The backend remains the source of truth for ranking and selection.
        The frontend validates that the returned result respects the
        declared top-fraction contract.
        """

        self._validate_claim_batch(
            claims
        )

        fraction = self._finite_number(
            review_fraction,
            field="review_fraction",
        )

        if not (
            0
            < fraction
            <= 1
        ):

            raise ValueError(
                (
                    "review_fraction must lie "
                    "in the interval (0, 1]."
                )
            )

        payload = self._request(
            "POST",
            "/top-review",
            json={
                "claims":
                    claims,

                "review_fraction":
                    fraction,
            },
        )

        predictions = self._require_list(
            payload,
            "predictions",
        )

        total_claims = (
            self._require_non_negative_int(
                payload.get(
                    "total_claims"
                ),
                field="total_claims",
            )
        )

        selected_claims = (
            self._require_non_negative_int(
                payload.get(
                    "selected_claims"
                ),
                field="selected_claims",
            )
        )

        response_fraction = (
            self._finite_number(
                payload.get(
                    "review_fraction"
                ),
                field="review_fraction",
            )
        )

        # ---------------------------------------------------------------------
        # Response envelope
        # ---------------------------------------------------------------------

        if (
            total_claims
            != len(
                claims
            )
        ):

            raise FraudAPIContractError(
                (
                    "top-review total_claims does not "
                    "match submitted portfolio size."
                )
            )

        if (
            selected_claims
            != len(
                predictions
            )
        ):

            raise FraudAPIContractError(
                (
                    "top-review selected_claims does not "
                    "match returned prediction count."
                )
            )

        if not math.isclose(
            response_fraction,
            fraction,
            rel_tol=0.0,
            abs_tol=(
                self.FLOAT_COMPARISON_TOLERANCE
            ),
        ):

            raise FraudAPIContractError(
                (
                    "top-review response fraction differs "
                    "from the requested fraction."
                )
            )

        expected_selected = min(
            len(
                claims
            ),
            max(
                1,
                math.ceil(
                    len(
                        claims
                    )
                    * fraction
                ),
            ),
        )

        if (
            selected_claims
            != expected_selected
        ):

            raise FraudAPIContractError(
                (
                    "top-review selected_claims does not "
                    "match the declared top-fraction policy."
                )
            )

        # ---------------------------------------------------------------------
        # Ranked predictions
        # ---------------------------------------------------------------------

        normalized_predictions: list[
            dict[str, Any]
        ] = []

        ranks: list[int] = []

        scores: list[float] = []

        for (
            index,
            prediction,
        ) in enumerate(
            predictions
        ):

            if not isinstance(
                prediction,
                dict,
            ):

                raise FraudAPIContractError(
                    (
                        f"Ranked prediction at index "
                        f"{index} is not an object."
                    )
                )

            self._validate_prediction(
                prediction
            )

            rank = (
                self._require_positive_int(
                    prediction.get(
                        "risk_rank"
                    ),
                    field="risk_rank",
                )
            )

            percentile = (
                self._finite_number(
                    prediction.get(
                        "risk_percentile"
                    ),
                    field="risk_percentile",
                )
            )

            if not (
                0
                < percentile
                <= 1
            ):

                raise FraudAPIContractError(
                    (
                        "risk_percentile must lie "
                        "in (0, 1]."
                    )
                )

            if (
                prediction.get(
                    "selected_for_review"
                )
                is not True
            ):

                raise FraudAPIContractError(
                    (
                        "Every /top-review prediction "
                        "must be selected_for_review=True."
                    )
                )

            returned_fraction = (
                self._finite_number(
                    prediction.get(
                        "review_fraction"
                    ),
                    field=(
                        "prediction."
                        "review_fraction"
                    ),
                )
            )

            if not math.isclose(
                returned_fraction,
                fraction,
                rel_tol=0.0,
                abs_tol=(
                    self.FLOAT_COMPARISON_TOLERANCE
                ),
            ):

                raise FraudAPIContractError(
                    (
                        "Ranked prediction review_fraction "
                        "differs from requested fraction."
                    )
                )

            score = (
                self._finite_number(
                    prediction[
                        "fraud_risk_score"
                    ],
                    field=(
                        "fraud_risk_score"
                    ),
                )
            )

            ranks.append(
                rank
            )

            scores.append(
                score
            )

            normalized_predictions.append(
                prediction
            )

        # ---------------------------------------------------------------------
        # Identity and model consistency
        # ---------------------------------------------------------------------

        self._validate_batch_identity(
            claims,
            normalized_predictions,
            allow_subset=True,
        )

        self._validate_model_consistency(
            normalized_predictions
        )

        # ---------------------------------------------------------------------
        # Ranking consistency
        # ---------------------------------------------------------------------

        expected_ranks = list(
            range(
                1,
                selected_claims + 1,
            )
        )

        if (
            ranks
            != expected_ranks
        ):

            raise FraudAPIContractError(
                (
                    "top-review risk ranks must be exactly "
                    "1, 2, ..., selected_claims."
                )
            )

        for (
            previous,
            current,
        ) in zip(
            scores,
            scores[1:],
        ):

            if (
                current
                > previous
                + self.FLOAT_COMPARISON_TOLERANCE
            ):

                raise FraudAPIContractError(
                    (
                        "top-review predictions are not "
                        "sorted by descending fraud risk."
                    )
                )

        return payload

    # =========================================================================
    # Claim validation
    # =========================================================================

    @staticmethod
    def _validate_claim(
        claim: dict[str, Any],
    ) -> None:
        """
        Validate one frontend claim payload.
        """

        if not isinstance(
            claim,
            dict,
        ):

            raise TypeError(
                "claim must be a dictionary."
            )

        if not claim:

            raise ValueError(
                "claim cannot be empty."
            )

    @classmethod
    def _validate_claim_batch(
        cls,
        claims: list[
            dict[str, Any]
        ],
    ) -> None:
        """
        Validate a complete claim portfolio before transmission.
        """

        if not isinstance(
            claims,
            list,
        ):

            raise TypeError(
                "claims must be a list."
            )

        if not claims:

            raise ValueError(
                "claims cannot be empty."
            )

        if (
            len(
                claims
            )
            > cls.MAX_BATCH_SIZE
        ):

            raise ValueError(
                (
                    "A batch cannot contain more than "
                    f"{cls.MAX_BATCH_SIZE:,} claims."
                )
            )

        invalid = [
            index
            for (
                index,
                claim,
            ) in enumerate(
                claims
            )
            if not isinstance(
                claim,
                dict,
            )
        ]

        if invalid:

            preview = ", ".join(
                str(
                    index
                )
                for index
                in invalid[:10]
            )

            raise TypeError(
                (
                    f"{len(invalid)} batch item(s) "
                    "are not claim dictionaries. "
                    f"Invalid indexes: {preview}"
                )
            )

        empty = [
            index
            for (
                index,
                claim,
            ) in enumerate(
                claims
            )
            if not claim
        ]

        if empty:

            preview = ", ".join(
                str(
                    index
                )
                for index
                in empty[:10]
            )

            raise ValueError(
                (
                    f"{len(empty)} batch claim(s) "
                    f"are empty. Indexes: {preview}"
                )
            )

    # =========================================================================
    # Claim identity validation
    # =========================================================================

    @staticmethod
    def _claim_id(
        claim: dict[str, Any],
    ) -> str | None:
        """
        Return a normalized claim identifier when available.
        """

        value = claim.get(
            "claim_id"
        )

        if value is None:
            return None

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
        )

    @classmethod
    def _validate_single_claim_identity(
        cls,
        claim: dict[str, Any],
        prediction: dict[str, Any],
    ) -> None:
        """
        Ensure a returned score/explanation belongs to the submitted claim.
        """

        submitted_id = (
            cls._claim_id(
                claim
            )
        )

        returned_id = (
            cls._claim_id(
                prediction
            )
        )

        if (
            submitted_id is not None
            and returned_id is None
        ):

            raise FraudAPIContractError(
                (
                    "API response omitted claim_id "
                    "for an identified submitted claim."
                )
            )

        if (
            submitted_id is not None
            and returned_id is not None
            and submitted_id
            != returned_id
        ):

            raise FraudAPIContractError(
                (
                    "Prediction claim_id differs from "
                    "the submitted claim_id."
                )
            )

    @classmethod
    def _validate_batch_identity(
        cls,
        claims: list[
            dict[str, Any]
        ],
        predictions: list[
            dict[str, Any]
        ],
        *,
        allow_subset: bool = False,
    ) -> None:
        """
        Validate claim identity preservation across portfolio responses.

        For /score-batch the returned claim set must exactly match the
        submitted portfolio.

        For /top-review the returned claims must form a unique subset
        of the submitted portfolio.
        """

        submitted_ids = [
            cls._claim_id(
                claim
            )
            for claim in claims
        ]

        returned_ids = [
            cls._claim_id(
                prediction
            )
            for prediction in predictions
        ]

        # Identity reconciliation is impossible when the entire source
        # portfolio intentionally omits claim_id.
        if all(
            claim_id is None
            for claim_id in submitted_ids
        ):

            return

        if any(
            claim_id is None
            for claim_id in submitted_ids
        ):

            raise ValueError(
                (
                    "Submitted portfolio contains an "
                    "inconsistent claim_id contract: "
                    "some claims expose claim_id while others do not."
                )
            )

        if (
            len(
                set(
                    submitted_ids
                )
            )
            != len(
                submitted_ids
            )
        ):

            raise ValueError(
                (
                    "Submitted portfolio contains "
                    "duplicate claim_id values."
                )
            )

        if any(
            claim_id is None
            for claim_id in returned_ids
        ):

            raise FraudAPIContractError(
                (
                    "API response omitted claim_id for "
                    "one or more predictions."
                )
            )

        if (
            len(
                set(
                    returned_ids
                )
            )
            != len(
                returned_ids
            )
        ):

            raise FraudAPIContractError(
                (
                    "API response contains duplicate "
                    "claim_id values."
                )
            )

        submitted_set = set(
            submitted_ids
        )

        returned_set = set(
            returned_ids
        )

        unexpected = (
            returned_set
            - submitted_set
        )

        if unexpected:

            examples = ", ".join(
                sorted(
                    unexpected
                )[:5]
            )

            raise FraudAPIContractError(
                (
                    "API returned claim IDs that were not "
                    "submitted. Examples: "
                    f"{examples}"
                )
            )

        if not allow_subset:

            missing = (
                submitted_set
                - returned_set
            )

            if missing:

                examples = ", ".join(
                    sorted(
                        missing
                    )[:5]
                )

                raise FraudAPIContractError(
                    (
                        "API omitted submitted claims from "
                        "the batch response. Examples: "
                        f"{examples}"
                    )
                )

    # =========================================================================
    # Model consistency
    # =========================================================================

    @staticmethod
    def _validate_model_consistency(
        predictions: list[
            dict[str, Any]
        ],
    ) -> None:
        """
        Ensure all predictions in one response use one model identity.
        """

        if not predictions:
            return

        identities = {
            (
                str(
                    prediction.get(
                        "model_name"
                    )
                ).strip(),

                str(
                    prediction.get(
                        "model_version"
                    )
                ).strip(),
            )
            for prediction
            in predictions
        }

        if (
            len(
                identities
            )
            != 1
        ):

            raise FraudAPIContractError(
                (
                    "Portfolio response contains predictions "
                    "from multiple model identities."
                )
            )

    # =========================================================================
    # Prediction validation
    # =========================================================================

    @classmethod
    def _validate_prediction(
        cls,
        prediction: dict[str, Any],
    ) -> None:
        """
        Validate the common scoring contract.
        """

        if not isinstance(
            prediction,
            dict,
        ):

            raise FraudAPIContractError(
                "Prediction must be an object."
            )

        required = {
            "fraud_risk_score",
            "model_name",
            "model_version",
        }

        missing = (
            required
            - set(
                prediction
            )
        )

        if missing:

            raise FraudAPIContractError(
                (
                    "Prediction response is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        score = (
            cls._finite_number(
                prediction[
                    "fraud_risk_score"
                ],
                field="fraud_risk_score",
            )
        )

        if not (
            0
            <= score
            <= 1
        ):

            raise FraudAPIContractError(
                (
                    "fraud_risk_score must lie "
                    "in [0, 1]."
                )
            )

        cls._require_non_empty_string(
            prediction[
                "model_name"
            ],
            field="model_name",
        )

        cls._require_non_empty_string(
            prediction[
                "model_version"
            ],
            field="model_version",
        )

        claim_id = prediction.get(
            "claim_id"
        )

        if (
            claim_id is not None
            and not str(
                claim_id
            ).strip()
        ):

            raise FraudAPIContractError(
                (
                    "claim_id cannot be an empty "
                    "string when present."
                )
            )

    # =========================================================================
    # SHAP explanation validation
    # =========================================================================

    @classmethod
    def _validate_explanation(
        cls,
        explanation: dict[str, Any],
    ) -> None:
        """
        Validate the complete local TreeSHAP contract.

        The frontend independently checks the numerical relationships
        reported by the backend instead of trusting consistency flags alone.
        """

        required = {
            "fraud_risk_score",
            "model_name",
            "model_version",
            "explanation_method",
            "explanation_space",
            "transformed_feature_count",
            "base_value",
            "shap_sum",
            "model_raw_margin",
            "reconstructed_raw_margin",
            "reconstructed_probability",
            "positive_drivers",
            "negative_drivers",
            "strongest_drivers",
            "all_contributions",
            "consistency",
        }

        missing = (
            required
            - set(
                explanation
            )
        )

        if missing:

            raise FraudAPIContractError(
                (
                    "Explanation response is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        cls._validate_prediction(
            explanation
        )

        if (
            explanation.get(
                "explanation_method"
            )
            != cls.EXPECTED_EXPLANATION_METHOD
        ):

            raise FraudAPIContractError(
                "Unexpected explanation method."
            )

        if (
            explanation.get(
                "explanation_space"
            )
            != cls.EXPECTED_EXPLANATION_SPACE
        ):

            raise FraudAPIContractError(
                (
                    "Unexpected SHAP explanation "
                    "output space."
                )
            )

        transformed_feature_count = (
            cls._require_positive_int(
                explanation[
                    "transformed_feature_count"
                ],
                field=(
                    "transformed_feature_count"
                ),
            )
        )

        base_value = (
            cls._finite_number(
                explanation[
                    "base_value"
                ],
                field="base_value",
            )
        )

        shap_sum = (
            cls._finite_number(
                explanation[
                    "shap_sum"
                ],
                field="shap_sum",
            )
        )

        model_raw_margin = (
            cls._finite_number(
                explanation[
                    "model_raw_margin"
                ],
                field="model_raw_margin",
            )
        )

        reconstructed_raw_margin = (
            cls._finite_number(
                explanation[
                    "reconstructed_raw_margin"
                ],
                field=(
                    "reconstructed_raw_margin"
                ),
            )
        )

        reconstructed_probability = (
            cls._finite_number(
                explanation[
                    "reconstructed_probability"
                ],
                field=(
                    "reconstructed_probability"
                ),
            )
        )

        if not (
            0
            <= reconstructed_probability
            <= 1
        ):

            raise FraudAPIContractError(
                (
                    "reconstructed_probability "
                    "must lie in [0, 1]."
                )
            )

        # ---------------------------------------------------------------------
        # Driver collections
        # ---------------------------------------------------------------------

        driver_fields = (
            "positive_drivers",
            "negative_drivers",
            "strongest_drivers",
            "all_contributions",
        )

        for field in driver_fields:

            drivers = explanation.get(
                field
            )

            if not isinstance(
                drivers,
                list,
            ):

                raise FraudAPIContractError(
                    (
                        f"Explanation field '{field}' "
                        "must be a list."
                    )
                )

            for (
                index,
                driver,
            ) in enumerate(
                drivers
            ):

                try:

                    cls._validate_shap_driver(
                        driver
                    )

                except FraudAPIContractError as exc:

                    raise FraudAPIContractError(
                        (
                            f"{field}[{index}]: "
                            f"{exc}"
                        )
                    ) from exc

        positive_drivers = (
            explanation[
                "positive_drivers"
            ]
        )

        negative_drivers = (
            explanation[
                "negative_drivers"
            ]
        )

        strongest_drivers = (
            explanation[
                "strongest_drivers"
            ]
        )

        all_contributions = (
            explanation[
                "all_contributions"
            ]
        )

        if (
            len(
                all_contributions
            )
            != transformed_feature_count
        ):

            raise FraudAPIContractError(
                (
                    "all_contributions count does not "
                    "match transformed_feature_count."
                )
            )

        # ---------------------------------------------------------------------
        # Driver sign semantics
        # ---------------------------------------------------------------------

        if any(
            float(
                driver[
                    "shap_value"
                ]
            )
            <= 0
            for driver
            in positive_drivers
        ):

            raise FraudAPIContractError(
                (
                    "positive_drivers contains a "
                    "non-positive SHAP contribution."
                )
            )

        if any(
            float(
                driver[
                    "shap_value"
                ]
            )
            >= 0
            for driver
            in negative_drivers
        ):

            raise FraudAPIContractError(
                (
                    "negative_drivers contains a "
                    "non-negative SHAP contribution."
                )
            )

        # ---------------------------------------------------------------------
        # Strongest-driver ordering
        # ---------------------------------------------------------------------

        strongest_absolute_values = [
            float(
                driver[
                    "absolute_shap_value"
                ]
            )
            for driver
            in strongest_drivers
        ]

        for (
            previous,
            current,
        ) in zip(
            strongest_absolute_values,
            strongest_absolute_values[
                1:
            ],
        ):

            if (
                current
                > previous
                + cls.SHAP_DRIVER_TOLERANCE
            ):

                raise FraudAPIContractError(
                    (
                        "strongest_drivers must be ordered "
                        "by descending absolute SHAP value."
                    )
                )

        # ---------------------------------------------------------------------
        # SHAP sum from complete contribution vector
        # ---------------------------------------------------------------------

        computed_shap_sum = sum(
            float(
                item[
                    "shap_value"
                ]
            )
            for item
            in all_contributions
        )

        # ---------------------------------------------------------------------
        # Consistency block
        # ---------------------------------------------------------------------

        consistency = (
            explanation.get(
                "consistency"
            )
        )

        if not isinstance(
            consistency,
            dict,
        ):

            raise FraudAPIContractError(
                (
                    "Explanation consistency "
                    "must be an object."
                )
            )

        required_consistency = {
            "shap_additivity_ok",
            "probability_consistency_ok",
            "raw_margin_absolute_error",
            "probability_absolute_error",
            "shap_tolerance",
            "probability_tolerance",
        }

        missing_consistency = (
            required_consistency
            - set(
                consistency
            )
        )

        if missing_consistency:

            raise FraudAPIContractError(
                (
                    "Explanation consistency "
                    "is missing: "
                    + ", ".join(
                        sorted(
                            missing_consistency
                        )
                    )
                )
            )

        if (
            consistency[
                "shap_additivity_ok"
            ]
            is not True
        ):

            raise FraudAPIContractError(
                (
                    "Backend SHAP explanation failed "
                    "the additivity check."
                )
            )

        if (
            consistency[
                "probability_consistency_ok"
            ]
            is not True
        ):

            raise FraudAPIContractError(
                (
                    "Backend SHAP explanation failed "
                    "the probability consistency check."
                )
            )

        raw_margin_error = (
            cls._non_negative_finite_number(
                consistency[
                    "raw_margin_absolute_error"
                ],
                field=(
                    "raw_margin_absolute_error"
                ),
            )
        )

        probability_error = (
            cls._non_negative_finite_number(
                consistency[
                    "probability_absolute_error"
                ],
                field=(
                    "probability_absolute_error"
                ),
            )
        )

        shap_tolerance = (
            cls._positive_float(
                consistency[
                    "shap_tolerance"
                ],
                field="shap_tolerance",
            )
        )

        probability_tolerance = (
            cls._positive_float(
                consistency[
                    "probability_tolerance"
                ],
                field=(
                    "probability_tolerance"
                ),
            )
        )

        # ---------------------------------------------------------------------
        # Independent mathematical consistency validation
        # ---------------------------------------------------------------------

        shap_sum_error = abs(
            computed_shap_sum
            - shap_sum
        )

        if (
            shap_sum_error
            > shap_tolerance
        ):

            raise FraudAPIContractError(
                (
                    "Reported shap_sum does not match "
                    "the sum of all SHAP contributions."
                )
            )

        reconstructed_from_payload = (
            base_value
            + shap_sum
        )

        if abs(
            reconstructed_from_payload
            - reconstructed_raw_margin
        ) > shap_tolerance:

            raise FraudAPIContractError(
                (
                    "base_value + shap_sum does not match "
                    "reconstructed_raw_margin."
                )
            )

        actual_raw_error = abs(
            reconstructed_raw_margin
            - model_raw_margin
        )

        if (
            actual_raw_error
            > shap_tolerance
        ):

            raise FraudAPIContractError(
                (
                    "SHAP reconstructed raw margin differs "
                    "from model raw margin beyond tolerance."
                )
            )

        if not math.isclose(
            actual_raw_error,
            raw_margin_error,
            rel_tol=0.0,
            abs_tol=max(
                shap_tolerance,
                cls.FLOAT_COMPARISON_TOLERANCE,
            ),
        ):

            raise FraudAPIContractError(
                (
                    "Reported raw-margin error is inconsistent "
                    "with explanation values."
                )
            )

        fraud_score = (
            cls._finite_number(
                explanation[
                    "fraud_risk_score"
                ],
                field="fraud_risk_score",
            )
        )

        actual_probability_error = abs(
            reconstructed_probability
            - fraud_score
        )

        if (
            actual_probability_error
            > probability_tolerance
        ):

            raise FraudAPIContractError(
                (
                    "SHAP reconstructed probability differs "
                    "from fraud_risk_score beyond tolerance."
                )
            )

        if not math.isclose(
            actual_probability_error,
            probability_error,
            rel_tol=0.0,
            abs_tol=max(
                probability_tolerance,
                cls.FLOAT_COMPARISON_TOLERANCE,
            ),
        ):

            raise FraudAPIContractError(
                (
                    "Reported probability error is inconsistent "
                    "with explanation values."
                )
            )

    @classmethod
    def _validate_shap_driver(
        cls,
        driver: Any,
    ) -> None:
        """
        Validate one transformed-feature SHAP contribution.
        """

        if not isinstance(
            driver,
            dict,
        ):

            raise FraudAPIContractError(
                (
                    "Each SHAP driver must "
                    "be an object."
                )
            )

        required = {
            "feature",
            "feature_value",
            "shap_value",
            "absolute_shap_value",
            "direction",
        }

        missing = (
            required
            - set(
                driver
            )
        )

        if missing:

            raise FraudAPIContractError(
                (
                    "SHAP driver is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        cls._require_non_empty_string(
            driver[
                "feature"
            ],
            field="feature",
        )

        cls._finite_number(
            driver[
                "feature_value"
            ],
            field="feature_value",
        )

        shap_value = (
            cls._finite_number(
                driver[
                    "shap_value"
                ],
                field="shap_value",
            )
        )

        absolute = (
            cls._non_negative_finite_number(
                driver[
                    "absolute_shap_value"
                ],
                field=(
                    "absolute_shap_value"
                ),
            )
        )

        if not math.isclose(
            absolute,
            abs(
                shap_value
            ),
            rel_tol=0.0,
            abs_tol=(
                cls.SHAP_DRIVER_TOLERANCE
            ),
        ):

            raise FraudAPIContractError(
                (
                    "absolute_shap_value is inconsistent "
                    "with shap_value."
                )
            )

        direction = driver[
            "direction"
        ]

        if (
            direction
            not in cls.SHAP_DIRECTION_VALUES
        ):

            raise FraudAPIContractError(
                (
                    "Invalid SHAP driver direction."
                )
            )

        expected_direction = (
            "increase"
            if shap_value > 0
            else "decrease"
            if shap_value < 0
            else "neutral"
        )

        if (
            direction
            != expected_direction
        ):

            raise FraudAPIContractError(
                (
                    "SHAP driver direction is inconsistent "
                    "with shap_value."
                )
            )

    # =========================================================================
    # Numeric / primitive validators
    # =========================================================================

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        field: str,
    ) -> float:
        """
        Convert an API contract value into a finite float.
        """

        if isinstance(
            value,
            bool,
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be numeric."
                )
            )

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be numeric."
                )
            ) from exc

        if not math.isfinite(
            result
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be finite."
                )
            )

        return result

    @classmethod
    def _non_negative_finite_number(
        cls,
        value: Any,
        *,
        field: str,
    ) -> float:
        """
        Require a finite API contract number >= 0.
        """

        result = cls._finite_number(
            value,
            field=field,
        )

        if result < 0:

            raise FraudAPIContractError(
                (
                    f"{field} cannot "
                    "be negative."
                )
            )

        return result

    @staticmethod
    def _positive_float(
        value: Any,
        *,
        field: str,
    ) -> float:
        """
        Require a finite configuration float > 0.
        """

        if isinstance(
            value,
            bool,
        ):

            raise ValueError(
                (
                    f"{field} must "
                    "be numeric."
                )
            )

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise ValueError(
                (
                    f"{field} must "
                    "be numeric."
                )
            ) from exc

        if (
            not math.isfinite(
                result
            )
            or result <= 0
        ):

            raise ValueError(
                (
                    f"{field} must be a finite "
                    "number greater than zero."
                )
            )

        return result

    @staticmethod
    def _require_non_empty_string(
        value: Any,
        *,
        field: str,
    ) -> str:
        """
        Require a non-empty string representation.
        """

        if value is None:

            raise FraudAPIContractError(
                (
                    f"{field} cannot "
                    "be empty."
                )
            )

        text = str(
            value
        ).strip()

        if not text:

            raise FraudAPIContractError(
                (
                    f"{field} cannot "
                    "be empty."
                )
            )

        return text

    @staticmethod
    def _require_positive_int(
        value: Any,
        *,
        field: str,
    ) -> int:
        """
        Require an API contract integer >= 1 without silent truncation.
        """

        if isinstance(
            value,
            bool,
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be an integer."
                )
            ) from exc

        if (
            not math.isfinite(
                number
            )
            or not number.is_integer()
            or number < 1
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must be an "
                    "integer >= 1."
                )
            )

        return int(
            number
        )

    @staticmethod
    def _require_non_negative_int(
        value: Any,
        *,
        field: str,
    ) -> int:
        """
        Require an API contract integer >= 0.
        """

        if isinstance(
            value,
            bool,
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise FraudAPIContractError(
                (
                    f"{field} must "
                    "be an integer."
                )
            ) from exc

        if (
            not math.isfinite(
                number
            )
            or not number.is_integer()
            or number < 0
        ):

            raise FraudAPIContractError(
                (
                    f"{field} must be an "
                    "integer >= 0."
                )
            )

        return int(
            number
        )

    @staticmethod
    def _require_non_negative_int_input(
        value: Any,
        *,
        field: str,
    ) -> int:
        """
        Require a local configuration integer >= 0.
        """

        if isinstance(
            value,
            bool,
        ):

            raise ValueError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise ValueError(
                (
                    f"{field} must "
                    "be an integer."
                )
            ) from exc

        if (
            not math.isfinite(
                number
            )
            or not number.is_integer()
            or number < 0
        ):

            raise ValueError(
                (
                    f"{field} must be an "
                    "integer >= 0."
                )
            )

        return int(
            number
        )

    @staticmethod
    def _require_int_in_range(
        value: Any,
        *,
        field: str,
        minimum: int,
        maximum: int,
    ) -> int:
        """
        Require a local configuration integer inside an inclusive range.
        """

        if isinstance(
            value,
            bool,
        ):

            raise ValueError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:

            raise ValueError(
                (
                    f"{field} must "
                    "be an integer."
                )
            ) from exc

        if (
            not math.isfinite(
                number
            )
            or not number.is_integer()
        ):

            raise ValueError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        result = int(
            number
        )

        if not (
            minimum
            <= result
            <= maximum
        ):

            raise ValueError(
                (
                    f"{field} must be between "
                    f"{minimum} and {maximum}."
                )
            )

        return result