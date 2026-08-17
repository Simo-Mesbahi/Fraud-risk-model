from __future__ import annotations

from typing import Any

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
    """Base frontend API exception."""


class FraudAPIUnavailableError(
    FraudAPIError
):
    """Inference API cannot be reached."""


class FraudAPITimeoutError(
    FraudAPIError
):
    """Inference request exceeded configured timeout."""


class FraudAPIResponseError(
    FraudAPIError
):
    """API returned an unsuccessful or malformed response."""


# =============================================================================
# Client
# =============================================================================


class FraudAPIClient:
    """
    HTTP client for the Health Fraud Intelligence backend.

    Features:
    - persistent HTTP connection pooling
    - separate connection/read timeouts
    - safe retries for idempotent GET requests
    - structured FastAPI validation errors
    - defensive response validation
    """

    MAX_BATCH_SIZE = 10_000


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

        self.base_url = (
            normalized
        )

        self.timeout = (
            float(
                connect_timeout
            ),
            float(
                read_timeout
            ),
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
                        "health-fraud-intelligence/"
                        "5.0"
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
    # HTTP transport
    # =========================================================================


    def _configure_retries(
        self,
        retry_total: int,
    ) -> None:

        retry_total = max(
            0,
            int(
                retry_total
            ),
        )

        retry = Retry(
            total=retry_total,

            connect=retry_total,

            read=0,

            status=retry_total,

            backoff_factor=0.25,

            status_forcelist=(
                429,
                502,
                503,
                504,
            ),

            # Automatic retries only for idempotent requests.
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
    # Error parsing
    # =========================================================================


    @staticmethod
    def _extract_error_detail(
        response: requests.Response,
    ) -> str:

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
                or (
                    f"HTTP "
                    f"{response.status_code}"
                )
            )

        if not isinstance(
            payload,
            dict,
        ):

            return str(
                payload
            )

        detail = (
            payload.get(
                "detail"
            )
        )

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

                location = (
                    item.get(
                        "loc",
                        []
                    )
                )

                message = (
                    item.get(
                        "msg",
                        "Validation error",
                    )
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

            return (
                "; ".join(
                    messages
                )
            )

        if detail is not None:

            return str(
                detail
            )

        message = (
            payload.get(
                "message"
            )
        )

        if message is not None:

            return str(
                message
            )

        return str(
            payload
        )


    # =========================================================================
    # Core request
    # =========================================================================


    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:

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

        try:

            response = (
                self.session.request(
                    method=method.upper(),
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
                    "the response timeout."
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
                    "failure: "
                    f"{exc}"
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
    # Health / contract
    # =========================================================================


    def health(
        self,
    ) -> dict[str, Any]:

        return self._request(
            "GET",
            "/health",
        )


    def model_info(
        self,
    ) -> dict[str, Any]:

        return self._request(
            "GET",
            "/model-info",
        )


    # =========================================================================
    # Single scoring
    # =========================================================================


    def score_claim(
        self,
        claim: dict[str, Any],
    ) -> dict[str, Any]:

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

        return self._request(
            "POST",
            "/score",
            json={
                "claim":
                    claim,
            },
        )


    # =========================================================================
    # Batch scoring
    # =========================================================================


    def score_batch(
        self,
        claims: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        self._validate_claim_batch(
            claims
        )

        return self._request(
            "POST",
            "/score-batch",
            json={
                "claims":
                    claims,
            },
        )


    # =========================================================================
    # Review ranking
    # =========================================================================


    def top_review(
        self,
        claims: list[
            dict[str, Any]
        ],
        review_fraction: float,
    ) -> dict[str, Any]:

        self._validate_claim_batch(
            claims
        )

        fraction = float(
            review_fraction
        )

        if not (
            0
            < fraction
            <= 1
        ):

            raise ValueError(
                (
                    "review_fraction must be "
                    "greater than 0 and at most 1."
                )
            )

        return self._request(
            "POST",
            "/top-review",
            json={
                "claims":
                    claims,

                "review_fraction":
                    fraction,
            },
        )


    # =========================================================================
    # Validation
    # =========================================================================


    @classmethod
    def _validate_claim_batch(
        cls,
        claims: list[
            dict[str, Any]
        ],
    ) -> None:

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

            raise TypeError(
                (
                    f"{len(invalid)} batch item(s) "
                    "are not claim dictionaries."
                )
            )