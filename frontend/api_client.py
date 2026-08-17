from __future__ import annotations

from typing import Any

import requests


class FraudAPIError(
    RuntimeError
):
    pass


class FraudAPIClient:

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
    ) -> None:
        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

        self.session = (
            requests.Session()
        )

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict[str, Any]:
        url = (
            f"{self.base_url}{path}"
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

        except requests.ConnectionError as exc:
            raise FraudAPIError(
                "The inference API is currently unavailable."
            ) from exc

        except requests.Timeout as exc:
            raise FraudAPIError(
                "The inference request timed out."
            ) from exc

        try:
            payload = response.json()

        except Exception:
            payload = {}

        if not response.ok:
            detail = payload.get(
                "detail",
                response.text,
            )

            raise FraudAPIError(
                str(detail)
            )

        return payload

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

    def score_claim(
        self,
        claim: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/score",
            json={
                "claim": claim,
            },
        )

    def score_batch(
        self,
        claims: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/score-batch",
            json={
                "claims": claims,
            },
        )

    def top_review(
        self,
        claims: list[dict[str, Any]],
        review_fraction: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/top-review",
            json={
                "claims": claims,
                "review_fraction":
                    review_fraction,
            },
        )