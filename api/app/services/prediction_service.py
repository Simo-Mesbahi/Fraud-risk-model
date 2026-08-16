from __future__ import annotations

from typing import Any

import pandas as pd

from health_fraud.models.predict import FraudScorer


class PredictionService:
    """
    Application service responsible for fraud scoring.

    The service separates the HTTP layer from the ML inference layer.
    """

    def __init__(
        self,
        scorer: FraudScorer,
    ) -> None:
        self.scorer = scorer

    @staticmethod
    def _to_dataframe(
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        if not records:
            raise ValueError(
                "At least one claim is required."
            )

        dataframe = pd.DataFrame(
            records
        )

        if dataframe.empty:
            raise ValueError(
                "Unable to construct claim dataset."
            )

        return dataframe

    @staticmethod
    def _records(
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Convert DataFrame output into JSON-compatible records.
        """

        result = dataframe.copy()

        result = result.replace(
            {
                float("inf"): None,
                float("-inf"): None,
            }
        )

        result = result.astype(
            object
        ).where(
            pd.notna(result),
            None,
        )

        return result.to_dict(
            orient="records"
        )

    def model_info(
        self,
    ) -> dict[str, Any]:
        return self.scorer.model_info()

    def score_single(
        self,
        claim: dict[str, Any],
    ) -> dict[str, Any]:
        dataframe = self._to_dataframe(
            [claim]
        )

        scored = self.scorer.score(
            dataframe
        )

        return self._records(
            scored
        )[0]

    def score_batch(
        self,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        dataframe = self._to_dataframe(
            claims
        )

        scored = self.scorer.score(
            dataframe
        )

        return self._records(
            scored
        )

    def top_review(
        self,
        claims: list[dict[str, Any]],
        review_fraction: float,
    ) -> list[dict[str, Any]]:
        dataframe = self._to_dataframe(
            claims
        )

        selected = (
            self.scorer
            .select_top_fraction(
                dataframe=dataframe,
                review_fraction=review_fraction,
            )
        )

        return self._records(
            selected
        )