from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from health_fraud.models.predict import (
    FraudScorer,
)


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_EXPLANATION_TOP_K = 8

MAX_EXPLANATION_TOP_K = 50


# =============================================================================
# Prediction service
# =============================================================================


class PredictionService:
    """
    Application service responsible for fraud inference.

    This layer separates the HTTP/API contract from the underlying
    machine-learning implementation.

    Responsibilities
    ----------------
    - Validate service-level claim payloads.
    - Convert JSON-compatible records to pandas DataFrames.
    - Execute single-claim scoring.
    - Execute batch scoring.
    - Rank and select claims for investigation.
    - Generate local TreeSHAP explanations.
    - Enforce model/explanation consistency before returning results.
    - Convert pandas / NumPy outputs into JSON-safe Python values.

    Important
    ---------
    This service is inference-only.

    It does not:
    - train or refit the model;
    - mutate the persisted preprocessor;
    - modify the frozen feature contract;
    - tune model thresholds;
    - decide whether fraud occurred.
    """

    def __init__(
        self,
        scorer: FraudScorer,
    ) -> None:

        if not isinstance(
            scorer,
            FraudScorer,
        ):
            raise TypeError(
                (
                    "scorer must be an instance "
                    "of FraudScorer."
                )
            )

        self.scorer = scorer


    # =========================================================================
    # Input validation
    # =========================================================================


    @staticmethod
    def _validate_claim(
        claim: dict[str, Any],
    ) -> None:
        """
        Validate one service-level claim payload.
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
    def _validate_claims(
        cls,
        claims: list[
            dict[str, Any]
        ],
    ) -> None:
        """
        Validate a collection of claim records.
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
                "At least one claim is required."
            )

        for (
            index,
            claim,
        ) in enumerate(
            claims
        ):

            try:

                cls._validate_claim(
                    claim
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise type(
                    exc
                )(
                    (
                        f"Invalid claim at index "
                        f"{index}: {exc}"
                    )
                ) from exc


    # =========================================================================
    # DataFrame conversion
    # =========================================================================


    @classmethod
    def _to_dataframe(
        cls,
        records: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        """
        Convert claim records into a validated pandas DataFrame.
        """

        cls._validate_claims(
            records
        )

        dataframe = pd.DataFrame(
            records
        )

        if dataframe.empty:
            raise ValueError(
                "Unable to construct claim dataset."
            )

        if not dataframe.columns.is_unique:

            duplicated = (
                dataframe.columns[
                    dataframe.columns
                    .duplicated()
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                (
                    "Duplicate claim fields detected: "
                    + ", ".join(
                        duplicated
                    )
                )
            )

        return dataframe


    # =========================================================================
    # JSON-safe serialization
    # =========================================================================


    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:
        """
        Recursively convert model output into JSON-safe Python values.

        Handles:
        - dictionaries;
        - lists / tuples;
        - NumPy scalar values;
        - pandas timestamps;
        - pandas timedeltas;
        - NaN;
        - positive/negative infinity.
        """

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
            ),
        ):

            return [
                cls._json_safe(
                    item
                )

                for item
                in value
            ]

        if isinstance(
            value,
            pd.Timestamp,
        ):

            return value.isoformat()

        if isinstance(
            value,
            pd.Timedelta,
        ):

            return value.isoformat()

        if isinstance(
            value,
            np.ndarray,
        ):

            return [
                cls._json_safe(
                    item
                )

                for item
                in value.tolist()
            ]

        if isinstance(
            value,
            np.generic,
        ):

            return cls._json_safe(
                value.item()
            )

        if isinstance(
            value,
            float,
        ):

            if not np.isfinite(
                value
            ):

                return None

            return float(
                value
            )

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

        return value


    @classmethod
    def _records(
        cls,
        dataframe: pd.DataFrame,
    ) -> list[
        dict[str, Any]
    ]:
        """
        Convert DataFrame output into JSON-compatible records.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                (
                    "Expected a pandas DataFrame "
                    "for record serialization."
                )
            )

        records = (
            dataframe.to_dict(
                orient="records"
            )
        )

        return [
            cls._json_safe(
                record
            )

            for record
            in records
        ]


    # =========================================================================
    # Model contract
    # =========================================================================


    def model_info(
        self,
    ) -> dict[str, Any]:
        """
        Return deployed model information and capabilities.
        """

        info = (
            self.scorer.model_info()
        )

        if not isinstance(
            info,
            dict,
        ):
            raise RuntimeError(
                (
                    "FraudScorer.model_info() returned "
                    "an invalid response."
                )
            )

        result = (
            self._json_safe(
                info
            )
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
                result
            )
        )

        if missing:

            raise RuntimeError(
                (
                    "Model information is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        return result


    # =========================================================================
    # Single scoring
    # =========================================================================


    def score_single(
        self,
        claim: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score exactly one health-insurance claim.
        """

        self._validate_claim(
            claim
        )

        dataframe = (
            self._to_dataframe(
                [
                    claim
                ]
            )
        )

        scored = (
            self.scorer.score(
                dataframe
            )
        )

        records = (
            self._records(
                scored
            )
        )

        if len(
            records
        ) != 1:

            raise RuntimeError(
                (
                    "Single-claim scoring returned "
                    "an invalid number of predictions."
                )
            )

        prediction = (
            records[
                0
            ]
        )

        self._validate_score_result(
            prediction
        )

        return prediction


    # =========================================================================
    # Batch scoring
    # =========================================================================


    def score_batch(
        self,
        claims: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:
        """
        Score multiple claims while preserving submitted row order.
        """

        dataframe = (
            self._to_dataframe(
                claims
            )
        )

        scored = (
            self.scorer.score(
                dataframe
            )
        )

        records = (
            self._records(
                scored
            )
        )

        if (
            len(
                records
            )
            != len(
                claims
            )
        ):

            raise RuntimeError(
                (
                    "Batch prediction count does not match "
                    "the number of submitted claims."
                )
            )

        for (
            index,
            prediction,
        ) in enumerate(
            records
        ):

            try:

                self._validate_score_result(
                    prediction
                )

            except RuntimeError as exc:

                raise RuntimeError(
                    (
                        f"Invalid prediction at batch "
                        f"index {index}: {exc}"
                    )
                ) from exc

        return records


    # =========================================================================
    # Investigation prioritization
    # =========================================================================


    def top_review(
        self,
        claims: list[
            dict[str, Any]
        ],
        review_fraction: float,
    ) -> list[
        dict[str, Any]
    ]:
        """
        Rank claims and return the highest-risk review population.

        review_fraction expresses investigation capacity, not an
        automatic fraud classification threshold.
        """

        dataframe = (
            self._to_dataframe(
                claims
            )
        )

        try:

            review_fraction = float(
                review_fraction
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                (
                    "review_fraction must "
                    "be numeric."
                )
            ) from exc

        if not (
            0
            < review_fraction
            <= 1
        ):

            raise ValueError(
                (
                    "review_fraction must lie "
                    "in the interval (0, 1]."
                )
            )

        selected = (
            self.scorer
            .select_top_fraction(
                dataframe=dataframe,
                review_fraction=(
                    review_fraction
                ),
            )
        )

        records = (
            self._records(
                selected
            )
        )

        if not records:

            raise RuntimeError(
                (
                    "Investigation ranking returned "
                    "an empty selection."
                )
            )

        expected_maximum = (
            len(
                claims
            )
        )

        if len(
            records
        ) > expected_maximum:

            raise RuntimeError(
                (
                    "Investigation ranking returned more "
                    "claims than were submitted."
                )
            )

        for (
            index,
            prediction,
        ) in enumerate(
            records
        ):

            try:

                self._validate_ranked_result(
                    prediction
                )

            except RuntimeError as exc:

                raise RuntimeError(
                    (
                        f"Invalid ranked prediction at "
                        f"index {index}: {exc}"
                    )
                ) from exc

        return records


    # =========================================================================
    # Local explainability
    # =========================================================================


    def explain_single(
        self,
        claim: dict[str, Any],
        *,
        top_k: int = (
            DEFAULT_EXPLANATION_TOP_K
        ),
    ) -> dict[str, Any]:
        """
        Generate one local TreeSHAP explanation.

        The explanation uses exactly the same:
        - source claim;
        - feature-engineering pipeline;
        - frozen preprocessing;
        - XGBoost model;

        as the standard scoring endpoint.

        The scorer verifies:

            base_value + sum(SHAP)
                ~= model raw margin

        and:

            sigmoid(reconstructed margin)
                ~= fraud-risk probability

        before the result reaches this service.
        """

        self._validate_claim(
            claim
        )

        try:

            top_k = int(
                top_k
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise TypeError(
                "top_k must be an integer."
            ) from exc

        if not (
            1
            <= top_k
            <= MAX_EXPLANATION_TOP_K
        ):

            raise ValueError(
                (
                    "top_k must be between 1 and "
                    f"{MAX_EXPLANATION_TOP_K}."
                )
            )

        dataframe = (
            self._to_dataframe(
                [
                    claim
                ]
            )
        )

        explanation = (
            self.scorer
            .explain_single(
                dataframe,
                top_k=top_k,
            )
        )

        if not isinstance(
            explanation,
            dict,
        ):

            raise RuntimeError(
                (
                    "FraudScorer.explain_single() "
                    "returned an invalid response."
                )
            )

        result = (
            self._json_safe(
                explanation
            )
        )

        self._validate_explanation_result(
            result
        )

        return result


    # =========================================================================
    # Result validation
    # =========================================================================


    @staticmethod
    def _validate_score_result(
        prediction: dict[str, Any],
    ) -> None:
        """
        Validate one common scoring result.
        """

        if not isinstance(
            prediction,
            dict,
        ):

            raise RuntimeError(
                "Prediction must be a dictionary."
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

            raise RuntimeError(
                (
                    "Prediction result is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        try:

            score = float(
                prediction[
                    "fraud_risk_score"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                (
                    "fraud_risk_score must "
                    "be numeric."
                )
            ) from exc

        if not np.isfinite(
            score
        ):

            raise RuntimeError(
                (
                    "fraud_risk_score must "
                    "be finite."
                )
            )

        if not (
            0
            <= score
            <= 1
        ):

            raise RuntimeError(
                (
                    "fraud_risk_score must lie "
                    "in [0, 1]."
                )
            )

        if not str(
            prediction[
                "model_name"
            ]
        ).strip():

            raise RuntimeError(
                "model_name cannot be empty."
            )

        if not str(
            prediction[
                "model_version"
            ]
        ).strip():

            raise RuntimeError(
                "model_version cannot be empty."
            )


    @classmethod
    def _validate_ranked_result(
        cls,
        prediction: dict[str, Any],
    ) -> None:
        """
        Validate one investigation-ranking result.
        """

        cls._validate_score_result(
            prediction
        )

        required = {
            "risk_rank",
            "risk_percentile",
            "review_fraction",
            "selected_for_review",
        }

        missing = (
            required
            - set(
                prediction
            )
        )

        if missing:

            raise RuntimeError(
                (
                    "Ranked result is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        try:

            rank = int(
                prediction[
                    "risk_rank"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                "risk_rank must be an integer."
            ) from exc

        if rank < 1:

            raise RuntimeError(
                "risk_rank must be at least 1."
            )

        try:

            percentile = float(
                prediction[
                    "risk_percentile"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                (
                    "risk_percentile must "
                    "be numeric."
                )
            ) from exc

        if not np.isfinite(
            percentile
        ):

            raise RuntimeError(
                (
                    "risk_percentile must "
                    "be finite."
                )
            )

        if not (
            0
            < percentile
            <= 1
        ):

            raise RuntimeError(
                (
                    "risk_percentile must lie "
                    "in (0, 1]."
                )
            )

        try:

            fraction = float(
                prediction[
                    "review_fraction"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                (
                    "review_fraction must "
                    "be numeric."
                )
            ) from exc

        if not (
            0
            < fraction
            <= 1
        ):

            raise RuntimeError(
                (
                    "review_fraction must lie "
                    "in (0, 1]."
                )
            )

        if (
            prediction[
                "selected_for_review"
            ]
            is not True
        ):

            raise RuntimeError(
                (
                    "A top-review result must have "
                    "selected_for_review=True."
                )
            )


    # =========================================================================
    # Explainability validation
    # =========================================================================


    @classmethod
    def _validate_explanation_result(
        cls,
        explanation: dict[str, Any],
    ) -> None:
        """
        Validate one local SHAP explanation before returning it to HTTP.

        This is intentionally stricter than normal scoring because a
        mathematically inconsistent explanation must never be presented
        as valid model evidence.
        """

        if not isinstance(
            explanation,
            dict,
        ):

            raise RuntimeError(
                (
                    "Explanation result must "
                    "be a dictionary."
                )
            )

        cls._validate_score_result(
            explanation
        )

        required = {
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

            raise RuntimeError(
                (
                    "Explanation result is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        if (
            explanation[
                "explanation_method"
            ]
            != "TreeSHAP"
        ):

            raise RuntimeError(
                (
                    "Unexpected explanation method."
                )
            )

        if (
            explanation[
                "explanation_space"
            ]
            != "raw_margin_log_odds"
        ):

            raise RuntimeError(
                (
                    "Unexpected explanation space."
                )
            )

        feature_count = (
            cls._finite_integer(
                explanation[
                    "transformed_feature_count"
                ],
                field=(
                    "transformed_feature_count"
                ),
            )
        )

        if feature_count < 1:

            raise RuntimeError(
                (
                    "transformed_feature_count "
                    "must be at least 1."
                )
            )

        for field in (
            "base_value",
            "shap_sum",
            "model_raw_margin",
            "reconstructed_raw_margin",
            "reconstructed_probability",
        ):

            cls._finite_number(
                explanation[
                    field
                ],
                field=field,
            )

        reconstructed_probability = float(
            explanation[
                "reconstructed_probability"
            ]
        )

        if not (
            0
            <= reconstructed_probability
            <= 1
        ):

            raise RuntimeError(
                (
                    "reconstructed_probability "
                    "must lie in [0, 1]."
                )
            )

        all_contributions = (
            explanation[
                "all_contributions"
            ]
        )

        if not isinstance(
            all_contributions,
            list,
        ):

            raise RuntimeError(
                (
                    "all_contributions must "
                    "be a list."
                )
            )

        if (
            len(
                all_contributions
            )
            != feature_count
        ):

            raise RuntimeError(
                (
                    "all_contributions count does not match "
                    "transformed_feature_count."
                )
            )

        for field in (
            "positive_drivers",
            "negative_drivers",
            "strongest_drivers",
            "all_contributions",
        ):

            drivers = (
                explanation[
                    field
                ]
            )

            if not isinstance(
                drivers,
                list,
            ):

                raise RuntimeError(
                    (
                        f"{field} must "
                        "be a list."
                    )
                )

            for driver in drivers:

                cls._validate_shap_driver(
                    driver
                )

        consistency = (
            explanation[
                "consistency"
            ]
        )

        if not isinstance(
            consistency,
            dict,
        ):

            raise RuntimeError(
                (
                    "consistency must "
                    "be a dictionary."
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

            raise RuntimeError(
                (
                    "Explanation consistency is missing: "
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

            raise RuntimeError(
                (
                    "SHAP explanation failed "
                    "the raw-margin additivity check."
                )
            )

        if (
            consistency[
                "probability_consistency_ok"
            ]
            is not True
        ):

            raise RuntimeError(
                (
                    "SHAP explanation failed "
                    "the probability consistency check."
                )
            )

        for field in (
            "raw_margin_absolute_error",
            "probability_absolute_error",
            "shap_tolerance",
            "probability_tolerance",
        ):

            value = cls._finite_number(
                consistency[
                    field
                ],
                field=field,
            )

            if value < 0:

                raise RuntimeError(
                    (
                        f"{field} cannot "
                        "be negative."
                    )
                )

        fraud_score = float(
            explanation[
                "fraud_risk_score"
            ]
        )

        probability_error = abs(
            fraud_score
            - reconstructed_probability
        )

        probability_tolerance = float(
            consistency[
                "probability_tolerance"
            ]
        )

        # Additional service-level guard. FraudScorer already performs
        # this check, but the service rejects any inconsistent object
        # before it reaches the HTTP layer.
        if (
            probability_error
            > max(
                probability_tolerance,
                1e-6,
            )
        ):

            raise RuntimeError(
                (
                    "Reconstructed SHAP probability "
                    "does not match fraud_risk_score."
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

            raise RuntimeError(
                (
                    "Each SHAP driver must "
                    "be a dictionary."
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

            raise RuntimeError(
                (
                    "SHAP driver is missing: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )
            )

        feature = str(
            driver[
                "feature"
            ]
        ).strip()

        if not feature:

            raise RuntimeError(
                (
                    "SHAP feature name "
                    "cannot be empty."
                )
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
            cls._finite_number(
                driver[
                    "absolute_shap_value"
                ],
                field=(
                    "absolute_shap_value"
                ),
            )
        )

        if absolute < 0:

            raise RuntimeError(
                (
                    "absolute_shap_value "
                    "cannot be negative."
                )
            )

        if (
            abs(
                absolute
                - abs(
                    shap_value
                )
            )
            > 1e-8
        ):

            raise RuntimeError(
                (
                    "absolute_shap_value "
                    "is inconsistent with shap_value."
                )
            )

        direction = (
            driver[
                "direction"
            ]
        )

        valid_directions = {
            "increase",
            "decrease",
            "neutral",
        }

        if direction not in (
            valid_directions
        ):

            raise RuntimeError(
                (
                    "Invalid SHAP contribution "
                    "direction."
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

            raise RuntimeError(
                (
                    "SHAP contribution direction "
                    "is inconsistent with shap_value."
                )
            )


    # =========================================================================
    # Numerical validation
    # =========================================================================


    @staticmethod
    def _finite_number(
        value: Any,
        *,
        field: str,
    ) -> float:
        """
        Convert a value to a finite float.
        """

        try:

            result = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                (
                    f"{field} must "
                    "be numeric."
                )
            ) from exc

        if not np.isfinite(
            result
        ):

            raise RuntimeError(
                (
                    f"{field} must "
                    "be finite."
                )
            )

        return result


    @staticmethod
    def _finite_integer(
        value: Any,
        *,
        field: str,
    ) -> int:
        """
        Convert a value to a finite integer without silent truncation.
        """

        try:

            numeric = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise RuntimeError(
                (
                    f"{field} must "
                    "be an integer."
                )
            ) from exc

        if not np.isfinite(
            numeric
        ):

            raise RuntimeError(
                (
                    f"{field} must "
                    "be finite."
                )
            )

        if not numeric.is_integer():

            raise RuntimeError(
                (
                    f"{field} must "
                    "be an integer."
                )
            )

        return int(
            numeric
        )