from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from health_fraud.features.build import (
    build_model_features,
)


# =============================================================================
# Constants
# =============================================================================


DEFAULT_EXPLANATION_TOP_K = 8

SHAP_ABSOLUTE_TOLERANCE = 1e-5
PROBABILITY_ABSOLUTE_TOLERANCE = 1e-6


FORBIDDEN_FEATURES = {
    "is_fraud",
    "latent_fraud_score",
    "synthetic_fraud_probability",
    "fraud_mechanism",
    "fraud_difficulty",
    "legitimate_anomaly",
    "legitimate_anomaly_type",
}


# =============================================================================
# Artifact container
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class FraudModelArtifacts:
    """
    Frozen production model artifacts.
    """

    model: Any
    preprocessor: Any
    metadata: dict[str, Any]


# =============================================================================
# Prepared model input
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PreparedModelInput:
    """
    Complete representation of a prepared inference batch.

    Prediction and explanation use this same representation to guarantee
    that both paths consume identical engineered and transformed features.
    """

    engineered: pd.DataFrame
    model_input: pd.DataFrame
    transformed: Any
    transformed_feature_names: tuple[str, ...]


# =============================================================================
# Fraud scorer
# =============================================================================


class FraudScorer:
    """
    Production-style fraud scoring and explainability wrapper.

    Responsibilities
    ----------------
    - Load frozen model artifacts.
    - Validate artifact compatibility.
    - Rebuild deterministic engineered features.
    - Validate the frozen feature contract.
    - Apply the persisted preprocessing pipeline.
    - Produce fraud-risk probabilities.
    - Rank claims by model risk.
    - Select claims according to investigation capacity.
    - Produce local TreeSHAP explanations.
    - Verify SHAP reconstruction against deployed model outputs.

    Important
    ---------
    This class performs inference only.

    It must never:
    - train or refit the model;
    - fit or mutate the preprocessor;
    - modify the frozen feature contract;
    - tune operational thresholds;
    - consume synthetic target/leakage variables.
    """

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
        metadata_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.preprocessor_path = Path(
            preprocessor_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self._validate_artifact_paths()

        self.artifacts = (
            self._load_artifacts()
        )

        self.model = (
            self.artifacts.model
        )

        self.preprocessor = (
            self.artifacts.preprocessor
        )

        self.metadata = (
            self.artifacts.metadata
        )

        self._validate_metadata()

        self.features: list[str] = list(
            self.metadata[
                "features"
            ]
        )

        self.feature_count = int(
            self.metadata[
                "feature_count"
            ]
        )

        self.target = str(
            self.metadata[
                "target"
            ]
        )

        self.model_name = str(
            self.metadata[
                "model_name"
            ]
        )

        self.model_version = str(
            self.metadata[
                "model_version"
            ]
        )

        self._validate_feature_contract()

        self.transformed_feature_names = (
            self._resolve_transformed_feature_names()
        )

        self._validate_transformed_feature_contract()

        self._shap_explainer = (
            shap.TreeExplainer(
                self.model,
                model_output="raw",
                feature_names=list(
                    self.transformed_feature_names
                ),
            )
        )

    # =========================================================================
    # Artifact loading
    # =========================================================================

    def _validate_artifact_paths(
        self,
    ) -> None:
        """
        Verify that all required persisted artifacts exist.
        """

        paths = {
            "model":
                self.model_path,

            "preprocessor":
                self.preprocessor_path,

            "metadata":
                self.metadata_path,
        }

        missing = [
            (
                name,
                path,
            )
            for (
                name,
                path,
            ) in paths.items()
            if not path.exists()
        ]

        if missing:

            formatted = "\n".join(
                f"  - {name}: {path}"
                for (
                    name,
                    path,
                ) in missing
            )

            raise FileNotFoundError(
                (
                    "Missing required model artifacts:\n"
                    + formatted
                )
            )

    def _load_artifacts(
        self,
    ) -> FraudModelArtifacts:
        """
        Load model, frozen preprocessing pipeline and metadata.
        """

        model = joblib.load(
            self.model_path
        )

        preprocessor = joblib.load(
            self.preprocessor_path
        )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "Model metadata must contain a JSON object."
            )

        return FraudModelArtifacts(
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
        )

    # =========================================================================
    # Metadata validation
    # =========================================================================

    def _validate_metadata(
        self,
    ) -> None:
        """
        Validate the minimum frozen model metadata contract.
        """

        required_keys = {
            "model_name",
            "model_version",
            "target",
            "feature_count",
            "features",
        }

        missing_keys = (
            required_keys
            - set(
                self.metadata
            )
        )

        if missing_keys:
            raise ValueError(
                (
                    "Model metadata is missing required keys: "
                    + ", ".join(
                        sorted(
                            missing_keys
                        )
                    )
                )
            )

        features = (
            self.metadata[
                "features"
            ]
        )

        if not isinstance(
            features,
            list,
        ):
            raise TypeError(
                "metadata['features'] must be a list."
            )

        if not all(
            isinstance(
                feature,
                str,
            )
            and feature.strip()
            for feature in features
        ):
            raise ValueError(
                (
                    "All frozen feature names must be "
                    "non-empty strings."
                )
            )

        feature_count = int(
            self.metadata[
                "feature_count"
            ]
        )

        if (
            len(
                features
            )
            != feature_count
        ):
            raise ValueError(
                (
                    "metadata feature_count does not match "
                    "the number of listed features."
                )
            )

        if (
            len(
                set(
                    features
                )
            )
            != len(
                features
            )
        ):
            raise ValueError(
                (
                    "Duplicate feature names detected "
                    "in metadata."
                )
            )

    def _validate_feature_contract(
        self,
    ) -> None:
        """
        Perform additional checks on the frozen feature contract.
        """

        if not self.features:
            raise ValueError(
                "The frozen feature list is empty."
            )

        leakage = (
            FORBIDDEN_FEATURES
            .intersection(
                self.features
            )
        )

        if leakage:
            raise ValueError(
                (
                    "Forbidden leakage features found "
                    "in model contract: "
                    + ", ".join(
                        sorted(
                            leakage
                        )
                    )
                )
            )

    # =========================================================================
    # Transformed feature contract
    # =========================================================================

    def _resolve_transformed_feature_names(
        self,
    ) -> tuple[str, ...]:
        """
        Resolve the feature names produced by the persisted preprocessor.
        """

        if not hasattr(
            self.preprocessor,
            "get_feature_names_out",
        ):
            raise RuntimeError(
                (
                    "The frozen preprocessor does not expose "
                    "get_feature_names_out()."
                )
            )

        try:
            names = (
                self.preprocessor
                .get_feature_names_out()
            )

        except Exception as exc:
            raise RuntimeError(
                (
                    "Unable to resolve transformed "
                    "feature names."
                )
            ) from exc

        names = np.asarray(
            names,
            dtype=object,
        )

        if names.ndim != 1:
            raise RuntimeError(
                (
                    "Expected a one-dimensional transformed "
                    "feature-name vector."
                )
            )

        normalized = tuple(
            str(
                name
            )
            for name in names.tolist()
        )

        if not normalized:
            raise RuntimeError(
                (
                    "The preprocessor produced no transformed "
                    "feature names."
                )
            )

        if (
            len(
                set(
                    normalized
                )
            )
            != len(
                normalized
            )
        ):
            raise RuntimeError(
                (
                    "Duplicate transformed feature names "
                    "were produced by the preprocessor."
                )
            )

        return normalized

    def _validate_transformed_feature_contract(
        self,
    ) -> None:
        """
        Validate preprocessor/model transformed-feature compatibility.
        """

        expected = len(
            self.transformed_feature_names
        )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if (
            model_feature_count
            is not None
            and int(
                model_feature_count
            )
            != expected
        ):
            raise ValueError(
                (
                    "Frozen model/preprocessor incompatibility: "
                    f"preprocessor produces {expected} features, "
                    f"model expects {model_feature_count}."
                )
            )

    # =========================================================================
    # Input validation
    # =========================================================================

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate generic inference input.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input DataFrame must not be empty."
            )

        if not dataframe.columns.is_unique:

            duplicated = (
                dataframe.columns[
                    dataframe.columns
                    .duplicated()
                ]
                .tolist()
            )

            raise ValueError(
                (
                    "Duplicate input columns detected: "
                    + ", ".join(
                        map(
                            str,
                            duplicated,
                        )
                    )
                )
            )

    def validate_input(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate an engineered dataframe against the frozen contract.
        """

        self._validate_dataframe(
            dataframe
        )

        missing_features = [
            feature
            for feature in self.features
            if feature not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                (
                    "Missing required model features: "
                    + ", ".join(
                        missing_features
                    )
                )
            )

    # =========================================================================
    # Feature preparation
    # =========================================================================

    def prepare_model_input(
        self,
        dataframe: pd.DataFrame,
    ) -> PreparedModelInput:
        """
        Build the exact representation consumed by the frozen model.

        Both scoring and explainability call this method so that they
        cannot silently diverge.
        """

        self._validate_dataframe(
            dataframe
        )

        enriched = (
            build_model_features(
                dataframe
            )
        )

        if not isinstance(
            enriched,
            pd.DataFrame,
        ):
            raise TypeError(
                (
                    "build_model_features() must return "
                    "a pandas DataFrame."
                )
            )

        if (
            len(
                enriched
            )
            != len(
                dataframe
            )
        ):
            raise RuntimeError(
                (
                    "Feature engineering changed "
                    "the number of rows."
                )
            )

        self.validate_input(
            enriched
        )

        model_input = (
            enriched.loc[
                :,
                self.features,
            ]
            .copy()
        )

        transformed = (
            self.preprocessor
            .transform(
                model_input
            )
        )

        if (
            transformed.shape[
                0
            ]
            != len(
                dataframe
            )
        ):
            raise RuntimeError(
                (
                    "Preprocessing changed "
                    "the number of rows."
                )
            )

        if (
            transformed.shape[
                1
            ]
            != len(
                self.transformed_feature_names
            )
        ):
            raise RuntimeError(
                (
                    "Transformed feature count does not match "
                    "get_feature_names_out()."
                )
            )

        return PreparedModelInput(
            engineered=enriched,
            model_input=model_input,
            transformed=transformed,
            transformed_feature_names=(
                self.transformed_feature_names
            ),
        )

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Backwards-compatible transformed-feature preparation.
        """

        return (
            self.prepare_model_input(
                dataframe
            )
            .transformed
        )

    # =========================================================================
    # Probability scoring
    # =========================================================================

    def _predict_from_transformed(
        self,
        transformed: Any,
    ) -> np.ndarray:
        """
        Score already-preprocessed samples.
        """

        probabilities = (
            self.model
            .predict_proba(
                transformed
            )[:, 1]
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.ndim != 1:
            raise RuntimeError(
                (
                    "Expected a one-dimensional "
                    "fraud probability vector."
                )
            )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise ValueError(
                (
                    "Model produced non-finite "
                    "probabilities."
                )
            )

        if (
            np.any(
                probabilities < 0
            )
            or np.any(
                probabilities > 1
            )
        ):
            raise ValueError(
                (
                    "Model produced probabilities "
                    "outside [0, 1]."
                )
            )

        return probabilities

    def predict_proba(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return fraud probability for each claim.
        """

        prepared = (
            self.prepare_model_input(
                dataframe
            )
        )

        probabilities = (
            self._predict_from_transformed(
                prepared.transformed
            )
        )

        if (
            len(
                probabilities
            )
            != len(
                dataframe
            )
        ):
            raise RuntimeError(
                (
                    "Prediction count does not "
                    "match input row count."
                )
            )

        return probabilities

    # =========================================================================
    # Basic scoring
    # =========================================================================

    def score(
        self,
        dataframe: pd.DataFrame,
        claim_id_column: str = "claim_id",
    ) -> pd.DataFrame:
        """
        Score claims while preserving row order.
        """

        probabilities = (
            self.predict_proba(
                dataframe
            )
        )

        result = pd.DataFrame(
            {
                "fraud_risk_score":
                    probabilities,
            }
        )

        if (
            claim_id_column
            in dataframe.columns
        ):
            result.insert(
                0,
                claim_id_column,
                dataframe[
                    claim_id_column
                ]
                .to_numpy(
                    copy=False
                ),
            )

        result[
            "model_name"
        ] = self.model_name

        result[
            "model_version"
        ] = self.model_version

        return result

    # =========================================================================
    # SHAP helpers
    # =========================================================================

    @staticmethod
    def _dense_row(
        transformed: Any,
        row_index: int,
    ) -> np.ndarray:
        """
        Return one transformed row as a dense 1-D numeric vector.
        """

        row = transformed[
            row_index
        ]

        if hasattr(
            row,
            "toarray",
        ):
            row = (
                row.toarray()
            )

        row = np.asarray(
            row,
            dtype=np.float64,
        )

        return np.ravel(
            row
        )

    @staticmethod
    def _shap_vector(
        values: Any,
    ) -> np.ndarray:
        """
        Normalize SHAP values to one feature vector.
        """

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim == 2:

            if array.shape[0] != 1:
                raise RuntimeError(
                    (
                        "Expected a SHAP explanation "
                        "for exactly one claim."
                    )
                )

            array = array[
                0
            ]

        elif array.ndim == 3:

            if array.shape[0] != 1:
                raise RuntimeError(
                    (
                        "Expected a SHAP explanation "
                        "for exactly one claim."
                    )
                )

            if array.shape[-1] == 1:
                array = array[
                    0,
                    :,
                    0,
                ]

            elif array.shape[-1] == 2:
                array = array[
                    0,
                    :,
                    1,
                ]

            else:
                raise RuntimeError(
                    (
                        "Unexpected SHAP output "
                        "dimension."
                    )
                )

        if array.ndim != 1:
            raise RuntimeError(
                (
                    "Unable to normalize SHAP values "
                    "to one feature vector."
                )
            )

        return array

    @staticmethod
    def _expected_value_scalar(
        expected_value: Any,
    ) -> float:
        """
        Normalize SHAP base value to one scalar.
        """

        values = np.asarray(
            expected_value,
            dtype=np.float64,
        )

        if values.ndim == 0:
            return float(
                values
            )

        values = values.ravel()

        if len(
            values
        ) == 1:
            return float(
                values[
                    0
                ]
            )

        if len(
            values
        ) == 2:
            return float(
                values[
                    1
                ]
            )

        raise RuntimeError(
            (
                "Unexpected TreeExplainer "
                "base-value structure."
            )
        )

    def _raw_margin(
        self,
        transformed: Any,
    ) -> np.ndarray:
        """
        Return raw XGBoost margins.
        """

        margin = (
            self.model.predict(
                transformed,
                output_margin=True,
            )
        )

        margin = np.asarray(
            margin,
            dtype=np.float64,
        )

        margin = np.ravel(
            margin
        )

        if not np.all(
            np.isfinite(
                margin
            )
        ):
            raise RuntimeError(
                (
                    "Model produced non-finite "
                    "raw margins."
                )
            )

        return margin

    @staticmethod
    def _feature_record(
        *,
        feature: str,
        transformed_value: float,
        shap_value: float,
    ) -> dict[str, Any]:
        """
        Build one JSON-friendly SHAP contribution record.
        """

        if shap_value > 0:
            direction = "increase"

        elif shap_value < 0:
            direction = "decrease"

        else:
            direction = "neutral"

        return {
            "feature":
                str(
                    feature
                ),

            "feature_value":
                float(
                    transformed_value
                ),

            "shap_value":
                float(
                    shap_value
                ),

            "absolute_shap_value":
                float(
                    abs(
                        shap_value
                    )
                ),

            "direction":
                direction,
        }

    # =========================================================================
    # Local SHAP explanation
    # =========================================================================

    def explain(
        self,
        dataframe: pd.DataFrame,
        *,
        top_k: int = DEFAULT_EXPLANATION_TOP_K,
        claim_id_column: str = "claim_id",
    ) -> list[dict[str, Any]]:
        """
        Produce local TreeSHAP explanations.

        Explanations are generated in XGBoost raw-margin/log-odds space.

        Numerical consistency is explicitly checked:

            base_value + sum(SHAP)
                ~= model raw margin

        followed by:

            sigmoid(reconstructed raw margin)
                ~= fraud probability

        This verifies that the explanation corresponds to the same frozen
        model used by the production scoring path.
        """

        self._validate_dataframe(
            dataframe
        )

        top_k = int(
            top_k
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        prepared = (
            self.prepare_model_input(
                dataframe
            )
        )

        probabilities = (
            self._predict_from_transformed(
                prepared.transformed
            )
        )

        raw_margins = (
            self._raw_margin(
                prepared.transformed
            )
        )

        if (
            len(
                raw_margins
            )
            != len(
                dataframe
            )
        ):
            raise RuntimeError(
                (
                    "Raw-margin prediction count does "
                    "not match input row count."
                )
            )

        feature_names = list(
            prepared.transformed_feature_names
        )

        explanations: list[
            dict[str, Any]
        ] = []

        for row_index in range(
            len(
                dataframe
            )
        ):

            transformed_row = (
                self._dense_row(
                    prepared.transformed,
                    row_index,
                )
            )

            if (
                len(
                    transformed_row
                )
                != len(
                    feature_names
                )
            ):
                raise RuntimeError(
                    (
                        "Transformed row width does not "
                        "match feature-name count."
                    )
                )

            shap_input = (
                transformed_row
                .reshape(
                    1,
                    -1,
                )
            )

            shap_explanation = (
                self._shap_explainer(
                    shap_input,
                    check_additivity=True,
                )
            )

            shap_values = (
                self._shap_vector(
                    shap_explanation.values
                )
            )

            if (
                len(
                    shap_values
                )
                != len(
                    feature_names
                )
            ):
                raise RuntimeError(
                    (
                        "SHAP feature count does not match "
                        "transformed model feature count."
                    )
                )

            base_value = (
                self._expected_value_scalar(
                    shap_explanation.base_values
                )
            )

            shap_sum = float(
                np.sum(
                    shap_values
                )
            )

            reconstructed_margin = float(
                base_value
                + shap_sum
            )

            model_margin = float(
                raw_margins[
                    row_index
                ]
            )

            reconstructed_probability = float(
                expit(
                    reconstructed_margin
                )
            )

            model_probability = float(
                probabilities[
                    row_index
                ]
            )

            margin_error = float(
                abs(
                    reconstructed_margin
                    - model_margin
                )
            )

            probability_error = float(
                abs(
                    reconstructed_probability
                    - model_probability
                )
            )

            margin_consistent = bool(
                margin_error
                <= SHAP_ABSOLUTE_TOLERANCE
            )

            probability_consistent = bool(
                probability_error
                <= PROBABILITY_ABSOLUTE_TOLERANCE
            )

            contributions = [
                self._feature_record(
                    feature=feature,
                    transformed_value=float(
                        transformed_row[
                            index
                        ]
                    ),
                    shap_value=float(
                        shap_values[
                            index
                        ]
                    ),
                )
                for (
                    index,
                    feature,
                ) in enumerate(
                    feature_names
                )
            ]

            positive_drivers = sorted(
                (
                    item
                    for item in contributions
                    if item[
                        "shap_value"
                    ] > 0
                ),
                key=lambda item:
                    item[
                        "shap_value"
                    ],
                reverse=True,
            )[
                :top_k
            ]

            negative_drivers = sorted(
                (
                    item
                    for item in contributions
                    if item[
                        "shap_value"
                    ] < 0
                ),
                key=lambda item:
                    item[
                        "shap_value"
                    ],
            )[
                :top_k
            ]

            strongest_drivers = sorted(
                contributions,
                key=lambda item:
                    item[
                        "absolute_shap_value"
                    ],
                reverse=True,
            )[
                :top_k
            ]

            claim_id: Any = None

            if (
                claim_id_column
                in dataframe.columns
            ):

                claim_id = (
                    dataframe.iloc[
                        row_index
                    ][
                        claim_id_column
                    ]
                )

                if pd.isna(
                    claim_id
                ):
                    claim_id = None

                elif hasattr(
                    claim_id,
                    "item",
                ):
                    try:
                        claim_id = (
                            claim_id.item()
                        )
                    except Exception:
                        pass

            explanations.append(
                {
                    "claim_id":
                        (
                            None
                            if claim_id is None
                            else str(
                                claim_id
                            )
                        ),

                    "fraud_risk_score":
                        model_probability,

                    "model_name":
                        self.model_name,

                    "model_version":
                        self.model_version,

                    "explanation_method":
                        "TreeSHAP",

                    "explanation_space":
                        "raw_margin_log_odds",

                    "transformed_feature_count":
                        len(
                            feature_names
                        ),

                    "base_value":
                        base_value,

                    "shap_sum":
                        shap_sum,

                    "model_raw_margin":
                        model_margin,

                    "reconstructed_raw_margin":
                        reconstructed_margin,

                    "reconstructed_probability":
                        reconstructed_probability,

                    "positive_drivers":
                        positive_drivers,

                    "negative_drivers":
                        negative_drivers,

                    "strongest_drivers":
                        strongest_drivers,

                    "all_contributions":
                        contributions,

                    "consistency":
                        {
                            "shap_additivity_ok":
                                margin_consistent,

                            "probability_consistency_ok":
                                probability_consistent,

                            "raw_margin_absolute_error":
                                margin_error,

                            "probability_absolute_error":
                                probability_error,

                            "shap_tolerance":
                                SHAP_ABSOLUTE_TOLERANCE,

                            "probability_tolerance":
                                PROBABILITY_ABSOLUTE_TOLERANCE,
                        },
                }
            )

        return explanations

    def explain_single(
        self,
        dataframe: pd.DataFrame,
        *,
        top_k: int = DEFAULT_EXPLANATION_TOP_K,
        claim_id_column: str = "claim_id",
    ) -> dict[str, Any]:
        """
        Explain exactly one claim.
        """

        self._validate_dataframe(
            dataframe
        )

        if (
            len(
                dataframe
            )
            != 1
        ):
            raise ValueError(
                (
                    "explain_single() requires "
                    "exactly one claim."
                )
            )

        return (
            self.explain(
                dataframe,
                top_k=top_k,
                claim_id_column=claim_id_column,
            )[
                0
            ]
        )

    # =========================================================================
    # Ranking
    # =========================================================================

    def rank(
        self,
        dataframe: pd.DataFrame,
        claim_id_column: str = "claim_id",
    ) -> pd.DataFrame:
        """
        Rank claims from highest to lowest fraud risk.

        risk_percentile uses a high-is-risky convention:
        the highest-risk claim receives 1.0.
        """

        scored = (
            self.score(
                dataframe=dataframe,
                claim_id_column=(
                    claim_id_column
                ),
            )
        )

        scored = (
            scored
            .sort_values(
                "fraud_risk_score",
                ascending=False,
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        n_rows = len(
            scored
        )

        scored[
            "risk_rank"
        ] = np.arange(
            1,
            n_rows + 1,
            dtype=np.int64,
        )

        if n_rows == 1:

            scored[
                "risk_percentile"
            ] = 1.0

        else:

            scored[
                "risk_percentile"
            ] = (
                1.0
                - (
                    (
                        scored[
                            "risk_rank"
                        ]
                        - 1
                    )
                    / n_rows
                )
            )

        return scored

    # =========================================================================
    # Investigation queue
    # =========================================================================

    def select_top_fraction(
        self,
        dataframe: pd.DataFrame,
        review_fraction: float = 0.03,
        claim_id_column: str = "claim_id",
    ) -> pd.DataFrame:
        """
        Select the highest-risk claims according to investigation capacity.

        Example
        -------
        review_fraction=0.03 selects approximately the top 3% of claims.
        """

        if not isinstance(
            review_fraction,
            (
                int,
                float,
            ),
        ):
            raise TypeError(
                "review_fraction must be numeric."
            )

        review_fraction = float(
            review_fraction
        )

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

        ranked = (
            self.rank(
                dataframe=dataframe,
                claim_id_column=(
                    claim_id_column
                ),
            )
        )

        n_selected = max(
            1,
            int(
                np.ceil(
                    len(
                        ranked
                    )
                    * review_fraction
                )
            ),
        )

        selected = (
            ranked
            .iloc[
                :n_selected
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        selected[
            "review_fraction"
        ] = review_fraction

        selected[
            "selected_for_review"
        ] = True

        return selected

    # =========================================================================
    # Model information
    # =========================================================================

    def model_info(
        self,
    ) -> dict[str, Any]:
        """
        Return model, deployment and explainability information.
        """

        return {
            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "target":
                self.target,

            "feature_count":
                self.feature_count,

            "transformed_feature_count":
                len(
                    self.transformed_feature_names
                ),

            "probability_method":
                "predict_proba",

            "explainability":
                {
                    "available":
                        True,

                    "method":
                        "TreeSHAP",

                    "output_space":
                        "raw_margin_log_odds",

                    "transformed_feature_count":
                        len(
                            self.transformed_feature_names
                        ),
                },

            "review_policy":
                self.metadata.get(
                    "review_policy"
                ),
        }