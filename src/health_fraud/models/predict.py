from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd

from health_fraud.features.build import (
    build_model_features,
)


# =============================================================================
# Artifact container
# =============================================================================


@dataclass(frozen=True)
class FraudModelArtifacts:
    """
    Container holding the frozen production model artifacts.
    """

    model: Any
    preprocessor: Any
    metadata: dict[str, Any]


# =============================================================================
# Fraud scorer
# =============================================================================


class FraudScorer:
    """
    Production-style fraud scoring wrapper.

    Responsibilities
    ----------------
    - Load the frozen model artifacts.
    - Validate artifact compatibility.
    - Rebuild deterministic engineered features.
    - Validate the frozen model feature contract.
    - Apply the frozen preprocessing pipeline.
    - Return fraud-risk probabilities.
    - Rank claims by fraud risk.
    - Select claims according to an investigation capacity.

    Important
    ---------
    This class performs inference only.

    It must not:
    - train the model;
    - modify the feature contract;
    - tune thresholds;
    - use synthetic leakage variables.
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
            "model": (
                self.model_path
            ),
            "preprocessor": (
                self.preprocessor_path
            ),
            "metadata": (
                self.metadata_path
            ),
        }

        missing = [
            (
                name,
                path,
            )
            for (
                name,
                path,
            )
            in paths.items()
            if not path.exists()
        ]

        if missing:
            formatted = "\n".join(
                f"  - {name}: {path}"
                for (
                    name,
                    path,
                )
                in missing
            )

            raise FileNotFoundError(
                "Missing required model "
                "artifacts:\n"
                + formatted
            )

    def _load_artifacts(
        self,
    ) -> FraudModelArtifacts:
        """
        Load model, preprocessing pipeline and metadata.
        """

        model = joblib.load(
            self.model_path
        )

        preprocessor = joblib.load(
            self.preprocessor_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
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
        Validate the minimum model metadata contract.
        """

        required_keys = {
            "model_name",
            "model_version",
            "target",
            "feature_count",
            "features",
        }

        metadata_keys = set(
            self.metadata
        )

        missing_keys = (
            required_keys
            - metadata_keys
        )

        if missing_keys:
            raise ValueError(
                "Model metadata is missing "
                "required keys: "
                + ", ".join(
                    sorted(
                        missing_keys
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
                "metadata['features'] "
                "must be a list."
            )

        feature_count = int(
            self.metadata[
                "feature_count"
            ]
        )

        if (
            len(features)
            != feature_count
        ):
            raise ValueError(
                "metadata feature_count "
                "does not match the number "
                "of listed features."
            )

        if (
            len(
                set(features)
            )
            != len(features)
        ):
            raise ValueError(
                "Duplicate feature names "
                "detected in metadata."
            )

    def _validate_feature_contract(
        self,
    ) -> None:
        """
        Perform additional checks on the frozen feature contract.
        """

        if not self.features:
            raise ValueError(
                "The frozen feature list "
                "is empty."
            )

        forbidden = {
            "is_fraud",
            "latent_fraud_score",
            "synthetic_fraud_probability",
            "fraud_mechanism",
            "fraud_difficulty",
            "legitimate_anomaly",
            "legitimate_anomaly_type",
        }

        leakage = (
            forbidden
            .intersection(
                self.features
            )
        )

        if leakage:
            raise ValueError(
                "Forbidden leakage features "
                "found in model contract: "
                + ", ".join(
                    sorted(
                        leakage
                    )
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
                "Input must be a "
                "pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input DataFrame "
                "must not be empty."
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
                "Duplicate input columns "
                "detected: "
                + ", ".join(
                    map(
                        str,
                        duplicated,
                    )
                )
            )

    def validate_input(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate an already feature-engineered dataframe
        against the frozen model feature contract.
        """

        self._validate_dataframe(
            dataframe
        )

        missing_features = [
            feature
            for feature
            in self.features
            if feature
            not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required "
                "model features: "
                + ", ".join(
                    missing_features
                )
            )

    # =========================================================================
    # Feature preparation
    # =========================================================================

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Build deterministic engineered features,
        validate the frozen contract and transform
        the dataframe with the persisted preprocessor.
        """

        self._validate_dataframe(
            dataframe
        )

        enriched = (
            build_model_features(
                dataframe
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
            transformed.shape[0]
            != len(dataframe)
        ):
            raise RuntimeError(
                "Preprocessing changed "
                "the number of rows."
            )

        return transformed

    # =========================================================================
    # Probability scoring
    # =========================================================================

    def predict_proba(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return fraud probability for each claim.
        """

        transformed = (
            self.prepare_features(
                dataframe
            )
        )

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

        if (
            probabilities.ndim
            != 1
        ):
            raise RuntimeError(
                "Expected a one-dimensional "
                "fraud probability vector."
            )

        if (
            len(probabilities)
            != len(dataframe)
        ):
            raise RuntimeError(
                "Prediction count does not "
                "match input row count."
            )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise ValueError(
                "Model produced "
                "non-finite probabilities."
            )

        if np.any(
            probabilities < 0
        ) or np.any(
            probabilities > 1
        ):
            raise ValueError(
                "Model produced probabilities "
                "outside [0, 1]."
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
                "fraud_risk_score": (
                    probabilities
                ),
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
    # Ranking
    # =========================================================================

    def rank(
        self,
        dataframe: pd.DataFrame,
        claim_id_column: str = "claim_id",
    ) -> pd.DataFrame:
        """
        Rank claims from highest to lowest fraud risk.
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

        scored[
            "risk_percentile"
        ] = (
            scored[
                "risk_rank"
            ]
            / n_rows
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
        Select the highest-risk claims according
        to a fixed investigation capacity.

        Example
        -------
        review_fraction=0.03 selects the top 3%
        highest-risk claims.
        """

        if not isinstance(
            review_fraction,
            (int, float),
        ):
            raise TypeError(
                "review_fraction must "
                "be numeric."
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
                "review_fraction must lie "
                "in the interval (0, 1]."
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
                    len(ranked)
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
        Return lightweight model information useful
        for APIs, health checks and logging.
        """

        return {
            "model_name": (
                self.model_name
            ),
            "model_version": (
                self.model_version
            ),
            "target": (
                self.target
            ),
            "feature_count": (
                self.feature_count
            ),
            "review_policy": (
                self.metadata.get(
                    "review_policy"
                )
            ),
        }