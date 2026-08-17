from __future__ import annotations

import os

from functools import lru_cache
from pathlib import Path
from typing import Final

from api.app.services.prediction_service import (
    PredictionService,
)

from health_fraud.models.predict import (
    FraudScorer,
)


# =============================================================================
# Project paths
# =============================================================================


PROJECT_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_ARTIFACTS_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
)


def _resolve_artifacts_root() -> Path:
    """
    Resolve the model-artifact root.

    Priority
    --------
    1. FRAUD_ARTIFACTS_ROOT environment variable.
    2. Project-local ./artifacts directory.

    This keeps local development, Codespaces and Docker deployments
    compatible without hardcoding runtime-specific filesystem paths.
    """

    configured = (
        os.getenv(
            "FRAUD_ARTIFACTS_ROOT"
        )
        or os.getenv(
            "ARTIFACTS_ROOT"
        )
    )

    if configured:

        return (
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )

    return (
        DEFAULT_ARTIFACTS_ROOT
        .resolve()
    )


ARTIFACTS_ROOT: Final[Path] = (
    _resolve_artifacts_root()
)


# =============================================================================
# Frozen artifact paths
# =============================================================================


MODEL_PATH: Final[Path] = (
    ARTIFACTS_ROOT
    / "models"
    / "health_fraud_xgboost.joblib"
)


PREPROCESSOR_PATH: Final[Path] = (
    ARTIFACTS_ROOT
    / "preprocessors"
    / "health_fraud_preprocessor.joblib"
)


METADATA_PATH: Final[Path] = (
    ARTIFACTS_ROOT
    / "metadata"
    / "health_fraud_model_metadata.json"
)


# =============================================================================
# Artifact validation
# =============================================================================


def _validate_artifact_file(
    *,
    name: str,
    path: Path,
) -> None:
    """
    Validate one required frozen-model artifact.
    """

    if not path.exists():

        raise FileNotFoundError(
            (
                f"Required {name} artifact was not found: "
                f"{path}"
            )
        )

    if not path.is_file():

        raise FileNotFoundError(
            (
                f"Required {name} artifact path is not a file: "
                f"{path}"
            )
        )

    try:

        size = (
            path.stat()
            .st_size
        )

    except OSError as exc:

        raise RuntimeError(
            (
                f"Unable to inspect {name} artifact: "
                f"{path}"
            )
        ) from exc

    if size <= 0:

        raise RuntimeError(
            (
                f"Required {name} artifact is empty: "
                f"{path}"
            )
        )


def validate_model_artifacts() -> None:
    """
    Validate all persisted artifacts required for inference.

    Validation occurs before deserialization so startup failures are
    explicit and easier to diagnose.
    """

    required = (
        (
            "model",
            MODEL_PATH,
        ),
        (
            "preprocessor",
            PREPROCESSOR_PATH,
        ),
        (
            "metadata",
            METADATA_PATH,
        ),
    )

    errors: list[str] = []

    for (
        name,
        path,
    ) in required:

        try:

            _validate_artifact_file(
                name=name,
                path=path,
            )

        except Exception as exc:

            errors.append(
                str(
                    exc
                )
            )

    if errors:

        details = (
            "\n".join(
                f"  - {message}"
                for message in errors
            )
        )

        raise RuntimeError(
            (
                "Fraud-model artifact validation failed:\n"
                f"{details}"
            )
        )


# =============================================================================
# Fraud scorer dependency
# =============================================================================


@lru_cache(
    maxsize=1
)
def get_fraud_scorer() -> FraudScorer:
    """
    Load and cache the frozen fraud model stack once per API process.

    The cached object contains:
    - frozen XGBoost model;
    - frozen preprocessing pipeline;
    - frozen metadata contract;
    - TreeSHAP explainer.

    No model fitting or preprocessing fitting occurs here.
    """

    validate_model_artifacts()

    return FraudScorer(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        metadata_path=METADATA_PATH,
    )


# =============================================================================
# Prediction service dependency
# =============================================================================


@lru_cache(
    maxsize=1
)
def get_prediction_service() -> PredictionService:
    """
    Return the process-local prediction application service.

    The service reuses the cached FraudScorer, guaranteeing that scoring,
    ranking and explainability use the same frozen model artifacts.
    """

    scorer = (
        get_fraud_scorer()
    )

    return PredictionService(
        scorer=scorer
    )


# =============================================================================
# Cache control
# =============================================================================


def clear_dependency_caches() -> None:
    """
    Clear process-local dependency caches.

    Intended primarily for tests and controlled development workflows.
    Production code should normally never call this function while the
    API is serving requests.
    """

    get_prediction_service.cache_clear()
    get_fraud_scorer.cache_clear()


# =============================================================================
# Runtime diagnostics
# =============================================================================


def dependency_info() -> dict[str, str]:
    """
    Return non-sensitive dependency-path information for diagnostics.
    """

    return {
        "artifacts_root":
            str(
                ARTIFACTS_ROOT
            ),

        "model_path":
            str(
                MODEL_PATH
            ),

        "preprocessor_path":
            str(
                PREPROCESSOR_PATH
            ),

        "metadata_path":
            str(
                METADATA_PATH
            ),
    }