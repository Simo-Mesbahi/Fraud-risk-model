from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.app.dependencies import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    get_fraud_scorer,
)

from api.app.services.prediction_service import (
    PredictionService,
)

from health_fraud.models.predict import (
    FraudScorer,
)


# =============================================================================
# Project paths
# =============================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEMO_CLAIMS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "claims.parquet"
)


# =============================================================================
# Artifact fixtures
# =============================================================================


@pytest.fixture(
    scope="session",
)
def model_path() -> Path:
    """
    Return the frozen XGBoost model artifact path.
    """

    assert MODEL_PATH.exists(), (
        f"Model artifact missing: {MODEL_PATH}"
    )

    return MODEL_PATH


@pytest.fixture(
    scope="session",
)
def preprocessor_path() -> Path:
    """
    Return the persisted preprocessing artifact path.
    """

    assert PREPROCESSOR_PATH.exists(), (
        (
            "Preprocessor artifact missing: "
            f"{PREPROCESSOR_PATH}"
        )
    )

    return PREPROCESSOR_PATH


@pytest.fixture(
    scope="session",
)
def metadata_path() -> Path:
    """
    Return frozen model metadata path.
    """

    assert METADATA_PATH.exists(), (
        f"Metadata artifact missing: {METADATA_PATH}"
    )

    return METADATA_PATH


# =============================================================================
# Model fixtures
# =============================================================================


@pytest.fixture(
    scope="session",
)
def fraud_scorer() -> FraudScorer:
    """
    Load the frozen production inference stack once per test session.

    The fixture intentionally uses the same dependency factory as the API
    so tests exercise the exact deployed artifact configuration.
    """

    scorer = get_fraud_scorer()

    assert isinstance(
        scorer,
        FraudScorer,
    )

    return scorer


@pytest.fixture(
    scope="session",
)
def prediction_service(
    fraud_scorer: FraudScorer,
) -> PredictionService:
    """
    Return the application inference service.
    """

    return PredictionService(
        scorer=fraud_scorer,
    )


# =============================================================================
# Dataset fixtures
# =============================================================================


@pytest.fixture(
    scope="session",
)
def demo_claims_frame() -> pd.DataFrame:
    """
    Load a small deterministic sample from the bundled claim dataset.
    """

    assert DEMO_CLAIMS_PATH.exists(), (
        (
            "Demo claims dataset missing: "
            f"{DEMO_CLAIMS_PATH}"
        )
    )

    frame = pd.read_parquet(
        DEMO_CLAIMS_PATH
    )

    assert not frame.empty

    return (
        frame
        .head(100)
        .copy()
    )


@pytest.fixture
def single_claim(
    demo_claims_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return one source claim record.
    """

    return (
        demo_claims_frame
        .iloc[0]
        .to_dict()
    )


@pytest.fixture
def second_claim(
    demo_claims_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a second independent source claim.
    """

    return (
        demo_claims_frame
        .iloc[1]
        .to_dict()
    )


@pytest.fixture
def claim_batch(
    demo_claims_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Return a deterministic batch for inference tests.
    """

    return (
        demo_claims_frame
        .head(20)
        .to_dict(
            orient="records"
        )
    )


@pytest.fixture
def review_claim_batch(
    demo_claims_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Return exactly 100 claims for top-3%-review assertions.
    """

    return (
        demo_claims_frame
        .head(100)
        .to_dict(
            orient="records"
        )
    )