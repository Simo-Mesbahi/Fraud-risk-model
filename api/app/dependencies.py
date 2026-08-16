from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from health_fraud.models.predict import FraudScorer
from api.app.services.prediction_service import PredictionService


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "health_fraud_xgboost.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessors"
    / "health_fraud_preprocessor.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
    / "health_fraud_model_metadata.json"
)


@lru_cache(maxsize=1)
def get_fraud_scorer() -> FraudScorer:
    """
    Load the frozen model artifacts once per API process.
    """

    return FraudScorer(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        metadata_path=METADATA_PATH,
    )


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """
    Return the application prediction service.
    """

    return PredictionService(
        scorer=get_fraud_scorer()
    )