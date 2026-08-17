from __future__ import annotations

import pandas as pd

from fastapi.testclient import TestClient

from api.app.main import app


client = TestClient(app)


def _serialize_claim(
    row: pd.Series,
) -> dict:
    record = {}

    for column, value in row.items():

        if pd.isna(value):
            record[column] = None

        elif isinstance(
            value,
            pd.Timestamp,
        ):
            record[column] = (
                value.isoformat()
            )

        elif hasattr(
            value,
            "item",
        ):
            try:
                record[column] = (
                    value.item()
                )

            except Exception:
                record[column] = value

        else:
            record[column] = value

    return record


def _load_claims(
    n: int = 10,
) -> list[dict]:
    claims = pd.read_parquet(
        "data/interim/claims.parquet"
    ).head(n)

    return [
        _serialize_claim(row)
        for _, row in claims.iterrows()
    ]


def test_single_score() -> None:
    claim = _load_claims(
        n=1
    )[0]

    response = client.post(
        "/score",
        json={
            "claim": claim
        },
    )

    assert response.status_code == 200

    payload = response.json()

    prediction = payload[
        "prediction"
    ]

    assert (
        prediction["claim_id"]
        == claim["claim_id"]
    )

    assert (
        0
        <= prediction[
            "fraud_risk_score"
        ]
        <= 1
    )

    assert (
        prediction["model_name"]
        == "XGBoost"
    )

    assert (
        prediction["model_version"]
        == "1.0.0"
    )


def test_batch_score() -> None:
    claims = _load_claims(
        n=10
    )

    response = client.post(
        "/score-batch",
        json={
            "claims": claims
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 10

    assert (
        len(
            payload["predictions"]
        )
        == 10
    )

    for prediction in payload[
        "predictions"
    ]:
        assert (
            0
            <= prediction[
                "fraud_risk_score"
            ]
            <= 1
        )


def test_top_review() -> None:
    claims = _load_claims(
        n=10
    )

    response = client.post(
        "/top-review",
        json={
            "claims": claims,
            "review_fraction": 0.30,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["total_claims"]
        == 10
    )

    assert (
        payload[
            "selected_claims"
        ]
        == 3
    )

    predictions = payload[
        "predictions"
    ]

    assert len(
        predictions
    ) == 3

    scores = [
        prediction[
            "fraud_risk_score"
        ]
        for prediction
        in predictions
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    for prediction in predictions:
        assert (
            prediction[
                "selected_for_review"
            ]
            is True
        )


def test_invalid_review_fraction() -> None:
    claims = _load_claims(
        n=2
    )

    response = client.post(
        "/top-review",
        json={
            "claims": claims,
            "review_fraction": 1.5,
        },
    )

    assert response.status_code == 422


def test_missing_claim_features() -> None:
    response = client.post(
        "/score",
        json={
            "claim": {
                "claim_id": "INVALID_TEST"
            }
        },
    )

    assert response.status_code == 422