from fastapi.testclient import TestClient

from api.app.main import app


client = TestClient(app)


def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_name"] == "XGBoost"
    assert payload["model_version"] == "1.0.0"
    assert payload["target"] == "is_fraud"
    assert payload["feature_count"] > 0

    assert payload["review_policy"] is not None