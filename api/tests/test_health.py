from fastapi.testclient import TestClient

from api.app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_name"] == "XGBoost"
    assert payload["model_version"] == "1.0.0"