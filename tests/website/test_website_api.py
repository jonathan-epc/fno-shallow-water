from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from website.app.main import app, model_handler

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Create a FastAPI test client instance."""
    return TestClient(app)


def test_website_root_page(client: TestClient):
    """Verify GET / returns 200 and loads HTML template."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert (
        "<title>" in response.text
        or "FNO" in response.text
        or "html" in response.text.lower()
    )


def test_website_models_endpoint(client: TestClient):
    """Verify GET /models returns JSON dictionary of available models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_website_predict_nonexistent_model(client: TestClient):
    """Verify POST /predict/{nonexistent} with valid schema returns 404 error."""
    payload = {
        "scalar_features": [0.2, 0.01, 0.03, 0.001],
        "field_data_flat": [0.0] * 4411,
    }
    response = client.post("/predict/nonexistent_model", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_website_predict_success(client: TestClient):
    """Verify POST /predict/{model_key} returns 200 and valid ModelOutput schema when prediction succeeds."""
    mock_prediction = {
        "scalar_predictions": [0.15, 0.01],
        "field_predictions_info": "Predicted H, U, V",
        "field_predictions_shape": [3, 11, 401],
        "field_predictions_flat": {
            "H": [0.15] * 4411,
            "U": [0.45] * 4411,
            "V": [0.0] * 4411,
        },
    }

    payload = {
        "scalar_features": [0.2, 0.01, 0.03, 0.001],
        "field_data_flat": [0.0] * 4411,
    }

    with patch.object(model_handler, "predict", return_value=mock_prediction):
        response = client.post("/predict/test_model_key", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["scalar_predictions"] == [0.15, 0.01]
        assert data["field_predictions_info"] == "Predicted H, U, V"
        assert "H" in data["field_predictions_flat"]
        assert len(data["field_predictions_flat"]["H"]) == 4411
