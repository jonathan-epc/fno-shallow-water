import pytest
from fastapi.testclient import TestClient

from website.app.main import app


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
