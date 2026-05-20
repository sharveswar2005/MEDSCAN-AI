from backend.app.core.config import settings

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "environment": settings.ENVIRONMENT}

def test_predict_endpoint_missing_file(client):
    response = client.post(f"{settings.API_V1_STR}/predict")
    # Should fail with 422 Unprocessable Entity since file is missing
    assert response.status_code == 422
