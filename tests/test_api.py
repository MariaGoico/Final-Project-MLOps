import pytest
import os
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import the app. 
# Note: This triggers the global code in api.py (loading models).
# We will mock the heavy parts in the fixtures.
from api.api import app
import api.api as api_module # Import module to patch globals

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def client():
    """Returns a FastAPI TestClient."""
    return TestClient(app)

@pytest.fixture
def mock_predictor():
    """
    Mocks the Breast Cancer Predictor.
    Injects a dummy object into api.api.predictor so we don't need real artifacts.
    """
    mock = MagicMock()
    # Mock properties
    mock.model_type = "xgboost"
    mock.threshold = 0.5
    mock.num_features = 30 # Standard breast cancer features
    
    # Mock predict method (return Benign=0, Prob=0.1)
    mock.predict_with_confidence.return_value = (0, 0.15)
    
    # Inject into the running API module
    original_predictor = api_module.predictor
    api_module.predictor = mock
    
    yield mock
    
    # Teardown: Restore original
    api_module.predictor = original_predictor

@pytest.fixture
def reset_retraining_state():
    """Resets global retraining variables before each test."""
    api_module.retraining_in_progress = False
    api_module.last_retrain_time = None
    api_module.retraining_status = {
        "status": "idle",
        "last_triggered": None,
        "error": None
    }
    yield

@pytest.fixture
def auth_headers():
    """Generates Basic Auth headers for the webhook."""
    secret = "your-webhook-secret" # Matches default in api.py
    token = base64.b64encode(f"admin:{secret}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

# ==========================================
# Tests: Basic Endpoints
# ==========================================

def test_home_endpoint(client, mock_predictor):
    """Test root endpoint returns status info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Breast Cancer Prediction API"
    assert "endpoints" in data

def test_health_check(client, mock_predictor):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    # Check if our custom metric is present
    assert "prediction_requests_total" in response.text

# ==========================================
# Tests: Prediction Logic
# ==========================================

def test_predict_success(client, mock_predictor):
    """Test valid CSV prediction."""
    # Create dummy CSV content
    csv_content = "radius_mean,texture_mean,perimeter_mean\n10.5,20.1,80.5"
    
    # Mock the predictor to allow flexible inputs (since we only send 3 cols)
    # The API code handles filling missing cols, so we just need predictor to accept it
    
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["diagnosis"] == "Benign (B)" # Based on our mock
    assert "latency_ms" in data["predictions"][0]

def test_predict_invalid_file_type(client):
    """Test uploading a non-CSV file."""
    files = {"file": ("image.png", b"fake_image_bytes", "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "must be CSV" in response.json()["detail"]

def test_predict_empty_csv(client, mock_predictor):
    """Test handling of empty CSVs or missing columns."""
    # CSV with just headers, no data
    csv_content = "col1,col2\n" 
    files = {"file": ("empty.csv", csv_content, "text/csv")}
    
    # This might pass (0 predictions) or fail depending on pandas handling.
    # Your code loops `for i, row in df.iterrows():`, so it should return success with 0 preds.
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 0

# ==========================================
# Tests: Retraining & Webhooks
# ==========================================
@patch("api.api.threading.Thread") # <--- Mock Threading
@patch("api.api.RetrainingPipeline")
def test_manual_retrain_trigger(MockPipeline, MockThread, client, reset_retraining_state):
    """Test manual retraining trigger."""
    # Setup Mock pipeline
    pipeline_instance = MockPipeline.return_value
    pipeline_instance.run.return_value = {"success": True}
    
    response = client.post("/retrain/manual")
    
    assert response.status_code == 200
    assert response.json()["status"] == "retraining_started"
    
    # Verify global state is set to 'manual_trigger'
    # Since the thread didn't actually run, the state won't change to 'github_triggered'
    assert api_module.retraining_status["status"] == "manual_trigger"
    
    # Verify thread was started
    MockThread.assert_called_once()

def test_webhook_unauthorized(client):
    """Test webhook without headers."""
    response = client.post("/webhook/retrain", json={})
    assert response.status_code == 401

@patch("api.api.RetrainingPipeline")
def test_webhook_alert_trigger(MockPipeline, client, auth_headers, reset_retraining_state):
    """Test Grafana alert payload triggers retraining."""
    # Setup Mock
    pipeline_instance = MockPipeline.return_value
    pipeline_instance.run.return_value = {"success": True}
    
    # Grafana Payload
    payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "ModelDrift"},
                "values": {"A": 0.5}
            }
        ]
    }
    
    response = client.post("/webhook/retrain", headers=auth_headers, json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "retraining_triggered"
    assert "ModelDrift" in response.json()["alert"]

def test_webhook_cooldown(client, auth_headers, reset_retraining_state):
    """Test that retraining is rejected if triggered too soon."""
    # Simulate a recent retrain
    api_module.last_retrain_time = datetime.now() - timedelta(minutes=1) # 1 min ago
    api_module.retraining_cooldown_minutes = 30
    
    payload = {"alerts": [{"status": "firing", "labels": {"alertname": "Drift"}}]}
    
    response = client.post("/webhook/retrain", headers=auth_headers, json=payload)
    
    # Should return status 200 (OK request) but with specific JSON status
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cooldown_active"
    assert "cooldown_minutes" in data
