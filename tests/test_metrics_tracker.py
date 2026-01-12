import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from api.metrics_tracker import (
    ModelMetricsTracker,
    model_f1_score_validation,
    data_drift_detected,
    data_drift_score,
    production_malignant_rate
)

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def tracker():
    """Returns a fresh ModelMetricsTracker instance."""
    # Disable simulation to test real logic
    return ModelMetricsTracker(window_size=100, enable_simulation=False)

def get_metric_value(metric):
    """Helper to read the current value of a Prometheus Gauge/Counter."""
    # .collect() returns a list of Metric samples
    # samples[0] contains the (name, labels, value) tuple
    return metric.collect()[0].samples[0].value

# ==========================================
# Tests: Validation Metrics
# ==========================================

def test_load_validation_metrics(tracker):
    """Test if validation metrics update Prometheus gauges."""
    metrics = {
        'f1_score': 0.85,
        'accuracy': 0.90,
        'roc_auc': 0.95,
        'confusion_matrix': {'tp': 10, 'tn': 10, 'fp': 1, 'fn': 1},
        'model_info': {'version': '1.0.0'}
    }
    
    tracker.load_validation_metrics(metrics)
    
    # Check Prometheus Gauge Value
    assert get_metric_value(model_f1_score_validation) == 0.85

def test_load_metrics_handles_missing_keys(tracker):
    """Test robustness against partial metrics dictionaries."""
    metrics = {'f1_score': 0.88} # Missing other keys
    tracker.load_validation_metrics(metrics)
    assert get_metric_value(model_f1_score_validation) == 0.88

# ==========================================
# Tests: Production Tracking
# ==========================================

def test_add_prediction(tracker):
    """Test adding predictions updates internal counters."""
    # Add 1 Malignant, 1 Benign
    tracker.add_prediction(pred=1, prob=0.9, features=[0.5])
    tracker.add_prediction(pred=0, prob=0.2, features=[0.1])
    
    assert len(tracker.predictions) == 2
    assert tracker.malignant_count == 1
    assert tracker.benign_count == 1
    
    # Check stats dict
    stats = tracker.get_stats()
    assert stats['total_predictions'] == 2
    assert stats['malignant_rate'] == 50.0

def test_production_metrics_calculation(tracker):
    """Test calculation of production rates (malignant vs benign)."""
    # Fill with 10 predictions (required minimum for calculation)
    for _ in range(5):
        tracker.add_prediction(pred=1, prob=0.9)
    for _ in range(5):
        tracker.add_prediction(pred=0, prob=0.1)
        
    tracker.calculate_metrics()
    
    # Check Prometheus Gauge
    # 5/10 = 50%
    assert get_metric_value(production_malignant_rate) == 50.0

# ==========================================
# Tests: Real Drift Detection Logic
# ==========================================

def test_data_drift_detection_no_baseline(tracker):
    """Should return False if baseline is not set."""
    drifted, score = tracker.detect_data_drift()
    assert drifted is False
    assert score == 0.0

def test_data_drift_detection_trigger(tracker):
    """Test that significantly different data triggers drift."""
    # 1. Set Baseline (Mean=0, Std=1) for 5 features
    # We need 5 features because the threshold is >= 3 drifted features
    tracker.set_feature_baseline(
        means=[0.0] * 5,
        stds=[1.0] * 5
    )
    
    # 2. Add Normal Data (Z-score near 0) - Need 30+ samples
    for _ in range(30):
        tracker.add_prediction(0, 0.5, features=[0.1] * 5)
    
    drifted, _ = tracker.detect_data_drift()
    assert not drifted
    
    # 3. Add Drifted Data (Mean=10 -> Z-score=10, huge drift)
    # The logic looks at the last 100 samples. 
    # Adding 30 drifted samples should skew the mean enough.
    for _ in range(30):
        tracker.add_prediction(0, 0.5, features=[10.0] * 5)
        
    drifted, score = tracker.detect_data_drift()
    
    # Should flag drift (Implicit bool check for numpy compatibility)
    assert drifted
    # Check Prometheus update
    assert get_metric_value(data_drift_detected) == 1.0
    assert get_metric_value(data_drift_score) > 0.0

def test_concept_drift_trigger(tracker):
    """Test concept drift (similar inputs, different outputs)."""
    # Use identical features so distance is 0.0 (always triggers pair check)
    feature_vector = [1.0, 1.0]
    
    # Strategy: Interleave 0s and 1s.
    # The logic compares neighbors. If we look like: 0, 1, 0, 1, 0, 1...
    # Every neighbor is different, leading to a high Flip Rate.
    for i in range(100):
        # Alternate labels: 0, 1, 0, 1...
        label = i % 2
        prob = 0.9 if label == 1 else 0.1
        tracker.add_prediction(pred=label, prob=prob, features=feature_vector)
    
    drifted, score = tracker.detect_concept_drift()
    
    # Debug print if it fails
    print(f"Drift Score: {score}") 
    
    # The flip rate should be extremely high (~50%)
    assert drifted
    assert score > 0.15 # Default threshold

def test_fairness_issue_trigger(tracker):
    """Test fairness detection (confidence gap)."""
    # Need 30 samples min per class
    
    # Class 0: High Confidence (0.9)
    for _ in range(30):
        tracker.add_prediction(0, 0.9, [0])
        
    # Class 1: Low Confidence (0.55) -> Gap = 0.35
    for _ in range(30):
        tracker.add_prediction(1, 0.55, [0])
        
    issue, details = tracker.detect_fairness_issues()
    
    # Implicit bool check (Numpy returns np.bool_)
    assert issue
    assert details['confidence_gap'] > 0.10 # Default threshold

# ==========================================
# Tests: Simulation Mode
# ==========================================

@patch("time.time")
def test_simulation_mode_data_drift(mock_time, tracker):
    """Test that simulation mode forces drift flags based on time."""
    tracker.enable_simulation = True
    
    # Time window 100s -> Inside Data Drift Window (60-300s)
    mock_time.return_value = 100.0
    
    tracker.calculate_metrics()
    
    # Check if flag was forced to 1
    assert get_metric_value(data_drift_detected) == 1.0

@patch("time.time")
def test_simulation_mode_normal(mock_time, tracker):
    """Test simulation normal state."""
    tracker.enable_simulation = True
    
    # Time window 0s -> Normal State
    mock_time.return_value = 0.0
    
    tracker.calculate_metrics()
    
    assert get_metric_value(data_drift_detected) == 0.0
