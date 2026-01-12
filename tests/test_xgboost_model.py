import pytest
import os
import shutil
import numpy as np
import pandas as pd
import json
import pickle
import xgboost as xgb
from unittest.mock import patch, MagicMock
from logic.xgboost_model import XGBoostBreastCancerClassifier

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_env(tmp_path):
    """
    Sets up a safe environment for the test:
    1. Creates a temp directory.
    2. Switches cwd to it.
    3. Creates the specific folder structure your code expects.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Create the nested directory structure your new code requires
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts/xgboost", exist_ok=True)
    os.makedirs("plots/xgboost", exist_ok=True)
    
    yield tmp_path
    
    os.chdir(original_cwd)

@pytest.fixture
def mock_data_csv(mock_env):
    """Creates a dummy CSV file with valid columns."""
    # 50 samples to ensure cross-validation (5 splits) works without issues
    n_samples = 50
    data = {
        "id": range(n_samples),
        "diagnosis": ["M", "B"] * 25,
        "feature_1": np.random.rand(n_samples),
        "feature_2": np.random.rand(n_samples),
        "feature_3": np.random.rand(n_samples),
        "feature_4": np.random.rand(n_samples),
        "feature_5": np.random.rand(n_samples)
    }
    df = pd.DataFrame(data)
    
    csv_path = "data/data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path

@pytest.fixture
def classifier(mock_data_csv, tmp_path):
    """Returns a classifier instance."""
    # We pass a dummy tracking URI, but since we mock MLflow in tests, it won't matter
    return XGBoostBreastCancerClassifier(
        data_path=mock_data_csv, 
        tracking_uri=f"file://{tmp_path}/mlruns"
    )

# ==========================================
# Integration Test (The Big One)
# ==========================================

@patch("logic.xgboost_model.mlflow")
def test_full_training_pipeline(mock_mlflow, classifier):
    """
    Runs the full train_and_optimize loop with n_trials=1.
    Verifies that ALL artifacts, plots, and metrics are generated in the correct folders.
    Mocks mlflow to prevent run-state conflicts.
    """
    # Ensure active_run() returns None initially to simulate clean state
    mock_mlflow.active_run.return_value = None
    
    model, threshold = classifier.train_and_optimize(n_trials=1)
    
    # 1. Verify Return Values
    assert isinstance(model, xgb.Booster)
    assert 0.0 < threshold < 1.0
    
    # 2. Verify Artifacts (artifacts/xgboost/)
    base_art = "artifacts/xgboost"
    assert os.path.exists(f"{base_art}/model.json")
    assert os.path.exists(f"{base_art}/threshold.json")
    assert os.path.exists(f"{base_art}/preprocessor.pkl")
    assert os.path.exists(f"{base_art}/metadata.json")
    assert os.path.exists(f"{base_art}/shap_global.json")
    assert os.path.exists(f"{base_art}/validation_metrics.json")
    assert os.path.exists(f"{base_art}/feature_baseline.npz")
    assert os.path.exists("artifacts/feature_baseline.json")
    
    # 3. Verify Plots (plots/xgboost/)
    base_plot = "plots/xgboost"
    assert os.path.exists(f"{base_plot}/roc_curve.png")
    assert os.path.exists(f"{base_plot}/confusion_matrix.png")
    assert os.path.exists(f"{base_plot}/shap_summary.png")
    assert os.path.exists(f"{base_plot}/threshold_f1.png")

# ==========================================
# Unit Tests (Specific Logic)
# ==========================================

def test_save_validation_metrics_logic(classifier):
    """Test if validation metrics JSON is constructed correctly."""
    # Mock input data
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0]) # 1 False Negative
    y_proba = np.array([0.1, 0.9, 0.2, 0.4])
    X_train_dummy = np.random.rand(10, 5) # Just for shape
    classifier.n_features = 5
    classifier.best_threshold = 0.5
    
    # We don't mock mlflow here because save_validation_metrics only calls log_artifact,
    # which is less prone to state crashes, but for safety we can just mock it locally if needed.
    with patch("logic.xgboost_model.mlflow"):
        metrics = classifier.save_validation_metrics(y_test, y_pred, y_proba, X_train_dummy)
    
    # Check structure
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "model_info" in metrics
    
    # Check specific calculation (Recall: 1 TP / 2 Positives = 0.5)
    assert metrics['recall'] == 0.5
    assert metrics['confusion_matrix']['fn'] == 1
    
    # Check if file was written
    with open("artifacts/xgboost/validation_metrics.json", "r") as f:
        loaded = json.load(f)
        assert loaded['recall'] == 0.5

def test_save_feature_baseline_logic(classifier):
    """Test if feature baseline calculates mean/std and saves NPZ correctly."""
    X_train = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    
    with patch("logic.xgboost_model.mlflow"):
        classifier.save_feature_baseline(X_train)
    
    # Load the NPZ file
    data = np.load("artifacts/xgboost/feature_baseline.npz")
    
    # Verify contents
    assert "means" in data
    assert "stds" in data
    
    np.testing.assert_array_equal(data["means"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(data["stds"], np.array([0.0, 0.0]))

def test_objective_function_runs(classifier):
    """Ensure the Optuna objective function returns a valid float score."""
    import optuna
    X_train, _, _, y_train, _, _ = classifier.load_and_prepare_data()
    cv = classifier.validator.get_stratified_kfold(n_splits=2)
    
    # Manually create a trial
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    
    # Mock mlflow inside the objective function
    with patch("logic.xgboost_model.mlflow"):
        score = classifier.objective(trial, X_train, y_train, cv)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

@patch("logic.xgboost_model.mlflow")
def test_mlflow_logging(mock_mlflow, classifier):
    """Verify that MLflow runs are actually being created via calls."""
    classifier.train_and_optimize(n_trials=1)
    
    # Check if critical MLflow methods were called
    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metric.called
    assert mock_mlflow.end_run.called
