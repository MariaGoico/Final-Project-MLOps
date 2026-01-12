import pytest
import os
import shutil
import numpy as np
import pandas as pd
import mlflow
import json
import pickle
import torch
import onnx
from pytorch_tabnet.tab_model import TabNetClassifier
from logic.tabnet_model import TabNetBreastCancerClassifier

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_env(tmp_path):
    """
    Sets up a safe environment for the test:
    1. Creates a temp directory.
    2. Switches cwd to it.
    3. Creates necessary subdirectories (artifacts, plots).
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Create the nested directory structure
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts/tabnet", exist_ok=True)
    os.makedirs("plots/tabnet", exist_ok=True)
    
    yield tmp_path
    
    os.chdir(original_cwd)

@pytest.fixture
def mock_data_csv(mock_env):
    """Creates a dummy CSV file compatible with TabNet."""
    # 50 samples to ensure train/val/test splits work
    n_samples = 50
    data = {
        "id": range(n_samples),
        "diagnosis": ["M", "B"] * 25,
        # TabNet likes float32, but pandas reads as float64 (processor handles conversion)
        "feature_1": np.random.rand(n_samples),
        "feature_2": np.random.rand(n_samples),
        "feature_3": np.random.rand(n_samples),
        "feature_4": np.random.rand(n_samples)
    }
    df = pd.DataFrame(data)
    
    csv_path = "data/data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path

@pytest.fixture
def classifier(mock_data_csv, tmp_path):
    """Returns a classifier instance pointing to a local temp MLflow."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    return TabNetBreastCancerClassifier(
        data_path=mock_data_csv, 
        tracking_uri=tracking_uri
    )

# ==========================================
# Integration Test (Full Pipeline)
# ==========================================

def test_full_training_pipeline(classifier):
    """
    Runs the full train_and_optimize loop with n_trials=1.
    Verifies that artifacts, ONNX export, and metrics work.
    """
    # Reduce epochs for speed in testing
    # Note: We can't easily inject max_epochs into the inner loop without patching,
    # so we rely on the small dataset making it fast.
    
    model, threshold = classifier.train_and_optimize(n_trials=1)
    
    # 1. Verify Model Return
    assert isinstance(model, TabNetClassifier)
    assert 0.0 < threshold < 1.0
    
    # 2. Verify Artifacts (artifacts/tabnet/)
    base_art = "artifacts/tabnet"
    assert os.path.exists(f"{base_art}/tabnet_model.zip")  # TabNet saves as zip
    assert os.path.exists(f"{base_art}/tabnet.onnx")       # ONNX export
    assert os.path.exists(f"{base_art}/threshold.json")
    assert os.path.exists(f"{base_art}/preprocessor.pkl")
    assert os.path.exists(f"{base_art}/metadata.json")
    assert os.path.exists(f"{base_art}/feature_importance.json")
    assert os.path.exists(f"{base_art}/validation_metrics.json")
    assert os.path.exists(f"{base_art}/feature_baseline.npz")
    
    # 3. Verify Plots (plots/tabnet/)
    base_plot = "plots/tabnet"
    assert os.path.exists(f"{base_plot}/roc_curve.png")
    assert os.path.exists(f"{base_plot}/confusion_matrix.png")
    assert os.path.exists(f"{base_plot}/feature_importance.png")
    assert os.path.exists(f"{base_plot}/threshold_f1.png")

# ==========================================
# Unit Tests (Specific Components)
# ==========================================

def test_onnx_export_validity(classifier):
    """
    Test if the exported ONNX model is actually valid and runnable.
    """
    # Create a tiny dummy model for testing export logic
    X_train = np.random.rand(10, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 10)
    
    model = TabNetClassifier(n_d=8, n_a=8, n_steps=3, verbose=0)
    model.fit(X_train, y_train, max_epochs=1)
    
    # Export
    save_path = "artifacts/tabnet/test_model.onnx"
    classifier.export_to_onnx(model, X_train, save_path)
    
    # Verify File Exists
    assert os.path.exists(save_path)
    
    # Verify ONNX Structure
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    # Check Inputs/Outputs
    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    assert input_name == "input"
    assert output_name == "output"

def test_objective_function_runs(classifier):
    """Test the Optuna objective function with a single trial."""
    import optuna
    
    # Prepare data manually
    X_train, X_val, _, y_train, y_val, _ = classifier.load_and_prepare_data()
    
    # Create trial
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    
    # Run objective
    score = classifier.objective(trial, X_train, y_train, X_val, y_val)
    
    # Expect AUC score
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_feature_importance_logging(classifier):
    """Test feature importance extraction and saving."""
    # Create dummy model with known importance
    X = np.random.rand(10, 3)
    feature_names = ["f1", "f2", "f3"]
    
    # Mock a fitted model object with feature_importances_ attribute
    class MockModel:
        feature_importances_ = np.array([0.1, 0.8, 0.1])
        
    classifier.log_feature_importance(MockModel(), feature_names)
    
    # Check JSON
    json_path = "artifacts/tabnet/feature_importance.json"
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["f2"] == 0.8
        
    # Check Plot
    assert os.path.exists("plots/tabnet/feature_importance.png")

def test_save_validation_metrics_structure(classifier):
    """Ensure validation metrics JSON has correct schema."""
    y_test = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.array([0.1, 0.9])
    X_train = np.zeros((10, 5))
    classifier.n_features = 5
    
    metrics = classifier.save_validation_metrics(y_test, y_pred, y_proba, X_train)
    
    assert metrics['accuracy'] == 1.0
    assert metrics['model_info']['algorithm'] == 'TabNet'
    assert os.path.exists("artifacts/tabnet/validation_metrics.json")
