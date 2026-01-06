import pytest
import os
import shutil
import numpy as np
import pandas as pd
import mlflow
import json
import pickle
import xgboost as xgb
from logic.model import XGBoostBreastCancerClassifier

# --- Fixtures ---

@pytest.fixture
def mock_env(tmp_path):
    """
    Sets up a safe environment for the test:
    1. Creates a temp directory.
    2. Changes the current working directory to it (so hardcoded paths like 'artifacts/' go here).
    3. Returns the path.
    """
    # Save original working directory
    original_cwd = os.getcwd()
    
    # Switch to temp directory
    os.chdir(tmp_path)
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    yield tmp_path
    
    # Restore original working directory after test
    os.chdir(original_cwd)

@pytest.fixture
def mock_data_csv(mock_env):
    """Creates a dummy CSV file with valid structure in the temp dir."""
    # 20 samples, 5 features + diagnosis
    data = {
        "id": range(20),
        "diagnosis": ["M", "B"] * 10,
        "feature_1": np.random.rand(20),
        "feature_2": np.random.rand(20),
        "feature_3": np.random.rand(20),
        "feature_4": np.random.rand(20),
        "feature_5": np.random.rand(20)
    }
    df = pd.DataFrame(data)
    
    # Save inside the temp env
    os.makedirs("data", exist_ok=True)
    csv_path = "data/data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path

@pytest.fixture
def classifier(mock_data_csv):
    """Returns a classifier instance pointing to the local MLflow backend."""
    # Point MLflow to a local temp folder to avoid messing up global runs
    tracking_uri = "sqlite:///mlflow.db" 
    return XGBoostBreastCancerClassifier(
        data_path=mock_data_csv, 
        tracking_uri=tracking_uri
    )

# --- Tests ---

def test_load_and_prepare_data(classifier):
    """Test if data is loaded, split, and scaled correctly."""
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.load_and_prepare_data()
    
    # Check shapes (approx 70/15/15 split of 20 samples)
    assert len(X_train) + len(X_val) + len(X_test) == 20
    assert classifier.n_features == 5
    
    # Check if scaling worked (mean should be close to 0)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1.0)
    
    # Check scale_pos_weight
    # We have equal classes (10 M, 10 B), so weight should be approx 1.0
    assert 0.8 <= classifier.scale_pos_weight <= 1.2

def test_full_training_pipeline(classifier, mock_env):
    """
    Integration test: Runs the full train_and_optimize loop.
    Checks if artifacts, models, and logs are actually created.
    """
    # Run optimization with minimal trials for speed
    # We mock xgb.train via parameters to make it fast (few rounds)
    # But since we can't easily inject params into the inner loop without mocking,
    # we just trust the small dataset (20 rows) to be fast.
    
    model, threshold = classifier.train_and_optimize(n_trials=1)
    
    # 1. Check Returns
    assert isinstance(model, xgb.Booster)
    assert 0.0 < threshold < 1.0
    
    # 2. Check Directory Creation
    assert os.path.exists("artifacts")
    assert os.path.exists("plots")
    assert os.path.exists("models")
    
    # 3. Check Specific Artifacts
    assert os.path.exists("artifacts/model.json")
    assert os.path.exists("artifacts/threshold.json")
    assert os.path.exists("artifacts/preprocessor.pkl")
    assert os.path.exists("plots/confusion_matrix.png")
    assert os.path.exists("plots/roc_curve.png")
    
    # 4. Verify Model loads correctly
    loaded_model = xgb.Booster()
    loaded_model.load_model("artifacts/model.json")
    assert loaded_model.num_features() == 5

def test_preprocessor_pickling(classifier):
    """Verify the processor pickle file works (addressing your code comment)."""
    # Run data prep to fit the processor
    classifier.load_and_prepare_data()
    
    # Manually save
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/preprocessor.pkl", "wb") as f:
        pickle.dump(classifier.processor, f)
        
    # Load back
    with open("artifacts/preprocessor.pkl", "rb") as f:
        loaded_processor = pickle.load(f)
        
    # Test if loaded processor works
    sample_data = np.random.rand(1, 5)
    
    # Should not raise error
    try:
        loaded_processor.transform(sample_data)
    except Exception as e:
        pytest.fail(f"Pickled processor failed to transform data: {e}")

def test_objective_function(classifier):
    """Test the Optuna objective function in isolation."""
    import optuna
    
    # Prepare data
    X_train, _, _, y_train, _, _ = classifier.load_and_prepare_data()
    cv = classifier.validator.get_stratified_kfold(n_splits=2) # 2 splits for tiny data
    
    # Mock an Optuna trial
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    
    # Run objective
    score = classifier.objective(trial, X_train, y_train, cv)
    
    # Score should be AUC (0 to 1)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_find_optimal_threshold(classifier):
    """Test threshold optimization logic."""
    # Create a dummy model that predicts probabilities
    class DummyModel:
        def predict(self, dmatrix):
            # Return fixed probabilities to force a specific result
            # 10 samples
            return np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.2, 0.9, 0.8])
            
    X_val = np.random.rand(10, 5)
    # create labels that match the high probs above
    y_val = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    
    threshold, f1 = classifier.find_optimal_threshold(DummyModel(), X_val, y_val, plot=False)
    
    # The threshold should likely be around 0.5 given the data above
    assert 0.3 < threshold < 0.7
    assert f1 > 0.8  # Should be a good score
