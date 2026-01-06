import pytest
import numpy as np
import os
import json
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from logic.breast_cancer_predictor import BreastCancerPredictor

# --- Fixtures: Setup Dummy Artifacts ---

@pytest.fixture
def mock_artifact_dir(tmp_path):
    """
    Creates a temporary directory with valid dummy artifacts:
    - model.json (Tiny XGBoost model)
    - preprocessor.pkl (Fitted StandardScaler)
    - threshold.json (Threshold value)
    """
    # 1. Create and save a dummy Preprocessor
    # We pretend our model uses 5 features
    X_train = np.random.rand(10, 5)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    with open(tmp_path / "preprocessor.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 2. Create and save a dummy XGBoost model
    # We train it briefly on random data so it has the correct structure
    y_train = np.random.randint(0, 2, 10)
    dtrain = xgb.DMatrix(scaler.transform(X_train), label=y_train)
    
    model = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=1)
    model.save_model(tmp_path / "model.json")

    # 3. Create threshold.json
    with open(tmp_path / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

    return str(tmp_path)

@pytest.fixture
def predictor(mock_artifact_dir):
    """Returns an instance of the predictor pointed at the dummy artifacts."""
    return BreastCancerPredictor(artifact_dir=mock_artifact_dir)

# --- Tests ---

def test_initialization(predictor):
    """Test if model, processor, and threshold load correctly."""
    assert isinstance(predictor.model, xgb.Booster)
    assert isinstance(predictor.processor, StandardScaler)
    assert predictor.threshold == 0.5

def test_prediction_shape_and_range(predictor):
    """Test that predict_proba returns valid probabilities."""
    # Input: 5 features (matching our dummy model)
    X_input = np.random.rand(1, 5) 
    
    proba = predictor.predict_proba(X_input)
    
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0

def test_prediction_classes(predictor):
    """Test that predict returns binary classes (0 or 1)."""
    X_input = np.random.rand(5, 5)  # Batch of 5 samples
    
    preds = predictor.predict(X_input)
    
    assert len(preds) == 5
    assert np.all(np.isin(preds, [0, 1]))  # All values must be 0 or 1

def test_predict_with_confidence(predictor):
    """Test the detailed return tuple (class, probability)."""
    X_input = np.random.rand(5)  # Single sample (1D array)
    
    cls, prob = predictor.predict_with_confidence(X_input)
    
    assert isinstance(cls, int)
    assert cls in [0, 1]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_input_dimension_handling(predictor):
    """Test if 1D array is correctly reshaped to 2D inside _prepare."""
    X_1d = np.random.rand(5)
    
    # This should not raise an error
    dmatrix = predictor._prepare(X_1d)
    
    # XGBoost DMatrix should have 1 row and 5 columns
    assert dmatrix.num_row() == 1
    assert dmatrix.num_col() == 5

def test_feature_mismatch_error(predictor):
    """Test if ValueError is raised when input features don't match model."""
    # Model expects 5 features, we pass 3
    X_bad = np.random.rand(1, 3)
    
    # Match the error coming from StandardScaler OR your custom error
    # We use a broader regex or simply match "features" to be safe against both cases
    with pytest.raises(ValueError, match="features"):
        predictor.predict(X_bad)

def test_threshold_logic(predictor):
    """Test if the threshold is actually applied correctly."""
    # Force a specific probability via mocking (harder with XGBoost) 
    # OR: Manually check logic given the probability return
    
    X_input = np.random.rand(1, 5)
    probs = predictor.predict_proba(X_input)
    
    # Get the raw probability
    raw_prob = probs[0]
    
    # Get the prediction
    prediction = predictor.predict(X_input)[0]
    
    # Verify the logic
    expected_prediction = 1 if raw_prob >= 0.5 else 0
    assert prediction == expected_prediction
