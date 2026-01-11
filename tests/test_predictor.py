import pytest
import numpy as np
import os
import json
import pickle
import xgboost as xgb
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler
from logic.breast_cancer_predictor import BreastCancerPredictor

# ==========================================
# ️ FIXTURES
# ==========================================

@pytest.fixture
def common_preprocessor(tmp_path):
    """Creates a dummy preprocessor used by both models."""
    X_train = np.random.rand(10, 5)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    path = tmp_path / "preprocessor.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    
    # Also save the threshold (common to both)
    with open(tmp_path / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)
        
    return scaler

@pytest.fixture
def xgboost_artifacts(tmp_path, common_preprocessor):
    """
    Sets up a 'Real' XGBoost environment.
    Writes model.json, metadata.json, and preprocessor.
    """
    # 1. Create Metadata (CRITICAL CHANGE)
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({
            "model_type": "xgboost",
            "n_features": 5
        }, f)

    # 2. Create and save a dummy XGBoost model
    # We train it briefly so it has the correct structure (5 features)
    X_dummy = np.random.rand(10, 5)
    X_scaled = common_preprocessor.transform(X_dummy)
    y_dummy = np.random.randint(0, 2, 10)
    
    dtrain = xgb.DMatrix(X_scaled, label=y_dummy)
    model = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=1)
    model.save_model(tmp_path / "model.json")

    return str(tmp_path)

@pytest.fixture
def tabnet_artifacts(tmp_path, common_preprocessor):
    """
    Sets up a 'Mocked' TabNet environment.
    Writes metadata.json and preprocessor, but Mocks the ONNX file.
    """
    # 1. Create Metadata indicating TabNet
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({
            "model_type": "tabnet", 
            "n_features": 5  # Must match the preprocessor
        }, f)
        
    # 2. Create a dummy ONNX file (content doesn't matter as we will mock the loader)
    with open(tmp_path / "tabnet.onnx", "w") as f:
        f.write("dummy content")
        
    return str(tmp_path)

# ==========================================
# 離 TESTS
# ==========================================

# --- 1. XGBoost Tests (Integration Style) ---

def test_xgboost_initialization(xgboost_artifacts):
    """Test that XGBoost loads correctly with the new metadata system."""
    predictor = BreastCancerPredictor(artifact_dir=xgboost_artifacts)
    
    assert predictor.model_type == "xgboost"
    assert isinstance(predictor.model, xgb.Booster)
    assert predictor.threshold == 0.5
    assert predictor.num_features == 5

def test_xgboost_prediction_flow(xgboost_artifacts):
    """Test end-to-end prediction with real XGBoost artifacts."""
    predictor = BreastCancerPredictor(artifact_dir=xgboost_artifacts)
    X_input = np.random.rand(1, 5)
    
    # Test Proba
    proba = predictor.predict_proba(X_input)
    assert 0.0 <= proba[0] <= 1.0
    
    # Test Class
    pred = predictor.predict(X_input)
    assert pred[0] in [0, 1]

# --- 2. TabNet Tests (Mocked Style) ---

def test_tabnet_initialization(tabnet_artifacts):
    """Test that TabNet initializes correctly using mocked ONNX runtime."""
    
    # We patch onnxruntime.InferenceSession so it doesn't try to read the dummy file
    with patch("onnxruntime.InferenceSession") as mock_session_cls:
        # Configure the mock session
        mock_session = mock_session_cls.return_value
        # Mock inputs info (needed for input_name retrieval)
        mock_input = MagicMock()
        mock_input.name = "input_node"
        mock_session.get_inputs.return_value = [mock_input]
        
        # Initialize
        predictor = BreastCancerPredictor(artifact_dir=tabnet_artifacts)
        
        assert predictor.model_type == "tabnet"
        assert predictor.num_features == 5
        # Verify it tried to load the onnx file
        mock_session_cls.assert_called_once()
        assert "tabnet.onnx" in str(mock_session_cls.call_args[0][0])

def test_tabnet_prediction_logic(tabnet_artifacts):
    """Test TabNet prediction logic including ONNX output parsing."""
    
    with patch("onnxruntime.InferenceSession") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_session.get_inputs.return_value = [MagicMock(name="input_node")]
        
        # Scenario A: ONNX returns (Batch, 2) probabilities -> [Prob_0, Prob_1]
        # We simulate a return of [[0.1, 0.9]] (90% confident positive)
        mock_session.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)]
        
        predictor = BreastCancerPredictor(artifact_dir=tabnet_artifacts)
        X_input = np.random.rand(1, 5)
        
        # Test
        prob = predictor.predict_proba(X_input)
        
        # Should extract the second column (0.9)
        assert len(prob) == 1
        assert prob[0] == 0.9
        
        # Scenario B: ONNX returns (Batch, 1) or flat -> just probability
        mock_session.run.return_value = [np.array([0.9], dtype=np.float32)]
        prob_flat = predictor.predict_proba(X_input)
        assert prob_flat[0] == 0.9

# --- 3. Shared Logic Tests (Parametrized) ---
# These verify that common logic (preprocessing, dimension handling) works for BOTH

@pytest.mark.parametrize("artifact_fixture", ["xgboost_artifacts", "tabnet_artifacts"])
def test_input_preprocessing_and_types(artifact_fixture, request):
    """Verify data preparation logic works for both backends."""
    
    # Resolve the fixture value from the string name
    artifact_path = request.getfixturevalue(artifact_fixture)
    
    # We need to mock TabNet if that's the current fixture
    if "tabnet" in artifact_fixture:
        with patch("onnxruntime.InferenceSession") as mock_sess:
            mock_sess.return_value.get_inputs.return_value = [MagicMock(name="in")]
            predictor = BreastCancerPredictor(artifact_dir=artifact_path)
    else:
        predictor = BreastCancerPredictor(artifact_dir=artifact_path)

    # 1. Test Dimension Handling
    X_1d = np.random.rand(5)
    X_proc = predictor._prepare(X_1d)
    
    assert X_proc.ndim == 2
    assert X_proc.shape == (1, 5)
    
    # 2. Test Type Enforcement (ONNX requires float32)
    assert X_proc.dtype == np.float32

    # 3. Test Feature Mismatch Error
    X_bad = np.random.rand(1, 3) # Only 3 features
    with pytest.raises(ValueError, match="features"):
        predictor._prepare(X_bad)