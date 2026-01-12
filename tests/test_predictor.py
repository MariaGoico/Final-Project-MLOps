import pytest
import os
import json
import pickle
import numpy as np
import xgboost as xgb
import torch
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from logic.breast_cancer_predictor import BreastCancerPredictor

# ==========================================
# Helpers for Dummy Artifact Creation
# ==========================================

def create_common_artifacts(path, n_features=5):
    """Creates preprocessor and threshold common to both models."""
    # 1. Preprocessor
    scaler = StandardScaler()
    scaler.fit(np.random.rand(10, n_features))
    with open(path / "preprocessor.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # 2. Threshold
    with open(path / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

def create_xgboost_artifacts(path, n_features=5):
    """Creates a dummy XGBoost model and metadata."""
    create_common_artifacts(path, n_features)
    
    # Metadata
    with open(path / "metadata.json", "w") as f:
        json.dump({"model_type": "xgboost", "n_features": n_features}, f)
        
    # Model
    X = np.random.rand(10, n_features)
    y = np.random.randint(0, 2, 10)
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train({}, dtrain, num_boost_round=1)
    model.save_model(path / "model.json")

def create_tabnet_artifacts(path, n_features=5):
    """Creates a dummy ONNX model and metadata."""
    create_common_artifacts(path, n_features)
    
    # Metadata
    with open(path / "metadata.json", "w") as f:
        json.dump({"model_type": "tabnet", "n_features": n_features}, f)
        
    # Create a tiny PyTorch model and export to ONNX
    class TinyModel(torch.nn.Module):
        def forward(self, x):
            # Simulate output [prob_0, prob_1]
            # Just return a fixed tensor for simplicity or simple math
            return torch.stack([x[:,0], x[:,0]], dim=1) 

    model = TinyModel()
    dummy_input = torch.randn(1, n_features)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        path / "tabnet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )

# ==========================================
# Tests
# ==========================================

def test_init_xgboost_mode(tmp_path):
    """Test if predictor correctly loads XGBoost backend."""
    create_xgboost_artifacts(tmp_path)
    
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    assert predictor.model_type == "xgboost"
    assert isinstance(predictor.model, xgb.Booster)
    # Ensure it didn't try to load ONNX session
    assert not hasattr(predictor, "session")
    assert predictor.feature_count == 5

def test_init_tabnet_mode(tmp_path):
    """Test if predictor correctly loads TabNet/ONNX backend."""
    create_tabnet_artifacts(tmp_path)
    
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    assert predictor.model_type == "tabnet"
    assert hasattr(predictor, "session")
    assert isinstance(predictor.session, ort.InferenceSession)
    # Ensure it didn't try to load XGBoost
    assert not hasattr(predictor, "model")
    assert predictor.feature_count == 5

def test_predict_xgboost(tmp_path):
    """Test prediction logic using XGBoost backend."""
    create_xgboost_artifacts(tmp_path, n_features=5)
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    X = np.random.rand(1, 5)
    
    # Probabilities
    proba = predictor.predict_proba(X)
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0
    
    # Classes
    pred = predictor.predict(X)
    assert pred[0] in [0, 1]

def test_predict_tabnet(tmp_path):
    """Test prediction logic using ONNX backend."""
    create_tabnet_artifacts(tmp_path, n_features=5)
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    X = np.random.rand(1, 5)
    
    # Probabilities
    # Our dummy ONNX model returns values, we just check shape/type logic
    proba = predictor.predict_proba(X)
    
    assert isinstance(proba, np.ndarray)
    assert len(proba) == 1
    
    # Classes
    pred = predictor.predict(X)
    assert pred[0] in [0, 1]

def test_predict_confidence_tuple(tmp_path):
    """Test the (class, probability) return format."""
    create_xgboost_artifacts(tmp_path)
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    X = np.random.rand(5) # 1D array
    
    cls, prob = predictor.predict_with_confidence(X)
    
    assert isinstance(cls, int)
    assert isinstance(prob, float)

def test_input_dimension_fix(tmp_path):
    """Test that 1D input is auto-reshaped to 2D."""
    create_xgboost_artifacts(tmp_path, n_features=3)
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    X_1d = np.array([0.1, 0.2, 0.3])
    
    # Should not crash
    predictor.predict(X_1d)

def test_error_missing_metadata(tmp_path):
    """Test specific error for missing metadata.json."""
    # Create an empty dir
    with pytest.raises(FileNotFoundError, match="Metadata not found"):
        BreastCancerPredictor(artifact_dir=str(tmp_path))

def test_error_unknown_model_type(tmp_path):
    """Test error when metadata contains garbage model type."""
    create_common_artifacts(tmp_path)
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"model_type": "random_forest"}, f)
        
    with pytest.raises(ValueError, match="Unknown model type"):
        BreastCancerPredictor(artifact_dir=str(tmp_path))

def test_error_feature_mismatch(tmp_path):
    """Test validation logic when input shape is wrong."""
    create_xgboost_artifacts(tmp_path, n_features=5)
    predictor = BreastCancerPredictor(artifact_dir=str(tmp_path))
    
    # Pass 3 features instead of 5
    X_bad = np.random.rand(1, 3)
    
    # Accepts "features" in message (covers both custom ValueError and sklearn errors)
    with pytest.raises(ValueError, match="features"):
        predictor.predict(X_bad)
