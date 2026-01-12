import pytest
import numpy as np
import pandas as pd
import os
from logic.data_module import CancerDataProcessor, CancerDataProcessorTabNet

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def processor():
    """Returns a fresh instance of the standard processor."""
    return CancerDataProcessor()

@pytest.fixture
def tabnet_processor():
    """Returns a fresh instance of the TabNet processor."""
    return CancerDataProcessorTabNet()

@pytest.fixture
def mock_csv(tmp_path):
    """Creates a temporary CSV file with dummy cancer data."""
    data = {
        "id": [101, 102, 103, 104],
        "diagnosis": ["M", "B", "M", "B"],  # M=1, B=0
        "radius_mean": [10.0, 20.0, 10.0, 20.0],
        "texture_mean": [15.0, 25.0, 15.0, 25.0]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_cancer_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

# ==========================================
# Tests: Standard CancerDataProcessor
# ==========================================

def test_load_data_success(processor, mock_csv):
    X, y = processor.load_data(mock_csv)
    assert X.shape == (4, 2)
    assert y.shape == (4,)
    # M->1, B->0
    np.testing.assert_array_equal(y, np.array([1, 0, 1, 0]))

def test_load_data_missing_column(processor, tmp_path):
    df = pd.DataFrame({"id": [1], "val": [10]})
    bad_file = tmp_path / "bad.csv"
    df.to_csv(bad_file, index=False)
    
    with pytest.raises(ValueError, match="Target column 'diagnosis' not found"):
        processor.load_data(str(bad_file))

def test_standard_fit_transform(processor):
    X = np.array([[0.0], [10.0]]) # Mean=5, Std=5
    X_scaled = processor.fit_transform(X)
    assert processor.is_fitted
    # (0-5)/5 = -1, (10-5)/5 = 1
    assert np.allclose(X_scaled, [[-1.0], [1.0]])

def test_standard_transform_unfitted_error(processor):
    with pytest.raises(ValueError, match="Preprocessor not fitted"):
        processor.transform(np.array([[1]]))

# ==========================================
# Tests: CancerDataProcessorTabNet
# ==========================================

def test_tabnet_load_data_properties(tabnet_processor, mock_csv):
    """Test that TabNet loader forces specific dtypes and captures feature names."""
    X, y = tabnet_processor.load_data(mock_csv)
    
    # Check Types (TabNet is strict about float32/int64)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    
    # Check Feature Names capture
    expected_features = ["radius_mean", "texture_mean"]
    assert tabnet_processor.feature_names_ == expected_features

def test_tabnet_variance_removal(tabnet_processor):
    """Test that columns with zero variance (constant values) are removed."""
    # Col 0: Varying, Col 1: Constant (5.0)
    X = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0]
    ], dtype=np.float32)
    
    # Manually set names to track which one gets removed
    tabnet_processor.feature_names_ = ["var_col", "const_col"]
    
    X_trans = tabnet_processor.fit_transform(X)
    
    # Result should have 1 column only
    assert X_trans.shape[1] == 1
    
    # "const_col" should be gone from feature_names_
    assert tabnet_processor.feature_names_ == ["var_col"]

def test_tabnet_nan_safety_check(tabnet_processor):
    """Test that fit_transform raises error if NaNs exist."""
    # Ensure the column with NaN varies (1.0 vs 2.0) so VarianceThreshold keeps it
    X_nan = np.array([
        [1.0, 5.0], 
        [2.0, np.nan], 
        [3.0, 6.0]
    ], dtype=np.float32)
    
    tabnet_processor.feature_names_ = ["col1", "col2"]
    
    with pytest.raises(ValueError, match="Non-finite values"):
        tabnet_processor.fit_transform(X_nan)

def test_tabnet_persistence(tabnet_processor, tmp_path):
    """Test save() and load() functionality."""
    # Setup state
    X = np.array([[1.0], [2.0]], dtype=np.float32)
    tabnet_processor.feature_names_ = ["col1"]
    tabnet_processor.fit_transform(X)
    
    # Save
    save_path = tmp_path / "pipeline" / "tabnet.pkl"
    tabnet_processor.save(str(save_path))
    
    # Check file exists
    assert save_path.exists()
    
    # Load
    loaded_processor = CancerDataProcessorTabNet.load(str(save_path))
    
    # Verify state matches
    assert loaded_processor.is_fitted is True
    assert loaded_processor.feature_names_ == ["col1"]
    
    # Verify behavior matches (transform should work)
    X_test = np.array([[1.5]], dtype=np.float32)
    res_orig = tabnet_processor.transform(X_test)
    res_load = loaded_processor.transform(X_test)
    
    np.testing.assert_array_equal(res_orig, res_load)

def test_tabnet_transform_unfitted_error(tabnet_processor):
    """Test that transform requires fitting first."""
    X = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="must be fitted"):
        tabnet_processor.transform(X)
