import pytest
import numpy as np
import pandas as pd
import os
from logic.data_module import CancerDataProcessor

# --- Fixtures ---

@pytest.fixture
def processor():
    """Returns a fresh instance of CancerDataProcessor for each test."""
    return CancerDataProcessor()

@pytest.fixture
def mock_csv(tmp_path):
    """
    Creates a temporary CSV file with dummy cancer data.
    Returns the path to the file.
    """
    # Create dummy data dictionary
    data = {
        "id": [101, 102, 103, 104],
        "diagnosis": ["M", "B", "M", "B"],  # M=1, B=0
        "radius_mean": [10.0, 20.0, 10.0, 20.0],
        "texture_mean": [15.0, 25.0, 15.0, 25.0]
    }
    df = pd.DataFrame(data)
    
    # Save to a temporary directory managed by pytest
    file_path = tmp_path / "test_cancer_data.csv"
    df.to_csv(file_path, index=False)
    
    return str(file_path)

# --- Tests for Data Loading ---

def test_load_data_success(processor, mock_csv):
    """Test if data loads, id is dropped, and diagnosis is mapped correctly."""
    X, y = processor.load_data(mock_csv)
    
    # 1. Check return types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # 2. Check Shapes
    # 4 rows, 2 features (radius, texture) - 'id' and 'diagnosis' should be gone
    assert X.shape == (4, 2) 
    assert y.shape == (4,)
    
    # 3. Check Label Mapping (M->1, B->0)
    # Original: M, B, M, B -> Expected: 1, 0, 1, 0
    np.testing.assert_array_equal(y, np.array([1, 0, 1, 0]))

def test_load_data_missing_column(processor, tmp_path):
    """Test if error is raised when 'diagnosis' column is missing."""
    # Create bad data
    df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    bad_file = tmp_path / "bad_data.csv"
    df.to_csv(bad_file, index=False)
    
    with pytest.raises(ValueError, match="Target column 'diagnosis' not found"):
        processor.load_data(str(bad_file))

# --- Tests for Preprocessing (Scaling) ---

def test_fit_transform_logic(processor):
    """Test if the scaler actually normalizes the data."""
    # Create simple data: column 1 has mean 10, col 2 has mean 100
    X = np.array([
        [0.0, 0.0],
        [20.0, 200.0]
    ])
    # Mean of col 1 is 10.0, Mean of col 2 is 100.0
    
    X_scaled = processor.fit_transform(X)
    
    assert processor.is_fitted is True
    
    # After standard scaling, mean should be approx 0 and std approx 1
    # Use np.isclose for floating point comparisons
    assert np.allclose(X_scaled.mean(axis=0), 0.0)
    assert np.allclose(X_scaled.std(axis=0), 1.0)

def test_transform_without_fit_error(processor):
    """Test if transform raises error when called before fitting."""
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match="Preprocessor not fitted"):
        processor.transform(X)

def test_transform_consistency(processor):
    """Test if transform applies the SAME scaling as fit_transform."""
    X_train = np.array([[0], [10]]) # Mean=5, Std=5
    X_test = np.array([[5]])        # Should be scaled to 0 (since it equals the mean)
    
    processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)
    
    # (5 - 5) / 5 = 0
    assert X_test_scaled[0][0] == 0.0
