import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold
from logic.utilities import DataValidator, set_seed

@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 5)
    y = np.array([0] * 90 + [1] * 10) # 100 samples
    return X, y

# --- TESTS ---
def test_set_seed():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)

def test_split_train_val_test_shapes(dummy_data):
    X, y = dummy_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataValidator.split_train_val_test(
        X, y, train_size=0.7, val_size=0.15, test_size=0.15
    )
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

def test_get_scale_pos_weight():
    y = np.array([0, 0, 0, 1]) 
    weight = DataValidator.get_scale_pos_weight(y)
    assert weight == 3.0

def test_get_class_weights():
    y = np.array([0, 0, 0, 1])
    weights = DataValidator.get_class_weights(y)
    assert weights[1] > weights[0]

def test_split_train_test(dummy_data):
    """Test the 2-way split (train/test) which was previously missed."""
    X, y = dummy_data
    X_train, X_test, y_train, y_test = DataValidator.split_train_test(
        X, y, test_size=0.2
    )
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    # Ensure stratification logic ran
    assert np.isclose(y_train.mean(), 0.1) 
    assert np.isclose(y_test.mean(), 0.1)

def test_split_train_test_error_handling(dummy_data):
    """Test that passing y=None raises ValueError."""
    X, _ = dummy_data
    with pytest.raises(ValueError, match="y cannot be None"):
        DataValidator.split_train_test(X, None)

def test_split_train_val_test_error_handling(dummy_data):
    """Test error handling for the 3-way split."""
    X, _ = dummy_data
    # Case 1: y is None
    with pytest.raises(ValueError, match="y cannot be None"):
        DataValidator.split_train_val_test(X, None)
        
    # Case 2: Ratios don't sum to 1.0 (already added before, but good to keep)
    with pytest.raises(ValueError, match="must equal 1.0"):
        DataValidator.split_train_val_test(X, np.zeros(100), train_size=0.5, val_size=0.1, test_size=0.1)

def test_get_stratified_kfold():
    """Test if the KFold object is created with correct parameters."""
    kf = DataValidator.get_stratified_kfold(n_splits=5, random_state=99)
    
    assert isinstance(kf, StratifiedKFold)
    assert kf.n_splits == 5
    assert kf.random_state == 99
    assert kf.shuffle is True

def test_utility_errors_null_y():
    """Trigger the 'y is None' checks in weight calculators."""
    
    # 1. Class Weights
    with pytest.raises(ValueError, match="y cannot be None"):
        DataValidator.get_class_weights(None)
        
    # 2. Scale Pos Weight
    with pytest.raises(ValueError, match="y cannot be None"):
        DataValidator.get_scale_pos_weight(None)
