import pytest
import os
import pandas as pd
import numpy as np
from logic.generate_baseline import generate_baseline

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_fs(tmp_path):
    """
    Sets up a temporary file system:
    1. Changes the Current Working Directory (CWD) to a temp folder.
    2. Creates the 'data/' directory.
    3. Creates a dummy 'data/data.csv' with known statistics.
    """
    # 1. Save original CWD and switch to temp
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # 2. Create data directory
    os.makedirs("data", exist_ok=True)
    
    # 3. Create dummy data
    # Feature 1: Constant 10 -> Mean=10, Std=0
    # Feature 2: 0, 10 -> Mean=5, Std=~7.07
    # ID & Diagnosis: Should be dropped
    df = pd.DataFrame({
        "id": [1, 2],
        "diagnosis": ["M", "B"],
        "const_feat": [10.0, 10.0],
        "var_feat": [0.0, 10.0]
    })
    
    csv_path = "data/data.csv"
    df.to_csv(csv_path, index=False)
    
    yield tmp_path
    
    # Teardown: Restore CWD
    os.chdir(original_cwd)

# ==========================================
# Tests
# ==========================================

def test_generate_baseline_success(mock_fs):
    """
    Test that the script:
    1. Reads data correctly.
    2. Drops 'id' and 'diagnosis'.
    3. Calculates correct Mean/Std.
    4. Saves the result to the correct artifacts path.
    """
    # Run the function under test
    generate_baseline()
    
    # Verify the output file was created
    expected_path = "artifacts/feature_baseline.npz"
    assert os.path.exists(expected_path)
    
    # Load the file and verify contents
    data = np.load(expected_path)
    
    # 1. Check Feature Names
    # Should only contain the numeric features, order maintained
    assert "feature_names" in data
    assert data["feature_names"].tolist() == ["const_feat", "var_feat"]
    
    # 2. Check Means
    # const_feat mean = 10.0
    # var_feat mean = 5.0
    expected_means = np.array([10.0, 5.0])
    np.testing.assert_array_equal(data["means"], expected_means)
    
    # 3. Check Stds
    # const_feat std = 0.0
    # var_feat std (population vs sample): Pandas default is sample std (ddof=1)
    # std([0, 10]) = sqrt(50) ≈ 7.071
    expected_stds = np.array([0.0, 7.0710678])
    np.testing.assert_allclose(data["stds"], expected_stds, rtol=1e-5)

def test_directory_creation(mock_fs):
    """Test that the artifacts directory is automatically created if it doesn't exist."""
    # Ensure artifacts dir does not exist yet
    if os.path.exists("artifacts"):
        import shutil
        shutil.rmtree("artifacts")
        
    generate_baseline()
    
    assert os.path.isdir("artifacts")

def test_missing_data_file(tmp_path):
    """Test that it raises an error if data.csv is missing."""
    # We purposefully do NOT use the mock_fs fixture here to avoid creating the CSV
    # But we still need to chdir to be safe
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Ensure no data file exists
    if os.path.exists("data/data.csv"):
        os.remove("data/data.csv")
        
    try:
        with pytest.raises(FileNotFoundError):
            generate_baseline()
    finally:
        os.chdir(original_cwd)
