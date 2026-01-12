import pytest
import os
import json
import shutil
from logic.evaluate import compare_and_promote

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_fs(tmp_path):
    """
    Sets up the artifacts directory structure.
    Returns the paths to the root, xgboost, and tabnet folders.
    """
    root_dir = tmp_path / "artifacts"
    xgb_dir = root_dir / "xgboost"
    tab_dir = root_dir / "tabnet"
    
    os.makedirs(xgb_dir)
    os.makedirs(tab_dir)
    
    return root_dir, xgb_dir, tab_dir

def create_mock_artifacts(directory, score, model_name):
    """Helper to create dummy metrics and artifact files."""
    # Create metrics.json
    metrics = {"roc_auc": score}
    with open(directory / "validation_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Create a dummy model file to check if it gets copied
    with open(directory / "model.bin", "w") as f:
        f.write(f"This is the {model_name} model")

# ==========================================
# Tests
# ==========================================

def test_xgboost_wins(mock_fs):
    """Test scenario where XGBoost has a higher score."""
    root_dir, xgb_dir, tab_dir = mock_fs
    
    # Setup: XGBoost (0.9) > TabNet (0.8)
    create_mock_artifacts(xgb_dir, 0.9, "xgboost")
    create_mock_artifacts(tab_dir, 0.8, "tabnet")
    
    # Run comparison
    compare_and_promote(
        xgboost_dir=str(xgb_dir),
        tabnet_dir=str(tab_dir),
        output_dir=str(root_dir)
    )
    
    # Check 1: Summary file says xgboost
    with open(root_dir / "evaluation_summary.json") as f:
        summary = json.load(f)
        assert summary["winner"] == "xgboost"
        
    # Check 2: The correct model file was copied
    with open(root_dir / "model.bin") as f:
        content = f.read()
        assert "xgboost" in content

def test_tabnet_wins(mock_fs):
    """Test scenario where TabNet has a higher score."""
    root_dir, xgb_dir, tab_dir = mock_fs
    
    # Setup: TabNet (0.95) > XGBoost (0.9)
    create_mock_artifacts(xgb_dir, 0.90, "xgboost")
    create_mock_artifacts(tab_dir, 0.95, "tabnet")
    
    compare_and_promote(
        xgboost_dir=str(xgb_dir),
        tabnet_dir=str(tab_dir),
        output_dir=str(root_dir)
    )
    
    # Check 1: Summary file says tabnet
    with open(root_dir / "evaluation_summary.json") as f:
        summary = json.load(f)
        assert summary["winner"] == "tabnet"

    # Check 2: The correct model file was copied
    with open(root_dir / "model.bin") as f:
        content = f.read()
        assert "tabnet" in content

def test_missing_metrics_default(mock_fs):
    """Test behavior when metrics files are missing (should default to XGBoost)."""
    root_dir, xgb_dir, tab_dir = mock_fs
    
    # Setup: Create artifacts but NO metrics.json
    with open(xgb_dir / "model.bin", "w") as f: f.write("xgb_model")
    
    # Run
    compare_and_promote(
        xgboost_dir=str(xgb_dir),
        tabnet_dir=str(tab_dir),
        output_dir=str(root_dir)
    )
    
    # Should default to "xgboost"
    with open(root_dir / "evaluation_summary.json") as f:
        summary = json.load(f)
        assert summary["winner"] == "xgboost"
        
    assert (root_dir / "model.bin").exists()

def test_cleanup_logic(mock_fs):
    """Test if old files in root are deleted before copying new ones."""
    root_dir, xgb_dir, tab_dir = mock_fs
    
    # Setup: Create an "old" file in root that should be deleted
    old_file = root_dir / "old_junk.txt"
    with open(old_file, "w") as f:
        f.write("delete me")
        
    # Create XGB artifacts to promote
    create_mock_artifacts(xgb_dir, 0.9, "xgboost")
    create_mock_artifacts(tab_dir, 0.8, "tabnet")
    
    compare_and_promote(
        xgboost_dir=str(xgb_dir),
        tabnet_dir=str(tab_dir),
        output_dir=str(root_dir)
    )
    
    # The old file should be gone
    assert not old_file.exists()
    
    # The new file should exist
    assert (root_dir / "model.bin").exists()

def test_subfolder_safety(mock_fs):
    """Test that subfolders (like artifacts/xgboost) are NOT deleted during cleanup."""
    root_dir, xgb_dir, tab_dir = mock_fs
    
    # Create dummy data
    create_mock_artifacts(xgb_dir, 0.9, "xgboost")
    create_mock_artifacts(tab_dir, 0.8, "tabnet")
    
    compare_and_promote(
        xgboost_dir=str(xgb_dir),
        tabnet_dir=str(tab_dir),
        output_dir=str(root_dir)
    )
    
    # Subfolders must still exist
    assert xgb_dir.exists()
    assert tab_dir.exists()
