import pytest
import os
from unittest.mock import patch, MagicMock
from logic.retraining_pipeline import RetrainingPipeline

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def mock_env():
    """Sets up the necessary environment variables for the pipeline."""
    # We use patch.dict to safely modify os.environ for just this test context
    env_vars = {
        'GITHUB_TOKEN': 'fake_token_123',
        'GITHUB_REPO_OWNER': 'my_org',
        'GITHUB_REPO_NAME': 'my_repo'
    }
    with patch.dict(os.environ, env_vars):
        yield

# ==========================================
# Tests
# ==========================================

def test_initialization_with_env_vars(mock_env):
    """Test if the class correctly loads config from environment variables."""
    pipeline = RetrainingPipeline()
    
    assert pipeline.github_token == 'fake_token_123'
    assert pipeline.repo_owner == 'my_org'
    assert pipeline.repo_name == 'my_repo'
    # Default values hardcoded in class
    assert pipeline.workflow_id == 'CICD.yml'
    assert pipeline.ref == 'main'

def test_missing_token_error():
    """Test behavior when GITHUB_TOKEN is missing."""
    # Ensure environment is empty regarding the token
    with patch.dict(os.environ, {}, clear=True):
        pipeline = RetrainingPipeline()
        # The __init__ might load None, or logic handles it in run()
        # Your code checks it inside run()
        
        result = pipeline.run("Test Reason")
        
        assert result['success'] is False
        assert result['error'] == 'Missing GITHUB_TOKEN'

@patch('requests.post')
def test_successful_trigger(mock_post, mock_env):
    """Test a successful 204 response from GitHub API."""
    # Setup the mock response
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response
    
    pipeline = RetrainingPipeline()
    result = pipeline.run("Drift Detected")
    
    # 1. Check Return Value
    assert result['success'] is True
    assert result['message'] == 'GitHub Action triggered successfully'
    assert 'timestamp' in result
    
    # 2. Check if requests.post was called with correct URL and Headers
    expected_url = "https://api.github.com/repos/my_org/my_repo/actions/workflows/CICD.yml/dispatches"
    
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    
    assert args[0] == expected_url
    assert kwargs['headers']['Authorization'] == "token fake_token_123"
    assert kwargs['json']['inputs']['reason'] == "Drift Detected"

@patch('requests.post')
def test_api_failure_response(mock_post, mock_env):
    """Test how the code handles a 401 or 404 error from GitHub."""
    # Setup mock for failure (e.g., Bad Credentials)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"message": "Bad credentials"}'
    mock_post.return_value = mock_response
    
    pipeline = RetrainingPipeline()
    result = pipeline.run("Manual Test")
    
    assert result['success'] is False
    assert "GitHub API Error" in result['error']
    assert "Bad credentials" in result['error']

@patch('requests.post')
def test_exception_handling(mock_post, mock_env):
    """Test what happens if the internet connection is down (requests raises Exception)."""
    # Configure mock to raise an exception instead of returning a response
    mock_post.side_effect = Exception("Connection Timed Out")
    
    pipeline = RetrainingPipeline()
    result = pipeline.run("Manual Test")
    
    assert result['success'] is False
    assert result['error'] == "Connection Timed Out"
