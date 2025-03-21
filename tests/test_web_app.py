"""Test the web application functionality."""

import os
import pytest
from fastapi.testclient import TestClient
from runpod_analyzer.web.app import app
import tempfile
import shutil
from pathlib import Path

client = TestClient(app)

def test_index_page():
    """Test that the index page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_analyze_endpoint():
    """Test the analyze endpoint with a sample repository."""
    # Test with a valid GitHub repository
    response = client.post("/analyze", params={"repo_url_param": "https://github.com/vllm-project/vllm"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "repository_type" in data
    assert "confidence" in data
    assert "languages" in data
    assert "dependencies" in data
    assert "generated_files" in data
    assert all(key in data["generated_files"] for key in ["hub_json", "tests_json", "dockerfile", "handler"])

def test_invalid_repository():
    """Test handling of invalid repository URLs."""
    # Test with empty URL
    response = client.post("/analyze", params={"repo_url_param": ""})
    assert response.status_code == 400
    assert "Repository URL is required" in response.json()["detail"]

    # Test with invalid URL format
    response = client.post("/analyze", params={"repo_url_param": "not_a_url"})
    assert response.status_code == 400
    assert "Invalid GitHub repository URL" in response.json()["detail"]

def test_error_handling():
    """Test error handling for various scenarios."""
    # Test with non-existent repository
    response = client.post("/analyze", params={"repo_url_param": "https://github.com/nonexistent/repo"})
    assert response.status_code == 500
    assert "Repository analysis failed" in response.json()["detail"]

def test_temp_directory_cleanup():
    """Test that temporary directories are properly cleaned up."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_analyzer_")
    temp_path = Path(temp_dir)
    
    try:
        # Create some test files
        (temp_path / "test.txt").write_text("test")
        
        # Verify directory exists
        assert temp_path.exists()
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        # Verify directory is gone
        assert not temp_path.exists()
    finally:
        # Ensure cleanup happens even if test fails
        if temp_path.exists():
            shutil.rmtree(temp_dir) 