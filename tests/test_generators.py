"""Tests for file generators."""

import os
import json
import shutil
import tempfile
from pathlib import Path
import pytest
from runpod_analyzer.core.types import RepoType
from runpod_analyzer.core.dockerfile_generator import generate_dockerfile, save_dockerfile
from runpod_analyzer.core.handler_generator import generate_handler, save_handler
from runpod_analyzer.core.metadata import generate_metadata, save_metadata
from runpod_analyzer.core.test_generator import generate_test_cases

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_dockerfile_generation(temp_dir):
    """Test Dockerfile generation."""
    # Test with various dependency combinations
    test_cases = [
        {
            "dependencies": {"torch", "transformers"},
            "expected_base": "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"
        },
        {
            "dependencies": {"tensorflow"},
            "expected_base": "tensorflow/tensorflow:2.13.0-gpu"
        },
        {
            "dependencies": {"flask", "fastapi"},
            "expected_base": "python:3.11-slim"
        }
    ]
    
    for case in test_cases:
        # Generate Dockerfile
        dockerfile = generate_dockerfile(temp_dir, case["dependencies"])
        
        # Check base image
        assert dockerfile.startswith(f"FROM {case['expected_base']}")
        
        # Check common elements
        assert "WORKDIR /app" in dockerfile
        assert "COPY requirements.txt" in dockerfile
        assert "RUN pip install" in dockerfile
        assert "COPY . ." in dockerfile
        assert "CMD" in dockerfile
        
        # Save and verify
        save_dockerfile(temp_dir, dockerfile)
        assert (temp_dir / "Dockerfile").exists()

def test_handler_generation(temp_dir):
    """Test handler.py generation."""
    # Test with various repository types and dependencies
    test_cases = [
        {
            "repo_type": RepoType.ML_MODEL,
            "dependencies": {"torch", "transformers"},
            "expected_imports": ["import torch", "from transformers import"]
        },
        {
            "repo_type": RepoType.ML_MODEL,
            "dependencies": {"tensorflow"},
            "expected_imports": ["import tensorflow as tf", "import numpy as np"]
        },
        {
            "repo_type": RepoType.API,
            "dependencies": {"flask"},
            "expected_imports": ["import runpod"]
        }
    ]
    
    for case in test_cases:
        # Generate handler
        handler = generate_handler(temp_dir, case["repo_type"], case["dependencies"])
        
        # Check imports
        for imp in case["expected_imports"]:
            assert imp in handler
        
        # Check common elements
        assert "@runpod.handler" in handler
        assert "def handler(event):" in handler
        assert "if __name__ == \"__main__\":" in handler
        
        # Save and verify
        save_handler(temp_dir, handler)
        assert (temp_dir / "handler.py").exists()

def test_metadata_generation(temp_dir):
    """Test hub.json metadata generation."""
    # Create test files
    readme = temp_dir / "README.md"
    readme.write_text("""
# Test Project

A machine learning project for testing.

## Features
- Deep learning
- PyTorch integration
- API endpoints
""")
    
    env_example = temp_dir / ".env.example"
    env_example.write_text("""
MODEL_NAME=bert-base-uncased
API_KEY=your-api-key-here
""")
    
    # Generate metadata
    metadata = generate_metadata(temp_dir)
    
    # Check basic structure
    assert "version" in metadata
    assert "title" in metadata
    assert "description" in metadata
    assert "tags" in metadata
    assert "env" in metadata
    
    # Check content
    assert metadata["title"] == "Test Project"
    assert "machine learning project" in metadata["description"].lower()
    assert "deep-learning" in metadata["tags"]
    assert any(env["name"] == "MODEL_NAME" for env in metadata["env"])
    
    # Save and verify
    save_metadata(temp_dir, metadata)
    assert (temp_dir / "hub.json").exists()
    
    # Verify saved content
    with open(temp_dir / "hub.json") as f:
        saved_metadata = json.load(f)
    assert saved_metadata == metadata

def test_test_case_generation():
    """Test test case generation."""
    # Test with various repository types
    test_cases = [
        {
            "repo_type": RepoType.ML_MODEL,
            "dependencies": {"torch", "transformers"},
            "expected_input": "text"
        },
        {
            "repo_type": RepoType.ML_MODEL,
            "dependencies": {"tensorflow"},
            "expected_input": "data"
        },
        {
            "repo_type": RepoType.API,
            "dependencies": set(),
            "expected_input": "input"
        }
    ]
    
    for case in test_cases:
        # Generate test cases
        tests = generate_test_cases(case["repo_type"], case["dependencies"])
        
        # Check structure
        assert isinstance(tests, list)
        assert len(tests) > 0
        assert "input" in tests[0]
        assert case["expected_input"] in str(tests[0]["input"]) 