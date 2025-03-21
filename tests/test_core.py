"""Unit tests for core repository analysis components."""

import os
import json
import shutil
import tempfile
from pathlib import Path
import pytest
from runpod_analyzer.core.analyzer import (
    detect_repository_type,
    detect_languages,
    get_dependencies,
    RepoType
)
from runpod_analyzer.main import get_repo_name
from runpod_analyzer.core.repo_patterns import analyze_repository

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

# Unit Tests

def test_detect_repository_type(temp_dir):
    """Test repository type detection."""
    # Create test files
    readme = temp_dir / "README.md"
    requirements = temp_dir / "requirements.txt"
    
    # Test ML Model detection
    readme.write_text("""
    # ML Project
    A deep learning project using PyTorch for image classification.
    """)
    requirements.write_text("""
    torch==2.0.0
    torchvision==0.15.0
    """)
    
    repo_type, confidence = detect_repository_type(temp_dir)
    assert repo_type == RepoType.ML_MODEL
    assert confidence >= 10.0
    
    # Test API detection
    readme.write_text("""
    # API Project
    A REST API built with FastAPI.
    """)
    requirements.write_text("""
    fastapi==0.68.0
    uvicorn==0.15.0
    """)
    
    repo_type, confidence = detect_repository_type(temp_dir)
    assert repo_type == RepoType.API
    assert confidence >= 10.0
    
    # Test unknown type
    readme.write_text("""
    # Generic Project
    A utility library.
    """)
    requirements.write_text("""
    requests==2.28.0
    """)
    
    repo_type, confidence = detect_repository_type(temp_dir)
    assert repo_type == RepoType.UNKNOWN
    assert confidence < 5.0

def test_detect_languages(temp_dir):
    """Test programming language detection."""
    # Create test files
    (temp_dir / "main.py").touch()
    (temp_dir / "utils.py").touch()
    (temp_dir / "app.js").touch()
    (temp_dir / "types.ts").touch()
    (temp_dir / "lib.cpp").touch()
    
    languages = detect_languages(temp_dir)
    
    assert "python" in languages
    assert "javascript" in languages
    assert "typescript" in languages
    assert "c++" in languages
    assert len(languages) == 4

def test_get_dependencies(temp_dir):
    """Test dependency detection from various files."""
    # Test requirements.txt
    requirements = temp_dir / "requirements.txt"
    requirements.write_text("""
    torch==2.0.0
    numpy>=1.24.0
    pandas[extras]>=2.0.0
    scikit-learn
    """)
    
    deps = get_dependencies(temp_dir)
    assert "torch" in deps
    assert "numpy" in deps
    assert "pandas" in deps
    assert "scikit-learn" in deps
    
    # Test setup.py
    setup_py = temp_dir / "setup.py"
    setup_py.write_text("""
    from setuptools import setup
    
    setup(
        name="test-project",
        install_requires=[
            "tensorflow>=2.13.0",
            "keras",
            "transformers[torch]"
        ]
    )
    """)
    
    deps = get_dependencies(temp_dir)
    assert "tensorflow" in deps
    assert "keras" in deps
    assert "transformers" in deps
    
    # Test pyproject.toml
    pyproject = temp_dir / "pyproject.toml"
    pyproject.write_text("""
    [tool.poetry]
    name = "test-project"
    
    [tool.poetry.dependencies]
    python = "^3.8"
    fastapi = "^0.95.0"
    pydantic = "^2.0.0"
    """)
    
    deps = get_dependencies(temp_dir)
    assert "fastapi" in deps
    assert "pydantic" in deps

def test_get_repo_name():
    """Test repository name extraction from URL."""
    test_cases = [
        {
            "url": "https://github.com/username/repo.git",
            "expected": "repo"
        },
        {
            "url": "https://github.com/org/project-name",
            "expected": "project-name"
        },
        {
            "url": "git@github.com:user/my_repo.git",
            "expected": "my_repo"
        },
        {
            "url": "https://gitlab.com/group/subgroup/repo.git",
            "expected": "repo"
        }
    ]
    
    for case in test_cases:
        assert get_repo_name(case["url"]) == case["expected"]

# Integration Tests

TEST_REPOS = [
    # Traditional ML
    {
        "url": "https://github.com/scikit-learn/scikit-learn.git",
        "expected_type": RepoType.ML_MODEL,
        "min_confidence": 10.0,
        "expected_languages": ["python"],
        "expected_dependencies": {"numpy", "scipy", "joblib"}
    },
    # NLP
    {
        "url": "https://github.com/huggingface/transformers.git",
        "expected_type": RepoType.ML_MODEL,
        "min_confidence": 10.0,
        "expected_languages": ["python"],
        "expected_dependencies": {"torch", "tensorflow", "transformers"}
    },
    # Deep Learning
    {
        "url": "https://github.com/keras-team/keras.git",
        "expected_type": RepoType.ML_MODEL,
        "min_confidence": 10.0,
        "expected_languages": ["python"],
        "expected_dependencies": {"tensorflow", "numpy"}
    },
    # ML UI Tool
    {
        "url": "https://github.com/gradio-app/gradio.git",
        "expected_type": RepoType.ML_MODEL,
        "min_confidence": 5.0,
        "expected_languages": ["python"],
        "expected_dependencies": {"fastapi", "pillow", "ffmpeg"}
    },
    # LangChain
    {
        "url": "https://github.com/langchain-ai/langchain.git",
        "expected_type": RepoType.ML_MODEL,
        "min_confidence": 5.0,
        "expected_languages": ["python"],
        "expected_dependencies": {"pydantic", "requests"}
    }
]

@pytest.mark.integration
@pytest.mark.parametrize("repo", TEST_REPOS)
def test_repository_analysis(temp_dir, repo):
    """Test repository analysis for various repositories."""
    # Analyze repository
    analysis = analyze_repository(repo["url"])
    
    # Check repository type
    assert analysis.repo_type == repo["expected_type"], \
        f"Expected type {repo['expected_type']}, got {analysis.repo_type}"
    
    # Check confidence score
    assert analysis.confidence >= repo["min_confidence"], \
        f"Expected confidence >= {repo['min_confidence']}, got {analysis.confidence}"
    
    # Check languages
    for lang in repo["expected_languages"]:
        assert lang in analysis.languages, \
            f"Expected language {lang} not found in {analysis.languages}"
    
    # Check dependencies
    for dep in repo["expected_dependencies"]:
        assert dep in analysis.dependencies, \
            f"Expected dependency {dep} not found in {analysis.dependencies}"
    
    # Check generated files
    assert (temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/hub.json").exists(), "hub.json not generated"
    assert (temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/tests.json").exists(), "tests.json not generated"
    assert (temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/Dockerfile").exists(), "Dockerfile not generated"
    assert (temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/handler.py").exists(), "handler.py not generated"
    
    # Validate hub.json structure
    with open(temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/hub.json") as f:
        hub_data = json.load(f)
        assert "version" in hub_data
        assert "title" in hub_data
        assert "description" in hub_data
        assert "tags" in hub_data
        assert "env" in hub_data
    
    # Validate tests.json structure
    with open(temp_dir / f"{repo['url'].split('/')[-1].replace('.git', '')}/tests.json") as f:
        test_data = json.load(f)
        assert "version" in test_data
        assert "tests" in test_data
        assert isinstance(test_data["tests"], list)
        assert len(test_data["tests"]) > 0

def test_invalid_repository(temp_dir):
    """Test analysis of an invalid repository."""
    with pytest.raises(Exception):
        analyze_repository("https://github.com/invalid/repo.git")

def test_repository_analysis(temp_dir):
    """Test repository analysis functionality."""
    repo = {
        "url": "https://github.com/vllm-project/vllm",
        "type": RepoType.ML_MODEL,
        "languages": ["python"],
        "dependencies": {"torch", "transformers"}
    }
    
    # Analyze repository
    analysis = analyze_repository(repo["url"])
    
    # Generate RunPod files
    from runpod_analyzer.core.metadata import generate_hub_json, save_hub_json
    from runpod_analyzer.core.test_generator import generate_test_payloads, save_tests_json
    from runpod_analyzer.core.dockerfile_generator import generate_dockerfile, save_dockerfile
    from runpod_analyzer.core.handler_generator import generate_handler, save_handler
    
    hub_json = generate_hub_json(temp_dir, analysis)
    save_hub_json(temp_dir, hub_json)
    
    test_cases = generate_test_payloads(temp_dir, analysis.repo_type)
    save_tests_json(temp_dir, test_cases)
    
    dockerfile = generate_dockerfile(temp_dir)
    save_dockerfile(temp_dir, dockerfile)
    
    handler = generate_handler(temp_dir, analysis.repo_type, analysis.dependencies)
    save_handler(temp_dir, handler)
    
    # Check for generated files in temp_dir
    assert (temp_dir / "hub.json").exists()
    assert (temp_dir / "tests.json").exists()
    assert (temp_dir / "Dockerfile").exists()
    assert (temp_dir / "handler.py").exists()
    
    # Validate hub.json structure
    with open(temp_dir / "hub.json") as f:
        hub_data = json.load(f)
        assert "title" in hub_data
        assert "description" in hub_data
        assert "tags" in hub_data
        assert "env" in hub_data 