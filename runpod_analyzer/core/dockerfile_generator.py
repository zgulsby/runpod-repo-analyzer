"""Dockerfile generation functionality."""

import os
from pathlib import Path
from typing import Set, List, Optional

def detect_python_version(repo_path: Path) -> str:
    """Detect Python version from repository files."""
    # Check runtime.txt
    runtime_file = repo_path / "runtime.txt"
    if runtime_file.exists():
        try:
            with open(runtime_file, "r") as f:
                content = f.read().strip()
                if content.startswith("python-"):
                    return content.replace("python-", "")
        except Exception:
            pass
    
    # Check .python-version
    py_version_file = repo_path / ".python-version"
    if py_version_file.exists():
        try:
            with open(py_version_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    
    # Default to Python 3.10
    return "3.10"

def get_system_dependencies(dependencies: Set[str]) -> List[str]:
    """Get required system dependencies based on Python packages."""
    system_deps = set()
    
    # Common system dependencies for ML/DL packages
    if any(dep in dependencies for dep in ['torch', 'tensorflow', 'jax']):
        system_deps.update([
            'build-essential',
            'cuda-toolkit',
            'libcudnn8',
            'nvidia-cuda-toolkit'
        ])
    
    if 'opencv-python' in dependencies:
        system_deps.update([
            'libgl1-mesa-glx',
            'libglib2.0-0'
        ])
    
    if any(dep in dependencies for dep in ['scipy', 'numpy']):
        system_deps.update([
            'libatlas-base-dev',
            'gfortran'
        ])
    
    return sorted(list(system_deps))

def generate_dockerfile(repo_path: Path, provided_dependencies: Optional[Set[str]] = None) -> str:
    """Generate a Dockerfile for the repository.
    
    Args:
        repo_path: Path to the repository
        provided_dependencies: Optional set of dependencies to include
        
    Returns:
        Generated Dockerfile content
    """
    python_version = detect_python_version(repo_path)
    dependencies = set()
    
    # Use provided dependencies if available
    if provided_dependencies:
        dependencies.update(provided_dependencies)
    
    # Collect dependencies from various files if no dependencies were provided
    if not provided_dependencies:
        req_files = ['requirements.txt', 'requirements-dev.txt', 'requirements/prod.txt']
        for req_file in req_files:
            req_path = repo_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                                if pkg:
                                    dependencies.add(pkg.lower())
                except Exception:
                    continue
    
    # Get system dependencies
    system_deps = get_system_dependencies(dependencies)
    
    # Determine base image based on dependencies
    if 'torch' in dependencies or 'pytorch' in dependencies:
        base_image = "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"
    elif 'tensorflow' in dependencies:
        base_image = "tensorflow/tensorflow:2.13.0-gpu"
    elif 'flask' in dependencies or 'fastapi' in dependencies:
        base_image = "python:3.11-slim"
    else:
        base_image = f"python:{python_version}-slim"
    
    # Generate Dockerfile content
    system_deps_str = ' '.join(system_deps)
    
    dockerfile = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y {system_deps_str} && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RUNPOD_DEBUG=1

# Install additional RunPod dependencies
RUN pip install --no-cache-dir runpod

# Set the entry point to the handler
CMD [ "python", "-u", "handler.py" ]"""
    
    return dockerfile

def save_dockerfile(repo_path: Path, dockerfile_content: str) -> None:
    """Save Dockerfile to the repository."""
    dockerfile_path = repo_path / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content) 