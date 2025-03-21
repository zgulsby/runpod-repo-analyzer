"""Core repository analysis functionality."""

import os
import re
import hashlib
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .types import RepositoryAnalysis, RepoType
from .dependency_analyzer import (
    parse_requirements_txt,
    parse_setup_py,
    parse_pyproject_toml,
    parse_package_json,
    parse_cargo_toml,
    parse_go_mod
)
from .dependency_analyzer import analyze_dependencies
from .repo_patterns import match_pattern
from .metadata import generate_hub_json, save_hub_json
from .test_generator import generate_test_payloads, save_tests_json
from .language_detector import detect_languages
from .dockerfile_generator import generate_dockerfile, save_dockerfile
from .handler_generator import generate_handler, save_handler

def get_all_files(repo_path: Path) -> Set[str]:
    """Get all files in the repository, excluding common ignore patterns."""
    ignore_patterns = {
        '.git', '__pycache__', 'node_modules', 'venv',
        'env', '.env', '.venv', '.idea', '.vscode'
    }
    
    files = set()
    for root, dirs, filenames in os.walk(repo_path):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_patterns]
        
        for filename in filenames:
            # Get path relative to repo root
            rel_path = os.path.relpath(os.path.join(root, filename), repo_path)
            files.add(rel_path)
    
    return files

def get_repo_cache_path() -> Path:
    """Get the path to the repository cache directory."""
    cache_dir = Path.home() / '.runpod_analyzer_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cached_repo_path(repo_url: str) -> Optional[Path]:
    """Get the path to a cached repository if it exists."""
    # Create a unique identifier for the repo
    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    cache_path = get_repo_cache_path() / repo_hash
    
    if cache_path.exists():
        # Check if the cache is fresh (less than 24 hours old)
        cache_time = cache_path.stat().st_mtime
        if (time.time() - cache_time) < 86400:  # 24 hours in seconds
            return cache_path
    
    return None

def setup_sparse_checkout(repo_path: Path) -> None:
    """Set up sparse checkout for the repository."""
    try:
        # Initialize sparse checkout without cone mode
        subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=repo_path, check=True)
        
        # Define patterns for files to include - more inclusive now
        patterns = [
            "/*",  # All files in root
            "**/*.py",  # All Python files
            "**/requirements*.txt",  # All requirements files
            "**/setup.py",  # All setup files
            "**/pyproject.toml",  # All pyproject files
            "**/package.json",  # Node.js package files
            "**/package-lock.json",  # Node.js lock files
            "**/yarn.lock",  # Yarn lock files
            "**/Cargo.toml",  # Rust package files
            "**/Cargo.lock",  # Rust lock files
            "**/go.mod",  # Go module files
            "**/go.sum",  # Go checksum files
            "**/README*",  # All README files
            "**/*.md",  # All markdown files
            ".git/*",  # Git metadata
            "**/Dockerfile*",  # All Dockerfile variants
            "**/*.env*",  # Environment files
            "**/config.*",  # Config files
            "**/manifest.*",  # Manifest files
            "**/composer.json",  # PHP package files
            "**/composer.lock",  # PHP lock files
            "**/Gemfile",  # Ruby package files
            "**/Gemfile.lock",  # Ruby lock files
            "**/pom.xml",  # Maven package files
            "**/build.gradle",  # Gradle package files
            "**/settings.gradle",  # Gradle settings files
        ]
        
        # Write patterns to sparse-checkout file
        sparse_checkout_file = repo_path / ".git" / "info" / "sparse-checkout"
        sparse_checkout_file.parent.mkdir(exist_ok=True)
        with open(sparse_checkout_file, "w") as f:
            f.write("\n".join(patterns))
        
    except subprocess.CalledProcessError:
        pass  # Ignore errors in sparse checkout setup

def get_default_branch(repo_url: str) -> str:
    """Get the default branch name for a repository."""
    try:
        result = subprocess.run(
            ['git', 'ls-remote', '--symref', repo_url, 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        match = re.search(r'ref: refs/heads/(\S+)\s+HEAD', result.stdout)
        if match:
            return match.group(1)
    except subprocess.CalledProcessError:
        pass
    
    # Try common branch names
    for branch in ['main', 'master', 'develop']:
        try:
            result = subprocess.run(
                ['git', 'ls-remote', repo_url, f'refs/heads/{branch}'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                return branch
        except subprocess.CalledProcessError:
            continue
    
    return 'main'  # Default to main if we can't detect

def clone_or_update_repo(repo_url: str, repo_path: Path) -> None:
    """Clone or update a repository."""
    # Create parent directory if it doesn't exist
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing directory if it exists
    if repo_path.exists():
        shutil.rmtree(repo_path)
    
    # Clone the repository
    subprocess.run(['git', 'init', str(repo_path)], check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=repo_path, check=True)
    subprocess.run(['git', 'fetch', '--depth', '1', 'origin'], cwd=repo_path, check=True)
    
    # Get default branch name
    result = subprocess.run(
        ['git', 'remote', 'show', 'origin'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    default_branch = 'main'
    for line in result.stdout.split('\n'):
        if 'HEAD branch:' in line:
            default_branch = line.split(':')[1].strip()
            break
    
    # Create and checkout branch
    subprocess.run(
        ['git', 'checkout', '-b', default_branch, f'origin/{default_branch}'],
        cwd=repo_path,
        check=True
    )

def detect_repository_type(repo_path: Path) -> Tuple[RepoType, float]:
    """Detect repository type and confidence score."""
    confidence = 0.0
    repo_type = RepoType.UNKNOWN
    
    # Check for ML infrastructure indicators
    infra_indicators = {
        'inference': 5.0,
        'serve': 4.0,
        'server': 3.0,
        'performance': 3.0,
        'throughput': 3.0,
        'latency': 3.0,
        'cuda': 4.0,
        'gpu': 4.0,
        'parallel': 3.0,
        'distributed': 4.0,
        'scale': 3.0,
        'optimization': 3.0,
        'memory': 3.0,
        'cache': 2.0,
        'pipeline': 2.0,
        'deployment': 2.0,
        'monitoring': 2.0,
        'metrics': 2.0
    }
    
    # Check for ML model indicators
    ml_indicators = {
        'model': 3.0,
        'train': 2.0,
        'predict': 2.0,
        'dataset': 2.0,
        'neural': 2.0,
        'deep learning': 3.0,
        'machine learning': 3.0
    }
    
    # Check for API indicators
    api_indicators = {
        'api': 4.0,
        'endpoint': 3.0,
        'route': 2.0,
        'rest': 3.0,
        'graphql': 4.0,
        'http': 2.0,
        'request': 1.0,
        'response': 1.0
    }
    
    # Read README content
    readme_path = repo_path / "README.md"
    readme_content = ""
    if readme_path.exists():
        with open(readme_path) as f:
            readme_content = f.read().lower()
    
    # Calculate infrastructure confidence
    infra_confidence = sum(
        score for term, score in infra_indicators.items()
        if term in readme_content
    )
    
    # Calculate ML model confidence
    ml_confidence = sum(
        score for term, score in ml_indicators.items()
        if term in readme_content
    )
    
    # Calculate API confidence
    api_confidence = sum(
        score for term, score in api_indicators.items()
        if term in readme_content
    )
    
    # Check dependencies
    requirements_txt = repo_path / "requirements.txt"
    if requirements_txt.exists():
        with open(requirements_txt) as f:
            requirements = f.read().lower()
            # Infrastructure packages
            if any(pkg in requirements for pkg in ['vllm', 'triton', 'ray', 'tensorrt', 'onnx', 'cuda', 'cupy']):
                infra_confidence += 15.0
            # ML packages
            if any(pkg in requirements for pkg in ['torch', 'tensorflow', 'keras', 'sklearn']):
                ml_confidence += 8.0
            # API packages
            if any(pkg in requirements for pkg in ['flask', 'fastapi', 'django', 'express']):
                api_confidence += 8.0
    
    # Check for C++ code which is common in ML infrastructure
    cpp_files = list(repo_path.glob('**/*.cpp')) + list(repo_path.glob('**/*.cu'))
    if cpp_files:
        infra_confidence += 10.0
    
    # Check for CUDA kernels
    cuda_files = list(repo_path.glob('**/*.cu')) + list(repo_path.glob('**/*.cuh'))
    if cuda_files:
        infra_confidence += 12.0
    
    # Determine repository type based on confidence scores
    max_confidence = max(infra_confidence, ml_confidence, api_confidence)
    
    if max_confidence > 5.0:
        if infra_confidence == max_confidence:
            repo_type = RepoType.ML_INFRASTRUCTURE
            confidence = min(infra_confidence * 2.0, 100.0)  # Scale up infrastructure confidence
        elif ml_confidence == max_confidence:
            repo_type = RepoType.ML_MODEL
            confidence = min(ml_confidence * 2.0, 100.0)
        elif api_confidence == max_confidence:
            repo_type = RepoType.API
            confidence = min(api_confidence * 2.0, 100.0)
    
    return repo_type, confidence

def detect_languages(repo_path: Path) -> Dict[str, float]:
    """Detect programming languages used in the repository with confidence scores."""
    languages = {}
    
    # Count files by extension
    extension_counts = {}
    total_files = 0
    
    for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
        count = len(list(repo_path.glob(f'**/*{ext}')))
        if count > 0:
            extension_counts[ext] = count
            total_files += count
    
    if total_files == 0:
        return {}
    
    # Calculate confidence scores based on file counts
    if extension_counts.get('.py', 0) > 0:
        languages['python'] = extension_counts['.py'] / total_files
    
    js_count = extension_counts.get('.js', 0)
    ts_count = extension_counts.get('.ts', 0)
    if js_count > 0 or ts_count > 0:
        languages['javascript'] = (js_count + ts_count) / total_files
        if ts_count > 0:
            languages['typescript'] = ts_count / total_files
    
    if extension_counts.get('.java', 0) > 0:
        languages['java'] = extension_counts['.java'] / total_files
    
    cpp_count = extension_counts.get('.cpp', 0)
    c_count = extension_counts.get('.c', 0)
    if cpp_count > 0 or c_count > 0:
        languages['c++'] = (cpp_count + c_count) / total_files
    
    return languages

def clean_package_name(pkg: str) -> str:
    """Clean a package name by removing quotes, version specifiers, and extras."""
    # Remove leading/trailing whitespace and quotes
    pkg = pkg.strip()
    pkg = pkg.strip("'").strip('"')
    pkg = pkg.strip()
    
    # Skip empty lines and comments
    if not pkg or pkg.startswith('#'):
        return ''
    
    # Remove version specifiers and extras
    for spec in ['==', '>=', '<=', '>', '<', '~=', '!=']:
        if spec in pkg:
            pkg = pkg.split(spec)[0]
    
    # Remove extras
    if '[' in pkg:
        pkg = pkg.split('[')[0]
    
    # Remove trailing comma and whitespace
    pkg = pkg.rstrip(',').strip()
    
    return pkg.lower()

def get_dependencies(repo_path: Path) -> Set[str]:
    """Get dependencies from various package management files."""
    dependencies = set()
    
    # Check requirements.txt
    requirements_txt = repo_path / "requirements.txt"
    if requirements_txt.exists():
        with open(requirements_txt) as f:
            for line in f:
                pkg = clean_package_name(line)
                if pkg:
                    dependencies.add(pkg)
    
    # Check setup.py
    setup_py = repo_path / "setup.py"
    if setup_py.exists():
        with open(setup_py) as f:
            content = f.read()
            # Look for install_requires list
            import re
            install_requires = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if install_requires:
                # Split by commas and newlines to handle both single-line and multi-line lists
                deps = re.split(r'[,\n]', install_requires.group(1))
                for dep in deps:
                    pkg = clean_package_name(dep)
                    if pkg:
                        dependencies.add(pkg)
    
    # Check pyproject.toml
    pyproject_toml = repo_path / "pyproject.toml"
    if pyproject_toml.exists():
        try:
            import toml
            with open(pyproject_toml) as f:
                data = toml.load(f)
                if 'tool' in data and 'poetry' in data['tool']:
                    deps = data['tool']['poetry'].get('dependencies', {})
                    for pkg in deps:
                        if pkg != 'python':
                            dependencies.add(pkg.lower())
        except:
            pass
    
    return dependencies

def clone_repository(repo_url: str, repo_path: Path) -> None:
    """Clone a repository to the specified path."""
    try:
        # Remove directory if it exists
        if repo_path.exists():
            shutil.rmtree(repo_path)
        
        # Clone the repository
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")

def analyze_characteristics(repo_path: Path) -> Dict[str, Any]:
    """Analyze repository characteristics."""
    characteristics = {
        'files': get_all_files(repo_path),
        'has_setup_py': 'setup.py' in get_all_files(repo_path),
        'has_requirements_txt': 'requirements.txt' in get_all_files(repo_path),
        'has_pyproject_toml': 'pyproject.toml' in get_all_files(repo_path),
        'has_readme': any(f.lower().startswith('readme.') for f in get_all_files(repo_path)),
        'has_dockerfile': 'Dockerfile' in get_all_files(repo_path),
        'has_license': any(f.lower().startswith('license') for f in get_all_files(repo_path)),
    }
    
    # Check for ML-related files
    ml_patterns = {
        'model', 'train', 'predict', 'inference',
        'pytorch', 'tensorflow', 'keras', 'torch',
        'dataset', 'neural', 'network', 'deep',
        'machine', 'learning', 'ai', 'artificial'
    }
    
    # Check for infrastructure-related files
    infra_patterns = {
        'server', 'service', 'api', 'endpoint',
        'docker', 'kubernetes', 'k8s', 'helm',
        'deploy', 'scale', 'monitor', 'metrics',
        'prometheus', 'grafana', 'logging'
    }
    
    # Check for web-related files
    web_patterns = {
        'web', 'html', 'css', 'js', 'javascript',
        'react', 'vue', 'angular', 'node', 'express',
        'flask', 'django', 'fastapi'
    }
    
    # Count pattern matches in filenames
    for file in characteristics['files']:
        file_lower = file.lower()
        for pattern in ml_patterns:
            if pattern in file_lower:
                characteristics['has_ml'] = True
                break
        for pattern in infra_patterns:
            if pattern in file_lower:
                characteristics['has_infrastructure'] = True
                break
        for pattern in web_patterns:
            if pattern in file_lower:
                characteristics['has_web'] = True
                break
    
    return characteristics 