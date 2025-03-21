"""Repository pattern detection and classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import os
import fnmatch
from .types import RepoType, RepositoryAnalysis
from .language_detector import detect_languages
from .dependency_analyzer import analyze_dependencies_across_languages
import tempfile
import subprocess
import shutil
import logging
import traceback
import time
import re

# Configure logging with more detailed format
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class RepoCharacteristics:
    """Characteristics of a repository."""
    has_web: bool = False
    has_api: bool = False
    has_ml: bool = False
    has_cli: bool = False
    has_docs: bool = False
    has_tests: bool = False
    has_docker: bool = False
    has_ci: bool = False
    has_deploy: bool = False
    
    @property
    def confidence_threshold(self) -> float:
        """Calculate confidence threshold based on characteristics."""
        # More characteristics mean we need higher confidence
        base_threshold = 0.4
        characteristic_count = sum(1 for v in vars(self).values() if v)
        return base_threshold + (characteristic_count * 0.05)

def match_pattern(repo_path: Path, languages: Dict[str, float], dependencies: Set[str]) -> Tuple[RepoType, float]:
    """Match repository patterns to determine type and confidence score."""
    scores = {
        RepoType.ML_MODEL: 0.0,
        RepoType.ML_APPLICATION: 0.0,
        RepoType.ML_FRAMEWORK: 0.0,
        RepoType.API: 0.0,
        RepoType.LIBRARY: 0.0,
        RepoType.LANGUAGE_COMPILER: 0.0,
        RepoType.ML_TOOL: 0.0
    }

    # Core ML dependencies that indicate ML-related repository
    ml_model_deps = {
        'torch', 'tensorflow', 'keras', 'sklearn', 'xgboost', 'lightgbm',
        'transformers', 'diffusers', 'timm', 'torchvision', 'detectron2'
    }
    
    ml_app_deps = {
        'gradio', 'streamlit', 'dash', 'plotly', 'panel', 'voila',
        'ipywidgets', 'flask', 'fastapi', 'django'
    }
    
    ml_framework_deps = {
        'vllm', 'deepspeed', 'accelerate', 'ray', 'horovod', 'lightning',
        'pytorch_lightning', 'jax', 'flax', 'optax', 'triton', 'onnx',
        'tensorrt', 'tvm', 'mxnet'
    }
    
    api_deps = {
        'fastapi', 'flask', 'django', 'express', 'graphql', 'grpc',
        'aiohttp', 'sanic', 'tornado', 'starlette', 'uvicorn'
    }
    
    library_deps = {
        'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib',
        'seaborn', 'requests', 'beautifulsoup4', 'pillow', 'opencv-python'
    }

    # Calculate base scores from dependencies
    for dep in dependencies:
        if dep in ml_model_deps:
            scores[RepoType.ML_MODEL] += 0.3
        if dep in ml_app_deps:
            scores[RepoType.ML_APPLICATION] += 0.3
        if dep in ml_framework_deps:
            scores[RepoType.ML_FRAMEWORK] += 0.4
        if dep in api_deps:
            scores[RepoType.API] += 0.3
        if dep in library_deps:
            scores[RepoType.LIBRARY] += 0.2

    # Adjust scores based on language distribution
    if languages.get('python', 0) > 0.7:
        scores[RepoType.ML_MODEL] *= 1.2
        scores[RepoType.ML_FRAMEWORK] *= 1.2
    if languages.get('javascript', 0) > 0.3 or languages.get('typescript', 0) > 0.3:
        scores[RepoType.ML_APPLICATION] *= 1.3
        scores[RepoType.API] *= 1.2
    if languages.get('c', 0) > 0.2 or languages.get('cpp', 0) > 0.2:
        scores[RepoType.ML_FRAMEWORK] *= 1.3
        scores[RepoType.LANGUAGE_COMPILER] *= 1.4

    # Check for name-based heuristics
    repo_name = repo_path.name.lower()
    name_heuristics = {
        'model': (RepoType.ML_MODEL, 0.3),
        'diffusion': (RepoType.ML_MODEL, 0.3),
        'bert': (RepoType.ML_MODEL, 0.3),
        'gpt': (RepoType.ML_MODEL, 0.3),
        'app': (RepoType.ML_APPLICATION, 0.3),
        'ui': (RepoType.ML_APPLICATION, 0.3),
        'web': (RepoType.ML_APPLICATION, 0.3),
        'webui': (RepoType.ML_APPLICATION, 0.4),
        'framework': (RepoType.ML_FRAMEWORK, 0.4),
        'engine': (RepoType.ML_FRAMEWORK, 0.3),
        'api': (RepoType.API, 0.3),
        'service': (RepoType.API, 0.2),
        'lib': (RepoType.LIBRARY, 0.2),
        'sdk': (RepoType.LIBRARY, 0.2),
        'compiler': (RepoType.LANGUAGE_COMPILER, 0.4),
        'tool': (RepoType.ML_TOOL, 0.3)
    }

    # Log heuristic matches for debugging
    for term, (repo_type, boost) in name_heuristics.items():
        if term in repo_name:
            scores[repo_type] += boost
            logger.info(f"Found heuristic term '{term}' in repository name, suggesting {repo_type.value}")

    # Special case adjustments
    if 'gradio' in dependencies or 'streamlit' in dependencies:
        scores[RepoType.ML_APPLICATION] = max(scores[RepoType.ML_APPLICATION], 0.8)
    if 'vllm' in dependencies or 'deepspeed' in dependencies:
        scores[RepoType.ML_FRAMEWORK] = max(scores[RepoType.ML_FRAMEWORK], 0.8)
    if 'transformers' in dependencies and languages.get('python', 0) > 0.8:
        scores[RepoType.ML_MODEL] = max(scores[RepoType.ML_MODEL], 0.7)

    # Log classification scores for debugging
    logger.info(f"Classification scores: " + ", ".join([f"{k.value}={v:.2f}" for k, v in scores.items()]))

    # Determine the most likely type
    max_score = max(scores.values())
    if max_score < 0.3:
        return RepoType.UNKNOWN, 0.5

    # Get the type with the highest score
    repo_type = max(scores.items(), key=lambda x: x[1])[0]
    
    # Log initial classification
    logger.info(f"Initial type classification: {repo_type} (confidence: {max_score})")

    # Apply final adjustments based on combined factors
    confidence = max_score
    
    # Boost confidence if multiple indicators align
    if repo_type == RepoType.ML_APPLICATION:
        if any(term in repo_name for term in ['ui', 'web', 'app']) and languages.get('javascript', 0) > 0.1:
            confidence = min(confidence * 1.1, 0.99)
            logger.info(f"Boosted ml_application confidence based on name: {confidence:.2f}")
    elif repo_type == RepoType.API:
        if 'api' in repo_name:
            confidence = min(confidence * 1.1, 0.99)
            logger.info(f"Boosted api confidence based on name: {confidence:.2f}")
    elif repo_type == RepoType.ML_FRAMEWORK:
        if languages.get('cpp', 0) > 0.1 and languages.get('python', 0) > 0.5:
            confidence = min(confidence * 1.1, 0.99)
            logger.info(f"Boosted ml_framework confidence based on language mix: {confidence:.2f}")

    logger.info(f"Determined repository type: {repo_type} (confidence: {confidence})")
    return repo_type, confidence

def clone_repository(url: str, target_dir: str) -> None:
    """Clone a git repository to the target directory."""
    try:
        logger.info(f"Cloning repository {url} to {target_dir}")
        
        # Remove existing directory if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # Clone with timeout
        process = subprocess.run(
            ["git", "clone", "--depth", "1", url, target_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if process.returncode != 0:
            error_msg = process.stderr.strip()
            logger.error(f"Git clone failed: {error_msg}")
            raise RuntimeError(f"Failed to clone repository: {error_msg}")
            
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        raise RuntimeError("Repository clone timed out")
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to clone repository: {str(e)}")

def get_cached_repo_path(repo_url: str) -> Optional[Path]:
    """
    Check if a repository has already been cloned to a cache directory.
    
    Args:
        repo_url: URL of the GitHub repository
        
    Returns:
        Path to the cached repository if it exists, None otherwise
    """
    # For now, just return None since we don't have cache logic implemented
    return None

def get_all_files(repo_path: Path) -> List[Path]:
    """Get all files in a repository directory, excluding common directories to ignore.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of file paths
    """
    all_files = []
    ignore_dirs = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', '.pytest_cache'}
    
    for root, dirs, files in os.walk(repo_path):
        # Remove directories to ignore
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # Add all files in the current directory
        for file in files:
            file_path = Path(root) / file
            all_files.append(file_path)
    
    return all_files

def clone_or_update_repo(repo_url: str, target_dir: Path) -> None:
    """Clone a repository or update it if it already exists.
    
    This function serves as a bridge between the high-level analysis code and
    the low-level clone_repository function. It's designed to make it easier
    to add repository caching or other optimization in the future without
    changing the interface used by analyze_repository.
    
    Args:
        repo_url: URL of the repository to clone
        target_dir: Directory to clone the repository to
    """
    # For now, just call clone_repository
    clone_repository(repo_url, str(target_dir))

def analyze_repository(repo_url: str, clone_path: Optional[str] = None) -> RepositoryAnalysis:
    """
    Analyze a GitHub repository and return insights about its type and structure.
    
    Args:
        repo_url: URL of the GitHub repository to analyze
        clone_path: Optional path to clone the repository to
        
    Returns:
        RepositoryAnalysis object with insights about the repository
    """
    temp_dir = None
    source_dir = None
    
    try:
        if not repo_url:
            raise ValueError("Repository URL cannot be empty")
            
        logging.info(f"Analyzing repository {repo_url}")
        
        # If clone_path is provided, use it as the source directory
        if clone_path:
            source_dir = clone_path
        else:
            # Otherwise, use a cache directory if available
            cached_path = get_cached_repo_path(repo_url)
            if cached_path:
                logging.info(f"Using cached repository at {cached_path}")
                source_dir = str(cached_path)
            else:
                # Create a temporary directory to clone the repository
                temp_dir = tempfile.mkdtemp(prefix="runpod_analyzer_")
                source_dir = temp_dir
                logging.info(f"Cloning repository {repo_url} to {source_dir}")
                
                try:
                    clone_or_update_repo(repo_url, Path(source_dir))
                except Exception as e:
                    raise RuntimeError(f"Failed to clone repository: {str(e)}")
                    
        # Get all files in the repository
        repo_files = get_all_files(Path(source_dir))
        
        # Detect programming languages
        languages = detect_languages(str(Path(source_dir)))
        
        # Analyze dependencies
        dependencies = analyze_dependencies(Path(source_dir))
        
        # Match repository patterns to determine type and other properties
        repo_type, confidence = match_pattern(Path(source_dir), languages, dependencies)
        
        # Create the analysis result
        analysis = RepositoryAnalysis(
            repo_type=repo_type,
            confidence=confidence,
            languages=languages,
            dependencies=dependencies,
            source_dir=source_dir,
            repo_url=repo_url  # Store the repository URL in the analysis object
        )
        
        return analysis
    finally:
        # Clean up the temporary directory if we created it ourselves
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")

def _find_imports_in_code(repo_path: str) -> Set[str]:
    """Analyze Python code to find imports directly from files."""
    imports = set()
    try:
        # Find Python files
        python_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
                    
        # Sample up to 50 files for faster analysis
        if len(python_files) > 50:
            python_files = python_files[:50]
            
        # Common ML dependencies to look for
        ml_deps = {
            'torch', 'tensorflow', 'keras', 'sklearn', 'transformers', 'huggingface',
            'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'streamlit', 'gradio',
            'fastapi', 'flask', 'django', 'ray', 'triton', 'onnx', 'jax', 'tqdm',
            'cv2', 'opencv', 'PIL', 'pillow', 'openai', 'langchain'
        }
        
        # Analyze each Python file
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for import statements
                    import_lines = [line.strip() for line in content.split('\n') 
                                   if line.strip().startswith(('import ', 'from ')) 
                                   and not line.strip().startswith('from .')]
                    
                    for line in import_lines:
                        parts = line.split()
                        if line.startswith('import '):
                            # Handle 'import torch' or 'import torch as t'
                            module = parts[1].split('.')[0].split(',')[0].strip()
                            if module.lower() in ml_deps:
                                logger.info(f"Found ML dependency '{module}' in code imports")
                                imports.add(module)
                        elif line.startswith('from '):
                            # Handle 'from torch import nn' - get the 'torch' part
                            if len(parts) > 1:
                                module = parts[1].split('.')[0].strip()
                                if module.lower() in ml_deps:
                                    logger.info(f"Found ML dependency '{module}' in code imports")
                                    imports.add(module)
            except Exception as e:
                logger.warning(f"Error analyzing imports in {py_file}: {str(e)}")
                continue
                
    except Exception as e:
        logger.warning(f"Error during code import analysis: {str(e)}")
        
    return imports

def _analyze_structure(repo_path: Path) -> Dict[str, float]:
    """Analyze repository structure to determine type."""
    scores = {
        'infrastructure': 0.0,
        'model': 0.0,
        'framework': 0.0
    }
    
    # Infrastructure indicators
    infra_dirs = {'core', 'engine', 'server', 'worker', 'scheduler', 'router', 'backend'}
    infra_files = {'Dockerfile', 'docker-compose.yml', 'kubernetes', 'config.yaml'}
    
    # Model indicators
    model_dirs = {'model', 'models', 'weights', 'checkpoints', 'pretrained'}
    model_files = {'model.py', 'inference.py', 'training.py', 'config.json'}
    
    # Framework indicators
    framework_dirs = {'framework', 'lib', 'utils', 'tools', 'extensions'}
    framework_files = {'setup.py', 'pyproject.toml', 'requirements.txt'}
    
    # Analyze directory structure
    for item in repo_path.iterdir():
        if item.is_dir():
            name = item.name.lower()
            if name in infra_dirs:
                scores['infrastructure'] += 0.3
            elif name in model_dirs:
                scores['model'] += 0.3
            elif name in framework_dirs:
                scores['framework'] += 0.3
    
    # Analyze file structure
    for item in repo_path.rglob('*'):
        if item.is_file():
            name = item.name.lower()
            if name in infra_files:
                scores['infrastructure'] += 0.2
            elif name in model_files:
                scores['model'] += 0.2
            elif name in framework_files:
                scores['framework'] += 0.2
    
    return scores

def _analyze_code(repo_path: Path) -> Dict[str, float]:
    """Analyze code structure and patterns to determine type."""
    scores = {
        'infrastructure': 0.0,
        'model': 0.0,
        'framework': 0.0
    }
    
    # Infrastructure patterns
    infra_patterns = {
        'class': ['Engine', 'Server', 'Worker', 'Scheduler', 'Router', 'Backend'],
        'function': ['serve', 'deploy', 'scale', 'monitor', 'route', 'schedule'],
        'import': ['fastapi', 'uvicorn', 'grpc', 'docker', 'kubernetes']
    }
    
    # Model patterns
    model_patterns = {
        'class': ['Model', 'Inference', 'Training', 'Checkpoint', 'Weights'],
        'function': ['inference', 'train', 'predict', 'generate', 'forward'],
        'import': ['torch', 'tensorflow', 'transformers', 'diffusers']
    }
    
    # Framework patterns
    framework_patterns = {
        'class': ['Framework', 'Library', 'Utils', 'Tools', 'Extensions'],
        'function': ['register', 'configure', 'extend', 'customize', 'plugin'],
        'import': ['setuptools', 'wheel', 'ninja', 'cmake']
    }
    
    # Analyze Python files
    for py_file in repo_path.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
                # Check for class definitions
                for pattern_type, patterns in infra_patterns.items():
                    for pattern in patterns:
                        if f'class {pattern}' in content:
                            scores['infrastructure'] += 0.2
                
                for pattern_type, patterns in model_patterns.items():
                    for pattern in patterns:
                        if f'class {pattern}' in content:
                            scores['model'] += 0.2
                
                for pattern_type, patterns in framework_patterns.items():
                    for pattern in patterns:
                        if f'class {pattern}' in content:
                            scores['framework'] += 0.2
                
                # Check for function definitions
                for pattern_type, patterns in infra_patterns.items():
                    for pattern in patterns:
                        if f'def {pattern}' in content:
                            scores['infrastructure'] += 0.1
                
                for pattern_type, patterns in model_patterns.items():
                    for pattern in patterns:
                        if f'def {pattern}' in content:
                            scores['model'] += 0.1
                
                for pattern_type, patterns in framework_patterns.items():
                    for pattern in patterns:
                        if f'def {pattern}' in content:
                            scores['framework'] += 0.1
                
                # Check for imports
                for pattern_type, patterns in infra_patterns.items():
                    for pattern in patterns:
                        if f'import {pattern}' in content:
                            scores['infrastructure'] += 0.1
                
                for pattern_type, patterns in model_patterns.items():
                    for pattern in patterns:
                        if f'import {pattern}' in content:
                            scores['model'] += 0.1
                
                for pattern_type, patterns in framework_patterns.items():
                    for pattern in patterns:
                        if f'import {pattern}' in content:
                            scores['framework'] += 0.1
        except Exception:
            continue
    
    return scores

def _analyze_documentation(repo_path: Path) -> Dict[str, float]:
    """Analyze documentation to determine type."""
    scores = {
        'infrastructure': 0.0,
        'model': 0.0,
        'framework': 0.0
    }
    
    # Infrastructure keywords
    infra_keywords = {
        'serve', 'deploy', 'scale', 'monitor', 'route', 'schedule',
        'server', 'worker', 'backend', 'infrastructure', 'engine',
        'kubernetes', 'docker', 'api', 'endpoint', 'service'
    }
    
    # Model keywords
    model_keywords = {
        'model', 'inference', 'training', 'predict', 'generate',
        'weights', 'checkpoint', 'pretrained', 'fine-tune', 'train'
    }
    
    # Framework keywords
    framework_keywords = {
        'framework', 'library', 'tool', 'extension', 'plugin',
        'customize', 'configure', 'extend', 'register'
    }
    
    # Analyze README and documentation
    for doc_file in repo_path.rglob('*.md'):
        try:
            with open(doc_file, 'r') as f:
                content = f.read().lower()
                
                # Count keyword occurrences
                for keyword in infra_keywords:
                    if keyword in content:
                        scores['infrastructure'] += 0.1
                
                for keyword in model_keywords:
                    if keyword in content:
                        scores['model'] += 0.1
                
                for keyword in framework_keywords:
                    if keyword in content:
                        scores['framework'] += 0.1
        except Exception:
            continue
    
    return scores

def _analyze_metadata(repo_path: Path) -> Dict[str, float]:
    """Analyze project metadata to determine type."""
    scores = {
        'infrastructure': 0.0,
        'model': 0.0,
        'framework': 0.0
    }
    
    # Infrastructure topics
    infra_topics = {
        'serving', 'deployment', 'scaling', 'monitoring', 'routing',
        'server', 'worker', 'backend', 'infrastructure', 'engine',
        'kubernetes', 'docker', 'api', 'endpoint', 'service'
    }
    
    # Model topics
    model_topics = {
        'model', 'inference', 'training', 'prediction', 'generation',
        'weights', 'checkpoint', 'pretrained', 'fine-tuning', 'training'
    }
    
    # Framework topics
    framework_topics = {
        'framework', 'library', 'tool', 'extension', 'plugin',
        'customization', 'configuration', 'extension', 'registration'
    }
    
    # Check for topics in various metadata files
    for metadata_file in repo_path.glob('*'):
        if metadata_file.name.lower() in {'topics.json', 'metadata.json', 'package.json'}:
            try:
                with open(metadata_file, 'r') as f:
                    content = f.read().lower()
                    
                    # Count topic occurrences
                    for topic in infra_topics:
                        if topic in content:
                            scores['infrastructure'] += 0.2
                    
                    for topic in model_topics:
                        if topic in content:
                            scores['model'] += 0.2
                    
                    for topic in framework_topics:
                        if topic in content:
                            scores['framework'] += 0.2
            except Exception:
                continue
    
    return scores

def _check_file_exists(repo_path: str, file_pattern: str) -> bool:
    """Check if a file matching the pattern exists in the repository."""
    # First try exact match
    if Path(repo_path).glob(f"**/{file_pattern}"):
        return True
    
    # Then try pattern matching
    for root, _, files in os.walk(repo_path):
        for file in files:
            if fnmatch.fnmatch(file.lower(), file_pattern.lower()):
                return True
    return False

def _check_file_pattern(repo_path: str, pattern: str) -> bool:
    """Check if any files match the given pattern in the repository."""
    # Handle directory patterns (e.g., kubernetes/*.yaml)
    if '/' in pattern:
        dir_part, file_part = pattern.split('/', 1)
        # Check if the directory exists
        dir_path = Path(repo_path) / dir_part
        if not dir_path.exists():
            return False
        # Check for files matching the pattern in the directory
        for root, _, files in os.walk(dir_path):
            for file in files:
                if fnmatch.fnmatch(file.lower(), file_part.lower()):
                    return True
        return False
    
    # Handle simple file patterns (e.g., *.cu)
    for root, _, files in os.walk(repo_path):
        for file in files:
            if fnmatch.fnmatch(file.lower(), pattern.lower()):
                return True
    return False

def _check_indicators(repo_path: str, indicators: set, patterns: set) -> bool:
    """Check if any indicators or patterns are present in the repository."""
    # Check for indicators (exact file names or simple patterns)
    if any(_check_file_exists(repo_path, i) for i in indicators):
        return True
    
    # Check for more complex patterns (e.g., directory patterns)
    if any(_check_file_pattern(repo_path, p) for p in patterns):
        return True
    
    return False

def _has_matching_deps(dependencies: Dict[str, List[str]], target_deps: set) -> bool:
    """Check if any of the target dependencies are present."""
    all_deps = set()
    for deps in dependencies.values():
        all_deps.update(deps)
    return bool(all_deps.intersection(target_deps))

def analyze_dependencies(repo_path: Path) -> Set[str]:
    """Analyze the repository to find dependencies.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Set of dependency names
    """
    dependencies = set()
    
    # Check for Python dependencies
    req_files = ['requirements.txt', 'requirements-dev.txt', 'requirements/prod.txt']
    for req_file in req_files:
        req_path = repo_path / req_file
        if req_path.exists():
            try:
                with open(req_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name, ignoring version specifiers
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('[')[0].strip()
                            if pkg:
                                dependencies.add(pkg.lower())
            except Exception:
                pass
    
    # Check for setup.py
    setup_py = repo_path / 'setup.py'
    if setup_py.exists():
        try:
            with open(setup_py, 'r') as f:
                content = f.read()
                # Look for install_requires list
                match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    deps_str = match.group(1)
                    # Parse the string to extract package names
                    for line in deps_str.split('\n'):
                        line = line.strip().strip(',\'\"')
                        if line and not line.startswith('#'):
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('[')[0].strip()
                            if pkg:
                                dependencies.add(pkg.lower())
        except Exception:
            pass
    
    # Check for package.json (Node.js)
    package_json = repo_path / 'package.json'
    if package_json.exists():
        try:
            import json
            with open(package_json, 'r') as f:
                data = json.load(f)
                # Get dependencies and devDependencies
                for dep_type in ['dependencies', 'devDependencies']:
                    if dep_type in data:
                        for pkg in data[dep_type]:
                            dependencies.add(pkg.lower())
        except Exception:
            pass
    
    # Also look for imports directly in Python files
    imports = _find_imports_in_code(str(repo_path))
    dependencies.update(imports)
    
    return dependencies 