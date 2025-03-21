from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional, Set
from .types import RepoType
import os

@dataclass
class TestCase:
    """Represents a test case in tests.json"""
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

def extract_function_params(content: str) -> List[Dict[str, Any]]:
    """Extract function parameters from Python code"""
    params = []
    # Look for function definitions
    func_matches = re.finditer(r'def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:', content)
    
    for match in func_matches:
        func_name = match.group(1)
        if func_name == 'handler':  # Focus on handler functions
            param_str = match.group(2)
            # Parse parameters
            param_parts = [p.strip() for p in param_str.split(',') if p.strip()]
            for param in param_parts:
                # Handle type hints
                if ':' in param:
                    name, type_hint = param.split(':', 1)
                    name = name.strip()
                    type_hint = type_hint.strip()
                    # Convert Python type hints to JSON schema types
                    if 'str' in type_hint:
                        params.append({"name": name, "type": "string"})
                    elif 'int' in type_hint:
                        params.append({"name": name, "type": "integer"})
                    elif 'float' in type_hint:
                        params.append({"name": name, "type": "number"})
                    elif 'bool' in type_hint:
                        params.append({"name": name, "type": "boolean"})
                    elif 'List' in type_hint or 'list' in type_hint:
                        params.append({"name": name, "type": "array"})
                    elif 'Dict' in type_hint or 'dict' in type_hint:
                        params.append({"name": name, "type": "object"})
                    else:
                        params.append({"name": name, "type": "any"})
                else:
                    # No type hint, assume any
                    params.append({"name": param, "type": "any"})
    
    return params

def extract_api_endpoints(repo_path: Path) -> List[Dict[str, Any]]:
    """Extract API endpoints and their parameters from common API frameworks"""
    endpoints = []
    
    # Look for FastAPI/Flask routes
    for py_file in repo_path.rglob('*.py'):
        with open(py_file, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                # FastAPI endpoints
                route_matches = re.finditer(r'@(?:app|router)\.(?:get|post|put|delete)\([\'"]([^\'"]+)[\'"]\)', content)
                for match in route_matches:
                    route = match.group(1)
                    # Find the function definition following the decorator
                    func_def = content[match.end():].split('\n')[0]
                    params = extract_function_params(func_def)
                    endpoints.append({
                        "path": route,
                        "method": "POST",  # Default to POST for RunPod
                        "parameters": params
                    })
            except UnicodeDecodeError:
                continue
    
    return endpoints

def extract_dependencies_from_file(file_path: str) -> set:
    """Extract dependencies from a Python package file."""
    dependencies = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract from requirements.txt
            if file_path.endswith('.txt'):
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                        dependencies.add(pkg.lower())
            
            # Extract from pyproject.toml
            elif file_path.endswith('.toml'):
                # Look for dependencies section
                dep_section = re.search(r'\[project\]\s*.*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if dep_section:
                    deps = dep_section.group(1)
                    for line in deps.split('\n'):
                        line = line.strip().strip('",\'')
                        if line:
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                            if pkg:
                                dependencies.add(pkg.lower())
                                
            # Extract from setup.py
            elif file_path.endswith('.py'):
                # Look for install_requires list
                install_req = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if install_req:
                    deps = install_req.group(1)
                    for line in deps.split('\n'):
                        line = line.strip().strip('",\'')
                        if line:
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                            if pkg:
                                dependencies.add(pkg.lower())
    except Exception:
        pass
    return dependencies

def extract_imports_from_python_file(file_path: str) -> set:
    """Extract import statements from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Match both 'import package' and 'from package import ...'
            import_lines = re.findall(r'^(?:from\s+([^\s.]+)|import\s+([^\s,]+))', content, re.MULTILINE)
            for from_pkg, import_pkg in import_lines:
                pkg = (from_pkg or import_pkg).lower()
                if pkg and not pkg.startswith('_'):  # Skip internal imports
                    imports.add(pkg)
    except Exception:
        pass
    return imports

def find_package_files(repo_path: str) -> List[str]:
    """Find all Python package files in the repository."""
    package_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file in {'requirements.txt', 'setup.py', 'pyproject.toml'}:
                package_files.append(os.path.join(root, file))
    return package_files

def find_python_files(repo_path: str) -> List[str]:
    """Find all Python files in the repository."""
    python_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def has_dependency(dependencies: set, target_deps: List[str]) -> bool:
    """Check if any of the target dependencies are present (case-insensitive)."""
    # Clean up dependency names
    deps_lower = {
        dep.split('[')[0]  # Remove extras
           .split('==')[0]  # Remove version specs
           .split('>=')[0]
           .split('<=')[0]
           .split('<')[0]
           .split('>')[0]
           .split('@')[0]  # Remove git refs
           .split('#')[0]  # Remove fragments
           .split(';')[0]  # Remove platform specs
           .replace('-', '_')  # Normalize dashes
           .strip()
           .lower()
        for dep in dependencies
        if dep and not dep.startswith(('-', '.', 'git+'))  # Skip flags and local paths
    }
    
    # Clean up target dependency names
    target_deps_lower = {
        dep.replace('-', '_').strip().lower()
        for dep in target_deps
    }
    
    return any(
        any(target in dep or dep in target 
            for target in target_deps_lower)
        for dep in deps_lower
    )

def generate_ml_model_test_payload(repo_path: str) -> Dict[str, Any]:
    """Generate test payload for ML model repositories."""
    # Collect all dependencies
    dependencies = set()
    for file in find_package_files(repo_path):
        deps = extract_dependencies_from_file(file)
        # Clean up dependency names
        deps = {dep.split('[')[0].split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip().lower() 
               for dep in deps}
        dependencies.update(deps)
    
    # Also check imports in Python files
    for file in find_python_files(repo_path):
        deps = extract_imports_from_python_file(file)
        # Clean up import names
        deps = {dep.split('.')[0].lower() for dep in deps}
        dependencies.update(deps)
    
    # Define test cases based on detected ML frameworks and libraries
    test_cases = []
    
    # Core ML/DL Framework Detection
    core_ml_deps = {
        'tensorflow', 'torch', 'pytorch', 'keras', 'sklearn', 'scikit-learn',
        'xgboost', 'lightgbm', 'fastai', 'mxnet', 'paddlepaddle', 'jax'
    }
    
    # NLP Framework Detection
    nlp_deps = {
        'transformers', 'spacy', 'nltk', 'gensim', 'allennlp',
        'langchain', 'llama-index', 'sentence-transformers'
    }
    
    # Computer Vision Framework Detection
    vision_deps = {
        'opencv-python', 'opencv', 'pillow', 'pil', 'torchvision', 'tensorflow-vision',
        'detectron2', 'albumentations', 'imgaug', 'kornia'
    }
    
    # Time Series Framework Detection
    timeseries_deps = {
        'prophet', 'statsmodels', 'pmdarima', 'neuralprophet',
        'sktime', 'tsai', 'darts'
    }
    
    # Cloud ML Service Detection
    cloud_deps = {
        'sagemaker', 'azureml-core', 'google-cloud-aiplatform',
        'vertexai', 'openai', 'anthropic', 'cohere'
    }
    
    # Determine the primary framework
    framework = None
    if 'torch' in dependencies or 'pytorch' in dependencies:
        framework = 'pytorch'
    elif 'tensorflow' in dependencies:
        framework = 'tensorflow'
    elif 'jax' in dependencies:
        framework = 'jax'
    elif 'sklearn' in dependencies or 'scikit-learn' in dependencies:
        framework = 'sklearn'
    
    # Add test cases based on detected frameworks
    if has_dependency(dependencies, core_ml_deps):
        # Generic ML model test case
        test_cases.append({
            "input": {
                "data": {
                    "features": [1.0, 2.0, 3.0, 4.0],
                    "parameters": {
                        "batch_size": 32,
                        "model_type": "classification",
                        "output_type": "numpy" if framework in ['sklearn', None] else framework
                    }
                }
            },
            "output": {
                "predictions": [0.5],
                "probabilities": [0.3, 0.7],
                "metadata": {
                    "model_type": "generic",
                    "output_shape": [1],
                    "framework": framework or "unknown"
                }
            }
        })
    
    if has_dependency(dependencies, nlp_deps):
        # NLP model test case
        nlp_framework = None
        if 'transformers' in dependencies:
            nlp_framework = 'transformers'
        elif 'spacy' in dependencies:
            nlp_framework = 'spacy'
        elif 'nltk' in dependencies:
            nlp_framework = 'nltk'
        elif 'langchain' in dependencies:
            nlp_framework = 'langchain'
        
        test_cases.append({
            "input": {
                "text": "Sample text for processing",
                "parameters": {
                    "max_length": 128,
                    "truncation": "true",
                    "padding": "max_length",
                    "return_tensors": "pt" if framework == "pytorch" else "tf" if framework == "tensorflow" else None
                }
            },
            "output": {
                "result": "Processed text output",
                "metadata": {
                    "model_type": "nlp",
                    "task_type": "text_processing",
                    "framework": nlp_framework or framework or "unknown"
                }
            }
        })
    
    if has_dependency(dependencies, vision_deps):
        # Computer vision model test case
        vision_framework = None
        if 'torchvision' in dependencies:
            vision_framework = 'torchvision'
        elif 'tensorflow' in dependencies:
            vision_framework = 'tensorflow-vision'
        elif any(dep.startswith('opencv') for dep in dependencies):
            vision_framework = 'opencv'
        
        test_cases.append({
            "input": {
                "image": {
                    "url": "https://example.com/image.jpg",
                    "format": "RGB",
                    "parameters": {
                        "resize": [224, 224],
                        "normalize": "true",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    }
                }
            },
            "output": {
                "predictions": [
                    {"label": "class_1", "score": 0.95},
                    {"label": "class_2", "score": 0.05}
                ],
                "metadata": {
                    "model_type": "vision",
                    "task_type": "classification",
                    "framework": vision_framework or framework or "unknown"
                }
            }
        })
    
    if has_dependency(dependencies, timeseries_deps):
        # Time series model test case
        ts_framework = None
        if 'prophet' in dependencies:
            ts_framework = 'prophet'
        elif 'statsmodels' in dependencies:
            ts_framework = 'statsmodels'
        elif 'neuralprophet' in dependencies:
            ts_framework = 'neuralprophet'
        
        test_cases.append({
            "input": {
                "data": {
                    "timestamps": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                    "values": [100.0, 101.0],
                    "parameters": {
                        "forecast_steps": 5,
                        "frequency": "1D",
                        "confidence_interval": 0.95
                    }
                }
            },
            "output": {
                "forecast": {
                    "timestamps": ["2024-01-03T00:00:00Z"],
                    "values": [102.0],
                    "confidence_intervals": {
                        "lower": [98.0],
                        "upper": [104.0]
                    }
                },
                "metadata": {
                    "model_type": "timeseries",
                    "task_type": "forecasting",
                    "framework": ts_framework or framework or "unknown"
                }
            }
        })
    
    if has_dependency(dependencies, cloud_deps):
        # Cloud ML service test case
        cloud_service = None
        if 'openai' in dependencies:
            cloud_service = 'openai'
        elif 'sagemaker' in dependencies:
            cloud_service = 'sagemaker'
        elif 'azureml-core' in dependencies:
            cloud_service = 'azure'
        elif 'vertexai' in dependencies:
            cloud_service = 'vertex'
        
        test_cases.append({
            "input": {
                "data": "Sample input for cloud service",
                "parameters": {
                    "api_version": "v1",
                    "timeout": 30,
                    "stream": "false",
                    "service": cloud_service or "unknown"
                }
            },
            "output": {
                "result": "Cloud service response",
                "metadata": {
                    "model_type": "cloud_service",
                    "latency": 0.5,
                    "tokens": {"prompt": 10, "completion": 20, "total": 30}
                }
            }
        })
    
    # If no specific test cases were generated, use a generic one
    if not test_cases:
        test_cases.append({
            "input": {
                "data": "Sample input",
                "parameters": {
                    "batch_size": 1,
                    "model_type": "unknown",
                    "framework": framework or "unknown"
                }
            },
            "output": {
                "result": "Sample output",
                "metadata": {
                    "model_type": "unknown",
                    "framework": framework or "unknown"
                }
            }
        })
    
    return {
        "version": "1.0.0",
        "schema_version": "1.0.0",
        "test_cases": test_cases
    }

def generate_api_test_payload(repo_path: str) -> Dict[str, Any]:
    """Generate test payload for API repositories."""
    return {
        "input": {
            "method": "POST",
            "path": "/predict",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            },
            "body": {
                "data": "Example input data",
                "parameters": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
        },
        "output": {
            "status_code": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "result": "Example output",
                "confidence": 0.95,
                "metadata": {
                    "model_version": "1.0.0",
                    "processing_time": "0.123s"
                }
            }
        }
    }

def generate_web_app_test_payload(repo_path: str) -> Dict[str, Any]:
    """Generate test payloads for web application repositories."""
    test_cases = []
    
    # Basic web app test case
    web_test = {
        "input": {
            "url": "http://localhost:3000",
            "method": "GET",
            "headers": {
                "Accept": "text/html"
            }
        },
        "output": {
            "status": 200,
            "content_type": "text/html",
            "body": "<!DOCTYPE html><html>...</html>"
        }
    }
    test_cases.append(web_test)
    
    return {
        "version": "1.0.0",
        "schema_version": "1.0.0",
        "test_cases": test_cases
    }

def generate_test_payloads(repo_path: Path, repo_type: RepoType) -> List[dict]:
    """Generate test payloads based on repository type."""
    if repo_type == RepoType.ML_MODEL:
        return generate_ml_model_test_payload(str(repo_path))["test_cases"]
    elif repo_type == RepoType.API:
        return [generate_api_test_payload(str(repo_path))]
    elif repo_type == RepoType.ML_APPLICATION:
        # ML_APPLICATION is used for web apps and other ML applications
        return generate_web_app_test_payload(str(repo_path))["test_cases"]
    elif repo_type == RepoType.ML_FRAMEWORK:
        # ML frameworks often have similar test patterns to ML models
        return generate_ml_model_test_payload(str(repo_path))["test_cases"]
    elif repo_type == RepoType.ML_TOOL:
        # ML tools often have CLI or API interfaces
        return generate_ml_tool_test_payload(str(repo_path))
    elif repo_type == RepoType.LANGUAGE_COMPILER:
        # Language/compiler repositories usually have API interfaces
        return generate_language_compiler_test_payload(str(repo_path))
    else:
        return [generate_default_test_payload()]

def generate_ml_tool_test_payload(repo_path: str) -> List[dict]:
    """Generate test payloads for ML tool repositories."""
    test_cases = []
    
    # Basic ML tool test case
    ml_tool_test = {
        "input": {
            "command": "run",
            "parameters": {
                "input_file": "sample.txt",
                "output_dir": "/tmp/output",
                "model_name": "default"
            }
        },
        "output": {
            "status": "success",
            "result_files": ["output.json"],
            "metrics": {
                "execution_time": "1.5s",
                "memory_usage": "2.3GB"
            }
        }
    }
    test_cases.append(ml_tool_test)
    
    # Tool-specific API test
    api_test = {
        "input": {
            "method": "POST",
            "endpoint": "/api/v1/process",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer YOUR_API_KEY"
            },
            "body": {
                "input": "Sample text for processing",
                "options": {
                    "format": "json",
                    "verbose": "true"
                }
            }
        },
        "output": {
            "status_code": 200,
            "body": {
                "status": "success",
                "result": "Processed result",
                "metadata": {
                    "processing_time": "0.5s"
                }
            }
        }
    }
    test_cases.append(api_test)
    
    return test_cases

def generate_language_compiler_test_payload(repo_path: str) -> List[dict]:
    """Generate test payloads for language/compiler repositories."""
    test_cases = []
    
    # Basic compiler test case
    compiler_test = {
        "input": {
            "source_code": "function hello() { return 'Hello, World!'; }",
            "options": {
                "target": "es2020",
                "module": "commonjs",
                "sourceMap": "true"
            }
        },
        "output": {
            "compiled_code": "function hello() { return 'Hello, World!'; }",
            "source_map": {},
            "diagnostics": []
        }
    }
    test_cases.append(compiler_test)
    
    # Language service API test
    lang_service_test = {
        "input": {
            "method": "POST",
            "endpoint": "/api/typecheck",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "source": "const x: number = 'string';",
                "options": {
                    "strict": "true"
                }
            }
        },
        "output": {
            "status_code": 200,
            "body": {
                "diagnostics": [
                    {
                        "message": "Type 'string' is not assignable to type 'number'",
                        "severity": "error",
                        "line": 1,
                        "column": 7
                    }
                ]
            }
        }
    }
    test_cases.append(lang_service_test)
    
    return test_cases

def generate_default_test_payload() -> dict:
    """Generate a default test payload for unknown repository types."""
    return {
        "input": {
            "data": "Sample input data",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            }
        },
        "output": {
            "result": "Sample output",
            "metadata": {
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    }

def save_tests_json(repo_path: Path, test_cases: List[dict]) -> None:
    """Save test cases to tests.json in the repository."""
    tests_json = {
        "version": "1.0.0",
        "schema_version": "1.0.0",
        "test_cases": test_cases
    }
    tests_path = repo_path / "tests.json"
    with open(tests_path, "w") as f:
        json.dump(tests_json, f, indent=2)

def generate_test_cases(repo_type: RepoType, dependencies: Set[str]) -> List[dict]:
    """Generate test cases for the repository based on its type and dependencies.
    
    This function provides a consistent interface for generating test cases across
    different repository types. It handles special cases for transformers and 
    tensorflow repositories, which need different input formats. For other
    repository types, it delegates to generate_test_payloads.
    
    Args:
        repo_type: Type of the repository
        dependencies: Set of dependencies used in the repository
        
    Returns:
        List of test cases
    """
    # Create a temporary directory to pass to generate_test_payloads
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # If we have dependencies for specific frameworks, create mock files to help detection
        if dependencies:
            # Create requirements.txt with the dependencies
            with open(temp_dir / "requirements.txt", "w") as f:
                f.write("\n".join(dependencies))
        
        # Special case for transformers
        if repo_type == RepoType.ML_MODEL and 'transformers' in dependencies:
            return [
                {
                    "input": {
                        "text": "Hello world",
                        "parameters": {
                            "max_length": 50,
                            "do_sample": True,
                            "temperature": 0.7
                        }
                    }
                }
            ]
        # Special case for tensorflow
        elif repo_type == RepoType.ML_MODEL and 'tensorflow' in dependencies:
            return [
                {
                    "input": {
                        "data": [1.0, 2.0, 3.0, 4.0, 5.0],
                        "parameters": {
                            "batch_size": 1,
                            "model_type": "classification"
                        }
                    }
                }
            ]
        # Default case
        else:
            # Generate test payloads based on repository type
            return generate_test_payloads(temp_dir, repo_type)
    finally:
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir) 