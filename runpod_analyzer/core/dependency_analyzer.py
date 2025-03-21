"""Dependency analysis for various languages and package managers."""

import json
import re
import tomli
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import os
import logging
import traceback

import yaml
from packaging import requirements

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def safe_read_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """Safely read a file with error handling."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None

def parse_requirements_txt(file_path: str) -> Dict[str, str]:
    """Parse Python requirements.txt file."""
    try:
        dependencies = {}
        content = safe_read_file(file_path)
        if not content:
            return dependencies
            
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Handle various dependency formats
                    if '>=' in line:
                        package, version = line.split('>=')
                    elif '==' in line:
                        package, version = line.split('==')
                    elif '~=' in line:
                        package, version = line.split('~=')
                    elif '!=' in line:
                        package, version = line.split('!=')
                    else:
                        package, version = line, '*'
                    
                    package = package.strip()
                    version = version.strip()
                    
                    if package:
                        dependencies[package] = version
                except Exception as e:
                    logger.warning(f"Error parsing line in requirements.txt: {line}, error: {str(e)}")
                    continue
        return dependencies
    except Exception as e:
        logger.error(f"Error parsing requirements.txt: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def parse_setup_py(content: str) -> List[str]:
    """Parse Python setup.py dependencies."""
    deps = []
    # Look for install_requires list
    match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        requirements_str = match.group(1)
        # Extract package names from quotes
        deps.extend(
            name.strip(' \'"') for name in re.findall(r'["\']([^"\']+)["\']', requirements_str)
        )
    return deps

def parse_pyproject_toml(file_path: str) -> Dict[str, str]:
    """Parse Python pyproject.toml file."""
    try:
        dependencies = {}
        content = safe_read_file(file_path)
        if not content:
            return dependencies
            
        data = tomli.loads(content)
        
        # Check for dependencies in [tool.poetry.dependencies]
        if 'tool' in data and 'poetry' in data['tool'] and 'dependencies' in data['tool']['poetry']:
            deps = data['tool']['poetry']['dependencies']
            for package, version in deps.items():
                if isinstance(version, str):
                    dependencies[package] = version
                elif isinstance(version, dict):
                    dependencies[package] = version.get('version', '*')
        
        # Check for dependencies in [project.dependencies]
        if 'project' in data and 'dependencies' in data['project']:
            deps = data['project']['dependencies']
            for dep in deps:
                try:
                    if '>=' in dep:
                        package, version = dep.split('>=')
                    elif '==' in dep:
                        package, version = dep.split('==')
                    else:
                        package, version = dep, '*'
                    
                    package = package.strip()
                    version = version.strip()
                    
                    if package:
                        dependencies[package] = version
                except Exception as e:
                    logger.warning(f"Error parsing dependency in pyproject.toml: {dep}, error: {str(e)}")
                    continue
                    
        return dependencies
    except Exception as e:
        logger.error(f"Error parsing pyproject.toml: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def parse_package_json(file_path: str) -> Dict[str, str]:
    """Parse Node.js package.json file."""
    try:
        dependencies = {}
        content = safe_read_file(file_path)
        if not content:
            return dependencies
            
        data = json.loads(content)
        
        # Check dependencies
        if 'dependencies' in data:
            for package, version in data['dependencies'].items():
                dependencies[package] = version
        
        # Check devDependencies
        if 'devDependencies' in data:
            for package, version in data['devDependencies'].items():
                dependencies[package] = version
                
        return dependencies
    except Exception as e:
        logger.error(f"Error parsing package.json: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def parse_cargo_toml(file_path: str) -> Dict[str, str]:
    """Parse Rust Cargo.toml file."""
    try:
        dependencies = {}
        content = safe_read_file(file_path)
        if not content:
            return dependencies
            
        data = tomli.loads(content)
        
        # Check dependencies section
        if 'dependencies' in data:
            for package, version in data['dependencies'].items():
                if isinstance(version, str):
                    dependencies[package] = version
                elif isinstance(version, dict):
                    dependencies[package] = version.get('version', '*')
                    
        return dependencies
    except Exception as e:
        logger.error(f"Error parsing Cargo.toml: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def parse_go_mod(file_path: str) -> Dict[str, str]:
    """Parse Go go.mod file."""
    try:
        dependencies = {}
        content = safe_read_file(file_path)
        if not content:
            return dependencies
            
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('require'):
                try:
                    # Handle both direct and indirect dependencies
                    parts = line.split()
                    if len(parts) >= 2:
                        package = parts[1]
                        version = parts[2] if len(parts) > 2 else '*'
                        dependencies[package] = version
                except Exception as e:
                    logger.warning(f"Error parsing line in go.mod: {line}, error: {str(e)}")
                    continue
        return dependencies
    except Exception as e:
        logger.error(f"Error parsing go.mod: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def extract_python_dependencies(file_path: str) -> Set[str]:
    """Extract Python dependencies from requirements.txt, setup.py, or pyproject.toml."""
    deps = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Handle requirements.txt
            if file_path.endswith('requirements.txt'):
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name without version
                        package = re.split(r'[=<>~\[]', line)[0].strip()
                        if package:
                            deps.add(package.lower())
            
            # Handle setup.py
            elif file_path.endswith('setup.py'):
                # Look for install_requires list
                match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    reqs = match.group(1)
                    for line in reqs.split('\n'):
                        line = line.strip().strip(',').strip("'").strip('"')
                        if line and not line.startswith('#'):
                            package = re.split(r'[=<>~\[]', line)[0].strip()
                            if package:
                                deps.add(package.lower())
            
            # Handle pyproject.toml
            elif file_path.endswith('pyproject.toml'):
                # Look for dependencies section
                match = re.search(r'\[tool\.poetry\.dependencies\](.*?)(\[|\Z)', content, re.DOTALL)
                if match:
                    deps_section = match.group(1)
                    for line in deps_section.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            package = line.split('=')[0].strip()
                            if package and package != 'python':
                                deps.add(package.lower())
    except Exception:
        # Skip files that can't be read
        pass
    
    return deps

def extract_node_dependencies(file_path: str) -> Set[str]:
    """Extract Node.js dependencies from package.json."""
    deps = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Get both dependencies and devDependencies
            all_deps = {}
            if 'dependencies' in data:
                all_deps.update(data['dependencies'])
            if 'devDependencies' in data:
                all_deps.update(data['devDependencies'])
            
            # Add package names without versions
            deps.update(all_deps.keys())
    except Exception:
        # Skip files that can't be read or aren't valid JSON
        pass
    
    return deps

def analyze_dependencies(repo_path: Path) -> Set[str]:
    """Analyze repository dependencies and return a set of package names."""
    dependencies = set()
    
    # Check requirements.txt
    requirements_txt = repo_path / "requirements.txt"
    if requirements_txt.exists():
        with open(requirements_txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name
                    pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                    if pkg:
                        dependencies.add(pkg.lower())

    # Check setup.py
    setup_py = repo_path / "setup.py"
    if setup_py.exists():
        with open(setup_py) as f:
            content = f.read()
            # Look for install_requires list
            import re
            install_requires = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if install_requires:
                for line in install_requires.group(1).split('\n'):
                    line = line.strip().strip("'").strip('"').strip(',')
                    if line and not line.startswith('#'):
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                        if pkg:
                            dependencies.add(pkg.lower())

    # Check pyproject.toml
    pyproject_toml = repo_path / "pyproject.toml"
    if pyproject_toml.exists():
        with open(pyproject_toml) as f:
            content = f.read()
            # Look for dependencies section
            import re
            dep_section = re.search(r'\[project\]\s*.*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if dep_section:
                for line in dep_section.group(1).split('\n'):
                    line = line.strip().strip("'").strip('"').strip(',')
                    if line and not line.startswith('#'):
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                        if pkg:
                            dependencies.add(pkg.lower())

    return dependencies

def analyze_dependencies_across_languages(repo_path: str) -> Set[str]:
    """Analyze dependencies across multiple programming languages."""
    try:
        dependencies = set()
        repo_path = Path(repo_path)
        
        # Python dependencies
        if (repo_path / 'requirements.txt').exists():
            deps = parse_requirements_txt(str(repo_path / 'requirements.txt'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Python dependencies in requirements.txt")
            
        if (repo_path / 'pyproject.toml').exists():
            deps = parse_pyproject_toml(str(repo_path / 'pyproject.toml'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Python dependencies in pyproject.toml")
        
        # Check for setup.py
        if (repo_path / 'setup.py').exists():
            try:
                with open(str(repo_path / 'setup.py'), 'r', encoding='utf-8') as f:
                    setup_content = f.read()
                    deps = parse_setup_py(setup_content)
                    dependencies.update(deps)
                    logger.info(f"Found {len(deps)} Python dependencies in setup.py")
            except Exception as e:
                logger.warning(f"Error parsing setup.py: {str(e)}")
            
        # Node.js dependencies
        if (repo_path / 'package.json').exists():
            deps = parse_package_json(str(repo_path / 'package.json'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Node.js dependencies in package.json")
            
        # Java dependencies
        if (repo_path / 'pom.xml').exists():
            deps = parse_pom_xml(str(repo_path / 'pom.xml'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Java dependencies in pom.xml")
            
        if (repo_path / 'build.gradle').exists():
            deps = parse_gradle_build(str(repo_path / 'build.gradle'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Gradle dependencies in build.gradle")
            
        # Rust dependencies
        if (repo_path / 'Cargo.toml').exists():
            deps = parse_cargo_toml(str(repo_path / 'Cargo.toml'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Rust dependencies in Cargo.toml")
            
        # Go dependencies
        if (repo_path / 'go.mod').exists():
            deps = parse_go_mod(str(repo_path / 'go.mod'))
            dependencies.update(deps.keys())
            logger.info(f"Found {len(deps)} Go dependencies in go.mod")
            
        # Special case for vLLM and other ML frameworks
        if (repo_path / 'setup.py').exists() and 'vllm' in repo_path.name.lower():
            dependencies.add('vllm')
            logger.info("Detected vLLM from repository name")
        
        # Look for additional requirements files
        for req_file in repo_path.glob('**/requirements*.txt'):
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-r'):
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                            if pkg:
                                dependencies.add(pkg.lower())
            except Exception:
                pass
                
        # Additional dependency scanning: check imports in Python files
        # Only if we haven't found many dependencies yet, to avoid excessive scanning
        if len(dependencies) < 5:
            python_source_deps = scan_python_imports(repo_path)
            dependencies.update(python_source_deps)
            logger.info(f"Found {len(python_source_deps)} dependencies from Python imports")
            
        # Check for common ML dependencies in source code
        ml_keywords = ['torch', 'tensorflow', 'cuda', 'triton', 'transformers', 'huggingface', 
                       'keras', 'sklearn', 'streamlit', 'gradio', 'fastapi', 'flask', 
                       'jax', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
                       'opencv', 'pillow', 'nltk', 'spacy', 'gensim']
                       
        for py_file in repo_path.glob('**/*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    for keyword in ml_keywords:
                        if keyword in content and (f"import {keyword}" in content.lower() or f"from {keyword}" in content.lower()):
                            dependencies.add(keyword)
                            logger.info(f"Found ML dependency '{keyword}' in code imports")
            except Exception:
                pass  # Silently ignore file reading errors
            
        return dependencies
        
    except Exception as e:
        logger.error(f"Error analyzing dependencies: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return set()

def scan_python_imports(repo_path: Path) -> Set[str]:
    """Scan Python files for import statements to identify dependencies."""
    dependencies = set()
    stdlib_modules = {
        'os', 'sys', 'datetime', 'time', 'math', 're', 'json', 'random', 
        'collections', 'itertools', 'functools', 'logging', 'io', 'tempfile',
        'shutil', 'pathlib', 'argparse', 'subprocess', 'multiprocessing',
        'threading', 'concurrent', 'urllib', 'http', 'socket', 'ssl',
        'email', 'xml', 'html', 'csv', 'sqlite3', 'pickle', 'copy',
        'typing', 'abc', 'hashlib', 'base64', 'zlib', 'gzip', 'zipfile',
        'tarfile', 'configparser', 'contextlib', 'dis', 'enum', 'inspect',
        'doctest', 'unittest', 'pdb', 'trace', 'array', 'types', 'uuid'
    }
    
    # Exclude directories that are likely to be test or build directories
    excluded_dirs = {'tests', 'test', 'docs', 'doc', 'examples', 'build', 'dist', '__pycache__', '.git', '.github'}
    
    # Counters for statistics
    py_files_scanned = 0
    imports_found = 0
    
    # Regular expressions for import statements
    import_pattern = re.compile(r'^import\s+([a-zA-Z0-9_., ]+)', re.MULTILINE)
    from_pattern = re.compile(r'^from\s+([a-zA-Z0-9_.]+)\s+import', re.MULTILINE)
    
    for py_file in repo_path.glob('**/*.py'):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in excluded_dirs):
            continue
            
        py_files_scanned += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Process 'import xxx' statements
                for match in import_pattern.finditer(content):
                    import_stmt = match.group(1)
                    for module in import_stmt.split(','):
                        module = module.strip().split('.')[0].strip()
                        if module and module.lower() not in stdlib_modules:
                            dependencies.add(module.lower())
                            imports_found += 1
                
                # Process 'from xxx import' statements
                for match in from_pattern.finditer(content):
                    module = match.group(1).split('.')[0].strip()
                    if module and module.lower() not in stdlib_modules:
                        dependencies.add(module.lower())
                        imports_found += 1
                        
        except Exception:
            pass
    
    logger.info(f"Scanned {py_files_scanned} Python files, found {imports_found} import statements")
    return dependencies

def extract_ml_dependencies_from_code(repo_path: Path) -> Set[str]:
    """Scan code files specifically for ML-related dependencies."""
    ml_dependencies = set()
    
    # List of ML-related libraries to look for
    ml_libraries = {
        # Deep Learning
        'torch': ['torch', 'pytorch'],
        'tensorflow': ['tensorflow', 'tf', 'keras'],
        'jax': ['jax', 'flax', 'optax'],
        
        # ML Libraries
        'sklearn': ['sklearn', 'scikit-learn', 'scikit_learn'],
        'xgboost': ['xgboost', 'xgb'],
        'lightgbm': ['lightgbm', 'lgbm'],
        
        # NLP
        'transformers': ['transformers', 'huggingface'],
        'spacy': ['spacy'],
        'nltk': ['nltk'],
        
        # Computer Vision
        'cv2': ['cv2', 'opencv'],
        'pillow': ['pil', 'pillow', 'image'],
        
        # Data Processing
        'numpy': ['numpy', 'np'],
        'pandas': ['pandas', 'pd'],
        
        # ML Applications
        'gradio': ['gradio', 'gr'],
        'streamlit': ['streamlit', 'st'],
        
        # API Frameworks
        'fastapi': ['fastapi'],
        'flask': ['flask'],
        
        # ML Ops
        'mlflow': ['mlflow'],
        'wandb': ['wandb', 'weights & biases'],
        'ray': ['ray', 'ray[tune]', 'ray[rllib]']
    }
    
    # Patterns to look for in code
    import_patterns = [
        re.compile(r'import\s+([a-zA-Z0-9_.]+)', re.IGNORECASE),
        re.compile(r'from\s+([a-zA-Z0-9_.]+)\s+import', re.IGNORECASE),
        re.compile(r'([a-zA-Z0-9_]+)\.', re.IGNORECASE)  # Usage patterns
    ]
    
    # Scan Python files
    for py_file in repo_path.glob('**/*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check import statements
                for pattern in import_patterns:
                    for match in pattern.finditer(content):
                        module = match.group(1).split('.')[0].lower()
                        
                        # Check if matches any of our ML library keys or aliases
                        for lib_name, aliases in ml_libraries.items():
                            if module in aliases:
                                ml_dependencies.add(lib_name)
                
                # Special check for specific usage patterns
                if 'torch.nn' in content or 'nn.Module' in content:
                    ml_dependencies.add('torch')
                if 'tf.keras' in content or 'model.fit(' in content:
                    ml_dependencies.add('tensorflow')
                if 'transformers.AutoModel' in content or 'from transformers import' in content:
                    ml_dependencies.add('transformers')
                if 'streamlit.app' in content or 'st.write(' in content:
                    ml_dependencies.add('streamlit')
                if 'gr.Interface' in content or 'gradio.Interface' in content:
                    ml_dependencies.add('gradio')
                        
        except Exception:
            continue
    
    return ml_dependencies 