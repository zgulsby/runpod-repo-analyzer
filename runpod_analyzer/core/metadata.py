from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import List, Optional, Dict, Any, Set
import os
import logging
import numpy as np
from .types import RepoType, RepositoryAnalysis

# Add exports for the new functions from metadata_generator.py
__all__ = [
    'HubMetadata', 
    'generate_hub_json', 
    'save_hub_json',
    'extract_title_from_readme',
    'extract_description_from_readme',
    'extract_env_vars',
    'generate_tags',
    'generate_metadata',
    'save_metadata'
]

@dataclass
class HubMetadata:
    """Represents the metadata structure for hub.json"""
    title: str
    description: str
    tags: List[str]
    env_vars: Dict[str, str]
    version: str = "1.0.0"
    schema_version: int = 1

def safe_read_file(file_path: Path) -> str:
    """Safely read a file with fallback encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    
    return ""  # Return empty string if all encodings fail

def clean_markdown(text: str) -> str:
    """Clean markdown and HTML tags from text"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown emphasis
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
    
    # Remove markdown code blocks
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove badges
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_title(readme_path: Path, repo_path: Path) -> str:
    """Extract title from README.md or use repository name"""
    if readme_path.exists():
        content = safe_read_file(readme_path)
        if content:
            # Try to find an image alt text that might contain the title
            alt_match = re.search(r'<img[^>]*alt="([^"]+)"[^>]*>', content)
            if alt_match:
                title = clean_markdown(alt_match.group(1))
                # Remove "Logo" suffix if present
                title = re.sub(r'\s+Logo$', '', title)
                return title
            
            # Try to find a h1 header
            h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if h1_match:
                return clean_markdown(h1_match.group(1))
            
            # Try to find the first non-empty line that's not a badge or HTML tag
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('[![') and not line.startswith('<'):
                    return clean_markdown(line)
    
    # Fallback to repository name
    return repo_path.name.replace('-', ' ').title()

def extract_description(readme_path: Path) -> str:
    """Extract a description from the README file."""
    if not readme_path.exists():
        logging.warning(f"README file not found at {readme_path}")
        return "No description available"
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to find a description paragraph after the title
        lines = content.split('\n')
        description_candidates = []
        
        # Find the first non-empty paragraph after any headers, badges, or links
        in_paragraph = False
        paragraph = []
        
        for line in lines[1:]:  # Skip the first line (title)
            line = line.strip()
            
            # Skip badge lines (common in READMEs)
            if re.search(r'!\[.*?\]\(.*?\)', line) or re.search(r'<img.*?>', line):
                continue
                
            # Skip horizontal rules and section headers
            if line.startswith('---') or line.startswith('===') or line.startswith('#'):
                if in_paragraph and paragraph:
                    description_candidates.append(' '.join(paragraph))
                    paragraph = []
                    in_paragraph = False
                continue
                
            # If the line is not empty, add it to the current paragraph
            if line:
                paragraph.append(line)
                in_paragraph = True
            # If the line is empty and we were in a paragraph, end the paragraph
            elif in_paragraph:
                description_candidates.append(' '.join(paragraph))
                paragraph = []
                in_paragraph = False
        
        # Add the last paragraph if it exists
        if paragraph:
            description_candidates.append(' '.join(paragraph))
        
        # Filter out very short descriptions
        description_candidates = [d for d in description_candidates if len(d) > 30]
        
        if description_candidates:
            # Select the first good candidate
            desc = description_candidates[0]
            
            # Truncate if too long (max 200 words)
            words = desc.split()
            if len(words) > 200:
                desc = ' '.join(words[:200]) + '...'
                
            return desc
        else:
            logging.warning("No suitable description found in README")
            return "No description available"
            
    except Exception as e:
        logging.error(f"Error extracting description from README: {str(e)}")
        return "No description available"

def extract_tags(repo_path: Path, readme_content: str) -> List[str]:
    """Infer tags from README content and repository structure"""
    tags = set()
    
    # Add tags based on detected files
    if (repo_path / 'requirements.txt').exists():
        tags.add('python')
    if (repo_path / 'package.json').exists():
        tags.add('nodejs')
    if list(repo_path.glob('*.py')):
        tags.add('python')
    if list(repo_path.glob('*.js')) or list(repo_path.glob('*.ts')):
        tags.add('javascript')
    
    # Add tags from README content
    if readme_content:
        content = readme_content.lower()
        if content:
            # Common technology keywords
            keywords = {
                'machine learning': ['ml', 'machine-learning', 'ai'],
                'deep learning': ['deep-learning', 'neural-network'],
                'api': ['api', 'rest', 'graphql'],
                'web': ['web', 'frontend', 'backend'],
                'database': ['database', 'sql', 'nosql'],
                'docker': ['docker', 'container'],
                'kubernetes': ['kubernetes', 'k8s'],
                'serverless': ['serverless', 'faas'],
            }
            
            for tech, tech_tags in keywords.items():
                if tech in content or any(tag in content for tag in tech_tags):
                    tags.add(tech_tags[0])
    
    return sorted(list(tags))

def extract_env_vars(repo_path: Path) -> Dict[str, str]:
    """Identify environment variables from common config files"""
    env_vars = {}
    
    # Check .env.example file
    env_example = repo_path / '.env.example'
    if env_example.exists():
        content = safe_read_file(env_example)
        if content:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip("'").strip('"')
                        env_vars[key] = value if value != '' else "Required"
    
    # Check for environment variables in Python files
    for py_file in repo_path.glob('**/*.py'):
        try:
            content = safe_read_file(py_file)
            if content:
                # Look for os.environ or os.getenv usage
                env_matches = re.finditer(r'os\.(environ|getenv)\[?[\'"](\w+)[\'"]\]?', content)
                for match in env_matches:
                    key = match.group(2)
                    if key not in env_vars:
                        env_vars[key] = "Required"
        except Exception:
            continue
    
    return env_vars

def generate_hub_json(repo_path: Path, analysis: RepositoryAnalysis) -> Dict[str, Any]:
    """
    Generate a hub.json file with metadata about the repository.
    """
    # Extract repository name as title
    readme_path = repo_path / "README.md"
    title = extract_title(readme_path, repo_path)
    
    # Extract description from README
    description = extract_description(readme_path)
    
    # Determine tags based on repository type and dependencies
    tags = extract_tags(repo_path, safe_read_file(readme_path))
    
    # Extract environment variables from code and config files
    env_vars = extract_env_vars(repo_path)
    
    # Determine repository type (serverless, api, app)
    repo_type = "serverless"  # Default to serverless
    if analysis.repo_type == RepoType.API:
        repo_type = "api"
    
    # Determine category based on repo type and analysis
    category_mapping = {
        RepoType.ML_MODEL: "machine-learning",
        RepoType.ML_APPLICATION: "machine-learning",
        RepoType.ML_FRAMEWORK: "machine-learning",
        RepoType.API: "api",
        RepoType.LIBRARY: "utility",
        RepoType.LANGUAGE_COMPILER: "utility",
        RepoType.ML_TOOL: "machine-learning"
    }
    category = category_mapping.get(analysis.repo_type, "utility")
    
    # Specific subcategories based on language tasks
    # Check tags and description for language-related keywords
    nlp_keywords = ["nlp", "language", "text", "translation", "sentiment", "chatbot", "llm", "gpt"]
    vision_keywords = ["vision", "image", "detection", "recognition", "segmentation", "camera"]
    audio_keywords = ["audio", "speech", "voice", "sound", "text-to-speech", "tts"]
    
    tags_str = " ".join(tags).lower()
    desc_str = description.lower()
    combined = tags_str + " " + desc_str
    
    if any(keyword in combined for keyword in nlp_keywords):
        category = "language"
    elif any(keyword in combined for keyword in vision_keywords):
        category = "vision"
    elif any(keyword in combined for keyword in audio_keywords):
        category = "audio"
    
    # Build the hub.json content
    hub_json = {
        "title": title,
        "description": description,
        "tags": tags,
        "type": repo_type,
        "category": category,
        "iconUrl": "https://raw.githubusercontent.com/runpod/runpod-icons/main/default.svg",
        "config": {
            "runsOn": "GPU",
            "containerDiskInGb": 10,
            "env": []
        }
    }
    
    # Common environment variables for ML models
    common_env_vars = [
        {
            "key": "MODEL_PATH",
            "input": {
                "name": "Model Path",
                "type": "text",
                "default": "model"
            }
        },
        {
            "key": "DEVICE",
            "input": {
                "name": "Device",
                "type": "select",
                "default": "cuda",
                "options": ["cuda", "cpu"]
            }
        },
        {
            "key": "MAX_LENGTH",
            "input": {
                "name": "Maximum Output Length",
                "type": "number",
                "default": 512
            }
        }
    ]
    
    # Add common environment variables
    hub_json["config"]["env"].extend(common_env_vars)
    
    # Convert env_vars to a dictionary format for compatibility
    env_vars_dict = {}
    if isinstance(env_vars, list):
        # If it's a list of dictionaries, convert to dict format
        for env_var in env_vars:
            if isinstance(env_var, dict) and "name" in env_var:
                env_vars_dict[env_var["name"]] = env_var.get("description", "")
    else:
        # Otherwise assume it's already a dictionary
        env_vars_dict = env_vars
    
    # Add other environment variables
    for key, value in env_vars_dict.items():
        # Skip if we already have this env var
        if any(e["key"] == key for e in hub_json["config"]["env"]):
            continue
            
        # Create a new env var entry
        new_env_var = {
            "key": key,
            "input": {
                "name": value if value else key,
                "type": "text",
                "default": value if value else ""
            }
        }
        hub_json["config"]["env"].append(new_env_var)
    
    return json.dumps(hub_json, indent=2)

def save_hub_json(repo_path: Path, metadata: Dict[str, Any]) -> None:
    """Save metadata to hub.json in the repository."""
    hub_path = repo_path / "hub.json"
    with open(hub_path, "w") as f:
        json.dump(metadata, f, indent=2)

def extract_title_from_readme(readme_content: str) -> str:
    """Extract title from README.md content."""
    if readme_content:
        try:
            # Look for first heading
            lines = readme_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    return line.lstrip('# ').strip()
                elif line.startswith('=='):
                    return lines[0].strip()
        except Exception:
            pass
    
    # If no README or no title found, return a generic title
    return "RunPod Serverless Project"

def extract_description_from_readme(readme_content: str) -> str:
    """Extract description from README.md content."""
    if readme_content:
        try:
            # Look for first paragraph after title
            lines = readme_content.split('\n')
            description = []
            found_title = False
            
            for line in lines:
                line = line.strip()
                if not found_title:
                    if line.startswith('# ') or line.startswith('=='):
                        found_title = True
                    continue
                
                if line and not line.startswith('#'):
                    description.append(line)
                elif description:  # Stop at next heading if we have description
                    break
            
            if description:
                return ' '.join(description)
        except Exception:
            pass
    
    return "A RunPod serverless endpoint"

def extract_env_vars(repo_path: str) -> List[Dict[str, str]]:
    """Extract environment variables from configuration files."""
    env_vars = []
    
    # Common environment variable file patterns
    env_files = ['.env.example', '.env.template', '.env.sample', 'config.yml', 'config.yaml']
    
    for file_name in env_files:
        file_path = os.path.join(repo_path, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract variables based on file type
                    if file_name.endswith(('.yml', '.yaml')):
                        # Parse YAML config
                        import yaml
                        config = yaml.safe_load(content)
                        if isinstance(config, dict):
                            for key in config:
                                env_vars.append({
                                    "name": str(key).upper(),
                                    "description": "Configuration parameter",
                                    "required": True
                                })
                    else:
                        # Parse .env style files
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '=' in line:
                                    key = line.split('=')[0].strip()
                                    env_vars.append({
                                        "name": key,
                                        "description": "Environment variable",
                                        "required": True
                                    })
            except Exception:
                pass
    
    # Add common environment variables based on repository type
    common_vars = {
        RepoType.ML_MODEL: [
            {"name": "MODEL_PATH", "description": "Path to the model file", "required": True},
            {"name": "DEVICE", "description": "Device to run the model on (cpu/cuda)", "required": False}
        ],
        RepoType.API: [
            {"name": "PORT", "description": "Port to run the service on", "required": False},
            {"name": "HOST", "description": "Host to bind the service to", "required": False}
        ]
    }
    
    return env_vars + common_vars.get(RepoType.ML_MODEL, [])

def generate_tags(repo_path: str, repo_type: RepoType) -> List[str]:
    """Generate tags based on repository characteristics."""
    tags = []
    
    # Add type-specific tags
    type_tags = {
        RepoType.ML_MODEL: ["ai", "ml", "model"],
        RepoType.API: ["api", "service", "web"],
        RepoType.LIBRARY: ["library", "package", "sdk"],
        RepoType.UNKNOWN: ["other"]
    }
    
    tags.extend(type_tags.get(repo_type, []))
    
    # Add tags from README topics
    readme_path = os.path.join(repo_path, "README.md")
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                # Look for common keywords
                keywords = {
                    "tensorflow": ["tensorflow", "deep-learning"],
                    "pytorch": ["pytorch", "deep-learning"],
                    "transformers": ["transformers", "nlp"],
                    "vision": ["computer-vision", "image"],
                    "nlp": ["nlp", "text"],
                    "api": ["api", "rest"],
                    "graphql": ["graphql", "api"],
                    "docker": ["docker", "container"]
                }
                
                for keyword, related_tags in keywords.items():
                    if keyword in content:
                        tags.extend(related_tags)
        except Exception:
            pass
    
    # Remove duplicates and sort
    return sorted(list(set(tags)))

def generate_metadata(repo_path: Path) -> Dict[str, Any]:
    """Generate hub.json metadata for the repository."""
    # Read README content
    readme_path = repo_path / "README.md"
    readme_content = ""
    if readme_path.exists():
        with open(readme_path) as f:
            readme_content = f.read()
    
    # Generate metadata
    metadata = {
        "version": "1.0.0",
        "title": extract_title_from_readme(readme_content),
        "description": extract_description_from_readme(readme_content),
        "tags": extract_tags(repo_path, readme_content),
        "env": extract_env_vars(repo_path),
        "ui": {
            "inputs": [
                {
                    "name": "input",
                    "type": "json",
                    "description": "Input data for processing"
                }
            ]
        }
    }
    
    return metadata

def save_metadata(repo_path: Path, metadata: Dict[str, Any]) -> None:
    """Save hub.json to the repository."""
    hub_path = repo_path / "hub.json"
    with open(hub_path, "w") as f:
        json.dump(metadata, f, indent=2) 