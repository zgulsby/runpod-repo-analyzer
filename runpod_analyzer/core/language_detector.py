from typing import Dict, List, Optional
from pathlib import Path
import os

def get_file_extension(file_path: str) -> str:
    """Get the file extension from a path."""
    return os.path.splitext(file_path)[1].lower()

def is_source_file(file_path: str) -> bool:
    """Check if a file is a source code file."""
    # Common source code file extensions
    source_extensions = {
        # Python
        '.py', '.pyx', '.pyi',
        # JavaScript/TypeScript
        '.js', '.jsx', '.ts', '.tsx',
        # Java
        '.java',
        # C/C++
        '.c', '.cpp', '.h', '.hpp',
        # Ruby
        '.rb',
        # Go
        '.go',
        # Rust
        '.rs',
        # PHP
        '.php',
        # Swift
        '.swift',
        # Kotlin
        '.kt',
        # Scala
        '.scala',
        # Shell
        '.sh', '.bash',
        # R
        '.r', '.R',
        # Julia
        '.jl'
    }
    return get_file_extension(file_path) in source_extensions

def get_language_from_extension(ext: str) -> str:
    """Map file extension to programming language."""
    extension_map = {
        # Python
        '.py': 'python',
        '.pyx': 'python',
        '.pyi': 'python',
        # JavaScript/TypeScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        # Java
        '.java': 'java',
        # C/C++
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        # Ruby
        '.rb': 'ruby',
        # Go
        '.go': 'go',
        # Rust
        '.rs': 'rust',
        # PHP
        '.php': 'php',
        # Swift
        '.swift': 'swift',
        # Kotlin
        '.kt': 'kotlin',
        # Scala
        '.scala': 'scala',
        # Shell
        '.sh': 'shell',
        '.bash': 'shell',
        # R
        '.r': 'r',
        '.R': 'r',
        # Julia
        '.jl': 'julia'
    }
    return extension_map.get(ext, 'unknown')

def count_lines_in_file(file_path: str) -> int:
    """Count the number of non-empty lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except (UnicodeDecodeError, IOError):
        # Skip binary files or files that can't be read
        return 0

def detect_languages(repo_path: str) -> Dict[str, float]:
    """
    Detect programming languages used in a repository.
    Returns a dictionary mapping language names to their percentage of use.
    """
    language_lines = {}
    total_lines = 0
    
    # Walk through the repository
    for root, _, files in os.walk(repo_path):
        # Skip hidden directories and common non-source directories
        if any(part.startswith('.') for part in Path(root).parts) or \
           any(part in ['node_modules', 'venv', 'env', 'build', 'dist'] for part in Path(root).parts):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            if is_source_file(file_path):
                ext = get_file_extension(file_path)
                lang = get_language_from_extension(ext)
                if lang != 'unknown':
                    lines = count_lines_in_file(file_path)
                    language_lines[lang] = language_lines.get(lang, 0) + lines
                    total_lines += lines
    
    # Calculate percentages
    if total_lines > 0:
        return {lang: count / total_lines for lang, count in language_lines.items()}
    else:
        return {} 