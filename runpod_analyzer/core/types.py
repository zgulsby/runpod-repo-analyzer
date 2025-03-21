from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

class RepoType(Enum):
    """Types of repositories that can be detected."""
    ML_MODEL = "ml_model"  # Pure ML models (e.g., Hugging Face model repos)
    ML_FRAMEWORK = "ml_framework"  # ML frameworks and tools (e.g., vLLM, Transformers)
    ML_APPLICATION = "ml_application"  # End-to-end ML applications
    ML_INFRASTRUCTURE = "ml_infrastructure"  # ML infrastructure tools
    ML_TOOL = "ml_tool"  # Specialized ML tools and utilities (e.g., Cog, Gradio)
    API = "api"  # General API services
    LIBRARY = "library"  # General libraries
    LANGUAGE_COMPILER = "language_compiler"  # Programming languages and compilers
    UNKNOWN = "unknown"  # Unclassified repositories

@dataclass(frozen=True)
class RepositoryAnalysis:
    """Analysis results for a repository."""
    repo_type: RepoType
    confidence: float
    languages: Dict[str, float]
    dependencies: Set[str]
    test_cases: List[dict]
    repository_url: Optional[str] = None  # URL of the repository being analyzed 