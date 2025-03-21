"""Type definitions for the RunPod Analyzer."""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

class RepoType(Enum):
    """Type of repository."""
    UNKNOWN = "unknown"
    ML_MODEL = "ml_model"
    ML_APPLICATION = "ml_application"
    ML_FRAMEWORK = "ml_framework"
    API = "api"
    LIBRARY = "library"
    LANGUAGE_COMPILER = "language_compiler"
    ML_TOOL = "ml_tool"

@dataclass
class RepositoryAnalysis:
    """Results of the repository analysis."""
    repo_type: RepoType
    confidence: float
    languages: Dict[str, float]
    dependencies: Set[str]
    source_dir: Optional[str] = None
    repo_url: Optional[str] = None
    test_cases: List[dict] = field(default_factory=list)
    repository_url: Optional[str] = None  # For backwards compatibility 