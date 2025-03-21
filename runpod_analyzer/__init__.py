"""RunPod Repository Analyzer - A tool for analyzing repositories and generating RunPod configurations."""

__version__ = "0.1.0"

from runpod_analyzer.core.repo_patterns import analyze_repository

__all__ = ["analyze_repository"] 