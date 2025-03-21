"""Main script for RunPod repository analyzer."""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from .core.repo_patterns import analyze_repository
from .core.types import RepoType

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a repository and generate RunPod-required files."
    )
    parser.add_argument(
        "repo_url",
        help="URL of the repository to analyze"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for generated files (default: ./output)",
        default="./output"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def get_repo_name(repo_url: str) -> str:
    """Extract repository name from URL."""
    # Remove .git extension if present
    repo_url = repo_url.rstrip(".git")
    
    # Get the last part of the URL
    return repo_url.split("/")[-1]

def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get repository name and create target directory
    repo_name = get_repo_name(args.repo_url)
    repo_path = output_dir / repo_name
    
    try:
        # Analyze repository and generate files
        if args.verbose:
            print(f"Analyzing repository: {args.repo_url}")
            print(f"Output directory: {repo_path}")
        
        analysis = analyze_repository(args.repo_url)
        
        # Print analysis results
        if args.verbose:
            print("\nAnalysis Results:")
            print(f"Repository Type: {analysis.repo_type.name}")
            print(f"Confidence Score: {analysis.confidence:.2f}")
            print(f"Languages: {', '.join(analysis.languages)}")
            print(f"Dependencies: {', '.join(sorted(analysis.dependencies))}")
            print(f"\nGenerated Files:")
            print(f"- {repo_path}/hub.json")
            print(f"- {repo_path}/tests.json")
            print(f"- {repo_path}/Dockerfile")
            print(f"- {repo_path}/handler.py")
        
        print("\nSuccess! RunPod files have been generated.")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 