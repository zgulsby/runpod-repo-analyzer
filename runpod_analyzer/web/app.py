"""Web interface for RunPod repository analyzer."""

import os
import logging
import tempfile
import json
import traceback
import shutil
import time
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Form, Query, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import sys
from contextlib import contextmanager

from ..core.repo_patterns import analyze_repository, RepoType
from ..core.types import RepositoryAnalysis
from ..core.metadata import generate_hub_json, safe_read_file
from ..core.test_generator import generate_test_payloads
from ..core.dockerfile_generator import generate_dockerfile
from ..core.handler_generator import generate_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RunPod Repository Analyzer",
    description="Analyze GitHub repositories and generate RunPod-required files",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    logger.error(f"Static directory not found: {static_dir}")
    raise RuntimeError("Static directory not found")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Remove unused model
# class RepositoryRequest(BaseModel):
#     """Repository analysis request model."""
#     repo_url: str

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions to prevent exposing sensitive information."""
    # Log the full exception for debugging
    error_id = f"error-{id(exc)}"
    logger.error(f"Unhandled exception [{error_id}]: {str(exc)}")
    logger.error(f"Request path: {request.url.path}")
    logger.error(traceback.format_exc())
    
    # Return a generic error message to the client
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An internal server error occurred",
            "error_id": error_id  # Include an ID so the specific error can be found in logs
        }
    )

def validate_github_url(url: str) -> bool:
    """Validate that the URL is a valid GitHub repository URL."""
    # Better sanitization
    if not url or not isinstance(url, str):
        return False
        
    # Strip any leading/trailing whitespace
    url = url.strip()
    
    # Check for potentially malicious inputs
    if '..' in url or '&' in url or ';' in url or '|' in url:
        logger.warning(f"Potentially unsafe characters in URL: {url}")
        return False
        
    # Validate GitHub URL format
    github_pattern = r'^https://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_\.]+$'
    return bool(re.match(github_pattern, url))

# Define a timeout error
class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass

# Timeout handler
@contextmanager
def timeout(seconds, error_message="Operation timed out"):
    """Context manager for timeouts."""
    def signal_handler(signum, frame):
        raise TimeoutError(error_message)
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def perform_repository_analysis(repo_url: str) -> Dict[str, Any]:
    """
    Common helper function to perform repository analysis and generate files.
    This avoids code duplication between GET and POST routes.
    """
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL is required")

    # Validate GitHub URL
    if not validate_github_url(repo_url):
        logging.error(f"Invalid GitHub URL: {repo_url}")
        raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")
    
    # For all repositories, use standard analysis
    logging.info(f"Analyzing repository: {repo_url}")
    temp_path = None

    try:
        # Create a single temporary directory for the whole process
        temp_path = tempfile.mkdtemp(prefix="runpod_analyzer_")
        logging.info(f"Created temporary directory for analysis: {temp_path}")
        
        # Analyze the repository with a timeout - pass temp_path to use for cloning
        try:
            with timeout(300, "Repository analysis timed out after 5 minutes"):
                # Use a manually created temp_path that we won't clean up until we're done
                analysis = analyze_repository(repo_url, clone_path=temp_path)
        except TimeoutError as e:
            logging.error(f"Timeout analyzing repository {repo_url}: {str(e)}")
            raise HTTPException(status_code=504, detail="Repository analysis timed out. Try again with a smaller repository.")
        
        if not hasattr(analysis, 'test_cases'):
            # This is a safety check in case the RepositoryAnalysis instance doesn't have test_cases
            logging.warning("Analysis result doesn't have test_cases attribute, adding empty list")
            # Create a new instance with test_cases
            from ..core.types import RepositoryAnalysis
            analysis = RepositoryAnalysis(
                repo_type=analysis.repo_type,
                confidence=analysis.confidence,
                languages=analysis.languages,
                dependencies=analysis.dependencies,
                test_cases=[]
            )

        # Check for README file availability to help with debugging
        readme_path = Path(temp_path) / "README.md"
        if readme_path.exists():
            logging.info(f"README.md found at: {readme_path}")
            readme_content = safe_read_file(readme_path)
            logging.info(f"README.md content length: {len(readme_content)}, preview: {readme_content[:100]}")
        else:
            # Look for README in all subdirectories
            readme_files = list(Path(temp_path).glob("**/README.md"))
            if readme_files:
                readme_path = readme_files[0]
                logging.info(f"Found README.md at: {readme_path}")
                try:
                    # Copy the README to the root for extraction
                    root_readme_path = Path(temp_path) / "README.md"
                    shutil.copy(readme_path, root_readme_path)
                    logging.info(f"Copied README from {readme_path} to {root_readme_path}")
                except Exception as e:
                    logging.error(f"Error copying README: {str(e)}")
            else:
                logging.warning(f"No README.md found in repository")

        # Generate RunPod files
        hub_json = generate_hub_json(Path(temp_path), analysis)
        
        # Add debug logging for hub.json to check description field
        try:
            hub_dict = json.loads(hub_json)
            logging.info(f"hub.json description: {hub_dict.get('description', 'No description in JSON')}")
        except json.JSONDecodeError:
            logging.error(f"Failed to parse hub.json for debugging: {hub_json[:100]}")
        
        tests_json = generate_test_payloads(Path(temp_path), analysis.repo_type)
        dockerfile = generate_dockerfile(Path(temp_path))
        handler = generate_handler(Path(temp_path), analysis.repo_type, analysis.dependencies)

        # Return the analysis results and generated files
        result = {
            "status": "success",
            "repository": {
                "type": analysis.repo_type.value,
                "confidence": analysis.confidence * 100,  # Convert to percentage
                "languages": list(analysis.languages.keys()),
                "dependencies": list(analysis.dependencies)
            },
            "files": {
                "hub_json": hub_json,
                "tests_json": tests_json,
                "dockerfile": dockerfile,
                "handler": handler
            }
        }
        
        return result
    except Exception as e:
        logging.error(f"Error during repository analysis: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {str(e)}")
    finally:
        # Clean up temporary directory
        if 'temp_path' in locals():
            try:
                shutil.rmtree(temp_path)
                logging.info(f"Cleaned up temporary directory: {temp_path}")
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the index page."""
    try:
        index_path = Path(__file__).parent / "static" / "index.html"
        if not index_path.exists():
            logger.error(f"Index file not found at {index_path}")
            raise HTTPException(status_code=404, detail="Index file not found")
            
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/analyze")
async def analyze_get(repo_url: str = Query(..., description="GitHub repository URL")):
    """Analyze a GitHub repository and generate RunPod files (GET endpoint)."""
    return perform_repository_analysis(repo_url)

@app.post("/analyze")
async def analyze(
    request: Request,
    repo_url: Optional[str] = Form(None),
    repo_url_param: Optional[str] = None
):
    """Analyze a GitHub repository and generate RunPod files."""
    try:
        # Log request information
        content_type = request.headers.get('content-type', '')
        logging.info(f"POST analyze called with content-type: {content_type}")
        
        # Get repo_url from request body, form data, or query parameter
        final_repo_url = None
        
        # Try to get JSON data
        if "application/json" in content_type:
            try:
                json_data = await request.json()
                logging.info(f"JSON data received: {json_data}")
                if json_data and isinstance(json_data, dict):
                    # Check for repo_url or url in the JSON data
                    if 'repo_url' in json_data:
                        final_repo_url = json_data['repo_url'].strip()
                        logging.info(f"Using repo_url from JSON body: {final_repo_url}")
                    elif 'url' in json_data:
                        final_repo_url = json_data['url'].strip()
                        logging.info(f"Using url from JSON body: {final_repo_url}")
            except Exception as e:
                logging.error(f"Error parsing JSON data: {str(e)}")
        # Try form data
        elif repo_url:
            final_repo_url = repo_url.strip()
            logging.info(f"Using repo_url from form: {final_repo_url}")
        # Try query parameter
        elif repo_url_param:
            final_repo_url = repo_url_param.strip()
            logging.info(f"Using repo_url from query param: {final_repo_url}")
                    
        # Use standard analysis for all repositories
        return perform_repository_analysis(final_repo_url)
        
    except HTTPException:
        # Re-raise HTTP exceptions with their original status code
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 