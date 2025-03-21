"""RunPod serverless handler generation functionality."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from .types import RepoType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def _generate_imports(dependencies: set) -> str:
    """Generate import statements based on detected dependencies."""
    imports = ["import runpod", "import os", "import json"]
    
    if 'torch' in dependencies:
        imports.extend([
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F"
        ])
    elif 'tensorflow' in dependencies:
        imports.extend([
            "import tensorflow as tf",
            "import numpy as np"
        ])
    elif 'transformers' in dependencies:
        imports.extend([
            "from transformers import AutoTokenizer, AutoModel",
            "import torch"
        ])
    
    return "\n".join(imports)

def _generate_handler_logic(repo_type: RepoType) -> str:
    """Generate the specific handler logic based on repository type."""
    if repo_type == RepoType.ML_MODEL or repo_type == RepoType.ML_FRAMEWORK:
        return """
        # Get the model path from environment variable
        model_path = os.environ.get('MODEL_PATH', 'model')
        device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # TODO: Load and run your model based on input_data
        # This is a placeholder implementation
        result = {
            "predictions": [0.1, 0.2, 0.7],
            "input_processed": input_data,
            "model_info": {
                "path": model_path,
                "device": device
            }
        }
        """
    elif repo_type == RepoType.API:
        return """
        # Process API request
        # TODO: Implement your API logic here
        result = {
            "success": "true",
            "message": "API request processed",
            "data": input_data
        }
        """
    else:
        return """
        # Generic processing logic
        # TODO: Implement your specific logic here
        result = {
            "success": "true",
            "message": "Request processed successfully",
            "data": input_data
        }
        """

def generate_handler(output_dir: Path, repo_type: RepoType, dependencies: Set[str]) -> str:
    """Generate handler.py file for the RunPod serverless handler."""
    import_section = _generate_imports(dependencies)
    handler_logic = _generate_handler_logic(repo_type)
    
    # Create the handler content without using complex f-strings
    handler_content = "# RunPod Serverless Handler\n"
    handler_content += import_section + "\n"
    handler_content += "import os\n"
    handler_content += "import time\n"
    handler_content += "import json\n"
    handler_content += "import logging\n"
    handler_content += "import traceback\n"
    handler_content += "import runpod\n\n"
    
    handler_content += "# Configure logging\n"
    handler_content += "logging.basicConfig(\n"
    handler_content += "    level=logging.INFO,\n"
    handler_content += "    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\n"
    handler_content += ")\n"
    handler_content += "logger = logging.getLogger(__name__)\n\n"
    
    handler_content += "# Input validation functions\n"
    handler_content += "def validate_input(event):\n"
    handler_content += '    """Validate that the input data contains the required fields."""\n'
    handler_content += "    if not isinstance(event, dict):\n"
    handler_content += '        return False, "Input must be a dictionary"\n'
    handler_content += '    \n'
    handler_content += '    if "input" not in event:\n'
    handler_content += '        return False, "Input dictionary is required"\n'
    handler_content += '        \n'
    handler_content += '    return True, ""\n\n'
    
    handler_content += "def sanitize_input(data):\n"
    handler_content += '    """Sanitize input data to prevent security issues."""\n'
    handler_content += "    if isinstance(data, str):\n"
    handler_content += "        # Remove potentially dangerous characters\n"
    handler_content += "        return data.replace(';', '').replace('|', '').replace('&', '')\n"
    handler_content += "    elif isinstance(data, dict):\n"
    handler_content += "        return {key: sanitize_input(value) for key, value in data.items()}\n"
    handler_content += "    elif isinstance(data, list):\n"
    handler_content += "        return [sanitize_input(item) for item in data]\n"
    handler_content += "    else:\n"
    handler_content += "        return data\n\n"
    
    handler_content += "# Main handler function\n"
    handler_content += "@runpod.handler\n"
    handler_content += "def handler(event):\n"
    handler_content += f'    """Process incoming requests for the {repo_type.value} model."""\n'
    handler_content += '    request_id = f"req-{int(time.time() * 1000)}-{os.getpid()}"\n'
    handler_content += '    logger.info(f"[{request_id}] Processing request")\n'
    handler_content += '    \n'
    handler_content += '    try:\n'
    handler_content += '        # Validate and sanitize input\n'
    handler_content += '        is_valid, error_msg = validate_input(event)\n'
    handler_content += '        if not is_valid:\n'
    handler_content += '            logger.error(f"[{request_id}] Invalid input: {error_msg}")\n'
    handler_content += '            return {"error": error_msg}\n'
    handler_content += '            \n'
    handler_content += '        input_data = sanitize_input(event.get("input", {}))\n'
    handler_content += '        \n'
    handler_content += f'        # Process based on repository type\n'
    handler_content += f'        logger.info(f"[{{request_id}}] Processing {repo_type.value} request")\n'
    handler_content += f'{handler_logic}\n'
    handler_content += '            \n'
    handler_content += '        return {"status": "success", "output": result}\n'
    handler_content += '        \n'
    handler_content += '    except Exception as e:\n'
    handler_content += '        error_id = f"error-{id(e)}"\n'
    handler_content += '        logger.error(f"[{request_id}] Unhandled exception [{error_id}]: {str(e)}")\n'
    handler_content += '        logger.error(traceback.format_exc())\n'
    handler_content += '        \n'
    handler_content += '        # Return error with identifier but without exposing implementation details\n'
    handler_content += '        return {\n'
    handler_content += '            "status": "error",\n'
    handler_content += '            "error": "An unexpected error occurred",\n'
    handler_content += '            "error_id": error_id  # For log correlation\n'
    handler_content += '        }\n\n'
    
    handler_content += 'if __name__ == "__main__":\n'
    handler_content += '    runpod.serverless.start({"handler": handler})\n'

    handler_path = output_dir / "handler.py"
    with open(handler_path, "w") as f:
        f.write(handler_content)
    logger.info(f"Generated handler.py: {handler_path}")
    
    return handler_content 