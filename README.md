# RunPod Repository Analyzer

A tool to analyze repositories and generate RunPod-required files for the RunPod Hub (`hub.json`, `tests.json`, `Dockerfile`, `handler.py`).

## Features

- **Repository Analysis**: Automatically detects repository type (ML Model, API) and programming languages used.
- **Dependency Detection**: Identifies dependencies from various package management files (`requirements.txt`, `setup.py`, `pyproject.toml`).
- **File Generation**:
  - `hub.json`: Metadata for RunPod Hub
  - `tests.json`: Test payloads for API testing
  - `Dockerfile`: Container configuration
  - `handler.py`: RunPod serverless handler
- **Port Conflict Resolution**: Automatically finds an available port if the default port (8000) is already in use.
- **Error Handling**: Improved error handling with unique error IDs for better debugging.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/runpod-repo-analyzer.git
cd runpod-repo-analyzer

# Install the package
pip install -e .
```

## Usage

### Web Server

```bash
# Start the web server
python run_server.py

# The server will start on port 8000 by default
# If port 8000 is in use, it will automatically find an available port
```

### Command Line Interface

```bash
# Basic usage
runpod-analyze https://github.com/username/repository.git

# Specify output directory
runpod-analyze -o ./output https://github.com/username/repository.git

# Enable verbose output
runpod-analyze -v https://github.com/username/repository.git
```

### Generated Files

1. **hub.json**: Contains metadata about your project
   ```json
   {
     "version": "1.0.0",
     "title": "Project Title",
     "description": "Project description from README",
     "tags": ["python", "machine-learning"],
     "env": [
       {
         "name": "MODEL_NAME",
         "description": "Name of the model to use",
         "default": "bert-base-uncased"
       }
     ]
   }
   ```

2. **tests.json**: Contains test payloads
   ```json
   {
     "version": "1.0.0",
     "tests": [
       {
         "input": {
           "text": "Sample input text"
         },
         "output": {
           "embeddings": []
         }
       }
     ]
   }
   ```

3. **Dockerfile**: Container configuration
   ```dockerfile
   FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   CMD ["python", "handler.py"]
   ```

4. **handler.py**: RunPod serverless handler
   ```python
   import runpod
   
   @runpod.handler
   def handler(event):
       try:
           input_data = event.get('input', {})
           # Process input and return results
           return {"status": "success", "output": {}}
       except Exception as e:
           return {"error": str(e)}
   
   if __name__ == "__main__":
       runpod.serverless.start({"handler": handler})
   ```

## Repository Type Detection

The tool uses various indicators to determine the repository type:

1. **ML Model Repository**:
   - Keywords in README: model, train, predict, inference, dataset
   - Dependencies: torch, tensorflow, keras, sklearn
   - File patterns: model files, training scripts

2. **API Repository**:
   - Keywords in README: api, endpoint, route, server
   - Dependencies: flask, fastapi, django, express
   - File patterns: route definitions, API documentation

## Server Scripts

The repository includes several scripts to help manage the server:

### Starting the Server

```bash
# Start server on default port 8000
./scripts/start_server.sh

# Start server on a specific port
./scripts/start_server.sh -p 9000

# Start server in background mode
./scripts/start_server.sh -b

# Get help on script options
./scripts/start_server.sh --help
```

### Checking Server Status

```bash
# Check basic server status
./scripts/status_server.sh

# Check detailed server status
./scripts/status_server.sh -v
```

### Stopping the Server

```bash
# Stop server gracefully (when running in background)
./scripts/stop_server.sh

# Force stop server (when running in background)
./scripts/stop_server.sh -f
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details 
