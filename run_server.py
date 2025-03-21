"""Run the FastAPI server."""
import os
import sys
import socket
import time
import logging
from runpod_analyzer.web.app import app
import uvicorn

def is_port_in_use(port: int) -> bool:
    """Check if the specified port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

if __name__ == "__main__":
    # Find an available port
    try:
        port = int(os.environ.get('PORT', 8000))
        if is_port_in_use(port):
            print(f"Port {port} is already in use. Looking for an available port...")
            port = find_available_port(port)
        print(f"Starting server on port {port}")
        uvicorn.run(app, host="127.0.0.1", port=port)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1) 