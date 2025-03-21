#!/bin/bash

# Define help function
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Start the RunPod Repository Analyzer server"
  echo ""
  echo "Options:"
  echo "  -p, --port PORT    Specify the port to run the server on (default: 8000)"
  echo "  -h, --help         Display this help message and exit"
  echo "  -b, --background   Run the server in the background"
  echo ""
  echo "Examples:"
  echo "  $0                 # Start server on default port 8000"
  echo "  $0 -p 9000         # Start server on port 9000"
  echo "  $0 --port 9000 -b  # Start server on port 9000 in background"
}

# Set default values
PORT=8000
BACKGROUND=false
LOG_FILE="analyzer_server.log"

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
        PORT=$2
        shift 2
      else
        echo "Error: Port must be a number" >&2
        exit 1
      fi
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -b|--background)
      BACKGROUND=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

# Set environment variable for the server to use
export PORT="$PORT"

# Check if Python and required packages are installed
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 is not installed" >&2
  exit 1
fi

# Ensure we're in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || { echo "Error: Could not change to project root directory"; exit 1; }

echo "Starting RunPod Repository Analyzer server on port $PORT..."

# Run the server
if [ "$BACKGROUND" = true ]; then
  echo "Running in background mode. Logs will be written to $LOG_FILE"
  python3 run_server.py > "$LOG_FILE" 2>&1 &
  
  # Save the PID to a file for later management
  echo $! > scripts/server.pid
  echo "Server started with PID $(cat scripts/server.pid)"
  echo "To stop the server, run: scripts/stop_server.sh"
else
  # Run in foreground
  python3 run_server.py
fi 