#!/bin/bash

# Script to stop the RunPod Repository Analyzer server running in the background

# Define help function
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Stop the RunPod Repository Analyzer server running in the background"
  echo ""
  echo "Options:"
  echo "  -h, --help         Display this help message and exit"
  echo "  -f, --force        Force kill the server process"
}

# Set default values
FORCE=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -f|--force)
      FORCE=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

# Ensure we're in the scripts directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Error: Could not change to scripts directory"; exit 1; }

# Check if server.pid file exists
if [ ! -f "server.pid" ]; then
  echo "Error: No server.pid file found. Is the server running in background mode?"
  echo "You might need to find and kill the process manually."
  exit 1
fi

# Read the PID from file
PID=$(cat server.pid)

if ! ps -p "$PID" > /dev/null; then
  echo "Process with PID $PID is not running."
  echo "Removing stale server.pid file."
  rm server.pid
  exit 0
fi

echo "Stopping RunPod Repository Analyzer server (PID: $PID)..."

if [ "$FORCE" = true ]; then
  # Force kill
  kill -9 "$PID" 2>/dev/null
  KILL_STATUS=$?
else
  # Graceful shutdown
  kill "$PID" 2>/dev/null
  KILL_STATUS=$?
  
  # Wait for up to 5 seconds for the process to terminate
  COUNTDOWN=5
  while ps -p "$PID" > /dev/null && [ $COUNTDOWN -gt 0 ]; do
    echo "Waiting for server to shut down... ($COUNTDOWN seconds remaining)"
    sleep 1
    COUNTDOWN=$((COUNTDOWN-1))
  done
  
  # Force kill if it's still running
  if ps -p "$PID" > /dev/null; then
    echo "Server did not shut down gracefully. Force killing..."
    kill -9 "$PID" 2>/dev/null
  fi
fi

if [ $KILL_STATUS -eq 0 ]; then
  echo "Server stopped successfully."
  rm server.pid
else
  echo "Error: Failed to stop server. Process may no longer exist."
  echo "Removing server.pid file."
  rm server.pid
fi 