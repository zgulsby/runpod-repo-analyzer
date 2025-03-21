#!/bin/bash

# Script to check the status of the RunPod Repository Analyzer server

# Define help function
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Check the status of the RunPod Repository Analyzer server"
  echo ""
  echo "Options:"
  echo "  -h, --help         Display this help message and exit"
  echo "  -v, --verbose      Show more detailed information about the server"
}

# Set default values
VERBOSE=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -v|--verbose)
      VERBOSE=true
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

# Function to check if a port is in use
is_port_in_use() {
  local port=$1
  if command -v nc &> /dev/null; then
    nc -z 127.0.0.1 "$port" &> /dev/null
    return $?
  elif command -v lsof &> /dev/null; then
    lsof -i:$port -sTCP:LISTEN &> /dev/null
    return $?
  else
    # Fallback to checking with /dev/tcp on systems where it's supported
    (echo > /dev/tcp/127.0.0.1/$port) &> /dev/null
    return $?
  fi
}

# Check for background server process
if [ -f "server.pid" ]; then
  PID=$(cat server.pid)
  if ps -p "$PID" > /dev/null; then
    echo "✅ Server is running in background mode with PID $PID"
    
    if [ "$VERBOSE" = true ]; then
      echo ""
      echo "Process details:"
      ps -p "$PID" -o pid,ppid,user,%cpu,%mem,vsz,rss,tty,stat,start,time,command
      echo ""
      
      # Check which ports are in use
      for port in {8000..8010}; do
        if is_port_in_use "$port"; then
          echo "Port $port is in use."
        fi
      done
      
      # Display the last 5 lines of the log if it exists
      if [ -f "../analyzer_server.log" ]; then
        echo ""
        echo "Last 5 lines from server log:"
        tail -n 5 "../analyzer_server.log"
      fi
    fi
  else
    echo "❌ Server PID file exists but process $PID is not running"
    echo "   You may want to remove the stale PID file: rm scripts/server.pid"
  fi
else
  # Check if the server might be running in foreground or started another way
  for port in {8000..8010}; do
    if is_port_in_use "$port"; then
      echo "⚠️  No server.pid file found, but a server appears to be running on port $port"
      if [ "$VERBOSE" = true ] && command -v lsof &> /dev/null; then
        echo ""
        echo "Process using port $port:"
        lsof -i:$port -sTCP:LISTEN
      fi
      exit 0
    fi
  done
  
  echo "❌ Server is not running"
fi 