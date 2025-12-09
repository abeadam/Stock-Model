#!/bin/bash

# Script to start Jupyter Notebook for local network access
# Usage: ./start_jupyter_local.sh [port]

set -e

# Default port
PORT=${1:-8888}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: venv directory not found. Please run this script from the project root."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Error: Jupyter is not installed. Installing..."
    pip install jupyter notebook
fi

# Generate config if it doesn't exist
CONFIG_DIR="$HOME/.jupyter"
CONFIG_FILE="$CONFIG_DIR/jupyter_notebook_config.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generating Jupyter configuration..."
    jupyter notebook --generate-config
    
    # Set some default options for network access
    cat >> "$CONFIG_FILE" << EOF

# Configuration for local network access
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = False
EOF
    echo "Configuration file created at $CONFIG_FILE"
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Warning: Port $PORT is already in use."
    echo "Please choose a different port or stop the existing process."
    echo "Usage: ./start_jupyter_local.sh [port]"
    exit 1
fi

# Get the server's IP address on the local network
# Try multiple methods to get the IP
SERVER_IP=""
if command -v hostname &> /dev/null; then
    # Try to get IP from hostname
    SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
fi

if [ -z "$SERVER_IP" ]; then
    # Try ip command
    SERVER_IP=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' || echo "")
fi

if [ -z "$SERVER_IP" ]; then
    # Try ifconfig
    SERVER_IP=$(ifconfig 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1 || echo "")
fi

echo ""
echo "=========================================="
echo "Starting Jupyter Notebook on port $PORT"
echo "=========================================="
echo ""
if [ -n "$SERVER_IP" ]; then
    echo "Server IP address: $SERVER_IP"
    echo ""
    echo "Access from other computers on your local network:"
    echo "  http://$SERVER_IP:$PORT"
    echo ""
fi
echo "Access from this computer:"
echo "  http://localhost:$PORT"
echo ""
echo "IMPORTANT SECURITY NOTES:"
echo "  - Anyone on your local network can access this Jupyter server"
echo "  - Make sure to use the token/password for authentication"
echo "  - Consider setting a password: jupyter notebook password"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start Jupyter
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0

