#!/bin/bash

# Script to start Jupyter Notebook for SSH access
# Usage: ./start_jupyter_ssh.sh [port]

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
    
    # Set some default options for remote access
    cat >> "$CONFIG_FILE" << EOF

# Configuration for SSH access
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
    echo "Usage: ./start_jupyter_ssh.sh [port]"
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting Jupyter Notebook on port $PORT"
echo "=========================================="
echo ""
echo "On your local machine, run:"
echo "  ssh -L $PORT:localhost:$PORT user@remote-server"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start Jupyter
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0

