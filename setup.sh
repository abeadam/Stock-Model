#!/bin/bash
# Setup script for SPX Prediction Model

echo "Setting up SPX Prediction Model dependencies..."

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "pip is not installed. Installing pip..."
    echo "Please run: sudo apt install python3-pip"
    exit 1
fi

# Install dependencies
echo "Installing required packages..."
if command -v pip3 &> /dev/null; then
    pip3 install --user -r requirements.txt
else
    python3 -m pip install --user -r requirements.txt
fi

echo "Setup complete!"
echo "You can now run: python3 spx_prediction_model.py"

