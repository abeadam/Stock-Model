#!/bin/bash
# Script to run the model with GPU support

# Set CUDA library paths BEFORE Python imports torch
# Include both runtime and driver libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Activate virtual environment
cd /home/wiseguy/dev
source venv/bin/activate

# Verify CUDA is available
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 | grep -v "UserWarning" | grep -v "Triggered"

# Run the model
echo "Starting training..."
python3 spx_prediction_model.py

