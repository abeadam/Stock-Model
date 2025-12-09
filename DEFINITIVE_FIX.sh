#!/bin/bash
# Definitive fix for CUDA issue - reinstall PyTorch properly

cd /home/wiseguy/dev
source venv/bin/activate

echo "=========================================="
echo "DEFINITIVE CUDA FIX"
echo "=========================================="
echo ""
echo "Current PyTorch version:"
pip show torch | grep Version
echo ""

echo "Step 1: Complete uninstall..."
pip uninstall torch torchvision torchaudio -y
pip cache purge

echo ""
echo "Step 2: Installing PyTorch with CUDA 11.8..."
echo "This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 3: Testing CUDA..."
python3 -c "
import torch
import sys
print('='*60)
print('INSTALLATION RESULT')
print('='*60)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA in build: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print('')
    print('✓ SUCCESS! GPU is working!')
    print('You can now run: python3 spx_prediction_model.py')
else:
    print('')
    print('✗ GPU still not available')
    print('')
    print('Trying CPU-only installation...')
    print('='*60)
    sys.exit(1)
print('='*60)
" || {
    echo ""
    echo "CUDA 11.8 didn't work. Installing CPU-only PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    echo ""
    echo "CPU-only PyTorch installed. Training will use CPU (slower but works)."
    echo "Run: python3 spx_prediction_model.py"
}


