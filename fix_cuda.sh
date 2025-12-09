#!/bin/bash
# Script to fix CUDA/PyTorch compatibility issue

echo "=========================================="
echo "Fixing PyTorch CUDA Installation"
echo "=========================================="

cd /home/wiseguy/dev
source venv/bin/activate

echo ""
echo "Step 1: Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.1 support (more compatible)..."
# Try CUDA 12.1 first (more stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# If that doesn't work, uncomment the line below to try CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 3: Verifying installation..."
python3 -c "
import torch
print('='*50)
print('PyTorch Installation Check')
print('='*50)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA built-in: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print('✓ GPU is working!')
else:
    print('✗ GPU is NOT available')
    print('')
    print('If GPU is still not available, try:')
    print('  1. Check NVIDIA driver: nvidia-smi')
    print('  2. Install CUDA toolkit if missing')
    print('  3. Check: python3 -c \"import torch; print(torch.cuda.is_available())\"')
print('='*50)
"

echo ""
echo "Done! If CUDA is now available, you can run:"
echo "  python3 spx_prediction_model.py"

