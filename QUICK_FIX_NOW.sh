#!/bin/bash
# Quick fix: Reinstall PyTorch to match system CUDA 12.0

cd /home/wiseguy/dev
source venv/bin/activate

echo "Your system has CUDA 12.0, but PyTorch was built for CUDA 12.1"
echo "Reinstalling PyTorch with compatible version..."
echo ""

pip uninstall torch torchvision torchaudio -y

echo ""
echo "Installing PyTorch with CUDA 11.8 (most compatible with CUDA 12.0)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Testing..."
python3 -c "
import torch
print('='*50)
print('RESULT')
print('='*50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✓ SUCCESS! GPU is working!')
else:
    print('⚠ GPU not available - will use CPU')
    print('Training will work but be slower')
print('='*50)
"

echo ""
echo "If CUDA is still False, try CPU-only:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"


