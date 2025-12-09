#!/bin/bash
# Fix CUDA library version mismatch

echo "=========================================="
echo "Fixing CUDA Library Version Mismatch"
echo "=========================================="

cd /home/wiseguy/dev
source venv/bin/activate

echo ""
echo "The error shows PyTorch's CUDA libraries don't match your system."
echo "This is a version compatibility issue."
echo ""

echo "Step 1: Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "Step 2: Installing PyTorch with CPU-only (will work, but slower)..."
echo "OR install matching CUDA version..."
echo ""

# Option 1: CPU-only (guaranteed to work)
read -p "Install CPU-only PyTorch? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo ""
    echo "CPU-only PyTorch installed. Training will use CPU (slower)."
else
    echo ""
    echo "Trying CUDA 11.8 (most compatible)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo ""
    echo "If this still doesn't work, you may need to:"
    echo "1. Install matching CUDA toolkit version"
    echo "2. Or use CPU-only PyTorch"
fi

echo ""
echo "Step 3: Testing..."
python3 -c "
import torch
print('='*50)
print('PyTorch Installation Check')
print('='*50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✓ GPU is working!')
else:
    print('⚠ Using CPU (slower but will work)')
print('='*50)
"


