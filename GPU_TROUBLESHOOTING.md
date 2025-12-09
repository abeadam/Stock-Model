# GPU Troubleshooting Guide

## Current Issue
PyTorch shows "CUDA unknown error" and cannot access your RTX 3060 GPU.

## Diagnosis
- ✅ GPU is physically present and healthy
- ✅ NVIDIA drivers installed (580.95.05)
- ✅ CUDA runtime libraries found (`libcudart.so.12`)
- ❌ PyTorch cannot initialize CUDA

## Root Cause
This is typically caused by:
1. **Version mismatch** between PyTorch's CUDA build and system CUDA
2. **Missing CUDA driver libraries** (different from runtime libraries)
3. **PyTorch installation issue** - CPU-only or wrong CUDA version

## Solutions (Try in Order)

### Solution 1: Reinstall PyTorch with Matching CUDA Version

Your system has CUDA 12.x libraries. Reinstall PyTorch:

```bash
cd /home/wiseguy/dev
source venv/bin/activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.4 (compatible with CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Solution 2: Install CUDA Toolkit (if missing)

If Solution 1 doesn't work, you may need the full CUDA toolkit:

```bash
# Check if CUDA toolkit is installed
nvcc --version

# If not installed, install CUDA toolkit 12.x
# Follow NVIDIA's installation guide for your Linux distribution
```

### Solution 3: Use the GPU Script

I've created `run_with_gpu.sh` that sets up the environment:

```bash
./run_with_gpu.sh
```

### Solution 4: Manual Environment Setup

Before running your script:

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda  # If CUDA toolkit is installed
source venv/bin/activate
python3 spx_prediction_model.py
```

## Quick Test

After trying solutions, test with:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## If Nothing Works

The issue might be that PyTorch 2.9.0+cu128 was built against CUDA 12.8, but your system might have a different CUDA version or missing driver libraries. In this case:

1. Check PyTorch installation: `pip show torch`
2. Try installing a different CUDA version of PyTorch
3. Consider using CPU for now (slower but will work)


