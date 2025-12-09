# Install CUDA Toolkit to Fix GPU Issue

## Current Problem
PyTorch has CUDA support but can't initialize CUDA. This usually means the **CUDA Toolkit** is missing (different from CUDA runtime libraries).

## Solution: Install CUDA Toolkit

### Option 1: Install via Package Manager (Easier)

```bash
# Update package list
sudo apt update

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

### Option 2: Install from NVIDIA (Recommended for latest version)

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select:
   - Linux
   - x86_64
   - Ubuntu (or your distro)
   - Version 12.x (matches your driver)
3. Follow installation instructions

### After Installation

1. **Set environment variables** (add to `~/.bashrc`):
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

2. **Reload shell**:
```bash
source ~/.bashrc
```

3. **Test**:
```bash
cd /home/wiseguy/dev
source venv/bin/activate
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Why This Is Needed

PyTorch needs:
- ✅ CUDA runtime libraries (you have these)
- ✅ CUDA driver (you have this)
- ❌ CUDA toolkit (likely missing) - provides `nvcc` and development libraries

The toolkit provides the development headers and libraries that PyTorch needs to communicate with CUDA.

## Alternative: Train on CPU

If installing CUDA toolkit is not possible, the model will automatically use CPU (slower but functional):

```bash
python3 spx_prediction_model.py
```

Training will be slower but will work.


