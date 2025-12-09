# Quick Fix for GPU Issue

## The Problem
PyTorch shows "CUDA unknown error" and cannot access your GPU, even though:
- ✅ GPU is physically present (RTX 3060)
- ✅ NVIDIA drivers installed
- ✅ CUDA libraries present
- ✅ PyTorch has CUDA support (2.6.0+cu124)

## The Solution

PyTorch can't find the CUDA driver library (`libcuda.so`) at runtime. You need to set `LD_LIBRARY_PATH` **before** running Python.

### Option 1: Use the GPU Script (Easiest)

```bash
./run_with_gpu.sh
```

This script automatically sets the library path and runs your model.

### Option 2: Manual Fix (Permanent)

Add to your `~/.bashrc` or `~/.profile`:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Then:
```bash
source ~/.bashrc  # or restart terminal
cd /home/wiseguy/dev
source venv/bin/activate
python3 spx_prediction_model.py
```

### Option 3: Quick Test

Test if this fixes it:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source venv/bin/activate
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

If it prints `CUDA: True`, the fix works!

## Why This Happens

PyTorch loads CUDA libraries when it's imported. If `libcuda.so` (the driver library) isn't in the library search path, PyTorch can't initialize CUDA, even though the GPU and drivers are fine.

## After Fixing

Once CUDA works, you'll see:
- GPU utilization increasing during training
- Much faster training (GPU is 10-100x faster than CPU)
- Model device showing `cuda:0`


