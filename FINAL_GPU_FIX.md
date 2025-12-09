# Final GPU Fix - PyTorch CUDA Initialization Error

## Current Status
- ✅ GPU: RTX 3060 (healthy)
- ✅ Driver: 580.95.05 (installed)
- ✅ CUDA libraries: Present
- ❌ PyTorch: Cannot initialize CUDA ("CUDA unknown error")

## Root Cause
PyTorch 2.6.0+cu124 may have a compatibility issue with your system's CUDA setup. The "CUDA unknown error" typically means PyTorch can't communicate with the CUDA driver.

## Solution: Reinstall PyTorch

### Step 1: Complete Uninstall
```bash
cd /home/wiseguy/dev
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip cache purge  # Clear pip cache
```

### Step 2: Install Compatible Version

**Option A: CUDA 12.1 (Recommended - try this first)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option B: CUDA 11.8 (More stable, widely compatible)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option C: Latest stable with CUDA 12.4**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Verify
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Alternative: Use CPU for Now

If GPU still doesn't work after reinstalling, you can train on CPU (slower but functional):

```bash
# The script will automatically use CPU if CUDA is not available
python3 spx_prediction_model.py
```

Training will be slower but will work.

## Quick Test Script

Run `./fix_cuda.sh` to automatically reinstall PyTorch and test CUDA.

## Why This Happens

The "CUDA unknown error" usually means:
1. PyTorch's CUDA build doesn't match your system's CUDA version
2. Missing or incompatible CUDA driver libraries
3. PyTorch installation corruption

Reinstalling usually fixes it.


