# Fix GPU Not Being Used

## Problem
PyTorch cannot detect your NVIDIA GPU, even though it's physically present and healthy.

## Root Cause
The error message shows:
```
CUDA unknown error - this may be due to an incorrectly set up environment
```

This typically means:
1. PyTorch was installed **without CUDA support** (CPU-only version)
2. CUDA version mismatch between PyTorch and system
3. Environment variable issues

## Solution

### Step 1: Check Current PyTorch Installation
```bash
source venv/bin/activate
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2: Uninstall Current PyTorch
```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
```

### Step 3: Install PyTorch with CUDA Support

For CUDA 13.0 (your system), install PyTorch with CUDA 12.x (compatible):
```bash
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

OR for CUDA 11.8 (more stable):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Step 5: Check Environment Variables
```bash
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH
```

If `CUDA_VISIBLE_DEVICES` is set incorrectly, unset it:
```bash
unset CUDA_VISIBLE_DEVICES
```

## Alternative: Use System CUDA

If the above doesn't work, you may need to:
1. Install CUDA toolkit matching your driver
2. Set LD_LIBRARY_PATH to include CUDA libraries
3. Reinstall PyTorch

## Quick Test After Fix

Run your script and check:
- GPU diagnostics should show CUDA available
- Model device should be `cuda:0`
- GPU utilization should increase during training


