# Immediate GPU Fix - Step by Step

## Problem Identified
- ✅ GPU hardware: Working (nvidia-smi works)
- ✅ CUDA driver: Installed (libcuda.so.1 loads)
- ❌ CUDA initialization: **FAILING** (cuInit() error 999)
- ❌ User not in `video` group

## The Fix (Do These Steps)

### Step 1: Add User to Required Groups

```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
```

**Important**: You must **log out and back in** (or reboot) for group changes to take effect.

### Step 2: REBOOT SYSTEM (Critical!)

```bash
sudo reboot
```

**This is the most important step!** The CUDA driver state is stuck, and a reboot clears it. This fixes the error 999 issue in most cases.

### Step 3: After Reboot - Reinstall PyTorch with CUDA

```bash
cd /home/wiseguy/dev
source venv/bin/activate

# Uninstall CPU-only version
pip uninstall torch torchvision torchaudio -y

# Install with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Step 4: Verify GPU Works

```bash
python3 spx_prediction_model.py
```

You should see in the output:
- `Using device: cuda`
- `GPU DIAGNOSTICS` section showing GPU info
- `✓ Model is on GPU`

## Why Reboot is Necessary

The error 999 from `cuInit()` means the CUDA driver **cannot initialize**. This is a driver state issue that persists until reboot. Common causes:
- Driver state corruption
- Stuck driver locks
- Permission cache issues

**Reboot is the standard fix** for this type of CUDA driver error.

## If Reboot Doesn't Fix It

1. **Check driver after reboot:**
   ```bash
   nvidia-smi
   python3 test_cuda_driver.py
   ```

2. **Reinstall NVIDIA driver:**
   ```bash
   sudo apt update
   sudo apt install --reinstall nvidia-driver-580
   sudo reboot
   ```

3. **Check for conflicting processes:**
   ```bash
   fuser -v /dev/nvidia*
   ```

## Quick Script

Run `./REBOOT_AND_FIX.sh` to see all steps.


