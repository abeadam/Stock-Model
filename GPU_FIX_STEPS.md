# Steps to Fix GPU Access

## Current Issue
PyTorch cannot communicate with CUDA driver ("CUDA unknown error"). This is a system-level issue, not a PyTorch issue.

## Step-by-Step Fix

### Step 1: Check Permissions

Run:
```bash
./fix_gpu_permissions.sh
```

If you're not in `video` or `render` groups, add yourself:
```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
```

**Then log out and back in** (or reboot) for groups to take effect.

### Step 2: Test CUDA Driver Access

```bash
python3 test_cuda_driver.py
```

This tests if Python can access the CUDA driver directly.

### Step 3: Reboot System

**This is often the fix!** CUDA driver state can get stuck, and a reboot clears it:

```bash
sudo reboot
```

After reboot, test:
```bash
cd /home/wiseguy/dev
source venv/bin/activate
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 4: Reinstall PyTorch with CUDA (After Reboot)

If reboot fixes it, reinstall PyTorch with CUDA:

```bash
cd /home/wiseguy/dev
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 5: If Still Failing - Check Driver

```bash
# Check driver status
nvidia-smi

# Check driver logs
dmesg | grep -i nvidia | tail -20

# Verify driver files
ls -la /usr/lib/x86_64-linux-gnu/libcuda.so*
```

### Step 6: Reinstall NVIDIA Driver (Last Resort)

If nothing else works:

```bash
# Check current driver
nvidia-smi

# Reinstall driver
sudo apt update
sudo apt install --reinstall nvidia-driver-580
sudo reboot
```

## Most Likely Fix

**Reboot your system.** The CUDA driver state is likely stuck, and a reboot usually fixes it.

After reboot:
1. Reinstall PyTorch with CUDA 11.8
2. Test CUDA access
3. Should work!

## Why This Happens

The "CUDA unknown error" means PyTorch's CUDA runtime can't call the CUDA driver. This is usually:
- Driver state issue (reboot fixes)
- Permission issue (groups fix)
- Library path issue (environment variables)

Reboot is the most common fix.


