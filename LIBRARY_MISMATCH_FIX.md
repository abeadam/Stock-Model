# Fix CUDA Library Version Mismatch

## The Real Problem

The error changed from "CUDA unknown error" to:
```
undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```

This means **PyTorch's bundled CUDA libraries don't match your system's CUDA version**.

## Quick Fix Options

### Option 1: Install CPU-Only PyTorch (Guaranteed to Work)

This will work immediately, but training will be slower:

```bash
cd /home/wiseguy/dev
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Pros**: Works immediately, no compatibility issues  
**Cons**: Training is slower (CPU only)

### Option 2: Reinstall PyTorch with Matching CUDA

Try different CUDA versions until one works:

```bash
cd /home/wiseguy/dev
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y

# Try CUDA 11.8 (most stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Option 3: Use the Fix Script

```bash
./fix_cuda_library_mismatch.sh
```

## Why This Happens

PyTorch bundles its own CUDA libraries. If these don't match your system's CUDA toolkit version, you get symbol mismatches.

Your system has:
- CUDA toolkit (nvcc works)
- NVIDIA driver 580.95.05
- But PyTorch's libraries are incompatible

## Recommendation

**For now, use CPU-only PyTorch** to get training working. You can fix GPU later:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Training will be slower but will work. Once training is working, you can troubleshoot GPU compatibility separately.


