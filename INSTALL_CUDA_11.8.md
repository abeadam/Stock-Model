# Installing CUDA 11.8 Support

## Important: You Don't Need Full CUDA Toolkit!

**PyTorch comes with CUDA libraries bundled.** You just need to install PyTorch built with CUDA 11.8 support, not the full CUDA toolkit.

## Option 1: Install PyTorch with CUDA 11.8 Support (Recommended - Easier)

This is what you actually need:

```bash
cd /home/wiseguy/dev
source venv/bin/activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**This is usually enough!** PyTorch includes all the CUDA libraries it needs.

## Option 2: Install Full CUDA 11.8 Toolkit (Only if Option 1 Doesn't Work)

If PyTorch still can't access GPU after Option 1, you may need the full toolkit:

### For Ubuntu/Debian:

1. **Download CUDA 11.8 from NVIDIA:**
   - Go to: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Select: Linux → x86_64 → Ubuntu → 22.04 (or your version)
   - Download the `.deb` installer

2. **Install:**
```bash
# Install the .deb package
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

3. **Set environment variables** (add to `~/.bashrc`):
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

4. **Reload:**
```bash
source ~/.bashrc
nvcc --version  # Should show CUDA 11.8
```

### Alternative: Use Package Manager

```bash
# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```

## Recommendation

**Start with Option 1** (just install PyTorch with cu118). This is usually sufficient because:
- PyTorch bundles its own CUDA libraries
- You don't need the full toolkit for PyTorch to work
- Much simpler and faster

Only try Option 2 if Option 1 doesn't work after testing.

## Quick Test Script

I've created `QUICK_FIX_NOW.sh` that does Option 1 automatically:

```bash
./QUICK_FIX_NOW.sh
```


