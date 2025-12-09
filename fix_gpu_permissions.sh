#!/bin/bash
# Fix GPU permissions and CUDA access

echo "=========================================="
echo "Fixing GPU Access Issues"
echo "=========================================="

# Check if user is in video/render groups
USER=$(whoami)
echo "Current user: $USER"
echo "Groups: $(groups)"

if ! groups | grep -q video; then
    echo ""
    echo "⚠ User not in 'video' group. Adding..."
    echo "Run this command (requires sudo):"
    echo "  sudo usermod -a -G video $USER"
    echo "Then log out and back in, or reboot"
fi

if ! groups | grep -q render; then
    echo ""
    echo "⚠ User not in 'render' group. Adding..."
    echo "Run this command (requires sudo):"
    echo "  sudo usermod -a -G render $USER"
    echo "Then log out and back in, or reboot"
fi

echo ""
echo "Checking /dev/nvidia* permissions..."
ls -la /dev/nvidia* 2>/dev/null | head -5 || echo "No /dev/nvidia* devices found"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Add user to video/render groups (if needed)"
echo "2. Reboot system (often fixes CUDA driver state)"
echo "3. After reboot, test: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""


