#!/usr/bin/env python3
"""Test CUDA driver access directly"""
import ctypes
import sys
import os

print("="*60)
print("Testing CUDA Driver Access")
print("="*60)

# Try to load libcuda directly
try:
    libcuda = ctypes.CDLL('libcuda.so.1')
    print("✓ libcuda.so.1 loaded successfully")
    
    # Try to get driver version
    try:
        cuInit = libcuda.cuInit
        cuInit.argtypes = [ctypes.c_uint]
        result = cuInit(0)
        if result == 0:
            print("✓ cuInit() succeeded - CUDA driver is accessible")
        else:
            print(f"✗ cuInit() failed with error code: {result}")
    except Exception as e:
        print(f"✗ cuInit() error: {e}")
        
except Exception as e:
    print(f"✗ Cannot load libcuda.so.1: {e}")
    print("\nPossible issues:")
    print("1. libcuda.so.1 not in library path")
    print("2. Permission denied")
    print("3. Driver not properly installed")

print("="*60)


