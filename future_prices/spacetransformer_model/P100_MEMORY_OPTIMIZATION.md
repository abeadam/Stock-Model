# P100 GPU Memory Optimization Guide

This guide explains how to maximize GPU memory usage on P100 (16GB VRAM) for faster training.

## Key Changes Made

### 1. **Aggressive Batch Size Settings**
- **Before**: `batch_size=2` (very conservative)
- **After**: `batch_size=32` (can go up to 64-128 with FP16)
- **Impact**: 16-64x larger batches = much faster training

### 2. **Full Model Capacity**
- **Before**: `d_model=64`, `d_ff=256`, `n_heads=4`, `context_length=48`
- **After**: `d_model=128`, `d_ff=512`, `n_heads=8`, `context_length=96`
- **Impact**: Better model capacity while still fitting in memory with FP16

### 3. **Enhanced DataLoader Configuration**
- **Workers**: Increased from 4 to 8 for better data throughput
- **Prefetch Factor**: Added prefetching (2 batches ahead) for larger batch sizes
- **Impact**: Keeps GPU fed with data, reducing idle time

### 4. **Memory Monitoring**
- Real-time GPU memory usage display
- Warnings when memory usage is low (suggests increasing batch size)
- **Impact**: Helps find optimal settings

### 5. **Automatic Batch Size Finder**
- Optional function to automatically find maximum batch size
- Tests progressively larger batches until OOM
- **Impact**: Automatically finds optimal settings for your specific model/data

## Recommended Settings for P100

### Conservative (Safe Start)
```python
batch_size = 16
d_model = 128
d_ff = 512
n_heads = 8
context_length = 96
use_amp = True
```

### Aggressive (Maximum Speed)
```python
batch_size = 64  # or even 128 with FP16
d_model = 128
d_ff = 512
n_heads = 8
context_length = 96
use_amp = True
gradient_accumulation_steps = 1
```

### Very Aggressive (If Memory Allows)
```python
batch_size = 128
d_model = 256  # Larger model
d_ff = 1024
n_heads = 16
context_length = 96
use_amp = True
```

## Memory Usage Breakdown (Approximate)

For a typical configuration with 434 features:

| Component | FP32 Memory | FP16 Memory |
|-----------|-------------|-------------|
| Model weights | ~200 MB | ~100 MB |
| Batch (size=32) | ~2.5 GB | ~1.25 GB |
| Gradients | ~200 MB | ~100 MB |
| Optimizer states | ~400 MB | ~200 MB |
| Activations | ~1-2 GB | ~0.5-1 GB |
| **Total** | **~4-5 GB** | **~2-2.5 GB** |

With FP16, you can use **2-4x larger batches** than FP32!

## Using the Auto Batch Size Finder

The `runner.py` now includes an optional batch size finder:

```python
# Set this to True in runner.py
USE_OPTIMAL_BATCH = True

# It will automatically test:
# batch_size = 16, 32, 64, 128
# and use the largest that fits
```

## Performance Expectations

### Before (Conservative Settings)
- Batch size: 2
- Training speed: ~X samples/sec
- GPU utilization: ~20-30%

### After (Optimized Settings)
- Batch size: 32-64
- Training speed: **10-30x faster**
- GPU utilization: **70-90%**

## Memory Optimization Tips

### 1. Use FP16 (Mixed Precision)
- **Always enabled** in optimized settings
- Saves ~50% memory
- 1.5-2x speedup on P100

### 2. Increase Batch Size Gradually
```python
# Start here
batch_size = 16

# If successful, try
batch_size = 32

# Then
batch_size = 64

# Finally (if memory allows)
batch_size = 128
```

### 3. Monitor Memory Usage
The training script now shows:
```
GPU Memory: 2.50 GB allocated, 3.20 GB reserved (20.0% utilization)
  → Consider increasing batch_size (currently using 20.0% of GPU memory)
```

### 4. Use Gradient Accumulation for Very Large Effective Batches
```python
batch_size = 32
gradient_accumulation_steps = 4
# Effective batch size = 128, but only uses memory for 32
```

## Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**
   ```python
   batch_size = 16  # or 8
   ```

2. **Use gradient accumulation**
   ```python
   batch_size = 8
   gradient_accumulation_steps = 4  # Effective batch = 32
   ```

3. **Reduce model size**
   ```python
   d_model = 64
   d_ff = 256
   n_heads = 4
   ```

4. **Reduce context length**
   ```python
   context_length = 48  # Instead of 96
   ```

### Low GPU Utilization

If you see warnings like:
```
→ Consider increasing batch_size (currently using 20.0% of GPU memory)
```

**Action**: Increase batch size until utilization is 70-90%

### Memory Fragmentation

If you get OOM errors even with small batches:

1. Restart Python process
2. Set environment variable:
   ```python
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
   ```

## Example: Finding Your Optimal Settings

```python
# Test 1: Start conservative
batch_size = 16
# If successful and memory < 50%, continue to Test 2

# Test 2: Increase batch
batch_size = 32
# If successful and memory < 70%, continue to Test 3

# Test 3: Aggressive
batch_size = 64
# If successful, you're at optimal!

# Test 4: Very aggressive (may fail)
batch_size = 128
# If OOM, use 64 from Test 3
```

## Current Default Settings in runner.py

The `runner.py` now uses these optimized defaults:

```python
batch_size = 32              # 16x larger than before!
context_length = 96          # Full context (was 48)
d_model = 128               # Medium model (was 64)
d_ff = 512                  # Standard (was 256)
n_heads = 8                 # Standard (was 4)
use_amp = True              # FP16 enabled
compile_model = True        # Compilation enabled
num_workers = 8             # More workers (was 4)
prefetch_factor = 2         # Prefetch batches
```

## Expected Speedup

| Setting | Batch Size | Speedup vs Original |
|---------|------------|---------------------|
| Original | 2 | 1x (baseline) |
| Conservative | 16 | ~8x |
| Optimized | 32 | ~16x |
| Aggressive | 64 | ~32x |
| Maximum | 128 | ~64x (if memory allows) |

**Note**: Actual speedup depends on model size, data complexity, and other factors.

## Next Steps

1. **Run with default optimized settings** (batch_size=32)
2. **Monitor GPU memory usage** during first epoch
3. **If memory < 70%**: Increase batch_size to 64
4. **If memory > 90%**: Reduce batch_size to 16
5. **If OOM**: Use gradient accumulation instead

## References

- See `GPU_OPTIMIZATION_GUIDE.md` for general GPU optimizations
- P100 specs: 16GB VRAM, Pascal architecture
- FP16 support: Native on P100


