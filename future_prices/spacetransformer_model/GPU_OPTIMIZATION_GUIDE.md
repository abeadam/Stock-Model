# GPU Optimization Guide for P100 Training

This guide explains the GPU optimizations added to maximize training speed on P100 GPUs.

## Key Optimizations Implemented

### 1. **Automatic Mixed Precision (FP16) Training**
- **What it does**: Uses FP16 (half precision) for forward passes, FP32 for loss computation
- **Speedup**: ~1.5-2x faster training on P100
- **Memory savings**: ~50% reduction in GPU memory usage
- **How to use**: Set `use_amp=True` (default)

### 2. **Optimized DataLoader Configuration**
- **`num_workers`**: Parallel data loading (up to 4 workers)
- **`pin_memory=True`**: Faster CPU→GPU transfers
- **`persistent_workers=True`**: Keeps workers alive between epochs
- **Speedup**: 10-30% faster data loading

### 3. **Gradient Accumulation**
- **What it does**: Accumulates gradients over multiple batches before updating weights
- **Benefit**: Simulates larger batch sizes without using more GPU memory
- **How to use**: Set `gradient_accumulation_steps=N` (e.g., 2, 4, 8)
- **Effective batch size**: `batch_size * gradient_accumulation_steps`

### 4. **Model Compilation (PyTorch 2.0+)**
- **What it does**: Compiles model to optimized kernels using `torch.compile()`
- **Speedup**: 10-30% faster inference/forward passes
- **How to use**: Set `compile_model=True` (requires PyTorch 2.0+)

### 5. **Non-blocking GPU Transfers**
- **What it does**: Overlaps data transfer with computation
- **Speedup**: 5-10% improvement in overall throughput

## Usage Examples

### Basic Usage (All Optimizations Enabled)
```python
from spacetransformer_model.spacetransformer import main

model, history, scaler, feature_names = main(
    data_path='../es_with_indicators.csv',
    context_length=96,
    target_length=24,
    batch_size=16,  # Can use larger batch with FP16
    use_amp=True,  # Enable mixed precision
    gradient_accumulation_steps=2,  # Effective batch size = 32
    compile_model=True  # Compile model (PyTorch 2.0+)
)
```

### Conservative Settings (If OOM Errors)
```python
model, history, scaler, feature_names = main(
    data_path='../es_with_indicators.csv',
    context_length=96,
    target_length=24,
    batch_size=8,  # Smaller batch
    use_amp=True,  # Still use FP16 for memory savings
    gradient_accumulation_steps=4,  # Effective batch size = 32
    compile_model=False  # Skip compilation if issues
)
```

### Maximum Performance (If GPU Memory Allows)
```python
model, history, scaler, feature_names = main(
    data_path='../es_with_indicators.csv',
    context_length=96,
    target_length=24,
    batch_size=32,  # Larger batch
    use_amp=True,
    gradient_accumulation_steps=1,  # No accumulation needed
    compile_model=True
)
```

## P100-Specific Recommendations

### Memory Constraints
- P100 has 16GB VRAM
- With FP16, you can typically use 2x larger batch sizes
- Start with `batch_size=16` and increase if memory allows

### Optimal Settings for P100
```python
# Recommended starting point
batch_size = 16
use_amp = True
gradient_accumulation_steps = 2  # Effective batch = 32
compile_model = True  # If PyTorch 2.0+
num_workers = 4  # For DataLoader (auto-set)
```

### Batch Size Guidelines
- **Small model** (d_model=64, n_heads=4): `batch_size=32-64`
- **Medium model** (d_model=128, n_heads=8): `batch_size=16-32`
- **Large model** (d_model=256, n_heads=16): `batch_size=8-16`

## Performance Monitoring

The training script now prints:
- GPU name and memory
- Mixed precision status
- Effective batch size
- CUDA version

Monitor GPU utilization with:
```bash
watch -n 1 nvidia-smi
```

## Expected Speedups

| Optimization | Speedup | Memory Savings |
|-------------|---------|----------------|
| FP16 (AMP) | 1.5-2x | ~50% |
| DataLoader optimization | 1.1-1.3x | None |
| Gradient accumulation | Variable | Allows larger effective batch |
| Model compilation | 1.1-1.3x | None |
| **Combined** | **2-3x** | **~50%** |

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce model size (`d_model`, `n_heads`, `d_ff`)
4. Reduce `context_length`

### Slow Training
1. Ensure `use_amp=True`
2. Check `num_workers` is set (auto-configured)
3. Verify `pin_memory=True` (auto-enabled for CUDA)
4. Try `compile_model=True` if PyTorch 2.0+

### Numerical Instabilities
- If FP16 causes NaN/Inf, try reducing learning rate
- Or disable AMP: `use_amp=False`

## Additional Tips

1. **Warm-up**: First epoch may be slower due to CUDA initialization
2. **Memory fragmentation**: Restart Python if memory issues persist
3. **Data loading**: Ensure data is on fast storage (SSD)
4. **CUDA version**: P100 works best with CUDA 11.0+

## Example: Finding Optimal Batch Size

```python
# Start small and increase
for batch_size in [8, 16, 32, 64]:
    try:
        model, history, scaler, feature_names = main(
            batch_size=batch_size,
            use_amp=True,
            gradient_accumulation_steps=1
        )
        print(f"✓ Success with batch_size={batch_size}")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ OOM with batch_size={batch_size}")
            continue
        raise
```

## References

- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [torch.compile Documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [DataLoader Performance Tuning](https://pytorch.org/docs/stable/data.html)


