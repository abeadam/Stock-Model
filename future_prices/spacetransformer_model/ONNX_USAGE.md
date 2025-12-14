# ONNX Runtime Optimization Guide

This guide shows how to convert your PyTorch model to ONNX and use ONNX Runtime for faster inference.

## Installation

```bash
# Install ONNX Runtime
pip install onnxruntime          # For CPU
pip install onnxruntime-gpu     # For GPU (NVIDIA)

# Optional: Install ONNX for model validation
pip install onnx
```

## Quick Start

### Step 1: Convert PyTorch Model to ONNX

```bash
python convert_to_onnx.py \
    --checkpoint checkpoints/spacetimeformer_best.pth \
    --output model.onnx \
    --device cpu
```

This will create:
- `model.onnx` - Full model for parallel prediction
- `model_encoder.onnx` - Encoder only (for future autoregressive optimization)

### Step 2: Use ONNX Runtime for Inference

```python
from spacetransformer_model.onnx_inference import ONNXPredictor

# Create predictor
predictor = ONNXPredictor(
    onnx_model_path='model.onnx',
    checkpoint_path='checkpoints/spacetimeformer_best.pth',  # For scaler
    use_parallel=True
)

# Predict
predictions = predictor.predict(
    context_data=context_data,
    target_indices=[0, 1, 2],  # High, Low, Close
    return_scaled=False  # Return actual prices
)
```

### Step 3: Benchmark Performance

```bash
python benchmark_pytorch_vs_onnx.py \
    --checkpoint checkpoints/spacetimeformer_best.pth \
    --onnx model.onnx \
    --n-runs 50 \
    --warmup-runs 5
```

This will compare:
- PyTorch (uncompiled)
- PyTorch (compiled with torch.compile)
- ONNX Runtime (CPU)
- ONNX Runtime (GPU if available)

## Expected Performance

Typical speedups:
- **ONNX Runtime (CPU)**: 1.5-2x faster than PyTorch
- **ONNX Runtime (GPU)**: 2-3x faster than PyTorch
- **Combined with quantization**: 3-5x faster

## API Comparison

### PyTorch (RealtimePredictor)
```python
from spacetransformer_model.realtime_inference import RealtimePredictor

predictor = RealtimePredictor(
    checkpoint_path='checkpoints/spacetimeformer_best.pth',
    use_parallel=True,
    compile_model=True,
    use_quantization=False
)
```

### ONNX Runtime (ONNXPredictor)
```python
from spacetransformer_model.onnx_inference import ONNXPredictor

predictor = ONNXPredictor(
    onnx_model_path='model.onnx',
    checkpoint_path='checkpoints/spacetimeformer_best.pth',
    use_parallel=True
)
```

Both have the same `predict()` API, so switching is easy!

## Troubleshooting

### ONNX Export Fails
- Make sure model is in eval mode
- Check that all operations are ONNX-compatible
- Try different `opset_version` (default: 17)

### ONNX Runtime Not Faster
- Ensure you're using `onnxruntime-gpu` for GPU
- Check that GPU providers are available
- Warmup runs are important (first few runs are slower)

### Model Size Issues
- ONNX models are typically similar size to PyTorch
- Use quantization for smaller models (future feature)

## Next Steps

1. **Quantization**: Convert ONNX model to INT8 for 2-4x additional speedup
2. **TensorRT**: For NVIDIA GPUs, convert ONNX → TensorRT for even more speed
3. **Mobile Deployment**: Use ONNX Runtime Mobile for edge devices

## Files

- `convert_to_onnx.py` - Convert PyTorch → ONNX
- `onnx_inference.py` - ONNX Runtime inference class
- `benchmark_pytorch_vs_onnx.py` - Performance comparison tool

