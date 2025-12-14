"""
Benchmark PyTorch vs ONNX Runtime inference performance.

Compares:
- PyTorch (uncompiled)
- PyTorch (compiled)
- ONNX Runtime (CPU)
- ONNX Runtime (GPU if available)

Usage:
    python benchmark_pytorch_vs_onnx.py --checkpoint checkpoints/spacetimeformer_best.pth --onnx model.onnx
"""

import sys
import os
import time
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.realtime_inference import RealtimePredictor

try:
    from spacetransformer_model.onnx_inference import ONNXPredictor
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime onnxruntime-gpu")


def benchmark_pytorch(
    checkpoint_path: str,
    context_data: np.ndarray,
    n_runs: int = 50,
    warmup_runs: int = 5,
    compile_model: bool = False,
    use_quantization: bool = False
) -> dict:
    """Benchmark PyTorch inference."""
    print(f"\n{'='*80}")
    print(f"PyTorch Benchmark ({'compiled' if compile_model else 'uncompiled'}, "
          f"{'quantized' if use_quantization else 'FP32'})")
    print(f"{'='*80}")
    
    # Initialize predictor
    init_start = time.time()
    predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,
        compile_model=compile_model,
        use_quantization=use_quantization
    )
    init_time = time.time() - init_start
    
    # Warmup
    if warmup_runs > 0:
        print(f"Warming up ({warmup_runs} runs)...")
        warmup_start = time.time()
        for _ in range(warmup_runs):
            _ = predictor.predict(context_data, return_scaled=True)
        warmup_time = time.time() - warmup_start
        print(f"  Warmup time: {warmup_time:.4f}s ({warmup_time/warmup_runs:.4f}s per run)")
    
    # Benchmark
    print(f"Running {n_runs} predictions...")
    start = time.time()
    for _ in range(n_runs):
        predictions = predictor.predict(context_data, return_scaled=True)
    total_time = time.time() - start
    
    avg_time = total_time / n_runs
    throughput = 1.0 / avg_time if avg_time > 0 else 0
    
    results = {
        'framework': 'PyTorch',
        'compiled': compile_model,
        'quantized': use_quantization,
        'init_time': init_time,
        'warmup_time': warmup_time if warmup_runs > 0 else 0,
        'total_time': total_time,
        'n_runs': n_runs,
        'avg_time': avg_time,
        'throughput': throughput,
        'predictions_shape': predictions.shape
    }
    
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average per prediction: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/second")
    
    return results


def benchmark_onnx(
    onnx_model_path: str,
    checkpoint_path: str,
    context_data: np.ndarray,
    n_runs: int = 50,
    warmup_runs: int = 5,
    use_gpu: bool = True
) -> dict:
    """Benchmark ONNX Runtime inference."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime not available")
    
    print(f"\n{'='*80}")
    print(f"ONNX Runtime Benchmark ({'GPU' if use_gpu else 'CPU'})")
    print(f"{'='*80}")
    
    # Initialize predictor
    init_start = time.time()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    predictor = ONNXPredictor(
        onnx_model_path=onnx_model_path,
        checkpoint_path=checkpoint_path,
        use_parallel=True,
        providers=providers
    )
    init_time = time.time() - init_start
    
    # Warmup
    if warmup_runs > 0:
        print(f"Warming up ({warmup_runs} runs)...")
        warmup_start = time.time()
        for _ in range(warmup_runs):
            _ = predictor.predict(context_data, return_scaled=True)
        warmup_time = time.time() - warmup_start
        print(f"  Warmup time: {warmup_time:.4f}s ({warmup_time/warmup_runs:.4f}s per run)")
    
    # Benchmark
    print(f"Running {n_runs} predictions...")
    start = time.time()
    for _ in range(n_runs):
        predictions = predictor.predict(context_data, return_scaled=True)
    total_time = time.time() - start
    
    avg_time = total_time / n_runs
    throughput = 1.0 / avg_time if avg_time > 0 else 0
    
    results = {
        'framework': 'ONNX Runtime',
        'device': 'GPU' if use_gpu else 'CPU',
        'init_time': init_time,
        'warmup_time': warmup_time if warmup_runs > 0 else 0,
        'total_time': total_time,
        'n_runs': n_runs,
        'avg_time': avg_time,
        'throughput': throughput,
        'predictions_shape': predictions.shape
    }
    
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Average per prediction: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {throughput:.2f} predictions/second")
    
    return results


def print_comparison(all_results: list):
    """Print comparison table of all results."""
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    # Find baseline (PyTorch uncompiled)
    baseline = None
    for r in all_results:
        if r['framework'] == 'PyTorch' and not r.get('compiled', False):
            baseline = r
            break
    
    if baseline is None:
        baseline = all_results[0]
    
    print(f"\nBaseline: {baseline['framework']} "
          f"({'compiled' if baseline.get('compiled') else 'uncompiled'}, "
          f"{'quantized' if baseline.get('quantized') else 'FP32'})")
    print(f"  Average time: {baseline['avg_time']*1000:.2f}ms")
    print(f"  Throughput: {baseline['throughput']:.2f} pred/s")
    
    print(f"\n{'Framework':<20} {'Config':<30} {'Time (ms)':<15} {'Speedup':<15} {'Throughput':<15}")
    print("-" * 95)
    
    for r in all_results:
        framework = r['framework']
        if framework == 'PyTorch':
            config = f"{'compiled' if r.get('compiled') else 'uncompiled'}, "
            config += f"{'quantized' if r.get('quantized') else 'FP32'}"
        else:
            config = r.get('device', 'CPU')
        
        time_ms = r['avg_time'] * 1000
        speedup = baseline['avg_time'] / r['avg_time'] if r['avg_time'] > 0 else 0
        throughput = r['throughput']
        
        print(f"{framework:<20} {config:<30} {time_ms:>10.2f}ms   {speedup:>10.2f}x   {throughput:>10.2f} pred/s")
    
    # Find fastest
    fastest = min(all_results, key=lambda x: x['avg_time'])
    print(f"\n🏆 Fastest: {fastest['framework']} "
          f"({fastest.get('device', '')} {fastest.get('compiled', '')} {fastest.get('quantized', '')})")
    print(f"   {fastest['avg_time']*1000:.2f}ms per prediction")
    print(f"   {fastest['throughput']:.2f} predictions/second")
    
    # Calculate speedup vs baseline
    if fastest != baseline:
        speedup = baseline['avg_time'] / fastest['avg_time']
        print(f"   {speedup:.2f}x faster than baseline")


def main():
    parser = argparse.ArgumentParser(description='Benchmark PyTorch vs ONNX Runtime')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint file'
    )
    parser.add_argument(
        '--onnx',
        type=str,
        help='Path to ONNX model file (optional, will skip ONNX benchmarks if not provided)'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=50,
        help='Number of prediction runs for benchmarking (default: 50)'
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=5,
        help='Number of warmup runs (default: 5)'
    )
    parser.add_argument(
        '--context-points',
        type=int,
        help='Context length (auto-detected from checkpoint if not provided)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        help='Number of features (auto-detected from checkpoint if not provided)'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if args.onnx and not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)
    
    # Get model dimensions
    print("Loading checkpoint to get model dimensions...")
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    context_points = args.context_points or checkpoint.get('context_points', 48)
    n_features = args.n_features or checkpoint.get('n_variables', 440)
    
    print(f"Model dimensions:")
    print(f"  Context points: {context_points}")
    print(f"  Features: {n_features}")
    
    # Create dummy context data
    context_data = np.random.randn(context_points, n_features).astype(np.float32)
    
    all_results = []
    
    # Benchmark 1: PyTorch uncompiled
    try:
        result = benchmark_pytorch(
            checkpoint_path=args.checkpoint,
            context_data=context_data,
            n_runs=args.n_runs,
            warmup_runs=args.warmup_runs,
            compile_model=False,
            use_quantization=False
        )
        all_results.append(result)
    except Exception as e:
        print(f"❌ PyTorch (uncompiled) benchmark failed: {e}")
    
    # Benchmark 2: PyTorch compiled
    try:
        result = benchmark_pytorch(
            checkpoint_path=args.checkpoint,
            context_data=context_data,
            n_runs=args.n_runs,
            warmup_runs=args.warmup_runs,
            compile_model=True,
            use_quantization=False
        )
        all_results.append(result)
    except Exception as e:
        print(f"❌ PyTorch (compiled) benchmark failed: {e}")
    
    # Benchmark 3: ONNX Runtime (CPU)
    if args.onnx and ONNX_AVAILABLE:
        try:
            result = benchmark_onnx(
                onnx_model_path=args.onnx,
                checkpoint_path=args.checkpoint,
                context_data=context_data,
                n_runs=args.n_runs,
                warmup_runs=args.warmup_runs,
                use_gpu=False
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ ONNX Runtime (CPU) benchmark failed: {e}")
    
    # Benchmark 4: ONNX Runtime (GPU)
    if args.onnx and ONNX_AVAILABLE:
        try:
            result = benchmark_onnx(
                onnx_model_path=args.onnx,
                checkpoint_path=args.checkpoint,
                context_data=context_data,
                n_runs=args.n_runs,
                warmup_runs=args.warmup_runs,
                use_gpu=True
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ ONNX Runtime (GPU) benchmark failed: {e}")
    
    # Print comparison
    if all_results:
        print_comparison(all_results)
    else:
        print("\n❌ No benchmarks completed successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()

