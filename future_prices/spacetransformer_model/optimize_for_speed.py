"""
Ultra-fast inference optimization guide and script.

This script tests various optimization combinations to achieve <1 second prediction time.
"""

import sys
import os
import time
import numpy as np
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.realtime_inference import RealtimePredictor


def optimize_inference(
    checkpoint_path: str,
    target_time_ms: float = 1000.0,  # 1 second
    context_data: np.ndarray = None
):
    """
    Test various optimization strategies to achieve target inference time.
    
    Strategies tested:
    1. Reduce target length (2 steps instead of 24)
    2. Use parallel mode (faster than autoregressive)
    3. Model compilation (torch.compile)
    4. FP16 half precision (2x faster, less memory)
    5. Quantization (INT8 - 2-4x faster)
    6. ONNX Runtime (if available)
    7. GPU acceleration
    """
    print("=" * 80)
    print("ULTRA-FAST INFERENCE OPTIMIZATION")
    print("=" * 80)
    print(f"\nTarget: < {target_time_ms}ms per prediction")
    
    # Load model dimensions
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    context_points = checkpoint.get('context_points', 48)
    n_features = checkpoint.get('n_variables', 440)
    
    if context_data is None:
        context_data = np.random.randn(context_points, n_features).astype(np.float32)
    
    print(f"\nModel dimensions:")
    print(f"  Context: {context_points}")
    print(f"  Features: {n_features}")
    
    results = []
    
    # Strategy 1: Minimal target length (2 steps), parallel, compiled
    print(f"\n{'='*80}")
    print("Strategy 1: Minimal target length (2 steps) + Parallel + Compiled")
    print(f"{'='*80}")
    try:
        predictor = RealtimePredictor(
            checkpoint_path=checkpoint_path,
            use_parallel=True,
            compile_model=True,
            use_quantization=False
        )
        
        # Warmup
        for _ in range(5):
            _ = predictor.predict(context_data, target_length=2, return_scaled=True)
        
        # Benchmark
        start = time.time()
        for _ in range(50):
            predictions = predictor.predict(context_data, target_length=2, return_scaled=True)
        elapsed = (time.time() - start) / 50 * 1000
        
        results.append({
            'strategy': '2 steps + Parallel + Compiled',
            'time_ms': elapsed,
            'meets_target': elapsed < target_time_ms,
            'predictions_shape': predictions.shape
        })
        print(f"  ✅ Average: {elapsed:.2f}ms")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Strategy 2: Add FP16 (half precision)
    print(f"\n{'='*80}")
    print("Strategy 2: 2 steps + Parallel + Compiled + FP16")
    print(f"{'='*80}")
    try:
        predictor = RealtimePredictor(
            checkpoint_path=checkpoint_path,
            use_parallel=True,
            compile_model=True,
            use_quantization=False
        )
        
        # Convert model to FP16
        if hasattr(predictor.model, 'half'):
            predictor.model = predictor.model.half()
            print("  -> Model converted to FP16")
        
        # Convert input to FP16
        context_data_fp16 = context_data.astype(np.float16)
        
        # Warmup
        for _ in range(5):
            _ = predictor.predict(context_data_fp16, target_length=2, return_scaled=True)
        
        # Benchmark
        start = time.time()
        for _ in range(50):
            predictions = predictor.predict(context_data_fp16, target_length=2, return_scaled=True)
        elapsed = (time.time() - start) / 50 * 1000
        
        results.append({
            'strategy': '2 steps + Parallel + Compiled + FP16',
            'time_ms': elapsed,
            'meets_target': elapsed < target_time_ms,
            'predictions_shape': predictions.shape
        })
        print(f"  ✅ Average: {elapsed:.2f}ms")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Strategy 3: Add quantization
    print(f"\n{'='*80}")
    print("Strategy 3: 2 steps + Parallel + Compiled + Quantized (INT8)")
    print(f"{'='*80}")
    try:
        predictor = RealtimePredictor(
            checkpoint_path=checkpoint_path,
            use_parallel=True,
            compile_model=True,
            use_quantization=True
        )
        
        # Warmup
        for _ in range(5):
            _ = predictor.predict(context_data, target_length=2, return_scaled=True)
        
        # Benchmark
        start = time.time()
        for _ in range(50):
            predictions = predictor.predict(context_data, target_length=2, return_scaled=True)
        elapsed = (time.time() - start) / 50 * 1000
        
        results.append({
            'strategy': '2 steps + Parallel + Compiled + Quantized',
            'time_ms': elapsed,
            'meets_target': elapsed < target_time_ms,
            'predictions_shape': predictions.shape
        })
        print(f"  ✅ Average: {elapsed:.2f}ms")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Strategy 4: Single step prediction
    print(f"\n{'='*80}")
    print("Strategy 4: 1 step + Parallel + Compiled")
    print(f"{'='*80}")
    try:
        predictor = RealtimePredictor(
            checkpoint_path=checkpoint_path,
            use_parallel=True,
            compile_model=True,
            use_quantization=False
        )
        
        # Warmup
        for _ in range(5):
            _ = predictor.predict(context_data, target_length=1, return_scaled=True)
        
        # Benchmark
        start = time.time()
        for _ in range(50):
            predictions = predictor.predict(context_data, target_length=1, return_scaled=True)
        elapsed = (time.time() - start) / 50 * 1000
        
        results.append({
            'strategy': '1 step + Parallel + Compiled',
            'time_ms': elapsed,
            'meets_target': elapsed < target_time_ms,
            'predictions_shape': predictions.shape
        })
        print(f"  ✅ Average: {elapsed:.2f}ms")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Strategy 5: ONNX Runtime (if available)
    try:
        from spacetransformer_model.onnx_inference import ONNXPredictor
        
        onnx_path = checkpoint_path.replace('.pth', '.onnx')
        if os.path.exists(onnx_path):
            print(f"\n{'='*80}")
            print("Strategy 5: ONNX Runtime + 2 steps")
            print(f"{'='*80}")
            try:
                predictor = ONNXPredictor(
                    onnx_model_path=onnx_path,
                    checkpoint_path=checkpoint_path,
                    use_parallel=True
                )
                
                # Warmup
                for _ in range(5):
                    _ = predictor.predict(context_data, return_scaled=True)
                
                # Benchmark (ONNX doesn't support target_length yet, so we'll slice output)
                start = time.time()
                for _ in range(50):
                    predictions_full = predictor.predict(context_data, return_scaled=True)
                    predictions = predictions_full[:2]  # Take first 2 steps
                elapsed = (time.time() - start) / 50 * 1000
                
                results.append({
                    'strategy': 'ONNX Runtime + 2 steps',
                    'time_ms': elapsed,
                    'meets_target': elapsed < target_time_ms,
                    'predictions_shape': predictions.shape
                })
                print(f"  ✅ Average: {elapsed:.2f}ms")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    except ImportError:
        pass
    
    # Print summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Strategy':<50} {'Time (ms)':<15} {'Status':<15}")
    print("-" * 80)
    
    for r in results:
        status = "✅ MEETS TARGET" if r['meets_target'] else "⚠️  ABOVE TARGET"
        print(f"{r['strategy']:<50} {r['time_ms']:>10.2f}ms   {status}")
    
    # Find fastest
    if results:
        fastest = min(results, key=lambda x: x['time_ms'])
        print(f"\n🏆 Fastest: {fastest['strategy']}")
        print(f"   Time: {fastest['time_ms']:.2f}ms")
        
        if fastest['meets_target']:
            print(f"   ✅ Meets target of < {target_time_ms}ms!")
        else:
            print(f"   ⚠️  Still above target. Additional optimizations needed.")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR <1 SECOND PREDICTION")
    print(f"{'='*80}")
    
    if results and any(r['meets_target'] for r in results):
        print("\n✅ You can achieve <1 second with:")
        for r in results:
            if r['meets_target']:
                print(f"   - {r['strategy']} ({r['time_ms']:.2f}ms)")
    else:
        print("\n💡 To achieve <1 second prediction:")
        print("   1. Use target_length=1 or 2 (fastest)")
        print("   2. Enable parallel mode (use_parallel=True)")
        print("   3. Compile model (compile_model=True)")
        print("   4. Use GPU if available (much faster than CPU)")
        print("   5. Consider quantization (use_quantization=True)")
        print("   6. Use ONNX Runtime (1.5-3x faster)")
        print("   7. Use FP16 half precision (2x faster on GPU)")
        print("   8. Consider model distillation (smaller model)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Optimize inference for <1 second prediction')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint'
    )
    parser.add_argument(
        '--target-time',
        type=float,
        default=1000.0,
        help='Target time in milliseconds (default: 1000ms = 1 second)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    optimize_inference(
        checkpoint_path=args.checkpoint,
        target_time_ms=args.target_time
    )


if __name__ == '__main__':
    main()

