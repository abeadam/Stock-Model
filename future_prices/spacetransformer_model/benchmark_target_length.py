"""
Benchmark inference speed for different target lengths.

Shows how prediction speed scales with number of future steps predicted.
Useful for determining optimal target length for real-time trading.

Usage:
    python benchmark_target_length.py --checkpoint checkpoints/spacetimeformer_best.pth
"""

import sys
import os
import time
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.realtime_inference import RealtimePredictor


def benchmark_target_lengths(
    checkpoint_path: str,
    target_lengths: list[int] = [1, 2, 4, 8, 12, 24],
    n_runs: int = 50,
    warmup_runs: int = 5,
    use_parallel: bool = True,
    compile_model: bool = True
):
    """
    Benchmark inference speed for different target lengths.
    
    Args:
        checkpoint_path: Path to model checkpoint
        target_lengths: List of target lengths to test
        n_runs: Number of prediction runs per target length
        warmup_runs: Number of warmup runs
        use_parallel: Use parallel prediction mode
        compile_model: Compile model for faster inference
    """
    print("=" * 80)
    print("Target Length vs Inference Speed Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Mode: {'Parallel' if use_parallel else 'Autoregressive'}")
    print(f"  Compiled: {compile_model}")
    print(f"  Test lengths: {target_lengths}")
    print(f"  Runs per length: {n_runs}")
    
    # Load predictor once
    print(f"\nLoading model...")
    predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=use_parallel,
        compile_model=compile_model,
        use_quantization=False
    )
    
    # Get model dimensions
    context_points = predictor.context_points
    n_features = predictor.n_variables
    
    # Create dummy context data
    context_data = np.random.randn(context_points, n_features).astype(np.float32)
    
    print(f"  Context points: {context_points}")
    print(f"  Features: {n_features}")
    
    results = []
    
    for target_len in target_lengths:
        print(f"\n{'='*80}")
        print(f"Testing target length: {target_len}")
        print(f"{'='*80}")
        
        # Warmup
        if warmup_runs > 0:
            print(f"Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                _ = predictor.predict(
                    context_data,
                    target_length=target_len,
                    return_scaled=True
                )
        
        # Benchmark
        print(f"Running {n_runs} predictions...")
        start = time.time()
        for _ in range(n_runs):
            predictions = predictor.predict(
                context_data,
                target_length=target_len,
                return_scaled=True
            )
        total_time = time.time() - start
        
        avg_time = total_time / n_runs
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        results.append({
            'target_length': target_len,
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': throughput,
            'predictions_shape': predictions.shape
        })
        
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Average per prediction: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {throughput:.2f} predictions/second")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Use first result as baseline
    baseline = results[0]
    
    print(f"\n{'Target Length':<15} {'Time (ms)':<15} {'Speedup':<15} {'Throughput':<15} {'Time per Step':<15}")
    print("-" * 75)
    
    for r in results:
        target_len = r['target_length']
        time_ms = r['avg_time'] * 1000
        speedup = baseline['avg_time'] / r['avg_time'] if r['avg_time'] > 0 else 0
        throughput = r['throughput']
        time_per_step = time_ms / target_len if target_len > 0 else 0
        
        print(f"{target_len:<15} {time_ms:>10.2f}ms   {speedup:>10.2f}x   {throughput:>10.2f} pred/s   {time_per_step:>10.2f}ms/step")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    # Find fastest
    fastest = min(results, key=lambda x: x['avg_time'])
    print(f"\n🏆 Fastest: {fastest['target_length']} steps ({fastest['avg_time']*1000:.2f}ms)")
    
    # Calculate speedup from 24 to 2
    result_24 = next((r for r in results if r['target_length'] == 24), None)
    result_2 = next((r for r in results if r['target_length'] == 2), None)
    
    if result_24 and result_2:
        speedup_24_to_2 = result_24['avg_time'] / result_2['avg_time']
        print(f"\n📊 Speedup (24 → 2 steps): {speedup_24_to_2:.2f}x faster")
        print(f"   Time saved: {(result_24['avg_time'] - result_2['avg_time'])*1000:.2f}ms per prediction")
        
        # Calculate time per step
        time_per_step_24 = result_24['avg_time'] / 24
        time_per_step_2 = result_2['avg_time'] / 2
        print(f"\n⏱️  Time per step:")
        print(f"   24 steps: {time_per_step_24*1000:.2f}ms/step")
        print(f"   2 steps:  {time_per_step_2*1000:.2f}ms/step")
        
        if use_parallel:
            print(f"\n💡 Parallel mode: Decoder processes all steps at once.")
            print(f"   Speedup is less dramatic (~{speedup_24_to_2:.1f}x) because attention")
            print(f"   complexity scales with sequence length.")
        else:
            print(f"\n💡 Autoregressive mode: Each step requires a separate forward pass.")
            print(f"   Speedup is more dramatic (~{speedup_24_to_2:.1f}x) because you do")
            print(f"   {result_2['target_length']} passes instead of {result_24['target_length']}.")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if use_parallel:
        print(f"\n✅ For real-time trading with parallel mode:")
        print(f"   - Predicting 2 steps: ~{result_2['avg_time']*1000:.1f}ms (if result_2 else 'N/A')")
        print(f"   - Predicting 24 steps: ~{result_24['avg_time']*1000:.1f}ms (if result_24 else 'N/A')")
        print(f"   - Use 2 steps if you only need immediate next values")
        print(f"   - Use 24 steps if you need longer horizon predictions")
    else:
        print(f"\n✅ For real-time trading with autoregressive mode:")
        print(f"   - Predicting 2 steps is {speedup_24_to_2:.1f}x faster than 24 steps")
        print(f"   - Use 2 steps for ultra-low latency (<{result_2['avg_time']*1000:.1f}ms)")
        print(f"   - Use 24 steps only if you need longer predictions")


def main():
    parser = argparse.ArgumentParser(description='Benchmark target length vs inference speed')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint file'
    )
    parser.add_argument(
        '--target-lengths',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 12, 24],
        help='Target lengths to test (default: 1 2 4 8 12 24)'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=50,
        help='Number of runs per target length (default: 50)'
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=5,
        help='Number of warmup runs (default: 5)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Use parallel prediction mode (default: True)'
    )
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='Use autoregressive prediction mode'
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        default=True,
        help='Compile model (default: True)'
    )
    parser.add_argument(
        '--no-compile',
        dest='compile',
        action='store_false',
        help='Do not compile model'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    benchmark_target_lengths(
        checkpoint_path=args.checkpoint,
        target_lengths=args.target_lengths,
        n_runs=args.n_runs,
        warmup_runs=args.warmup_runs,
        use_parallel=args.parallel,
        compile_model=args.compile
    )


if __name__ == '__main__':
    main()

