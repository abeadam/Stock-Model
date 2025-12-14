"""
Example script showing how to use RealtimePredictor for fast inference.

This demonstrates the usage pattern for real-time trading systems.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.realtime_inference import RealtimePredictor


def get_checkpoint_path():
    """Get checkpoint path relative to this script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'checkpoints', 'spacetimeformer_best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}\n"
            "Please ensure the checkpoint file exists or update the path."
        )
    return checkpoint_path


def example_single_prediction():
    """Example: Single prediction for one context sequence."""
    print("=" * 80)
    print("Example 1: Single Prediction")
    print("=" * 80)
    
    # Initialize predictor (fast parallel mode for real-time trading)
    import time
    
    checkpoint_path = get_checkpoint_path()
    print("\nInitializing predictor...")
    init_start = time.time()
    predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,  # Fast mode: all 24 steps predicted in one pass
        compile_model=False
    )

    
    # Simulate context data: (context_points, n_features)
    # In real usage, this would be your actual historical data
    context_points = predictor.context_points
    n_features = predictor.n_variables
    
    print(f"\nContext shape: ({context_points}, {n_features})")
    
    # Create dummy context data (replace with real data)
    context_data = np.random.randn(context_points, n_features)
    
    # Predict next 24 steps for High, Low, Close
    # Note: You need to know the actual indices of High, Low, Close in your feature set
    # For this example, we'll assume they're at indices 0, 1, 2
    target_indices = [0, 2, 3]  # Adjust based on your actual feature order
    
    print(f"Predicting next {predictor.target_points} steps...")
    predictions = predictor.predict(
        context_data=context_data,
        target_indices=target_indices,
        return_scaled=False  # Return actual prices (inverse transformed)
    )
    
    # predictions shape: (24, 3) - 24 time steps, 3 targets (High, Low, Close)
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 steps of predictions:\n{predictions[:5]}")
    init_time = time.time() - init_start
    print(f"  -> Predictor took {init_time:.4f} seconds")
    
    return predictions


def example_batch_prediction():
    """Example: Batch prediction for multiple context sequences."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Prediction")
    print("=" * 80)
    
    checkpoint_path = get_checkpoint_path()
    predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True
    )
    
    # Batch of context sequences: (batch_size, context_points, n_features)
    batch_size = 10
    context_points = predictor.context_points
    n_features = predictor.n_variables
    
    print(f"\nBatch context shape: ({batch_size}, {context_points}, {n_features})")
    
    # Create dummy batch data
    batch_context = np.random.randn(batch_size, context_points, n_features)
    
    # Predict for entire batch
    target_indices = [0, 1, 2]  # High, Low, Close
    
    print(f"Predicting next {predictor.target_points} steps for {batch_size} samples...")
    batch_predictions = predictor.predict_batch(
        context_batch=batch_context,
        target_indices=target_indices,
        return_scaled=False
    )
    
    # batch_predictions shape: (batch_size, 24, 3)
    print(f"Batch predictions shape: {batch_predictions.shape}")
    print(f"First sample, first 5 steps:\n{batch_predictions[0, :5, :]}")
    
    return batch_predictions


def example_with_real_data():
    """Example: Using real data from CSV file."""
    print("\n" + "=" * 80)
    print("Example 3: Using Real Data")
    print("=" * 80)
    
    try:
        from spacetransformer_model.data_loader import load_data
        from spacetransformer_model.utils import extract_cyclical_time_features
        import pandas as pd
        
        checkpoint_path = get_checkpoint_path()
        # Resolve data path relative to project root
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(script_dir, 'es_with_indicators.csv')
        
        # Initialize predictor
        predictor = RealtimePredictor(
            checkpoint_path=checkpoint_path,
            use_parallel=True
        )
        
        # Load and prepare data
        print(f"\nLoading data from {data_path}...")
        X_train, X_test, scaler, feature_names = load_data(
            data_path=data_path,
            skip_split=True,
            skip_scaling=True,  # We'll use the scaler from checkpoint
            max_rows=1000
        )
        
        # Get the last context_points rows as context
        context_points = predictor.context_points
        if len(X_train) < context_points:
            print(f"Warning: Not enough data. Need {context_points}, got {len(X_train)}")
            return None
        
        context_data = X_train[-context_points:]  # Last context_points rows
        
        # Find target indices (High, Low, Close)
        target_columns = ['High', 'Low', 'Close']
        target_indices = [feature_names.index(col) for col in target_columns if col in feature_names]
        
        if len(target_indices) != 3:
            print(f"Warning: Could not find all target columns. Found: {target_indices}")
            # Use first 3 features as fallback
            target_indices = [0, 1, 2]
        
        print(f"Target indices: {target_indices}")
        print(f"Context data shape: {context_data.shape}")
        
        # Make prediction
        print(f"\nPredicting next {predictor.target_points} steps...")
        predictions = predictor.predict(
            context_data=context_data,
            target_indices=target_indices,
            return_scaled=False
        )
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions (first 5 steps):\n{predictions[:5]}")
        
        return predictions
        
    except Exception as e:
        print(f"Error in real data example: {e}")
        print("This example requires the data file and may need path adjustments.")
        return None


def example_comparison_modes():
    """Example: Compare compiled vs uncompiled, and parallel vs autoregressive."""
    print("\n" + "=" * 80)
    print("Example 4: Speed Comparison (Compiled vs Uncompiled)")
    print("=" * 80)
    
    import time
    
    checkpoint_path = get_checkpoint_path()
    
    # Use actual model dimensions
    # Load a temporary predictor just to get dimensions
    temp_predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,
        compile_model=False  # Quick load just for dimensions
    )
    context_points = temp_predictor.context_points
    n_features = temp_predictor.n_variables
    context_data = np.random.randn(context_points, n_features)
    del temp_predictor  # Free memory
    
    print(f"\nUsing model dimensions: context_points={context_points}, n_features={n_features}")
    print(f"Testing with {len(context_data)} context points and {n_features} features")
    
    # Test 1: Uncompiled model
    print("\n1. Uncompiled model (parallel mode):")
    predictor_uncompiled = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,
        compile_model=False
    )
    
    # Warmup run
    _ = predictor_uncompiled.predict(context_data, return_scaled=True)
    
    # Time multiple runs
    n_runs = 50
    start = time.time()
    for _ in range(n_runs):
        _ = predictor_uncompiled.predict(context_data, return_scaled=True)
    uncompiled_time = time.time() - start
    print(f"   Time for {n_runs} predictions: {uncompiled_time:.4f}s")
    print(f"   Average per prediction: {uncompiled_time/n_runs:.4f}s")
    
    # Test 2: Compiled model
    print("\n2. Compiled model (parallel mode):")
    predictor_compiled = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,
        compile_model=True
    )
    
    # Warmup runs (compilation happens here)
    print("   Warming up compiled model (first few runs are slower)...")
    warmup_start = time.time()
    for _ in range(5):
        _ = predictor_compiled.predict(context_data, return_scaled=True)
    warmup_time = time.time() - warmup_start
    print(f"   Warmup time (5 runs): {warmup_time:.4f}s")
    
    # Time multiple runs after warmup
    start = time.time()
    for _ in range(n_runs):
        _ = predictor_compiled.predict(context_data, return_scaled=True)
    compiled_time = time.time() - start
    print(f"   Time for {n_runs} predictions (after warmup): {compiled_time:.4f}s")
    print(f"   Average per prediction: {compiled_time/n_runs:.4f}s")
    
    # Compare
    if compiled_time < uncompiled_time:
        speedup = uncompiled_time / compiled_time
        print(f"\n   ✅ Compiled is {speedup:.2f}x faster (after warmup)")
    else:
        slowdown = compiled_time / uncompiled_time
        print(f"\n   ⚠️  Compiled is {slowdown:.2f}x slower")
        print("   This can happen on CPU/MPS or with many graph breaks.")
        print("   For real-time trading, uncompiled may be better.")
    
    # Test 3: Autoregressive mode comparison
    print("\n3. Autoregressive mode (uncompiled):")
    predictor_ar = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=False,
        compile_model=False
    )
    
    # Warmup
    _ = predictor_ar.predict(context_data, return_scaled=True)
    
    start = time.time()
    for _ in range(10):  # Fewer runs since autoregressive is slower
        _ = predictor_ar.predict(context_data, return_scaled=True)
    ar_time = time.time() - start
    print(f"   Time for 10 predictions: {ar_time:.4f}s")
    print(f"   Average per prediction: {ar_time/10:.4f}s")
    
    parallel_speedup = ar_time / (uncompiled_time/n_runs * 10)
    print(f"\n   Parallel mode is {parallel_speedup:.2f}x faster than autoregressive")


if __name__ == "__main__":
    print("SpaceTimeFormer Real-Time Inference Examples")
    print("=" * 80)
    
    # Run examples
    try:
        # example_single_prediction()
        # example_batch_prediction()
        # example_with_real_data()
        example_comparison_modes()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the checkpoint file exists and paths are correct.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

