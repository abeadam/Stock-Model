"""
Runner script for SpaceTimeFormer model training

This script provides a simple way to train the SpaceTimeFormer model.
Optimized for high-end hardware (e.g., Apple M-series Max/Ultra or P100/A100 GPUs).
"""

import sys
import os
import torch

# Set PyTorch CUDA memory allocation to use expandable segments
# This helps reduce memory fragmentation, especially when there's
# a lot of reserved but unallocated memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable memory-efficient attention and other optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.spacetransformer import main
from spacetransformer_model.utils import get_device, empty_cache

def find_optimal_batch_size(start_batch=32, max_batch=128, context_length=96, 
                           d_model=128, n_heads=8, d_ff=512, enc_layers=3, dec_layers=3):
    """
    Find the maximum batch size that fits in GPU memory.
    Tries progressively larger batches until OOM, then uses the last working size.
    """
    device = get_device()
    if device.type == 'cpu':
        return start_batch
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    future_prices_dir = os.path.dirname(script_dir)
    data_path = os.path.join(future_prices_dir, 'es_with_indicators.csv')
    
    batch_sizes = []
    current = start_batch
    while current <= max_batch:
        batch_sizes.append(current)
        current *= 2
    
    print("=" * 80)
    print(f"Finding Optimal Batch Size for {device.type.upper()}")
    print("=" * 80)
    
    optimal_batch = start_batch
    for batch_size in batch_sizes:
        print(f"\nTrying batch_size={batch_size}...")
        try:
            # Clear cache before test
            empty_cache(device)
            
            # Quick test with minimal epochs
            model, history, scaler, feature_names = main(
                data_path=data_path,
                context_length=context_length,
                target_length=24,
                batch_size=batch_size,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                enc_layers=enc_layers,
                dec_layers=dec_layers,
                n_epochs=1,  # Just test memory, not full training
                use_amp=True,  # Use FP16 to save memory
                gradient_accumulation_steps=1,
                compile_model=False,  # Skip compilation for speed test
                golden_test=True  # Use minimal data for testing
            )
            
            optimal_batch = batch_size
            print(f"✓ Success with batch_size={batch_size}")
            
            # Clean up
            del model, history, scaler
            empty_cache(device)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                print(f"✗ OOM with batch_size={batch_size}")
                empty_cache(device)
                break
            else:
                raise
    
    print(f"\n{'='*80}")
    print(f"Optimal batch size: {optimal_batch}")
    print(f"{'='*80}\n")
    return optimal_batch

if __name__ == '__main__':
    # ============================================================================
    # CONFIGURATION FLAGS
    # ============================================================================
    # Set this to True to enable all GPU optimizations (larger batches, more workers, etc.)
    USE_GPU_OPTIMIZATIONS = True
    
    # Set to True to auto-find optimal batch size (only used if USE_GPU_OPTIMIZATIONS=True)
    USE_OPTIMAL_BATCH = False
    
    # Set to True for quick testing with minimal data
    GOLDEN_TEST = False
    
    # ============================================================================
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the future_prices directory (parent of script_dir)
    future_prices_dir = os.path.dirname(script_dir)
    # Construct path to data file
    data_path = os.path.join(future_prices_dir, 'es_with_indicators.csv')
    
    # ============================================================================
    # Configure settings based on flags
    # ============================================================================
    if USE_GPU_OPTIMIZATIONS:
        # Configuration for High-End Mac (40-core GPU / 52GB Memory)
        # Strategy: Since batch size is capped at 128 due to OOM, we maximize
        # computation per sample by:
        # 1. Increasing model dimensions (d_model, d_ff, layers)
        # 2. Using gradient accumulation to simulate larger effective batch size
        # 3. Deeper networks = more parallel computation per sample
        
        if USE_OPTIMAL_BATCH:
            # Use the same model dimensions as the optimized config
            optimal_batch = find_optimal_batch_size(
                start_batch=64,      # Start lower to find safe batch size
                max_batch=256,       # Test up to 256, but expect ~128 to be max
                context_length=96,
                d_model=512,         # Match optimized config (memory-efficient)
                n_heads=16,
                d_ff=2048,
                enc_layers=4,
                dec_layers=4
            )
            batch_size = optimal_batch
        else:
            # Batch size is capped at 128 due to memory, so we maximize compute per sample
            batch_size = 128  # Maximum before OOM
        
        # Balanced model capacity to fit in ~80GB memory while maximizing GPU utilization
        # Strategy: Moderate model size + gradient accumulation + deeper networks
        context_length = 96  # Full context length
        d_model = 512        # Large but memory-efficient model size
        d_ff = 2048          # 4x d_model for feed-forward (standard ratio)
        n_heads = 16         # d_model must be divisible by n_heads (512/16=32)
        enc_layers = 4       # Deeper encoder (increased from 3, but not too deep)
        dec_layers = 4       # Deeper decoder (increased from 3, but not too deep)
        
        # GPU optimizations enabled
        use_amp = True
        # Gradient accumulation simulates larger batch size without using more memory
        # Effective batch size = batch_size * gradient_accumulation_steps
        # This helps with training stability and better gradient estimates
        gradient_accumulation_steps = 4  # Effective batch = 128 * 4 = 512
        compile_model = True
    else:
        # Conservative default settings (original behavior)
        batch_size = 2
        context_length = 48  # Reduced to save memory
        d_model = 64         # Smaller model
        d_ff = 256          # Smaller feed-forward
        n_heads = 4         # Fewer heads
        
        # GPU optimizations disabled
        use_amp = False
        gradient_accumulation_steps = 1
        compile_model = False
    
    # Common settings
    target_length = 24
    # enc_layers and dec_layers are set above in USE_GPU_OPTIMIZATIONS block
    if not USE_GPU_OPTIMIZATIONS:
        enc_layers = 3
        dec_layers = 3
    n_epochs = 100
    learning_rate = 1e-3
    
    # ============================================================================
    # Resume Logic
    # ============================================================================
    # Path to the best model saved from previous runs
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    resume_path = os.path.join(checkpoint_dir, 'spacetimeformer_best.pth')
    
    # Only resume if the file exists
    if not os.path.exists(resume_path):
        resume_path = None
    else:
        print(f"Found existing checkpoint at {resume_path}")

    # ============================================================================
    # Run training
    # ============================================================================
    model, history, scaler, feature_names = main(
        data_path=data_path,
        context_length=context_length,
        target_length=target_length,
        batch_size=batch_size,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        # GPU optimizations (controlled by flag)
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        compile_model=compile_model,
        use_gpu_optimizations=USE_GPU_OPTIMIZATIONS,
        golden_test=GOLDEN_TEST,
        resume_path=resume_path
    )

