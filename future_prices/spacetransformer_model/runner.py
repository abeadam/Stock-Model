"""
Runner script for SpaceTimeFormer model training

This script provides a simple way to train the SpaceTimeFormer model.
Optimized for P100 GPU (16GB VRAM) with aggressive memory usage for maximum speed.
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

def find_optimal_batch_size(start_batch=32, max_batch=128, context_length=96, 
                           d_model=128, n_heads=8, d_ff=512):
    """
    Find the maximum batch size that fits in GPU memory.
    Tries progressively larger batches until OOM, then uses the last working size.
    """
    if not torch.cuda.is_available():
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
    print("Finding Optimal Batch Size for P100 GPU")
    print("=" * 80)
    
    optimal_batch = start_batch
    for batch_size in batch_sizes:
        print(f"\nTrying batch_size={batch_size}...")
        try:
            # Clear cache before test
            torch.cuda.empty_cache()
            
            # Quick test with minimal epochs
            model, history, scaler, feature_names = main(
                data_path=data_path,
                context_length=context_length,
                target_length=24,
                batch_size=batch_size,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                enc_layers=3,
                dec_layers=3,
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
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ OOM with batch_size={batch_size}")
                torch.cuda.empty_cache()
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
        # P100-optimized settings: Use more GPU memory for faster training
        # With FP16, we can use 2-4x larger batches than FP32
        
        if USE_OPTIMAL_BATCH:
            optimal_batch = find_optimal_batch_size(
                start_batch=16,
                max_batch=128,
                context_length=96,
                d_model=128,
                n_heads=8,
                d_ff=512
            )
            batch_size = optimal_batch
        else:
            # Aggressive settings for P100 with FP16
            # These should work well for 16GB VRAM
            batch_size = 128  # Can go up to 64-128 with FP16
        
        # Optimized model settings
        context_length = 96  # Full context length for better model capacity
        d_model = 128        # Medium model size - good balance
        d_ff = 512           # Standard feed-forward dimension
        n_heads = 8          # Standard number of heads
        
        # GPU optimizations enabled
        use_amp = True
        gradient_accumulation_steps = 1
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

