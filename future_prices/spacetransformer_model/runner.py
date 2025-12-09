"""
Runner script for SpaceTimeFormer model training

This script provides a simple way to train the SpaceTimeFormer model.
"""

import sys
import os

# Set PyTorch CUDA memory allocation to use expandable segments
# This helps reduce memory fragmentation, especially when there's
# a lot of reserved but unallocated memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.spacetransformer import main

if __name__ == '__main__':
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the future_prices directory (parent of script_dir)
    future_prices_dir = os.path.dirname(script_dir)
    # Construct path to data file
    data_path = os.path.join(future_prices_dir, 'es_with_indicators.csv')
    
    model, history, scaler, feature_names = main(
        data_path=data_path,
        context_length=48,  # Reduced from 96 to save memory (sequence length = 48 * 434 = 20,832)
        target_length=24,
        batch_size=2,  # Further reduced batch size to avoid OOM
        d_model=64,    # Reduce model dimension to save memory
        d_ff=256,      # Reduce feed-forward dimension
        n_heads=4      # Reduce number of heads to save memory
    )

