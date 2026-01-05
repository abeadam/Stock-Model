"""
Generate plots comparing real vs predicted values from trained SpaceTimeFormer model.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.data_loader import load_data
from spacetransformer_model.model import SpaceTimeFormer
from spacetransformer_model.train import TimeSeriesDataset


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_info)
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Validate checkpoint structure
    if 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    state_dict = checkpoint['model_state_dict']
    if state_dict is None:
        raise ValueError("Checkpoint contains 'model_state_dict' but it is None")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint 'model_state_dict' is not a dict, got {type(state_dict)}"
        )
    
    print(f"  -> State dict contains {len(state_dict)} parameters")
    
    # Required architecture parameters
    required_params = [
        'n_variables', 'd_model', 'n_heads', 'enc_layers', 'dec_layers',
        'd_ff', 'context_points', 'target_points'
    ]
    
    missing_params = [p for p in required_params if p not in checkpoint]
    if missing_params:
        raise ValueError(
            f"Checkpoint is missing required architecture parameters: {missing_params}. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    
    # Extract architecture parameters
    n_variables = checkpoint['n_variables']
    d_model = checkpoint['d_model']
    n_heads = checkpoint['n_heads']
    enc_layers = checkpoint['enc_layers']
    dec_layers = checkpoint['dec_layers']
    d_ff = checkpoint['d_ff']
    dropout = checkpoint.get('dropout', 0.1)
    max_seq_length = checkpoint.get('max_seq_length', 1000)
    context_points = checkpoint['context_points']
    target_points = checkpoint['target_points']
    use_windowed_attn = checkpoint.get('use_windowed_attn', True)
    window_size = checkpoint.get('window_size', 200)
    
    print("  -> Loaded architecture parameters from checkpoint")
    
    # Recreate model
    model = SpaceTimeFormer(
        n_variables=n_variables,
        d_model=d_model,
        n_heads=n_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length,
        context_points=context_points,
        target_points=target_points,
        use_windowed_attn=use_windowed_attn,
        window_size=window_size
    )
    
    # Load state dict
    result = model.load_state_dict(state_dict, strict=False)
    # Handle both tuple and _IncompatibleKeys object
    if isinstance(result, tuple):
        missing_keys, unexpected_keys = result
    else:
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
    
    if missing_keys:
        print(f"  -> WARNING: Missing keys in state_dict ({len(missing_keys)} total): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  -> WARNING: Unexpected keys in state_dict ({len(unexpected_keys)} total): {unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    print(f"  -> Model loaded successfully")
    print(f"  -> n_variables: {n_variables}")
    print(f"  -> d_model: {d_model}")
    print(f"  -> Context points: {context_points}, Target points: {target_points}")
    
    return model, checkpoint


def generate_predictions(
    model: SpaceTimeFormer,
    test_loader: DataLoader,
    device: torch.device,
    target_indices: list[int],
    scaler: Optional[Any] = None,
    n_samples: int = 10,
    max_prediction_points: int = 200
) -> tuple:
    """
    Generate predictions on test data.
    
    Args:
        model: Trained SpaceTimeFormer model
        test_loader: DataLoader for test data
        device: Device to run inference on
        target_indices: Indices of target variables (High, Low, Close)
        scaler: Scaler for inverse transforming predictions to actual prices
        n_samples: Number of samples to plot
        max_prediction_points: Maximum number of prediction points to generate (for speed)
    
    Returns:
        Tuple of (all_predictions, all_targets, sample_indices) - values are in original (unscaled) space
    """
    print(f"\nGenerating predictions (limited to {max_prediction_points} points for speed)...")
    
    all_predictions = []
    all_targets = []
    sample_indices = []
    sample_count = 0
    total_prediction_points = 0
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_target_X) in enumerate(test_loader):
            # Stop if we've reached max prediction points or n_samples for plotting
            if sample_count >= n_samples or total_prediction_points >= max_prediction_points:
                break
                
            batch_X = batch_X.to(device)
            batch_target_X = batch_target_X.to(device)
            
            # Autoregressive generation (same as validation)
            batch_size = batch_X.shape[0]
            target_length = batch_target_X.shape[1]
            
            # Check if processing this batch would exceed max_prediction_points
            batch_points = batch_size * target_length
            if total_prediction_points + batch_points > max_prediction_points:
                # Process only enough samples to reach max_prediction_points
                remaining_points = max_prediction_points - total_prediction_points
                samples_to_process = max(1, remaining_points // target_length)
                if samples_to_process < batch_size:
                    # Only process first N samples in this batch
                    batch_X = batch_X[:samples_to_process]
                    batch_target_X = batch_target_X[:samples_to_process]
                    batch_size = samples_to_process
            
            # Encode context once
            encoder_output = model.encode(batch_X)
            
            # Initial input: Last context step, sliced to match model's expected features
            # The model expects model.n_variables features, but batch_X might have more
            current_input = batch_X[:, -1:, :model.n_variables]
            
            # Store predictions
            predictions_list = []
            
            for t in range(target_length):
                step_predictions = model.decode(current_input, encoder_output)
                next_pred = step_predictions[:, -1:, :]
                predictions_list.append(next_pred)
                # Concatenate for next iteration (both should have model.n_variables features)
                current_input = torch.cat([current_input, next_pred], dim=1)
            
            # Concatenate all predictions
            predictions = torch.cat(predictions_list, dim=1)
            
            # Extract target columns (still in scaled space)
            pred_targets_scaled = predictions[:, :, target_indices].cpu().numpy()
            true_targets_scaled = batch_target_X[:, :, target_indices].cpu().numpy()
            
            # Inverse transform to actual prices if scaler is provided
            if scaler is not None:
                # Reshape to (batch * time, n_features) for scaler
                # We need to reconstruct full feature vectors for inverse transform
                batch_size_actual, target_len, n_targets = pred_targets_scaled.shape
                
                # Create full feature arrays (zeros for non-target features)
                n_features = model.n_variables
                pred_full = np.zeros((batch_size_actual, target_len, n_features))
                true_full = np.zeros((batch_size_actual, target_len, n_features))
                
                # Place target values at correct indices
                for i, target_idx in enumerate(target_indices):
                    pred_full[:, :, target_idx] = pred_targets_scaled[:, :, i]
                    true_full[:, :, target_idx] = true_targets_scaled[:, :, i]
                
                # Reshape for scaler: (batch * time, features)
                pred_flat = pred_full.reshape(-1, n_features)
                true_flat = true_full.reshape(-1, n_features)
                
                # Inverse transform
                pred_flat = scaler.inverse_transform(pred_flat)
                true_flat = scaler.inverse_transform(true_flat)
                
                # Extract target columns from inverse-transformed data
                pred_targets = pred_flat[:, target_indices].reshape(batch_size_actual, target_len, n_targets)
                true_targets = true_flat[:, target_indices].reshape(batch_size_actual, target_len, n_targets)
            else:
                # No scaler, return scaled values
                pred_targets = pred_targets_scaled
                true_targets = true_targets_scaled
            
            all_predictions.append(pred_targets)
            all_targets.append(true_targets)
            
            # Track which samples we're using and update counters
            for i in range(batch_size):
                if sample_count < n_samples and total_prediction_points < max_prediction_points:
                    sample_indices.append((batch_idx, i))
                    sample_count += 1
                    total_prediction_points += target_length
                else:
                    break
            
            # Stop if we've reached the limit
            if total_prediction_points >= max_prediction_points:
                break
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    actual_points = len(all_predictions) * all_predictions.shape[1]
    max_iterations = all_predictions.shape[1]  # Number of future time steps predicted
    print(f"  -> Generated predictions for {len(all_predictions)} samples")
    print(f"  -> Total prediction points: {actual_points}")
    print(f"  -> Max predicted iterations per sample: {max_iterations} future time steps")
    
    return all_predictions, all_targets, sample_indices


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
    sample_indices: list,
    save_dir: Optional[str] = None,
    scaler_used: Optional[Any] = None
):
    """
    Plot real vs predicted values.
    
    Args:
        predictions: Predicted values (n_samples, target_length, n_targets)
        targets: True values (n_samples, target_length, n_targets)
        target_names: Names of target variables
        sample_indices: List of (batch_idx, sample_idx) tuples
        save_dir: Directory to save plots (optional)
    """
    n_samples = min(len(predictions), 10)  # Plot up to 10 samples
    n_targets = len(target_names)
    target_length = predictions.shape[1]
    time_steps = np.arange(1, target_length + 1)  # Start from 1 to show these are future steps
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_targets, n_samples, figsize=(4*n_samples, 3*n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Real vs Predicted Values (Next {target_length} Time Steps)', fontsize=16, fontweight='bold')
    
    for target_idx, target_name in enumerate(target_names):
        for sample_idx in range(n_samples):
            ax = axes[target_idx, sample_idx]
            
            # Get data for this sample and target
            pred = predictions[sample_idx, :, target_idx]
            true = targets[sample_idx, :, target_idx]
            
            # Plot
            ax.plot(time_steps, true, 'b-', label='Real', linewidth=2, alpha=0.7)
            ax.plot(time_steps, pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)
            
            # Formatting
            ax.set_title(f'{target_name} - Sample {sample_idx+1} (Max: {target_length} steps ahead)', fontsize=10)
            ax.set_xlabel(f'Future Time Step (1 to {target_length})')
            ax.set_ylabel('Price' if scaler_used else 'Scaled Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'predictions_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    # Also create a summary plot with all targets for first few samples
    n_summary_samples = min(n_samples, 5)
    fig2, axes2 = plt.subplots(n_summary_samples, 1, figsize=(12, 3*n_summary_samples))
    if n_summary_samples == 1:
        axes2 = [axes2]
    
    fig2.suptitle(f'All Targets - Real vs Predicted (First 5 Samples, Next {target_length} Steps)', fontsize=14, fontweight='bold')
    
    # Define colors and styles for each target
    colors = {'High': 'red', 'Low': 'blue', 'Close': 'green'}
    linestyles_real = {'High': '-', 'Low': '-', 'Close': '-'}
    linestyles_pred = {'High': '--', 'Low': '--', 'Close': '--'}
    
    for sample_idx in range(n_summary_samples):
        ax = axes2[sample_idx]
        
        for target_idx, target_name in enumerate(target_names):
            pred = predictions[sample_idx, :, target_idx]
            true = targets[sample_idx, :, target_idx]
            
            color = colors.get(target_name, 'black')
            ax.plot(time_steps, true, linestyle=linestyles_real[target_name], 
                   color=color, label=f'{target_name} (Real)', linewidth=1.5, alpha=0.7)
            ax.plot(time_steps, pred, linestyle=linestyles_pred[target_name], 
                   color=color, label=f'{target_name} (Pred)', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'Sample {sample_idx+1} - Predicting Next {target_length} Iterations', fontsize=10)
        ax.set_xlabel(f'Future Time Step (1 to {target_length})')
        ax.set_ylabel('Price' if scaler_used else 'Scaled Value')
        # Increase legend columns and adjust position to show all labels
        ax.legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path2 = os.path.join(save_dir, 'predictions_summary.png')
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path2}")
    else:
        plt.show()


def main():
    """Main function to generate plots."""
    # Configuration
    checkpoint_path = '/home/wiseguy/dev/future_prices/spacetransformer_model/checkpoints/spacetimeformer_best.pth'
    data_path = '/home/wiseguy/dev/future_prices/es_with_indicators.csv'
    
    # Model hyperparameters (should match training)
    context_length = 96
    target_length = 24
    batch_size = 128
    test_size = 0.3
    random_state = 42
    
    # Target columns
    target_columns = ['High', 'Low', 'Close']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model first (architecture parameters are in checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Try to load scaler from checkpoint
    scaler = None
    if 'scaler' in checkpoint:
        import pickle
        import io
        print("\nLoading scaler from checkpoint...")
        scaler_bytes = io.BytesIO(checkpoint['scaler'])
        scaler = pickle.load(scaler_bytes)
        print("  -> Scaler loaded from checkpoint")
    else:
        print("\n  -> WARNING: No scaler found in checkpoint, will fit new scaler (may cause metric mismatch)")
    
    # Load data for inference (no split needed, limited to 1000 rows for faster processing)
    print("\nLoading data for inference (limited to 1000 rows)...")
    X_train, X_test, scaler_new, feature_names = load_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        golden_test=False,
        batch_size=batch_size,
        max_rows=1000,
        skip_split=True,  # No need to split for inference/plotting
        skip_scaling=(scaler is not None)  # Skip scaling if we have checkpoint scaler
    )
    
    # Use scaler from checkpoint if available, otherwise use newly fitted one
    if scaler is not None:
        print("  -> Using scaler from checkpoint to transform data")
        # Transform raw data with checkpoint scaler
        X_full = scaler.transform(X_test)
    else:
        print("  -> Using newly fitted scaler (WARNING: metrics may not match training)")
        X_full = X_test  # Already scaled by load_data
    
    print(f"  -> Number of features: {X_full.shape[1]}")
    
    # Find target indices
    target_indices = []
    for col in target_columns:
        if col in feature_names:
            target_indices.append(feature_names.index(col))
            print(f"  -> Found '{col}' at index {feature_names.index(col)}")
        else:
            print(f"  -> WARNING: '{col}' not found in features")
    
    if not target_indices:
        print("  -> ERROR: No target columns found!")
        return
    
    # Create dataset from full data
    test_dataset = TimeSeriesDataset(
        X_full,
        context_length=context_length,
        target_length=target_length,
        stride=1
    )
    
    # Randomly sample 200 sequences from test dataset
    n_sequences = len(test_dataset)
    n_samples_to_use = min(200, n_sequences)
    print(f"  -> Total test sequences: {n_sequences}")
    
    if n_sequences > n_samples_to_use:
        # Randomly select 200 sequences
        rng = np.random.RandomState(random_state)
        selected_indices = rng.choice(n_sequences, size=n_samples_to_use, replace=False)
        selected_indices = sorted(selected_indices)  # Keep sorted for reproducibility
        
        # Create a subset dataset with only selected indices
        # We'll modify the dataset's valid_indices
        original_indices = test_dataset.valid_indices.copy()
        test_dataset.valid_indices = [original_indices[i] for i in selected_indices]
        print(f"  -> Randomly selected {n_samples_to_use} sequences for testing")
    else:
        print(f"  -> Using all {n_sequences} available sequences")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"  -> Test sequences: {len(test_dataset)}")
    print(f"  -> Test batches: {len(test_loader)}")
    
    # Get scaler for inverse transform (use checkpoint scaler if available, otherwise None)
    scaler_for_inverse = None
    if 'scaler' in checkpoint:
        import pickle
        import io
        scaler_bytes = io.BytesIO(checkpoint['scaler'])
        scaler_for_inverse = pickle.load(scaler_bytes)
    
    # Generate predictions (limited to 200 points for speed)
    predictions, targets, sample_indices = generate_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        target_indices=target_indices,
        scaler=scaler_for_inverse,
        n_samples=10,
        max_prediction_points=200
    )
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Prediction Metrics")
    print("="*80)
    print("\nNOTE: Metrics below are calculated on UNSCALED (actual price) data.")
    print("      For model quality assessment, refer to training validation metrics")
    print("      which are calculated on scaled data (R² ≈ 0.85 indicates good performance).")
    print("="*80)
    
    for target_idx, target_name in enumerate(target_columns):
        if target_idx >= len(target_indices):
            continue
            
        pred_flat = predictions[:, :, target_idx].flatten()
        true_flat = targets[:, :, target_idx].flatten()
        
        mse = np.mean((pred_flat - true_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - true_flat))
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors (more interpretable than R² on unscaled data)
        mean_price = np.mean(np.abs(true_flat))
        mape = (mae / mean_price * 100) if mean_price > 0 else 0
        rmse_percent = (rmse / mean_price * 100) if mean_price > 0 else 0
        
        # R² score (may be misleading on unscaled data due to scale differences)
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\n{target_name}:")
        print(f"  MSE:      {mse:.6f}")
        print(f"  MAE:      {mae:.6f}  ({mape:.2f}% of mean price)")
        print(f"  RMSE:     {rmse:.6f}  ({rmse_percent:.2f}% of mean price)")
        print(f"  R²:       {r2:.6f}  (NOTE: R² on unscaled data can be misleading)")
        print(f"  Mean Price: {mean_price:.2f}")
        
        # Additional context
        if r2 < 0:
            print(f"  → Negative R² indicates prediction errors are larger than")
            print(f"    variance of target, but this can be misleading on unscaled data.")
            print(f"    Focus on RMSE/MAE percentages instead.")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    save_dir = os.path.join(os.path.dirname(checkpoint_path), 'plots')
    plot_predictions(
        predictions=predictions,
        targets=targets,
        target_names=target_columns,
        sample_indices=sample_indices,
        save_dir=save_dir,
        scaler_used=scaler_for_inverse
    )
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()

