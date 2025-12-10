"""
SpaceTimeFormer model for futures price prediction

This module provides a complete pipeline for training and using SpaceTimeFormer
to predict PctChange_ToMaxHigh_5 values from multivariate time series data.

SpaceTimeFormer learns spatiotemporal patterns by:
- Flattening multivariate sequences into tokens (one per variable per timestep)
- Using spatiotemporal attention to learn relationships across variables and time
- Predicting future values using an encoder-decoder architecture

Reference: "Long-Range Transformers for Dynamic Spatiotemporal Forecasting"
https://github.com/QData/spacetimeformer
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from typing import Optional

from .data_loader import load_data
from .model import SpaceTimeFormer
from .train import TimeSeriesDataset, train_model


def main(
    data_path: str = '../es_with_indicators.csv',
    context_length: int = 96,
    target_length: int = 24,
    d_model: int = 128,
    n_heads: int = 8,
    enc_layers: int = 3,
    dec_layers: int = 3,
    d_ff: int = 512,
    dropout: float = 0.1,
    batch_size: int = 32,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model_path: Optional[str] = None,
    golden_test: bool = False
):
    """
    Main function to train SpaceTimeFormer model.
    
    Args:
        data_path: Path to CSV file with features and targets
        context_length: Number of timesteps in context sequence
        target_length: Number of timesteps to predict
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        enc_layers: Number of encoder layers
        dec_layers: Number of decoder layers
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        test_size: Proportion of data for testing
        random_state: Random seed
        save_model_path: Path to save trained model (optional)
        golden_test: If True, run in golden test mode (minimal data)
    """
    print("=" * 80)
    print("SpaceTimeFormer Training Pipeline")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    loss_target_cols = ['Low', 'High', 'Close']
    X_train, X_test, scaler, feature_names = load_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        golden_test=golden_test,
        batch_size=batch_size
    )
    
    n_features = X_train.shape[1]
    print(f"  -> Training samples: {len(X_train)}")
    print(f"  -> Test samples: {len(X_test)}")
    print(f"  -> Number of features: {n_features}")

    # Determine target indices for loss (High, Low, Close)
    target_columns = ['Close', 'High', 'Low']
    target_indices = []
    for col in target_columns:
        if col in feature_names:
            target_indices.append(feature_names.index(col))
        else:
            print(f"  -> WARNING: Target column '{col}' not found in features; loss will skip it.")
    if not target_indices:
        print("  -> WARNING: No target columns found; defaulting target_indices to [0].")
        target_indices = [0]
    
    # Create datasets (sequences are created on-the-fly to avoid memory issues)
    print("\n[2/5] Creating datasets...")
    print(f"  -> Context length: {context_length}")
    print(f"  -> Target length: {target_length}")
    
    # Create full training dataset first to get sequence count
    full_train_dataset = TimeSeriesDataset(
        X_train,
        context_length=context_length,
        target_length=target_length,
        stride=1
    )
    
    # Split training data into train and validation using dataset indices
    n_sequences = len(full_train_dataset)
    val_size = int(n_sequences * 0.2)
    indices = np.random.RandomState(random_state).permutation(n_sequences)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train dataset with subset of indices
    train_dataset = TimeSeriesDataset(
        X_train,
        context_length=context_length,
        target_length=target_length,
        stride=1
    )
    train_dataset.valid_indices = [train_dataset.valid_indices[i] for i in train_indices]
    
    # Create validation dataset with subset of indices
    val_dataset = TimeSeriesDataset(
        X_train,
        context_length=context_length,
        target_length=target_length,
        stride=1
    )
    val_dataset.valid_indices = [val_dataset.valid_indices[i] for i in val_indices]
    
    # Create test dataset
    test_dataset = TimeSeriesDataset(
        X_test,
        context_length=context_length,
        target_length=target_length,
        stride=1
    )
    
    print(f"  -> Training sequences: {len(train_dataset)}")
    print(f"  -> Validation sequences: {len(val_dataset)}")
    print(f"  -> Test sequences: {len(test_dataset)}")
    
    # Create data loaders
    print("\n[3/5] Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  -> Train batches: {len(train_loader)}")
    print(f"  -> Val batches: {len(val_loader)}")
    print(f"  -> Test batches: {len(test_loader)}")
    
    # Create model
    print("\n[4/5] Creating SpaceTimeFormer model...")
    # Use windowed attention to reduce memory usage
    # With many features, the flattened sequence can be very long
    # (e.g., 434 features * 96 timesteps = 41,664 tokens)
    # Windowed attention limits attention to local windows to avoid OOM errors
    seq_length = context_length * n_features
    window_size = min(200, seq_length // 10)  # Adaptive window size (10% of sequence)
    window_size = max(50, window_size)  # Minimum window size
    
    print(f"  -> Sequence length (flattened): {seq_length}")
    print(f"  -> Using windowed attention with window size: {window_size}")
    print(f"  -> NOTE: Windowed attention may reduce model capacity compared to full attention")
    print(f"  ->      Consider feature selection or reducing context_length for better results")
    
    model = SpaceTimeFormer(
        n_variables=n_features,
        d_model=d_model,
        n_heads=n_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        d_ff=d_ff,
        dropout=dropout,
        context_points=context_length,
        target_points=target_length,
        use_windowed_attn=True,  # Enable windowed attention for memory efficiency
        window_size=window_size
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Total parameters: {total_params:,}")
    print(f"  -> Trainable parameters: {trainable_params:,}")
    
    # Set save path
    if save_model_path is None:
        save_model_path = os.path.join(
            os.path.dirname(__file__),
            'checkpoints',
            'spacetimeformer_best.pth'
        )
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    
    # Identify target indices for loss calculation
    # We want to minimize loss on 'High', 'Low', 'Close'
    target_indices = []
    
    print(f"  -> Identifying target indices for columns: {loss_target_cols}")
    for col in loss_target_cols:
        try:
            idx = feature_names.index(col)
            target_indices.append(idx)
            print(f"    - Found '{col}' at index {idx}")
        except ValueError:
            print(f"    - WARNING: Target column '{col}' not found in features!")
            
    if not target_indices:
        print("  -> ERROR: No target columns found! Defaulting to index 0.")
        target_indices = [0]
    
    # Train model
    print("\n[5/5] Training model...")
    print("-" * 80)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        patience=15,
        save_path=save_model_path,
        target_indices=target_indices,
        scaler=scaler  # Pass scaler to save in checkpoint
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved to: {save_model_path}")
    print(f"Best validation R²: {max(history['val_r2']):.6f}")
    print(f"Best validation RMSE: {min(history['val_rmse']):.6f}")
    
    return model, history, scaler, feature_names


if __name__ == '__main__':
    # Example usage
    model, history, scaler, feature_names = main(
        data_path='../es_with_indicators.csv',
        context_length=96,
        target_length=24,
        d_model=128,
        n_heads=8,
        enc_layers=3,
        dec_layers=3,
        batch_size=32,
        n_epochs=100,
        learning_rate=1e-3
    )
