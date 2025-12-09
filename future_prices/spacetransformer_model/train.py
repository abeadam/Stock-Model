"""
Training utilities for SpaceTimeFormer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np
import gc
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.
    Creates sequences of context and target windows from time series data.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        context_length: int = 96,
        target_length: int = 24,
        stride: int = 1
    ):
        """
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            context_length: Number of timesteps to use as context
            target_length: Number of timesteps to predict
            stride: Stride for creating sequences
        """
        self.X = X
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride
        
        # Calculate valid sequence indices
        self.valid_indices = []
        for i in range(len(X) - context_length - target_length + 1):
            if i % stride == 0:
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Get context sequence (features)
        context_X = self.X[start_idx:start_idx + self.context_length]
        
        # Get target sequence (full features for target window)
        target_start = start_idx + self.context_length
        target_X = self.X[target_start:target_start + self.target_length]
        
        # Convert to tensors
        context_X = torch.FloatTensor(context_X)
        target_X = torch.FloatTensor(target_X)
        
        return context_X, target_X


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_indices: list[int]
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: SpaceTimeFormer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        target_indices: Indices of the target variables in the feature set (for loss calculation)
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(train_loader)
    
    for batch_idx, (batch_X, batch_target_X) in enumerate(train_loader):
        batch_X = batch_X.to(device)  # (batch_size, context_length, n_features)
        batch_target_X = batch_target_X.to(device) # (batch_size, target_length, n_features)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        
        # Use full target features for teacher forcing
        predictions = model(batch_X, batch_target_X)
        
        # Predictions shape: (batch_size, target_length, n_variables)
        # We compute loss on the specific target columns
        target_predictions = predictions[:, :, target_indices]
        
        # Ground truth for these specific columns
        target_truth = batch_target_X[:, :, target_indices]
        
        # Calculate loss
        loss = criterion(target_predictions, target_truth)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Store loss value before deleting tensors
        loss_value = loss.item()
        
        # Delete intermediate tensors to free memory
        del batch_X, batch_target_X, predictions, target_predictions, target_truth, loss
        
        total_loss += loss_value
        n_batches += 1
        
        # Print progress every 10% of batches or every 50 batches, whichever is more frequent
        progress_pct = ((batch_idx + 1) / total_batches) * 100
        avg_loss = total_loss / n_batches
        print(f"    Batch [{batch_idx + 1}/{total_batches}] ({progress_pct:.1f}%) | "
              f"Avg Loss: {avg_loss:.6f} | Current: {loss_value:.6f}")
    
    avg_train_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"  Training complete. Average loss: {avg_train_loss:.6f}")
    return avg_train_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_indices: list[int]
) -> Tuple[float, dict]:
    """
    Validate model.
    
    Args:
        model: SpaceTimeFormer model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    n_batches = 0
    total_batches = len(val_loader)
    
    print(f"  Starting validation ({total_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_target_X) in enumerate(val_loader):
            batch_X = batch_X.to(device)
            batch_target_X = batch_target_X.to(device)
            
            # Autoregressive generation
            # For true multi-step forecasting, we feed predictions back as input
            batch_size = batch_X.shape[0]
            target_length = batch_target_X.shape[1]
            
            # Initial input: Last context step
            # Shape: (batch_size, 1, n_features)
            current_input = batch_X[:, -1:, :]
            
            # Store predictions
            predictions_list = []
            
            for t in range(target_length):
                # Predict next step using accumulated sequence
                # Note: Re-running full forward pass is inefficient but correct for this architecture
                step_predictions = model(batch_X, current_input)
                
                # Get the prediction for the last step
                # Shape: (batch_size, 1, n_features)
                next_pred = step_predictions[:, -1:, :]
                
                predictions_list.append(next_pred)
                
                # Append to input for next iteration
                current_input = torch.cat([current_input, next_pred], dim=1)
            
            # Concatenate all predictions
            # Shape: (batch_size, target_length, n_features)
            predictions = torch.cat(predictions_list, dim=1)
            
            # Predictions shape: (batch_size, target_length, n_variables)
            # We compute loss on the specific target columns
            target_predictions = predictions[:, :, target_indices]
            
            # Ground truth
            target_truth = batch_target_X[:, :, target_indices]
            
            # Calculate loss
            loss = criterion(target_predictions, target_truth)
            
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.append(target_predictions.cpu().numpy())
            all_targets.append(target_truth.cpu().numpy())
            
            n_batches += 1
            
            # Print progress every 10% of batches or every 20 batches, whichever is more frequent
            progress_interval = max(1, min(5, total_batches // 10))
            if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
                progress_pct = ((batch_idx + 1) / total_batches) * 100
                avg_loss = total_loss / n_batches
                print(f"    Batch [{batch_idx + 1}/{total_batches}] ({progress_pct:.1f}%) | "
                      f"Avg Loss: {avg_loss:.6f} | Current: {loss.item():.6f}")
            
            # Clear CUDA cache every 10 iterations to free up memory
            # if n_batches % 10 == 0 and device.type == 'cuda':
            #     torch.cuda.empty_cache()
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Flatten for metric calculation
    predictions_flat = all_predictions.flatten()
    targets_flat = all_targets.flatten()
    
    mse = mean_squared_error(targets_flat, predictions_flat)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_flat, predictions_flat)
    
    metrics = {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"  Validation complete. Loss: {metrics['loss']:.6f}, R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    return metrics['loss'], metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    patience: int = 10,
    save_path: Optional[str] = None,
    target_indices: Optional[list[int]] = None
) -> dict:
    """
    Train SpaceTimeFormer model.
    
    Args:
        model: SpaceTimeFormer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (defaults to cuda if available)
        patience: Early stopping patience
        save_path: Path to save best model
        target_idx: Index of the target variable in the feature set (for loss calculation)
    
    Returns:
        Training history dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_val_r2 = float('-inf')
    best_val_rmse = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    if target_indices is None:
        target_indices = [0]
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, target_indices)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, target_indices)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Check for improvements
        is_best_loss = val_loss < best_val_loss
        is_best_r2 = val_metrics['r2'] > best_val_r2
        is_best_rmse = val_metrics['rmse'] < best_val_rmse
        
        if is_best_loss:
            best_val_loss = val_loss
        if is_best_r2:
            best_val_r2 = val_metrics['r2']
        if is_best_rmse:
            best_val_rmse = val_metrics['rmse']
        
        # Print detailed epoch results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{n_epochs} (Time: {epoch_time:.2f}s)")
        print(f"{'='*80}")
        print(f"Training Metrics:")
        print(f"  Loss: {train_loss:.6f}")
        print(f"\nValidation Metrics:")
        print(f"  Loss:     {val_loss:.6f} {'✓ BEST' if is_best_loss else ''}")
        print(f"  MSE:      {val_metrics['mse']:.6f}")
        print(f"  MAE:      {val_metrics['mae']:.6f}")
        print(f"  RMSE:     {val_metrics['rmse']:.6f} {'✓ BEST' if is_best_rmse else ''}")
        print(f"  R²:       {val_metrics['r2']:.6f} {'✓ BEST' if is_best_r2 else ''}")
        print(f"\nBest So Far:")
        print(f"  Loss:     {best_val_loss:.6f}")
        print(f"  RMSE:     {best_val_rmse:.6f}")
        print(f"  R²:       {best_val_r2:.6f}")
        print(f"\nLearning Rate: {current_lr:.2e}", end="")
        if new_lr != current_lr:
            print(f" → {new_lr:.2e} (reduced)")
        else:
            print()
        
        # GPU memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        print(f"{'='*80}")
        
        # Early stopping and model saving
        if is_best_loss:
            patience_counter = 0
            
            # Save best model
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'history': history
                }, save_path)
                print(f"  → Saved best model to: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"   No improvement for {patience} epochs")
                break
    
    return history

