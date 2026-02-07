"""
Training utilities for SpaceTimeFormer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, TYPE_CHECKING, Callable
import torch.amp
import numpy as np
import time
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .utils import get_device, empty_cache

if TYPE_CHECKING:
    from .model import SpaceTimeFormer


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
    target_indices: list[int],
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    batch_save_callback: Optional[Callable[[int], None]] = None,
    save_interval: int = 5
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
        use_amp: Whether to use Automatic Mixed Precision (FP16)
        scaler: GradScaler for mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(train_loader)
    
    for batch_idx, (batch_X, batch_target_X) in enumerate(train_loader):
        # Non-blocking transfer for better GPU utilization
        batch_X = batch_X.to(device, non_blocking=True)  # (batch_size, context_length, n_features)
        batch_target_X = batch_target_X.to(device, non_blocking=True) # (batch_size, target_length, n_features)
        
        # Zero gradients only at the start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        # autocast requires device_type as string
        device_type = device.type
        if device_type not in ['cuda', 'cpu', 'mps']:
            device_type = 'cpu'
            
        with torch.autocast(device_type=device_type, enabled=use_amp):
            # Use full target features for teacher forcing
            predictions = model(batch_X, batch_target_X)
            
            # Predictions shape: (batch_size, target_length, n_variables)
            # We compute loss on the specific target columns
            target_predictions = predictions[:, :, target_indices]
            
            # Ground truth for these specific columns
            target_truth = batch_target_X[:, :, target_indices]
            
            # Calculate loss (scale by accumulation steps for correct averaging)
            loss = criterion(target_predictions, target_truth) / gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping for stability
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Store loss value (multiply by accumulation steps to get true loss)
        loss_value = loss.item() * gradient_accumulation_steps
        
        # Delete intermediate tensors to free memory
        del batch_X, batch_target_X, predictions, target_predictions, target_truth, loss
        
        # Periodic memory cleanup for large batches
        # Increased interval to 200 for high-end Macs to reduce overhead
        if n_batches % 200 == 0:
            empty_cache(device)
        
        total_loss += loss_value
        n_batches += 1
        
        # Print progress every 10% of batches or every 100 batches, whichever is more frequent
        # Reducing frequency to avoid console I/O bottleneck with fast GPUs
        progress_pct = ((batch_idx + 1) / total_batches) * 100
        avg_loss = total_loss / n_batches
        
        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or (batch_idx + 1) % 100 == 0:
            print(f"    Batch [{batch_idx + 1}/{total_batches}] ({progress_pct:.1f}%) | "
                  f"Avg Loss: {avg_loss:.6f} | Current: {loss_value:.6f}")
        
        # Periodic batch saving if requested
        if batch_save_callback is not None and (batch_idx + 1) % save_interval == 0:
            batch_save_callback(batch_idx + 1)
    
    avg_train_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"  Training complete. Average loss: {avg_train_loss:.6f}")
    return avg_train_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_indices: list[int],
    use_amp: bool = False
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
            batch_X = batch_X.to(device, non_blocking=True)
            batch_target_X = batch_target_X.to(device, non_blocking=True)
            
            # Autoregressive generation with mixed precision
            # For true multi-step forecasting, we feed predictions back as input
            batch_size = batch_X.shape[0]
            target_length = batch_target_X.shape[1]
            
            device_type = device.type
            if device_type not in ['cuda', 'cpu', 'mps']:
                device_type = 'cpu'
            with torch.autocast(device_type=device_type, enabled=use_amp):
                encoder_output = model.encode(batch_X) # type: ignore
            
            # Initial input: Last context step
            # Shape: (batch_size, 1, n_features)
            current_input = batch_X[:, -1:, :]
            
            # Store predictions
            predictions_list = []
            
            for t in range(target_length):
                # Predict next step using accumulated sequence and cached encoder output
                # Note: We still re-process the decoder sequence, but avoid re-encoding context
                
                device_type = device.type
                if device_type not in ['cuda', 'cpu', 'mps']:
                    device_type = 'cpu'
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    step_predictions = model.decode(current_input, encoder_output) # type: ignore
                
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
            
            # Calculate loss with mixed precision
            device_type = device.type
            if device_type not in ['cuda', 'cpu', 'mps']:
                device_type = 'cpu'
            with torch.autocast(device_type=device_type, enabled=use_amp):
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


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    batch_idx: Optional[int],
    val_loss: float,
    val_metrics: dict,
    history: dict,
    save_path: str,
    scaler: Optional[object] = None
):
    """
    Save model checkpoint with architecture parameters.
    """
    # Extract architecture parameters from model
    try:
        # Get n_heads from first encoder layer
        encoder_layer = list(model.encoder)[0]  # type: ignore
        if hasattr(encoder_layer.attention, 'n_heads'):
            n_heads = encoder_layer.attention.n_heads
        elif hasattr(encoder_layer.attention, 'attention') and hasattr(encoder_layer.attention.attention, 'n_heads'):
            n_heads = encoder_layer.attention.attention.n_heads
        else:
            n_heads = 4  # Default fallback
        
        # Get d_ff from first encoder layer's feed-forward network
        if hasattr(encoder_layer, 'ff') and len(encoder_layer.ff) > 0:
            d_ff = encoder_layer.ff[0].out_features  # type: ignore
        else:
            d_ff = 256
        
        # Get use_windowed_attn and window_size
        use_windowed_attn = hasattr(encoder_layer.attention, 'window_size')
        window_size = encoder_layer.attention.window_size if use_windowed_attn else 50  # type: ignore
        
        # Get dropout
        dropout = encoder_layer.dropout.p if hasattr(encoder_layer, 'dropout') else 0.1  # type: ignore
        
        # Get max_seq_length from embedding
        max_seq_length = model.embedding.max_seq_length if hasattr(model.embedding, 'max_seq_length') else 1000  # type: ignore
        
        # Get layer counts
        enc_layers = len(list(model.encoder))  # type: ignore
        dec_layers = len(list(model.decoder))  # type: ignore
        
    except Exception as e:
        print(f"  -> Warning: Could not extract all architecture parameters: {e}")
        # Use defaults or reasonable estimates
        n_heads = getattr(model, 'n_heads', 4)
        d_ff = getattr(model, 'd_ff', 512)
        use_windowed_attn = getattr(model, 'use_windowed_attn', True)
        window_size = getattr(model, 'window_size', 200)
        dropout = getattr(model, 'dropout', 0.1)
        max_seq_length = getattr(model, 'max_seq_length', 1000)
        enc_layers = len(list(model.encoder)) if hasattr(model, 'encoder') else 3
        dec_layers = len(list(model.decoder)) if hasattr(model, 'decoder') else 3
    
    checkpoint_data = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'history': history,
        # Architecture parameters
        'n_variables': model.n_variables,
        'd_model': model.d_model,
        'n_heads': n_heads,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'max_seq_length': max_seq_length,
        'context_points': model.context_points,
        'target_points': model.target_points,
        'use_windowed_attn': use_windowed_attn,
        'window_size': window_size,
        'timestamp': time.time()
    }
    
    # Add scaler if provided
    if scaler is not None:
        import pickle
        import io
        scaler_bytes = io.BytesIO()
        pickle.dump(scaler, scaler_bytes)
        checkpoint_data['scaler'] = scaler_bytes.getvalue()
    
    # Save checkpoint
    torch.save(checkpoint_data, save_path)
    # Also save a temporary latest checkpoint if requested frequently
    if batch_idx is not None:
        latest_path = save_path.replace('.pth', '_latest.pth')
        torch.save(checkpoint_data, latest_path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    patience: int = 10,
    save_path: Optional[str] = None,
    target_indices: Optional[list[int]] = None,
    scaler: Optional[object] = None,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    compile_model: bool = False,
    start_epoch: int = 0,
    resume_history: Optional[dict] = None,
    resume_optimizer_state: Optional[dict] = None,
    resume_scaler_state: Optional[bytes] = None
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
        device = get_device()
    
    model = model.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # Note: torch.compile has limited support on MPS currently
    if compile_model and hasattr(torch, 'compile'):
        if device.type == 'mps':
            print("  -> Warning: torch.compile() has limited support on MPS. Skipping compilation.")
        else:
            print("  -> Compiling model with torch.compile() for faster execution...")
            try:
                # torch.compile returns a callable, but we can still use it as a model
                compiled_model = torch.compile(model, mode='reduce-overhead')  # type: ignore
                model = compiled_model  # type: ignore
                print("  -> Model compiled successfully")
            except Exception as e:
                print(f"  -> Warning: Could not compile model: {e}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    # Using fused=True if supported for faster weight updates on GPU
    optimizer_kwargs = {'lr': learning_rate, 'weight_decay': 1e-4}
    
    # Fused AdamW is faster on CUDA and recent MPS versions
    if device.type in ['cuda', 'mps']:
        try:
            # Check if fused is supported in this PyTorch version
            optimizer = optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            print("  -> Using Fused AdamW optimizer for faster weight updates")
        except Exception:
            optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
            print("  -> Using Standard AdamW optimizer")
    else:
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    if resume_optimizer_state:
        print("  -> Resuming optimizer state")
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except Exception as e:
            print(f"  -> Warning: Could not resume optimizer state: {e}")
    
    # Mixed precision scaler
    amp_scaler = None
    if use_amp:
        if device.type == 'cuda':
            amp_scaler = torch.amp.GradScaler(device_type='cuda')
            print("  -> Using Automatic Mixed Precision (FP16) for faster training on CUDA")
        elif device.type == 'mps':
            try:
                # GradScaler support for MPS is in newer PyTorch
                amp_scaler = torch.amp.GradScaler(device_type='mps')
                print("  -> Using Automatic Mixed Precision (FP16) for faster training on MPS")
            except Exception as e:
                print(f"  -> Warning: MPS GradScaler not supported or failed: {e}. Disabling AMP.")
                use_amp = False
        
        # Resume scaler state if available
        if amp_scaler and resume_scaler_state:
            print("  -> Resuming scaler state")
            try:
                import pickle
                import io
                scaler_state = pickle.loads(resume_scaler_state)
                amp_scaler.load_state_dict(scaler_state.state_dict())
            except Exception as e:
                print(f"  -> Warning: Could not resume scaler state: {e}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5
    )
    
    # Training history
    if resume_history:
        history = resume_history
        print(f"  -> Resuming from history with {len(history['train_loss'])} entries")
    else:
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': []
        }
    
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    best_val_r2 = max(history['val_r2']) if history['val_r2'] else float('-inf')
    best_val_rmse = min(history['val_rmse']) if history['val_rmse'] else float('inf')
    patience_counter = 0
    
    # Define batch save callback
    def on_batch_end(batch_idx):
        if save_path:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                batch_idx=batch_idx,
                val_loss=best_val_loss if best_val_loss != float('inf') else 0.0,
                val_metrics=history.get('val_metrics', {}), # Use last val metrics or empty
                history=history,
                save_path=save_path,
                scaler=scaler
            )

    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    effective_batch_size = (train_loader.batch_size or 1) * gradient_accumulation_steps
    print(f"Effective Batch Size: {effective_batch_size}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {total_memory:.2f} GB")
        
        # Show current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        
        try:
            # Try to get CUDA version if available
            version_module = getattr(torch, 'version', None)
            if version_module:
                cuda_version = getattr(version_module, 'cuda', 'Unknown')
                if cuda_version:
                    print(f"CUDA Version: {cuda_version}")
        except (AttributeError, TypeError):
            pass
    elif device.type == 'mps':
        print(f"GPU: Apple Metal (MPS)")
        allocated = 0.0
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / 1e9
            print(f"  Allocated: {allocated:.2f} GB")
        
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'recommended_max_memory'):
            total_memory = torch.mps.recommended_max_memory() / 1e9
            print(f"  Total Memory (Recommended): {total_memory:.2f} GB")
            print(f"  Available (Estimated): {max(0, total_memory - allocated):.2f} GB")
    print(f"{'='*80}\n")
    
    if target_indices is None:
        target_indices = [0]
    
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, target_indices,
            use_amp=use_amp, scaler=amp_scaler, gradient_accumulation_steps=gradient_accumulation_steps,
            batch_save_callback=on_batch_end,
            save_interval=5
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, target_indices, use_amp=use_amp)
        
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
            total = torch.cuda.get_device_properties(device).total_memory / 1e9
            utilization = (reserved / total) * 100
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved ({utilization:.1f}% utilization)")
        elif device.type == 'mps':
            utilization = 0.0
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory() / 1e9
                print(f"MPS Memory: {allocated:.2f} GB allocated")
                
                if hasattr(torch.mps, 'recommended_max_memory'):
                    total = torch.mps.recommended_max_memory() / 1e9
                    utilization = (allocated / total) * 100
                    print(f"  Utilization (Recommended): {utilization:.1f}%")
            
            # Warn if memory usage is low (could use larger batch)
            if utilization > 0 and utilization < 60:
                print(f"  → Consider increasing batch_size (currently using {utilization:.1f}% of GPU memory)")
        
        print(f"{'='*80}")
        
        # Early stopping and model saving
        if is_best_loss:
            patience_counter = 0
            
            # Save best model
            if save_path:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    batch_idx=None, # None indicates end of epoch
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    history=history,
                    save_path=save_path,
                    scaler=scaler
                )
                print(f"  → Saved best model to: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"   No improvement for {patience} epochs")
                break
    
    return history

