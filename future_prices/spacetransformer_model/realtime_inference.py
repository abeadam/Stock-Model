"""
Fast real-time inference for SpaceTimeFormer model.
Optimized for low-latency predictions in trading systems.

Usage:
    from realtime_inference import RealtimePredictor
    
    predictor = RealtimePredictor(checkpoint_path='path/to/checkpoint.pth')
    predictions = predictor.predict(context_data)  # Returns predictions for next 24 steps
"""

import sys
import os
import torch
import numpy as np
import pickle
import io
import time
import warnings
from typing import Optional, Tuple
from pathlib import Path

# Suppress TracerWarnings during torch.compile() - these are safe for inference
# Filter by both message pattern and category
warnings.filterwarnings('ignore', message='.*TracerWarning.*')
warnings.filterwarnings('ignore', message='.*Converting a tensor.*')
warnings.filterwarnings('ignore', message='.*torch.tensor results are registered.*')
warnings.filterwarnings('ignore', message='.*torch.as_tensor results.*')
# Also suppress at logging level
import logging
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
logging.getLogger('torch.jit').setLevel(logging.ERROR)

# Add parent directory to path to allow imports when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.model import SpaceTimeFormer
from spacetransformer_model.utils import get_device


class RealtimePredictor:
    """
    Fast real-time predictor for SpaceTimeFormer model.
    Optimized for low-latency inference in trading systems.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_parallel: bool = True,
        compile_model: bool = True,
        use_quantization: bool = False,
        use_fp16: bool = False
    ):
        """
        Initialize the predictor by loading model and scaler from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            use_parallel: If True, predict all steps in parallel (faster but less accurate)
                         If False, use autoregressive prediction (slower but more accurate)
            compile_model: If True, compile model with torch.compile() for faster inference (PyTorch 2.0+)
            use_quantization: If True, use INT8 quantization for 2-4x speedup (may reduce accuracy slightly)
            use_fp16: If True, use FP16 half precision for 2x speedup on GPU (less memory, faster)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self.use_parallel = use_parallel
        self.use_quantization = use_quantization
        self.use_fp16 = use_fp16
        
        print(f"Loading model from {checkpoint_path}...")
        print(f"  -> Device: {self.device}")
        print(f"  -> Prediction mode: {'parallel (fast)' if use_parallel else 'autoregressive (accurate)'}")
        
        start_time = time.time()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Validate checkpoint
        if 'model_state_dict' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'model_state_dict'. Keys: {list(checkpoint.keys())}")
        
        # Extract architecture parameters
        required_params = [
            'n_variables', 'd_model', 'n_heads', 'enc_layers', 'dec_layers',
            'd_ff', 'context_points', 'target_points'
        ]
        missing_params = [p for p in required_params if p not in checkpoint]
        if missing_params:
            raise ValueError(f"Checkpoint missing parameters: {missing_params}")
        
        # Create model
        self.model = SpaceTimeFormer(
            n_variables=checkpoint['n_variables'],
            d_model=checkpoint['d_model'],
            n_heads=checkpoint['n_heads'],
            enc_layers=checkpoint['enc_layers'],
            dec_layers=checkpoint['dec_layers'],
            d_ff=checkpoint['d_ff'],
            dropout=checkpoint.get('dropout', 0.1),
            max_seq_length=checkpoint.get('max_seq_length', 1000),
            context_points=checkpoint['context_points'],
            target_points=checkpoint['target_points'],
            use_windowed_attn=checkpoint.get('use_windowed_attn', True),
            window_size=checkpoint.get('window_size', 200)
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store model dtype for consistent tensor creation
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Apply FP16 half precision if requested (2x speedup on GPU, less memory)
        if use_fp16:
            if self.device.type == 'cuda':
                # Use automatic mixed precision or convert to half
                self.model = self.model.half()
                self.model_dtype = torch.float16
                print("  -> Model converted to FP16 (half precision)")
            elif self.device.type == 'mps':
                # MPS also supports float16
                self.model = self.model.to(torch.float16)
                self.model_dtype = torch.float16
                print("  -> Model converted to FP16 (half precision) for MPS")
            else:
                print("  -> FP16 only available on CUDA/MPS, skipping")
                use_fp16 = False
        
        # Apply quantization if requested (2-4x speedup, slight accuracy loss)
        if use_quantization:
            try:
                if hasattr(torch.quantization, 'quantize_dynamic'):
                    # Dynamic quantization: quantize linear layers to INT8
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, 
                        {torch.nn.Linear}, 
                        dtype=torch.qint8
                    )
                    print("  -> Model quantized to INT8 (2-4x faster, slight accuracy loss)")
                else:
                    print("  -> Quantization not available in this PyTorch version")
                    use_quantization = False
            except Exception as e:
                print(f"  -> Quantization failed: {e}, using full precision")
                use_quantization = False
        
        # Optimize model for faster inference
        if compile_model:
            try:
                # Set faster matmul precision (CUDA/NVIDIA GPUs only)
                # Note: Mac MPS doesn't support this, but it's safe to call
                if hasattr(torch, 'set_float32_matmul_precision') and self.device.type == 'cuda':
                    torch.set_float32_matmul_precision('high')  # or 'medium' for better accuracy
                
                # Enable cuDNN benchmarking for consistent input shapes (faster)
                if self.device.type == 'cuda' and hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False  # Faster, but non-deterministic
                
                # Configure dynamo to capture scalar outputs (fixes graph breaks from .item() calls)
                if hasattr(torch, '_dynamo') and hasattr(torch._dynamo.config, 'capture_scalar_outputs'):
                    torch._dynamo.config.capture_scalar_outputs = True
                
                # Suppress TracerWarnings during compilation
                import logging
                logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
                
                # Disable gradient checkpointing during inference for speed
                # Checkpointing is only needed during training, not inference
                import spacetransformer_model.attention as attention_module
                from torch.utils.checkpoint import checkpoint as original_checkpoint
                
                def no_checkpoint_inference(func, *args, **kwargs):
                    """Disable checkpointing during inference - just call function directly"""
                    export_kwargs = {k: v for k, v in kwargs.items() if k != 'use_reentrant'}
                    return func(*args, **export_kwargs)
                
                # Store original and replace with no-op checkpoint for inference
                self._original_checkpoint = attention_module.checkpoint
                attention_module.checkpoint = no_checkpoint_inference
                self._checkpoint_patched = True
                
                # Compile model for faster inference (PyTorch 2.0+)
                # Works on both CUDA (NVIDIA) and MPS (Mac) backends
                # 'reduce-overhead' mode: Optimizes for inference speed
                # 'max-autotune' mode: Maximum optimization (slower first compilation, faster runtime)
                if hasattr(torch, 'compile'):
                    # On Mac MPS, 'reduce-overhead' is typically the best mode
                    # On CUDA, both modes work, but 'reduce-overhead' is faster to compile
                    compile_mode = 'reduce-overhead'
                    print(f"  -> Compiling model (this may take 30-60 seconds, please wait)...")
                    print(f"     Mode: {compile_mode}")
                    compile_start = time.time()
                    try:
                        self.model = torch.compile(self.model, mode=compile_mode)
                        compile_time = time.time() - compile_start
                        backend = 'MPS (Mac)' if self.device.type == 'mps' else 'CUDA' if self.device.type == 'cuda' else 'CPU'
                        print(f"  -> Model compiled successfully in {compile_time:.1f}s (mode: {compile_mode}, backend: {backend})")
                    except Exception as compile_error:
                        compile_time = time.time() - compile_start
                        print(f"  -> Compilation failed after {compile_time:.1f}s: {compile_error}")
                        print(f"  -> Using uncompiled model (will be slower but should work)")
                        # Model is already loaded, just not compiled
                else:
                    print("  -> torch.compile not available (PyTorch < 2.0), using standard model")
            except Exception as e:
                print(f"  -> Model compilation failed: {e}, using standard model")
        else:
            print("  -> Model compilation disabled")
        
        # Load scaler
        self.scaler = None
        if 'scaler' in checkpoint:
            scaler_bytes = io.BytesIO(checkpoint['scaler'])
            self.scaler = pickle.load(scaler_bytes)
            print("  -> Scaler loaded from checkpoint")
        else:
            print("  -> WARNING: No scaler in checkpoint, predictions will be in scaled space")
        
        # Store architecture info
        self.context_points = checkpoint['context_points']
        self.target_points = checkpoint['target_points']
        self.n_variables = checkpoint['n_variables']
        
        # Track if checkpoint was patched
        self._checkpoint_patched = False
        
        init_time = time.time() - start_time
        print(f"  -> Model loaded: {self.context_points} context -> {self.target_points} target steps")
        print(f"  -> Initialization completed in {init_time:.4f} seconds")
        
        # Pre-warmup the model to ensure compilation is complete
        if compile_model:
            print("  -> Pre-warming model (ensuring compilation is complete)...")
            try:
                warmup_data = torch.randn(1, self.context_points, self.n_variables, 
                                         device=self.device, dtype=self.model_dtype)
                if self.use_parallel:
                    dummy_target = warmup_data[:, -1:, :].repeat(1, min(2, self.target_points), 1)
                    _ = self.model(warmup_data, dummy_target)
                else:
                    _ = self.model.encode(warmup_data)
                    dummy_target = warmup_data[:, -1:, :]
                    _ = self.model.decode(dummy_target, _)
                print("  -> Model warmed up and ready")
            except Exception as e:
                print(f"  -> Warmup failed (non-critical): {e}")
        
        print("  -> Ready for inference")
    
    def predict(
        self,
        context_data: np.ndarray,
        target_indices: Optional[list[int]] = None,
        return_scaled: bool = False,
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict future values given context data.
        
        Args:
            context_data: Input context sequence
                         Shape: (context_points, n_features) or (1, context_points, n_features)
                         If 2D, will be expanded to batch dimension
            target_indices: Indices of target variables to return (e.g., [High_idx, Low_idx, Close_idx])
                          If None, returns all variables
            return_scaled: If True, return scaled predictions. If False, inverse transform using scaler.
            target_length: Number of future steps to predict (default: self.target_points, typically 24)
                          Use smaller values (e.g., 2) for faster inference
        
        Returns:
            Predictions array
            Shape: (target_length, n_targets) if target_indices specified
                   (target_length, n_variables) if target_indices is None
            Values are in original (unscaled) space unless return_scaled=True
        """
        # Convert to numpy if needed
        if isinstance(context_data, torch.Tensor):
            context_data = context_data.cpu().numpy()
        
        # Ensure 3D: (batch, time, features)
        if context_data.ndim == 2:
            context_data = context_data[np.newaxis, :, :]  # Add batch dimension
        
        batch_size, seq_len, n_features = context_data.shape
        
        # Validate input shape
        if seq_len != self.context_points:
            raise ValueError(
                f"Context length mismatch: got {seq_len}, expected {self.context_points}"
            )
        
        # Ensure features match model (slice if needed)
        if n_features > self.n_variables:
            context_data = context_data[:, :, :self.n_variables]
            n_features = self.n_variables
        elif n_features < self.n_variables:
            raise ValueError(
                f"Feature count mismatch: got {n_features}, model expects {self.n_variables}"
            )
        
        # Scale if scaler is available and we want unscaled output
        if self.scaler is not None and not return_scaled:
            # Reshape for scaler: (batch * time, features)
            context_flat = context_data.reshape(-1, n_features)
            context_scaled = self.scaler.transform(context_flat)
            context_data = context_scaled.reshape(batch_size, seq_len, n_features)
        
        # Convert to tensor (use model's dtype for consistency)
        context_tensor = torch.tensor(context_data, dtype=self.model_dtype).to(self.device)
        
        # Determine target length
        actual_target_length = target_length if target_length is not None else self.target_points
        
        # Generate predictions
        # Use torch.inference_mode() instead of no_grad() for faster inference
        # inference_mode is faster because it disables autograd entirely
        with torch.inference_mode():
            if self.use_parallel:
                # Fast parallel prediction: all steps at once
                last_step = context_tensor[:, -1:, :]  # (batch, 1, features)
                dummy_target = last_step.repeat(1, actual_target_length, 1)  # (batch, target_length, features)
                predictions = self.model(context_tensor, dummy_target)  # (batch, target_length, n_variables)
            else:
                # Accurate autoregressive prediction: one step at a time
                # Optimized: encode once, pre-allocate output tensor
                encoder_output = self.model.encode(context_tensor)  # type: ignore
                
                # Pre-allocate predictions tensor for better performance
                predictions = torch.zeros(
                    batch_size, actual_target_length, self.n_variables,
                    device=self.device, dtype=context_tensor.dtype
                )
                
                # Start with last context step
                current_input = context_tensor[:, -1:, :]
                
                # Optimized decode loop: only decode the new token each step
                for t in range(actual_target_length):
                    # Decode current sequence (grows by 1 each step)
                    step_pred = self.model.decode(current_input, encoder_output)  # type: ignore
                    next_pred = step_pred[:, -1:, :]  # Get last step prediction
                    
                    # Store prediction
                    predictions[:, t:t+1, :] = next_pred
                    
                    # Append only the new prediction (more efficient than full concat)
                    current_input = torch.cat([current_input, next_pred], dim=1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        
        # Extract target indices if specified
        if target_indices is not None:
            predictions = predictions[:, :, target_indices]
        
        # Remove batch dimension if single sample
        if batch_size == 1:
            predictions = predictions[0]  # (target_points, n_targets or n_variables)
        
        # Inverse transform if scaler available and we want unscaled output
        if self.scaler is not None and not return_scaled:
            # Reshape for scaler: (time, features)
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            
            # Create full feature array for inverse transform
            if target_indices is not None:
                # Need to reconstruct full feature vector
                n_targets = len(target_indices)
                n_features = self.n_variables
                pred_full = np.zeros((pred_flat.shape[0], n_features))
                for i, idx in enumerate(target_indices):
                    pred_full[:, idx] = pred_flat[:, i]
                pred_flat = pred_full
            
            # Inverse transform
            pred_unscaled = self.scaler.inverse_transform(pred_flat)
            
            # Extract target columns if needed
            if target_indices is not None:
                pred_unscaled = pred_unscaled[:, target_indices]
            
            # Reshape back
            predictions = pred_unscaled.reshape(predictions.shape[0], -1)
        
        return predictions
    
    def predict_batch(
        self,
        context_batch: np.ndarray,
        target_indices: Optional[list[int]] = None,
        return_scaled: bool = False
    ) -> np.ndarray:
        """
        Predict for a batch of context sequences.
        
        Args:
            context_batch: Batch of context sequences
                         Shape: (batch_size, context_points, n_features)
            target_indices: Indices of target variables to return
            return_scaled: If True, return scaled predictions
        
        Returns:
            Predictions array
            Shape: (batch_size, target_points, n_targets)
        """
        # Ensure 3D
        if context_batch.ndim == 2:
            context_batch = context_batch[np.newaxis, :, :]
        
        batch_size, seq_len, n_features = context_batch.shape
        
        # Validate
        if seq_len != self.context_points:
            raise ValueError(f"Context length mismatch: got {seq_len}, expected {self.context_points}")
        
        # Slice features if needed
        if n_features > self.n_variables:
            context_batch = context_batch[:, :, :self.n_variables]
        elif n_features < self.n_variables:
            raise ValueError(f"Feature count mismatch: got {n_features}, expected {self.n_variables}")
        
        # Scale if needed
        if self.scaler is not None and not return_scaled:
            context_flat = context_batch.reshape(-1, n_features)
            context_scaled = self.scaler.transform(context_flat)
            context_batch = context_scaled.reshape(batch_size, seq_len, n_features)
        
        # Convert to tensor (use model's dtype for consistency)
        context_tensor = torch.tensor(context_batch, dtype=self.model_dtype).to(self.device)
        
        # Generate predictions
        # Use torch.inference_mode() instead of no_grad() for faster inference
        with torch.inference_mode():
            if self.use_parallel:
                last_step = context_tensor[:, -1:, :]
                dummy_target = last_step.repeat(1, self.target_points, 1)
                predictions = self.model(context_tensor, dummy_target)
            else:
                # Optimized autoregressive: encode once, pre-allocate output
                encoder_output = self.model.encode(context_tensor)  # type: ignore
                
                # Pre-allocate predictions tensor
                predictions = torch.zeros(
                    batch_size, self.target_points, self.n_variables,
                    device=self.device, dtype=context_tensor.dtype
                )
                
                current_input = context_tensor[:, -1:, :]
                
                for t in range(self.target_points):
                    step_pred = self.model.decode(current_input, encoder_output)  # type: ignore
                    next_pred = step_pred[:, -1:, :]
                    predictions[:, t:t+1, :] = next_pred
                    current_input = torch.cat([current_input, next_pred], dim=1)
        
        # Convert to numpy (move to CPU only once)
        if predictions.device.type == 'cuda':
            predictions = predictions.cpu()
        predictions = predictions.numpy()
        
        # Extract targets
        if target_indices is not None:
            predictions = predictions[:, :, target_indices]
        
        # Inverse transform
        if self.scaler is not None and not return_scaled:
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            
            if target_indices is not None:
                n_targets = len(target_indices)
                n_features = self.n_variables
                pred_full = np.zeros((pred_flat.shape[0], n_features))
                for i, idx in enumerate(target_indices):
                    pred_full[:, idx] = pred_flat[:, i]
                pred_flat = pred_full
            
            pred_unscaled = self.scaler.inverse_transform(pred_flat)
            
            if target_indices is not None:
                pred_unscaled = pred_unscaled[:, target_indices]
            
            predictions = pred_unscaled.reshape(predictions.shape)
        
        return predictions


# Example usage
if __name__ == "__main__":
    import time
    
    # Example: Load model and make predictions
    # Resolve checkpoint path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'checkpoints', 'spacetimeformer_best.pth')
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the checkpoint file exists or update the checkpoint_path variable.")
        sys.exit(1)
    
    # Initialize predictor (fast parallel mode with compilation)
    print("Initializing RealtimePredictor...")
    start_time = time.time()
    predictor = RealtimePredictor(
        checkpoint_path=checkpoint_path,
        use_parallel=True,  # Fast mode for real-time trading
        compile_model=True,  # Compile for speed
        use_quantization=False  # Set to True for 2-4x speedup (slight accuracy loss)
    )

    
    # Example: Single prediction
    # Assume you have context data: (context_points, n_features)
    # context_data = np.random.randn(48, 438)  # Example: 48 time steps, 438 features
    
    # Predict next 24 steps for High, Low, Close (assuming indices 0, 1, 2)
    # predictions = predictor.predict(
    #     context_data=context_data,
    #     target_indices=[0, 1, 2],  # High, Low, Close indices
    #     return_scaled=False  # Return actual prices
    # )
    # # predictions shape: (24, 3) - 24 time steps, 3 targets

    # init_time = time.time() - start_time
    # print(f"  -> Initialization completed in {init_time:.4f} seconds")
    
    # Example: Batch prediction
    # batch_context = np.random.randn(10, 48, 438)  # 10 samples
    # batch_predictions = predictor.predict_batch(
    #     context_batch=batch_context,
    #     target_indices=[0, 1, 2],
    #     return_scaled=False
    # )
    # # batch_predictions shape: (10, 24, 3)
    
    print("\nExample usage:")
    print("  from spacetransformer_model.realtime_inference import RealtimePredictor")
    print("  predictor = RealtimePredictor('path/to/checkpoint.pth',")
    print("                                  use_parallel=True,")
    print("                                  compile_model=True,")
    print("                                  use_quantization=False)  # True for 2-4x speedup")
    print("  predictions = predictor.predict(context_data, target_indices=[0, 1, 2])")
    print("  # Returns: (24, 3) array with predictions for next 24 steps")
    print("\nOptimization options:")
    print("  - use_parallel=True: Predict all steps at once (~24x faster)")
    print("  - compile_model=True: Compile model for faster inference (1.5-3x speedup)")
    print("  - use_quantization=True: INT8 quantization (2-4x speedup, slight accuracy loss)")
    print("\nNote: First prediction may be slower (compilation warmup), subsequent predictions will be faster.")

