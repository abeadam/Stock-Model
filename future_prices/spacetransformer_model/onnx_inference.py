"""
ONNX Runtime inference for SpaceTimeFormer model.
Faster than PyTorch for inference, especially with quantization.

Usage:
    from onnx_inference import ONNXPredictor
    
    predictor = ONNXPredictor('model.onnx')
    predictions = predictor.predict(context_data)
"""

import sys
import os
import numpy as np
import pickle
import io
import time
from typing import Optional, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore  # Set to None if not available
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime onnxruntime-gpu")


class ONNXPredictor:
    """
    Fast ONNX Runtime predictor for SpaceTimeFormer model.
    Typically 1.5-3x faster than PyTorch for inference.
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        checkpoint_path: Optional[str] = None,
        use_parallel: bool = True,
        providers: Optional[list] = None,
        use_quantization: bool = False
    ):
        """
        Initialize ONNX Runtime predictor.
        
        Args:
            onnx_model_path: Path to ONNX model file
            checkpoint_path: Optional path to PyTorch checkpoint (for scaler and metadata)
            use_parallel: If True, predict all steps in parallel
            providers: ONNX Runtime execution providers (default: auto-detect)
            use_quantization: If True, use quantized model (requires separate quantized ONNX file)
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Install with: "
                "pip install onnxruntime onnxruntime-gpu"
            )
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        self.onnx_model_path = onnx_model_path
        self.use_parallel = use_parallel
        
        print(f"Loading ONNX model from {onnx_model_path}...")
        start_time = time.time()
        
        # Setup execution providers
        if providers is None:
            # Auto-detect: prefer GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Remove providers that aren't available
            if ort is not None:  # Type guard: ort is available if we got here
                available_providers = ort.get_available_providers()
                providers = [p for p in providers if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']
        
        print(f"  -> Execution providers: {providers}")
        
        # Create inference session
        # Type guard: ort is guaranteed to be available here (checked at start of __init__)
        assert ort is not None, "onnxruntime not available"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Use all available threads
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  -> Inputs: {self.input_names}")
        print(f"  -> Outputs: {self.output_names}")
        
        # Load metadata from checkpoint if available
        self.scaler = None
        self.context_points = None
        self.target_points = None
        self.n_variables = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Load scaler
                if 'scaler' in checkpoint:
                    scaler_bytes = io.BytesIO(checkpoint['scaler'])
                    self.scaler = pickle.load(scaler_bytes)
                    print("  -> Scaler loaded from checkpoint")
                
                # Load architecture info
                self.context_points = checkpoint.get('context_points')
                self.target_points = checkpoint.get('target_points')
                self.n_variables = checkpoint.get('n_variables')
                
                if self.context_points:
                    print(f"  -> Context points: {self.context_points}")
                if self.target_points:
                    print(f"  -> Target points: {self.target_points}")
                if self.n_variables:
                    print(f"  -> Variables: {self.n_variables}")
            except Exception as e:
                print(f"  -> Warning: Could not load checkpoint metadata: {e}")
        
        # If metadata not available, try to infer from model
        if self.context_points is None:
            # Try to get from input shape
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 2:
                self.context_points = input_shape[1] if input_shape[1] != 'context_length' else 48
            if self.n_variables is None and len(input_shape) >= 3:
                self.n_variables = input_shape[2] if input_shape[2] != 'n_variables' else 440
        
        init_time = time.time() - start_time
        print(f"  -> Model loaded in {init_time:.4f} seconds")
        print("  -> Ready for inference")
    
    def predict(
        self,
        context_data: np.ndarray,
        target_indices: Optional[list[int]] = None,
        return_scaled: bool = False
    ) -> np.ndarray:
        """
        Predict future values given context data.
        
        Args:
            context_data: Input context sequence
                         Shape: (context_points, n_features) or (1, context_points, n_features)
            target_indices: Indices of target variables to return
            return_scaled: If True, return scaled predictions
        
        Returns:
            Predictions array
            Shape: (target_points, n_targets) if target_indices specified
                   (target_points, n_variables) if target_indices is None
        """
        # Convert to numpy if needed (handle torch tensors if imported)
        try:
            import torch
            if isinstance(context_data, torch.Tensor):
                context_data = context_data.cpu().numpy()
        except ImportError:
            pass  # torch not available, assume it's already numpy
        
        # Ensure 3D: (batch, time, features)
        if context_data.ndim == 2:
            context_data = context_data[np.newaxis, :, :]
        
        batch_size, seq_len, n_features = context_data.shape
        
        # Validate input shape
        if self.context_points and seq_len != self.context_points:
            raise ValueError(
                f"Context length mismatch: got {seq_len}, expected {self.context_points}"
            )
        
        # Ensure features match model
        if self.n_variables and n_features > self.n_variables:
            context_data = context_data[:, :, :self.n_variables]
            n_features = self.n_variables
        elif self.n_variables and n_features < self.n_variables:
            raise ValueError(
                f"Feature count mismatch: got {n_features}, model expects {self.n_variables}"
            )
        
        # Scale if scaler is available and we want unscaled output
        if self.scaler is not None and not return_scaled:
            context_flat = context_data.reshape(-1, n_features)
            context_scaled = self.scaler.transform(context_flat)
            context_data = context_scaled.reshape(batch_size, seq_len, n_features)
        
        # Prepare inputs for ONNX
        if self.use_parallel:
            # Parallel mode: need both context and target
            if self.target_points is None:
                raise ValueError("target_points not set, cannot use parallel mode")
            
            # Create dummy target (repeat last context step)
            last_step = context_data[:, -1:, :]
            dummy_target = np.repeat(last_step, self.target_points, axis=1)
            
            # Run inference
            inputs = {
                self.input_names[0]: context_data.astype(np.float32),
                self.input_names[1]: dummy_target.astype(np.float32)
            }
        else:
            # Autoregressive mode: only context (if encoder is separate)
            # For now, fall back to parallel mode
            if self.target_points is None:
                raise ValueError("target_points not set")
            
            last_step = context_data[:, -1:, :]
            dummy_target = np.repeat(last_step, self.target_points, axis=1)
            
            inputs = {
                self.input_names[0]: context_data.astype(np.float32),
                self.input_names[1]: dummy_target.astype(np.float32)
            }
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        # ONNX Runtime returns numpy arrays (or SparseTensor in rare cases)
        # For our model, it's always a numpy array
        predictions = np.asarray(outputs[0])  # (batch, target_points, n_variables)
        
        # Extract target indices if specified
        if target_indices is not None:
            predictions = predictions[:, :, target_indices]
        
        # Remove batch dimension if single sample
        if batch_size == 1:
            predictions = predictions[0]
        
        # Inverse transform if scaler available and we want unscaled output
        if self.scaler is not None and not return_scaled:
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            
            if target_indices is not None:
                n_targets = len(target_indices)
                n_features = self.n_variables
                if n_features is None:
                    raise ValueError("n_variables not set, cannot perform inverse transform")
                # Ensure n_features is an integer for np.zeros
                n_features_int = int(n_features)
                pred_full = np.zeros((pred_flat.shape[0], n_features_int), dtype=pred_flat.dtype)
                for i, idx in enumerate(target_indices):
                    pred_full[:, idx] = pred_flat[:, i]
                pred_flat = pred_full
            
            pred_unscaled = self.scaler.inverse_transform(pred_flat)
            
            if target_indices is not None:
                pred_unscaled = pred_unscaled[:, target_indices]
            
            predictions = pred_unscaled.reshape(predictions.shape)
        
        return predictions


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX Runtime inference example')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint (for scaler)')
    parser.add_argument('--context-points', type=int, default=48, help='Context length')
    parser.add_argument('--n-features', type=int, default=440, help='Number of features')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = ONNXPredictor(
        onnx_model_path=args.model,
        checkpoint_path=args.checkpoint,
        use_parallel=True
    )
    
    # Create dummy context
    context_data = np.random.randn(args.context_points, args.n_features)
    
    # Predict
    print("\nRunning prediction...")
    start = time.time()
    predictions = predictor.predict(context_data, target_indices=[0, 1, 2])
    elapsed = time.time() - start
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Time: {elapsed:.4f}s")
    print(f"First 5 steps:\n{predictions[:5]}")


if __name__ == '__main__':
    main()

