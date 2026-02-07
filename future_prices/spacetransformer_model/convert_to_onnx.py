"""
Convert PyTorch SpaceTimeFormer model to ONNX format for faster inference.

Usage:
    python convert_to_onnx.py --checkpoint checkpoints/spacetimeformer_best.pth --output model.onnx
"""

import sys
import os
import torch
import argparse
import time
import signal
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetransformer_model.model import SpaceTimeFormer
from spacetransformer_model.utils import get_device


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("ONNX export timed out")


def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    device: str = 'cpu',
    opset_version: int = 17,
    verbose: bool = False,
    debug: bool = False,
    timeout_seconds: Optional[int] = None,
    use_full_attention: bool = False
):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        device: Device to load model on ('cpu' or 'cuda')
        opset_version: ONNX opset version (17 is recommended for PyTorch 2.0+)
        verbose: If True, print detailed progress information
        debug: If True, enable debug mode with extra logging and error details
    """
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        torch.set_printoptions(threshold=10)  # Limit tensor printing in debug
        print("DEBUG MODE ENABLED")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Device: {device}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract architecture parameters
    required_params = [
        'n_variables', 'd_model', 'n_heads', 'enc_layers', 'dec_layers',
        'd_ff', 'context_points', 'target_points'
    ]
    missing_params = [p for p in required_params if p not in checkpoint]
    if missing_params:
        raise ValueError(f"Checkpoint missing parameters: {missing_params}")
    
    # Create model
    model = SpaceTimeFormer(
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
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    context_points = checkpoint['context_points']
    target_points = checkpoint['target_points']
    n_variables = checkpoint['n_variables']
    
    print(f"Model architecture:")
    print(f"  - Context points: {context_points}")
    print(f"  - Target points: {target_points}")
    print(f"  - Variables: {n_variables}")
    
    # Create dummy inputs for export
    # For parallel mode (all steps at once)
    dummy_context = torch.randn(1, context_points, n_variables).to(device)
    dummy_target = torch.randn(1, target_points, n_variables).to(device)
    
    print(f"\nExporting model to ONNX...")
    print(f"  - Output: {output_path}")
    print(f"  - Opset version: {opset_version}")
    if verbose or debug:
        print(f"  - Input shape: {dummy_context.shape}")
        print(f"  - Target shape: {dummy_target.shape}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Patch checkpoint to disable it during ONNX export
    # checkpoint is incompatible with ONNX tracing - it creates non-traceable graph breaks
    print("\n📤 Starting ONNX export...")
    print("   ⚠️  Temporarily disabling gradient checkpointing for ONNX compatibility...")
    
    from torch.utils.checkpoint import checkpoint as original_checkpoint
    
    def no_checkpoint(func, *args, **kwargs):
        """Replace checkpoint with direct call for ONNX export"""
        # Remove use_reentrant and other checkpoint-specific kwargs
        export_kwargs = {k: v for k, v in kwargs.items() if k != 'use_reentrant'}
        return func(*args, **export_kwargs)
    
    # Temporarily replace checkpoint in attention module
    import spacetransformer_model.attention as attention_module
    original_checkpoint_ref = attention_module.checkpoint
    attention_module.checkpoint = no_checkpoint
    
    # Suppress windowed attention progress prints during export
    # These prints can slow down the export significantly
    import sys
    import io
    original_stdout = sys.stdout
    class FilteredWriter:
        def __init__(self, original):
            self.original = original
            self.buffer = ""
        
        def write(self, text):
            # Filter out windowed attention progress messages
            if "Windowed attention:" not in text and "Completery" not in text and "Completey" not in text:
                self.original.write(text)
                self.original.flush()
        
        def flush(self):
            self.original.flush()
    
    # Redirect stdout to filter progress messages during export
    sys.stdout = FilteredWriter(original_stdout)
    
    # Export model for parallel prediction (forward method)
    try:
        if debug:
            print("\n[DEBUG] Starting ONNX export...")
            print(f"[DEBUG] Model device: {next(model.parameters()).device}")
            print(f"[DEBUG] Dummy context device: {dummy_context.device}")
            print(f"[DEBUG] Dummy target device: {dummy_target.device}")
        
        # Warn about CUDA export with windowed attention
        if device == 'cuda' and checkpoint.get('use_windowed_attn', True):
            print("\n⚠️  WARNING: Exporting with CUDA and windowed attention can be very slow.")
            print("   Consider using --device cpu for faster export (model will still work on GPU after export).")
            print("   This export may take 10-30 minutes or more...")
        
        # Set up timeout if specified
        if timeout_seconds:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            print(f"\n⏱️  Setting timeout to {timeout_seconds} seconds...")
        
        export_start = time.time()
        print("   🔄 Tracing model graph...")
        print("   ⚠️  WARNING: Windowed attention with chunking may cause ONNX export to hang.")
        print("   💡 If export hangs, consider using PyTorch inference instead (realtime_inference.py)")
        print("   💡 ONNX export traces the entire computation graph, which can be very slow")
        print("      for models with complex attention mechanisms and loops.")
        
        # Try exporting without dynamic axes first (faster, but less flexible)
        use_dynamic_axes = True
        if verbose or debug:
            print(f"   Using dynamic axes: {use_dynamic_axes}")
        
        # Add a progress callback if possible
        last_progress_time = time.time()
        def progress_callback():
            nonlocal last_progress_time
            current_time = time.time()
            if current_time - last_progress_time > 30:  # Print every 30 seconds
                elapsed = current_time - export_start
                print(f"   ⏳ Still tracing... ({elapsed:.0f}s elapsed)")
                last_progress_time = current_time
        
        try:
            # Note: torch.onnx.export doesn't have a built-in progress callback
            # We'll rely on timeout and user interruption
            torch.onnx.export(
                model,
                (dummy_context, dummy_target),
                output_path,
                input_names=['context', 'target'],
                output_names=['predictions'],
                dynamic_axes={
                    'context': {0: 'batch_size', 1: 'context_length'},
                    'target': {0: 'batch_size', 1: 'target_length'},
                    'predictions': {0: 'batch_size', 1: 'target_length'}
                } if use_dynamic_axes else None,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=verbose or debug,
                # Additional options to help with tracing
                training=torch.onnx.TrainingMode.EVAL,
                # Disable some optimizations that might cause issues
                strip_doc_string=False,
            )
        except Exception as e:
            print(f"\n   ⚠️  Export with dynamic axes failed: {e}")
            print("   🔄 Retrying without dynamic axes (faster but less flexible)...")
            # Retry without dynamic axes
            torch.onnx.export(
                model,
                (dummy_context, dummy_target),
                output_path,
                input_names=['context', 'target'],
                output_names=['predictions'],
                dynamic_axes=None,  # Fixed shapes only
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=verbose or debug,
                training=torch.onnx.TrainingMode.EVAL,
            )
            print("   ✅ Export succeeded without dynamic axes")
            print("   ⚠️  Note: Model will only accept fixed input shapes")
        
        # Cancel timeout
        if timeout_seconds:
            signal.alarm(0)
        
        export_time = time.time() - export_start
        if debug or verbose:
            print(f"[DEBUG] Export completed in {export_time:.2f}s")
        print(f"  ✅ Successfully exported to {output_path} in {export_time:.1f}s")
        
        # Verify ONNX model
        try:
            import onnx  # type: ignore  # Optional dependency for validation
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✅ ONNX model validation passed")
            
            # Print model info
            print(f"\nModel info:")
            print(f"  - Inputs: {len(onnx_model.graph.input)}")
            print(f"  - Outputs: {len(onnx_model.graph.output)}")
            print(f"  - File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        except ImportError:
            print("  ⚠️  onnx package not installed, skipping validation")
        except Exception as e:
            print(f"  ⚠️  Validation warning: {e}")
            
    except TimeoutError as e:
        # Cancel timeout
        if timeout_seconds:
            signal.alarm(0)
        print(f"\n  ❌ Export timed out after {timeout_seconds} seconds")
        print(f"  💡 Suggestions:")
        print(f"     - Try using --device cpu (often faster for export)")
        print(f"     - Increase timeout with --timeout <seconds>")
        print(f"     - The model may be too complex for ONNX export")
        raise
    except Exception as e:
        # Cancel timeout
        if timeout_seconds:
            signal.alarm(0)
        print(f"  ❌ Export failed: {e}")
        print(f"  💡 Try using --device cpu for export (model will still work on GPU after export)")
        raise
    finally:
        # Always restore original checkpoint function and stdout
        attention_module.checkpoint = original_checkpoint_ref
        sys.stdout = original_stdout
    
    # Also export encoder separately (for autoregressive mode)
    encoder_output_path = output_path.replace('.onnx', '_encoder.onnx')
    print(f"\nExporting encoder separately...")
    try:
        # Create a wrapper for encoder
        # ONNX export requires a module with forward() method, not just a method call
        class EncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x_context):
                return self.model.encode(x_context)
        
        encoder_wrapper = EncoderWrapper(model)
        # Checkpoint is already patched from main export, but ensure it's still patched
        torch.onnx.export(
            encoder_wrapper,
            (dummy_context,),  # Wrap in tuple - torch.onnx.export expects tuple of inputs
            encoder_output_path,
            input_names=['context'],
            output_names=['encoder_output'],
            dynamic_axes={
                'context': {0: 'batch_size', 1: 'context_length'},
                'encoder_output': {0: 'batch_size', 1: 'seq_length'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        print(f"  ✅ Encoder exported to {encoder_output_path}")
    except Exception as e:
        print(f"  ⚠️  Encoder export failed (optional): {e}")
    
    # Also export decoder separately (for autoregressive mode)
    decoder_output_path = output_path.replace('.onnx', '_decoder.onnx')
    print(f"\nExporting decoder separately...")
    try:
        # Create a wrapper for decoder
        # ONNX export requires a module with forward() method, not just a method call
        # Decoder needs both target input and encoder output
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x_target, encoder_output):
                return self.model.decode(x_target, encoder_output)
        
        # Create dummy encoder output for decoder export
        with torch.no_grad():
            dummy_encoder_output = model.encode(dummy_context)
        
        decoder_wrapper = DecoderWrapper(model)
        torch.onnx.export(
            decoder_wrapper,
            (dummy_target, dummy_encoder_output),  # Both inputs as tuple
            decoder_output_path,
            input_names=['target', 'encoder_output'],
            output_names=['predictions'],
            dynamic_axes={
                'target': {0: 'batch_size', 1: 'target_length'},
                'encoder_output': {0: 'batch_size', 1: 'seq_length'},
                'predictions': {0: 'batch_size', 1: 'target_length'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        print(f"  ✅ Decoder exported to {decoder_output_path}")
    except Exception as e:
        print(f"  ⚠️  Decoder export failed (optional): {e}")
    
    print(f"\n✅ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Install ONNX Runtime: pip install onnxruntime onnxruntime-gpu")
    print(f"  2. Use onnx_inference.py for faster inference")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.onnx',
        help='Output ONNX file path (default: model.onnx)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for conversion (default: auto-detect)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (default: 17)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with extra logging and error details'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for export (default: no timeout)'
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Device setup
    device_name = args.device
    if device_name is None:
        device_name = str(get_device())
    
    convert_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=device_name,
        opset_version=args.opset,
        verbose=args.verbose,
        debug=args.debug,
        timeout_seconds=args.timeout
    )


if __name__ == '__main__':
    main()

