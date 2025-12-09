"""
Attention Mechanisms for SpaceTimeFormer

This module implements various attention mechanisms including:
- Spatiotemporal attention (full attention across all variables and timesteps)
- Windowed attention (efficient local attention)
- Efficient attention variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
from torch.utils.checkpoint import checkpoint
from typing import Optional


class MultiHeadSpatiotemporalAttention(nn.Module):
    """
    Multi-head spatiotemporal attention.
    
    This attention mechanism allows each token (variable at timestep) to attend to
    all other tokens across both spatial (variables) and temporal (time) dimensions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V and split into heads
        # Shape: (batch_size, seq_len, n_heads, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # Shape: (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # Shape: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        
        return output


def _check_and_clear_gpu_memory(
    device: torch.device,
    total_memory: Optional[int],
    memory_threshold: float,
    i: int,
    seq_len: int,
    iteration: int,
    print_progress: bool = True
) -> None:
    """
    Check GPU memory usage and clear cache if it exceeds threshold.
    
    Args:
        device: CUDA device
        total_memory: Total GPU memory in bytes
        memory_threshold: Memory usage threshold (0.0-1.0) to trigger cache clearing
        i: Current iteration index
        seq_len: Total sequence length
        iteration: Iteration number for progress printing
        print_progress: Whether to print progress messages (default: True)
    """
    should_clear_cache = False
    if device.type == 'cuda' and total_memory is not None:
        reserved = torch.cuda.memory_reserved(device)
        memory_usage = reserved / total_memory  # Use reserved as it's what matters for OOM
        
        # Clear cache if memory usage exceeds threshold
        if memory_usage > memory_threshold:
            should_clear_cache = True
            if print_progress:
                progress_pct = ((i + 1) / seq_len) * 100
                print(f"      Windowed attention: [{i + 1}/{seq_len}] ({progress_pct:.1f}%) | "
                      f"GPU Memory: {memory_usage*100:.1f}% ({reserved/1e9:.2f}GB/{total_memory/1e9:.2f}GB) - Clearing cache", end='\r')
    
    # Print progress periodically (every 10 iterations) regardless of memory
    if print_progress and iteration % 10 == 0 and device.type == 'cuda':
        progress_pct = ((i + 1) / seq_len) * 100
        if not should_clear_cache:  # Only print if we didn't already print above
            if total_memory is not None:
                reserved = torch.cuda.memory_reserved(device)
                memory_usage = reserved / total_memory
                print(f"      Windowed attention: [{i + 1}/{seq_len}] ({progress_pct:.1f}%) | "
                      f"GPU Memory: {memory_usage*100:.1f}%", end='\r')
            else:
                print(f"      Windowed attention: [{i + 1}/{seq_len}] ({progress_pct:.1f}%)", end='\r')
    
    # Only synchronize and clear cache when memory is actually getting full
    if should_clear_cache:
        torch.cuda.empty_cache()
        gc.collect()  # Force Python garbage collection
        torch.cuda.synchronize()  # Ensure operations complete before clearing
        
        # Debug: Print all CUDA tensors to identify memory leaks
        print(f"\n      DEBUG: CUDA Tensors after cache clear (iteration {i+1}):")
        tensor_count = 0
        total_memory_mb = 0.0
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_count += 1
                obj_id = id(obj)
                obj_size_bytes = obj.element_size() * obj.nelement()
                obj_size_mb = obj_size_bytes / (1024 * 1024)
                total_memory_mb += obj_size_mb
                
                # Try to identify tensor type
                tensor_type = "Unknown"
                shape = tuple(obj.shape)
                if shape == (2, 48, 434):
                    tensor_type = "Input Context"
                elif shape == (2, 24):
                    tensor_type = "Target Values"
                elif shape == (2, 24, 1):
                    tensor_type = "Target Expanded/Predictions"
                elif shape == (20832,):
                    tensor_type = "Flattened Context"
                elif len(shape) == 3 and shape[2] == 64:
                    tensor_type = f"Embedding/Attention Output (seq_len={shape[1]})"
                elif len(shape) == 4 and shape[3] == 16:  # d_k = 64/4 = 16
                    tensor_type = f"Attention Q/K/V (heads={shape[1]}, seq={shape[2]})"
                
                print(f"        [{tensor_count}] {type(obj).__name__} | "
                      f"Shape: {shape} | "
                      f"Type: {tensor_type} | "
                      f"Dtype: {obj.dtype} | "
                      f"Size: {obj_size_mb:.2f} MB | "
                      f"Device: {obj.device} | "
                      f"ID: {hex(obj_id)}")
        if tensor_count == 0:
            print("        No CUDA tensors found")
        else:
            print(f"        Total: {tensor_count} tensors, {total_memory_mb:.2f} MB")
        print()  # Add blank line for readability


class WindowedAttention(nn.Module):
    """
    Windowed attention for efficient local attention patterns.
    
    Instead of attending to all tokens, each token only attends to tokens within
    a local window. This reduces computational complexity from O(n²) to O(n*w)
    where w is the window size.
    
    Uses chunked attention to avoid creating large mask matrices.
    """
    
    def __init__(self, d_model: int, n_heads: int, window_size: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Size of attention window (number of tokens to attend to)
            dropout: Dropout probability
        """
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = (d_model // n_heads) ** -0.5
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _compute_chunk(self, Q_chunk: torch.Tensor, K: torch.Tensor, V: torch.Tensor, start_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute attention for a chunk of the sequence.
        This function is designed to be used with torch.utils.checkpoint to save memory.
        """
        # start_idx is passed as a tensor from checkpoint, convert to int
        start_idx_val = int(start_idx.item())
        
        # Q_chunk shape: (batch_size, n_heads, chunk_len, d_k)
        chunk_len = Q_chunk.shape[2]
        seq_len = K.shape[2]
        half_window = self.window_size // 2
        
        chunk_outputs = []
        
        # Get device info for memory management
        device = Q_chunk.device
        memory_threshold = 0.974
        total_memory = None
        if device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(device).total_memory
        
        for i in range(chunk_len):
            # Global index in the sequence
            global_i = start_idx_val + i
            
            # Define window boundaries for this position
            start = max(0, global_i - half_window)
            end = min(seq_len, global_i + half_window + 1)
            
            # Extract Q for current position (relative to chunk)
            q_i = Q_chunk[:, :, i:i+1, :].contiguous()
            
            # Extract K and V for window (from full K, V)
            k_window = K[:, :, start:end, :].contiguous()
            v_window = V[:, :, start:end, :].contiguous()
            
            # Compute attention scores
            k_window_t = k_window.transpose(-2, -1)
            
            # Check memory before first matmul
            _check_and_clear_gpu_memory(
                device=device,
                total_memory=total_memory,
                memory_threshold=memory_threshold,
                i=global_i,
                seq_len=seq_len,
                iteration=global_i + 1
            )
            
            scores = torch.matmul(q_i, k_window_t) / self.scale
            del k_window_t
            
            attn_weights = torch.softmax(scores, dim=-1)
            del scores
            attn_weights = self.dropout(attn_weights)
            
            # Check memory before second matmul (no progress print)
            _check_and_clear_gpu_memory(
                device=device,
                total_memory=total_memory,
                memory_threshold=memory_threshold,
                i=global_i,
                seq_len=seq_len,
                iteration=global_i + 1,
                print_progress=False
            )
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v_window)
            
            # Append directly to list
            chunk_outputs.append(attn_output)
            
            # Cleanup
            del q_i, k_window, v_window, attn_weights, attn_output
        
        # Concatenate chunk outputs
        return torch.cat(chunk_outputs, dim=2)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, 
                value: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply windowed attention using chunked computation.
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, d_model = query.shape
        
        # Project to Q, K, V
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Clear original Q, K, V projections inputs to free memory
        del query, key, value
        
        # Use Gradient Checkpointing with chunking to save memory
        # Divide sequence into chunks and checkpoint each chunk
        # This prevents storing the entire autograd graph for the loop
        chunk_size = 512 # Adjust based on available memory
        output_list = []
        
        for start_idx in range(0, seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Extract chunk of Q - this slice requires grad, so it works with checkpoint
            Q_chunk = Q[:, :, start_idx:end_idx, :]
            
            # Use checkpoint for this chunk
            # Note: start_idx is passed as a tensor so it can be passed to checkpoint
            chunk_out = checkpoint(self._compute_chunk, Q_chunk, K, V, torch.tensor(start_idx), use_reentrant=False)
            output_list.append(chunk_out)
        
        # Print completion message
        if Q.device.type == 'cuda':
            print(f"      Windowed attention: [{seq_len}/{seq_len}] (100.0%) - Complete")
        
        # Concatenate all results along the sequence dimension
        output = torch.cat(output_list, dim=2)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(output)
        
        return output


class SpatiotemporalTransformerBlock(nn.Module):
    """
    Transformer block with spatiotemporal attention.
    
    Combines multi-head attention with feed-forward network and residual connections.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, use_windowed: bool = False, 
                 window_size: Optional[int] = None):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_windowed: If True, use windowed attention instead of full attention
            window_size: Window size for windowed attention (if use_windowed=True)
        """
        super().__init__()
        
        # Attention mechanism
        if use_windowed and window_size is not None:
            self.attention = WindowedAttention(d_model, n_heads, window_size, dropout)
        else:
            self.attention = MultiHeadSpatiotemporalAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

