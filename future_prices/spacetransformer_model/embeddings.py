"""
Spatiotemporal Embeddings for SpaceTimeFormer

This module implements embeddings that combine temporal (time) and spatial (variable) 
information. Each token represents a single variable at a specific timestep.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SpatiotemporalEmbedding(nn.Module):
    """
    Spatiotemporal embedding that combines:
    - Variable embeddings (spatial dimension - which feature/variable)
    - Temporal embeddings (time dimension - which timestep)
    - Value embeddings (the actual feature value)
    
    This creates tokens where each token represents: variable_i at time_t with value_v
    """
    
    def __init__(self, d_model: int, n_variables: int, max_seq_length: int = 1000):
        """
        Args:
            d_model: Model dimension (embedding size)
            n_variables: Number of variables/features in the multivariate time series
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.d_model = d_model
        self.n_variables = n_variables
        self.max_seq_length = max_seq_length
        
        # Variable embedding: learns which variable/feature this token represents
        # Each variable gets its own learnable embedding
        self.variable_embedding = nn.Embedding(n_variables, d_model)
        
        # Temporal embedding: learns the position in time
        # Uses sinusoidal positional encoding similar to standard transformers
        self.temporal_embedding = TemporalPositionalEncoding(d_model, max_seq_length)
        
        # Value projection: projects the actual feature value to d_model
        # This allows the model to incorporate the actual data values
        self.value_projection = nn.Linear(1, d_model)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, variable_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create spatiotemporal embeddings from input data.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_variables)
               Each element is a feature value
            variable_ids: Optional tensor of shape (batch_size, seq_length, n_variables)
                         with variable indices. If None, uses sequential indices.
        
        Returns:
            Embedded tensor of shape (batch_size, seq_length * n_variables, d_model)
            The sequence is flattened: [var0_t0, var1_t0, ..., varN_t0, var0_t1, ...]
        """
        batch_size, seq_length, n_variables = x.shape
        
        # Flatten to create tokens: each token is one variable at one timestep
        # Shape: (batch_size, seq_length * n_variables, 1)
        x_flat = x.reshape(batch_size, seq_length * n_variables, 1)
        
        # Project values to d_model
        # Shape: (batch_size, seq_length * n_variables, d_model)
        value_emb = self.value_projection(x_flat)
        
        # Create variable IDs if not provided
        if variable_ids is None:
            # Create variable indices: [0, 1, 2, ..., n_variables-1] repeated for each timestep
            variable_ids = torch.arange(n_variables, device=x.device).repeat(seq_length)
            variable_ids = variable_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            variable_ids = variable_ids.reshape(batch_size, -1)
        
        # Get variable embeddings
        # Shape: (batch_size, seq_length * n_variables, d_model)
        var_emb = self.variable_embedding(variable_ids)
        
        # Create temporal positions
        # For each timestep, repeat the position for all variables
        # [0, 0, ..., 0 (n_vars times), 1, 1, ..., 1 (n_vars times), ...]
        temporal_positions = torch.arange(seq_length, device=x.device).repeat_interleave(n_variables)
        temporal_positions = temporal_positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get temporal embeddings
        # Shape: (batch_size, seq_length * n_variables, d_model)
        temp_emb = self.temporal_embedding(temporal_positions)
        
        # Combine all embeddings: value + variable + temporal
        # Shape: (batch_size, seq_length * n_variables, d_model)
        embeddings = value_emb + var_emb + temp_emb
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal dimension.
    Similar to standard transformer positional encoding but applied to temporal positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get positional encodings for given positions.
        
        Args:
            positions: Tensor of shape (batch_size, seq_length) with position indices
        
        Returns:
            Positional encodings of shape (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = positions.shape
        
        # Get encodings for each position
        # positions: (batch_size, seq_length) -> (batch_size, seq_length, d_model)
        # Access the registered buffer as a tensor
        # Type cast to help linter understand this is a Tensor
        pe_tensor = torch.as_tensor(self.pe)  # Shape: (1, max_len, d_model)
        max_pos = int(pe_tensor.shape[1])  # Convert to int for clamp
        
        # Clamp positions to valid range
        positions_clamped = torch.clamp(positions, min=0, max=max_pos - 1)
        
        # Get positional encodings by indexing
        # pe_tensor shape: (1, max_len, d_model)
        # Index along dimension 1 (the sequence length dimension)
        pos_encodings = pe_tensor[0, positions_clamped]  # Shape: (batch_size, seq_length, d_model)
        
        return pos_encodings

