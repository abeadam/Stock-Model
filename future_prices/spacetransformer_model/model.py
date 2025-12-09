"""
SpaceTimeFormer Model Architecture

This module implements the main SpaceTimeFormer model for multivariate time series forecasting.
The model uses spatiotemporal attention to learn patterns across both variables and time.
"""

import gc
import torch
import torch.nn as nn
from typing import Optional

from .embeddings import SpatiotemporalEmbedding
from .attention import SpatiotemporalTransformerBlock


class SpaceTimeFormer(nn.Module):
    """
    SpaceTimeFormer: A Transformer for multivariate time series forecasting.
    
    The model learns spatiotemporal patterns by:
    1. Flattening multivariate sequences into tokens (one per variable per timestep)
    2. Using spatiotemporal attention to learn relationships across variables and time
    3. Predicting future values using an encoder-decoder architecture
    
    Reference: "Long-Range Transformers for Dynamic Spatiotemporal Forecasting"
    https://github.com/QData/spacetimeformer
    """
    
    def __init__(
        self,
        n_variables: int,
        d_model: int = 128,
        n_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
        context_points: int = 96,
        target_points: int = 24,
        use_windowed_attn: bool = False,
        window_size: int = 50
    ):
        """
        Initialize SpaceTimeFormer model.
        
        Args:
            n_variables: Number of variables/features in the multivariate time series
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            enc_layers: Number of encoder layers
            dec_layers: Number of decoder layers
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length for positional encoding
            context_points: Number of timesteps in context (input sequence)
            target_points: Number of timesteps to predict (output sequence)
            use_windowed_attn: If True, use windowed attention for efficiency
            window_size: Window size for windowed attention
        """
        super().__init__()
        
        self.n_variables = n_variables
        self.d_model = d_model
        self.context_points = context_points
        self.target_points = target_points
        
        # Spatiotemporal embedding
        self.embedding = SpatiotemporalEmbedding(
            d_model=d_model,
            n_variables=n_variables,
            max_seq_length=max_seq_length
        )
        
        # Encoder: processes context sequence
        self.encoder = nn.ModuleList([
            SpatiotemporalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_windowed=use_windowed_attn,
                window_size=window_size
            )
            for _ in range(enc_layers)
        ])
        
        # Decoder: generates predictions
        # For simplicity, we use a similar architecture but can be extended
        # with cross-attention between encoder and decoder
        self.decoder = nn.ModuleList([
            SpatiotemporalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_windowed=use_windowed_attn,
                window_size=window_size
            )
            for _ in range(dec_layers)
        ])
        
        # Output projection: maps from d_model to 1 (single target value per variable)
        # For predicting PctChange_ToMaxHigh_5, we output a single value
        self.output_projection = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        x_context: torch.Tensor,
        x_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through SpaceTimeFormer.
        
        Args:
            x_context: Context sequence (batch_size, context_points, n_variables)
                      Historical data used for prediction
            x_target: Optional target sequence for teacher forcing during training
                     (batch_size, target_points, n_variables)
        
        Returns:
            Predictions (batch_size, target_points, 1)
            For PctChange_ToMaxHigh_5, we predict a single value per timestep
        """
        batch_size = x_context.shape[0]
        
        # Embed context sequence
        # Shape: (batch_size, context_points * n_variables, d_model)
        context_emb = self.embedding(x_context)
        
        # Pass through encoder layers
        encoder_output = context_emb
        for layer in self.encoder:
            encoder_output = layer(encoder_output)
            if encoder_output.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        
        # For prediction, we need to generate target sequence
        # Strategy: Use the last timestep's representation to initialize decoder
        # and generate predictions autoregressively or in parallel
        
        if x_target is not None:
            # Training mode: use target sequence with teacher forcing
            # Embed target sequence
            target_emb = self.embedding(x_target)
            
            # Decoder processes target sequence (with masking for autoregressive prediction)
            decoder_output = target_emb
            for layer in self.decoder:
                decoder_output = layer(decoder_output)
        else:
            # Inference mode: generate predictions
            # Use encoder output to initialize decoder
            # For simplicity, we'll use the last context representation
            # In a full implementation, this would be more sophisticated
            
            # Take the last timestep's representation for each variable
            # Reshape encoder output back to (batch_size, context_points, n_variables, d_model)
            context_len = self.context_points * self.n_variables
            encoder_reshaped = encoder_output[:, -self.n_variables:, :]  # Last variables
            
            # Repeat for target length (simple approach - can be improved)
            decoder_input = encoder_reshaped.unsqueeze(1).expand(-1, self.target_points, -1, -1)
            decoder_input = decoder_input.reshape(batch_size, self.target_points * self.n_variables, self.d_model)
            
            # Pass through decoder
            decoder_output = decoder_input
            for layer in self.decoder:
                decoder_output = layer(decoder_output)
        
        # Project to output dimension
        # We only need predictions for the target variable (PctChange_ToMaxHigh_5)
        # For simplicity, we'll take the first variable's representation
        # In practice, you might want to learn which variable corresponds to the target
        
        # Reshape decoder output: (batch_size, target_points * n_variables, d_model)
        # -> (batch_size, target_points, n_variables, d_model)
        # Determine actual sequence length processed in decoder
        # decoder_output shape: (batch_size, seq_len * n_variables, d_model)
        current_seq_len = decoder_output.shape[1] // self.n_variables
        
        decoder_reshaped = decoder_output.reshape(
            batch_size, current_seq_len, self.n_variables, self.d_model
        )
        
        # Project to output
        # Shape: (batch_size, seq_len, n_variables, 1)
        predictions = self.output_projection(decoder_reshaped)
        
        # Squeeze the last dimension
        # Shape: (batch_size, seq_len, n_variables)
        predictions = predictions.squeeze(-1)
        
        return predictions
    
    def predict(self, x_context: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions for given context.
        
        Args:
            x_context: Context sequence (batch_size, context_points, n_variables)
        
        Returns:
            Predictions (batch_size, target_points, 1)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_context)
        return predictions

