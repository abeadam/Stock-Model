"""
SpaceTimeFormer Model Package

A Transformer-based model for multivariate time series forecasting that learns
spatiotemporal patterns across both variables and time dimensions.
"""

from .model import SpaceTimeFormer
from .embeddings import SpatiotemporalEmbedding, TemporalPositionalEncoding
from .attention import (
    MultiHeadSpatiotemporalAttention,
    WindowedAttention,
    SpatiotemporalTransformerBlock
)
from .train import TimeSeriesDataset, train_model, train_epoch, validate
from .data_loader import load_data

__all__ = [
    'SpaceTimeFormer',
    'SpatiotemporalEmbedding',
    'TemporalPositionalEncoding',
    'MultiHeadSpatiotemporalAttention',
    'WindowedAttention',
    'SpatiotemporalTransformerBlock',
    'TimeSeriesDataset',
    'train_model',
    'train_epoch',
    'validate',
    'load_data'
]

