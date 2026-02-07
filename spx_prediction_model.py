"""
SPX Prediction Model using PyTorch LSTM

Install dependencies:
    pip install torch pandas numpy scikit-learn
    OR
    pip install -r requirements.txt
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
from typing import Tuple, List, Any
import warnings
import gc  # Garbage collection for memory management
import logging

# Suppress PyTorch warnings
warnings.filterwarnings('ignore')
# Suppress PyTorch allocator config deprecation warning
logging.getLogger('torch').setLevel(logging.ERROR)

# Handle deprecated PYTORCH_CUDA_ALLOC_CONF environment variable
if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ and 'PYTORCH_ALLOC_CONF' not in os.environ:
    # Migrate to new environment variable name
    os.environ['PYTORCH_ALLOC_CONF'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    # Optionally remove old variable (commented out to avoid breaking external configs)
    # del os.environ['PYTORCH_CUDA_ALLOC_CONF']

# Note: LD_LIBRARY_PATH must be set BEFORE importing torch for CUDA to work
# If CUDA is not available, set it before running: export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Set random seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

class StockDataset(Dataset):
    """Dataset class for time series stock data
    
    Memory-efficient: Generates sequences on-the-fly instead of pre-creating them.
    This dramatically reduces memory usage by only creating sequences when needed.
    Ensures all data points in a sequence are from the same trading day.
    """
    def __init__(self, features_scaled, targets_scaled, day_ids, sequence_length, target_index=None):
        # Store normalized features and targets (not sequences)
        # sequences will be created on-the-fly in __getitem__
        self.features_scaled = features_scaled  # numpy array: (total_samples, num_features)
        self.targets_scaled = targets_scaled    # numpy array: (total_samples, 2) or (total_samples, 1) if target_index specified
        self.day_ids = day_ids  # numpy array: (total_samples,) - tracks which day each row belongs to
        self.sequence_length = sequence_length
        self.target_index = target_index  # None for both targets, 0 for High only, 1 for Low only
        
        # Calculate valid sequence indices - only sequences where all points are from the same day
        self.valid_indices = self._calculate_valid_indices()
        self.num_sequences = len(self.valid_indices)
    
    def _calculate_valid_indices(self):
        """Calculate indices where sequences can be created from the same trading day"""
        valid_indices = []
        for i in range(len(self.features_scaled) - self.sequence_length - 1):
            # Check if all sequence points and target are from the same day
            seq_start_idx = i
            seq_end_idx = i + self.sequence_length
            target_idx = i + self.sequence_length + 1
            
            # Get day IDs for sequence and target
            seq_days = self.day_ids[seq_start_idx:seq_end_idx]
            target_day = self.day_ids[target_idx]
            
            # All sequence points must be from the same day, and target must be from same day
            if len(set(seq_days)) == 1 and seq_days[0] == target_day:
                valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get the actual index from valid_indices
        actual_idx = self.valid_indices[idx]
        
        # Generate sequence on-the-fly from the normalized features
        # This avoids storing all sequences in memory
        # Sequence: features from actual_idx to actual_idx + sequence_length - 1
        seq = self.features_scaled[actual_idx:actual_idx + self.sequence_length]  # (sequence_length, num_features)
        # Target: High and Low from the NEXT iteration (actual_idx + sequence_length + 1)
        target_full = self.targets_scaled[actual_idx + self.sequence_length + 1]  # (2,) or (1,)
        
        # If target_index is specified, extract only that target (for single-target models)
        if self.target_index is not None:
            target = target_full[self.target_index]  # Scalar
        else:
            target = target_full  # (2,)
        
        # Runtime validation: Ensure all sequence points and target are from the same day
        # (This is redundant since valid_indices already enforces this, but provides extra safety)
        seq_days = self.day_ids[actual_idx:actual_idx + self.sequence_length]
        target_day = self.day_ids[actual_idx + self.sequence_length + 1]
        assert len(set(seq_days)) == 1, f"Sequence contains multiple days: {set(seq_days)}"
        assert seq_days[0] == target_day, f"Target day {target_day} doesn't match sequence day {seq_days[0]}"
        
        # Convert to tensor
        # For single targets, return scalar tensor (will be batched to shape (batch_size,))
        # For dual targets, return shape (2,) (will be batched to shape (batch_size, 2))
        if self.target_index is not None:
            target_tensor = torch.FloatTensor([target])  # Shape (1,) -> batched to (batch_size, 1)
        else:
            target_tensor = torch.FloatTensor(target)  # Shape (2,) -> batched to (batch_size, 2)
        return torch.FloatTensor(seq), target_tensor


class LSTMPredictor(nn.Module):
    """LSTM model for predicting % change in SPX High and Low
    
    Note: This model only processes sequences from the same trading day.
    The StockDataset class ensures all input sequences contain data from a single day.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output: % change in High and Low
        
        # Initialize final layer to match target scale (percentage changes ~0.006% std)
        # Targets are raw percentage changes with std ~0.006
        # Use moderate gain (0.1) to allow model to learn variance while staying in correct scale
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)  # Moderate gain to allow variance learning
        nn.init.zeros_(self.fc3.bias)  # Start at zero (target mean is ~0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Note: All sequences in x are guaranteed to be from the same trading day
        # (enforced by StockDataset._calculate_valid_indices)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with optional projection for dimension matching"""
    def __init__(self, in_features, out_features, dropout=0.1, use_projection=False):
        super(ResidualBlock, self).__init__()
        self.use_projection = use_projection
        
        # Main transformation
        self.norm1 = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer if dimensions don't match
        if use_projection:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None
    
    def forward(self, x):
        # Store residual
        residual = x
        
        # Main path
        out = self.norm1(x)
        out = self.linear(out)
        out = self.norm2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        # Add residual (with projection if needed)
        if self.use_projection and self.projection is not None:
            residual = self.projection(residual)
        
        out = out + residual
        return out


class TransformerPredictor(nn.Module):
    """Transformer model for predicting % change in SPX High and Low
    
    Uses self-attention to maintain information from all previous timesteps,
    allowing the model to attend to any part of the sequence when making predictions.
    
    Note: This model only processes sequences from the same trading day.
    The StockDataset class ensures all input sequences contain data from a single day.
    The self-attention mechanism operates only within the same-day sequence.
    """
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.3, max_seq_length=100):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers with residual connections
        # First layer: d_model -> 128 (with projection since dimensions differ)
        self.res_block1 = ResidualBlock(d_model, 128, dropout=dropout, use_projection=True)
        
        # Second layer: 128 -> 64 (with projection)
        self.res_block2 = ResidualBlock(128, 64, dropout=dropout, use_projection=True)
        
        # Third layer: 64 -> 32 (with projection)
        self.res_block3 = ResidualBlock(64, 32, dropout=dropout, use_projection=True)
        
        # Fourth layer: 32 -> 16 (with projection)
        self.res_block4 = ResidualBlock(32, 16, dropout=dropout, use_projection=True)
        
        # Fifth layer: 16 -> 8 (with projection)
        self.res_block5 = ResidualBlock(16, 8, dropout=dropout, use_projection=True)
        
        # Intermediate layer: 8 -> 4 (with residual for better representation)
        self.res_block6 = ResidualBlock(8, 4, dropout=dropout, use_projection=True)
        
        # Final output layer: 4 -> 2 (no residual - final prediction layer)
        self.fc_final = nn.Linear(4, 2)  # Output: % change in High and Low
        
        # Initialize final layer to match target scale (percentage changes ~0.006% std)
        # Targets are raw percentage changes with std ~0.006
        # Use moderate gain (0.1) to allow model to learn variance while staying in correct scale
        nn.init.xavier_uniform_(self.fc_final.weight, gain=0.1)  # Moderate gain to allow variance learning
        nn.init.zeros_(self.fc_final.bias)  # Start at zero (target mean is ~0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Note: All sequences in x are guaranteed to be from the same trading day
        # (enforced by StockDataset._calculate_valid_indices)
        # The self-attention mechanism operates only within the same-day sequence
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder (self-attention across all timesteps in the sequence)
        # This allows the model to attend to ALL previous timesteps within the same day
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Use the last timestep's output (which has attended to all previous timesteps in the same day)
        last_output = transformer_out[:, -1, :]  # (batch_size, d_model)
        
        # Fully connected layers with residual connections
        # Residual blocks (with dimension matching via projection)
        out = self.res_block1(last_output)  # d_model -> 128
        out = self.res_block2(out)          # 128 -> 64
        out = self.res_block3(out)          # 64 -> 32
        out = self.res_block4(out)          # 32 -> 16
        out = self.res_block5(out)          # 16 -> 8
        out = self.res_block6(out)          # 8 -> 4
        
        # Final output layer (no residual - final prediction layer)
        out = self.fc_final(out)            # 4 -> 2
        
        return out


class LSTMPredictorSingle(nn.Module):
    """LSTM model for predicting % change in SPX High OR Low (single target)
    
    Note: This model only processes sequences from the same trading day.
    The StockDataset class ensures all input sequences contain data from a single day.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMPredictorSingle, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: % change in High OR Low (single value)
        
        # Initialize final layer to match target scale (percentage changes ~0.006% std)
        # Targets are raw percentage changes with std ~0.006
        # Use moderate gain (0.1) to allow model to learn variance while staying in correct scale
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)  # Moderate gain to allow variance learning
        nn.init.zeros_(self.fc3.bias)  # Start at zero (target mean is ~0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Note: All sequences in x are guaranteed to be from the same trading day
        # (enforced by StockDataset._calculate_valid_indices)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)  # Shape: (batch_size, 1)
        
        return out.squeeze(-1)  # Remove last dimension: (batch_size,)


class TransformerPredictorSingle(nn.Module):
    """Transformer model for predicting % change in SPX High OR Low (single target)
    
    Uses self-attention to maintain information from all previous timesteps,
    allowing the model to attend to any part of the sequence when making predictions.
    
    Note: This model only processes sequences from the same trading day.
    The StockDataset class ensures all input sequences contain data from a single day.
    The self-attention mechanism operates only within the same-day sequence.
    """
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.3, max_seq_length=100):
        super(TransformerPredictorSingle, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers with residual connections
        # First layer: d_model -> 128 (with projection since dimensions differ)
        self.res_block1 = ResidualBlock(d_model, 128, dropout=dropout, use_projection=True)
        
        # Second layer: 128 -> 64 (with projection)
        self.res_block2 = ResidualBlock(128, 64, dropout=dropout, use_projection=True)
        
        # Third layer: 64 -> 32 (with projection)
        self.res_block3 = ResidualBlock(64, 32, dropout=dropout, use_projection=True)
        
        # Fourth layer: 32 -> 16 (with projection)
        self.res_block4 = ResidualBlock(32, 16, dropout=dropout, use_projection=True)
        
        # Fifth layer: 16 -> 8 (with projection)
        self.res_block5 = ResidualBlock(16, 8, dropout=dropout, use_projection=True)
        
        # Intermediate layer: 8 -> 4 (with residual for better representation)
        self.res_block6 = ResidualBlock(8, 4, dropout=dropout, use_projection=True)

        self.res_block7 = ResidualBlock(4, 2, dropout=dropout, use_projection=True) 
        self.fc_final = nn.Linear(2, 1)  # Output: % change in High OR Low (single value)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Note: All sequences in x are guaranteed to be from the same trading day
        # (enforced by StockDataset._calculate_valid_indices)
        # The self-attention mechanism operates only within the same-day sequence
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder (self-attention across all timesteps in the sequence)
        # This allows the model to attend to ALL previous timesteps within the same day
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Use the last timestep's output (which has attended to all previous timesteps in the same day)
        last_output = transformer_out[:, -1, :]  # (batch_size, d_model)
        
        # Fully connected layers with residual connections
        # Residual blocks (with dimension matching via projection)
        out = self.res_block1(last_output)  # d_model -> 128
        out = self.res_block2(out)          # 128 -> 64
        out = self.res_block3(out)          # 64 -> 32
        out = self.res_block4(out)          # 32 -> 16
        out = self.res_block5(out)          # 16 -> 8
        out = self.res_block6(out)          # 8 -> 4
        out = self.res_block7(out)          # 4 -> 2
        
        # Final output layer (no residual - final prediction layer)
        out = self.fc_final(out)            # 4 -> 1
        return out.squeeze(-1)  # Remove last dimension: (batch_size,)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer to understand sequence order"""
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix - use float32
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        seq_len = x.size(1)
        # Add positional encoding (pe is a registered buffer tensor, shape (1, max_len, d_model))
        x = x + self.pe[:, :seq_len, :]  # type: ignore
        return self.dropout(x)


def load_data(data_dir: str) -> pd.DataFrame:
    """Load and combine SPX, SPY, and VIX data"""
    print("Loading data files...")
    
    # Get all date prefixes
    spx_files = glob.glob(os.path.join(data_dir, "*_SPX.txt"))
    date_prefixes = set()
    for file in spx_files:
        basename = os.path.basename(file)
        date_prefix = basename.replace("_SPX.txt", "")
        date_prefixes.add(date_prefix)
    
    all_data = []
    
    for date_prefix in sorted(date_prefixes):
        spx_file = os.path.join(data_dir, f"{date_prefix}_SPX.txt")
        spy_file = os.path.join(data_dir, f"{date_prefix}_SPY.txt")
        vix_file = os.path.join(data_dir, f"{date_prefix}_VIX.txt")
        
        # Check if all files exist
        if not all(os.path.exists(f) for f in [spx_file, spy_file, vix_file]):
            continue
        
        try:
            # Load data
            spx_df = pd.read_csv(spx_file)
            spy_df = pd.read_csv(spy_file)
            vix_df = pd.read_csv(vix_file)
            
            # Merge on Date (timestamp)
            # SPX: Remove Volume
            merged = spx_df[['Date', 'Open', 'High', 'Low', 'Close']].merge(
                spy_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']],
                on='Date',
                suffixes=('_SPX', '_SPY')
            )
            # VIX: Remove Volume
            merged = merged.merge(
                vix_df[['Date', 'Open', 'High', 'Low', 'Close']],
                on='Date',
                suffixes=('', '_VIX')
            )
            
            # Rename VIX columns
            merged = merged.rename(columns={
                'Open': 'Open_VIX',
                'High': 'High_VIX',
                'Low': 'Low_VIX',
                'Close': 'Close_VIX'
            })
            
            # Add day identifier to track which trading day this data belongs to
            merged['Day_ID'] = date_prefix
            
            all_data.append(merged)
        except Exception as e:
            print(f"Error loading {date_prefix}: {e}")
            continue
    if not all_data:
        raise ValueError("No data files could be loaded!")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(combined_df)} total rows from {len(all_data)} days")
    
    # Add time-of-day column (time from start of trading day)
    combined_df = add_time_of_day(combined_df)
    gc.collect()  # Free memory after processing
    
    # Add technical indicators
    combined_df = add_technical_indicators(combined_df)
    gc.collect()  # Free memory after technical indicators
    
    return combined_df


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day column representing seconds from start of trading day (9:30 AM ET)"""
    df = df.copy()
    
    # Convert timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['Date'], unit='s')
    
    # Extract date and time components
    df['Date_only'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    
    # Calculate seconds from midnight
    df['Seconds_from_midnight'] = (
        df['DateTime'].dt.hour * 3600 + 
        df['DateTime'].dt.minute * 60 + 
        df['DateTime'].dt.second
    )
    
    # Trading day starts at 9:30 AM ET = 34200 seconds from midnight
    # Trading day ends at 4:00 PM ET = 57600 seconds from midnight
    TRADING_START_SECONDS = 34200  # 9:30 AM
    TRADING_END_SECONDS = 57600    # 4:00 PM
    
    # Calculate time from start of trading day (normalized to 0-1)
    df['Time_of_Day'] = (df['Seconds_from_midnight'] - TRADING_START_SECONDS) / (TRADING_END_SECONDS - TRADING_START_SECONDS)
    
    # Clip to valid trading hours (0-1 range)
    df['Time_of_Day'] = df['Time_of_Day'].clip(0, 1)
    
    # Drop intermediate columns
    df = df.drop(columns=['DateTime', 'Date_only', 'Time', 'Seconds_from_midnight'])
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for SPX, SPY, and VIX
    Calculates indicators per trading day to avoid cross-day contamination"""
    df = df.copy()
    
    # Sort by Date to ensure proper calculation
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Check if Day_ID exists - if not, we can't ensure per-day calculation
    if 'Day_ID' not in df.columns:
        print("Warning: Day_ID not found. Calculating indicators across all data (may include cross-day contamination)")
        day_groups = [df]
    else:
        # Group by Day_ID to calculate indicators per trading day
        day_groups = [group for _, group in df.groupby('Day_ID')]
        print(f"Calculating technical indicators per trading day ({len(day_groups)} days)")
    
    # Process each day separately
    processed_days = []
    for day_df in day_groups:
        # For each symbol (SPX, SPY, VIX)
        for symbol in ['SPX', 'SPY', 'VIX']:
            close_col = f'Close_{symbol}'
            high_col = f'High_{symbol}'
            low_col = f'Low_{symbol}'
            open_col = f'Open_{symbol}'
            
            # Moving Averages
            day_df[f'MA_5_{symbol}'] = day_df[close_col].rolling(window=5, min_periods=1).mean()
            day_df[f'MA_10_{symbol}'] = day_df[close_col].rolling(window=10, min_periods=1).mean()
            day_df[f'MA_20_{symbol}'] = day_df[close_col].rolling(window=20, min_periods=1).mean()
            
            # Exponential Moving Averages
            day_df[f'EMA_12_{symbol}'] = day_df[close_col].ewm(span=12, adjust=False).mean()
            day_df[f'EMA_26_{symbol}'] = day_df[close_col].ewm(span=26, adjust=False).mean()
            
            # RSI (Relative Strength Index)
            delta = day_df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
            day_df[f'RSI_{symbol}'] = 100 - (100 / (1 + rs))
            
            # MACD
            day_df[f'MACD_{symbol}'] = day_df[f'EMA_12_{symbol}'] - day_df[f'EMA_26_{symbol}']
            day_df[f'MACD_Signal_{symbol}'] = day_df[f'MACD_{symbol}'].ewm(span=9, adjust=False).mean()
            day_df[f'MACD_Hist_{symbol}'] = day_df[f'MACD_{symbol}'] - day_df[f'MACD_Signal_{symbol}']
            
            # Bollinger Bands
            ma_20 = day_df[close_col].rolling(window=20, min_periods=1).mean()
            std_20 = day_df[close_col].rolling(window=20, min_periods=1).std()
            day_df[f'BB_Upper_{symbol}'] = ma_20 + (std_20 * 2)
            day_df[f'BB_Lower_{symbol}'] = ma_20 - (std_20 * 2)
            day_df[f'BB_Width_{symbol}'] = day_df[f'BB_Upper_{symbol}'] - day_df[f'BB_Lower_{symbol}']
            day_df[f'BB_Position_{symbol}'] = (day_df[close_col] - day_df[f'BB_Lower_{symbol}']) / (day_df[f'BB_Width_{symbol}'] + 1e-10)
            
            # Bollinger Band distance indicators
            # (BB_Lower - Low) / Low: Shows relative distance of low from lower band
            day_df[f'BB_Lower_Low_Dist_{symbol}'] = (day_df[f'BB_Lower_{symbol}'] - day_df[low_col]) / (day_df[low_col] + 1e-10)
            # (BB_Upper - High) / High: Shows relative distance of high from upper band
            day_df[f'BB_Upper_High_Dist_{symbol}'] = (day_df[f'BB_Upper_{symbol}'] - day_df[high_col]) / (day_df[high_col] + 1e-10)
            
            # Stochastic Oscillator
            low_14 = day_df[low_col].rolling(window=14, min_periods=1).min()
            high_14 = day_df[high_col].rolling(window=14, min_periods=1).max()
            day_df[f'Stoch_K_{symbol}'] = 100 * ((day_df[close_col] - low_14) / (high_14 - low_14 + 1e-10))
            day_df[f'Stoch_D_{symbol}'] = day_df[f'Stoch_K_{symbol}'].rolling(window=3, min_periods=1).mean()
            
            # Average True Range (ATR)
            high_low = day_df[high_col] - day_df[low_col]
            high_close = abs(day_df[high_col] - day_df[close_col].shift(1))
            low_close = abs(day_df[low_col] - day_df[close_col].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            day_df[f'ATR_{symbol}'] = tr.rolling(window=14, min_periods=1).mean()
            
            # Price Change and Returns
            day_df[f'Price_Change_{symbol}'] = day_df[close_col].diff()
            day_df[f'Price_Change_Pct_{symbol}'] = day_df[close_col].pct_change()
            
            # Volatility (rolling standard deviation of returns)
            day_df[f'Volatility_{symbol}'] = day_df[f'Price_Change_Pct_{symbol}'].rolling(window=20, min_periods=1).std()
            
            # High-Low Spread
            day_df[f'HL_Spread_{symbol}'] = day_df[high_col] - day_df[low_col]
            day_df[f'HL_Spread_Pct_{symbol}'] = day_df[f'HL_Spread_{symbol}'] / (day_df[close_col] + 1e-10)
        
        # Fill NaN values with forward fill then backward fill for this day
        day_df = day_df.ffill().bfill()
        processed_days.append(day_df)
    
    # Concatenate all processed days back together
    df = pd.concat(processed_days, ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Replace any remaining NaN with 0
    df = df.fillna(0)
    
    return df


def prepare_data(data: pd.DataFrame, sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, Any]:
    """Prepare normalized features and targets for on-the-fly sequence generation
    
    Returns normalized features and targets (not sequences) to save memory.
    Sequences will be created on-the-fly during training.
    """
    print(f"Preparing data for on-the-fly sequence generation (sequence_length={sequence_length})...")
    
    # Select features: SPX, SPY, and VIX OHLC data (Volume removed from SPX and VIX)
    # Plus technical indicators and time-of-day
    base_columns = [
        'Open_SPX', 'High_SPX', 'Low_SPX', 'Close_SPX',
        'Open_SPY', 'High_SPY', 'Low_SPY', 'Close_SPY', 'Volume_SPY',
        'Open_VIX', 'High_VIX', 'Low_VIX', 'Close_VIX',
        'Time_of_Day'
    ]
    
    # Technical indicator columns
    tech_indicators = []
    for symbol in ['SPX', 'SPY', 'VIX']:
        tech_indicators.extend([
            f'MA_5_{symbol}', f'MA_10_{symbol}', f'MA_20_{symbol}',
            f'EMA_12_{symbol}', f'EMA_26_{symbol}',
            f'RSI_{symbol}',
            f'MACD_{symbol}', f'MACD_Signal_{symbol}', f'MACD_Hist_{symbol}',
            f'BB_Upper_{symbol}', f'BB_Lower_{symbol}', f'BB_Width_{symbol}', f'BB_Position_{symbol}',
            f'BB_Lower_Low_Dist_{symbol}', f'BB_Upper_High_Dist_{symbol}',
            f'Stoch_K_{symbol}', f'Stoch_D_{symbol}',
            f'ATR_{symbol}',
            f'Price_Change_{symbol}', f'Price_Change_Pct_{symbol}',
            f'Volatility_{symbol}',
            f'HL_Spread_{symbol}', f'HL_Spread_Pct_{symbol}'
        ])
    
    # Combine all feature columns
    feature_columns = base_columns + tech_indicators
    
    # Filter to only columns that exist in the dataframe
    feature_columns = [col for col in feature_columns if col in data.columns]
    
    print(f"Using {len(feature_columns)} features for prediction")
    
    # Extract features - convert to float32 to halve memory usage
    features = data[feature_columns].values.astype(np.float32)
    
    # Extract High and Low values for percentage change calculation
    high_values = data['High_SPX'].values.astype(np.float32)
    low_values = data['Low_SPX'].values.astype(np.float32)
    
    # Calculate percentage change from previous step
    # pct_change[i] = (value[i] - value[i-1]) / value[i-1] * 100
    # For the first row, we'll use 0 (no previous value)
    high_pct_change = np.zeros_like(high_values)
    low_pct_change = np.zeros_like(low_values)
    
    # Calculate percentage change (skip first row as it has no previous value)
    for i in range(1, len(high_values)):
        if high_values[i-1] != 0:  # Avoid division by zero
            high_pct_change[i] = ((high_values[i] - high_values[i-1]) / high_values[i-1]) * 100.0
        if low_values[i-1] != 0:  # Avoid division by zero
            low_pct_change[i] = ((low_values[i] - low_values[i-1]) / low_values[i-1]) * 100.0
    
    # Stack into targets array: [High_pct_change, Low_pct_change]
    targets = np.column_stack([high_pct_change, low_pct_change]).astype(np.float32)
    
    # Extract day IDs to ensure sequences only use same-day data
    if 'Day_ID' not in data.columns:
        raise ValueError("Day_ID column missing! Cannot ensure same-day sequences.")
    day_ids = np.array(data['Day_ID'].values)  # Convert to numpy array for type consistency
    
    # Normalize features
    scaler_features = StandardScaler()
    features_scaled = scaler_features.fit_transform(features).astype(np.float32)
    
    # DON'T normalize targets - percentage changes are already in a reasonable range
    # Normalizing targets causes the model to predict the mean (0) in normalized space,
    # which maps to ~0% in actual space, causing model collapse
    # Use raw percentage changes as targets
    targets_scaled = targets.astype(np.float32)  # Use raw percentage changes
    
    # Create a dummy scaler for inverse_transform compatibility
    # This scaler will just return the values as-is (identity transform)
    class IdentityScaler:
        def __init__(self):
            self.mean_ = np.array([0.0, 0.0])
            self.scale_ = np.array([1.0, 1.0])
        def inverse_transform(self, X):
            return X  # Identity transform - return as-is
    
    scaler_targets = IdentityScaler()
    
    print(f"Data type: float32 (halves memory usage compared to float64)")
    print(f"Note: Sequences will only use data from the same trading day")
    print(f"Note: Predicting % change in High/Low from the NEXT iteration (future prediction)")
    print(f"Note: Target is percentage change from previous step (e.g., +1.5% means 1.5% increase)")
    print(f"⚠️ IMPORTANT: Targets are NOT normalized (using raw percentage changes)")
    print(f"   Target range - High: [{targets_scaled[:, 0].min():.4f}%, {targets_scaled[:, 0].max():.4f}%], mean={targets_scaled[:, 0].mean():.4f}%, std={targets_scaled[:, 0].std():.4f}%")
    print(f"   Target range - Low:  [{targets_scaled[:, 1].min():.4f}%, {targets_scaled[:, 1].max():.4f}%], mean={targets_scaled[:, 1].mean():.4f}%, std={targets_scaled[:, 1].std():.4f}%")
    gc.collect()  # Free memory after data preparation
    return features_scaled, targets_scaled, day_ids, scaler_features, scaler_targets


class VarianceEncouragingLoss(nn.Module):
    """Loss function that encourages variance in predictions and prevents mean collapse
    
    Combines MSE with multiple regularization terms:
    1. Variance matching: encourages predictions to have similar variance to targets
    2. Diversity loss: encourages predictions to be spread out (not all the same)
    3. Mean penalty: discourages predictions from collapsing to zero (target mean)
    """
    def __init__(self, mse_weight=1.0, variance_weight=0.1, diversity_weight=0.05, mean_penalty_weight=0.02):
        super(VarianceEncouragingLoss, self).__init__()
        self.mse_weight = mse_weight
        self.variance_weight = variance_weight
        self.diversity_weight = diversity_weight
        self.mean_penalty_weight = mean_penalty_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Standard MSE loss
        mse = self.mse_loss(predictions, targets)
        
        # Variance matching term: penalize if prediction variance is too different from target variance
        # Calculate variance for each output dimension
        pred_var_high = torch.var(predictions[:, 0])
        pred_var_low = torch.var(predictions[:, 1])
        target_var_high = torch.var(targets[:, 0])
        target_var_low = torch.var(targets[:, 1])
        
        # Variance loss: squared difference between prediction and target variance
        # This encourages the model to predict values with similar variance to targets
        var_loss_high = (pred_var_high - target_var_high) ** 2
        var_loss_low = (pred_var_low - target_var_low) ** 2
        variance_loss = (var_loss_high + var_loss_low) / 2.0
        
        # Diversity loss: encourage predictions to be different from each other
        # Penalize if predictions are too similar (low pairwise distance)
        # This prevents all predictions from collapsing to the same value
        batch_size = predictions.size(0)
        if batch_size > 1:
            # Calculate pairwise squared distances between predictions
            pred_high_diff = predictions[:, 0].unsqueeze(1) - predictions[:, 0].unsqueeze(0)
            pred_low_diff = predictions[:, 1].unsqueeze(1) - predictions[:, 1].unsqueeze(0)
            # Mean squared pairwise distance (higher = more diverse)
            # We want to penalize when this is LOW (predictions are similar)
            # So we use the negative (or inverse) to create a penalty
            mean_pairwise_dist_high = torch.mean(pred_high_diff ** 2)
            mean_pairwise_dist_low = torch.mean(pred_low_diff ** 2)
            # Diversity loss: penalize when mean pairwise distance is too small
            # Use a softer penalty: 1 / (1 + k * distance) gives moderate penalty when distance is small
            # This is less aggressive than exponential and won't dominate the loss
            k = 100.0  # Scaling factor - adjust based on typical distance values
            diversity_high = 1.0 / (1.0 + k * mean_pairwise_dist_high)  # Softer penalty
            diversity_low = 1.0 / (1.0 + k * mean_pairwise_dist_low)
            diversity_loss = (diversity_high + diversity_low) / 2.0
        else:
            diversity_loss = torch.tensor(0.0, device=predictions.device)
        
        # Mean penalty: discourage predictions from collapsing to zero (target mean)
        # Penalize if predictions are too close to zero
        # This encourages the model to make non-zero predictions
        mean_penalty_high = torch.mean(predictions[:, 0] ** 2)  # Penalize if predictions are close to 0
        mean_penalty_low = torch.mean(predictions[:, 1] ** 2)
        # But we want predictions to be near zero on average, so we only penalize if they're TOO close
        # Use a threshold: only penalize if variance is very low (indicating collapse)
        threshold = 1e-6  # Very small threshold
        mean_penalty = torch.max(torch.tensor(0.0, device=predictions.device), 
                                 threshold - (mean_penalty_high + mean_penalty_low) / 2.0)
        
        # Combined loss
        total_loss = (self.mse_weight * mse + 
                     self.variance_weight * variance_loss +
                     self.diversity_weight * diversity_loss +
                     self.mean_penalty_weight * mean_penalty)
        
        return total_loss

def train_model(model, train_loader, val_loader, test_loader, device, num_epochs=50, learning_rate=0.001, early_stopping_patience=10, warmup_epochs=5):
    """Train the model with gradient clipping and early stopping"""
    # Detect if this is a single-target model by checking output shape
    model.eval()
    with torch.no_grad():
        sample_seq, sample_target = next(iter(train_loader))
        sample_seq = sample_seq.to(device)
        sample_output = model(sample_seq)
        is_single_target = sample_output.dim() == 1 or (sample_output.dim() == 2 and sample_output.shape[1] == 1)
    model.train()
    
    # Use appropriate loss function
    if is_single_target:
        # For single-target models, use MSELoss (simpler and more appropriate)
        criterion = nn.MSELoss()
        print("Using MSELoss for single-target model")
    else:
        # For dual-target models, use variance-encouraging loss
        criterion = VarianceEncouragingLoss(
            mse_weight=1.0, 
            variance_weight=0.2,  # Increased to emphasize variance matching
            diversity_weight=0.01,  # Reduced - was too aggressive and dominating loss
            mean_penalty_weight=0.01  # Reduced - less emphasis on mean penalty
        )
        print("Using VarianceEncouragingLoss for dual-target model")
    
    # MSE criterion for reporting metrics (pure MSE for evaluation)
    mse_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.999))  # Increased weight decay for stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    
    # Adaptive learning rate: increase LR when model gets stuck (loss not changing + small gradients)
    stuck_epochs = 0
    STUCK_THRESHOLD = 1  # Number of consecutive epochs with no progress before boosting LR (aggressive: boost after 1 epoch)
    LR_BOOST_FACTOR = 10.0  # Always multiply LR by 10 (order of magnitude boost)
    MAX_LR = learning_rate * 100  # Maximum LR cap (increased to allow 10x boosts)
    lr_boosted = False  # Track if we've boosted LR (to preserve it during warmup)
    
    # Gradient clipping threshold - prevent exploding gradients while allowing natural gradient flow
    # Use higher clipping for single model (dual-target), standard for separate models (single-target)
    if is_single_target:
        MAX_GRAD_NORM = 1.0  # Standard clipping for single-target models (separate models)
    else:
        MAX_GRAD_NORM = 10.0  # Higher clipping for dual-target models (single model)
    
    # Gradient accumulation: accumulate gradients over multiple batches before updating
    # This smooths out gradient updates and can help with stability
    GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate over 4 batches before updating
    
    train_losses_huber = []  # Actually MSE now
    train_losses_mse = []
    val_losses_huber = []  # Actually MSE now
    val_losses_mse = []
    test_losses_huber = []  # Actually MSE now
    test_losses_mse = []
    best_val_loss_huber = float('inf')  # Actually stores MSE
    patience_counter = 0
    best_epoch = 0
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA cache cleared. Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\nStarting training...")
    if isinstance(criterion, VarianceEncouragingLoss):
        print(f"Loss function: VarianceEncouragingLoss with regularization:")
        print(f"  - MSE weight: {criterion.mse_weight} (accuracy)")
        print(f"  - Variance weight: {criterion.variance_weight} (match target variance)")
        print(f"  - Diversity weight: {criterion.diversity_weight} (encourage diverse predictions)")
        print(f"  - Mean penalty weight: {criterion.mean_penalty_weight} (prevent collapse to zero)")
    else:
        print(f"Loss function: MSELoss (single-target model)")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print(f"Learning rate warmup: {warmup_epochs} epochs")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps (smooths gradient updates)")
    print(f"Gradient clipping: max_norm={MAX_GRAD_NORM}")
    
    # Learning rate warmup
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return learning_rate * (epoch + 1) / warmup_epochs
        return learning_rate
    
    for epoch in range(num_epochs):
        # Learning rate warmup (only if we haven't boosted LR)
        if epoch < warmup_epochs and not lr_boosted:
            current_lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        # Training phase
        model.train()
        train_loss_huber = 0.0
        train_loss_mse = 0.0
        batch_count = 0
        epoch_grad_norms = []  # Track gradient norms throughout epoch
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            
            # Diagnostic: Check first batch to see if model outputs vary
            if epoch == 0 and batch_idx == 0:
                output_std = outputs.std().item()
                target_std = targets.std().item()
                print(f"\n⚠️ First batch diagnostic:")
                print(f"  Model output: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}, std={output_std:.6f}")
                print(f"  Target:        min={targets.min().item():.6f}, max={targets.max().item():.6f}, mean={targets.mean().item():.6f}, std={target_std:.6f}")
                if output_std < 1e-5:
                    print(f"  ⚠️ WARNING: Model outputs are constant! Check initialization.")
                elif output_std < target_std * 0.1:
                    print(f"  ⚠️ WARNING: Model outputs have very low variance (std={output_std:.6f} vs target std={target_std:.6f})")
                    print(f"  💡 Model may collapse to constant predictions. Consider:")
                    print(f"     - Increasing final layer initialization gain")
                    print(f"     - Reducing learning rate")
                    print(f"     - Checking for vanishing gradients")
                else:
                    print(f"  ✓ Output variance looks reasonable (std ratio: {output_std/target_std:.2f})")
            
            # Scale loss by accumulation steps
            loss = criterion(outputs, targets) / GRADIENT_ACCUMULATION_STEPS
            
            # Track all loss components for diagnostics (first batch only)
            if epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    mse_only = mse_criterion(outputs, targets).item()
                    
                    # Only show detailed breakdown for dual-target models with VarianceEncouragingLoss
                    if not is_single_target and isinstance(criterion, VarianceEncouragingLoss):
                        pred_var_high = torch.var(outputs[:, 0]).item()
                        pred_var_low = torch.var(outputs[:, 1]).item()
                        target_var_high = torch.var(targets[:, 0]).item()
                        target_var_low = torch.var(targets[:, 1]).item()
                        var_loss_high = (pred_var_high - target_var_high) ** 2
                        var_loss_low = (pred_var_low - target_var_low) ** 2
                        variance_loss = (var_loss_high + var_loss_low) / 2.0
                        
                        # Calculate diversity loss
                        batch_size = outputs.size(0)
                        if batch_size > 1:
                            pred_high_diff = outputs[:, 0].unsqueeze(1) - outputs[:, 0].unsqueeze(0)
                            pred_low_diff = outputs[:, 1].unsqueeze(1) - outputs[:, 1].unsqueeze(0)
                            mean_pairwise_dist_high = torch.mean(pred_high_diff ** 2).item()
                            mean_pairwise_dist_low = torch.mean(pred_low_diff ** 2).item()
                            diversity_high = np.exp(-10.0 * mean_pairwise_dist_high)
                            diversity_low = np.exp(-10.0 * mean_pairwise_dist_low)
                            diversity_loss = (diversity_high + diversity_low) / 2.0
                        else:
                            diversity_loss = 0.0
                        
                        # Calculate mean penalty
                        mean_penalty_high = torch.mean(outputs[:, 0] ** 2).item()
                        mean_penalty_low = torch.mean(outputs[:, 1] ** 2).item()
                        threshold = 1e-6
                        mean_penalty = max(0.0, threshold - (mean_penalty_high + mean_penalty_low) / 2.0)
                        
                        print(f"  Loss breakdown:")
                        print(f"    MSE: {mse_only:.6f} (weight: {criterion.mse_weight})")
                        print(f"    Variance loss: {variance_loss:.6f} (weight: {criterion.variance_weight})")
                        print(f"    Diversity loss: {diversity_loss:.6f} (weight: {criterion.diversity_weight})")
                        print(f"    Mean penalty: {mean_penalty:.6f} (weight: {criterion.mean_penalty_weight})")
                        print(f"  Prediction variance - High: {pred_var_high:.6f} (target: {target_var_high:.6f})")
                        print(f"  Prediction variance - Low:  {pred_var_low:.6f} (target: {target_var_low:.6f})")
                    else:
                        # Simple diagnostic for single-target models
                        pred_var = torch.var(outputs).item() if outputs.dim() > 0 else 0.0
                        target_var = torch.var(targets).item() if targets.dim() > 0 else 0.0
                        print(f"  First batch - MSE: {mse_only:.6f}")
                        print(f"  Prediction variance: {pred_var:.6f} (target: {target_var:.6f})")
            
            loss.backward()
            
            train_loss_huber += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale for logging
            batch_count += 1
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping to prevent explosion - use more aggressive threshold
                # clip_grad_norm_ returns the norm BEFORE clipping, then clips
                grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                epoch_grad_norms.append(grad_norm_before.item())  # Track for epoch summary
                
                # Track gradient norm for diagnostics (first accumulation step only)
                if epoch == 0 and batch_idx < GRADIENT_ACCUMULATION_STEPS:
                    # After clipping, the actual norm should be <= MAX_GRAD_NORM
                    # Calculate actual norm after clipping to verify
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    actual_norm_after = total_norm ** (1. / 2)
                    
                    print(f"  Gradient norm: {grad_norm_before.item():.2f} -> {actual_norm_after:.2f} (clipped to {MAX_GRAD_NORM})", end="")
                    if grad_norm_before.item() < 1e-6:
                        print(" ⚠️ VANISHING GRADIENTS!")
                    elif grad_norm_before.item() > 100.0:  # Only warn on truly large gradients (>100)
                        print(f" ⚠️ LARGE GRADIENTS ({grad_norm_before.item():.1f}) - clipping applied")
                    else:
                        print(" ✓")
                
                # Store weights before update to check if they actually change
                weight_sample_before = None
                if epoch >= 5:  # Start checking after a few epochs
                    weight_sample_before = next(model.parameters()).data.clone()
                
                optimizer.step()
                
                # Check if weights actually changed (diagnostic for stuck training)
                if epoch >= 5 and weight_sample_before is not None:
                    weight_sample_after = next(model.parameters()).data
                    weight_change = (weight_sample_after - weight_sample_before).abs().max().item()
                    if weight_change < 1e-8:
                        print(f"  ⚠️ WARNING: Weights not updating! Max weight change: {weight_change:.2e}")
                        # Check gradient norm
                        if grad_norm_before.item() < 1e-6:
                            print(f"     → Gradients are vanishing (norm: {grad_norm_before.item():.2e})")
                        else:
                            print(f"     → Gradients exist (norm: {grad_norm_before.item():.2e}) but weights not changing!")
                            print(f"     → Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                
                optimizer.zero_grad()  # Zero gradients after update
            # Also calculate MSE for reporting
            train_loss_mse += mse_criterion(outputs, targets).item()
        
        train_loss_huber /= batch_count
        train_loss_mse /= batch_count
        train_losses_huber.append(train_loss_huber)
        train_losses_mse.append(train_loss_mse)
        
        # Clear CUDA cache after training phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Validation phase - calculate both Huber and MSE
        model.eval()
        val_loss_huber = 0.0
        val_loss_mse = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                val_loss_huber += criterion(outputs, targets).item()
                val_loss_mse += mse_criterion(outputs, targets).item()
        
        val_loss_huber /= len(val_loader)
        val_loss_mse /= len(val_loader)
        val_losses_huber.append(val_loss_huber)
        val_losses_mse.append(val_loss_mse)
        
        # Calculate test loss for monitoring (both Huber and MSE)
        model.eval()
        test_loss_huber = 0.0
        test_loss_mse = 0.0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                test_loss_huber += criterion(outputs, targets).item()
                test_loss_mse += mse_criterion(outputs, targets).item()
        
        test_loss_huber /= len(test_loader)
        test_loss_mse /= len(test_loader)
        test_losses_huber.append(test_loss_huber)
        test_losses_mse.append(test_loss_mse)
        
        # Clear CUDA cache after validation phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use MSE loss for scheduler (since we're using MSE for training now)
        scheduler.step(val_loss_mse)
        
        # Save best model based on validation MSE (matches training objective)
        if val_loss_mse < best_val_loss_huber:  # Reusing variable name for compatibility
            best_val_loss_huber = val_loss_mse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation HuberLoss: {best_val_loss_huber:.6f} at epoch {best_epoch}")
            break
        
        # Print both HuberLoss and MSE for all sets after every epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - HuberLoss: {train_loss_huber:.6f}, MSE: {train_loss_mse:.6f}")
        print(f"  Val   - HuberLoss: {val_loss_huber:.6f}, MSE: {val_loss_mse:.6f}")
        print(f"  Test  - HuberLoss: {test_loss_huber:.6f}, MSE: {test_loss_mse:.6f}")
        print(f"  LR: {current_lr:.2e}, Patience: {patience_counter}/{early_stopping_patience}")
        
        # Diagnostic: Check if loss is decreasing
        train_loss_change = None
        train_mse_change = None
        if epoch > 0:
            train_loss_change = train_losses_huber[-2] - train_loss_huber
            train_mse_change = train_losses_mse[-2] - train_loss_mse  # Check MSE directly
            val_loss_change = val_losses_huber[-2] - val_loss_huber
            if train_loss_change < 0:
                print(f"  ⚠️ Training loss increased by {abs(train_loss_change):.6f}")
            elif train_loss_change < 1e-6:
                print(f"  ⚠️ Training loss barely changed ({train_loss_change:.6f}) - model may not be learning")
            if val_loss_change < 0:
                print(f"  ⚠️ Validation loss increased by {abs(val_loss_change):.6f}")
        
        # Diagnostic: Check gradient norms to detect vanishing gradients
        is_stuck = False
        avg_grad_norm = None
        if epoch_grad_norms:
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
            max_grad_norm = max(epoch_grad_norms)
            min_grad_norm = min(epoch_grad_norms)
            if avg_grad_norm < 1e-6:
                print(f"  ⚠️ VANISHING GRADIENTS! Avg norm: {avg_grad_norm:.2e} (min: {min_grad_norm:.2e}, max: {max_grad_norm:.2e})")
                is_stuck = True
            elif avg_grad_norm < 1e-4:
                print(f"  ⚠️ Very small gradients! Avg norm: {avg_grad_norm:.2e} (min: {min_grad_norm:.2e}, max: {max_grad_norm:.2e})")
            elif train_loss_change is not None and train_loss_change < 1e-6 and avg_grad_norm < 0.01:
                print(f"  ⚠️ Loss not changing AND gradients are small! Avg grad norm: {avg_grad_norm:.4f}")
                is_stuck = True
        
        # Track stuck epochs: More robust detection
        # Consider stuck if: MSE not changing significantly (within 1e-4) OR (small gradients AND loss not changing)
        # Also check if MSE is oscillating around the same value (variance of last few MSEs is small)
        if epoch > 0:
            mse_stuck = train_mse_change is not None and abs(train_mse_change) < 1e-4  # More lenient threshold
            loss_stuck = train_loss_change is not None and abs(train_loss_change) < 1e-4
            grad_stuck = avg_grad_norm is not None and avg_grad_norm < 0.01
            
            # Check if MSE is oscillating around the same value (last 3 epochs have low variance)
            mse_oscillating = False
            if len(train_losses_mse) >= 3:
                recent_mse = train_losses_mse[-3:]
                mse_variance = np.var(recent_mse)
                mse_mean = np.mean(recent_mse)
                # If variance is very small relative to mean, MSE is essentially constant
                if mse_mean > 0 and mse_variance / mse_mean < 1e-4:
                    mse_oscillating = True
            
            if mse_stuck or mse_oscillating or (loss_stuck and grad_stuck) or is_stuck:
                stuck_epochs += 1
                if stuck_epochs == 1:
                    grad_norm_str = f"{avg_grad_norm:.4f}" if avg_grad_norm is not None else "N/A"
                    reason = []
                    if mse_stuck:
                        reason.append(f"MSE change: {train_mse_change:.2e}")
                    if mse_oscillating:
                        reason.append("MSE oscillating")
                    if grad_stuck:
                        reason.append(f"Grad norm: {grad_norm_str}")
                    print(f"  📍 Model appears stuck ({', '.join(reason)})")
                elif stuck_epochs < STUCK_THRESHOLD:
                    print(f"  📍 Still stuck ({stuck_epochs}/{STUCK_THRESHOLD} epochs) - will boost LR at {STUCK_THRESHOLD}")
            else:
                # Model is making progress - reset stuck counter
                if stuck_epochs > 0:
                    print(f"  ✓ Model making progress again (reset stuck counter)")
                stuck_epochs = 0
        
        # Adaptive learning rate boost when stuck
        # Always boost by 10x (order of magnitude) when stuck
        if stuck_epochs >= STUCK_THRESHOLD:
            current_lr = optimizer.param_groups[0]['lr']
            # Always use 10x boost
            new_lr = min(current_lr * LR_BOOST_FACTOR, MAX_LR)
            if new_lr > current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                lr_boosted = True  # Mark that we've boosted LR (preserve it during warmup)
                warmup_note = " (during warmup)" if epoch < warmup_epochs else ""
                print(f"  🚀 BOOSTING LEARNING RATE: {current_lr:.2e} → {new_lr:.2e} (10x boost, model stuck for {stuck_epochs} epochs{warmup_note})")
                stuck_epochs = 0  # Reset counter after boosting
        
        # Periodic cleanup
        if (epoch + 1) % 10 == 0:
            gc.collect()  # Periodic garbage collection during training
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if (epoch + 1) % 20 == 0:  # Print memory info every 20 epochs
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    print(f"  GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    gc.collect()  # Free memory after training
    # Clear CUDA cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nTraining completed. Best validation MSE: {best_val_loss_huber:.6f} at epoch {best_epoch}")
    
    return (train_losses_huber, train_losses_mse), (val_losses_huber, val_losses_mse), (test_losses_huber, test_losses_mse)


def evaluate_model(model, data_loader, scaler_targets, device, dataset_name="Dataset"):
    """Evaluate the model and calculate performance metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Clear CUDA cache after evaluation batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Ensure predictions are 2D (add dimension if single target)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    # Detect if single-target model
    is_single_target = predictions.shape[1] == 1
    
    # Diagnostic: Show raw predictions BEFORE inverse transform
    print(f"\n{dataset_name} Set - Raw Model Outputs (Before Inverse Transform):")
    if is_single_target:
        print(f"  Predictions: min={predictions[:, 0].min():.8f}, max={predictions[:, 0].max():.8f}, mean={predictions[:, 0].mean():.8f}, std={predictions[:, 0].std():.8f}")
        print(f"  Targets:      min={targets[:, 0].min():.8f}, max={targets[:, 0].max():.8f}, mean={targets[:, 0].mean():.8f}, std={targets[:, 0].std():.8f}")
    else:
        print(f"  Predictions - High: min={predictions[:, 0].min():.8f}, max={predictions[:, 0].max():.8f}, mean={predictions[:, 0].mean():.8f}, std={predictions[:, 0].std():.8f}")
        print(f"  Predictions - Low:  min={predictions[:, 1].min():.8f}, max={predictions[:, 1].max():.8f}, mean={predictions[:, 1].mean():.8f}, std={predictions[:, 1].std():.8f}")
        print(f"  Targets      - High: min={targets[:, 0].min():.8f}, max={targets[:, 0].max():.8f}, mean={targets[:, 0].mean():.8f}, std={targets[:, 0].std():.8f}")
        print(f"  Targets      - Low:  min={targets[:, 1].min():.8f}, max={targets[:, 1].max():.8f}, mean={targets[:, 1].mean():.8f}, std={targets[:, 1].std():.8f}")
    
    # Clear intermediate lists to free memory
    del all_predictions, all_targets
    gc.collect()
    
    # Inverse transform to get actual percentage changes (not absolute values)
    predictions_pct = scaler_targets.inverse_transform(predictions)  # Percentage changes
    targets_pct = scaler_targets.inverse_transform(targets)  # Percentage changes
    
    # Diagnostic: Check for NaN or Inf values
    if np.any(np.isnan(predictions_pct)) or np.any(np.isinf(predictions_pct)):
        print(f"WARNING: {dataset_name} Set contains NaN or Inf in predictions!")
    if np.any(np.isnan(targets_pct)) or np.any(np.isinf(targets_pct)):
        print(f"WARNING: {dataset_name} Set contains NaN or Inf in targets!")
    
    # Diagnostic: Check prediction ranges (percentage changes)
    print(f"\n{dataset_name} Set - Prediction Statistics (Percentage Changes):")
    if is_single_target:
        print(f"  Predictions: min={predictions_pct[:, 0].min():.4f}%, max={predictions_pct[:, 0].max():.4f}%, mean={predictions_pct[:, 0].mean():.4f}%")
        print(f"  Targets:      min={targets_pct[:, 0].min():.4f}%, max={targets_pct[:, 0].max():.4f}%, mean={targets_pct[:, 0].mean():.4f}%")
    else:
        print(f"  Predictions - High % Change: min={predictions_pct[:, 0].min():.4f}%, max={predictions_pct[:, 0].max():.4f}%, mean={predictions_pct[:, 0].mean():.4f}%")
        print(f"  Targets      - High % Change: min={targets_pct[:, 0].min():.4f}%, max={targets_pct[:, 0].max():.4f}%, mean={targets_pct[:, 0].mean():.4f}%")
        print(f"  Predictions - Low % Change:  min={predictions_pct[:, 1].min():.4f}%, max={predictions_pct[:, 1].max():.4f}%, mean={predictions_pct[:, 1].mean():.4f}%")
        print(f"  Targets      - Low % Change:  min={targets_pct[:, 1].min():.4f}%, max={targets_pct[:, 1].max():.4f}%, mean={targets_pct[:, 1].mean():.4f}%")
    
    # Calculate baseline (mean prediction) for comparison (on percentage changes)
    # Baseline predicts the mean of targets (which should be close to 0 for percentage changes)
    target_mean = targets_pct[:, 0].mean()
    baseline = np.full_like(targets_pct[:, 0], target_mean)
    baseline_mse = np.mean((targets_pct[:, 0] - baseline) ** 2)  # This is the variance of targets
    
    # Initialize dual-target variables (only used when not single target, but initialize to avoid linter errors)
    if not is_single_target:
        target_mean_high = targets_pct[:, 0].mean()
        target_mean_low = targets_pct[:, 1].mean()
        baseline_high = np.full_like(targets_pct[:, 0], target_mean_high)
        baseline_low = np.full_like(targets_pct[:, 1], target_mean_low)
        baseline_mse_high = np.mean((targets_pct[:, 0] - baseline_high) ** 2)
        baseline_mse_low = np.mean((targets_pct[:, 1] - baseline_low) ** 2)
    else:
        # Initialize to dummy values (won't be used, but needed for linter)
        target_mean_high = 0.0
        target_mean_low = 0.0
        baseline_high = np.array([])
        baseline_low = np.array([])
        baseline_mse_high = 0.0
        baseline_mse_low = 0.0
    
    # Diagnostic: Show baseline info
    print(f"\n{dataset_name} Set - Baseline (Mean Prediction) Info:")
    if is_single_target:
        print(f"  Target mean: {target_mean:.6f}%")
        print(f"  Target variance: {baseline_mse:.6f}")
        print(f"  Target std: {np.std(targets_pct[:, 0]):.6f}%")
    else:
        print(f"  Target mean - High: {target_mean_high:.6f}%, Low: {target_mean_low:.6f}%")
        print(f"  Target variance - High: {baseline_mse_high:.6f}, Low: {baseline_mse_low:.6f}")
        print(f"  Target std - High: {np.std(targets_pct[:, 0]):.6f}%, Low: {np.std(targets_pct[:, 1]):.6f}%")
    
    # Diagnostic: Check prediction vs target correlation
    pred_std = np.std(predictions_pct[:, 0])
    target_std = np.std(targets_pct[:, 0])
    print(f"\n{dataset_name} Set - Prediction Variance Analysis:")
    if is_single_target:
        print(f"  Prediction std: {pred_std:.6f}% (target std: {target_std:.6f}%)")
        if pred_std < target_std * 0.1:
            print(f"  ⚠️ WARNING: Predictions have very low variance! Model may be predicting constant values.")
    else:
        pred_std_high = np.std(predictions_pct[:, 0])
        pred_std_low = np.std(predictions_pct[:, 1])
        print(f"  Prediction std - High: {pred_std_high:.6f}% (target std: {np.std(targets_pct[:, 0]):.6f}%)")
        print(f"  Prediction std - Low:  {pred_std_low:.6f}% (target std: {np.std(targets_pct[:, 1]):.6f}%)")
        if pred_std_high < np.std(targets_pct[:, 0]) * 0.1:
            print(f"  ⚠️ WARNING: Predictions have very low variance! Model may be predicting constant values.")
        if pred_std_low < np.std(targets_pct[:, 1]) * 0.1:
            print(f"  ⚠️ WARNING: Predictions have very low variance! Model may be predicting constant values.")
    
    # Calculate metrics (on percentage changes)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if is_single_target:
        # Single target metrics
        mse = mean_squared_error(targets_pct[:, 0], predictions_pct[:, 0])
        mae = mean_absolute_error(targets_pct[:, 0], predictions_pct[:, 0])
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_pct[:, 0], predictions_pct[:, 0])
        mape = np.mean(np.abs(targets_pct[:, 0] - predictions_pct[:, 0]))
        
        print(f"\n{dataset_name} Set - Prediction Metrics:")
        print(f"  MSE:  {mse:.4f} (Baseline MSE: {baseline_mse:.4f})")
        print(f"  MAE:  {mae:.4f}% (Mean absolute error in % change)")
        print(f"  RMSE: {rmse:.4f}% (Root mean squared error in % change)")
        print(f"  R²:   {r2:.4f}", end="")
        if r2 < 0:
            print(f" ⚠️ NEGATIVE - Model worse than baseline!")
        else:
            print()
        
        # For single target, return simplified results
        return {
            'predictions_pct': predictions_pct,
            'targets_pct': targets_pct,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    else:
        # Dual target metrics
        mse_high = mean_squared_error(targets_pct[:, 0], predictions_pct[:, 0])
        mae_high = mean_absolute_error(targets_pct[:, 0], predictions_pct[:, 0])
        rmse_high = np.sqrt(mse_high)
        r2_high = r2_score(targets_pct[:, 0], predictions_pct[:, 0])
        
        mse_low = mean_squared_error(targets_pct[:, 1], predictions_pct[:, 1])
        mae_low = mean_absolute_error(targets_pct[:, 1], predictions_pct[:, 1])
        rmse_low = np.sqrt(mse_low)
        r2_low = r2_score(targets_pct[:, 1], predictions_pct[:, 1])
        
        # Overall metrics
        mse_overall = mean_squared_error(targets_pct, predictions_pct)
        mae_overall = mean_absolute_error(targets_pct, predictions_pct)
        rmse_overall = np.sqrt(mse_overall)
        
        # Calculate absolute percentage errors
        high_mape = np.mean(np.abs(targets_pct[:, 0] - predictions_pct[:, 0]))
        low_mape = np.mean(np.abs(targets_pct[:, 1] - predictions_pct[:, 1]))
        
        # Print results with dataset name (all metrics are on percentage changes)
        print(f"\n{dataset_name} Set - SPX High % Change Prediction Metrics:")
        print(f"  MSE:  {mse_high:.4f} (Baseline MSE: {baseline_mse_high:.4f})")
        print(f"  MAE:  {mae_high:.4f}% (Mean absolute error in % change)")
        print(f"  RMSE: {rmse_high:.4f}% (Root mean squared error in % change)")
        print(f"  R²:   {r2_high:.4f}", end="")
        if r2_high < 0:
            print(f" ⚠️ NEGATIVE - Model worse than baseline!")
        else:
            print()
        
        print(f"\n{dataset_name} Set - SPX Low % Change Prediction Metrics:")
        print(f"  MSE:  {mse_low:.4f} (Baseline MSE: {baseline_mse_low:.4f})")
        print(f"  MAE:  {mae_low:.4f}% (Mean absolute error in % change)")
        print(f"  RMSE: {rmse_low:.4f}% (Root mean squared error in % change)")
        print(f"  R²:   {r2_low:.4f}", end="")
        if r2_low < 0:
            print(f" ⚠️ NEGATIVE - Model worse than baseline!")
        else:
            print()
    
        print(f"\n{dataset_name} Set - Overall Metrics (on % changes):")
        print(f"  MSE:  {mse_overall:.4f}")
        print(f"  MAE:  {mae_overall:.4f}%")
        print(f"  RMSE: {rmse_overall:.4f}%")
        
        print(f"\n{dataset_name} Set - Mean Absolute Error in % Change Prediction:")
        print(f"  High: {high_mape:.4f}% (average error in predicting % change)")
        print(f"  Low:  {low_mape:.4f}% (average error in predicting % change)")
        
        return {
            'high': {
                'mse': mse_high,
                'mae': mae_high,
                'rmse': rmse_high,
                'r2': r2_high
            },
            'low': {
                'mse': mse_low,
                'mae': mae_low,
                'rmse': rmse_low,
                'r2': r2_low
            },
            'overall': {
                'mse': mse_overall,
                'mae': mae_overall,
                'rmse': rmse_overall
            },
            'predictions_pct': predictions_pct,  # Percentage changes
            'targets_pct': targets_pct  # Percentage changes
        }


def main():
    # Configuration
    DATA_DIR = 'daily_data'
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 100  # Increased to allow more training
    LEARNING_RATE = 0.0005  # Increased from 0.00001 - too small LR prevents learning
    EARLY_STOPPING_PATIENCE = 10  # Increased patience - allow more time to improve
    LEARNING_RATE_WARMUP_EPOCHS = 5  # Warmup epochs for learning rate
    
    # Model selection: 'lstm' or 'transformer'
    MODEL_TYPE = 'transformer'  # Change to 'lstm' to use LSTM model
    
    # Use separate models for High and Low (each model focuses on one target)
    USE_SEPARATE_MODELS = True  # Set to False to use single model for both targets
    
    # LSTM-specific parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    
    # Transformer-specific parameters
    D_MODEL = 256  # Model dimension
    NHEAD = 16  # Number of attention heads
    TRANSFORMER_LAYERS = 10  # Number of transformer encoder layers
    DIM_FEEDFORWARD = 1024  # Feedforward network dimension
    
    # Check for best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Detailed GPU diagnostics
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")  # pyright: ignore[reportAttributeAccessIssue]
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*60}\n")
        
        torch.cuda.empty_cache()
        print(f"CUDA cache cleared. GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"\n{'='*60}")
        print("METAL (MPS) DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        print(f"MPS Built: {torch.backends.mps.is_built()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'='*60}\n")
        
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            print("MPS cache cleared.")
    else:
        print("\n⚠️ WARNING: Neither CUDA nor MPS is available! Training will use CPU (much slower).")
        print("Check:")
        print("  1. NVIDIA drivers are installed")
        print("  2. CUDA toolkit is installed")
        print("  3. PyTorch was installed with CUDA support")
        print("  4. Run: python -c 'import torch; print(torch.cuda.is_available())'")
    
    # Load data
    data = load_data(DATA_DIR)
    gc.collect()  # Free memory after data loading
    
    # Prepare normalized data (not sequences - sequences created on-the-fly)
    features_scaled, targets_scaled, day_ids, scaler_features, scaler_targets = prepare_data(data, SEQUENCE_LENGTH)
    del data  # Free the original dataframe
    gc.collect()  # Free memory after data preparation
    
    # Split data: 70% train, 15% validation, 15% test
    # Split indices instead of data to maintain temporal order
    total_samples = len(features_scaled)
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)
    
    # Split features, targets, and day_ids using indices
    train_features = features_scaled[:train_end + SEQUENCE_LENGTH]
    train_targets = targets_scaled[:train_end + SEQUENCE_LENGTH]
    train_day_ids = day_ids[:train_end + SEQUENCE_LENGTH]
    
    val_features = features_scaled[train_end:val_end + SEQUENCE_LENGTH]
    val_targets = targets_scaled[train_end:val_end + SEQUENCE_LENGTH]
    val_day_ids = day_ids[train_end:val_end + SEQUENCE_LENGTH]
    
    test_features = features_scaled[val_end:]
    test_targets = targets_scaled[val_end:]
    test_day_ids = day_ids[val_end:]
    
    # Initialize dataset variables to avoid linter errors (will be properly assigned in conditional)
    train_dataset_high = None
    val_dataset_high = None
    test_dataset_high = None
    train_dataset_low = None
    val_dataset_low = None
    test_dataset_low = None
    train_dataset = None
    val_dataset = None
    test_dataset = None
    input_size = 0
    
    # Create datasets with on-the-fly sequence generation (only same-day sequences)
    if USE_SEPARATE_MODELS:
        print("\n" + "="*60)
        print("TRAINING SEPARATE MODELS FOR HIGH AND LOW")
        print("="*60)
        # Create separate datasets for High (target_index=0) and Low (target_index=1)
        train_dataset_high = StockDataset(train_features, train_targets, train_day_ids, SEQUENCE_LENGTH, target_index=0)
        val_dataset_high = StockDataset(val_features, val_targets, val_day_ids, SEQUENCE_LENGTH, target_index=0)
        test_dataset_high = StockDataset(test_features, test_targets, test_day_ids, SEQUENCE_LENGTH, target_index=0)
        
        train_dataset_low = StockDataset(train_features, train_targets, train_day_ids, SEQUENCE_LENGTH, target_index=1)
        val_dataset_low = StockDataset(val_features, val_targets, val_day_ids, SEQUENCE_LENGTH, target_index=1)
        test_dataset_low = StockDataset(test_features, test_targets, test_day_ids, SEQUENCE_LENGTH, target_index=1)
        
        print(f"\nData split (separate models, same-day only):")
        print(f"  High Model - Training: {len(train_dataset_high)}, Validation: {len(val_dataset_high)}, Test: {len(test_dataset_high)}")
        print(f"  Low Model  - Training: {len(train_dataset_low)}, Validation: {len(val_dataset_low)}, Test: {len(test_dataset_low)}")
        
        # Get input size
        sample_seq, _ = train_dataset_high[0]
        input_size = sample_seq.shape[1]
    else:
        train_dataset = StockDataset(train_features, train_targets, train_day_ids, SEQUENCE_LENGTH)
        val_dataset = StockDataset(val_features, val_targets, val_day_ids, SEQUENCE_LENGTH)
        test_dataset = StockDataset(test_features, test_targets, test_day_ids, SEQUENCE_LENGTH)
        
        print(f"\nData split (single model, same-day only):")
        print(f"  Training: {len(train_dataset)} valid sequences")
        print(f"  Validation: {len(val_dataset)} valid sequences")
        print(f"  Test: {len(test_dataset)} valid sequences")
        
        # Get input size
        sample_seq, _ = train_dataset[0]
        input_size = sample_seq.shape[1]
    
    # Free intermediate data
    del train_features, train_targets, train_day_ids, val_features, val_targets, val_day_ids, test_features, test_targets, test_day_ids
    gc.collect()  # Free memory after dataset creation
    
    # For time series, we should NOT shuffle to maintain temporal order
    # Shuffling can break temporal dependencies and cause the model to learn
    # patterns that don't respect the chronological sequence
    # Multiprocessing: Use multiple workers for faster data loading
    # pin_memory=True speeds up GPU transfer if using CUDA
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())  # Use up to 4 workers, or CPU count if less
    pin_memory = torch.cuda.is_available()  # Enable pin_memory for GPU
    
    print(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}")
    
    # Initialize DataLoader variables to avoid linter errors (will be properly assigned in conditional)
    train_loader_high = None
    val_loader_high = None
    test_loader_high = None
    train_loader_low = None
    val_loader_low = None
    test_loader_low = None
    train_loader = None
    val_loader = None
    test_loader = None
    
    if USE_SEPARATE_MODELS:
        # Type assertions: we know these are not None in this branch
        assert train_dataset_high is not None and val_dataset_high is not None and test_dataset_high is not None
        assert train_dataset_low is not None and val_dataset_low is not None and test_dataset_low is not None
        
        train_loader_high = DataLoader(train_dataset_high, batch_size=BATCH_SIZE, shuffle=False, 
                                      num_workers=num_workers, pin_memory=pin_memory, 
                                      persistent_workers=True if num_workers > 0 else False)
        val_loader_high = DataLoader(val_dataset_high, batch_size=BATCH_SIZE, shuffle=False, 
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=True if num_workers > 0 else False)
        test_loader_high = DataLoader(test_dataset_high, batch_size=BATCH_SIZE, shuffle=False, 
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    persistent_workers=True if num_workers > 0 else False)
        
        train_loader_low = DataLoader(train_dataset_low, batch_size=BATCH_SIZE, shuffle=False, 
                                     num_workers=num_workers, pin_memory=pin_memory, 
                                     persistent_workers=True if num_workers > 0 else False)
        val_loader_low = DataLoader(val_dataset_low, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=True if num_workers > 0 else False)
        test_loader_low = DataLoader(test_dataset_low, batch_size=BATCH_SIZE, shuffle=False, 
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=True if num_workers > 0 else False)
    else:
        # Type assertion: we know these are not None in this branch
        assert train_dataset is not None and val_dataset is not None and test_dataset is not None
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=num_workers, pin_memory=pin_memory, 
                                  persistent_workers=True if num_workers > 0 else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=num_workers, pin_memory=pin_memory,
                               persistent_workers=True if num_workers > 0 else False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=True if num_workers > 0 else False)
    
    if USE_SEPARATE_MODELS:
        # Create separate models for High and Low
        if MODEL_TYPE.lower() == 'transformer':
            model_high = TransformerPredictorSingle(
                input_size=input_size,
                d_model=D_MODEL,
                nhead=NHEAD,
                num_layers=TRANSFORMER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=0.3,
                max_seq_length=SEQUENCE_LENGTH
            ).to(device)
            model_low = TransformerPredictorSingle(
                input_size=input_size,
                d_model=D_MODEL,
                nhead=NHEAD,
                num_layers=TRANSFORMER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=0.3,
                max_seq_length=SEQUENCE_LENGTH
            ).to(device)
            model_type_name = "Transformer"
        else:
            model_high = LSTMPredictorSingle(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=0.2
            ).to(device)
            model_low = LSTMPredictorSingle(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=0.2
            ).to(device)
            model_type_name = "LSTM"
        
        print(f"\n{'='*60}")
        print(f"High Model ({model_type_name})")
        print(f"{'='*60}")
        print(f"  Total parameters: {sum(p.numel() for p in model_high.parameters()):,}")
        print(f"\n{'='*60}")
        print(f"Low Model ({model_type_name})")
        print(f"{'='*60}")
        print(f"  Total parameters: {sum(p.numel() for p in model_low.parameters()):,}")
        
        # Train High model
        print(f"\n{'='*60}")
        print("TRAINING HIGH MODEL")
        print(f"{'='*60}")
        (train_losses_huber_high, train_losses_mse_high), (val_losses_huber_high, val_losses_mse_high), (test_losses_huber_high, test_losses_mse_high) = train_model(
            model_high, train_loader_high, val_loader_high, test_loader_high, device,
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            warmup_epochs=LEARNING_RATE_WARMUP_EPOCHS
        )
        
        # Train Low model
        print(f"\n{'='*60}")
        print("TRAINING LOW MODEL")
        print(f"{'='*60}")
        (train_losses_huber_low, train_losses_mse_low), (val_losses_huber_low, val_losses_mse_low), (test_losses_huber_low, test_losses_mse_low) = train_model(
            model_low, train_loader_low, val_loader_low, test_loader_low, device,
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            warmup_epochs=LEARNING_RATE_WARMUP_EPOCHS
        )
        
        # Evaluate both models and combine results
        print(f"\n{'='*60}")
        print("EVALUATING SEPARATE MODELS")
        print(f"{'='*60}")
        
        # Evaluate High model
        print("\n" + "="*60)
        print("HIGH MODEL EVALUATION")
        print("="*60)
        results_high = evaluate_model(model_high, test_loader_high, scaler_targets, device, "Test (High)")
        
        # Evaluate Low model
        print("\n" + "="*60)
        print("LOW MODEL EVALUATION")
        print("="*60)
        results_low = evaluate_model(model_low, test_loader_low, scaler_targets, device, "Test (Low)")
        
        # Combine predictions for overall metrics
        predictions_high = results_high['predictions_pct'].reshape(-1, 1)
        predictions_low = results_low['predictions_pct'].reshape(-1, 1)
        targets_high = results_high['targets_pct'].reshape(-1, 1)
        targets_low = results_low['targets_pct'].reshape(-1, 1)
        
        # Combine into (N, 2) format
        predictions_combined = np.column_stack([predictions_high, predictions_low])
        targets_combined = np.column_stack([targets_high, targets_low])
        
        # Calculate combined metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse_combined = mean_squared_error(targets_combined, predictions_combined)
        mae_combined = mean_absolute_error(targets_combined, predictions_combined)
        rmse_combined = np.sqrt(mse_combined)
        r2_combined = r2_score(targets_combined, predictions_combined)
        
        print(f"\n{'='*60}")
        print("COMBINED RESULTS (High + Low)")
        print(f"{'='*60}")
        print(f"MSE: {mse_combined:.8f}")
        print(f"MAE: {mae_combined:.8f}")
        print(f"RMSE: {rmse_combined:.8f}")
        print(f"R²: {r2_combined:.6f} {'⚠️ NEGATIVE - Model worse than baseline!' if r2_combined < 0 else ''}")
        
        # Use combined results for plotting
        train_losses_huber = [(h + l) / 2 for h, l in zip(train_losses_huber_high, train_losses_huber_low)]
        train_losses_mse = [(h + l) / 2 for h, l in zip(train_losses_mse_high, train_losses_mse_low)]
        val_losses_huber = [(h + l) / 2 for h, l in zip(val_losses_huber_high, val_losses_huber_low)]
        val_losses_mse = [(h + l) / 2 for h, l in zip(val_losses_mse_high, val_losses_mse_low)]
        test_losses_huber = [(h + l) / 2 for h, l in zip(test_losses_huber_high, test_losses_huber_low)]
        test_losses_mse = [(h + l) / 2 for h, l in zip(test_losses_mse_high, test_losses_mse_low)]
        
        # Store combined results for plotting
        results = {
            'predictions_pct': predictions_combined,
            'targets_pct': targets_combined
        }
    else:
        # Single model for both targets
        if MODEL_TYPE.lower() == 'transformer':
            model = TransformerPredictor(
                input_size=input_size,
                d_model=D_MODEL,
                nhead=NHEAD,
                num_layers=TRANSFORMER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=0.3,
                max_seq_length=SEQUENCE_LENGTH
            ).to(device)
            
            print(f"\nModel architecture: Transformer")
            print(f"  Input size: {input_size}")
            print(f"  Model dimension (d_model): {D_MODEL}")
            print(f"  Attention heads: {NHEAD}")
            print(f"  Transformer layers: {TRANSFORMER_LAYERS}")
            print(f"  Feedforward dimension: {DIM_FEEDFORWARD}")
            print(f"  Sequence length: {SEQUENCE_LENGTH}")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=0.2
            ).to(device)
            
            print(f"\nModel architecture: LSTM")
            print(f"  Input size: {input_size}")
            print(f"  Hidden size: {HIDDEN_SIZE}")
            print(f"  Number of layers: {NUM_LAYERS}")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Verify model is on GPU
        if torch.cuda.is_available():
            model_device = next(model.parameters()).device
            print(f"  Model device: {model_device}")
            if model_device.type != 'cuda':
                print(f"  ⚠️ WARNING: Model is on {model_device}, not GPU!")
            else:
                print(f"  ✓ Model is on GPU")
        
        # Train model
        (train_losses_huber, train_losses_mse), (val_losses_huber, val_losses_mse), (test_losses_huber, test_losses_mse) = train_model(
            model, train_loader, val_loader, test_loader, device,
            num_epochs=NUM_EPOCHS, 
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            warmup_epochs=LEARNING_RATE_WARMUP_EPOCHS
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, scaler_targets, device, "Test")
        
        # Evaluate model on training and test sets
        print("\n" + "="*60)
        print("EVALUATION RESULTS - TRAINING SET")
        print("="*60)
        
        train_results = evaluate_model(model, train_loader, scaler_targets, device, dataset_name="Training")
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS - TEST SET")
        print("="*60)
        
        test_results = evaluate_model(model, test_loader, scaler_targets, device, dataset_name="Test")
        
        # Comparison between Training and Test sets
        print("\n" + "="*60)
        print("TRAINING vs TEST SET COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'Training':<15} {'Test':<15} {'Difference':<15}")
        print("-" * 70)
        
        # High metrics comparison
        print(f"{'High - MSE':<25} {train_results['high']['mse']:<15.4f} {test_results['high']['mse']:<15.4f} {train_results['high']['mse'] - test_results['high']['mse']:<15.4f}")
        print(f"{'High - MAE':<25} {train_results['high']['mae']:<15.4f} {test_results['high']['mae']:<15.4f} {train_results['high']['mae'] - test_results['high']['mae']:<15.4f}")
        print(f"{'High - RMSE':<25} {train_results['high']['rmse']:<15.4f} {test_results['high']['rmse']:<15.4f} {train_results['high']['rmse'] - test_results['high']['rmse']:<15.4f}")
        print(f"{'High - R²':<25} {train_results['high']['r2']:<15.4f} {test_results['high']['r2']:<15.4f} {train_results['high']['r2'] - test_results['high']['r2']:<15.4f}")
        
        print()
        
        # Low metrics comparison
        print(f"{'Low - MSE':<25} {train_results['low']['mse']:<15.4f} {test_results['low']['mse']:<15.4f} {train_results['low']['mse'] - test_results['low']['mse']:<15.4f}")
        print(f"{'Low - MAE':<25} {train_results['low']['mae']:<15.4f} {test_results['low']['mae']:<15.4f} {train_results['low']['mae'] - test_results['low']['mae']:<15.4f}")
        print(f"{'Low - RMSE':<25} {train_results['low']['rmse']:<15.4f} {test_results['low']['rmse']:<15.4f} {train_results['low']['rmse'] - test_results['low']['rmse']:<15.4f}")
        print(f"{'Low - R²':<25} {train_results['low']['r2']:<15.4f} {test_results['low']['r2']:<15.4f} {train_results['low']['r2'] - test_results['low']['r2']:<15.4f}")
        
        print()
        
        # Overall metrics comparison
        print(f"{'Overall - MSE':<25} {train_results['overall']['mse']:<15.4f} {test_results['overall']['mse']:<15.4f} {train_results['overall']['mse'] - test_results['overall']['mse']:<15.4f}")
        print(f"{'Overall - MAE':<25} {train_results['overall']['mae']:<15.4f} {test_results['overall']['mae']:<15.4f} {train_results['overall']['mae'] - test_results['overall']['mae']:<15.4f}")
        print(f"{'Overall - RMSE':<25} {train_results['overall']['rmse']:<15.4f} {test_results['overall']['rmse']:<15.4f} {train_results['overall']['rmse'] - test_results['overall']['rmse']:<15.4f}")
        
        # Use test results for final output
        results = test_results
    
    # Sample predictions from test set (showing percentage changes)
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 test samples - Percentage Changes)")
    print("="*60)
    print(f"{'Index':<8} {'Actual High %':<15} {'Pred High %':<15} {'Error %':<12} {'Actual Low %':<15} {'Pred Low %':<15} {'Error %':<12}")
    print("-" * 100)
    # Use 'predictions_pct' and 'targets_pct' keys (consistent across both model types)
    predictions_pct = results.get('predictions_pct', results.get('predictions', None))
    targets_pct = results.get('targets_pct', results.get('targets', None))
    if predictions_pct is not None and targets_pct is not None:
        for i in range(min(20, len(predictions_pct))):
            high_actual = targets_pct[i, 0]  # % change
            high_pred = predictions_pct[i, 0]  # % change
            high_error = high_actual - high_pred
            low_actual = targets_pct[i, 1]  # % change
            low_pred = predictions_pct[i, 1]  # % change
            low_error = low_actual - low_pred
            print(f"{i:<8} {high_actual:<15.4f}% {high_pred:<15.4f}% {high_error:<12.4f}% {low_actual:<15.4f}% {low_pred:<15.4f}% {low_error:<12.4f}%")
    else:
        print("  Warning: Could not find predictions/targets in results dictionary")
    
    print("\n" + "="*60)
    print("Model training and evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()

