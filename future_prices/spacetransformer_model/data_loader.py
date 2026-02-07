"""
Data loading utilities for SpaceTransformer model
Similar to MVE_SSNs_model data loading functionality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


def load_data(data_path: str,
              test_size: float = 0.2, random_state: int = 42,
              golden_test: bool = False, batch_size: int = 256,
              max_rows: Optional[int] = None,
              skip_split: bool = False,
              skip_scaling: bool = False) -> Tuple:
    """
    Load and prepare data for training or inference
    
    Args:
        data_path: Path to CSV file with features
        test_size: Proportion of data for testing (ignored if skip_split=True)
        random_state: Random seed
        golden_test: If True, only read minimal data needed for testing (batch_size + small validation)
        batch_size: Batch size for golden test mode (only used if golden_test=True)
        max_rows: Maximum number of rows to read from CSV (None = read all)
        skip_split: If True, return full dataset without train/test split (for inference)
    
    Returns:
        (X_train, X_test, scaler, feature_names)
        If skip_split=True, returns (X_full, X_full, scaler, feature_names) where X_full is the full dataset
    """
    print(f"Loading data from {data_path}...")
    
    # No target columns; we predict all and focus loss elsewhere
    
    # For golden test, only read the rows we need
    # Need enough rows to create at least a few sequences after train/test split
    if golden_test:
        # Read enough rows to create sequences (need context_length + target_length per sequence)
        # With 80/20 split, we need at least (context_length + target_length) * 2 / 0.8 rows
        # For safety, read at least 500 rows or batch_size * 10, whichever is larger
        nrows_to_read = max(500, batch_size * 10)
        print(f"GOLDEN TEST MODE: Reading only first {nrows_to_read} rows from CSV")
        df = pd.read_csv(data_path, nrows=nrows_to_read)
    elif max_rows is not None:
        print(f"Reading only first {max_rows} rows from CSV")
        df = pd.read_csv(data_path, nrows=max_rows)
    else:
        df = pd.read_csv(data_path)
    
    # Select feature columns
    # Forward-looking columns contain future information
    forward_looking_cols = ['PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
    
    # Exclude non-feature columns and forward-looking columns
    # We keep the target_cols (Low, High, Close) in features
    exclude_cols = ['Date', 'DateTime', 'DateTime_ET'] + forward_looking_cols
    
    # Remove duplicates
    exclude_cols = list(set(exclude_cols))
    # feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'BB_20_MA', 'BB_20_2_Upper', 'BB_20_2_Lower', 'MFI_14']
    print(f"Using {len(feature_cols)} features")
    
    # Extract features and target(s)
    X = df[feature_cols].values
    
    # Check for null values in features before handling
    X_array = np.asarray(X)
    nan_count = np.isnan(X_array).sum()
    nan_percentage = (nan_count / X_array.size) * 100 if X_array.size > 0 else 0
    
    if nan_count > 0:
        print(f"Found {nan_count:,} NaN values in features ({nan_percentage:.2f}% of total values)")
    else:
        print("No NaN values found in features")
    
    # Split data (features only) or use full dataset
    if skip_split:
        # Return full dataset without splitting (for inference/plotting)
        X_full = X
        if skip_scaling:
            # Return raw data without scaling (caller will scale with their own scaler)
            scaler = None
            print(f"Loaded {len(X_full)} samples (no split, no scaling)")
        else:
            # Scale features on full dataset
            scaler = StandardScaler()
            X_full = scaler.fit_transform(X_full)
            print(f"Loaded {len(X_full)} samples (no split)")
        # Return same data for both X_train and X_test to maintain return signature
        return X_full, X_full, scaler, feature_cols
    else:
        # Split data for training
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, X_test, scaler, feature_cols

