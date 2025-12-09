"""
Data loading utilities for SpaceTransformer model
Similar to MVE_SSNs_model data loading functionality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_data(data_path: str,
              test_size: float = 0.2, random_state: int = 42,
              golden_test: bool = False, batch_size: int = 256) -> Tuple:
    """
    Load and prepare data for training
    
    Args:
        data_path: Path to CSV file with features
        test_size: Proportion of data for testing
        random_state: Random seed
        golden_test: If True, only read minimal data needed for testing (batch_size + small validation)
        batch_size: Batch size for golden test mode (only used if golden_test=True)
    
    Returns:
        (X_train, X_test, scaler, feature_names)
    """
    print(f"Loading data from {data_path}...")
    
    # No target columns; we predict all and focus loss elsewhere
    
    # For golden test, only read the rows we need
    if golden_test:
        # Read only: batch_size for training + small validation set (e.g., 10% of batch_size)
        # Add some buffer for test set
        val_samples = max(1, batch_size // 10)
        test_samples = max(1, batch_size // 10)
        nrows_to_read = batch_size + val_samples + test_samples
        print(f"GOLDEN TEST MODE: Reading only first {nrows_to_read} rows from CSV")
        df = pd.read_csv(data_path, nrows=nrows_to_read)
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
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
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
    
    # Split data (features only)
    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, scaler, feature_cols

