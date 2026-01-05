"""
Utility functions for SpaceTimeFormer model.
"""

import pandas as pd
import numpy as np


def extract_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclical time features (hour, day, month) from datetime columns.
    
    This function adds sin/cos encoded time features to the dataframe:
    - hour_sin, hour_cos (if sub-daily data)
    - day_sin, day_cos (day of week)
    - month_sin, month_cos (month of year)
    
    Args:
        df: DataFrame with datetime columns (DateTime_ET, DateTime, or Date)
    
    Returns:
        DataFrame with added cyclical time features
    """
    # Prefer DateTime_ET, fallback to DateTime, then Date
    date_col = None
    if 'DateTime_ET' in df.columns and False:
        date_col = 'DateTime_ET'
    elif 'DateTime' in df.columns:
        date_col = 'DateTime'
    elif 'Date' in df.columns:
        date_col = 'Date'
        
    if date_col:
        print(f"Extracting cyclical time features from {date_col}...")
        try:
            # Convert to datetime
            dt_series = pd.to_datetime(df[date_col], errors='coerce')
            
            # Check for NaT values and warn if found
            nat_count = dt_series.isna().sum()
            if nat_count > 0:
                print(f"  -> Warning: {nat_count} NaT (Not a Time) values found in {date_col}")
                print(f"  -> These will be filled with 0 for cyclical features")
            
            # Extract features and apply cyclical encoding (sin/cos)
            # Always add all cyclical features for consistency, even if some are constant
            # This ensures the same number of features regardless of data subset
            
            # 1. Hour of day (0-23)
            # Always add hour features for consistency (will be constant if no sub-daily variation)
            # Fill NaT with 0 (midnight) for hour calculation
            hour_values = dt_series.dt.hour.fillna(0)
            df['hour_sin'] = np.sin(2 * np.pi * hour_values / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour_values / 24)
            if hour_values.nunique() > 1:
                print("  -> Added hour_sin/cos (variable)")
            else:
                print("  -> Added hour_sin/cos (constant - no sub-daily variation)")
                
            # 2. Day of week (0-6)
            # Fill NaT with 0 (Monday) for dayofweek calculation
            dayofweek_values = dt_series.dt.dayofweek.fillna(0)
            df['day_sin'] = np.sin(2 * np.pi * dayofweek_values / 7)
            df['day_cos'] = np.cos(2 * np.pi * dayofweek_values / 7)
            print("  -> Added day_sin/cos")
            
            # 3. Month of year (1-12)
            # Fill NaT with 1 (January) for month calculation
            month_values = dt_series.dt.month.fillna(1)
            df['month_sin'] = np.sin(2 * np.pi * month_values / 12)
            df['month_cos'] = np.cos(2 * np.pi * month_values / 12)
            print("  -> Added month_sin/cos")
            
        except Exception as e:
            print(f"Warning: Could not extract time features: {e}")
    
    return df

