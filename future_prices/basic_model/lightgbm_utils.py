"""
Shared utilities for LightGBM CloseToHigh / CloseToLow models.

Provides: load_data, prepare_features_and_target (parameterized by target_column),
train_lightgbm, scale_target, unscale_target, evaluate_model, get_feature_importance,
plot_predictions_vs_actual, run_correlation_analysis, run_kfold_cv, write_metrics_file.
"""

# Standard library imports
import os
import time
import warnings
from pathlib import Path

# Suppress sklearn feature name warnings (LightGBM works fine with numpy arrays)
warnings.filterwarnings('ignore', category=UserWarning,
                        message='X does not have valid feature names')

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

# Check for MPS availability (disabled for stability - MPS tensor operations can crash)
USE_MPS = False
DEVICE = None
try:
    import torch
    if torch.backends.mps.is_available():
        print("ℹ MPS (Metal) available but disabled - Using CPU for stability")
    elif torch.cuda.is_available():
        print("ℹ CUDA available but disabled - Using CPU for stability")
    else:
        print("ℹ Using CPU for feature engineering")
except ImportError:
    print("ℹ PyTorch not available - Feature engineering will use NumPy")


def load_data(csv_path, sample_size=None, chunk_size=100000):
    """
    Load data from CSV file, handling large files efficiently.

    Args:
        csv_path: Path to the CSV file
        sample_size: If specified, only load this many rows (for testing)
        chunk_size: Number of rows to process at a time for large files

    Returns:
        DataFrame with features and target
    """
    print(f"Loading data from {csv_path}...")

    # Check file size
    file_size_gb = os.path.getsize(csv_path) / (1024 ** 3)
    print(f"File size: {file_size_gb:.2f} GB")

    # For very large files, read in chunks
    if file_size_gb > 1.0:
        print("Large file detected. Reading in chunks...")
        chunks = []
        total_rows = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if sample_size and total_rows >= sample_size:
                break

            chunks.append(chunk)
            total_rows += len(chunk)

            if sample_size and total_rows >= sample_size:
                excess = total_rows - sample_size
                if excess > 0:
                    chunks[-1] = chunks[-1].iloc[:-excess]
                break

            if len(chunks) % 10 == 0:
                print(f"  Loaded {total_rows:,} rows...", end='\r', flush=True)

        print(f"  Loaded {total_rows:,} rows total" + " " * 20)
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df):,} total rows")
    else:
        df = pd.read_csv(csv_path)
        if sample_size:
            df = df.head(sample_size)
        print(f"Loaded {len(df):,} rows")

    return df


def prepare_features_and_target(df, target_column='PctChange_CloseToHigh'):
    """
    Prepare features and target variable.

    The target is the existing column given by target_column (e.g. PctChange_CloseToHigh
    or PctChange_CloseToLow). We exclude it from features to avoid leakage.

    Args:
        df: DataFrame with all columns
        target_column: Name of the target column (e.g. 'PctChange_CloseToHigh', 'PctChange_CloseToLow')

    Returns:
        X: Feature matrix
        y: Target vector (scaled)
        feature_cols: List of feature column names
        TARGET_SCALE: Scale factor used for target
        SCALING_METHOD: Method used for scaling
        y_original: Original unscaled target (for correlation analysis)
    """
    print("\nPreparing features and target...")

    if target_column not in df.columns:
        raise ValueError(f"DataFrame must contain column '{target_column}'")
    y = df[target_column].values.astype(np.float64)

    store_abs_for_original = False

    SCALING_METHOD = "asinh"

    if SCALING_METHOD == "none":
        y_scaled = y.astype(np.float64)
        TARGET_SCALE = 1.0
        print(f"\n  Using raw percentage changes (no scaling)...")
        print(f"  Target std: {np.std(y_scaled):.6f}%")
        print(f"  LightGBM can learn from small values directly")

    elif SCALING_METHOD == "linear":
        TARGET_SCALE = 1000.0
        y_scaled = (y * TARGET_SCALE).astype(np.float64)
        print(f"\n  Using linear scaling (scale_factor={TARGET_SCALE}x)...")
        print(f"  Original target std: {np.std(y):.6f}%")
        print(f"  Scaled target std: {np.std(y_scaled):.4f}")

    elif SCALING_METHOD == "log":
        LOG_SCALE = 100.0
        y_scaled = np.sign(y) * np.log1p(np.abs(y) * LOG_SCALE)
        TARGET_SCALE = LOG_SCALE
        print(f"\n  Using log transformation (log1p scale={LOG_SCALE})...")
        print(f"  Original target std: {np.std(y):.6f}%")
        print(f"  Transformed target std: {np.std(y_scaled):.6f}")
        print(f"  Formula: sign(y) * log1p(abs(y) * {LOG_SCALE})")

    elif SCALING_METHOD == "exponential":
        EXP_SCALE = 50.0
        MAX_EXP_ARG = 10.0
        y_abs_scaled = np.clip(np.abs(y) * EXP_SCALE, 0, MAX_EXP_ARG)
        y_scaled = np.sign(y) * (np.exp(y_abs_scaled) - 1.0)
        TARGET_SCALE = EXP_SCALE
        print(f"\n  Using exponential transformation (exp scale={EXP_SCALE}, max_arg={MAX_EXP_ARG})...")
        print(f"  Original target std: {np.std(y):.6f}%")
        print(f"  Transformed target std: {np.std(y_scaled):.6f}")
        print(f"  Original target range: [{np.min(y):.6f}%, {np.max(y):.6f}%]")
        print(f"  Transformed target range: [{np.min(y_scaled):.6f}, {np.max(y_scaled):.6f}]")
        print(f"  Formula: sign(y) * (exp(clip(abs(y) * {EXP_SCALE}, 0, {MAX_EXP_ARG})) - 1)")
        print(f"  Warning: Very aggressive amplification - small values become much larger")
        example_values = np.array([0.001, 0.01, 0.1, 1.0])
        example_transformed = np.sign(example_values) * (np.exp(np.clip(np.abs(example_values) * EXP_SCALE, 0, MAX_EXP_ARG)) - 1.0)
        print(f"  Example transformations:")
        for orig, trans in zip(example_values, example_transformed):
            print(f"    {orig:.3f}% -> {trans:.3f}")

    elif SCALING_METHOD == "asinh":
        ASINH_SCALE = 50.0
        y_scaled = np.sign(y) * np.arcsinh(np.abs(y) * ASINH_SCALE)
        TARGET_SCALE = ASINH_SCALE
        print(f"\n  Using asinh transformation (scale={ASINH_SCALE})...")
        print(f"  Linear for small |y|, log-like for large |y| (combines log + exponential benefits)")
        print(f"  Original target std: {np.std(y):.6f}%")
        print(f"  Transformed target std: {np.std(y_scaled):.6f}")
        print(f"  Formula: sign(y) * asinh(|y| * {ASINH_SCALE})")

    elif SCALING_METHOD == "absolute":
        store_abs_for_original = True
        y_raw = np.abs(y).astype(np.float64)
        y_scaled = y_raw
        TARGET_SCALE = 1.0
        SCALING_METHOD = "none"

        print(f"\n  Using absolute value of {target_column} (no scaling)...")
        print(f"  Absolute change - Mean: {np.mean(y_raw):.4f}, Std: {np.std(y_raw):.4f}")
        print(f"  Note: Predicting magnitude only (always positive), not direction")

    elif SCALING_METHOD == "absolute_log":
        store_abs_for_original = True
        y_raw = np.abs(y).astype(np.float64)
        LOG_SCALE = 0.1
        y_scaled = np.log1p(y_raw * LOG_SCALE)
        TARGET_SCALE = LOG_SCALE
        print(f"\n  Using log of absolute value of change (log1p scale={LOG_SCALE})...")
        print(f"  Absolute change - Mean: {np.mean(y_raw):.4f}, Std: {np.std(y_raw):.4f}")
        print(f"  Log-scaled target - Mean: {np.mean(y_scaled):.6f}, Std: {np.std(y_scaled):.6f}")
        print(f"  Formula: log1p(|change| * {LOG_SCALE})")

    else:
        raise ValueError(f"Unknown scaling method: {SCALING_METHOD}")

    # Check time intervals if DateTime column exists
    if 'DateTime' in df.columns:
        try:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            time_diffs = df['DateTime'].diff().iloc[1:-1]
            avg_interval = time_diffs.mean()
            print(f"\n  Time interval analysis:")
            print(f"    Average time between rows: {avg_interval}")
            if avg_interval.total_seconds() < 60:
                print(f"    ⚠ Very short intervals ({avg_interval.total_seconds():.1f} seconds)")
                print(f"    Percentage changes may be genuinely very small for such short intervals")
        except Exception:
            pass

    base_columns = [
        'Volume',
        'BB_10_STD',
        'BB_20_STD',
        'BB_50_STD',
        'Hours_From_Formal_Trading',
        'Hours_From_Overnight_Trading',
        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',
        'month_sin',
        'month_cos',
        'BB_20_1_Upper',
        'BB_20_1_Lower',
        'BB_20_2_Upper',
        'BB_20_2_Lower',
        'BB_20_3_Upper',
        'BB_20_3_Lower',
        'BB_50_MA',
        'BB_50_1_Upper',
        'BB_50_1_Lower',
        'BB_50_2_Upper',
        'BB_50_2_Lower',
        'BB_50_3_Upper',
        'BB_50_3_Lower',
        'EMA_10',
        'EMA_20',
        'EMA_50',
        'RSI_7',
        'RSI_14',
        'RSI_28',
        'MFI_7',
        'MFI_14',
        'MFI_28',
        'ATR_14',
        'VXM_Open',
        'VXM_High',
        'VXM_Low',
        'VXM_Close',
        'VXM_Volume',
        'VXM_BB_10_STD',
        'VXM_BB_20_STD',
        'VXM_BB_50_STD',
        'VXM_BB_20_1_Upper',
        'VXM_BB_20_1_Lower',
        'VXM_BB_20_2_Upper',
        'VXM_BB_20_2_Lower',
        'VXM_BB_20_3_Upper',
        'VXM_BB_20_3_Lower',
        'VXM_BB_50_MA',
        'VXM_BB_50_1_Upper',
        'VXM_BB_50_1_Lower',
        'VXM_BB_50_2_Upper',
        'VXM_BB_50_2_Lower',
        'VXM_BB_50_3_Upper',
        'VXM_BB_50_3_Lower',
        'VXM_EMA_10',
        'VXM_EMA_20',
        'VXM_EMA_50',
    ]

    available_base_cols = [col for col in base_columns if col in df.columns and col != target_column]
    missing_cols = [col for col in base_columns if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Some columns not found: {missing_cols}")

    feature_df = df[available_base_cols].copy()

    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        use_gpu = False
        high_t = None
        low_t = None
        close_t = None

        if 'BB_20_1_Upper' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_1_upper_t = torch.tensor(feature_df['BB_20_1_Upper'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_1_Upper_vs_High'] = ((bb20_1_upper_t - high_t) / high_t).cpu().numpy()
                feature_df['BB_20_1_Upper_vs_Close'] = ((bb20_1_upper_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_1_Upper_vs_Low'] = ((bb20_1_upper_t - low_t) / low_t).cpu().numpy()
            else:
                bb20_1_upper = feature_df['BB_20_1_Upper'].values
                feature_df['BB_20_1_Upper_vs_High'] = (bb20_1_upper - high) / high
                feature_df['BB_20_1_Upper_vs_Close'] = (bb20_1_upper - close) / close
                feature_df['BB_20_1_Upper_vs_Low'] = (bb20_1_upper - low) / low

        if 'BB_20_2_Upper' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_2_upper_t = torch.tensor(feature_df['BB_20_2_Upper'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_2_Upper_vs_High'] = ((bb20_2_upper_t - high_t) / high_t).cpu().numpy()
                feature_df['BB_20_2_Upper_vs_Close'] = ((bb20_2_upper_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_2_Upper_vs_Low'] = ((bb20_2_upper_t - low_t) / low_t).cpu().numpy()
            else:
                bb20_2_upper = feature_df['BB_20_2_Upper'].values
                feature_df['BB_20_2_Upper_vs_High'] = (bb20_2_upper - high) / high
                feature_df['BB_20_2_Upper_vs_Close'] = (bb20_2_upper - close) / close
                feature_df['BB_20_2_Upper_vs_Low'] = (bb20_2_upper - low) / low

        if 'BB_20_3_Upper' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_3_upper_t = torch.tensor(feature_df['BB_20_3_Upper'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_3_Upper_vs_High'] = ((bb20_3_upper_t - high_t) / high_t).cpu().numpy()
                feature_df['BB_20_3_Upper_vs_Close'] = ((bb20_3_upper_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_3_Upper_vs_Low'] = ((bb20_3_upper_t - low_t) / low_t).cpu().numpy()
            else:
                bb20_3_upper = feature_df['BB_20_3_Upper'].values
                feature_df['BB_20_3_Upper_vs_High'] = (bb20_3_upper - high) / high
                feature_df['BB_20_3_Upper_vs_Close'] = (bb20_3_upper - close) / close
                feature_df['BB_20_3_Upper_vs_Low'] = (bb20_3_upper - low) / low

        if 'BB_20_1_Lower' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_1_lower_t = torch.tensor(feature_df['BB_20_1_Lower'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_1_Lower_vs_Low'] = ((bb20_1_lower_t - low_t) / low_t).cpu().numpy()
                feature_df['BB_20_1_Lower_vs_Close'] = ((bb20_1_lower_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_1_Lower_vs_High'] = ((bb20_1_lower_t - high_t) / high_t).cpu().numpy()
            else:
                bb20_1_lower = feature_df['BB_20_1_Lower'].values
                feature_df['BB_20_1_Lower_vs_Low'] = (bb20_1_lower - low) / low
                feature_df['BB_20_1_Lower_vs_Close'] = (bb20_1_lower - close) / close
                feature_df['BB_20_1_Lower_vs_High'] = (bb20_1_lower - high) / high

        if 'BB_20_2_Lower' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_2_lower_t = torch.tensor(feature_df['BB_20_2_Lower'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_2_Lower_vs_Low'] = ((bb20_2_lower_t - low_t) / low_t).cpu().numpy()
                feature_df['BB_20_2_Lower_vs_Close'] = ((bb20_2_lower_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_2_Lower_vs_High'] = ((bb20_2_lower_t - high_t) / high_t).cpu().numpy()
            else:
                bb20_2_lower = feature_df['BB_20_2_Lower'].values
                feature_df['BB_20_2_Lower_vs_Low'] = (bb20_2_lower - low) / low
                feature_df['BB_20_2_Lower_vs_Close'] = (bb20_2_lower - close) / close
                feature_df['BB_20_2_Lower_vs_High'] = (bb20_2_lower - high) / high

        if 'BB_20_3_Lower' in feature_df.columns:
            if use_gpu and high_t is not None and low_t is not None and close_t is not None:
                import torch
                bb20_3_lower_t = torch.tensor(feature_df['BB_20_3_Lower'].values, device=DEVICE, dtype=torch.float32)
                feature_df['BB_20_3_Lower_vs_Low'] = ((bb20_3_lower_t - low_t) / low_t).cpu().numpy()
                feature_df['BB_20_3_Lower_vs_Close'] = ((bb20_3_lower_t - close_t) / close_t).cpu().numpy()
                feature_df['BB_20_3_Lower_vs_High'] = ((bb20_3_lower_t - high_t) / high_t).cpu().numpy()
            else:
                bb20_3_lower = feature_df['BB_20_3_Lower'].values
                feature_df['BB_20_3_Lower_vs_Low'] = (bb20_3_lower - low) / low
                feature_df['BB_20_3_Lower_vs_Close'] = (bb20_3_lower - close) / close
                feature_df['BB_20_3_Lower_vs_High'] = (bb20_3_lower - high) / high

    print("\nAdding lagged features...")
    pct_changes_full = df['Close'].pct_change() * 100

    for lag in [1, 2, 3, 5, 10, 20]:
        lagged_pct = pct_changes_full.shift(lag).values
        feature_df[f'PctChange_Lag_{lag}'] = lagged_pct
        print(f"  Added PctChange_Lag_{lag}")

    if len(pct_changes_full) > 5:
        momentum_5 = (pct_changes_full - pct_changes_full.shift(5)).values
        momentum_10 = (pct_changes_full - pct_changes_full.shift(10)).values
        feature_df['Momentum_5'] = momentum_5
        feature_df['Momentum_10'] = momentum_10
        print(f"  Added Momentum_5 and Momentum_10")

    print("Adding rolling statistics...")
    for window in [5, 10, 20, 50]:
        rolling_mean = pct_changes_full.rolling(window).mean().values
        rolling_std = pct_changes_full.rolling(window).std().values
        rolling_min = pct_changes_full.rolling(window).min().values
        rolling_max = pct_changes_full.rolling(window).max().values
        feature_df[f'RollingMean_{window}'] = rolling_mean
        feature_df[f'RollingStd_{window}'] = rolling_std
        feature_df[f'RollingMin_{window}'] = rolling_min
        feature_df[f'RollingMax_{window}'] = rolling_max
        print(f"  Added RollingMean_{window}, RollingStd_{window}, RollingMin_{window}, RollingMax_{window}")

    if 'Close' in df.columns:
        close_values = df['Close'].values
        for window in [10, 20, 50]:
            rolling_min_price = df['Close'].rolling(window).min().values
            rolling_max_price = df['Close'].rolling(window).max().values
            rolling_range = rolling_max_price - rolling_min_price
            # Avoid divide-by-zero: only divide where range > 0 (safe_denom avoids evaluating 0/0)
            safe_denom = np.where(rolling_range > 0, rolling_range, 1.0)
            price_position = np.where(rolling_range > 0,
                                      (close_values - rolling_min_price) / safe_denom,
                                      0.5)
            feature_df[f'PricePosition_{window}'] = price_position
            print(f"  Added PricePosition_{window}")

    if all(col in df.columns for col in ['VXM_High', 'VXM_Low', 'VXM_Close']):
        vxm_high = df['VXM_High'].values
        vxm_low = df['VXM_Low'].values
        vxm_close = df['VXM_Close'].values

        if 'VXM_BB_20_1_Upper' in feature_df.columns:
            vxm_bb20_1_upper = feature_df['VXM_BB_20_1_Upper'].values
            feature_df['VXM_BB_20_1_Upper_vs_VXM_High'] = (vxm_bb20_1_upper - vxm_high) / vxm_high
            feature_df['VXM_BB_20_1_Upper_vs_VXM_Close'] = (vxm_bb20_1_upper - vxm_close) / vxm_close
            feature_df['VXM_BB_20_1_Upper_vs_VXM_Low'] = (vxm_bb20_1_upper - vxm_low) / vxm_low
            print(f"  Added VXM_BB_20_1_Upper derived features")

        if 'VXM_BB_20_1_Lower' in feature_df.columns:
            vxm_bb20_1_lower = feature_df['VXM_BB_20_1_Lower'].values
            feature_df['VXM_BB_20_1_Lower_vs_VXM_Low'] = (vxm_bb20_1_lower - vxm_low) / vxm_low
            feature_df['VXM_BB_20_1_Lower_vs_VXM_Close'] = (vxm_bb20_1_lower - vxm_close) / vxm_close
            feature_df['VXM_BB_20_1_Lower_vs_VXM_High'] = (vxm_bb20_1_lower - vxm_high) / vxm_high
            print(f"  Added VXM_BB_20_1_Lower derived features")

        if 'VXM_Close' in df.columns:
            vxm_pct_change = df['VXM_Close'].pct_change() * 100
            feature_df['VXM_PctChange'] = vxm_pct_change.values
            feature_df['VXM_PctChange_Lag_1'] = vxm_pct_change.shift(1).values
            feature_df['VXM_PctChange_Lag_2'] = vxm_pct_change.shift(2).values
            print(f"  Added VXM_PctChange and lagged features")

        if 'Close' in df.columns:
            es_close = df['Close'].values
            vxm_close = df['VXM_Close'].values
            vxm_es_ratio = np.where(es_close > 0, vxm_close / es_close, 0)
            feature_df['VXM_ES_Ratio'] = vxm_es_ratio
            print(f"  Added VXM_ES_Ratio")

    if len(pct_changes_full) > 20:
        for window in [10, 20, 50]:
            volatility = pct_changes_full.rolling(window).std().values
            feature_df[f'Volatility_{window}'] = volatility
            print(f"  Added Volatility_{window}")

        if len(pct_changes_full) > 50:
            vol_short = pct_changes_full.rolling(10).std().values
            vol_long = pct_changes_full.rolling(50).std().values
            # Avoid divide-by-zero: only divide where vol_long > 0
            safe_vol_long = np.where(vol_long > 0, vol_long, 1.0)
            vol_ratio = np.where(vol_long > 0, vol_short / safe_vol_long, 1.0)
            feature_df['Volatility_Ratio'] = vol_ratio
            print(f"  Added Volatility_Ratio")

    if 'RSI_7' in feature_df.columns and 'Volume' in feature_df.columns:
        feature_df['RSI_7_x_Volume'] = feature_df['RSI_7'] * feature_df['Volume']
        print(f"  Added RSI_7_x_Volume interaction")

    close_values = df['Close'].values if 'Close' in df.columns else None
    for atr_col, pct_col in [('ATR_7', 'ATR_7_Pct'), ('ATR_14', 'ATR_14_Pct'), ('ATR_28', 'ATR_28_Pct')]:
        if atr_col in feature_df.columns and close_values is not None:
            atr_values = feature_df[atr_col].values
            atr_pct = np.where(close_values > 0, (atr_values / close_values) * 100, 0)
            feature_df[pct_col] = atr_pct
            print(f"  Added {pct_col} ({atr_col} as % of price)")

    if len(pct_changes_full) > 20:
        # Replace NaN with 0 so np.sign and astype(int8) don't produce invalid values
        pct_vals = np.asarray(pct_changes_full.values, dtype=np.float64)
        pct_vals = np.where(np.isfinite(pct_vals), pct_vals, 0.0)
        pct_signs = np.sign(pct_vals).astype(np.int8)
        for window in [10, 20]:
            trend_strength = np.full(len(pct_changes_full), 0.5, dtype=np.float32)
            for i in range(window, len(pct_signs)):
                window_signs = pct_signs[i-window:i]
                current_sign = pct_signs[i]
                matches = np.sum(window_signs == current_sign)
                trend_strength[i] = matches / window
            feature_df[f'TrendStrength_{window}'] = trend_strength
            print(f"  Added TrendStrength_{window} (optimized)")

    feature_cols = list(feature_df.columns)
    X = feature_df.values

    valid_mask = ~np.isnan(y_scaled)
    feature_nan_mask = ~np.isnan(X).any(axis=1)
    valid_mask = valid_mask & feature_nan_mask

    raw_y = np.asarray(y, dtype=np.float64)
    if store_abs_for_original:
        y_original = np.abs(raw_y)[valid_mask]
    else:
        y_original = raw_y[valid_mask]

    X = X[valid_mask]
    y = y_scaled[valid_mask]

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    if SCALING_METHOD == "absolute":
        print(f"Target: Absolute value of price change (no scaling)")
    elif SCALING_METHOD == "absolute_log":
        print(f"Target: Log of absolute value of price change (log1p scale={TARGET_SCALE})")
    elif SCALING_METHOD == "asinh":
        print(f"Target: Percentage change (asinh scale={TARGET_SCALE}: linear for small |y|, log-like for large)")
    elif TARGET_SCALE == 1.0:
        print(f"Target: {target_column} (no scaling)")
    elif TARGET_SCALE == 100.0:
        print(f"Target: {target_column} (log-transformed for training)")
    else:
        print(f"Target: {target_column} (linearly scaled by {TARGET_SCALE}x for training)")

    if SCALING_METHOD == "absolute" or SCALING_METHOD == "absolute_log":
        print(f"  Original (abs change) Mean: {np.mean(y_original):.4f}")
        print(f"  Original (abs change) Std: {np.std(y_original):.4f}")
    else:
        print(f"  Original Mean: {np.mean(y_original):.6f}%")
        print(f"  Original Std: {np.std(y_original):.6f}%")
    print(f"  Scaled Mean: {np.mean(y):.4f}")
    print(f"  Scaled Std: {np.std(y):.4f}")
    print(f"  Scaled Min: {np.min(y):.4f}")
    print(f"  Scaled Max: {np.max(y):.4f}")
    print(f"  Positive values: {np.sum(y > 0) / len(y) * 100:.2f}%")
    print(f"  Negative values: {np.sum(y < 0) / len(y) * 100:.2f}%")

    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")

    return X, y, feature_cols, TARGET_SCALE, SCALING_METHOD, y_original


def train_lightgbm(X_train, y_train, n_estimators=200,
                   max_depth=6, num_leaves=31, learning_rate=0.1,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l1=0.1, lambda_l2=1.0, min_data_in_leaf=20,
                   min_gain_to_split=0.0, random_state=42, n_jobs=-1, verbose=-1,
                   sample_size=None):
    """Train a LightGBM model (CPU, multi-threading). Returns (model, None)."""
    print("\nTraining LightGBM model...")
    print(f"  Device: CPU (MPS tensor operations disabled for stability)")
    print(f"  Parallel jobs: {n_jobs} ({'all cores' if n_jobs == -1 else f'{n_jobs} cores'})")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  num_leaves: {num_leaves}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  feature_fraction: {feature_fraction}")
    print(f"  bagging_fraction: {bagging_fraction}")
    print(f"  lambda_l1: {lambda_l1}")
    print(f"  lambda_l2: {lambda_l2}")
    print(f"  min_data_in_leaf: {min_data_in_leaf}")
    print(f"  min_gain_to_split: {min_gain_to_split}")

    print("\nUsing raw features (LightGBM handles scaling internally)...")
    X_train_raw = X_train

    print("\nStarting LightGBM training...")
    print(f"  Building {n_estimators} boosting rounds...")
    start_time = time.time()

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        min_data_in_leaf=min_data_in_leaf,
        min_gain_to_split=min_gain_to_split,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        force_col_wise=True,
        boosting_type='gbdt',
        objective='regression',
        metric='rmse'
    )

    print("  Using validation set for early stopping...")
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
        X_train_raw, y_train, test_size=0.1, random_state=random_state, shuffle=False
    )

    if sample_size and sample_size < 10000:
        stopping_rounds = 50
        log_period = 25
    else:
        stopping_rounds = 100
        log_period = 50

    print(f"  Early stopping patience: {stopping_rounds} rounds")
    print(f"  Logging every: {log_period} rounds")

    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val_fit, y_val_fit)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True),
            lgb.log_evaluation(period=log_period)
        ]
    )

    y_train_pred_sample = model.predict(X_train_fit[:10000])
    print(f"\n  Training prediction sample statistics (first 10000):")
    print(f"    Predicted Mean: {np.mean(y_train_pred_sample):.6f}")
    print(f"    Predicted Std: {np.std(y_train_pred_sample):.6f}")
    print(f"    Predicted Min: {np.min(y_train_pred_sample):.6f}")
    print(f"    Predicted Max: {np.max(y_train_pred_sample):.6f}")
    print(f"    Actual target Mean: {np.mean(y_train_fit[:10000]):.6f}")
    print(f"    Actual target Std: {np.std(y_train_fit[:10000]):.6f}")
    print(f"    Prediction range: {np.max(y_train_pred_sample) - np.min(y_train_pred_sample):.6f}")
    print(f"    Actual range: {np.max(y_train_fit[:10000]) - np.min(y_train_fit[:10000]):.6f}")

    sample_r2 = r2_score(y_train_fit[:10000], y_train_pred_sample)
    print(f"    Sample R²: {sample_r2:.4f}")

    pred_std = np.std(y_train_pred_sample)
    actual_std = np.std(y_train_fit[:10000])
    if pred_std < actual_std * 0.1:
        print(f"  ⚠ WARNING: Predictions have very low variance!")
        print(f"    Prediction std ({pred_std:.6f}) is < 10% of actual std ({actual_std:.6f})")
        print(f"    Model may be predicting the mean instead of learning patterns")

    correlation = np.corrcoef(y_train_fit[:10000], y_train_pred_sample)[0, 1]
    print(f"    Prediction-Actual Correlation: {correlation:.4f}")
    if correlation < 0.1:
        print(f"  ⚠ WARNING: Very low correlation! Model is not learning patterns.")

    scaler = None
    elapsed_time = time.time() - start_time

    actual_trees = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None else model.n_estimators
    print(f"\n  Actual trees used: {actual_trees} (out of {n_estimators} requested)")
    if actual_trees < n_estimators:
        print(f"  ⚠ Early stopping triggered at iteration {actual_trees}")
        print(f"  This means the model stopped improving on validation set")
    else:
        print(f"  ✓ Model used all {n_estimators} trees (early stopping didn't trigger)")

    print(f"\n{'='*60}")
    print(f"✓ Model training complete!")
    print(f"{'='*60}")
    print(f"  Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  Average time per round: {elapsed_time/actual_trees:.4f} seconds")
    print(f"  Rounds per second: {actual_trees/elapsed_time:.2f}")

    return model, None


def scale_target(y_unscaled, target_scale, scaling_method, exp_max_arg=10.0):
    """Forward scale: original target -> model space. Inverse of unscale_target."""
    if scaling_method == "none":
        return np.asarray(y_unscaled)
    if scaling_method == "absolute":
        return np.asarray(y_unscaled)
    if scaling_method == "absolute_log":
        return np.log1p(np.asarray(y_unscaled) * target_scale)
    if scaling_method == "log":
        return np.sign(y_unscaled) * np.log1p(np.abs(y_unscaled) * target_scale)
    if scaling_method == "exponential":
        y_abs = np.clip(np.abs(y_unscaled) * target_scale, 0, exp_max_arg)
        return np.sign(y_unscaled) * (np.exp(y_abs) - 1.0)
    if scaling_method == "asinh":
        return np.sign(y_unscaled) * np.arcsinh(np.abs(y_unscaled) * target_scale)
    if scaling_method == "linear":
        return np.asarray(y_unscaled) * target_scale
    raise ValueError(f"Unknown scaling method: {scaling_method}")


def unscale_target(y_scaled, target_scale, scaling_method):
    """Unscale target from model space back to percentage change."""
    if scaling_method == "none":
        return y_scaled
    if scaling_method == "absolute":
        return np.asarray(y_scaled)
    if scaling_method == "absolute_log":
        return np.expm1(np.asarray(y_scaled)) / target_scale
    if scaling_method == "log":
        return np.sign(y_scaled) * (np.exp(np.abs(y_scaled)) - 1.0) / target_scale
    if scaling_method == "exponential":
        y_abs = np.abs(y_scaled)
        return np.sign(y_scaled) * np.log1p(y_abs) / target_scale
    if scaling_method == "asinh":
        return np.sign(y_scaled) * np.sinh(np.abs(y_scaled)) / target_scale
    if scaling_method == "linear":
        return np.asarray(y_scaled) / target_scale
    raise ValueError(f"Unknown scaling method: {scaling_method}")


def evaluate_model(model, scaler, X_test, y_test, target_scale=1.0, scaling_method="none", verbose=True):
    """Evaluate the model on test set. Returns (metrics dict, y_pred array)."""
    if verbose:
        print("\nEvaluating on test set...")

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_pred_scaled = model.predict(X_test_scaled)

    if scaling_method == "none":
        y_pred = y_pred_scaled
        y_test_unscaled = y_test
    elif scaling_method == "absolute":
        y_pred = np.asarray(y_pred_scaled)
        y_test_unscaled = np.asarray(y_test)
    elif scaling_method == "absolute_log":
        y_pred = np.expm1(y_pred_scaled) / target_scale
        y_test_unscaled = np.expm1(y_test) / target_scale
    elif scaling_method == "log":
        y_pred = np.sign(y_pred_scaled) * (np.exp(np.abs(y_pred_scaled)) - 1.0) / target_scale
        y_test_unscaled = np.sign(y_test) * (np.exp(np.abs(y_test)) - 1.0) / target_scale
    elif scaling_method == "exponential":
        y_pred_abs = np.abs(y_pred_scaled)
        y_test_abs = np.abs(y_test)
        y_pred = np.sign(y_pred_scaled) * np.log1p(y_pred_abs) / target_scale
        y_test_unscaled = np.sign(y_test) * np.log1p(y_test_abs) / target_scale
        if verbose:
            print(f"\n  Exponential scaling diagnostics:")
            print(f"    Predictions (transformed) - Mean: {np.mean(y_pred_scaled):.6f}, Std: {np.std(y_pred_scaled):.6f}")
            print(f"    Actual (transformed) - Mean: {np.mean(y_test):.6f}, Std: {np.std(y_test):.6f}")
            print(f"    Predictions (untransformed) - Mean: {np.mean(y_pred):.6f}%, Std: {np.std(y_pred):.6f}%")
            print(f"    Actual (untransformed) - Mean: {np.mean(y_test_unscaled):.6f}%, Std: {np.std(y_test_unscaled):.6f}%")
            if np.std(y_pred_scaled) < np.std(y_test) * 0.1:
                print(f"    ⚠ WARNING: Predictions in transformed space have very low variance!")
    elif scaling_method == "asinh":
        y_pred = np.sign(y_pred_scaled) * np.sinh(np.abs(y_pred_scaled)) / target_scale
        y_test_unscaled = np.sign(y_test) * np.sinh(np.abs(y_test)) / target_scale
    elif scaling_method == "linear":
        y_pred = y_pred_scaled / target_scale
        y_test_unscaled = y_test / target_scale
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    mse = mean_squared_error(y_test_unscaled, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred)
    r2 = r2_score(y_test_unscaled, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = np.where(
            np.abs(y_test_unscaled) > 1e-10,
            np.abs((y_test_unscaled - y_pred) / y_test_unscaled),
            0.0
        )
    mape = np.mean(pct_errors) * 100

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

    if verbose:
        print(f"\nTest Set Results (Percentage Change - Unscaled):")
        print(f"  MSE: {mse:.6f} (%²)")
        print(f"  RMSE: {rmse:.6f}%")
        print(f"  MAE: {mae:.6f}%")
        print(f"  R²: {r2:.4f}")
        if not np.isnan(mape):
            print(f"  MAPE: {mape:.2f}%")

    return metrics, y_pred


def get_feature_importance(model, feature_cols, top_n=20):
    """
    Get feature importances (split and gain). Returns DataFrame with columns
    feature, importance_split, importance_gain. For LightGBM both are computed
    from the booster; for other models only importance_split is set (gain=NaN).
    """
    split_imp = model.feature_importances_  # default is split for LGBM
    if hasattr(model, "booster_") and model.booster_ is not None:
        gain_imp = model.booster_.feature_importance(importance_type="gain")
    else:
        gain_imp = np.full(len(feature_cols), np.nan)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_split": split_imp,
        "importance_gain": gain_imp,
    })
    # Sort by gain (fall back to split if gain is all NaN, e.g. non-LightGBM)
    sort_col = "importance_gain" if pd.Series(gain_imp).notna().any() else "importance_split"
    importance_df = importance_df.sort_values(sort_col, ascending=False)
    print(f"\nTop {top_n} Most Important Features (by {sort_col.replace('importance_', '')}):")
    print(importance_df.head(top_n).to_string(index=False))
    return importance_df


def plot_predictions_vs_actual(y_actual, y_pred, n_steps=100, save_path=None, target_label=None):
    """Plot actual vs predicted. Returns path to saved plot."""
    y_axis_label = target_label if target_label is not None else "Target value"
    print(f"\nGenerating plot of actual vs predicted ({y_axis_label}) for {n_steps} consecutive steps...")

    n_steps = min(n_steps, len(y_actual), len(y_pred))
    y_actual_plot = np.asarray(y_actual[:n_steps])
    y_pred_plot = np.asarray(y_pred[:n_steps])

    fig, ax = plt.subplots(figsize=(14, 8))
    steps = np.arange(1, n_steps + 1)

    ax.plot(steps, y_actual_plot, label='Actual (unscaled)',
            color='#2E86AB', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax.plot(steps, y_pred_plot, label='Predicted (unscaled)',
            color='#A23B72', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax.fill_between(steps, y_actual_plot, y_pred_plot,
                    alpha=0.2, color='gray', label='Prediction Error')

    ax.set_xlabel('Step (Consecutive Rows)', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_axis_label, fontsize=12, fontweight='bold')
    ax.set_title(f'LightGBM: Actual vs Predicted (unscaled)\n(First {n_steps} Consecutive Steps)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    if np.all(y_actual_plot >= 0) and np.all(y_pred_plot >= 0):
        ax.set_ylim(bottom=0)

    mse = mean_squared_error(y_actual_plot, y_pred_plot)
    mae = mean_absolute_error(y_actual_plot, y_pred_plot)
    r2 = r2_score(y_actual_plot, y_pred_plot)
    stats_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).parent / 'lightgbm_predictions_vs_actual.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()
    return save_path


def run_correlation_analysis(X, y_original, feature_cols, save_path):
    """
    Compute feature-target correlations, print top 20, save CSV, and print summary stats.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_original: Original unscaled target, aligned with X
        feature_cols: List of feature names
        save_path: Path to save correlations CSV

    Returns:
        correlations: List of (feature_name, correlation) sorted by |correlation| descending
    """
    print("\n" + "=" * 60)
    print("Feature-Target Correlation Analysis")
    print("=" * 60)

    correlations = []
    for i, feature_name in enumerate(feature_cols):
        feature_values = X[:, i]
        valid_corr_mask = ~(np.isnan(feature_values) | np.isnan(y_original))
        if np.sum(valid_corr_mask) > 10:
            try:
                corr = np.corrcoef(feature_values[valid_corr_mask], y_original[valid_corr_mask])[0, 1]
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations.append((feature_name, corr))
            except Exception:
                pass

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 20 Features by Absolute Correlation with Target:")
    print(f"{'Feature':<50} {'Correlation':>12} {'Abs Corr':>12}")
    print("-" * 76)
    for feature_name, corr in correlations[:20]:
        abs_corr = abs(corr)
        direction = "↑" if corr > 0 else "↓"
        print(f"{feature_name:<50} {corr:>12.6f} {abs_corr:>12.6f} {direction}")

    corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df.to_csv(save_path, index=False)
    print(f"\n  Full correlation analysis saved to: {save_path}")

    abs_corrs = [abs(corr) for _, corr in correlations]
    print("\n  Correlation Statistics:")
    print(f"    Mean absolute correlation: {np.mean(abs_corrs):.6f}")
    print(f"    Max absolute correlation: {np.max(abs_corrs):.6f}")
    print(f"    Features with |corr| > 0.1: {sum(1 for c in abs_corrs if c > 0.1)}")
    print(f"    Features with |corr| > 0.05: {sum(1 for c in abs_corrs if c > 0.05)}")
    print(f"    Features with |corr| < 0.01: {sum(1 for c in abs_corrs if c < 0.01)}")

    return correlations


def run_top_n_feature_search(
    X,
    y,
    correlations,
    feature_cols,
    target_scale,
    scaling_method,
    base_train_kwargs,
    top_n_candidates,
    n_folds=5,
    sample_size=None,
):
    """
    Grid search over number of top features (by |correlation| with target).
    Runs k-fold CV for each candidate N and picks the one with best mean R².

    Args:
        X: Full feature matrix
        y: Target (scaled)
        correlations: List of (feature_name, correlation) sorted by |corr| desc (from run_correlation_analysis)
        feature_cols: List of feature names (same order as X columns)
        target_scale: Scale factor for target
        scaling_method: Scaling method for unscale/scale_target
        base_train_kwargs: Dict of train_lightgbm kwargs (lambda_l1/l2 overridden when N is not None)
        top_n_candidates: List of int or None, e.g. [None, 10, 20, 30] (None = all features)
        n_folds: Number of folds for each CV run
        sample_size: Optional sample size for train_lightgbm

    Returns:
        best_n: Chosen N (None = use all features)
        best_r2: Mean CV R² for best_n
        results: List of (n, r2) for each candidate tried
    """
    print("\n" + "=" * 60)
    print("Grid search: optimizing number of top features (by |correlation|)")
    print("=" * 60)
    best_n = None
    best_r2 = -np.inf
    results = []
    for n in top_n_candidates:
        if n is not None and len(correlations) < n:
            continue
        if n is not None:
            top_names = [name for name, _ in correlations[:n]]
            col_indices = [feature_cols.index(name) for name in top_names]
            X_sub = X[:, col_indices]
            kwargs = {**base_train_kwargs, "lambda_l1": 0.0, "lambda_l2": 0.05}
        else:
            X_sub = X
            kwargs = base_train_kwargs
        metrics_mean, _, _, _, _ = run_kfold_cv(
            X_sub,
            y,
            n_folds,
            target_scale,
            scaling_method,
            sample_size=sample_size,
            report_weak_folds=False,
            quiet=True,
            **kwargs,
        )
        r2 = metrics_mean["r2"]
        results.append((n, r2))
        label = "all" if n is None else str(n)
        print(f"  TOP_N={label:>3} -> CV R² = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_n = n
    best_label = "all" if best_n is None else str(best_n)
    print(f"\n  Best: TOP_N={best_label} (CV R² = {best_r2:.4f})")
    return best_n, best_r2, results


def apply_top_n_features(X, correlations, feature_cols, top_n):
    """
    Subset X and feature_cols to top top_n features by |correlation|.
    If top_n is None or len(correlations) < top_n, returns X and feature_cols unchanged.

    Returns:
        X_sub: Feature matrix (possibly subset)
        feature_cols_sub: List of feature names (possibly subset)
    """
    if top_n is None or len(correlations) < top_n:
        return X, feature_cols
    top_names = [name for name, _ in correlations[:top_n]]
    col_indices = [feature_cols.index(name) for name in top_names]
    return X[:, col_indices], top_names


def run_kfold_cv(
    X, y, n_folds, target_scale, scaling_method, sample_size=None,
    report_weak_folds=True, weak_r2_threshold=0.7, quiet=False, **train_kwargs
):
    """
    Run k-fold cross-validation: train per fold, collect OOF predictions, report mean±std.

    Args:
        X: Feature matrix
        y: Target (scaled)
        n_folds: Number of folds
        target_scale: Scale factor for target
        scaling_method: Scaling method name for unscale/scale_target
        sample_size: Optional sample size (for train_lightgbm)
        report_weak_folds: If True, print warning for folds with R² below threshold
        weak_r2_threshold: R² threshold for weak-fold warning
        quiet: If True, skip all print output (for grid search over meta-params)
        **train_kwargs: Passed to train_lightgbm (n_estimators, max_depth, etc.)

    Returns:
        metrics_mean: Dict of mean metric across folds
        metrics_std: Dict of std of each metric
        oof_y_true: Concatenated unscaled actual (out-of-fold)
        oof_y_pred: Concatenated unscaled predicted (out-of-fold)
        cv_metrics_list: List of per-fold metric dicts
    """
    if not quiet:
        print("\n" + "=" * 60)
        print(f"{n_folds}-Fold Cross-Validation Evaluation")
        print("=" * 60)
    kf = KFold(n_splits=n_folds, shuffle=False)
    cv_metrics_list = []
    oof_y_true_list = []
    oof_y_pred_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model_fold, _ = train_lightgbm(
            X_tr, y_tr,
            sample_size=sample_size,
            **train_kwargs
        )
        metrics_fold, y_pred_fold = evaluate_model(
            model_fold, None, X_te, y_te,
            target_scale=target_scale, scaling_method=scaling_method, verbose=False
        )
        cv_metrics_list.append(metrics_fold)
        y_te_unscaled = unscale_target(y_te, target_scale, scaling_method)
        oof_y_true_list.append(y_te_unscaled)
        oof_y_pred_list.append(y_pred_fold)
        if not quiet:
            print(f"  Fold {fold + 1}/{n_folds} - R²: {metrics_fold['r2']:.4f}, RMSE: {metrics_fold['rmse']:.6f}%")

    oof_y_true = np.concatenate(oof_y_true_list)
    oof_y_pred = np.concatenate(oof_y_pred_list)

    fold_r2s = [m['r2'] for m in cv_metrics_list]
    if not quiet:
        if report_weak_folds:
            weak_folds = [i + 1 for i, r2 in enumerate(fold_r2s) if r2 < weak_r2_threshold]
            if weak_folds:
                print(f"\n  ⚠ Weak folds (R² < {weak_r2_threshold}): {weak_folds} — likely different regime in middle of series.")
                print("    Consider: more folds (e.g. N_FOLDS=10), stronger regularization, or time-based splits.")
        print(f"  Per-fold R²: {[f'{r:.3f}' for r in fold_r2s]}")

    if not quiet:
        print("\nScaling verification (actual vs predicted both unscaled with same transform):")
        exp_max_arg = 10.0
        if scaling_method in ("exponential", "log", "linear", "absolute_log", "asinh"):
            sample = min(500, len(oof_y_true))
            idx = np.linspace(0, len(oof_y_true) - 1, sample, dtype=int)
            y_sample = oof_y_true[idx]
            y_roundtrip = unscale_target(
                scale_target(y_sample, target_scale, scaling_method, exp_max_arg=exp_max_arg),
                target_scale, scaling_method
            )
            rt_err = np.abs(y_roundtrip - y_sample)
            print(f"  Round-trip (actual -> scale -> unscale): max error = {np.max(rt_err):.2e}, mean error = {np.mean(rt_err):.2e}")
        print(f"  OOF actual  (unscaled): mean = {np.mean(oof_y_true):.6f}, std = {np.std(oof_y_true):.6f}")
        print(f"  OOF predict (unscaled): mean = {np.mean(oof_y_pred):.6f}, std = {np.std(oof_y_pred):.6f}")
        if np.std(oof_y_pred) < np.std(oof_y_true) * 0.1:
            print("  Note: Prediction std << actual std => model outputs near-constant in scaled space (same unscaling applied to both).")

    metrics_mean = {k: np.mean([m[k] for m in cv_metrics_list]) for k in cv_metrics_list[0]}
    metrics_std = {k: np.std([m[k] for m in cv_metrics_list]) for k in cv_metrics_list[0]}
    if not quiet:
        print(f"\n{n_folds}-Fold CV Results (mean ± std):")
        print(f"  MSE:   {metrics_mean['mse']:.6f} ± {metrics_std['mse']:.6f} (%²)")
        print(f"  RMSE:  {metrics_mean['rmse']:.6f} ± {metrics_std['rmse']:.6f}%")
        print(f"  MAE:   {metrics_mean['mae']:.6f} ± {metrics_std['mae']:.6f}%")
        print(f"  R²:    {metrics_mean['r2']:.4f} ± {metrics_std['r2']:.4f}")
        if not np.isnan(metrics_mean['mape']):
            print(f"  MAPE:  {metrics_mean['mape']:.2f} ± {metrics_std['mape']:.2f}%")

    return metrics_mean, metrics_std, oof_y_true, oof_y_pred, cv_metrics_list


def write_metrics_file(path, title, n_folds, metrics_mean, metrics_std, train_metrics):
    """
    Write metrics summary to a text file (CV mean±std and final model on full data).

    Args:
        path: Output file path
        title: First line (e.g. "LightGBM Model (highest) Metrics - PctChange_CloseToHigh")
        n_folds: Number of CV folds
        metrics_mean: Dict from run_kfold_cv
        metrics_std: Dict from run_kfold_cv
        train_metrics: Dict from evaluate_model on full data
    """
    with open(path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{n_folds}-FOLD CROSS-VALIDATION (mean ± std):\n")
        f.write("-" * 60 + "\n")
        f.write(f"MSE:   {metrics_mean['mse']:.6f} ± {metrics_std['mse']:.6f} (%²)\n")
        f.write(f"RMSE:  {metrics_mean['rmse']:.6f} ± {metrics_std['rmse']:.6f}%\n")
        f.write(f"MAE:   {metrics_mean['mae']:.6f} ± {metrics_std['mae']:.6f}%\n")
        f.write(f"R²:    {metrics_mean['r2']:.4f} ± {metrics_std['r2']:.4f}\n")
        if not np.isnan(metrics_mean['mape']):
            f.write(f"MAPE:  {metrics_mean['mape']:.2f} ± {metrics_std['mape']:.2f}%\n")
        f.write("\n")
        f.write("FINAL MODEL ON FULL DATA (training fit):\n")
        f.write("-" * 60 + "\n")
        f.write(f"MSE: {train_metrics['mse']:.6f} (%²)\n")
        f.write(f"RMSE: {train_metrics['rmse']:.6f}%\n")
        f.write(f"MAE: {train_metrics['mae']:.6f}%\n")
        f.write(f"R²: {train_metrics['r2']:.4f}\n")
        if not np.isnan(train_metrics['mape']):
            f.write(f"MAPE: {train_metrics['mape']:.2f}%\n")
