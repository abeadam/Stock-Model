"""
Futures Price Analysis - ES (E-mini S&P 500) Data with Technical Indicators

Calculates and plots:
- Bollinger Bands: 20 period (1, 2, 3 std dev), 50 period (1, 2, 3 std dev)
- MFI (Money Flow Index): 14 and 28 periods
- RSI (Relative Strength Index): 14 and 28 periods
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import Optional
from datetime import datetime
import pytz


def load_es_data(data_dir: str = 'daily_data') -> pd.DataFrame:
    """Load ES (E-mini S&P 500) data from CSV files"""
    print("Loading ES data files...")
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve the data directory relative to script location
        data_dir = os.path.join(script_dir, data_dir)
        # Normalize the path (handles .. and .)
        data_dir = os.path.normpath(data_dir)
    
    # Get all ES files
    es_files = glob.glob(os.path.join(data_dir, "*_ES.txt"))
    
    if not es_files:
        raise ValueError(f"No ES files found in {data_dir}. Checked path: {os.path.abspath(data_dir)}")
    
    all_data = []
    
    for es_file in sorted(es_files):
        try:
            df = pd.read_csv(es_file)
            # Ensure we have required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {es_file} missing required columns, skipping")
                continue
            
            # ES files should have Volume, but check just in case
            if 'Volume' not in df.columns:
                print(f"Warning: {es_file} missing Volume column, will use dummy volume for MFI")
                df['Volume'] = 0  # Default volume
            
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {es_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No ES data files could be loaded!")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(combined_df)} total rows from {len(all_data)} files")
    
    return combined_df


def load_vxm_data(data_dir: str = 'daily_data') -> pd.DataFrame:
    """Load VXM (VX Mid-Term) data from CSV files"""
    print("Loading VXM data files...")
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve the data directory relative to script location
        data_dir = os.path.join(script_dir, data_dir)
        # Normalize the path (handles .. and .)
        data_dir = os.path.normpath(data_dir)
    
    # Get all VXM files
    vxm_files = glob.glob(os.path.join(data_dir, "*_VXM.txt"))
    
    if not vxm_files:
        raise ValueError(f"No VXM files found in {data_dir}. Checked path: {os.path.abspath(data_dir)}")
    
    all_data = []
    
    for vxm_file in sorted(vxm_files):
        try:
            df = pd.read_csv(vxm_file)
            # Ensure we have required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {vxm_file} missing required columns, skipping")
                continue
            
            # VXM files should have Volume, but check just in case
            if 'Volume' not in df.columns:
                print(f"Warning: {vxm_file} missing Volume column, will use dummy volume")
                df['Volume'] = 0  # Default volume
            
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {vxm_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No VXM data files could be loaded!")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(combined_df)} total rows from {len(all_data)} files")
    
    return combined_df


def load_stock_data(ticker: str, data_dir: str = 'daily_data') -> pd.DataFrame:
    """Load stock data from CSV files for a given ticker"""
    print(f"Loading {ticker} data files...")
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve the data directory relative to script location
        data_dir = os.path.join(script_dir, data_dir)
        # Normalize the path (handles .. and .)
        data_dir = os.path.normpath(data_dir)
    
    # Get all files for this ticker
    stock_files = glob.glob(os.path.join(data_dir, f"*_{ticker}.txt"))
    
    if not stock_files:
        print(f"Warning: No {ticker} files found in {data_dir}")
        return pd.DataFrame()  # Return empty dataframe
    
    all_data = []
    
    for stock_file in sorted(stock_files):
        try:
            df = pd.read_csv(stock_file)
            # Ensure we have required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {stock_file} missing required columns, skipping")
                continue
            
            # Stock files should have Volume, but check just in case
            if 'Volume' not in df.columns:
                print(f"Warning: {stock_file} missing Volume column, will use dummy volume")
                df['Volume'] = 0  # Default volume
            
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {stock_file}: {e}")
            continue
    
    if not all_data:
        print(f"Warning: No {ticker} data files could be loaded!")
        return pd.DataFrame()  # Return empty dataframe
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(combined_df)} total rows from {len(all_data)} {ticker} files")
    
    return combined_df


def process_stock_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Calculate all technical indicators for a stock dataframe"""
    if df.empty:
        return df
    
    print(f"\nCalculating indicators for {ticker}...")
    
    # Calculate Bollinger Bands for 10 period (1, 2, 3 std dev)
    df = calculate_bollinger_bands(df, period=10, std_dev=1.0)
    df = calculate_bollinger_bands(df, period=10, std_dev=2.0)
    df = calculate_bollinger_bands(df, period=10, std_dev=3.0)
    
    # Calculate Bollinger Bands for 20 period (1, 2, 3 std dev)
    df = calculate_bollinger_bands(df, period=20, std_dev=1.0)
    df = calculate_bollinger_bands(df, period=20, std_dev=2.0)
    df = calculate_bollinger_bands(df, period=20, std_dev=3.0)
    
    # Calculate Bollinger Bands for 50 period (1, 2, 3 std dev)
    df = calculate_bollinger_bands(df, period=50, std_dev=1.0)
    df = calculate_bollinger_bands(df, period=50, std_dev=2.0)
    df = calculate_bollinger_bands(df, period=50, std_dev=3.0)
    
    # Calculate EMA for 10, 20, and 50 periods
    df = calculate_ema(df, period=10)
    df = calculate_ema(df, period=20)
    df = calculate_ema(df, period=50)
    
    # Calculate RSI for 7, 14, and 28 periods
    df = calculate_rsi(df, period=7)
    df = calculate_rsi(df, period=14)
    df = calculate_rsi(df, period=28)
    
    # Calculate MFI for 7, 14, and 28 periods
    df = calculate_mfi(df, period=7)
    df = calculate_mfi(df, period=14)
    df = calculate_mfi(df, period=28)
    
    # Calculate current percent changes
    df = calculate_current_percent_changes(df)
    
    # Skip first 50 values (to avoid NaN from rolling windows)
    if len(df) > 50:
        df = df.iloc[50:].reset_index(drop=True)
    else:
        # If not enough data, return empty dataframe
        print(f"Warning: {ticker} has insufficient data ({len(df)} rows), skipping")
        return pd.DataFrame()
    
    return df


def calculate_bollinger_bands(df: pd.DataFrame, period: int, std_dev: float, 
                              price_col: str = 'Close') -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    df = df.copy()
    
    # Calculate moving average
    ma_col = f'BB_{period}_MA'
    df[ma_col] = df[price_col].rolling(window=period).mean()
    
    # Calculate standard deviation
    std_col = f'BB_{period}_STD'
    df[std_col] = df[price_col].rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_col = f'BB_{period}_{int(std_dev)}_Upper'
    lower_col = f'BB_{period}_{int(std_dev)}_Lower'
    
    df[upper_col] = df[ma_col] + (df[std_col] * std_dev)
    df[lower_col] = df[ma_col] - (df[std_col] * std_dev)
    
    return df


def calculate_ema(df: pd.DataFrame, period: int, price_col: str = 'Close') -> pd.DataFrame:
    """Calculate Exponential Moving Average (EMA)"""
    df = df.copy()
    
    ema_col = f'EMA_{period}'
    df[ema_col] = df[price_col].ewm(span=period, adjust=False).mean()
    
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'Close') -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI)"""
    df = df.copy()
    
    # Calculate price changes
    delta = df[price_col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss using exponential moving average
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_col = f'RSI_{period}'
    df[rsi_col] = 100 - (100 / (1 + rs))
    
    return df


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Money Flow Index (MFI)"""
    df = df.copy()
    
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * df['Volume']
    
    # Determine positive and negative money flow
    money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    # Calculate 14-period sums
    positive_flow_sum = money_flow_positive.rolling(window=period).sum()
    negative_flow_sum = money_flow_negative.rolling(window=period).sum()
    
    # Calculate MFI
    mfi_col = f'MFI_{period}'
    
    # Handle division by zero: if negative_flow_sum is zero, set MFI to 100
    # Otherwise calculate: 100 - (100 / (1 + money_flow_ratio))
    # where money_flow_ratio = positive_flow_sum / negative_flow_sum
    df[mfi_col] = np.where(
        negative_flow_sum == 0,
        100.0,  # No negative flow means all positive flow = maximum MFI
        100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    )
    
    return df


def calculate_forward_percent_changes(df: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """Calculate percent change from current close to max high and min low of next N periods"""
    df = df.copy()
    
    # Initialize arrays for forward-looking calculations
    max_highs = np.full(len(df), np.nan)
    min_lows = np.full(len(df), np.nan)
    
    # For each row, look forward at the next N periods
    for i in range(len(df)):
        # Get the next N periods (i+1 to i+periods)
        end_idx = min(i + periods + 1, len(df))
        if i + 1 < len(df):
            # Get the High and Low values for the next N periods
            future_highs = df['High'].iloc[i+1:end_idx].values
            future_lows = df['Low'].iloc[i+1:end_idx].values
            
            if len(future_highs) > 0:
                max_highs[i] = np.max(future_highs)
            else:
                max_highs[i] = df['High'].iloc[i]
            
            if len(future_lows) > 0:
                min_lows[i] = np.min(future_lows)
            else:
                min_lows[i] = df['Low'].iloc[i]
        else:
            # For the last row, use current values
            max_highs[i] = df['High'].iloc[i]
            min_lows[i] = df['Low'].iloc[i]
    
    # Calculate percent change from current Close to max high (next N periods)
    df['PctChange_ToMaxHigh_5'] = ((max_highs - df['Close'].values) / df['Close'].values) * 100
    
    # Calculate percent change from current Close to min low (next N periods)
    df['PctChange_ToMinLow_5'] = ((min_lows - df['Close'].values) / df['Close'].values) * 100
    
    return df


def calculate_current_percent_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percent change from current Close to High and Low"""
    df = df.copy()
    
    # Percent change from Close to High: (High - Close) / Close
    df['PctChange_CloseToHigh'] = ((df['High'] - df['Close']) / df['Close']) * 100
    
    # Percent change from Close to Low: (Low - Close) / Close
    df['PctChange_CloseToLow'] = ((df['Low'] - df['Close']) / df['Close']) * 100
    
    return df


def calculate_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate hours from start of formal trading (9 AM ET) and overnight trading (6 PM ET)"""
    df = df.copy()
    
    # Convert Date to datetime if it's not already
    if df['Date'].dtype != 'datetime64[ns]':
        try:
            df['DateTime'] = pd.to_datetime(df['Date'], unit='s')
        except:
            df['DateTime'] = pd.to_datetime(df['Date'])
    else:
        df['DateTime'] = df['Date']
    
    # Set Eastern timezone
    eastern = pytz.timezone('US/Eastern')
    
    # Convert to Eastern time
    if df['DateTime'].dt.tz is None:
        # Assume UTC if no timezone info
        df['DateTime_ET'] = pd.to_datetime(df['DateTime']).dt.tz_localize('UTC').dt.tz_convert(eastern)
    else:
        df['DateTime_ET'] = df['DateTime'].dt.tz_convert(eastern)
    
    # Initialize columns
    df['Hours_From_Formal_Trading'] = np.nan
    df['Hours_From_Overnight_Trading'] = np.nan
    
    # For each row, calculate hours from trading start
    for idx in df.index:
        time_et = df.at[idx, 'DateTime_ET']
        date_only = time_et.date()
        
        # Formal trading starts at 9:00 AM Eastern
        formal_start_time = datetime.strptime('09:00:00', '%H:%M:%S').time()
        formal_start = eastern.localize(datetime.combine(date_only, formal_start_time))
        
        # Overnight trading starts at 6:00 PM Eastern (previous day)
        # If current time is before 6 PM, use previous day's 6 PM
        overnight_start_time = datetime.strptime('18:00:00', '%H:%M:%S').time()
        overnight_start = eastern.localize(datetime.combine(date_only, overnight_start_time))
        if time_et < overnight_start:
            # Use previous day's 6 PM
            from datetime import timedelta
            prev_day = date_only - timedelta(days=1)
            overnight_start = eastern.localize(datetime.combine(prev_day, overnight_start_time))
        
        # Calculate hours from formal trading start
        hours_from_formal = (time_et - formal_start).total_seconds() / 3600.0
        df.at[idx, 'Hours_From_Formal_Trading'] = hours_from_formal
        
        # Calculate hours from overnight trading start
        hours_from_overnight = (time_et - overnight_start).total_seconds() / 3600.0
        df.at[idx, 'Hours_From_Overnight_Trading'] = hours_from_overnight
    
    return df


def plot_results(df: pd.DataFrame, save_path: Optional[str] = None, last_n_points: int = 400):
    """Plot all calculated indicators (only last N points)"""
    
    # Work on a copy to avoid modifying the original dataframe
    df_plot = df.tail(last_n_points).copy()
    
    # Convert Date to datetime if it's not already (only in the copy)
    if df_plot['Date'].dtype != 'datetime64[ns]':
        # Try to convert from timestamp
        try:
            df_plot['Date'] = pd.to_datetime(df_plot['Date'], unit='s')
        except:
            df_plot['Date'] = pd.to_datetime(df_plot['Date'])
    
    # Create figure with subplots (now 7 rows x 2 cols to include all new indicators)
    fig = plt.figure(figsize=(20, 28))
    
    # Plot 1: Price with Bollinger Bands 20
    ax1 = plt.subplot(7, 2, 1)
    ax1.plot(df_plot['Date'], df_plot['Close'], label='Close Price', linewidth=1.5, color='black')
    ax1.plot(df_plot['Date'], df_plot['BB_20_MA'], label='BB 20 MA', linewidth=1, color='blue', linestyle='--')
    ax1.plot(df_plot['Date'], df_plot['BB_20_1_Upper'], label='BB 20 Upper (1σ)', linewidth=1, color='red', alpha=0.7)
    ax1.plot(df_plot['Date'], df_plot['BB_20_1_Lower'], label='BB 20 Lower (1σ)', linewidth=1, color='red', alpha=0.7)
    ax1.plot(df_plot['Date'], df_plot['BB_20_2_Upper'], label='BB 20 Upper (2σ)', linewidth=1, color='orange', alpha=0.7)
    ax1.plot(df_plot['Date'], df_plot['BB_20_2_Lower'], label='BB 20 Lower (2σ)', linewidth=1, color='orange', alpha=0.7)
    ax1.plot(df_plot['Date'], df_plot['BB_20_3_Upper'], label='BB 20 Upper (3σ)', linewidth=1, color='purple', alpha=0.7)
    ax1.plot(df_plot['Date'], df_plot['BB_20_3_Lower'], label='BB 20 Lower (3σ)', linewidth=1, color='purple', alpha=0.7)
    ax1.fill_between(df_plot['Date'], df_plot['BB_20_1_Lower'], df_plot['BB_20_1_Upper'], alpha=0.1, color='red')
    ax1.fill_between(df_plot['Date'], df_plot['BB_20_2_Lower'], df_plot['BB_20_2_Upper'], alpha=0.1, color='orange')
    ax1.fill_between(df_plot['Date'], df_plot['BB_20_3_Lower'], df_plot['BB_20_3_Upper'], alpha=0.1, color='purple')
    ax1.set_title(f'Price with Bollinger Bands (20 period) - Last {last_n_points} points', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price with Bollinger Bands 50
    ax2 = plt.subplot(7, 2, 2)
    ax2.plot(df_plot['Date'], df_plot['Close'], label='Close Price', linewidth=1.5, color='black')
    ax2.plot(df_plot['Date'], df_plot['BB_50_MA'], label='BB 50 MA', linewidth=1, color='blue', linestyle='--')
    ax2.plot(df_plot['Date'], df_plot['BB_50_1_Upper'], label='BB 50 Upper (1σ)', linewidth=1, color='red', alpha=0.7)
    ax2.plot(df_plot['Date'], df_plot['BB_50_1_Lower'], label='BB 50 Lower (1σ)', linewidth=1, color='red', alpha=0.7)
    ax2.plot(df_plot['Date'], df_plot['BB_50_2_Upper'], label='BB 50 Upper (2σ)', linewidth=1, color='orange', alpha=0.7)
    ax2.plot(df_plot['Date'], df_plot['BB_50_2_Lower'], label='BB 50 Lower (2σ)', linewidth=1, color='orange', alpha=0.7)
    ax2.plot(df_plot['Date'], df_plot['BB_50_3_Upper'], label='BB 50 Upper (3σ)', linewidth=1, color='purple', alpha=0.7)
    ax2.plot(df_plot['Date'], df_plot['BB_50_3_Lower'], label='BB 50 Lower (3σ)', linewidth=1, color='purple', alpha=0.7)
    ax2.fill_between(df_plot['Date'], df_plot['BB_50_1_Lower'], df_plot['BB_50_1_Upper'], alpha=0.1, color='red')
    ax2.fill_between(df_plot['Date'], df_plot['BB_50_2_Lower'], df_plot['BB_50_2_Upper'], alpha=0.1, color='orange')
    ax2.fill_between(df_plot['Date'], df_plot['BB_50_3_Lower'], df_plot['BB_50_3_Upper'], alpha=0.1, color='purple')
    ax2.set_title(f'Price with Bollinger Bands (50 period) - Last {last_n_points} points', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI 7
    ax3 = plt.subplot(7, 2, 3)
    ax3.plot(df_plot['Date'], df_plot['RSI_7'], label='RSI 7', linewidth=1.5, color='purple')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax3.fill_between(df_plot['Date'], 30, 70, alpha=0.1, color='yellow')
    ax3.set_title('RSI (7 period)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RSI 14
    ax4 = plt.subplot(7, 2, 4)
    ax4.plot(df_plot['Date'], df_plot['RSI_14'], label='RSI 14', linewidth=1.5, color='blue')
    ax4.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax4.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax4.fill_between(df_plot['Date'], 30, 70, alpha=0.1, color='yellow')
    ax4.set_title('RSI (14 period)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('RSI')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: RSI 28
    ax5 = plt.subplot(7, 2, 5)
    ax5.plot(df_plot['Date'], df_plot['RSI_28'], label='RSI 28', linewidth=1.5, color='blue')
    ax5.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax5.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax5.fill_between(df_plot['Date'], 30, 70, alpha=0.1, color='yellow')
    ax5.set_title('RSI (28 period)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('RSI')
    ax5.set_ylim(0, 100)
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: MFI 7
    ax6 = plt.subplot(7, 2, 6)
    ax6.plot(df_plot['Date'], df_plot['MFI_7'], label='MFI 7', linewidth=1.5, color='purple')
    ax6.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax6.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax6.fill_between(df_plot['Date'], 20, 80, alpha=0.1, color='yellow')
    ax6.set_title('Money Flow Index (7 period)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('MFI')
    ax6.set_ylim(0, 100)
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: MFI 14
    ax7 = plt.subplot(7, 2, 7)
    ax7.plot(df_plot['Date'], df_plot['MFI_14'], label='MFI 14', linewidth=1.5, color='green')
    ax7.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax7.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax7.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax7.fill_between(df_plot['Date'], 20, 80, alpha=0.1, color='yellow')
    ax7.set_title('Money Flow Index (14 period)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('MFI')
    ax7.set_ylim(0, 100)
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: MFI 28
    ax8 = plt.subplot(7, 2, 8)
    ax8.plot(df_plot['Date'], df_plot['MFI_28'], label='MFI 28', linewidth=1.5, color='green')
    ax8.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax8.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax8.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
    ax8.fill_between(df_plot['Date'], 20, 80, alpha=0.1, color='yellow')
    ax8.set_title('Money Flow Index (28 period)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('MFI')
    ax8.set_ylim(0, 100)
    ax8.legend(loc='best', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Combined RSI comparison (7, 14, 28)
    ax9 = plt.subplot(7, 2, 9)
    ax9.plot(df_plot['Date'], df_plot['RSI_7'], label='RSI 7', linewidth=1.5, color='purple', alpha=0.7)
    ax9.plot(df_plot['Date'], df_plot['RSI_14'], label='RSI 14', linewidth=1.5, color='blue', alpha=0.7)
    ax9.plot(df_plot['Date'], df_plot['RSI_28'], label='RSI 28', linewidth=1.5, color='red', alpha=0.7)
    ax9.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax9.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax9.set_title('RSI Comparison (7 vs 14 vs 28)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Date')
    ax9.set_ylabel('RSI')
    ax9.set_ylim(0, 100)
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Combined MFI comparison (7, 14, 28)
    ax10 = plt.subplot(7, 2, 10)
    ax10.plot(df_plot['Date'], df_plot['MFI_7'], label='MFI 7', linewidth=1.5, color='purple', alpha=0.7)
    ax10.plot(df_plot['Date'], df_plot['MFI_14'], label='MFI 14', linewidth=1.5, color='blue', alpha=0.7)
    ax10.plot(df_plot['Date'], df_plot['MFI_28'], label='MFI 28', linewidth=1.5, color='red', alpha=0.7)
    ax10.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    ax10.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    ax10.set_title('MFI Comparison (7 vs 14 vs 28)', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Date')
    ax10.set_ylabel('MFI')
    ax10.set_ylim(0, 100)
    ax10.legend(loc='best', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Percent Change to Max High (next 5 periods)
    ax11 = plt.subplot(7, 2, 11)
    ax11.plot(df_plot['Date'], df_plot['PctChange_ToMaxHigh_5'], label='% Change to Max High (next 5)', 
             linewidth=1.5, color='green', alpha=0.8)
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax11.set_title('Percent Change: Current Close to Max High (next 5 periods)', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Date')
    ax11.set_ylabel('Percent Change (%)')
    ax11.legend(loc='best', fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Percent Change to Min Low (next 5 periods)
    ax12 = plt.subplot(7, 2, 12)
    ax12.plot(df_plot['Date'], df_plot['PctChange_ToMinLow_5'], label='% Change to Min Low (next 5)', 
              linewidth=1.5, color='red', alpha=0.8)
    ax12.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax12.set_title('Percent Change: Current Close to Min Low (next 5 periods)', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Date')
    ax12.set_ylabel('Percent Change (%)')
    ax12.legend(loc='best', fontsize=8)
    ax12.grid(True, alpha=0.3)
    
    # Plot 13: Percent Change from Close to High (current bar)
    ax13 = plt.subplot(7, 2, 13)
    ax13.plot(df_plot['Date'], df_plot['PctChange_CloseToHigh'], label='% Change: Close to High', 
              linewidth=1.5, color='green', alpha=0.8)
    ax13.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax13.set_title('Percent Change: Current Close to High', fontsize=12, fontweight='bold')
    ax13.set_xlabel('Date')
    ax13.set_ylabel('Percent Change (%)')
    ax13.legend(loc='best', fontsize=8)
    ax13.grid(True, alpha=0.3)
    
    # Plot 14: Percent Change from Close to Low (current bar)
    ax14 = plt.subplot(7, 2, 14)
    ax14.plot(df_plot['Date'], df_plot['PctChange_CloseToLow'], label='% Change: Close to Low', 
              linewidth=1.5, color='red', alpha=0.8)
    ax14.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax14.set_title('Percent Change: Current Close to Low', fontsize=12, fontweight='bold')
    ax14.set_xlabel('Date')
    ax14.set_ylabel('Percent Change (%)')
    ax14.legend(loc='best', fontsize=8)
    ax14.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to load data, calculate indicators, and plot results"""
    
    # Configuration
    # Use relative path - will be resolved in load_es_data function
    DATA_DIR = '../daily_data'
    
    try:
        # Load ES data
        df = load_es_data(DATA_DIR)
        
        print("Calculating trading hours...")
        # Calculate hours from formal trading (9 AM ET) and overnight trading (6 PM ET)
        df = calculate_trading_hours(df)
        
        # Process all technical indicators using the shared function
        initial_len = len(df)
        df = process_stock_indicators(df, "ES")
        if df.empty:
            raise ValueError("ES data processing failed - insufficient data")
        
        # Calculate forward percent changes (ES only)
        print("Calculating forward percent changes for ES...")
        df = calculate_forward_percent_changes(df, periods=5)
        
        # Skip last 5 values (to avoid NaN from forward calculations)
        initial_len_after_indicators = len(df)
        df = df.iloc[:-5].reset_index(drop=True)
        print(f"\nSkipped first 50 rows and last 5 rows. Data points: {initial_len} -> {initial_len_after_indicators} -> {len(df)}")
        
        # Check for NaN values after trimming
        nan_check = df.isna().sum()
        nan_columns = nan_check[nan_check > 0]
        if len(nan_columns) > 0:
            print(f"\n⚠️  WARNING: Found NaN values in {len(nan_columns)} columns after trimming:")
            for col, count in nan_columns.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count:,} NaN values ({percentage:.2f}% of rows)")
        else:
            print("\n✓ No NaN values found in trimmed data")
        
        # Display summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total data points: {len(df)}")
        print(f"\nBollinger Bands 10:")
        print(f"  Mean: {df['BB_10_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df['BB_10_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df['BB_10_2_Lower'].mean():.2f}")
        
        print(f"\nBollinger Bands 20:")
        print(f"  Mean: {df['BB_20_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df['BB_20_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df['BB_20_2_Lower'].mean():.2f}")
        
        print(f"\nBollinger Bands 50:")
        print(f"  Mean: {df['BB_50_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df['BB_50_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df['BB_50_2_Lower'].mean():.2f}")
        
        print(f"\nExponential Moving Averages (EMA):")
        print(f"  EMA 10: {df['EMA_10'].mean():.2f}")
        print(f"  EMA 20: {df['EMA_20'].mean():.2f}")
        print(f"  EMA 50: {df['EMA_50'].mean():.2f}")
        
        print(f"\nRSI 7: Mean={df['RSI_7'].mean():.2f}, Min={df['RSI_7'].min():.2f}, Max={df['RSI_7'].max():.2f}")
        print(f"RSI 14: Mean={df['RSI_14'].mean():.2f}, Min={df['RSI_14'].min():.2f}, Max={df['RSI_14'].max():.2f}")
        print(f"RSI 28: Mean={df['RSI_28'].mean():.2f}, Min={df['RSI_28'].min():.2f}, Max={df['RSI_28'].max():.2f}")
        print(f"MFI 7: Mean={df['MFI_7'].mean():.2f}, Min={df['MFI_7'].min():.2f}, Max={df['MFI_7'].max():.2f}")
        print(f"MFI 14: Mean={df['MFI_14'].mean():.2f}, Min={df['MFI_14'].min():.2f}, Max={df['MFI_14'].max():.2f}")
        print(f"MFI 28: Mean={df['MFI_28'].mean():.2f}, Min={df['MFI_28'].min():.2f}, Max={df['MFI_28'].max():.2f}")
        print(f"\nPercent Changes (current bar):")
        print(f"  Close to High: Mean={df['PctChange_CloseToHigh'].mean():.4f}%, Min={df['PctChange_CloseToHigh'].min():.4f}%, Max={df['PctChange_CloseToHigh'].max():.4f}%")
        print(f"  Close to Low: Mean={df['PctChange_CloseToLow'].mean():.4f}%, Min={df['PctChange_CloseToLow'].min():.4f}%, Max={df['PctChange_CloseToLow'].max():.4f}%")
        print(f"\nPercent Changes (next 5 periods):")
        print(f"  To Max High: Mean={df['PctChange_ToMaxHigh_5'].mean():.4f}%, Min={df['PctChange_ToMaxHigh_5'].min():.4f}%, Max={df['PctChange_ToMaxHigh_5'].max():.4f}%")
        print(f"  To Min Low: Mean={df['PctChange_ToMinLow_5'].mean():.4f}%, Min={df['PctChange_ToMinLow_5'].min():.4f}%, Max={df['PctChange_ToMinLow_5'].max():.4f}%")
        
        # Plot results
        print("\nGenerating plots...")
        # Get the script's directory to save files in the same folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, 'futures_price_analysis.png')
        plot_results(df, save_path=plot_path)
        
        # ========== VXM Data Processing ==========
        print("\n" + "="*60)
        print("Processing VXM Data")
        print("="*60)
        
        # Load VXM data
        df_vxm = load_vxm_data(DATA_DIR)
        
        print("\nCalculating Bollinger Bands for VXM...")
        # Calculate Bollinger Bands for 10 period (1, 2, 3 std dev)
        df_vxm = calculate_bollinger_bands(df_vxm, period=10, std_dev=1.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=10, std_dev=2.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=10, std_dev=3.0)
        
        # Calculate Bollinger Bands for 20 period (1, 2, 3 std dev)
        df_vxm = calculate_bollinger_bands(df_vxm, period=20, std_dev=1.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=20, std_dev=2.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=20, std_dev=3.0)
        
        # Calculate Bollinger Bands for 50 period (1, 2, 3 std dev)
        df_vxm = calculate_bollinger_bands(df_vxm, period=50, std_dev=1.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=50, std_dev=2.0)
        df_vxm = calculate_bollinger_bands(df_vxm, period=50, std_dev=3.0)
        
        print("Calculating Exponential Moving Averages (EMA) for VXM...")
        # Calculate EMA for 10, 20, and 50 periods
        df_vxm = calculate_ema(df_vxm, period=10)
        df_vxm = calculate_ema(df_vxm, period=20)
        df_vxm = calculate_ema(df_vxm, period=50)
        
        # Skip first 50 values (to avoid NaN from rolling windows)
        initial_len_vxm = len(df_vxm)
        df_vxm = df_vxm.iloc[50:].reset_index(drop=True)
        print(f"\nSkipped first 50 rows. VXM data points: {initial_len_vxm} -> {len(df_vxm)}")
        
        # Check for NaN values after trimming
        nan_check_vxm = df_vxm.isna().sum()
        nan_columns_vxm = nan_check_vxm[nan_check_vxm > 0]
        if len(nan_columns_vxm) > 0:
            print(f"\n⚠️  WARNING: Found NaN values in {len(nan_columns_vxm)} columns after trimming:")
            for col, count in nan_columns_vxm.items():
                percentage = (count / len(df_vxm)) * 100
                print(f"  {col}: {count:,} NaN values ({percentage:.2f}% of rows)")
        else:
            print("\n✓ No NaN values found in trimmed VXM data")
        
        # Display summary statistics for VXM
        print("\n=== VXM Summary Statistics ===")
        print(f"Total VXM data points: {len(df_vxm)}")
        print(f"\nVXM Bollinger Bands 10:")
        print(f"  Mean: {df_vxm['BB_10_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df_vxm['BB_10_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df_vxm['BB_10_2_Lower'].mean():.2f}")
        
        print(f"\nVXM Bollinger Bands 20:")
        print(f"  Mean: {df_vxm['BB_20_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df_vxm['BB_20_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df_vxm['BB_20_2_Lower'].mean():.2f}")
        
        print(f"\nVXM Bollinger Bands 50:")
        print(f"  Mean: {df_vxm['BB_50_MA'].mean():.2f}")
        print(f"  Upper 2σ: {df_vxm['BB_50_2_Upper'].mean():.2f}")
        print(f"  Lower 2σ: {df_vxm['BB_50_2_Lower'].mean():.2f}")
        
        print(f"\nVXM Exponential Moving Averages (EMA):")
        print(f"  EMA 10: {df_vxm['EMA_10'].mean():.2f}")
        print(f"  EMA 20: {df_vxm['EMA_20'].mean():.2f}")
        print(f"  EMA 50: {df_vxm['EMA_50'].mean():.2f}")
        
        # ========== Stock Data Processing ==========
        print("\n" + "="*60)
        print("Processing Stock Data")
        print("="*60)
        
        # List of stocks to process (note: user wrote APPL but using AAPL)
        stocks = ['TSLA', 'NVDA', 'MSFT', 'META', 'JPM', 'GOOG', 'AVGO', 'AMZN', 'AAPL']
        
        stock_dataframes = {}
        
        for ticker in stocks:
            # Load stock data
            df_stock = load_stock_data(ticker, DATA_DIR)
            
            if df_stock.empty:
                print(f"Skipping {ticker} - no data found")
                continue
            
            # Process all indicators for this stock
            df_stock = process_stock_indicators(df_stock, ticker)
            
            if df_stock.empty:
                print(f"Skipping {ticker} - insufficient data after processing")
                continue
            
            # Rename columns to have ticker prefix (except Date which is used for joining)
            stock_columns_rename = {col: f'{ticker}_{col}' for col in df_stock.columns if col != 'Date'}
            df_stock_renamed = df_stock.rename(columns=stock_columns_rename)
            
            stock_dataframes[ticker] = df_stock_renamed
            print(f"Processed {ticker}: {len(df_stock_renamed)} data points")
        
        # ========== Merge All Data ==========
        print("\n" + "="*60)
        print("Merging All Data (ES, VXM, and Stocks)")
        print("="*60)
        
        # Rename VXM columns to have "VXM_" prefix (except Date which is used for joining)
        vxm_columns_rename = {col: f'VXM_{col}' for col in df_vxm.columns if col != 'Date'}
        df_vxm_renamed = df_vxm.rename(columns=vxm_columns_rename)
        
        # Start with ES dataframe
        df_combined = df.copy()
        print(f"Starting with ES: {len(df_combined)} data points")
        
        # Merge VXM
        print(f"Merging VXM: {len(df_vxm_renamed)} data points")
        df_combined = pd.merge(df_combined, df_vxm_renamed, on='Date', how='outer')
        print(f"After VXM merge: {len(df_combined)} data points")
        
        # Merge all stocks
        for ticker, df_stock in stock_dataframes.items():
            print(f"Merging {ticker}: {len(df_stock)} data points")
            df_combined = pd.merge(df_combined, df_stock, on='Date', how='outer')
            print(f"After {ticker} merge: {len(df_combined)} data points")
        
        # Sort by Date
        df_combined = df_combined.sort_values('Date').reset_index(drop=True)
        
        # Forward-fill missing values (replicate the value of the row before it)
        print("\nForward-filling missing values...")
        initial_missing = df_combined.isna().sum().sum()
        df_combined = df_combined.ffill()
        after_ffill_missing = df_combined.isna().sum().sum()
        print(f"Forward-filled {initial_missing - after_ffill_missing} missing values")
        
        # Drop rows where there's still missing data (no row before it to replicate from)
        print("\nDropping rows with missing data that couldn't be forward-filled...")
        initial_len = len(df_combined)
        df_combined = df_combined.dropna()
        final_len = len(df_combined)
        print(f"Dropped {initial_len - final_len} rows with missing data (no previous value to replicate)")
        
        print(f"\nFinal combined data points: {final_len}")
        print(f"Date range: {df_combined['Date'].min()} to {df_combined['Date'].max()}")
        
        # Save the combined dataframe with all indicators
        output_file = os.path.join(script_dir, 'es_with_indicators.csv')
        df_combined.to_csv(output_file, index=False)
        print(f"\nCombined data (ES, VXM, and all stocks) with all indicators saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

