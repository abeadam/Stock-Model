"""
Futures Price Analysis - ES (E-mini S&P 500) Data with Technical Indicators
CHUNKED VERSION - Optimized for 20 GB RAM

This version processes data in chunks to minimize memory usage:
- Reads CSV files in chunks
- Processes indicators in chunks
- Frees memory after each operation
- Uses efficient merging strategies
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import Optional, List, Iterator
from datetime import datetime
import pytz
import gc
from pathlib import Path


# Memory management constants
CHUNK_SIZE = 100000  # Process 100k rows at a time
MAX_MEMORY_GB = 20
TEMP_DIR = 'temp_chunks'  # Directory for temporary chunk files


def ensure_temp_dir():
    """Create temporary directory for chunk files if it doesn't exist"""
    Path(TEMP_DIR).mkdir(exist_ok=True)


def clear_temp_dir():
    """Remove all temporary chunk files"""
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    ensure_temp_dir()


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # If psutil is not available, return 0
        return 0.0


def free_memory():
    """Force garbage collection to free memory"""
    gc.collect()


def load_es_data_chunked(data_dir: str = 'daily_data', chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Load ES (E-mini S&P 500) data from CSV files in chunks"""
    print("Loading ES data files in chunks...")
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
        data_dir = os.path.normpath(data_dir)
    
    # Get all ES files
    es_files = glob.glob(os.path.join(data_dir, "*_ES.txt"))
    
    if not es_files:
        raise ValueError(f"No ES files found in {data_dir}. Checked path: {os.path.abspath(data_dir)}")
    
    print(f"Found {len(es_files)} ES files")
    
    # Read all files and yield chunks
    for es_file in sorted(es_files):
        try:
            # Read file in chunks
            chunk_reader = pd.read_csv(es_file, chunksize=chunk_size)
            for chunk in chunk_reader:
                # Ensure we have required columns
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in chunk.columns for col in required_cols):
                    print(f"Warning: {es_file} missing required columns, skipping chunk")
                    continue
                
                if 'Volume' not in chunk.columns:
                    chunk['Volume'] = 0
                
                yield chunk
        except Exception as e:
            print(f"Error loading {es_file}: {e}")
            continue


def load_vxm_data_chunked(data_dir: str = 'daily_data', chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Load VXM data from CSV files in chunks"""
    print("Loading VXM data files in chunks...")
    
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
        data_dir = os.path.normpath(data_dir)
    
    vxm_files = glob.glob(os.path.join(data_dir, "*_VXM.txt"))
    
    if not vxm_files:
        raise ValueError(f"No VXM files found in {data_dir}")
    
    print(f"Found {len(vxm_files)} VXM files")
    
    for vxm_file in sorted(vxm_files):
        try:
            chunk_reader = pd.read_csv(vxm_file, chunksize=chunk_size)
            for chunk in chunk_reader:
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in chunk.columns for col in required_cols):
                    print(f"Warning: {vxm_file} missing required columns, skipping chunk")
                    continue
                
                if 'Volume' not in chunk.columns:
                    chunk['Volume'] = 0
                
                yield chunk
        except Exception as e:
            print(f"Error loading {vxm_file}: {e}")
            continue


def load_stock_data_chunked(ticker: str, data_dir: str = 'daily_data', chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Load stock data from CSV files in chunks"""
    print(f"Loading {ticker} data files in chunks...")
    
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
        data_dir = os.path.normpath(data_dir)
    
    stock_files = glob.glob(os.path.join(data_dir, f"*_{ticker}.txt"))
    
    if not stock_files:
        print(f"Warning: No {ticker} files found in {data_dir}")
        return
    
    print(f"Found {len(stock_files)} {ticker} files")
    
    for stock_file in sorted(stock_files):
        try:
            chunk_reader = pd.read_csv(stock_file, chunksize=chunk_size)
            for chunk in chunk_reader:
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in chunk.columns for col in required_cols):
                    print(f"Warning: {stock_file} missing required columns, skipping chunk")
                    continue
                
                if 'Volume' not in chunk.columns:
                    chunk['Volume'] = 0
                
                yield chunk
        except Exception as e:
            print(f"Error loading {stock_file}: {e}")
            continue


def process_chunk_indicators(df: pd.DataFrame, ticker: str, min_period: int = 50) -> pd.DataFrame:
    """Calculate technical indicators for a chunk of data"""
    if df.empty:
        return df
    
    # Calculate Bollinger Bands
    for period in [10, 20, 50]:
        for std_dev in [1.0, 2.0, 3.0]:
            df = calculate_bollinger_bands(df, period=period, std_dev=std_dev)
    
    # Calculate EMA
    for period in [10, 20, 50]:
        df = calculate_ema(df, period=period)
    
    # Calculate RSI
    for period in [7, 14, 28]:
        df = calculate_rsi(df, period=period)
    
    # Calculate MFI
    for period in [7, 14, 28]:
        df = calculate_mfi(df, period=period)
    
    # Calculate current percent changes
    df = calculate_current_percent_changes(df)
    
    # Skip rows that have NaN from rolling windows (first min_period rows)
    if len(df) > min_period:
        df = df.iloc[min_period:].reset_index(drop=True)
    else:
        return pd.DataFrame()
    
    return df


def calculate_bollinger_bands(df: pd.DataFrame, period: int, std_dev: float, 
                              price_col: str = 'Close') -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    df = df.copy()
    
    ma_col = f'BB_{period}_MA'
    df[ma_col] = df[price_col].rolling(window=period).mean()
    
    std_col = f'BB_{period}_STD'
    df[std_col] = df[price_col].rolling(window=period).std()
    
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
    
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi_col = f'RSI_{period}'
    df[rsi_col] = 100 - (100 / (1 + rs))
    
    return df


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Money Flow Index (MFI)"""
    df = df.copy()
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_flow_sum = money_flow_positive.rolling(window=period).sum()
    negative_flow_sum = money_flow_negative.rolling(window=period).sum()
    
    mfi_col = f'MFI_{period}'
    df[mfi_col] = np.where(
        negative_flow_sum == 0,
        100.0,
        100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    )
    
    return df


def calculate_current_percent_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percent change from current Close to High and Low"""
    df = df.copy()
    
    df['PctChange_CloseToHigh'] = ((df['High'] - df['Close']) / df['Close']) * 100
    df['PctChange_CloseToLow'] = ((df['Low'] - df['Close']) / df['Close']) * 100
    
    return df


def calculate_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate hours from start of formal trading (9 AM ET) and overnight trading (6 PM ET)"""
    df = df.copy()
    
    if df['Date'].dtype != 'datetime64[ns]':
        try:
            df['DateTime'] = pd.to_datetime(df['Date'], unit='s')
        except:
            df['DateTime'] = pd.to_datetime(df['Date'])
    else:
        df['DateTime'] = df['Date']
    
    eastern = pytz.timezone('US/Eastern')
    
    if df['DateTime'].dt.tz is None:
        df['DateTime_ET'] = pd.to_datetime(df['DateTime']).dt.tz_localize('UTC').dt.tz_convert(eastern)
    else:
        df['DateTime_ET'] = df['DateTime'].dt.tz_convert(eastern)
    
    df['Hours_From_Formal_Trading'] = np.nan
    df['Hours_From_Overnight_Trading'] = np.nan
    
    for idx in df.index:
        time_et = df.at[idx, 'DateTime_ET']
        date_only = time_et.date()
        
        formal_start_time = datetime.strptime('09:00:00', '%H:%M:%S').time()
        formal_start = eastern.localize(datetime.combine(date_only, formal_start_time))
        
        overnight_start_time = datetime.strptime('18:00:00', '%H:%M:%S').time()
        overnight_start = eastern.localize(datetime.combine(date_only, overnight_start_time))
        if time_et < overnight_start:
            from datetime import timedelta
            prev_day = date_only - timedelta(days=1)
            overnight_start = eastern.localize(datetime.combine(prev_day, overnight_start_time))
        
        hours_from_formal = (time_et - formal_start).total_seconds() / 3600.0
        df.at[idx, 'Hours_From_Formal_Trading'] = hours_from_formal
        
        hours_from_overnight = (time_et - overnight_start).total_seconds() / 3600.0
        df.at[idx, 'Hours_From_Overnight_Trading'] = hours_from_overnight
    
    return df


def process_data_in_chunks(data_loader_func, ticker: str, data_dir: str, 
                           output_file: str, process_forward_changes: bool = False):
    """
    Process data in chunks and save to temporary files, then combine.
    This minimizes memory usage by processing and saving chunks separately.
    """
    ensure_temp_dir()
    
    chunk_files = []
    chunk_num = 0
    total_rows = 0
    
    print(f"\nProcessing {ticker} data in chunks...")
    print(f"Memory before processing: {get_memory_usage_mb():.2f} MB")
    
    # Process each chunk
    for chunk in data_loader_func(data_dir):
        chunk_num += 1
        total_rows += len(chunk)
        
        if chunk_num % 10 == 0:
            print(f"  Processed {chunk_num} chunks, {total_rows:,} rows, Memory: {get_memory_usage_mb():.2f} MB")
        
        # Calculate trading hours for ES only
        if ticker == "ES":
            chunk = calculate_trading_hours(chunk)
        
        # Process indicators
        chunk = process_chunk_indicators(chunk, ticker)
        
        if chunk.empty:
            continue
        
        # Rename columns with ticker prefix (except Date)
        if ticker != "ES":
            rename_dict = {col: f'{ticker}_{col}' for col in chunk.columns if col != 'Date'}
            chunk = chunk.rename(columns=rename_dict)
        
        # Save chunk to temporary file (use CSV if parquet not available)
        chunk_file = os.path.join(TEMP_DIR, f'{ticker}_chunk_{chunk_num}.csv')
        try:
            chunk.to_parquet(chunk_file.replace('.csv', '.parquet'), index=False, compression='snappy')
            chunk_files.append(chunk_file.replace('.csv', '.parquet'))
        except (ImportError, AttributeError):
            # Fallback to CSV if parquet not available
            chunk.to_csv(chunk_file, index=False)
            chunk_files.append(chunk_file)
        
        # Free memory
        del chunk
        free_memory()
    
    print(f"Processed {chunk_num} chunks, {total_rows:,} total rows")
    print(f"Memory after processing chunks: {get_memory_usage_mb():.2f} MB")
    
    # Combine chunks efficiently
    print(f"\nCombining {len(chunk_files)} chunks for {ticker}...")
    combined_chunks = []
    
    for i, chunk_file in enumerate(chunk_files):
        if i % 50 == 0:
            print(f"  Loading chunk {i+1}/{len(chunk_files)}, Memory: {get_memory_usage_mb():.2f} MB")
        
        # Try to read as parquet, fallback to CSV
        try:
            chunk_df = pd.read_parquet(chunk_file)
        except (ImportError, AttributeError, ValueError):
            chunk_df = pd.read_csv(chunk_file)
        
        combined_chunks.append(chunk_df)
        
        # If we have too many chunks in memory, combine and save intermediate result
        if len(combined_chunks) >= 20:
            temp_combined = pd.concat(combined_chunks, ignore_index=True)
            temp_combined = temp_combined.sort_values('Date').reset_index(drop=True)
            
            # Save intermediate result
            temp_file = os.path.join(TEMP_DIR, f'{ticker}_temp_combined_{i}.csv')
            try:
                temp_combined.to_parquet(temp_file.replace('.csv', '.parquet'), index=False, compression='snappy')
                temp_file = temp_file.replace('.csv', '.parquet')
            except (ImportError, AttributeError):
                temp_combined.to_csv(temp_file, index=False)
            
            combined_chunks = [temp_combined]
            
            del temp_combined
            free_memory()
    
    # Final combination
    actual_output_file = output_file
    if combined_chunks:
        print(f"Final combination of {len(combined_chunks)} chunks...")
        final_df = pd.concat(combined_chunks, ignore_index=True)
        final_df = final_df.sort_values('Date').reset_index(drop=True)
        
        # Save final result
        try:
            final_df.to_parquet(output_file, index=False, compression='snappy')
        except (ImportError, AttributeError):
            # Fallback to CSV
            actual_output_file = output_file.replace('.parquet', '.csv')
            final_df.to_csv(actual_output_file, index=False)
        print(f"Saved {ticker} data: {len(final_df):,} rows to {actual_output_file}")
        print(f"Memory after final combination: {get_memory_usage_mb():.2f} MB")
        
        del final_df
        free_memory()
    else:
        print(f"Warning: No chunks to combine for {ticker}")
        return None
    
    # Clean up chunk files
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    
    return actual_output_file


def merge_dataframes_chunked(data_files: List[str], output_file: str):
    """
    Merge multiple dataframes efficiently by reading and merging in chunks.
    Uses Date column for merging.
    """
    print(f"\nMerging {len(data_files)} data files...")
    print(f"Memory before merging: {get_memory_usage_mb():.2f} MB")
    
    # Start with first file
    print(f"Loading base file: {data_files[0]}")
    try:
        df_combined = pd.read_parquet(data_files[0])
    except (ImportError, AttributeError, ValueError):
        df_combined = pd.read_csv(data_files[0])
    print(f"Base file loaded: {len(df_combined):,} rows, Memory: {get_memory_usage_mb():.2f} MB")
    
    # Merge other files one by one
    for i, data_file in enumerate(data_files[1:], 1):
        print(f"\nMerging file {i+1}/{len(data_files)-1}: {data_file}")
        try:
            df_to_merge = pd.read_parquet(data_file)
        except (ImportError, AttributeError, ValueError):
            df_to_merge = pd.read_csv(data_file)
        print(f"  File to merge: {len(df_to_merge):,} rows")
        
        # Merge on Date column
        df_combined = pd.merge(df_combined, df_to_merge, on='Date', how='outer')
        print(f"  After merge: {len(df_combined):,} rows, Memory: {get_memory_usage_mb():.2f} MB")
        
        # Free memory
        del df_to_merge
        free_memory()
        
        # If memory is getting high, save intermediate result and continue
        if get_memory_usage_mb() > MAX_MEMORY_GB * 1024 * 0.8:  # 80% of max memory
            print(f"  Memory high ({get_memory_usage_mb():.2f} MB), saving intermediate result...")
            temp_file = os.path.join(TEMP_DIR, f'merged_temp_{i}.csv')
            try:
                df_combined.to_parquet(temp_file.replace('.csv', '.parquet'), index=False, compression='snappy')
                temp_file = temp_file.replace('.csv', '.parquet')
            except (ImportError, AttributeError):
                df_combined.to_csv(temp_file, index=False)
            
            try:
                df_combined = pd.read_parquet(temp_file)
            except (ImportError, AttributeError, ValueError):
                df_combined = pd.read_csv(temp_file)
            
            os.remove(temp_file)
            free_memory()
    
    # Sort by Date
    print("\nSorting merged data by Date...")
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    # Forward-fill missing values
    print("Forward-filling missing values...")
    initial_missing = df_combined.isna().sum().sum()
    df_combined = df_combined.ffill()
    after_ffill_missing = df_combined.isna().sum().sum()
    print(f"Forward-filled {initial_missing - after_ffill_missing:,} missing values")
    
    # Drop rows with remaining missing data
    print("Dropping rows with remaining missing data...")
    initial_len = len(df_combined)
    df_combined = df_combined.dropna()
    final_len = len(df_combined)
    print(f"Dropped {initial_len - final_len:,} rows")
    
    # Save final result
    print(f"\nSaving final merged data: {final_len:,} rows")
    df_combined.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    print(f"Final memory usage: {get_memory_usage_mb():.2f} MB")
    
    return df_combined


def plot_results(df: pd.DataFrame, save_path: Optional[str] = None, last_n_points: int = 400):
    """Plot all calculated indicators (only last N points)"""
    
    df_plot = df.tail(last_n_points).copy()
    
    if df_plot['Date'].dtype != 'datetime64[ns]':
        try:
            df_plot['Date'] = pd.to_datetime(df_plot['Date'], unit='s')
        except:
            df_plot['Date'] = pd.to_datetime(df_plot['Date'])
    
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
    if 'PctChange_ToMaxHigh_5' in df_plot.columns:
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
    if 'PctChange_ToMinLow_5' in df_plot.columns:
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
    if 'PctChange_CloseToHigh' in df_plot.columns:
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
    if 'PctChange_CloseToLow' in df_plot.columns:
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
    """Main function to load data, calculate indicators, and plot results - CHUNKED VERSION"""
    
    DATA_DIR = '../daily_data'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Clear temp directory
        clear_temp_dir()
        
        print("="*60)
        print("FUTURES PRICE ANALYSIS - CHUNKED VERSION (20 GB RAM)")
        print("="*60)
        print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")
        
        # Process ES data
        es_output = os.path.join(TEMP_DIR, 'ES_processed.parquet')
        es_output_actual = process_data_in_chunks(load_es_data_chunked, "ES", DATA_DIR, es_output, process_forward_changes=True)
        if es_output_actual:
            es_output = es_output_actual
        free_memory()
        
        # Process VXM data
        vxm_output = os.path.join(TEMP_DIR, 'VXM_processed.parquet')
        vxm_output_actual = process_data_in_chunks(load_vxm_data_chunked, "VXM", DATA_DIR, vxm_output, process_forward_changes=False)
        if vxm_output_actual:
            vxm_output = vxm_output_actual
        free_memory()
        
        # Process stock data
        stocks = ['TSLA', 'NVDA', 'MSFT', 'META', 'JPM', 'GOOG', 'AVGO', 'AMZN', 'AAPL']
        stock_outputs = []
        
        for ticker in stocks:
            stock_output = os.path.join(TEMP_DIR, f'{ticker}_processed.parquet')
            try:
                stock_output_actual = process_data_in_chunks(
                    lambda data_dir: load_stock_data_chunked(ticker, data_dir),
                    ticker, DATA_DIR, stock_output, process_forward_changes=False
                )
                if stock_output_actual and os.path.exists(stock_output_actual):
                    stock_outputs.append(stock_output_actual)
                elif os.path.exists(stock_output):
                    stock_outputs.append(stock_output)
                free_memory()
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        # Rename VXM columns
        print("\nRenaming VXM columns...")
        try:
            vxm_df = pd.read_parquet(vxm_output)
        except (ImportError, AttributeError, ValueError):
            vxm_df = pd.read_csv(vxm_output.replace('.parquet', '.csv'))
        
        vxm_rename = {col: f'VXM_{col}' for col in vxm_df.columns if col != 'Date'}
        vxm_df = vxm_df.rename(columns=vxm_rename)
        
        try:
            vxm_df.to_parquet(vxm_output, index=False, compression='snappy')
        except (ImportError, AttributeError):
            vxm_output = vxm_output.replace('.parquet', '.csv')
            vxm_df.to_csv(vxm_output, index=False)
        
        del vxm_df
        free_memory()
        
        # Merge all data
        all_files = [es_output, vxm_output] + stock_outputs
        final_output = os.path.join(script_dir, 'es_with_indicators.csv')
        df_final = merge_dataframes_chunked(all_files, final_output)
        
        # Generate summary statistics (on sample if too large)
        print("\n=== Summary Statistics ===")
        if len(df_final) > 100000:
            sample_df = df_final.sample(n=100000)
            print(f"Using sample of 100,000 rows for statistics (total: {len(df_final):,})")
        else:
            sample_df = df_final
        
        print(f"Total data points: {len(df_final):,}")
        if 'BB_20_MA' in sample_df.columns:
            print(f"BB_20_MA Mean: {sample_df['BB_20_MA'].mean():.2f}")
        if 'RSI_14' in sample_df.columns:
            print(f"RSI_14 Mean: {sample_df['RSI_14'].mean():.2f}, Min: {sample_df['RSI_14'].min():.2f}, Max: {sample_df['RSI_14'].max():.2f}")
        
        # Plot results (using last 400 points)
        print("\nGenerating plots...")
        plot_path = os.path.join(script_dir, 'futures_price_analysis.png')
        
        # Load only last 400 points for plotting
        df_plot = df_final.tail(400).copy()
        plot_results(df_plot, save_path=plot_path)
        
        # Clean up
        clear_temp_dir()
        print(f"\nFinal memory usage: {get_memory_usage_mb():.2f} MB")
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp directory
        clear_temp_dir()


if __name__ == "__main__":
    main()

