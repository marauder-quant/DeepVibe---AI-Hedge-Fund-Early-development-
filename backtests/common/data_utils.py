"""
Data utilities for backtesting strategies.

This module provides common utilities for fetching and preprocessing market data
that can be shared across different strategy implementations.
"""

import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Try to import vectorbt and alpaca libraries, with clear error messages if missing
try:
    import vectorbt as vbt
except ImportError:
    raise ImportError("vectorbt is required. Install with: pip install vectorbt")

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Parse a timeframe string into amount and unit.
    
    Args:
        timeframe: String in format like '1d', '4h', '30m'
        
    Returns:
        Tuple of (amount, unit)
    """
    if timeframe.endswith('d'):
        return int(timeframe[:-1]) if len(timeframe) > 1 else 1, 'day'
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) if len(timeframe) > 1 else 1, 'hour'
    elif timeframe.endswith('m'):
        return int(timeframe[:-1]) if len(timeframe) > 1 else 1, 'minute'
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) if len(timeframe) > 1 else 1, 'week'
    else:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")


def get_alpaca_timeframe(timeframe: str) -> TimeFrame:
    """
    Convert a timeframe string to Alpaca TimeFrame object.
    
    Args:
        timeframe: String in format like '1d', '4h', '30m'
        
    Returns:
        Alpaca TimeFrame object
    """
    if not ALPACA_AVAILABLE:
        raise ImportError("Alpaca API libraries not available. Install with: pip install alpaca-py")
        
    amount, unit = parse_timeframe(timeframe)
    
    if unit == 'day':
        return TimeFrame(amount, TimeFrameUnit.Day)
    elif unit == 'hour':
        return TimeFrame(amount, TimeFrameUnit.Hour)
    elif unit == 'minute':
        return TimeFrame(amount, TimeFrameUnit.Minute)
    elif unit == 'week':
        return TimeFrame(amount, TimeFrameUnit.Week)
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")


def fetch_market_data(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    timeframe: str = '1d',
    include_buffer: bool = True,
    buffer_size: int = 100,
    columns: List[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Fetch market data from Alpaca or fall back to Yahoo Finance.
    
    Args:
        symbol: Trading symbol
        start_date: Start date for data
        end_date: End date for data
        timeframe: Timeframe for data (e.g., '1d', '1h', '15m')
        include_buffer: Whether to add a buffer before start_date for calculations
        buffer_size: Number of periods to use as buffer
        columns: List of columns to include (defaults to OHLCV)
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with market data
    """
    # Set default columns
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert string dates to datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Add timezone info if not present
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=pytz.UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=pytz.UTC)
    
    # Calculate buffer time
    amount, unit = parse_timeframe(timeframe)
    
    if include_buffer:
        if unit == 'day':
            time_buffer = timedelta(days=buffer_size * amount)
        elif unit == 'hour':
            time_buffer = timedelta(hours=buffer_size * amount)
        elif unit == 'minute':
            time_buffer = timedelta(minutes=buffer_size * amount)
        elif unit == 'week':
            time_buffer = timedelta(weeks=buffer_size * amount)
        buffered_start = start_date - time_buffer
    else:
        buffered_start = start_date
        time_buffer = timedelta(0)
    
    try:
        if verbose:
            print(f"Attempting to download data from Alpaca for {symbol}...")
        
        # Check if Alpaca is available
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca API libraries not available")
        
        # Get Alpaca API credentials
        api_key = os.environ.get('alpaca_paper_key')
        api_secret = os.environ.get('alpaca_paper_secret')
        
        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        tf = get_alpaca_timeframe(timeframe)
        
        # Determine if it's crypto or stock
        if '/' in symbol:
            client = CryptoHistoricalDataClient()
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=buffered_start.isoformat(),
                end=end_date.isoformat()
            )
            bars = client.get_crypto_bars(request_params)
        else:
            client = StockHistoricalDataClient(api_key, api_secret)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=buffered_start.isoformat(),
                end=end_date.isoformat(),
                adjustment='all'
            )
            bars = client.get_stock_bars(request_params)
            
        # Convert to dataframe
        df = bars.df
        
        if len(df) == 0:
            raise ValueError(f"No data returned from Alpaca for {symbol}")
            
        # Process MultiIndex if present
        if isinstance(df.index, pd.MultiIndex):
            if verbose:
                print("Converting MultiIndex to DatetimeIndex")
            # Extract timestamp from the MultiIndex
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.drop(columns=['symbol'], errors='ignore')
        
        # Rename columns to expected format
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        # Drop any extra columns not needed
        if all(col in df.columns for col in columns):
            df = df[columns]
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if verbose:
            print(f"Data shape from Alpaca: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    except Exception as e:
        if verbose:
            print(f"Error getting data from Alpaca: {str(e)}")
            print("Falling back to Yahoo Finance...")
        
        # Fallback to Yahoo Finance
        df = vbt.YFData.download(
            symbol, 
            start=buffered_start, 
            end=end_date
        ).get(columns)
    
    # Convert to float64
    df = df.astype(np.float64)
    
    # Create a copy without buffer if requested
    if include_buffer:
        # Create masks for with and without buffer
        with_buffer_mask = (df.index >= buffered_start) & (df.index <= end_date)
        without_buffer_mask = (df.index >= start_date) & (df.index <= end_date)
        
        # Return different DataFrames based on include_buffer
        df_with_buffer = df.loc[with_buffer_mask, :]
        df_without_buffer = df.loc[without_buffer_mask, :]
        
        if verbose:
            print(f"With buffer: {len(df_with_buffer)} rows from {df_with_buffer.index.min()} to {df_with_buffer.index.max()}")
            print(f"Without buffer: {len(df_without_buffer)} rows from {df_without_buffer.index.min()} to {df_without_buffer.index.max()}")
        
        return {
            'with_buffer': df_with_buffer,
            'without_buffer': df_without_buffer,
            'buffer_size': time_buffer
        }
    else:
        return df


def apply_splits(
    df: pd.DataFrame,
    split_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply data splitting using vectorbt's splitter.
    
    Args:
        df: DataFrame containing market data
        split_config: Configuration for data splitting
        
    Returns:
        Tuple of (in_sample_df, out_of_sample_df)
    """
    from backtests.common.data_splitting import convert_to_vectorbt_params
    
    # Convert configuration to vectorbt parameters
    vbt_params = convert_to_vectorbt_params(split_config)
    
    # Use vectorbt's splitting functionality
    # Note: Different versions of vectorbt have different APIs
    try:
        # Newer versions of vectorbt
        splitter = vbt.splits.RollingSplitter(**vbt_params)
        price_tuple = splitter.split(df)
    except (TypeError, AttributeError):
        try:
            # Try older versions of vectorbt
            price_tuple = vbt.utils.splits.rolling_split(df, **vbt_params)
        except (TypeError, AttributeError):
            # Fallback to the most basic approach
            n_splits = vbt_params.get('n', 5)
            window_len = vbt_params.get('window_len', 252)
            set_lens = vbt_params.get('set_lens', (63,))
            
            # Create manual splits
            total_len = len(df)
            all_indices = np.arange(total_len)
            
            in_sample_indices = []
            out_sample_indices = []
            
            for i in range(n_splits):
                # Calculate boundaries for this split
                if vbt_params.get('left_to_right', True):
                    start_idx = i * (window_len + set_lens[0])
                    train_end_idx = start_idx + window_len
                    test_end_idx = min(train_end_idx + set_lens[0], total_len)
                else:
                    test_end_idx = total_len - i * set_lens[0]
                    train_end_idx = test_end_idx - set_lens[0]
                    start_idx = max(0, train_end_idx - window_len)
                
                # Get indices for this split
                train_indices = all_indices[start_idx:train_end_idx]
                test_indices = all_indices[train_end_idx:test_end_idx]
                
                # Skip if test set is empty
                if len(test_indices) == 0:
                    continue
                
                in_sample_indices.append(train_indices)
                out_sample_indices.append(test_indices)
            
            # Create DataFrames from indices
            in_sample_df = pd.DataFrame()
            out_sample_df = pd.DataFrame()
            
            for i in range(len(in_sample_indices)):
                in_sample_df[i] = pd.Series(df.iloc[in_sample_indices[i]].values)
                out_sample_df[i] = pd.Series(df.iloc[out_sample_indices[i]].values)
            
            return in_sample_df, out_sample_df
    
    # Extract the results
    try:
        # Standard format: ((in_price, in_mask), (out_price, out_mask))
        in_sample_df = price_tuple[0][0]
        out_sample_df = price_tuple[1][0]
    except (IndexError, TypeError):
        # Alternative format: (in_price, out_price)
        in_sample_df = price_tuple[0]
        out_sample_df = price_tuple[1]
    
    return in_sample_df, out_sample_df 