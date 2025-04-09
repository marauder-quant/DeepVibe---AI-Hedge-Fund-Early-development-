#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quad Moving Average Crossover (QMAC) strategy implementation using vectorbt and Alpaca data.
This module provides functionality to run QMAC strategy backtests, analyze window combinations,
and visualize results. QMAC uses two separate crossover systems - one for buy signals and one for sell signals.
"""

import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import gc
import os
from dotenv import load_dotenv
import argparse
from dateutil.parser import parse
import itertools
from tqdm.auto import tqdm  # For progress bars
import numba as nb  # For performance optimization
import time  # For timing operations
import multiprocessing as mp
import ray  # For distributed computing

# Import configuration - use direct import instead of relative import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# Load environment variables for Alpaca API keys
load_dotenv()

#############################################
######## CONFIGURATION PARAMETERS ###########
#############################################
# All configuration parameters are now imported from config.py

# Initialize Ray for distributed computing
ray.init(ignore_reinit_error=True)

@nb.njit
def calculate_cross_signals(fast_ma, slow_ma):
    """
    Calculate crossover signals using numba for performance.
    
    Args:
        fast_ma (numpy.ndarray): Fast moving average values
        slow_ma (numpy.ndarray): Slow moving average values
        
    Returns:
        numpy.ndarray: Boolean array with True at crossover points
    """
    signals = np.zeros(len(fast_ma), dtype=np.bool_)
    
    # Skip the first value as we can't determine a cross
    for i in range(1, len(fast_ma)):
        # Check if fast MA crossed above slow MA
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            signals[i] = True
    
    return signals

@nb.njit
def calculate_cross_below_signals(fast_ma, slow_ma):
    """
    Calculate crossover below signals using numba for performance.
    
    Args:
        fast_ma (numpy.ndarray): Fast moving average values
        slow_ma (numpy.ndarray): Slow moving average values
        
    Returns:
        numpy.ndarray: Boolean array with True at crossover points
    """
    signals = np.zeros(len(fast_ma), dtype=np.bool_)
    
    # Skip the first value as we can't determine a cross
    for i in range(1, len(fast_ma)):
        # Check if fast MA crossed below slow MA
        if fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
            signals[i] = True
    
    return signals

def run_qmac_strategy(symbol, start_date, end_date, 
                      buy_fast_window, buy_slow_window, 
                      sell_fast_window, sell_slow_window,
                      init_cash=INITIAL_CASH, fees=FEES, slippage=SLIPPAGE, 
                      timeframe='1d', verbose=True):
    """
    Run a Quad Moving Average Crossover strategy with specific window parameters.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        start_date (datetime): Start date for the backtest
        end_date (datetime): End date for the backtest
        buy_fast_window (int): Fast moving average window size for buy signals
        buy_slow_window (int): Slow moving average window size for buy signals
        sell_fast_window (int): Fast moving average window size for sell signals
        sell_slow_window (int): Slow moving average window size for sell signals
        init_cash (float): Initial cash amount
        fees (float): Fee percentage (e.g., 0.0025 for 0.25%)
        slippage (float): Slippage percentage
        timeframe (str): Timeframe for data (e.g., '1d', '1h', '15m')
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Dictionary containing the backtest results
    """
    start_time = time.time()
    if verbose:
        print(f"Starting QMAC strategy backtest with window parameters:")
        print(f"  Buy: Fast={buy_fast_window}, Slow={buy_slow_window}")
        print(f"  Sell: Fast={sell_fast_window}, Slow={sell_slow_window}")
    
    # Set portfolio parameters
    vbt.settings.portfolio['init_cash'] = init_cash
    vbt.settings.portfolio['fees'] = fees
    vbt.settings.portfolio['slippage'] = slippage
    
    # Add time buffer for SMA/EMA calculation
    time_buffer = timedelta(days=500)
    
    # Download data with time buffer using Alpaca or fallback to Yahoo Finance if needed
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        if verbose:
            print(f"Attempting to download data from Alpaca for {symbol}...")
        # Set Alpaca API credentials from environment variables
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        
        # Parse timeframe string
        if timeframe.endswith('d'):
            tf_amount = int(timeframe[:-1]) if len(timeframe) > 1 else 1
            tf = TimeFrame(tf_amount, TimeFrameUnit.Day)
        elif timeframe.endswith('h'):
            tf_amount = int(timeframe[:-1]) if len(timeframe) > 1 else 1
            tf = TimeFrame(tf_amount, TimeFrameUnit.Hour)
        elif timeframe.endswith('m'):
            tf_amount = int(timeframe[:-1]) if len(timeframe) > 1 else 1
            tf = TimeFrame(tf_amount, TimeFrameUnit.Minute)
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
        
        # Determine if it's crypto or stock
        if '/' in symbol:
            client = CryptoHistoricalDataClient()
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=(start_date-time_buffer).isoformat(),
                end=end_date.isoformat()
            )
            bars = client.get_crypto_bars(request_params)
        else:
            api_key = os.environ.get('alpaca_paper_key')
            api_secret = os.environ.get('alpaca_paper_secret')
            client = StockHistoricalDataClient(api_key, api_secret)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=(start_date-time_buffer).isoformat(),
                end=end_date.isoformat(),
                adjustment='all'
            )
            bars = client.get_stock_bars(request_params)
            
        # Convert to dataframe
        df = bars.df
        
        if len(df) == 0:
            raise ValueError(f"No data returned from Alpaca for {symbol}")
            
        # Check and rename columns if needed
        if verbose:
            print(f"Columns in dataframe: {df.columns.tolist()}")
            print(f"Index type: {type(df.index)}")
        
        # If we have a MultiIndex, reset and use only the timestamp
        if isinstance(df.index, pd.MultiIndex):
            if verbose:
                print("Converting MultiIndex to DatetimeIndex")
            # Extract timestamp from the MultiIndex
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.drop(columns=['symbol'], errors='ignore')
        
        # Rename columns to match expected format
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
            
        # Drop any extra columns not needed (like trade_count, vwap)
        if all(col in df.columns for col in cols):
            df = df[cols]
        else:
            missing_cols = [col for col in cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if verbose:
            print(f"Data shape after cleaning: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        ohlcv_wbuf = df
        
    except Exception as e:
        print(f"Error getting data from Alpaca: {str(e)}")
        print("Falling back to Yahoo Finance...")
        ohlcv_wbuf = vbt.YFData.download(symbol, start=start_date-time_buffer, end=end_date).get(cols)
    
    data_time = time.time()
    if verbose:
        print(f"Data acquisition completed in {data_time - start_time:.2f} seconds")
    
    # Convert to float64
    ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)
    
    # Create a copy of data without time buffer
    wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date)
    ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]
    
    if verbose:
        print("Calculating moving averages...")
    
    # Pre-calculate running windows on data with time buffer
    buy_fast_ma = vbt.MA.run(ohlcv_wbuf['Close'], buy_fast_window)
    buy_slow_ma = vbt.MA.run(ohlcv_wbuf['Close'], buy_slow_window)
    sell_fast_ma = vbt.MA.run(ohlcv_wbuf['Close'], sell_fast_window)
    sell_slow_ma = vbt.MA.run(ohlcv_wbuf['Close'], sell_slow_window)
    
    # Remove time buffer
    buy_fast_ma = buy_fast_ma[wobuf_mask]
    buy_slow_ma = buy_slow_ma[wobuf_mask]
    sell_fast_ma = sell_fast_ma[wobuf_mask]
    sell_slow_ma = sell_slow_ma[wobuf_mask]
    
    if verbose:
        print("Generating trading signals...")
    
    # Generate crossover signals using numba-optimized functions
    # For buy signals: buy_fast_ma crosses above buy_slow_ma
    # For sell signals: sell_fast_ma crosses below sell_slow_ma
    
    # Convert the VectorBT objects to numpy arrays for numba
    buy_fast_arr = buy_fast_ma.ma.values
    buy_slow_arr = buy_slow_ma.ma.values
    sell_fast_arr = sell_fast_ma.ma.values
    sell_slow_arr = sell_slow_ma.ma.values
    
    # Use numba-accelerated functions to find crossover points
    qmac_entries_arr = calculate_cross_signals(buy_fast_arr, buy_slow_arr)
    qmac_exits_arr = calculate_cross_below_signals(sell_fast_arr, sell_slow_arr)
    
    # Convert back to pandas Series
    qmac_entries = pd.Series(qmac_entries_arr, index=ohlcv.index)
    qmac_exits = pd.Series(qmac_exits_arr, index=ohlcv.index)
    
    # Set frequency for metrics based on METRICS_FREQUENCY
    freq = None
    if METRICS_FREQUENCY == 'd':
        freq = '1D'
    elif METRICS_FREQUENCY == 'h':
        freq = '1H'
    elif METRICS_FREQUENCY == 'm':
        freq = '1min'
    elif METRICS_FREQUENCY == 'w':
        freq = '1W'
    else:
        freq = '1D'  # Default to daily
    
    if verbose:
        print(f"Using metrics frequency: {freq}")
        print(f"Building portfolios...")
    
    # Build portfolio
    qmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], qmac_entries, qmac_exits, freq=freq)
    
    # Build hold portfolio for comparison
    hold_entries = pd.Series.vbt.signals.empty_like(qmac_entries)
    hold_entries.iloc[0] = True
    hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
    hold_exits.iloc[-1] = True
    hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits, freq=freq)
    
    end_time = time.time()
    if verbose:
        print(f"Strategy execution completed in {end_time - start_time:.2f} seconds")
        print(f"Number of trades: {len(qmac_pf.trades)}")
    
    return {
        'ohlcv': ohlcv,
        'ohlcv_wbuf': ohlcv_wbuf,
        'buy_fast_ma': buy_fast_ma,
        'buy_slow_ma': buy_slow_ma,
        'sell_fast_ma': sell_fast_ma,
        'sell_slow_ma': sell_slow_ma,
        'qmac_entries': qmac_entries,
        'qmac_exits': qmac_exits,
        'qmac_pf': qmac_pf,
        'hold_pf': hold_pf
    }

@nb.njit(fastmath=True, cache=True)
def evaluate_window_combination(prices, buy_fast, buy_slow, sell_fast, sell_slow, window_size):
    """
    Evaluate a single window combination using numba for performance.
    This function calculates a simplified version of the strategy returns.
    
    Args:
        prices (numpy.ndarray): Price array
        buy_fast, buy_slow, sell_fast, sell_slow (int): Window sizes
        window_size (int): Maximum window size for pre-calculation
        
    Returns:
        float: Simplified return metric
    """
    n = len(prices)
    
    # Skip if insufficient data
    if n <= window_size:
        return 0.0
    
    # Pre-allocate arrays
    buy_fast_ma = np.zeros(n)
    buy_slow_ma = np.zeros(n)
    sell_fast_ma = np.zeros(n)
    sell_slow_ma = np.zeros(n)
    
    # Pre-compute sums for moving averages to avoid repeated calculations
    # Buy fast MA
    buy_fast_sum = 0.0
    for i in range(buy_fast):
        buy_fast_sum += prices[i]
    buy_fast_ma[buy_fast-1] = buy_fast_sum / buy_fast
    
    # Buy slow MA
    buy_slow_sum = 0.0
    for i in range(buy_slow):
        buy_slow_sum += prices[i]
    buy_slow_ma[buy_slow-1] = buy_slow_sum / buy_slow
    
    # Sell fast MA
    sell_fast_sum = 0.0
    for i in range(sell_fast):
        sell_fast_sum += prices[i]
    sell_fast_ma[sell_fast-1] = sell_fast_sum / sell_fast
    
    # Sell slow MA
    sell_slow_sum = 0.0
    for i in range(sell_slow):
        sell_slow_sum += prices[i]
    sell_slow_ma[sell_slow-1] = sell_slow_sum / sell_slow
    
    # Calculate MAs with rolling sum (much faster than np.mean)
    for i in range(buy_fast, n):
        buy_fast_sum = buy_fast_sum - prices[i-buy_fast] + prices[i-1]
        buy_fast_ma[i-1] = buy_fast_sum / buy_fast
    
    for i in range(buy_slow, n):
        buy_slow_sum = buy_slow_sum - prices[i-buy_slow] + prices[i-1]
        buy_slow_ma[i-1] = buy_slow_sum / buy_slow
    
    for i in range(sell_fast, n):
        sell_fast_sum = sell_fast_sum - prices[i-sell_fast] + prices[i-1]
        sell_fast_ma[i-1] = sell_fast_sum / sell_fast
    
    for i in range(sell_slow, n):
        sell_slow_sum = sell_slow_sum - prices[i-sell_slow] + prices[i-1]
        sell_slow_ma[i-1] = sell_slow_sum / sell_slow
    
    # Calculate signals and performance
    position = 0  # 0: out of market, 1: in the market
    entry_price = 0.0
    total_return = 0.0
    
    for i in range(window_size+1, n):
        # Buy signal: fast crosses above slow
        if position == 0 and buy_fast_ma[i-2] <= buy_slow_ma[i-2] and buy_fast_ma[i-1] > buy_slow_ma[i-1]:
            position = 1
            entry_price = prices[i-1]
        
        # Sell signal: fast crosses below slow
        elif position == 1 and sell_fast_ma[i-2] >= sell_slow_ma[i-2] and sell_fast_ma[i-1] < sell_slow_ma[i-1]:
            position = 0
            exit_price = prices[i-1]
            # Calculate return (accounting for fees/slippage)
            trade_return = (exit_price / entry_price) * (1.0 - 0.005) - 1.0  # 0.5% for fees+slippage
            total_return += trade_return
    
    # If still in position at the end, close position
    if position == 1:
        final_return = (prices[-1] / entry_price) * (1.0 - 0.0025) - 1.0  # Only exit fee
        total_return += final_return
    
    return total_return

def calculate_total_possible_combinations(min_window, max_window, window_step):
    """
    Calculate the total possible unique window combinations before sampling.
    
    Args:
        min_window (int): Minimum window size
        max_window (int): Maximum window size
        window_step (int): Step size between window values
        
    Returns:
        int: Total number of valid combinations
    """
    print("Calculating total possible combinations...")
    start_time = time.time()
    
    window_values = list(range(min_window, max_window+1, window_step))
    n_windows = len(window_values)
    
    if n_windows < 4:
        return 0
    
    # First, count potential buy pairs where fast < slow
    buy_pairs = []
    for i in range(n_windows):
        for j in range(i+1, n_windows):
            buy_pairs.append((i, j))
    
    # Display progress bar for counting combinations
    total_combinations = 0
    buy_pair_count = len(buy_pairs)
    
    print(f"Found {buy_pair_count:,} possible buy pairs. Calculating unique combinations...")
    
    with tqdm(total=buy_pair_count, desc="Counting Combinations") as pbar:
        # For each buy pair, count how many unique sell pairs we can form
        for buy_fast_idx, buy_slow_idx in buy_pairs:
            # This is a valid buy pair
            buy_fast = window_values[buy_fast_idx]
            buy_slow = window_values[buy_slow_idx]
            buy_values = {buy_fast, buy_slow}
            
            # Count valid sell pairs that don't overlap with buy values
            for sell_fast_idx in range(n_windows):
                sell_fast = window_values[sell_fast_idx]
                if sell_fast in buy_values:
                    continue  # Skip if not unique
                
                for sell_slow_idx in range(sell_fast_idx+1, n_windows):
                    sell_slow = window_values[sell_slow_idx]
                    if sell_slow in buy_values or sell_slow == sell_fast:
                        continue  # Skip if not unique
                    
                    # This is a valid combination
                    total_combinations += 1
            
            pbar.update(1)
    
    end_time = time.time()
    print(f"Calculation completed in {end_time - start_time:.2f} seconds")
    
    return total_combinations

def sample_unique_windows(min_window, max_window, window_step, count=100, total_possible=None):
    """
    Sample unique combinations of 4 window sizes more efficiently.
    
    Args:
        min_window (int): Minimum window size
        max_window (int): Maximum window size
        window_step (int): Step size between windows
        count (int): Number of combinations to sample
        total_possible (int, optional): Pre-calculated total possible combinations
        
    Returns:
        list: List of unique window combinations (buy_fast, buy_slow, sell_fast, sell_slow)
    """
    start_time = time.time()
    
    window_values = list(range(min_window, max_window+1, window_step))
    
    # First calculate how many window values we have
    n_windows = len(window_values)
    
    if n_windows < 4:
        # Not enough distinct window values for 4 unique MAs
        raise ValueError(f"Need at least 4 distinct window values. Current range produces only {n_windows}")
    
    print(f"Sampling from {n_windows} possible window sizes...")
    
    # Calculate total possible combinations if not provided
    if total_possible is None:
        total_possible = calculate_total_possible_combinations(min_window, max_window, window_step)
    else:
        print(f"Using pre-calculated total of {total_possible:,} possible combinations")
    
    # If requested count exceeds total possible, adjust
    if count > total_possible and count != 100000:  # 100000 is our proxy for unlimited
        count = total_possible
        print(f"Adjusted target to {count:,} combinations (maximum possible)")
    
    # For lower memory usage, we'll create a smaller initial set of valid combinations
    # by focusing on reasonable pairings and then sample from those
    valid_combinations = []
    
    # PHASE 1: Systematic sampling
    print("\nPHASE 1: Systematic sampling of window combinations...")
    
    # Generate combinations more efficiently
    # Start with a smaller sample of buy-side windows
    buy_indices = []
    for i in range(0, n_windows-1, max(1, (n_windows-1)//10)):  # Pick ~10 indices for fast
        for j in range(i+1, n_windows, max(1, (n_windows-i-1)//5)):  # Pick ~5 slow for each fast
            buy_indices.append((i, j))
    
    # Start with a smaller sample of sell-side windows - different from buy side
    sell_indices = []
    for i in range(0, n_windows-1, max(1, (n_windows-1)//10)):
        for j in range(i+1, n_windows, max(1, (n_windows-i-1)//5)):
            sell_indices.append((i, j))
    
    print(f"Testing {len(buy_indices)} buy pairs × {len(sell_indices)} sell pairs = {len(buy_indices)*len(sell_indices):,} potential combinations")
    
    # Create combinations ensuring all 4 values are unique with tqdm progress bar
    with tqdm(total=min(len(buy_indices) * len(sell_indices), count), desc="Systematic Sampling") as pbar:
        for buy_i, buy_j in buy_indices:
            buy_fast = window_values[buy_i]
            buy_slow = window_values[buy_j]
            
            for sell_i, sell_j in sell_indices:
                sell_fast = window_values[sell_i]
                sell_slow = window_values[sell_j]
                
                # Check if all 4 windows are unique
                if len({buy_fast, buy_slow, sell_fast, sell_slow}) == 4:
                    valid_combinations.append((buy_fast, buy_slow, sell_fast, sell_slow))
                    pbar.update(1)
                    
                    if len(valid_combinations) >= count:
                        # If we have enough combinations, stop
                        print(f"\nFound {len(valid_combinations):,} combinations in Phase 1")
                        print(f"Window sampling completed in {time.time() - start_time:.2f} seconds")
                        return valid_combinations
    
    print(f"\nFound {len(valid_combinations):,} combinations in Phase 1")
    
    # PHASE 2: If we haven't reached the required count yet, use a comprehensive approach
    if len(valid_combinations) < count and len(valid_combinations) < total_possible:
        remaining = min(count - len(valid_combinations), total_possible - len(valid_combinations))
        print(f"\nPHASE 2: Comprehensive generation of remaining {remaining:,} combinations...")
        
        # Comprehensive approach: check all possible combinations
        # We can be smarter by starting where we left off in the systematic approach
        existing_set = set(valid_combinations)
        
        # Setup progress tracking
        total_buy_pairs = n_windows * (n_windows - 1) // 2
        progress_interval = max(1, total_buy_pairs // 100)  # Update every 1%
        progress_counter = 0
        last_update_time = time.time()
        last_progress = 0
        
        # Initialize progress bar for overall Phase 2 progress
        with tqdm(total=remaining, desc="Finding Combinations") as main_pbar:
            inner_pbar = tqdm(total=total_buy_pairs, desc="Scanning Buy Pairs")
            
            # For all possible buy pairs
            for buy_fast_idx in range(n_windows):
                for buy_slow_idx in range(buy_fast_idx+1, n_windows):
                    buy_fast = window_values[buy_fast_idx]
                    buy_slow = window_values[buy_slow_idx]
                    
                    # Update buy pair progress
                    progress_counter += 1
                    if progress_counter % progress_interval == 0:
                        inner_pbar.update(progress_interval)
                    
                    # For all possible sell pairs
                    combinations_found_in_this_pass = 0
                    for sell_fast_idx in range(n_windows):
                        sell_fast = window_values[sell_fast_idx]
                        
                        # Skip if not unique with buy values
                        if sell_fast == buy_fast or sell_fast == buy_slow:
                            continue
                        
                        for sell_slow_idx in range(sell_fast_idx+1, n_windows):
                            sell_slow = window_values[sell_slow_idx]
                            
                            # Skip if not unique with other values
                            if sell_slow == buy_fast or sell_slow == buy_slow or sell_slow == sell_fast:
                                continue
                            
                            combo = (buy_fast, buy_slow, sell_fast, sell_slow)
                            if combo not in existing_set:
                                valid_combinations.append(combo)
                                existing_set.add(combo)
                                combinations_found_in_this_pass += 1
                                
                                if len(valid_combinations) >= count or len(valid_combinations) >= total_possible:
                                    # Close progress bars
                                    inner_pbar.close()
                                    main_pbar.update(combinations_found_in_this_pass)
                                    
                                    end_time = time.time()
                                    print(f"\nPhase 2 completed. Total combinations found: {len(valid_combinations):,}")
                                    print(f"Window sampling completed in {end_time - start_time:.2f} seconds")
                                    return valid_combinations
                    
                    # Update overall progress bar based on combinations found
                    if combinations_found_in_this_pass > 0:
                        main_pbar.update(combinations_found_in_this_pass)
                    
                    # Estimate and print time remaining periodically
                    current_time = time.time()
                    if current_time - last_update_time > 5:  # Update every 5 seconds
                        current_progress = progress_counter / total_buy_pairs
                        progress_since_last = current_progress - last_progress
                        time_since_last = current_time - last_update_time
                        
                        if progress_since_last > 0:
                            est_total_time = time_since_last * (1.0 / progress_since_last)
                            est_remaining = est_total_time * (1.0 - current_progress)
                            
                            # Format as hours:minutes:seconds
                            hours, remainder = divmod(est_remaining, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            
                            if hours >= 1:
                                time_str = f"{int(hours)}h {int(minutes)}m remaining"
                            elif minutes >= 1:
                                time_str = f"{int(minutes)}m {int(seconds)}s remaining"
                            else:
                                time_str = f"{int(seconds)}s remaining"
                            
                            print(f"\nProgress: {current_progress:.1%}, {time_str}, found {len(valid_combinations):,} combinations")
                            
                            last_update_time = current_time
                            last_progress = current_progress
            
            # Close inner progress bar if it hasn't been closed yet
            inner_pbar.close()
    
    end_time = time.time()
    print(f"\nTotal combinations found: {len(valid_combinations):,}")
    print(f"Window sampling completed in {end_time - start_time:.2f} seconds")
    
    return valid_combinations

# Ray remote function for distributed combination evaluation
@ray.remote
def evaluate_window_combinations_batch(combinations, prices, max_window_size):
    """
    Evaluate a batch of window combinations using Ray for distributed computing.
    
    Args:
        combinations (list): List of window combinations to evaluate
        prices (numpy.ndarray): Price array
        max_window_size (int): Maximum window size for pre-calculation
        
    Returns:
        list: List of (combination, performance) tuples
    """
    results = []
    for buy_fast, buy_slow, sell_fast, sell_slow in combinations:
        perf = evaluate_window_combination(prices, buy_fast, buy_slow, sell_fast, sell_slow, max_window_size)
        results.append(((buy_fast, buy_slow, sell_fast, sell_slow), perf))
    return results

def analyze_window_combinations_ray(symbol, start_date, end_date, 
                                   min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                                   window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                                   timeframe=DEFAULT_TIMEFRAME, verbose=True,
                                   max_combinations=MAX_COMBINATIONS, num_cpus=None):
    """
    Analyze window combinations using Ray for distributed computing.
    
    Args:
        symbol (str): Trading symbol
        start_date (datetime): Start date for backtest
        end_date (datetime): End date for backtest
        min_window (int): Minimum window size
        max_window (int): Maximum window size
        window_step (int): Step size between windows
        metric (str): Performance metric
        timeframe (str): Data timeframe
        verbose (bool): Whether to print detailed output
        max_combinations (int): Maximum combinations to test
        num_cpus (int): Number of CPUs to use (None for auto)
        
    Returns:
        dict: Analysis results
    """
    start_time = time.time()
    
    # Get CPU count for Ray
    if num_cpus is None:
        num_cpus = os.cpu_count()
    
    print(f"Running Ray-powered distributed optimization using {num_cpus} CPUs")
    
    # Get data for testing
    single_result = run_qmac_strategy(
        symbol, start_date, end_date, 
        buy_fast_window=min_window, 
        buy_slow_window=max_window,
        sell_fast_window=min_window+1,
        sell_slow_window=max_window-1,
        timeframe=timeframe,
        verbose=verbose
    )
    
    # Get price data
    prices = single_result['ohlcv']['Close'].values
    
    # Sample window combinations
    window_combinations = sample_unique_windows(
        min_window, max_window, window_step, 
        count=max_combinations)
    
    total_combinations = len(window_combinations)
    print(f"Testing {total_combinations:,} combinations with Ray distributed computing")
    
    # Split combinations into batches
    batch_size = max(1, total_combinations // (num_cpus * 10))  # 10 batches per CPU
    batches = [window_combinations[i:i+batch_size] for i in range(0, total_combinations, batch_size)]
    num_batches = len(batches)
    
    print(f"Splitting work into {num_batches} batches of ~{batch_size} combinations each")
    
    # Maximum window size for calculating MAs
    max_window_size = max(max_window, 
                        max([max(c[1], c[3]) for c in window_combinations]))
    
    # Put prices in Ray object store
    ray_prices = ray.put(prices)
    
    # Submit tasks to Ray
    print("Submitting tasks to Ray cluster...")
    future_results = [evaluate_window_combinations_batch.remote(batch, ray_prices, max_window_size) 
                     for batch in batches]
    
    # Process results as they complete
    print("Processing results (may take some time)...")
    all_results = []
    
    with tqdm(total=len(future_results), desc="Processing batches") as pbar:
        while future_results:
            # Wait for a batch to complete
            done_id, future_results = ray.wait(future_results, num_returns=1)
            
            # Get the result and add to all_results
            batch_results = ray.get(done_id[0])
            all_results.extend(batch_results)
            
            # Update progress bar
            pbar.update(1)
    
    # Find the best combination
    best_result = max(all_results, key=lambda x: x[1])
    best_combination, best_perf = best_result
    
    print(f"\nOptimal window combination:")
    print(f"Buy signals: Fast MA = {best_combination[0]}, Slow MA = {best_combination[1]}")
    print(f"Sell signals: Fast MA = {best_combination[2]}, Slow MA = {best_combination[3]}")
    print(f"Optimal performance ({metric}): {best_perf:.2%}")
    
    # Run optimal strategy to get full results
    buy_fast, buy_slow, sell_fast, sell_slow = best_combination
    optimal_results = run_qmac_strategy(
        symbol, start_date, end_date,
        buy_fast_window=buy_fast,
        buy_slow_window=buy_slow,
        sell_fast_window=sell_fast,
        sell_slow_window=sell_slow,
        timeframe=timeframe,
        verbose=False
    )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {'buy_fast': c[0], 'buy_slow': c[1], 'sell_fast': c[2], 'sell_slow': c[3], metric: p}
        for (c, p) in all_results
    ])
    
    end_time = time.time()
    print(f"Ray-powered optimization completed in {end_time - start_time:.2f} seconds")
    
    return {
        'ohlcv': single_result['ohlcv'],
        'performance_df': results_df,
        'optimal_windows': best_combination,
        'optimal_performance': best_perf,
        'optimal_results': optimal_results  # Add the optimal results to the return dictionary
    }

def analyze_window_combinations(symbol, start_date, end_date, 
                              min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                              window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                              timeframe=DEFAULT_TIMEFRAME, single_result=None, verbose=True,
                              max_combinations=MAX_COMBINATIONS, total_possible=None, 
                              use_ray=True, num_cpus=None):
    """
    Analyze multiple window combinations for QMAC strategy.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        start_date (datetime): Start date for the backtest
        end_date (datetime): End date for the backtest
        min_window (int): Minimum window size to test
        max_window (int): Maximum window size to test
        window_step (int): Step size between window values
        metric (str): Performance metric to optimize for
        timeframe (str): Timeframe for data (e.g., '1d', '1h', '15m')
        single_result (dict, optional): Result from a previous run_qmac_strategy call
        verbose (bool): Whether to print detailed output
        max_combinations (int): Maximum number of window combinations to test (-1 for unlimited)
        total_possible (int, optional): Pre-calculated total possible combinations
        use_ray (bool): Whether to use Ray for distributed computation
        num_cpus (int): Number of CPUs to use (None for auto)
        
    Returns:
        dict: Dictionary containing the analysis results
    """
    # If Ray is enabled, use Ray-based distributed analysis
    if use_ray:
        return analyze_window_combinations_ray(
            symbol, start_date, end_date,
            min_window=min_window, max_window=max_window,
            window_step=window_step, metric=metric,
            timeframe=timeframe, verbose=verbose,
            max_combinations=max_combinations, num_cpus=num_cpus
        )
    
    # Otherwise use the original implementation
    start_time = time.time()
    
    # Handle unlimited combinations mode
    if max_combinations < 0:
        print(f"Running in UNLIMITED combinations mode. This may take a long time...")
        # Set a very high number instead of truly unlimited to avoid memory issues
        max_combinations = 100000
    else:
        print(f"Starting window optimization with up to {max_combinations:,} combinations...")
    
    # Set portfolio parameters
    vbt.settings.portfolio['init_cash'] = INITIAL_CASH
    vbt.settings.portfolio['fees'] = FEES
    vbt.settings.portfolio['slippage'] = SLIPPAGE
    
    # Get data from single_result or run a new strategy
    if single_result is None:
        if verbose:
            print("No existing result provided, downloading data...")
        # Use basic windows for initial data download
        single_result = run_qmac_strategy(
            symbol, start_date, end_date, 
            buy_fast_window=min_window, 
            buy_slow_window=max_window,
            sell_fast_window=min_window+1,
            sell_slow_window=max_window-1,
            timeframe=timeframe,
            verbose=verbose
        )
    
    # Use the data directly from the single result
    ohlcv = single_result['ohlcv']
    prices = ohlcv['Close'].values  # Convert to numpy array for numba
    
    # Try-except block to handle potential errors in sampling
    try:
        # Get a sample of unique window combinations
        window_combinations = sample_unique_windows(
            min_window, max_window, window_step, 
            count=max_combinations,
            total_possible=total_possible)
    except Exception as e:
        print(f"Error sampling window combinations: {str(e)}")
        print("Using a simplified sampling approach...")
        
        # Fallback to a simpler approach
        window_values = list(range(min_window, max_window+1, window_step))
        n_windows = len(window_values)
        
        if n_windows < 4:
            # Not enough values for 4 unique windows
            print("WARNING: Not enough window values available for 4 unique windows")
            window_values = list(range(min_window, max_window+1, 1))
            n_windows = len(window_values)
        
        # Create some simple combinations
        window_combinations = []
        for i in tqdm(range(10), desc="Creating backup combinations"):  # Try 10 combinations with increasing spacing
            spacing = max(1, n_windows // 8 * (i+1))
            indices = [min(n_windows-1, max(0, n_windows//8 * j)) for j in range(4)]
            
            buy_fast = window_values[indices[0]]
            buy_slow = window_values[indices[1]]
            sell_fast = window_values[indices[2]]
            sell_slow = window_values[indices[3]]
            
            # Ensure buy_fast < buy_slow and sell_fast < sell_slow
            if buy_fast >= buy_slow:
                buy_fast, buy_slow = buy_slow, buy_fast
            if sell_fast >= sell_slow:
                sell_fast, sell_slow = sell_slow, sell_fast
                
            # Ensure all 4 are unique - adjust if needed
            values = [buy_fast, buy_slow, sell_fast, sell_slow]
            if len(set(values)) < 4:
                # Try to make them unique by adding small offsets
                values = list(set(values))
                while len(values) < 4 and values[-1] < max_window:
                    values.append(values[-1] + window_step)
                
                if len(values) >= 4:
                    buy_fast, buy_slow, sell_fast, sell_slow = sorted(values[:4])
                    
            window_combinations.append((buy_fast, buy_slow, sell_fast, sell_slow))
    
    if verbose:
        print(f"Testing {len(window_combinations)} window combinations...")

    # Check for existing checkpoint
    checkpoint_file = f"backtests/qmac_strategy/checkpoints/qmac_checkpoint_{symbol}_{timeframe}.json"
    resume_from_checkpoint = False
    start_batch = 0
    best_perf_so_far = 0.0
    best_combo_so_far = None
    
    try:
        import json
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_combinations = checkpoint_data.get('completed_combinations', 0)
                start_batch = completed_combinations // 10000  # Using batch size 10000
                best_perf_so_far = checkpoint_data.get('best_performance', 0.0)
                best_combo_so_far = checkpoint_data.get('best_combination')
                
                print(f"Resuming from checkpoint: {completed_combinations:,}/{len(window_combinations):,} combinations tested")
                if best_combo_so_far:
                    print(f"Best combination so far: {best_combo_so_far} with performance: {best_perf_so_far:.2%}")
                resume_from_checkpoint = True
    except Exception as e:
        print(f"Error reading checkpoint: {e}. Starting from beginning.")
    
    # Convert combinations to numpy arrays for Numba
    buy_fast_arr = np.array([c[0] for c in window_combinations])
    buy_slow_arr = np.array([c[1] for c in window_combinations])
    sell_fast_arr = np.array([c[2] for c in window_combinations])
    sell_slow_arr = np.array([c[3] for c in window_combinations])
    
    # Maximum window size for calculating MAs
    max_window_size = max(np.max(buy_slow_arr), np.max(sell_slow_arr))
    
    # Process in batches to show progress and prevent memory issues
    batch_size = 10000
    num_batches = (len(window_combinations) + batch_size - 1) // batch_size
    
    print(f"Starting Numba-accelerated parallel processing...")
    print(f"Testing a total of {len(window_combinations):,} combinations")
    
    # Add a prominent, static visual separator for testing phase
    print("\n" + "="*100)
    print(f"{'='*30} STARTING TESTING OF {len(window_combinations):,} COMBINATIONS {'='*30}")
    print("="*100)
    
    # Display estimated completion time
    estimated_combinations_per_second = 90  # Based on previous runs
    estimated_seconds = len(window_combinations) / estimated_combinations_per_second
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\nEstimated processing speed: ~{estimated_combinations_per_second} combinations/second")
    print(f"Estimated completion time: {estimated_hours:.1f} hours ({estimated_days:.1f} days)")
    
    # Create a static progress counter display
    print("\nPROGRESS TRACKING:")
    print(f"[{'_'*50}] 0% complete")
    print(f"0/{len(window_combinations):,} combinations processed")
    print(f"Starting from batch {start_batch+1}/{num_batches}")
    print("="*100 + "\n")
    
    # Prepare results array
    if resume_from_checkpoint and start_batch > 0:
        # Load partial results from checkpoint if available
        try:
            results = np.zeros(len(window_combinations))
            results_file = f"backtests/qmac_strategy/results/qmac_results_{symbol}_{timeframe}.npy"
            if os.path.exists(results_file):
                partial_results = np.load(results_file)
                # Make sure arrays are compatible
                if len(partial_results) == len(results):
                    results = partial_results
                    print(f"Loaded partial results from {results_file}")
        except Exception as e:
            print(f"Error loading partial results: {e}. Starting with fresh results array.")
            results = np.zeros(len(window_combinations))
    else:
        results = np.zeros(len(window_combinations))
    
    # Track overall progress for periodic reports
    total_tested = start_batch * batch_size
    last_report_time = time.time()
    testing_start_time = time.time()
    
    # Array to store timing data for individual combinations
    timing_data = []
    
    # Enhanced progress bar setup
    from tqdm.auto import tqdm, trange
    
    # Create main progress bar for all combinations
    main_pbar = tqdm(total=len(window_combinations), 
                     desc="Testing combinations", 
                     unit="combo",
                     initial=total_tested,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {desc}]")
    
    # Warm up Numba by testing a few combinations first (helps with JIT compilation)
    print("Warming up Numba JIT compiler...")
    for _ in range(5):
        _ = evaluate_window_combination(prices, 10, 20, 15, 30, max_window_size)
    
    # Process batches
    try:
        for batch_idx in range(start_batch, num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(window_combinations))
            batch_size_actual = end_idx - start_idx
            
            # Extract batch arrays
            bf_batch = buy_fast_arr[start_idx:end_idx]
            bs_batch = buy_slow_arr[start_idx:end_idx]
            sf_batch = sell_fast_arr[start_idx:end_idx]
            ss_batch = sell_slow_arr[start_idx:end_idx]
            
            # Create batch progress bar
            batch_desc = f"Batch {batch_idx+1}/{num_batches}"
            batch_pbar = tqdm(total=batch_size_actual, desc=batch_desc, 
                            leave=False, position=1, unit="combo",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            
            # Signal that testing is actually beginning
            if batch_idx == start_batch and start_idx == 0:
                print("\n" + "🚀"*40)
                print("TESTING PHASE ACTIVELY STARTING - FIRST COMBINATIONS BEING PROCESSED")
                print("🚀"*40 + "\n")
            elif batch_idx == start_batch:
                print("\n" + "🚀"*40)
                print(f"RESUMING TESTING FROM COMBINATION {start_idx+1:,}")
                print("🚀"*40 + "\n")
            
            # Process batch using numba-accelerated function
            batch_results = np.zeros(batch_size_actual)
            
            # Timing variables for performance stats
            combo_times = []
            
            for i in range(batch_size_actual):
                combo_start = time.time()
                
                # Use the numba-optimized evaluation function
                batch_results[i] = evaluate_window_combination(
                    prices, 
                    bf_batch[i], bs_batch[i], 
                    sf_batch[i], ss_batch[i], 
                    max_window_size)
                
                # Calculate time taken
                combo_end = time.time()
                combo_time = combo_end - combo_start
                combo_times.append(combo_time)
                
                # Update main progress and batch progress
                main_pbar.update(1)
                batch_pbar.update(1)
                
                # Update total tested count
                total_tested += 1
                
                # Update timing stats periodically (every 25 combinations)
                if len(combo_times) >= 25:
                    avg_time = sum(combo_times) / len(combo_times)
                    median_time = sorted(combo_times)[len(combo_times)//2]
                    min_time = min(combo_times)
                    max_time = max(combo_times)
                    
                    # Store timing data
                    timing_data.append({
                        'batch': batch_idx,
                        'combinations_tested': total_tested,
                        'avg_time': avg_time,
                        'median_time': median_time,
                        'min_time': min_time,
                        'max_time': max_time
                    })
                    
                    # Update description with timing information
                    main_pbar.set_description(f"Testing: {avg_time*1000:.1f}ms/combo")
                    
                    # Reset combo times
                    combo_times = []
                
                # Generate more detailed logs periodically
                current_time = time.time()
                if current_time - last_report_time > 30:
                    elapsed = current_time - testing_start_time
                    combinations_per_second = total_tested / elapsed if elapsed > 0 else 0
                    
                    # Estimate time remaining
                    if combinations_per_second > 0:
                        remaining_combinations = len(window_combinations) - total_tested
                        est_seconds_remaining = remaining_combinations / combinations_per_second
                        
                        # Format time remaining
                        if est_seconds_remaining < 60:
                            time_remaining = f"{est_seconds_remaining:.1f} seconds"
                        elif est_seconds_remaining < 3600:
                            time_remaining = f"{est_seconds_remaining/60:.1f} minutes"
                        elif est_seconds_remaining < 86400:
                            hours = int(est_seconds_remaining // 3600)
                            minutes = int((est_seconds_remaining % 3600) // 60)
                            time_remaining = f"{hours}h {minutes}m"
                        else:
                            days = int(est_seconds_remaining // 86400)
                            hours = int((est_seconds_remaining % 86400) // 3600)
                            time_remaining = f"{days}d {hours}h"
                        
                        # Calculate progress percentage
                        progress_percent = total_tested / len(window_combinations) * 100
                        
                        # Create a progress bar
                        bar_length = 50
                        filled_length = int(bar_length * total_tested // len(window_combinations))
                        bar = '█' * filled_length + '_' * (bar_length - filled_length)
                        
                        # Print a prominent progress update
                        print("\n" + "-"*100)
                        print(f"PROGRESS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"[{bar}] {progress_percent:.2f}% complete")
                        print(f"Combinations tested: {total_tested:,}/{len(window_combinations):,}")
                        print(f"Processing speed: {combinations_per_second:.2f} combinations/second")
                        print(f"Time elapsed: {elapsed/3600:.1f} hours")
                        print(f"Estimated time remaining: {time_remaining}")
                        print(f"Current batch: {batch_idx+1}/{num_batches}")
                        if best_combo_so_far:
                            print(f"Best combination so far: {best_combo_so_far} with performance: {best_perf_so_far:.2%}")
                        print("-"*100)
                    
                    last_report_time = current_time
            
            # Close batch progress bar
            batch_pbar.close()
            
            # Add batch completion indicator
            batch_progress_percent = (batch_idx + 1) / num_batches * 100
            progress_bar = '█' * int(batch_progress_percent // 2) + '░' * (50 - int(batch_progress_percent // 2))
            print(f"\nBatch {batch_idx+1}/{num_batches} completed [{progress_bar}] {batch_progress_percent:.1f}%")
            print(f"Total combinations processed: {total_tested:,}/{len(window_combinations):,} ({total_tested/len(window_combinations)*100:.2f}%)")
            
            # Store batch results
            results[start_idx:end_idx] = batch_results
            
            # Find best result in this batch
            batch_best_idx = np.argmax(batch_results)
            batch_best_perf = batch_results[batch_best_idx]
            
            # Update overall best if this batch has a better result
            if batch_best_perf > best_perf_so_far:
                best_perf_so_far = float(batch_best_perf)
                best_combo_so_far = window_combinations[start_idx + batch_best_idx]
                
                # Run the actual strategy with these parameters to get real returns
                buy_fast, buy_slow, sell_fast, sell_slow = best_combo_so_far
                print(f"\nNew potential best combination found in batch {batch_idx+1}: {best_combo_so_far}")
                print(f"Running full strategy to calculate actual returns...")
                
                try:
                    # Run the full strategy with best parameters
                    actual_result = run_qmac_strategy(
                        symbol, start_date, end_date, 
                        buy_fast_window=buy_fast, 
                        buy_slow_window=buy_slow,
                        sell_fast_window=sell_fast, 
                        sell_slow_window=sell_slow,
                        timeframe=timeframe,
                        verbose=False
                    )
                    
                    # Get the actual total return
                    actual_return = actual_result['qmac_pf'].total_return()
                    print(f"New best combination found: {best_combo_so_far} with actual return: {actual_return:.2%}")
                    
                    # Also save this information in the checkpoint
                    best_actual_return = actual_return
                except Exception as e:
                    print(f"Error calculating actual return: {e}")
                    print(f"Using estimated performance: {best_perf_so_far:.2%}")
                    best_actual_return = None
            
            # Save checkpoint after each batch
            checkpoint_data = {
                'completed_combinations': total_tested,
                'total_combinations': len(window_combinations),
                'best_combination_idx': int(np.argmax(results[:end_idx])),
                'best_combination': window_combinations[int(np.argmax(results[:end_idx]))],
                'best_performance': float(np.max(results[:end_idx])),
                'best_actual_return': best_actual_return if 'best_actual_return' in locals() else None,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': current_time - testing_start_time
            }
            
            # Save checkpoint to file
            try:
                import json
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"Saved checkpoint to {checkpoint_file}")
                
                # Also save results array for potential resume
                results_file = f"backtests/qmac_strategy/results/qmac_results_{symbol}_{timeframe}.npy"
                np.save(results_file, results)
                
                # Save timing data
                timing_file = f"backtests/qmac_strategy/timing/qmac_timing_{symbol}_{timeframe}.json"
                with open(timing_file, 'w') as f:
                    json.dump(timing_data, f)
            except Exception as e:
                print(f"Error saving files: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        # Save checkpoint
        try:
            checkpoint_data = {
                'completed_combinations': total_tested,
                'total_combinations': len(window_combinations),
                'best_combination_idx': int(np.argmax(results[:total_tested])),
                'best_combination': window_combinations[int(np.argmax(results[:total_tested]))],
                'best_performance': float(np.max(results[:total_tested])),
                'best_actual_return': best_actual_return if 'best_actual_return' in locals() else None,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': time.time() - testing_start_time,
                'interrupted': True
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"Saved checkpoint to {checkpoint_file}")
            
            # Save partial results
            results_file = f"backtests/qmac_strategy/results/qmac_results_{symbol}_{timeframe}.npy"
            np.save(results_file, results)
            print(f"Saved partial results to {results_file}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    finally:
        # Close progress bar
        main_pbar.close()
    
    # Find best combination overall
    if total_tested > 0:
        best_idx = np.argmax(results[:total_tested])
        optimal_windows = window_combinations[best_idx]
        optimal_perf = results[best_idx]
    else:
        # Default if no combinations were tested
        optimal_windows = (min_window, max_window, min_window+1, max_window-1)
        optimal_perf = 0.0
    
    # Convert results to DataFrame for easier analysis
    columns = ['buy_fast', 'buy_slow', 'sell_fast', 'sell_slow', metric]
    performance_df = pd.DataFrame({
        'buy_fast': buy_fast_arr[:total_tested],
        'buy_slow': buy_slow_arr[:total_tested],
        'sell_fast': sell_fast_arr[:total_tested],
        'sell_slow': sell_slow_arr[:total_tested],
        metric: results[:total_tested]
    })
    
    # Save final results
    try:
        results_csv = f"backtests/qmac_strategy/results/qmac_results_{symbol}_{timeframe}.csv"
        performance_df.to_csv(results_csv)
        print(f"Saved complete results to {results_csv}")
    except Exception as e:
        print(f"Error saving results CSV: {e}")
    
    # Print optimal windows
    buy_fast, buy_slow, sell_fast, sell_slow = optimal_windows
    print(f"\nOptimal window combination:")
    print(f"Buy signals: Fast MA = {buy_fast}, Slow MA = {buy_slow}")
    print(f"Sell signals: Fast MA = {sell_fast}, Slow MA = {sell_slow}")
    print(f"Optimal performance ({metric}): {optimal_perf:.2%}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Optimization completed in {total_time:.2f} seconds")
    
    if total_tested > 0:
        avg_time_per_combo = total_time / total_tested
        print(f"Average time per combination: {avg_time_per_combo*1000:.2f} ms")
        print(f"Combinations per second: {1.0/avg_time_per_combo:.2f}")
    
    return {
        'ohlcv': ohlcv,
        'performance_df': performance_df,
        'optimal_windows': optimal_windows,
        'optimal_performance': optimal_perf
    }

def calculate_performance_standard(price_series, buy_fast, buy_slow, sell_fast, sell_slow, freq, metric):
    """
    Calculate performance using vectorbt's standard method.
    
    Args:
        price_series (Series): Price series
        buy_fast, buy_slow, sell_fast, sell_slow (int): Window parameters
        freq (str): Frequency for metrics calculation
        metric (str): Performance metric to return
        
    Returns:
        float: Performance value
    """
    # Calculate moving averages
    buy_fast_ma = vbt.MA.run(price_series, buy_fast)
    buy_slow_ma = vbt.MA.run(price_series, buy_slow)
    sell_fast_ma = vbt.MA.run(price_series, sell_fast)
    sell_slow_ma = vbt.MA.run(price_series, sell_slow)
    
    # Generate signals
    entries = buy_fast_ma.ma_crossed_above(buy_slow_ma)
    exits = sell_fast_ma.ma_crossed_below(sell_slow_ma)
    
    # Create portfolio
    pf = vbt.Portfolio.from_signals(price_series, entries, exits, freq=freq)
    
    # Get performance metric
    return pf.deep_getattr(metric)

def plot_qmac_strategy(results):
    """
    Plot the QMAC strategy results.
    
    Args:
        results (dict): Results from run_qmac_strategy
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Plot the OHLC data with MA lines and entry/exit points
    fig = results['ohlcv']['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
    fig = results['buy_fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Buy Fast MA'), fig=fig)
    fig = results['buy_slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Buy Slow MA'), fig=fig)
    fig = results['sell_fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Sell Fast MA'), fig=fig)
    fig = results['sell_slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Sell Slow MA'), fig=fig)
    fig = results['qmac_entries'].vbt.signals.plot_as_entry_markers(results['ohlcv']['Open'], fig=fig)
    fig = results['qmac_exits'].vbt.signals.plot_as_exit_markers(results['ohlcv']['Open'], fig=fig)
    
    # Plot equity comparison
    value_fig = results['qmac_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (QMAC)'))
    results['hold_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=value_fig)
    
    # Plot trades
    trades_fig = results['qmac_pf'].trades.plot()
    
    return {
        'strategy_fig': fig,
        'value_fig': value_fig,
        'trades_fig': trades_fig
    }

def create_parameter_space_visualization(performance_df, symbol, start_date, end_date):
    """
    Create visualizations of the 4D parameter space.
    
    Args:
        performance_df (DataFrame): DataFrame containing performance data for all combinations
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Calculate ratios for visualization
    performance_df['buy_ratio'] = performance_df['buy_fast'] / performance_df['buy_slow']
    performance_df['sell_ratio'] = performance_df['sell_fast'] / performance_df['sell_slow']
    
    # Create 2D heatmaps for buy and sell parameters
    buy_heatmap = go.Figure(data=go.Heatmap(
        z=performance_df.groupby(['buy_fast', 'buy_slow'])['total_return'].mean().unstack(),
        x=performance_df['buy_slow'].unique(),
        y=performance_df['buy_fast'].unique(),
        colorscale='RdYlGn',
        colorbar=dict(title='Average Return')
    ))
    buy_heatmap.update_layout(
        title=f'Buy Parameters Performance Heatmap - {symbol}',
        xaxis_title='Buy Slow Window',
        yaxis_title='Buy Fast Window'
    )
    
    sell_heatmap = go.Figure(data=go.Heatmap(
        z=performance_df.groupby(['sell_fast', 'sell_slow'])['total_return'].mean().unstack(),
        x=performance_df['sell_slow'].unique(),
        y=performance_df['sell_fast'].unique(),
        colorscale='RdYlGn',
        colorbar=dict(title='Average Return')
    ))
    sell_heatmap.update_layout(
        title=f'Sell Parameters Performance Heatmap - {symbol}',
        xaxis_title='Sell Slow Window',
        yaxis_title='Sell Fast Window'
    )
    
    # Create 3D surface plot
    surface_plot = go.Figure(data=go.Surface(
        z=performance_df.groupby(['buy_ratio', 'sell_ratio'])['total_return'].mean().unstack(),
        x=performance_df['sell_ratio'].unique(),
        y=performance_df['buy_ratio'].unique(),
        colorscale='RdYlGn'
    ))
    surface_plot.update_layout(
        title=f'Performance Surface - {symbol}',
        scene=dict(
            xaxis_title='Sell Ratio',
            yaxis_title='Buy Ratio',
            zaxis_title='Return'
        )
    )
    
    # Create parallel coordinates plot
    top_performers = performance_df.nlargest(100, 'total_return')
    parallel_plot = go.Figure(data=go.Parcoords(
        line=dict(
            color=top_performers['total_return'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Return')
        ),
        dimensions=[
            dict(label='Buy Fast', values=top_performers['buy_fast']),
            dict(label='Buy Slow', values=top_performers['buy_slow']),
            dict(label='Sell Fast', values=top_performers['sell_fast']),
            dict(label='Sell Slow', values=top_performers['sell_slow']),
            dict(label='Return', values=top_performers['total_return'])
        ]
    ))
    parallel_plot.update_layout(
        title=f'Top 100 Performers - {symbol}'
    )
    
    # Create scatter plot of top performers
    scatter_plot = go.Figure(data=go.Scatter(
        x=top_performers['buy_ratio'],
        y=top_performers['sell_ratio'],
        mode='markers',
        marker=dict(
            size=10,
            color=top_performers['total_return'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Return')
        ),
        text=[f"Buy: {row['buy_fast']}/{row['buy_slow']}<br>Sell: {row['sell_fast']}/{row['sell_slow']}<br>Return: {row['total_return']:.2%}" 
              for _, row in top_performers.iterrows()]
    ))
    scatter_plot.update_layout(
        title=f'Top 100 Performers by Ratio - {symbol}',
        xaxis_title='Buy Ratio',
        yaxis_title='Sell Ratio'
    )
    
    return {
        'buy_heatmap': buy_heatmap,
        'sell_heatmap': sell_heatmap,
        'surface_plot': surface_plot,
        'parallel_plot': parallel_plot,
        'scatter_plot': scatter_plot
    }

def save_plots(figures, symbol, start_date, end_date, output_dir='plots'):
    """
    Save all plots to the specified directory.
    
    Args:
        figures (dict): Dictionary of figures to save
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        output_dir (str): Directory to save plots to
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Format dates for filenames
    date_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
    
    # Create safe symbol name for filenames
    symbol_safe = symbol.replace('/', '_')
    
    # Save each figure
    for name, fig in figures.items():
        filename = os.path.join(output_dir, f"{symbol_safe}_{name}_{date_str}.html")
        try:
            fig.write_html(filename)
            print(f"Saved {name} plot to {filename}")
        except Exception as e:
            print(f"Error saving {name} plot: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Quad Moving Average Crossover Strategy Backtest')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help='Trading symbol (e.g., SPY)')
    parser.add_argument('--start', type=str, default=DEFAULT_START_DATE, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=DEFAULT_END_DATE, 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, 
                        help='Timeframe (e.g., 1d, 30m, 1h, 15m)')
    parser.add_argument('--min-window', type=int, default=DEFAULT_MIN_WINDOW, 
                        help='Minimum window size')
    parser.add_argument('--max-window', type=int, default=DEFAULT_MAX_WINDOW, 
                        help='Maximum window size')
    parser.add_argument('--window-step', type=int, default=DEFAULT_WINDOW_STEP,
                        help='Step size between window values (higher = faster but less precise)')
    parser.add_argument('--max-combinations', type=int, default=MAX_COMBINATIONS,
                        help='Maximum number of window combinations to test (use -1 for unlimited)')
    parser.add_argument('--calculate-only', action='store_true', 
                        help='Only calculate possible combinations without running backtest')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode (limit window size range for quick testing)')
    parser.add_argument('--use-ray', action='store_true', default=True,
                        help='Use Ray for distributed computing')
    parser.add_argument('--num-cpus', type=int, default=None,
                        help='Number of CPUs to use for Ray (default: auto)')
    
    args = parser.parse_args()
    
    # Set up parameters from arguments
    symbol = args.symbol
    start_date = parse(args.start).replace(tzinfo=pytz.utc)
    end_date = parse(args.end).replace(tzinfo=pytz.utc)
    timeframe = args.timeframe
    min_window = args.min_window
    max_window = args.max_window
    window_step = args.window_step
    max_combinations = args.max_combinations
    use_ray = args.use_ray
    num_cpus = args.num_cpus
    
    # Fast mode - use limited window range for quick testing
    if args.fast:
        print("Running in FAST MODE with limited window range")
        min_window = 5
        max_window = 50
        window_step = 5
    
    # Adjust window sizes based on timeframe
    if timeframe == '1d':
        if min_window == DEFAULT_MIN_WINDOW:  # If default was used
            min_window = DEFAULT_MIN_WINDOW
        if max_window == DEFAULT_MAX_WINDOW:  # If default was used
            max_window = 252
    
    # Display strategy information
    print(f"QMAC Strategy Parameters:")
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Timeframe: {timeframe}")
    print(f"Window Range: {min_window} to {max_window} with step {window_step}")
    print(f"Computation: {'Ray distributed' if use_ray else 'Sequential'}")
    
    # Start timer for total execution
    total_start_time = time.time()
    
    # Calculate total possible combinations
    total_combinations = calculate_total_possible_combinations(min_window, max_window, window_step)
    print(f"Total Possible Unique Window Combinations: {total_combinations:,}")
    
    # If max_combinations is -1 (unlimited), set it to the total possible
    if max_combinations < 0:
        if total_combinations > 0:
            max_combinations = total_combinations
            print(f"Running with ALL {total_combinations:,} possible window combinations")
        else:
            max_combinations = 100  # Fallback if there are no valid combinations
            print(f"WARNING: No valid combinations possible with current settings, using {max_combinations} random samples")
    else:
        if max_combinations > total_combinations and total_combinations > 0:
            max_combinations = total_combinations
            print(f"Adjusted to maximum possible {total_combinations:,} combinations")
        else:
            print(f"Testing {max_combinations:,} of {total_combinations:,} possible combinations")
    
    # If calculate-only flag is set, exit after displaying combination counts
    if args.calculate_only:
        print("\nCalculation complete. Exiting without running backtest.")
        print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
        exit(0)
    
    # Analyze multiple window combinations
    print("\nAnalyzing window combinations...")
    window_results = analyze_window_combinations(
        symbol, start_date, end_date, min_window=min_window, max_window=max_window, 
        window_step=window_step, timeframe=timeframe, use_ray=use_ray, num_cpus=num_cpus,
        max_combinations=max_combinations)
    
    buy_fast, buy_slow, sell_fast, sell_slow = window_results['optimal_windows']
    optimal_perf = window_results['optimal_performance']
    
    print(f"\nOptimal window combination:")
    print(f"Buy signals: Fast MA = {buy_fast}, Slow MA = {buy_slow}")
    print(f"Sell signals: Fast MA = {sell_fast}, Slow MA = {sell_slow}")
    print(f"Optimal performance (total_return): {optimal_perf:.2%}")
    
    # Run strategy with optimal parameters
    print(f"\nRunning QMAC strategy with optimal parameters")
    optimal_results = window_results['optimal_results']
    
    # Print strategy stats
    print("\nOptimal QMAC Strategy Performance:")
    optimal_stats = optimal_results['qmac_pf'].stats()
    print(optimal_stats)
    
    # Print hold strategy stats for comparison
    print("\nBuy & Hold Strategy Performance:")
    hold_stats = optimal_results['hold_pf'].stats()
    print(hold_stats)
    
    # Print performance comparison
    qmac_return = optimal_stats['Total Return [%]']
    hold_return = hold_stats['Total Return [%]']
    outperformance = qmac_return - hold_return
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Optimal QMAC Return: {qmac_return:.2f}%")
    print(f"Buy & Hold Return: {hold_return:.2f}%")
    print(f"QMAC Outperformance: {outperformance:.2f}%")
    print(f"QMAC Max Drawdown: {optimal_stats['Max Drawdown [%]']:.2f}%")
    print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
    print(f"QMAC Sharpe Ratio: {optimal_stats['Sharpe Ratio']:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
    
    # Generate and save strategy plots
    strategy_figures = plot_qmac_strategy(optimal_results)
    
    # Create parameter space visualizations
    param_space_figures = create_parameter_space_visualization(
        window_results['performance_df'], 
        symbol, 
        start_date, 
        end_date
    )
    
    # Combine all figures
    all_figures = {**strategy_figures, **param_space_figures}
    
    # Save all plots
    timeframe_str = timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
    plot_dir = f'plots/{timeframe_str}_{symbol}_qmac_backtest'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    save_plots(all_figures, symbol, start_date, end_date, output_dir=plot_dir)
    
    print("\nDone!") 