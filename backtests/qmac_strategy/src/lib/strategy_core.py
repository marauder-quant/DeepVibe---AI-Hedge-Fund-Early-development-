#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core implementation of the Quad Moving Average Crossover (QMAC) strategy.
This module contains the main strategy logic and trade signal generation.
"""

import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import time
from dotenv import load_dotenv
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from lib.utils import calculate_cross_signals, calculate_cross_below_signals

# Load environment variables for Alpaca API keys
load_dotenv()

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