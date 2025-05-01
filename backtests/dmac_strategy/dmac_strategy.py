#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dual Moving Average Crossover (DMAC) strategy implementation using vectorbt and Alpaca data.
This module provides functionality to run DMAC strategy backtests, analyze window combinations,
and visualize results.
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

# Import settings from config
from backtests.dmac_strategy.config import (
    DEFAULT_SYMBOL, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME,
    DEFAULT_MIN_WINDOW, DEFAULT_MAX_WINDOW, DEFAULT_WINDOW_STEP,
    INITIAL_CASH, FEES, SLIPPAGE, 
    METRICS_FREQUENCY, DEFAULT_PLOTS_DIR
)

# Load environment variables for Alpaca API keys
load_dotenv()

def run_dmac_strategy(symbol, start_date, end_date, fast_window, slow_window, 
                      init_cash=INITIAL_CASH, fees=FEES, slippage=SLIPPAGE, timeframe=DEFAULT_TIMEFRAME, verbose=True):
    """
    Run a Dual Moving Average Crossover strategy with specific window parameters.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        start_date (datetime): Start date for the backtest
        end_date (datetime): End date for the backtest
        fast_window (int): Fast moving average window size
        slow_window (int): Slow moving average window size
        init_cash (float): Initial cash amount
        fees (float): Fee percentage (e.g., 0.0025 for 0.25%)
        slippage (float): Slippage percentage
        timeframe (str): Timeframe for data (e.g., '1d', '1h', '15m')
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Dictionary containing the backtest results
    """
    # Set portfolio parameters
    vbt.settings.portfolio['init_cash'] = init_cash
    vbt.settings.portfolio['fees'] = fees
    vbt.settings.portfolio['slippage'] = slippage
    
    # Add time buffer for SMA/EMA calculation - at least 500 days to ensure enough data
    # This is especially important for large MA windows
    time_buffer = timedelta(days=max(500, slow_window * 3))
    
    # Download data with time buffer using Alpaca or fallback to Yahoo Finance if needed
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        if verbose:
            print(f"Attempting to download data from Alpaca for {symbol}...")
            print(f"Downloading with buffer of {time_buffer.days} days to ensure enough history for MA calculations")
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
    
    # Convert to float64
    ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)
    
    # Create a copy of data without time buffer
    wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date)
    ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]
    
    if verbose:
        print(f"Full data shape (with buffer): {ohlcv_wbuf.shape}")
        print(f"Analysis window data shape: {ohlcv.shape}")
    
    # Check if there's enough data for the moving averages
    if len(ohlcv_wbuf) < slow_window * 2:
        print(f"WARNING: Not enough historical data for proper MA calculation. " +
              f"Have {len(ohlcv_wbuf)} bars, but need at least {slow_window * 2} for reliable results.")
    
    # Pre-calculate running windows on data with time buffer
    fast_ma = vbt.MA.run(ohlcv_wbuf['Close'], fast_window)
    slow_ma = vbt.MA.run(ohlcv_wbuf['Close'], slow_window)
    
    # Remove time buffer
    fast_ma = fast_ma[wobuf_mask]
    slow_ma = slow_ma[wobuf_mask]
    
    # Generate crossover signals
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    
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
        print(f"Entry signals count: {dmac_entries.sum()}")
        print(f"Exit signals count: {dmac_exits.sum()}")
    
    # Build portfolio
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits, freq=freq)
    
    # Build hold portfolio for comparison
    hold_entries = pd.Series.vbt.signals.empty_like(dmac_entries)
    hold_entries.iloc[0] = True
    hold_exits = pd.Series.vbt.signals.empty_like(hold_entries)
    hold_exits.iloc[-1] = True
    hold_pf = vbt.Portfolio.from_signals(ohlcv['Close'], hold_entries, hold_exits, freq=freq)
    
    return {
        'ohlcv': ohlcv,
        'ohlcv_wbuf': ohlcv_wbuf,
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'dmac_entries': dmac_entries,
        'dmac_exits': dmac_exits,
        'dmac_pf': dmac_pf,
        'hold_pf': hold_pf
    }

def analyze_window_combinations(symbol, start_date, end_date, 
                              min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                              window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                              timeframe=DEFAULT_TIMEFRAME, single_result=None, verbose=True):
    """
    Analyze multiple window combinations for DMAC strategy.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USD')
        start_date (datetime): Start date for the backtest
        end_date (datetime): End date for the backtest
        min_window (int): Minimum window size to test
        max_window (int): Maximum window size to test
        window_step (int): Step size between window values
        metric (str): Performance metric to optimize for
        timeframe (str): Timeframe for data (e.g., '1d', '1h', '15m')
        single_result (dict, optional): Result from a previous run_dmac_strategy call
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Dictionary containing the analysis results
    """
    # Set portfolio parameters
    vbt.settings.portfolio['init_cash'] = INITIAL_CASH
    vbt.settings.portfolio['fees'] = FEES
    vbt.settings.portfolio['slippage'] = SLIPPAGE
    
    # Get data from single_result or run a new strategy
    if single_result is None:
        if verbose:
            print("No existing result provided, downloading data...")
        single_result = run_dmac_strategy(
            symbol, start_date, end_date, 
            fast_window=min_window, 
            slow_window=max_window, 
            timeframe=timeframe,
            verbose=verbose
        )
    
    # Use the data directly from the single result
    ohlcv = single_result['ohlcv']
    
    if verbose:
        window_count = len(range(min_window, max_window+1, window_step))
        print(f"Running window optimization with {window_count} window combinations...")
    
    # Pre-calculate running windows on data
    fast_ma, slow_ma = vbt.MA.run_combs(
        ohlcv['Close'], np.arange(min_window, max_window+1, window_step), 
        r=2, short_names=['fast_ma', 'slow_ma'])
    
    # Generate crossover signals
    dmac_entries = fast_ma.ma_crossed_above(slow_ma)
    dmac_exits = fast_ma.ma_crossed_below(slow_ma)
    
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
    
    # Build portfolio
    dmac_pf = vbt.Portfolio.from_signals(ohlcv['Close'], dmac_entries, dmac_exits, freq=freq)
    
    # Calculate performance of each window combination
    dmac_perf = dmac_pf.deep_getattr(metric)
    
    # Convert array into a matrix
    dmac_perf_matrix = dmac_perf.vbt.unstack_to_df(symmetric=True, 
        index_levels='fast_ma_window', column_levels='slow_ma_window')
    
    # Find optimal window combination
    optimal_windows = dmac_perf.idxmax()
    
    # Print optimal windows if verbose
    if verbose:
        optimal_fast, optimal_slow = optimal_windows
        optimal_perf = dmac_perf[optimal_windows]
        print(f"\nOptimal window combination: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
        print(f"Optimal performance ({metric}): {optimal_perf:.2%}")
    
    return {
        'ohlcv': ohlcv,
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'dmac_entries': dmac_entries,
        'dmac_exits': dmac_exits,
        'dmac_pf': dmac_pf,
        'dmac_perf': dmac_perf,
        'dmac_perf_matrix': dmac_perf_matrix,
        'optimal_windows': optimal_windows
    }

def plot_dmac_strategy(results):
    """
    Plot the DMAC strategy results.
    
    Args:
        results (dict): Results from run_dmac_strategy
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Plot the OHLC data with MA lines and entry/exit points
    fig = results['ohlcv']['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
    fig = results['fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
    fig = results['slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
    fig = results['dmac_entries'].vbt.signals.plot_as_entry_markers(results['ohlcv']['Open'], fig=fig)
    fig = results['dmac_exits'].vbt.signals.plot_as_exit_markers(results['ohlcv']['Open'], fig=fig)
    
    # Plot equity comparison
    value_fig = results['dmac_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (DMAC)'))
    results['hold_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=value_fig)
    
    # Plot trades
    trades_fig = results['dmac_pf'].trades.plot()
    
    return {
        'strategy_fig': fig,
        'value_fig': value_fig,
        'trades_fig': trades_fig
    }

def plot_heatmap(perf_matrix, metric='total_return'):
    """
    Plot a heatmap of strategy performance across window combinations.
    
    Args:
        perf_matrix (DataFrame): Performance matrix
        metric (str): Performance metric
        
    Returns:
        Figure: Heatmap figure
    """
    heatmap = perf_matrix.vbt.heatmap(
        xaxis_title='Slow window', 
        yaxis_title='Fast window',
        title=f'{metric} by window combination')
    
    return heatmap

def save_plots(figures, symbol, start_date, end_date, output_dir=DEFAULT_PLOTS_DIR):
    """
    Save all plots to the specified directory.
    
    Args:
        figures (dict): Dictionary of figures to save
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        output_dir (str): Directory to save plots to
    
    Note: This is a legacy function. Use lib.visualization.save_plots instead.
    """
    # Import the visualization module's save_plots function
    from backtests.dmac_strategy.lib.visualization import save_plots as viz_save_plots
    
    # Call the improved save_plots function
    return viz_save_plots(figures, symbol, start_date, end_date, output_dir)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Dual Moving Average Crossover Strategy Backtest')
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
    
    args = parser.parse_args()
    
    # Set up parameters from arguments
    symbol = args.symbol
    start_date = parse(args.start).replace(tzinfo=pytz.utc)
    end_date = parse(args.end).replace(tzinfo=pytz.utc)
    timeframe = args.timeframe
    min_window = args.min_window
    max_window = args.max_window
    window_step = args.window_step
    
    # Adjust window sizes based on timeframe
    if timeframe == '1d':
        if min_window == DEFAULT_MIN_WINDOW:  # If default was used
            min_window = DEFAULT_MIN_WINDOW
        if max_window == DEFAULT_MAX_WINDOW:  # If default was used
            max_window = 252
    
    print(f"Running DMAC strategy for {symbol} from {start_date} to {end_date} on {timeframe} chart")
    print(f"Window range: {min_window} to {max_window} with step {window_step}")
    
    # Analyze multiple window combinations
    print("\nAnalyzing window combinations...")
    window_results = analyze_window_combinations(
        symbol, start_date, end_date, min_window=min_window, max_window=max_window, 
        window_step=window_step, timeframe=timeframe)
    
    optimal_fast, optimal_slow = window_results['optimal_windows']
    optimal_perf = window_results['dmac_perf'][window_results['optimal_windows']]
    
    print(f"\nOptimal window combination: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
    print(f"Optimal performance (total_return): {optimal_perf:.2%}")
    
    # Run strategy with optimal parameters
    print(f"\nRunning DMAC strategy with optimal parameters: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
    optimal_results = run_dmac_strategy(
        symbol, start_date, end_date, optimal_fast, optimal_slow, 
        timeframe=timeframe, verbose=False)
    
    # Print strategy stats
    print("\nOptimal DMAC Strategy Performance:")
    optimal_stats = optimal_results['dmac_pf'].stats()
    print(optimal_stats)
    
    # Print hold strategy stats for comparison
    print("\nBuy & Hold Strategy Performance:")
    hold_stats = optimal_results['hold_pf'].stats()
    print(hold_stats)
    
    # Print performance comparison
    dmac_return = optimal_stats['Total Return [%]']
    hold_return = hold_stats['Total Return [%]']
    outperformance = dmac_return - hold_return
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Optimal DMAC Return: {dmac_return:.2f}%")
    print(f"Buy & Hold Return: {hold_return:.2f}%")
    print(f"DMAC Outperformance: {outperformance:.2f}%")
    print(f"DMAC Max Drawdown: {optimal_stats['Max Drawdown [%]']:.2f}%")
    print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
    print(f"DMAC Sharpe Ratio: {optimal_stats['Sharpe Ratio']:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
    
    # Generate and save strategy plots
    strategy_figures = plot_dmac_strategy(optimal_results)
    timeframe_str = timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
    # Create more descriptive folder name
    plot_dir = f'plots/{timeframe_str}_{symbol}_backtest'
    
    # Generate heatmap and add to strategy_figures dictionary
    heatmap = plot_heatmap(window_results['dmac_perf_matrix'], 'total_return')
    strategy_figures['heatmap'] = heatmap
    
    # Save all plots including the heatmap
    save_plots(strategy_figures, symbol, start_date, end_date, output_dir=plot_dir)
    
    print(f"\nSaved visualization files in folder: {plot_dir}/")
    print(f"1. Strategy plot: strategy_fig")
    print(f"2. Performance comparison: value_fig")
    print(f"3. Trades visualization: trades_fig")
    print(f"4. Window optimization heatmap: heatmap")
    
    print("\nDone!") 