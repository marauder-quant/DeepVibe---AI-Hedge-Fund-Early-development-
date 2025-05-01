#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimizer module for DMAC strategy.
This module provides functions for analyzing window combinations and optimizing parameters.
"""

import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime
from tqdm import tqdm
from itertools import combinations_with_replacement
import random
import gc

# Import from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmac_strategy import run_dmac_strategy
from config import DEFAULT_MIN_WINDOW, DEFAULT_MAX_WINDOW, DEFAULT_WINDOW_STEP, TOP_N_PARAMS, MAX_COMBINATIONS

# Import from local modules
from lib.database import save_top_parameters_to_db

def sample_unique_windows(min_window, max_window, window_step, count, total_possible=None):
    """
    Sample unique window combinations for testing.
    
    Args:
        min_window (int): Minimum window size
        max_window (int): Maximum window size
        window_step (int): Step size between window values
        count (int): Number of combinations to sample
        total_possible (int, optional): Pre-calculated total possible combinations
        
    Returns:
        list: List of window combinations (tuples of (fast_window, slow_window))
    """
    # Create a list of all possible window values
    window_values = list(range(min_window, max_window + 1, window_step))
    
    # Calculate all possible combinations where fast_window < slow_window
    all_combinations = []
    for fast_window in window_values:
        for slow_window in window_values:
            if fast_window < slow_window:
                all_combinations.append((fast_window, slow_window))
    
    print(f"Generated {len(all_combinations)} potential window combinations")
    
    # If count is greater than the total possible combinations, return all combinations
    if count >= len(all_combinations) or count == -1:
        print(f"Using all {len(all_combinations)} window combinations")
        return all_combinations
    
    # Otherwise, sample randomly
    print(f"Sampling {count} combinations from {len(all_combinations)} possibilities")
    return random.sample(all_combinations, count)

def analyze_window_combinations(symbol, start_date, end_date, 
                              min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                              window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                              timeframe='1d', single_result=None, verbose=True,
                              max_combinations=MAX_COMBINATIONS, save_to_db=True, top_n=TOP_N_PARAMS):
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
        max_combinations (int): Maximum number of window combinations to test
        save_to_db (bool): Whether to save results to database
        top_n (int): Number of top parameter combinations to save
        
    Returns:
        dict: Dictionary containing the analysis results
    """
    start_time = time.time()
    
    print(f"Starting optimization for {symbol} from {start_date} to {end_date}")
    print(f"Window parameters: min={min_window}, max={max_window}, step={window_step}")
    print(f"Will test up to {max_combinations} combinations")
    
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
    
    # Generate window combinations to test
    window_combinations = sample_unique_windows(min_window, max_window, window_step, max_combinations)
    
    if verbose:
        print(f"Testing {len(window_combinations)} window combinations...")
    
    # Create progress bar
    pbar = tqdm(total=len(window_combinations), desc="Testing window combinations")
    
    # Store results for each combination
    results = []
    
    # Test each window combination
    for fast_window, slow_window in window_combinations:
        try:
            # Run strategy with current windows
            dmac_result = run_dmac_strategy(
                symbol, start_date, end_date, 
                fast_window=fast_window, 
                slow_window=slow_window,
                timeframe=timeframe,
                verbose=False
            )
            
            # Extract performance metric
            if metric == 'total_return':
                perf = dmac_result['dmac_pf'].total_return()
            elif metric == 'sharpe_ratio':
                perf = dmac_result['dmac_pf'].sharpe_ratio()
            else:
                # Default to total return if metric is not recognized
                perf = dmac_result['dmac_pf'].total_return()
            
            # Store additional metrics for database
            total_return = dmac_result['dmac_pf'].total_return()
            
            # Create a result entry with column names matching database expectations
            result_entry = {
                'fast_window': fast_window,
                'slow_window': slow_window,
                'total_return': total_return
            }
            
            # Store result
            results.append(result_entry)
            
            # Free up memory
            del dmac_result
            gc.collect()
            
        except Exception as e:
            if verbose:
                print(f"Error testing windows ({fast_window}, {slow_window}): {str(e)}")
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid results found. Check parameters and try again.")
        return {
            'ohlcv': ohlcv,
            'performance_df': pd.DataFrame(),
            'optimal_windows': None,
            'optimal_performance': None
        }
    
    print("\nOptimization completed, finding best parameters...")
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Total return range: {results_df['total_return'].min():.4f} to {results_df['total_return'].max():.4f}")
    
    # Find optimal window combination
    best_idx = results_df['total_return'].idxmax()
    optimal_row = results_df.iloc[best_idx]
    optimal_windows = (int(optimal_row['fast_window']), int(optimal_row['slow_window']))
    optimal_perf = optimal_row['total_return']
    
    if verbose:
        print(f"\nOptimal window combination: Fast MA = {optimal_windows[0]}, Slow MA = {optimal_windows[1]}")
        print(f"Optimal performance ({metric}): {optimal_perf:.2%}")
    
    # Run strategy with optimal parameters to get detailed results
    print(f"Running strategy with optimal parameters for detailed results...")
    optimal_results = run_dmac_strategy(
        symbol, start_date, end_date, 
        fast_window=optimal_windows[0], 
        slow_window=optimal_windows[1],
        timeframe=timeframe,
        verbose=True
    )
    
    # Save top parameters to database if requested
    if save_to_db:
        try:
            print(f"Attempting to save top parameters to database...")
            print(f"Results DataFrame shape: {results_df.shape}")
            print(f"Saving {top_n} top combinations")
            print(f"Symbol: {symbol}, Timeframe: {timeframe}")
            print(f"Date range: {start_date} to {end_date}")
            
            save_top_parameters_to_db(
                symbol, timeframe, start_date, end_date, 
                results_df, optimal_results, top_n
            )
            if verbose:
                print(f"Saved top {top_n} parameter combinations to database")
        except Exception as e:
            print(f"Error saving to database: {e}")
            import traceback
            traceback.print_exc()
    
    end_time = time.time()
    if verbose:
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    return {
        'ohlcv': ohlcv,
        'performance_df': results_df,
        'optimal_windows': optimal_windows,
        'optimal_performance': optimal_perf,
        'optimal_results': optimal_results
    } 