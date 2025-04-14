#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization utilities for the QMAC strategy.
This module contains functions for finding optimal window parameters.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import ray
import psutil
import json
import sys
from tqdm.auto import tqdm, trange

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from lib.utils import evaluate_window_combination, sample_unique_windows
from lib.strategy_core import run_qmac_strategy
from lib.database import save_top_parameters_to_db

# Ray remote function for distributed combination evaluation
@ray.remote(num_cpus=1)  # Explicitly specify resources
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
    # Ensure price array is contiguous
    prices = np.ascontiguousarray(prices)
    
    # Process combinations in mini-batches of 100 for better memory management
    results = []
    total = len(combinations)
    
    # Process 100 combinations at a time to avoid memory pressure
    for i in range(0, total, 100):
        batch = combinations[i:i+100]
        batch_results = []
        
        for buy_fast, buy_slow, sell_fast, sell_slow in batch:
            perf = evaluate_window_combination(prices, buy_fast, buy_slow, sell_fast, sell_slow, max_window_size)
            batch_results.append(((buy_fast, buy_slow, sell_fast, sell_slow), perf))
        
        results.extend(batch_results)
        
        # Explicitly trigger garbage collection to prevent memory leaks
        if i % 1000 == 0 and i > 0:
            import gc
            gc.collect()
    
    return results

def analyze_window_combinations_ray(symbol, start_date, end_date, 
                                   min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                                   window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                                   timeframe=DEFAULT_TIMEFRAME, verbose=True,
                                   max_combinations=MAX_COMBINATIONS, num_cpus=None,
                                   save_to_db=True, top_n=TOP_N_PARAMS):
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
        save_to_db (bool): Whether to save results to database
        top_n (int): Number of top parameter combinations to save
        
    Returns:
        dict: Analysis results
    """
    # Start timing
    start_time = time.time()
    
    # Get CPU count for Ray
    if num_cpus is None:
        num_cpus = os.cpu_count()
    
    print(f"Running Ray-powered distributed optimization using {num_cpus} CPUs")
    
    # Initialize Ray for distributed computing with proper memory limits
    if not ray.is_initialized():
        max_memory = int(0.7 * psutil.virtual_memory().total)  # Reduced to 70% of available memory
        object_store_memory = int(0.5 * max_memory)  # 50% of max_memory for object store
        try:
            ray.init(num_cpus=num_cpus, _memory=max_memory, 
                    object_store_memory=object_store_memory, 
                    ignore_reinit_error=True)
        except:
            print("Warning: Failed to initialize Ray with memory parameters. Trying default init...")
            ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    
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
    
    # Get price data and ensure it's contiguous for optimal Numba performance
    prices = np.ascontiguousarray(single_result['ohlcv']['Close'].values)
    
    # Sample window combinations
    window_combinations = sample_unique_windows(
        min_window, max_window, window_step, 
        count=max_combinations)
    
    total_combinations = len(window_combinations)
    print(f"Testing {total_combinations:,} combinations with Ray distributed computing")
    
    # Calculate optimal batch size based on total combinations and CPUs
    # Aim for smaller but more numerous batches to improve parallelism
    optimal_batches = num_cpus * 4  # 4 batches per CPU for better scheduling
    
    # Check for environment variable to override batch size
    env_batch_size = os.environ.get("QMAC_BATCH_SIZE")
    if env_batch_size and env_batch_size.isdigit():
        batch_size = int(env_batch_size)
        print(f"Using environment-specified batch size: {batch_size}")
    else:
        batch_size = max(100, total_combinations // optimal_batches)  # Ensure minimum batch size
    
    # Split combinations into batches
    batches = [window_combinations[i:i+batch_size] for i in range(0, total_combinations, batch_size)]
    num_batches = len(batches)
    
    print(f"Splitting work into {num_batches} batches of ~{batch_size} combinations each")
    
    # Maximum window size for calculating MAs
    max_window_size = max_window  # Start with the max_window
    
    # Sample a subset of combinations to find max window size
    sample_size = min(1000, len(window_combinations))
    for i in range(sample_size):
        combo = window_combinations[i]
        max_slow = max(combo[1], combo[3])  # Max of buy_slow and sell_slow
        max_window_size = max(max_window_size, max_slow)
    
    # Put prices in Ray object store
    ray_prices = ray.put(prices)
    
    # Process batches in smaller groups to avoid overwhelming the scheduler
    all_results = []
    
    # Submit tasks in smaller groups
    max_concurrent_batches = min(num_cpus * 2, 24)  # Limited concurrency - only process batches = 2x CPU count
    
    with tqdm(total=num_batches, desc="Processing batch groups") as group_pbar:
        for batch_group_idx in range(0, num_batches, max_concurrent_batches):
            end_idx = min(batch_group_idx + max_concurrent_batches, num_batches)
            current_batches = batches[batch_group_idx:end_idx]
            
            # Submit this group of batches
            future_results = [evaluate_window_combinations_batch.remote(batch, ray_prices, max_window_size) 
                            for batch in current_batches]
            
            # Wait for all futures in this group to complete
            with tqdm(total=len(future_results), desc=f"Group {batch_group_idx//max_concurrent_batches + 1}") as batch_pbar:
                while future_results:
                    # Wait for the fastest batch to complete
                    done_id, future_results = ray.wait(future_results, num_returns=1)
                    
                    # Get the result and add to all_results
                    batch_results = ray.get(done_id[0])
                    all_results.extend(batch_results)
                    
                    # Update progress bar
                    batch_pbar.update(1)
            
            # Update group progress
            group_pbar.update(end_idx - batch_group_idx)
    
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
    
    # Save top parameters to database if requested
    if save_to_db:
        try:
            save_top_parameters_to_db(
                symbol, timeframe, start_date, end_date, 
                results_df, optimal_results, top_n
            )
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    end_time = time.time()
    print(f"Ray-powered optimization completed in {end_time - start_time:.2f} seconds")
    
    return {
        'ohlcv': single_result['ohlcv'],
        'performance_df': results_df,
        'optimal_windows': best_combination,
        'optimal_performance': best_perf,
        'optimal_results': optimal_results
    }

def analyze_window_combinations(symbol, start_date, end_date, 
                              min_window=DEFAULT_MIN_WINDOW, max_window=DEFAULT_MAX_WINDOW, 
                              window_step=DEFAULT_WINDOW_STEP, metric='total_return', 
                              timeframe=DEFAULT_TIMEFRAME, single_result=None, verbose=True,
                              max_combinations=MAX_COMBINATIONS, total_possible=None, 
                              use_ray=True, num_cpus=None, save_to_db=True, top_n=TOP_N_PARAMS):
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
        save_to_db (bool): Whether to save results to database
        top_n (int): Number of top parameter combinations to save
        
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
            max_combinations=max_combinations, num_cpus=num_cpus,
            save_to_db=save_to_db, top_n=top_n
        )
    
    # Otherwise use the original implementation (sequential processing)
    start_time = time.time()
    
    # Handle unlimited combinations mode
    if max_combinations < 0:
        print(f"Running in UNLIMITED combinations mode. This may take a long time...")
        # Set a very high number instead of truly unlimited to avoid memory issues
        max_combinations = 100000
    else:
        print(f"Starting window optimization with up to {max_combinations:,} combinations...")
    
    # Set portfolio parameters
    import vectorbt as vbt
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
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    resume_from_checkpoint = False
    start_batch = 0
    best_perf_so_far = 0.0
    best_combo_so_far = None
    
    try:
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
            results_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_file)), "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            results_file = os.path.join(results_dir, f"qmac_results_{symbol}_{timeframe}.npy")
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
    
    # Enhanced progress bar setup
    
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
    best_actual_return = None
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
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"Saved checkpoint to {checkpoint_file}")
                
                # Also save results array for potential resume
                results_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_file)), "results")
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    
                results_file = os.path.join(results_dir, f"qmac_results_{symbol}_{timeframe}.npy")
                np.save(results_file, results)
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
            results_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_file)), "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            results_file = os.path.join(results_dir, f"qmac_results_{symbol}_{timeframe}.npy")
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
        results_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_file)), "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_csv = os.path.join(results_dir, f"qmac_results_{symbol}_{timeframe}.csv")
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
    
    # Save top parameters to database if requested
    if save_to_db:
        try:
            optimal_results = None
            if 'optimal_windows' in locals():
                # Run optimal strategy to get full results for database
                buy_fast, buy_slow, sell_fast, sell_slow = optimal_windows
                optimal_results = run_qmac_strategy(
                    symbol, start_date, end_date,
                    buy_fast_window=buy_fast,
                    buy_slow_window=buy_slow,
                    sell_fast_window=sell_fast,
                    sell_slow_window=sell_slow,
                    timeframe=timeframe,
                    verbose=False
                )
            
            save_top_parameters_to_db(
                symbol, timeframe, start_date, end_date, 
                performance_df, optimal_results, top_n
            )
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    return {
        'ohlcv': ohlcv,
        'performance_df': performance_df,
        'optimal_windows': optimal_windows,
        'optimal_performance': optimal_perf
    } 