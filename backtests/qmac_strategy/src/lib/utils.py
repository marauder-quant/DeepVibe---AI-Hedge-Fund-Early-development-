#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the QMAC strategy implementation.
This module contains numba-optimized functions for performance-critical operations.
"""

import numpy as np
import numba as nb
from tqdm.auto import tqdm

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

@nb.njit(parallel=True, fastmath=True, cache=True)
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
            
        # Early stopping for clearly unpromising combinations
        if position == 1 and total_return < -0.10:  # -10% threshold
            # If already seeing significant losses, don't waste time on this combination
            return total_return
    
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
    import time
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
    import time
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