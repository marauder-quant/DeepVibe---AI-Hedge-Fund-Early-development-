#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the QMAC strategy implementation.
This module contains numba-optimized functions for performance-critical operations.
"""

import numpy as np
import numba as nb
from tqdm.auto import tqdm
import psutil
import os
import time
import gc

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
    
    # Pre-allocate arrays - use float32 to reduce memory usage
    buy_fast_ma = np.zeros(n, dtype=np.float32)
    buy_slow_ma = np.zeros(n, dtype=np.float32)
    sell_fast_ma = np.zeros(n, dtype=np.float32)
    sell_slow_ma = np.zeros(n, dtype=np.float32)
    
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
    trades = 0
    losing_trades = 0
    
    # Enhanced early stopping with multiple criteria
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
            trades += 1
            
            # Track losing trades
            if trade_return < 0:
                losing_trades += 1
        
        # Enhanced early stopping for clearly unpromising combinations
        # Multiple criteria for stopping the evaluation early:
        # 1. If already seeing significant losses (-10%)
        # 2. If we've had several trades and most were losing
        # 3. If we've had multiple trades with no significant overall gains
        if position == 1 and total_return < -0.10:  # -10% threshold
            return total_return
        elif trades >= 5 and losing_trades / trades > 0.8:  # Over 80% losing trades
            return total_return * 0.9  # Apply penalty to discourage this combination
        elif trades >= 10 and total_return < 0.01:  # Many trades but minimal returns
            return total_return
    
    # If still in position at the end, close position
    if position == 1:
        final_return = (prices[-1] / entry_price) * (1.0 - 0.0025) - 1.0  # Only exit fee
        total_return += final_return
    
    return total_return

def get_system_resources():
    """
    Get current system resource usage information.
    
    Returns:
        dict: Dictionary containing system resource information
    """
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Get memory information
    memory = psutil.virtual_memory()
    memory_used_percent = memory.percent
    memory_available_gb = memory.available / (1024**3)
    
    # Get disk information
    disk = psutil.disk_usage('/')
    disk_used_percent = disk.percent
    
    # Get process information
    process = psutil.Process(os.getpid())
    process_memory_gb = process.memory_info().rss / (1024**3)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used_percent': memory_used_percent,
        'memory_available_gb': memory_available_gb,
        'disk_used_percent': disk_used_percent,
        'process_memory_gb': process_memory_gb,
        'timestamp': time.time()
    }

def adaptive_batch_size(current_batch_size, resources, min_batch_size=50, max_batch_size=10000):
    """
    Dynamically adjust batch size based on system resource usage.
    
    Args:
        current_batch_size (int): Current batch size
        resources (dict): System resource information from get_system_resources()
        min_batch_size (int): Minimum batch size
        max_batch_size (int): Maximum batch size
        
    Returns:
        int: New batch size
    """
    # If memory is getting tight, reduce batch size
    if resources['memory_used_percent'] > 85:
        # Significant memory pressure, reduce batch size substantially
        new_batch_size = max(min_batch_size, int(current_batch_size * 0.5))
    elif resources['memory_used_percent'] > 75:
        # Moderate memory pressure, reduce batch size slightly
        new_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
    elif resources['memory_used_percent'] < 50 and resources['cpu_percent'] < 70:
        # Plenty of resources available, increase batch size
        new_batch_size = min(max_batch_size, int(current_batch_size * 1.2))
    else:
        # Resources are in an acceptable range, maintain current batch size
        new_batch_size = current_batch_size
    
    return new_batch_size

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

def should_throttle_processing(resources, history=None, critical_memory_threshold=90, high_memory_threshold=80, critical_cpu_threshold=95):
    """
    Determine if processing should be throttled or paused based on system resources.
    
    Args:
        resources (dict): Current system resource information from get_system_resources()
        history (list, optional): History of resource measurements for trend analysis
        critical_memory_threshold (int): Memory percentage above which to pause processing
        high_memory_threshold (int): Memory percentage above which to throttle processing
        critical_cpu_threshold (int): CPU percentage above which to throttle processing
        
    Returns:
        tuple: (throttle_level, should_pause, reason)
            throttle_level: 0 (none), 1 (light), 2 (medium), 3 (heavy)
            should_pause: Boolean indicating if processing should pause completely
            reason: String explaining the reason for throttling/pausing
    """
    throttle_level = 0
    should_pause = False
    reason = "Normal operation"
    
    # Check for critical memory pressure (most important resource to monitor)
    if resources['memory_used_percent'] >= critical_memory_threshold:
        should_pause = True
        reason = f"Critical memory pressure: {resources['memory_used_percent']}% used"
        return (3, should_pause, reason)
    
    # Check for high memory pressure
    if resources['memory_used_percent'] >= high_memory_threshold:
        throttle_level = max(throttle_level, 2)
        reason = f"High memory pressure: {resources['memory_used_percent']}% used"
    
    # Check for critical CPU pressure
    if resources['cpu_percent'] >= critical_cpu_threshold:
        throttle_level = max(throttle_level, 2)
        if "memory" not in reason:
            reason = f"High CPU pressure: {resources['cpu_percent']}% used"
    
    # Check if available memory is critically low in absolute terms
    if resources['memory_available_gb'] < 1.0:  # Less than 1 GB free
        throttle_level = max(throttle_level, 3)
        should_pause = True
        reason = f"Critically low memory: only {resources['memory_available_gb']:.2f} GB available"
    
    # Process growth trends if history is provided
    if history and len(history) >= 3:
        # Check if memory usage is rapidly increasing
        recent_memory = [h['memory_used_percent'] for h in history[-3:]]
        memory_growth_rate = (recent_memory[-1] - recent_memory[0]) / 2  # % change per interval
        
        if memory_growth_rate > 5:  # More than 5% increase per interval
            throttle_level = max(throttle_level, 2)
            if "memory" not in reason:
                reason = f"Rapidly increasing memory usage: {memory_growth_rate:.1f}% per interval"
            
            # If already high memory and growing fast, should pause
            if resources['memory_used_percent'] > 70 and memory_growth_rate > 10:
                should_pause = True
                reason = f"High memory ({resources['memory_used_percent']}%) growing rapidly ({memory_growth_rate:.1f}% per interval)"
    
    # If process is using excessive memory, throttle
    if resources['process_memory_gb'] > 0.5 * resources['memory_available_gb']:
        throttle_level = max(throttle_level, 1)
        if throttle_level == 1:  # Only set reason if not already set to something more severe
            reason = f"Process using significant memory: {resources['process_memory_gb']:.2f} GB"
    
    return (throttle_level, should_pause, reason)

def calculate_adaptive_concurrency(current_concurrency, resources, history=None, min_concurrency=1, max_concurrency=None):
    """
    Calculate adaptive concurrency level based on system resources.
    
    Args:
        current_concurrency (int): Current concurrency level
        resources (dict): Current system resource information
        history (list, optional): History of resource measurements for trend analysis
        min_concurrency (int): Minimum concurrency level
        max_concurrency (int, optional): Maximum concurrency level (defaults to CPU count)
        
    Returns:
        int: New concurrency level
    """
    if max_concurrency is None:
        max_concurrency = os.cpu_count()
    
    # Get throttling recommendation
    throttle_level, should_pause, reason = should_throttle_processing(resources, history)
    
    # If we should pause, return minimum concurrency
    if should_pause:
        return min_concurrency
    
    # Adjust concurrency based on throttle level
    if throttle_level == 3:  # Heavy throttling
        new_concurrency = max(min_concurrency, int(current_concurrency * 0.5))
    elif throttle_level == 2:  # Medium throttling
        new_concurrency = max(min_concurrency, int(current_concurrency * 0.7))
    elif throttle_level == 1:  # Light throttling
        new_concurrency = max(min_concurrency, current_concurrency - 1)
    else:  # No throttling needed
        # If resources are plentiful, cautiously increase concurrency
        if resources['memory_used_percent'] < 50 and resources['cpu_percent'] < 70:
            new_concurrency = min(max_concurrency, current_concurrency + 1)
        else:
            new_concurrency = current_concurrency
    
    return new_concurrency

def handle_resource_pressure(resources, resource_history=None, recovery_time=15):
    """
    Handle system resource pressure with graceful degradation.
    
    Args:
        resources (dict): Current system resource information
        resource_history (list, optional): History of resource measurements
        recovery_time (int): Base recovery time in seconds
        
    Returns:
        tuple: (pause_duration, action_taken)
    """
    pause_duration = 0
    action_taken = "none"
    
    # Check if we're in a critical state
    throttle_level, should_pause, reason = should_throttle_processing(
        resources, resource_history)
    
    if should_pause:
        # Critical resource pressure - take immediate action
        print(f"\n⚠️ CRITICAL RESOURCE PRESSURE: {reason}")
        
        # Calculate appropriate pause duration based on severity
        if "memory" in reason and resources['memory_used_percent'] > 95:
            # Severe memory pressure needs more recovery time
            pause_duration = recovery_time * 2
            action_taken = "long_pause"
            
            print(f"Pausing for {pause_duration}s to allow memory recovery")
            
            # Force garbage collection before pausing
            gc.collect()
            time.sleep(pause_duration)
            
            # Check if memory improved
            new_resources = get_system_resources()
            if new_resources['memory_used_percent'] > 90:
                # Still critical - more aggressive action needed
                print("Memory still critical after pause - forcing additional garbage collection")
                
                # Try to free more memory
                import sys
                try:
                    # Clear any large variables in global namespace
                    for var_name in list(globals().keys()):
                        if var_name.startswith('__'):
                            continue
                        obj = globals()[var_name]
                        if isinstance(obj, (list, dict, set)) and sys.getsizeof(obj) > 1024*1024:
                            globals()[var_name] = None
                except Exception as e:
                    print(f"Error while trying to free globals: {e}")
                
                # Multiple GC runs and longer pause
                for _ in range(3):
                    gc.collect()
                    time.sleep(2)
                
                action_taken = "emergency_recovery"
                pause_duration += 6  # Additional 6 seconds
        else:
            # Standard pause for other resource issues
            pause_duration = recovery_time
            action_taken = "standard_pause"
            
            print(f"Pausing for {pause_duration}s to allow resource recovery")
            gc.collect()
            time.sleep(pause_duration)
    
    elif throttle_level >= 2:
        # High resource pressure - take preventive action
        print(f"\n⚠️ HIGH RESOURCE PRESSURE: {reason}")
        pause_duration = recovery_time // 2
        action_taken = "short_pause"
        
        print(f"Brief {pause_duration}s pause for resource stabilization")
        gc.collect()
        time.sleep(pause_duration)
    
    elif throttle_level == 1:
        # Moderate resource pressure - light intervention
        print(f"\nResource pressure: {reason}")
        gc.collect()  # Just garbage collect without pausing
        action_taken = "gc_only"
    
    return (pause_duration, action_taken) 