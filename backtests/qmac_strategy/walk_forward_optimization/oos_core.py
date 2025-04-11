#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core functionality for QMAC strategy out-of-sample testing.
"""

import os
import json
import logging
import multiprocessing as mp
import random
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import ray
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Import configuration and other modules
from backtests.qmac_strategy.walk_forward_optimization.oos_config import (
    DEFAULT_OOS_STOCKS, MAX_OOS_WINDOWS, DEFAULT_OOS_WINDOW_LENGTH,
    DEFAULT_TOP_PARAMS, EARLY_TERMINATION_ENABLED, EARLY_TERMINATION_MIN_TESTS,
    EARLY_TERMINATION_CONFIDENCE_THRESHOLD, EARLY_TERMINATION_CONSECUTIVE_CHECKS
)
from backtests.qmac_strategy.walk_forward_optimization.oos_utils import (
    get_sp500_tickers, generate_random_periods, test_parameters_on_stock_period,
    display_live_confidence_update
)
from backtests.qmac_strategy.walk_forward_optimization.oos_confidence import (
    load_confidence_tracker, update_confidence_after_batch, update_confidence_tracker,
    update_confidence_after_single_test
)

# Set up logging
log = logging.getLogger("rich")
console = Console()

# Initialize Ray for distributed processing if not already initialized
if not ray.is_initialized():
    try:
        ray.init(ignore_reinit_error=True)
    except:
        ray.init(local_mode=True, ignore_reinit_error=True)
else:
    log.info("Ray is already initialized, reusing existing Ray instance")

@ray.remote
def process_stock_batch(stock_batch, periods, params, timeframe='1d'):
    """Process a batch of stocks in parallel using Ray."""
    results = []
    
    # Ensure the correct import for test_parameters_on_stock_period
    from backtests.qmac_strategy.walk_forward_optimization.oos_utils import test_parameters_on_stock_period
    
    # Create a ProcessPoolExecutor for the periods
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(periods))) as executor:
        # Submit all test combinations
        futures = []
        for stock in stock_batch:
            for period_start, period_end in periods:
                futures.append(executor.submit(
                    test_parameters_on_stock_period,
                    params, stock, period_start, period_end, timeframe
                ))
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                log.error(f"Error processing stock batch: {e}")
    
    return results

def run_out_of_sample_test_parallel(params, start_date=None, end_date=None, n_periods=1, 
                          period_length=60, n_stocks=30, timeframe='1d', n_cores=None):
    """Run out-of-sample testing on random stocks and periods using parallel processing."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Get S&P 500 tickers
    log.info("Getting S&P 500 tickers...")
    sp500_tickers = get_sp500_tickers()
    selected_stocks = random.sample(sp500_tickers, min(n_stocks, len(sp500_tickers)))
    log.info(f"Selected {len(selected_stocks)} stocks for testing")
    
    # Generate random periods
    if start_date is None:
        start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
    if end_date is None:
        end_date = datetime.now(pytz.UTC)
    
    log.info("Generating test periods...")
    periods = generate_random_periods(start_date, end_date, n_periods, period_length)
    
    # Calculate total tests for progress tracking
    total_tests = len(selected_stocks) * len(periods)
    log.info(f"Running {total_tests} tests with parallel processing...")
    
    # Create timestamp for this batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Split stocks into batches for Ray workers
    # Determine the number of batches based on available CPU cores and Ray workers
    num_ray_workers = min(32, n_cores)  # Limit to reasonable number to prevent overloading
    batch_size = max(1, len(selected_stocks) // num_ray_workers)
    stock_batches = [selected_stocks[i:i+batch_size] for i in range(0, len(selected_stocks), batch_size)]
    
    log.info(f"Distributing work across {len(stock_batches)} Ray workers")
    
    # Flag for early termination
    early_termination = False
    
    # Set up progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Testing...", total=total_tests)
        
        # Submit tasks to Ray workers
        futures = [process_stock_batch.remote(batch, periods, params, timeframe) for batch in stock_batches]
        
        # Collect results and update confidence as they complete
        results = []
        remaining_futures = futures
        
        # Track low confidence count - only terminate if confidence stays low
        low_confidence_count = 0
        
        while remaining_futures:
            try:
                # Wait for a batch to complete
                completed_future, remaining_futures = ray.wait(remaining_futures, num_returns=1)
                
                # Get results from completed batch
                batch_results = ray.get(completed_future[0])
                results.extend(batch_results)
                
                # Update progress
                progress.update(task, advance=len(batch_results))
                
                # Update confidence after getting results from each batch
                if len(results) > 1:
                    results_so_far = results.copy()
                    results_df = pd.DataFrame(results_so_far)
                    current_summary, overall_confidence = update_confidence_after_batch(params, results_df, timestamp, timeframe)
                    
                    # Check if we should do early termination after minimum number of tests
                    if EARLY_TERMINATION_ENABLED and len(results) >= EARLY_TERMINATION_MIN_TESTS:
                        # Get current confidence level
                        tracker = load_confidence_tracker()
                        if (tracker and 'confidence_metrics' in tracker and 
                            'overall_confidence' in tracker['confidence_metrics']):
                            confidence = tracker['confidence_metrics']['overall_confidence']
                            
                            # If confidence is low, increment counter
                            if confidence < EARLY_TERMINATION_CONFIDENCE_THRESHOLD:
                                low_confidence_count += 1
                                # Only terminate if we've had low confidence for a few consecutive checks
                                if low_confidence_count >= EARLY_TERMINATION_CONSECUTIVE_CHECKS:
                                    console.print(f"[red]Early termination: Confidence below {EARLY_TERMINATION_CONFIDENCE_THRESHOLD} for {EARLY_TERMINATION_CONSECUTIVE_CHECKS} consecutive checks after {len(results)} tests")
                                    early_termination = True
                                    remaining_futures = []  # Clear remaining futures to terminate
                                    break
                            else:
                                # Reset counter if confidence improves
                                low_confidence_count = 0
            except Exception as e:
                log.error(f"Error in Ray execution: {e}")
                # Try to continue with next batch
                if remaining_futures:
                    continue
                else:
                    break
    
    # Handle case with no successful results
    if not results:
        console.print("[red]No successful test results obtained. Check if stocks data is available.")
        return None, None, early_termination
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter results_df to only include the requested columns
    if 'start_date' in results_df.columns and 'end_date' in results_df.columns:
        results_df = results_df[['symbol', 'start_date', 'end_date', 'total_return', 'alpha', 'theta']]
    
    # Calculate summary statistics
    summary = {
        'n_tests': len(results),
        'avg_return': results_df['total_return'].mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean(),
        'confidence': (results_df['alpha'] > 0).mean() * 0.5 + (results_df['theta'] > 0).mean() * 0.3 + (results_df['total_return'] > 0).mean() * 0.2
    }
    
    # Save results
    results_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_results_{timestamp}.csv')
    summary_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_summary_{timestamp}.json')
    
    results_df.to_csv(results_file, index=False)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Display summary
    summary_table = Table(title="Out-of-Sample Test Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    for key, value in summary.items():
        if isinstance(value, float):
            if key in ['avg_return', 'avg_alpha', 'alpha_success_rate', 'avg_theta', 'theta_success_rate', 'confidence']:
                summary_table.add_row(key, f"{value:.2%}")
            else:
                summary_table.add_row(key, f"{value:.2f}")
        else:
            summary_table.add_row(key, str(value))
    
    console.print(summary_table)
    console.print(f"\nResults saved to {results_file}")
    console.print(f"Summary saved to {summary_file}")
    
    # Final update to confidence tracker with complete results
    update_confidence_tracker(params, summary, results_df, timestamp, timeframe)
    
    # Return results for further analysis
    return results_df, summary, early_termination

def run_out_of_sample_test(params, start_date=None, end_date=None, n_periods=1, 
                          period_length=60, n_stocks=30, timeframe='1d', n_cores=None,
                          selected_stocks=None, selected_periods=None):
    """Run out-of-sample testing on random stocks and periods."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Get S&P 500 tickers
    if selected_stocks is None:
        log.info("Getting S&P 500 tickers...")
        sp500_tickers = get_sp500_tickers()
        selected_stocks = random.sample(sp500_tickers, min(n_stocks, len(sp500_tickers)))
        log.info(f"Selected {len(selected_stocks)} stocks for testing")
    else:
        log.info(f"Using pre-selected {len(selected_stocks)} stocks")
    
    # Generate random periods
    if selected_periods is None:
        if start_date is None:
            start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
        if end_date is None:
            end_date = datetime.now(pytz.UTC)
        
        log.info("Generating test periods...")
        periods = generate_random_periods(start_date, end_date, n_periods, period_length)
    else:
        periods = selected_periods
        log.info(f"Using pre-selected {len(periods)} time periods")
    
    # Prepare test cases
    test_cases = []
    for stock in selected_stocks:
        for period_start, period_end in periods:
            test_cases.append((params, stock, period_start, period_end, timeframe))
    
    total_tests = len(test_cases)
    log.info(f"Running {total_tests} tests one by one with live confidence updates...")
    
    # Run tests one by one for live updates
    results = []
    
    # Create timestamp for this batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Flag for early termination
    early_termination = False
    
    # Track low confidence count - only terminate if confidence stays low
    low_confidence_count = 0
    
    # Set up progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Testing...", total=total_tests)
        
        for i, case in enumerate(test_cases):
            # Run test for this case
            result = test_parameters_on_stock_period(*case)
            
            if result is not None:
                results.append(result)
                
                # Update progress
                progress.update(task, advance=1)
                
                # Update confidence after each test
                if i > 0 and len(results) > 1:  # Skip first test as we need at least 2 for meaningful confidence
                    current_summary, overall_confidence = update_confidence_after_single_test(params, result, i+1, results, timestamp, timeframe)
                    
                    # Display live update
                    display_live_confidence_update(result, i+1, current_summary, overall_confidence)
                    
                    # Check if we should do early termination after minimum number of tests
                    if EARLY_TERMINATION_ENABLED and len(results) >= EARLY_TERMINATION_MIN_TESTS:
                        # Get current confidence level
                        tracker = load_confidence_tracker()
                        if (tracker and 'confidence_metrics' in tracker and 
                            'overall_confidence' in tracker['confidence_metrics']):
                            confidence = tracker['confidence_metrics']['overall_confidence']
                            
                            # If confidence is low, increment counter
                            if confidence < EARLY_TERMINATION_CONFIDENCE_THRESHOLD:
                                low_confidence_count += 1
                                # Only terminate if we've had low confidence for a few consecutive tests
                                if low_confidence_count >= EARLY_TERMINATION_CONSECUTIVE_CHECKS:
                                    console.print(f"[red]Early termination: Confidence below {EARLY_TERMINATION_CONFIDENCE_THRESHOLD} for {EARLY_TERMINATION_CONSECUTIVE_CHECKS} consecutive checks after {len(results)} tests")
                                    early_termination = True
                                    break  # Exit the test loop early
                            else:
                                # Reset counter if confidence improves
                                low_confidence_count = 0
    
    # Handle case with no successful results
    if not results:
        console.print("[red]No successful test results obtained. Check if stocks data is available.")
        return None, None, early_termination
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter results_df to only include the requested columns
    if 'start_date' in results_df.columns and 'end_date' in results_df.columns:
        results_df = results_df[['symbol', 'start_date', 'end_date', 'total_return', 'alpha', 'theta']]
    
    # Calculate summary statistics
    summary = {
        'n_tests': len(results),
        'avg_return': results_df['total_return'].mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean(),
        'confidence': (results_df['alpha'] > 0).mean() * 0.5 + (results_df['theta'] > 0).mean() * 0.3 + (results_df['total_return'] > 0).mean() * 0.2
    }
    
    # Save results
    results_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_results_{timestamp}.csv')
    summary_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_summary_{timestamp}.json')
    
    results_df.to_csv(results_file, index=False)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Display summary
    summary_table = Table(title="Out-of-Sample Test Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    for key, value in summary.items():
        if isinstance(value, float):
            if key in ['avg_return', 'avg_alpha', 'alpha_success_rate', 'avg_theta', 'theta_success_rate', 'confidence']:
                summary_table.add_row(key, f"{value:.2%}")
            else:
                summary_table.add_row(key, f"{value:.2f}")
        else:
            summary_table.add_row(key, str(value))
    
    console.print(summary_table)
    console.print(f"\nResults saved to {results_file}")
    console.print(f"Summary saved to {summary_file}")
    
    # Final update to confidence tracker with complete results
    update_confidence_tracker(params, summary, results_df, timestamp, timeframe)
    
    # Return results for further analysis
    return results_df, summary, early_termination 