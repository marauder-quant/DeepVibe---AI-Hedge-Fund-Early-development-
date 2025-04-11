#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Out-of-sample testing for QMAC strategy.
This script tests the best parameters found from optimization on different stocks and time periods.

Note: The confidence system has been modified to use only Alpha Quality as the confidence metric.
Alpha Quality = 60% alpha success rate + 30% mean alpha + 10% t-statistic significance.
"""

import os
import sys
import json
import logging
import multiprocessing as mp
import random
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.logging import RichHandler
import warnings
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import ray
import sqlite3

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import QMAC strategy
try:
    from backtests.qmac_strategy.src.qmac_strategy import run_qmac_strategy
    from backtests.qmac_strategy.src.config import (
        DEFAULT_OOS_STOCKS, MAX_OOS_WINDOWS, DEFAULT_OOS_WINDOW_LENGTH,
        DEFAULT_TOP_PARAMS, EARLY_TERMINATION_ENABLED, EARLY_TERMINATION_MIN_TESTS,
        EARLY_TERMINATION_CONFIDENCE_THRESHOLD, EARLY_TERMINATION_CONSECUTIVE_CHECKS,
        DEFAULT_TIMEFRAME
    )
    from backtests.qmac_strategy.src.qmac_db_query import get_parameters_from_db, get_available_timeframes
except ImportError:
    # If direct import fails, try relative import
    from src.qmac_strategy import run_qmac_strategy
    from src.config import (
        DEFAULT_OOS_STOCKS, MAX_OOS_WINDOWS, DEFAULT_OOS_WINDOW_LENGTH,
        DEFAULT_TOP_PARAMS, EARLY_TERMINATION_ENABLED, EARLY_TERMINATION_MIN_TESTS,
        EARLY_TERMINATION_CONFIDENCE_THRESHOLD, EARLY_TERMINATION_CONSECUTIVE_CHECKS,
        DEFAULT_TIMEFRAME
    )
    from src.qmac_db_query import get_parameters_from_db, get_available_timeframes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")
console = Console()

# Suppress warnings
warnings.filterwarnings('ignore')

# Confidence tracking constants
CONFIDENCE_TRACKER_FILE = 'backtests/qmac_strategy/results/confidence_tracker.json'

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
                    update_confidence_after_batch(params, results_df, timestamp, timeframe)
                    
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

def update_confidence_after_batch(params, results_df, timestamp, timeframe):
    """Update the confidence metrics after processing a batch of results."""
    # Calculate current summary statistics
    current_summary = {
        'n_tests': len(results_df),
        'avg_return': results_df['total_return'].mean(),
        'median_return': results_df['total_return'].median(),
        'std_return': results_df['total_return'].std() if len(results_df) > 1 else 0,
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_trades': results_df['n_trades'].mean(),
        'success_rate': (results_df['total_return'] > 0).mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean()
    }
    
    # Get in-sample performance data
    in_sample_data = get_latest_in_sample_performance(timeframe)
    
    # Calculate confidence metric - only using alpha_quality now
    alpha_quality = calculate_alpha_quality(results_df)
    overall_confidence = alpha_quality  # Directly use alpha_quality as the overall confidence
    
    # Create a mini table for batch update
    batch_table = Table(title=f"Batch Update: {len(results_df)} Tests Completed")
    batch_table.add_column("Metric", style="cyan")
    batch_table.add_column("Value", style="green")
    
    batch_table.add_row("Avg Return", f"{current_summary['avg_return']:.2%}")
    batch_table.add_row("Avg Alpha", f"{current_summary['avg_alpha']:.2%}")
    batch_table.add_row("Alpha Success", f"{current_summary['alpha_success_rate']:.2%}")
    batch_table.add_row("Avg Theta", f"{current_summary['avg_theta']:.2%}")
    batch_table.add_row("Current Confidence", f"{overall_confidence:.2%}")
    
    console.print(batch_table)

def get_sp500_tickers():
    """Get list of S&P 500 tickers from Wikipedia."""
    try:
        # Try to get from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        print(f"Error getting S&P 500 tickers from Wikipedia: {e}")
        # Fallback to a sample list
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']

def generate_random_periods(start_date, end_date, n_periods=30, period_length=60):
    """Generate random 2-month periods within the date range."""
    periods = []
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    for _ in range(n_periods):
        # Generate random start date
        days_range = (end_date - start_date).days - period_length
        if days_range <= 0:
            continue
            
        random_start = start_date + timedelta(days=random.randint(0, days_range))
        random_end = random_start + timedelta(days=period_length)
        periods.append((random_start, random_end))
    
    return periods

def test_parameters_on_stock_period(params, symbol, start_date, end_date, timeframe='1d'):
    """Test QMAC parameters on a specific stock and period."""
    try:
        # Run strategy with given parameters
        result = run_qmac_strategy(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            buy_fast_window=params['buy_fast'],
            buy_slow_window=params['buy_slow'],
            sell_fast_window=params['sell_fast'],
            sell_slow_window=params['sell_slow'],
            timeframe=timeframe,
            verbose=False
        )
        
        # Check if result contains required keys
        if 'qmac_pf' not in result:
            print(f"Missing portfolio data for {symbol} from {start_date} to {end_date}")
            return None
        
        # Get performance metrics from QMAC strategy
        qmac_pf = result['qmac_pf']
        total_return = qmac_pf.total_return()
        
        # Get buy and hold performance metrics
        if 'hold_pf' in result:
            # Use the hold portfolio provided by run_qmac_strategy
            hold_pf = result['hold_pf']
            buy_and_hold_return = hold_pf.total_return()
            buy_and_hold_drawdown = hold_pf.stats()['Max Drawdown [%]'] / 100
            print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (from strategy), drawdown: {buy_and_hold_drawdown:.2%}")
        else:
            # Fallback method using the price data directly
            try:
                if 'ohlcv' in result and len(result['ohlcv']) > 1:
                    # Calculate from price data
                    data = result['ohlcv']
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    buy_and_hold_return = (last_price / first_price) - 1
                    
                    # Calculate drawdown for buy and hold
                    equity_curve = data['Close'] / first_price
                    peak = equity_curve.cummax()
                    drawdown = (equity_curve - peak) / peak
                    buy_and_hold_drawdown = abs(drawdown.min())
                    
                    print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (calculated from prices), drawdown: {buy_and_hold_drawdown:.2%}")
                else:
                    # No price data available, use a benchmark estimate
                    period_days = (end_date - start_date).days
                    buy_and_hold_return = 0.08 / 365 * period_days  # Annualized 8% return
                    buy_and_hold_drawdown = 0.03  # Default 3% drawdown
                    print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (estimated), drawdown: {buy_and_hold_drawdown:.2%}")
            except Exception as e:
                print(f"Error calculating buy & hold from prices for {symbol}: {e}")
                buy_and_hold_return = 0.01  # Default 1% return
                buy_and_hold_drawdown = 0.03  # Default 3% drawdown
        
        # Calculate alpha (strategy return - buy and hold return)
        alpha = total_return - buy_and_hold_return
        
        # Calculate theta (buy and hold drawdown - strategy drawdown)
        strategy_drawdown = qmac_pf.stats()['Max Drawdown [%]'] / 100
        theta = buy_and_hold_drawdown - strategy_drawdown
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_return': total_return,
            'sharpe_ratio': qmac_pf.stats()['Sharpe Ratio'],
            'max_drawdown': strategy_drawdown,
            'n_trades': len(qmac_pf.trades),
            'buy_and_hold_return': buy_and_hold_return,
            'buy_and_hold_drawdown': buy_and_hold_drawdown,
            'alpha': alpha,
            'theta': theta
        }
    except Exception as e:
        print(f"Error testing {symbol} from {start_date} to {end_date}: {e}")
        return None

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
                    update_confidence_after_single_test(params, result, i+1, results, timestamp, timeframe)
                    
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

def update_confidence_after_single_test(params, result, test_num, results_so_far, timestamp, timeframe):
    """Update confidence tracker after a single test for live updates."""
    log.info(f"Updating confidence after test {test_num}...")
    
    # Convert results so far to DataFrame
    results_df = pd.DataFrame(results_so_far)
    
    # Calculate current summary statistics
    current_summary = {
        'n_tests': len(results_df),
        'avg_return': results_df['total_return'].mean(),
        'median_return': results_df['total_return'].median(),
        'std_return': results_df['total_return'].std() if len(results_df) > 1 else 0,
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_trades': results_df['n_trades'].mean(),
        'success_rate': (results_df['total_return'] > 0).mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean()
    }
    
    # Get in-sample performance data
    in_sample_data = get_latest_in_sample_performance(timeframe)
    
    # Calculate confidence metric - only using alpha_quality now
    alpha_quality = calculate_alpha_quality(results_df)
    overall_confidence = alpha_quality  # Directly use alpha_quality as the overall confidence
    
    # Load existing tracker
    tracker = load_confidence_tracker()
    
    # Create test entry for current state
    test_entry = {
        'timestamp': f"{timestamp}_test{test_num}",
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params,
        'timeframe': timeframe,
        'in_sample_return': in_sample_data.get('total_return', 0),
        'out_of_sample_return': current_summary['avg_return'],
        'return_delta': current_summary['avg_return'] - in_sample_data.get('total_return', 0),
        'success_rate': current_summary['success_rate'],
        'n_tests': current_summary['n_tests'],
        'avg_alpha': current_summary['avg_alpha'],
        'alpha_success_rate': current_summary['alpha_success_rate'],
        'avg_theta': current_summary['avg_theta'],
        'theta_success_rate': current_summary['theta_success_rate'],
        'confidence_metrics': {
            'alpha_quality': alpha_quality,
            'p_value': calculate_statistical_confidence(results_df['total_return']) if len(results_df) > 1 else 1.0
        }
    }
    
    # If there are previous entries from this batch, remove them
    # to avoid cluttering the tracker with intermediate results
    if 'tests' in tracker:
        tracker['tests'] = [t for t in tracker['tests'] 
                           if 'timestamp' in t and not t['timestamp'].startswith(timestamp)]
    else:
        tracker['tests'] = []
    
    # Add current test entry
    tracker['tests'].append(test_entry)
    
    # Calculate alpha quality score across all tests
    alpha_quality_score = calculate_avg_alpha_quality(tracker['tests'])
    
    # Update overall confidence metrics - use alpha_quality_score as the overall confidence
    tracker['confidence_metrics'] = {
        'alpha_quality_score': alpha_quality_score,
        'overall_confidence': alpha_quality_score  # Use alpha_quality_score as overall_confidence
    }
    tracker['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated tracker
    save_confidence_tracker(tracker)
    
    # Display current confidence metrics
    display_live_confidence_update(result, test_num, current_summary, overall_confidence)

def display_live_confidence_update(latest_result, test_num, current_summary, overall_confidence):
    """Display a live update of confidence metrics after each test."""
    # Create a mini table for the latest test result
    latest_table = Table(title=f"Test #{test_num} Result: {latest_result['symbol']}")
    latest_table.add_column("Metric", style="cyan")
    latest_table.add_column("Value", style="green")
    
    latest_table.add_row("Return", f"{latest_result['total_return']:.2%}")
    latest_table.add_row("Buy & Hold", f"{latest_result['buy_and_hold_return']:.2%}")
    latest_table.add_row("Alpha", f"{latest_result['alpha']:.2%}")
    latest_table.add_row("Drawdown", f"{latest_result['max_drawdown']:.2%}")
    latest_table.add_row("B&H Drawdown", f"{latest_result['buy_and_hold_drawdown']:.2%}")
    latest_table.add_row("Theta", f"{latest_result['theta']:.2%}")
    
    # Create a summary table
    summary_table = Table(title="Running Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Avg Return", f"{current_summary['avg_return']:.2%}")
    summary_table.add_row("Avg Alpha", f"{current_summary['avg_alpha']:.2%}")
    summary_table.add_row("Alpha Success", f"{current_summary['alpha_success_rate']:.2%}")
    summary_table.add_row("Avg Theta", f"{current_summary['avg_theta']:.2%}")
    summary_table.add_row("Theta Success", f"{current_summary['theta_success_rate']:.2%}")
    summary_table.add_row("Current Confidence", f"{overall_confidence:.2%}")
    
    # Choose color based on confidence level
    confidence_color = "[green]" if overall_confidence >= 0.7 else "[yellow]" if overall_confidence >= 0.4 else "[red]"
    
    # Display tables side by side
    console.print(Panel.fit(
        Columns([latest_table, summary_table]), 
        title=f"{confidence_color}Confidence Update after {test_num} tests"
    ))

def load_confidence_tracker():
    """Load the confidence tracking data from file or initialize if not exists."""
    if os.path.exists(CONFIDENCE_TRACKER_FILE):
        with open(CONFIDENCE_TRACKER_FILE, 'r') as f:
            return json.load(f)
    else:
        # Initialize new confidence tracker
        return {
            'tracker_version': '1.0',
            'tests': [],
            'confidence_metrics': {
                'alpha_quality_score': 0,
                'overall_confidence': 0
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def save_confidence_tracker(data):
    """Save the confidence tracking data to file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(CONFIDENCE_TRACKER_FILE), exist_ok=True)
    with open(CONFIDENCE_TRACKER_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def update_confidence_tracker(params, summary, results_df, timestamp, timeframe):
    """Update the confidence tracker with new test results."""
    log.info("Updating confidence tracker...")
    
    # Load existing tracker data
    tracker = load_confidence_tracker()
    
    # Calculate confidence metric
    confidence = summary['alpha_success_rate'] * 0.5 + summary['theta_success_rate'] * 0.3 + (results_df['total_return'] > 0).mean() * 0.2
    
    # Create new test entry with only the requested fields
    test_entry = {
        'timestamp': timestamp,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date_from': results_df['start_date'].min().strftime('%Y-%m-%d') if 'start_date' in results_df.columns else None,
        'date_to': results_df['end_date'].max().strftime('%Y-%m-%d') if 'end_date' in results_df.columns else None,
        'out_of_sample_return': summary['avg_return'],
        'avg_alpha': summary['avg_alpha'],
        'alpha_success_rate': summary['alpha_success_rate'],
        'avg_theta': summary['avg_theta'],
        'theta_success_rate': summary['theta_success_rate'],
        'confidence': confidence
    }
    
    # Add test to tracker
    tracker['tests'].append(test_entry)
    
    # Update overall confidence metrics
    tracker['confidence_metrics'] = {
        'overall_confidence': confidence
    }
    tracker['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated tracker
    save_confidence_tracker(tracker)
    
    # Generate confidence report
    generate_confidence_report(tracker, timestamp)
    
    log.info(f"Confidence tracker updated. Overall confidence score: {confidence:.2%}")
    
    return confidence

def get_latest_in_sample_performance(timeframe):
    """Get the latest in-sample performance metrics."""
    results_dir = 'backtests/qmac_strategy/results'
    # Look for in-sample result files with this timeframe
    csv_files = [f for f in os.listdir(results_dir) 
                if f.endswith('.csv') and f'qmac_results_SPY_{timeframe}' in f]
    
    if not csv_files:
        log.warning(f"No in-sample results found for timeframe {timeframe}")
        return {'total_return': 0}
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    results_df = pd.read_csv(os.path.join(results_dir, latest_file))
    
    # Get the best parameters
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    
    # Create result dict with available columns
    result = {'total_return': best_row['total_return']}
    
    # Add optional metrics if they exist in the dataframe
    for metric in ['sharpe_ratio', 'max_drawdown']:
        if metric in best_row:
            result[metric] = best_row[metric]
    
    return result

def calculate_alpha_quality(results_df):
    """
    Calculate alpha quality score based on alpha distribution.
    Higher values indicate strategy adds more value over buy-and-hold.
    """
    # If no results, return zero
    if len(results_df) == 0:
        return 0.0
    
    # Calculate percentage of positive alphas
    alpha_success_rate = (results_df['alpha'] > 0).mean()
    
    # Calculate average alpha
    mean_alpha = results_df['alpha'].mean()
    
    # Calculate alpha t-statistic
    if len(results_df) > 1:
        alpha_t_stat = stats.ttest_1samp(results_df['alpha'], 0)[0]
        # Normalize t-stat to [0, 1] range, cap at 3 (very significant)
        normalized_t = min(abs(alpha_t_stat) / 3, 1.0) if alpha_t_stat > 0 else 0.0
    else:
        normalized_t = 0.0
    
    # Combine metrics: 60% on success rate, 30% on mean alpha, 10% on t-stat
    alpha_quality = (
        0.6 * alpha_success_rate + 
        0.3 * min(max(mean_alpha * 3, 0), 1) +  # Scale mean alpha, cap at 0-1
        0.1 * normalized_t
    )
    
    return alpha_quality

def calculate_statistical_confidence(returns):
    """Calculate statistical confidence using t-test."""
    # One-sample t-test against zero
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    
    # If t-statistic is positive, we're testing if mean is > 0
    # If negative, we don't have evidence strategy works
    if t_stat <= 0:
        return 1.0  # worst p-value
    
    return p_value

def calculate_avg_alpha_quality(tests):
    """Calculate average alpha quality from all tests."""
    if not tests:
        return 0.0
    
    alphas = []
    for test in tests:
        if 'confidence_metrics' in test and 'alpha_quality' in test['confidence_metrics']:
            alphas.append(test['confidence_metrics']['alpha_quality'])
    
    if not alphas:
        return 0.0
        
    return sum(alphas) / len(alphas)

def generate_confidence_report(tracker, timestamp):
    """Generate and save a visual confidence report."""
    report_file = os.path.join('backtests/qmac_strategy/results', f'confidence_report_{timestamp}.png')
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Extract data from tracker
    tests = tracker['tests']
    if len(tests) < 2:  # Need at least 2 tests for meaningful plots
        log.info("Not enough data to generate confidence report yet")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([{
        'date': test['date'],
        'start_date': test.get('date_from', test['date']),
        'end_date': test.get('date_to', test['date']),
        'avg_return': test['out_of_sample_return'],
        'avg_alpha': test.get('avg_alpha', 0),
        'alpha_success': test.get('alpha_success_rate', 0),
        'avg_theta': test.get('avg_theta', 0),
        'theta_success': test.get('theta_success_rate', 0),
        'confidence': test.get('confidence_metrics', {}).get('alpha_quality', 0) * 0.5 + 
                      test.get('theta_success_rate', 0) * 0.3 + 
                      test.get('success_rate', 0) * 0.2
    } for test in tests])
    df['date'] = pd.to_datetime(df['date'])
    
    # Configure the plot - use a modern style that's available in all matplotlib versions
    try:
        plt.style.use('ggplot')
    except:
        # Fallback to default style if ggplot is not available
        plt.style.use('default')
    
    # Plot 1: Average Return and Alpha
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['avg_return'], 'g-', label='Avg Return')
    plt.plot(df['date'], df['avg_alpha'], 'r-', label='Avg Alpha')
    plt.title('Performance Metrics Over Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Plot 2: Alpha and Theta Success Rates
    plt.subplot(2, 2, 2)
    plt.plot(df['date'], df['alpha_success'], 'b-', label='Alpha Success')
    plt.plot(df['date'], df['theta_success'], 'y-', label='Theta Success')
    plt.title('Success Rates Over Time')
    plt.ylabel('Rate (0-1)')
    plt.legend()
    plt.tight_layout()
    
    # Plot 3: Average Theta
    plt.subplot(2, 2, 3)
    plt.plot(df['date'], df['avg_theta'], 'm-', label='Avg Theta')
    plt.title('Theta Performance Over Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Plot 4: Confidence
    plt.subplot(2, 2, 4)
    plt.plot(df['date'], df['confidence'], 'g-', label='Confidence')
    plt.title('Strategy Confidence Over Time')
    plt.ylabel('Confidence (0-1)')
    plt.legend()
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle(f'QMAC Strategy Performance Summary - {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(report_file)
    log.info(f"Confidence report saved to {report_file}")
    plt.close()

def display_confidence_summary():
    """Display a summary of the current confidence in the strategy."""
    if not os.path.exists(CONFIDENCE_TRACKER_FILE):
        console.print("[yellow]No confidence tracking data available yet. Run out-of-sample tests to generate data.")
        return
    
    tracker = load_confidence_tracker()
    
    # Create a rich table
    table = Table(title="QMAC Strategy Confidence Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add confidence metrics
    metrics = tracker['confidence_metrics']
    table.add_row("Overall Confidence", f"{metrics.get('overall_confidence', 0):.2%}")
    
    # Add alpha metrics if available
    if 'alpha_quality_score' in metrics:
        table.add_row("Alpha Quality", f"{metrics['alpha_quality_score']:.2%}")
    
    # Add latest test info if available
    if tracker['tests']:
        latest = tracker['tests'][-1]
        table.add_row("Latest Test Date", latest['date'])
        table.add_row("In-sample Return", f"{latest['in_sample_return']:.2%}")
        table.add_row("Out-of-sample Return", f"{latest['out_of_sample_return']:.2%}")
        
        # Add alpha and theta info if available
        if 'avg_alpha' in latest:
            table.add_row("Alpha", f"{latest['avg_alpha']:.2%}")
            table.add_row("Alpha Success Rate", f"{latest['alpha_success_rate']:.2%}")
        
        if 'avg_theta' in latest:
            table.add_row("Theta", f"{latest['avg_theta']:.2%}")
            table.add_row("Theta Success Rate", f"{latest['theta_success_rate']:.2%}")
        
        table.add_row("Success Rate", f"{latest['success_rate']:.2%}")
    
    table.add_row("Total Tests Run", str(len(tracker['tests'])))
    table.add_row("Last Updated", tracker['last_updated'])
    
    console.print(table)
    
    # If confidence is high, print a success message
    if metrics.get('overall_confidence', 0) >= 0.7:
        console.print("[green]High confidence in strategy parameters!")
    elif metrics.get('overall_confidence', 0) >= 0.4:
        console.print("[yellow]Moderate confidence in strategy parameters.")
    else:
        console.print("[red]Low confidence in strategy parameters. Consider re-optimizing.")

# OOS Database settings
OOS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db', 'qmac_oos_parameters.db')

def initialize_oos_database():
    """
    Initialize the SQLite database for storing best out-of-sample parameters.
    
    Returns:
        None
    """
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(OOS_DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    c = conn.cursor()
    
    # Create table for OOS parameter combinations if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS best_oos_parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        rank INTEGER NOT NULL,
        buy_fast INTEGER NOT NULL,
        buy_slow INTEGER NOT NULL,
        sell_fast INTEGER NOT NULL,
        sell_slow INTEGER NOT NULL,
        avg_return REAL NOT NULL,
        avg_alpha REAL,
        alpha_success_rate REAL,
        avg_theta REAL,
        theta_success_rate REAL,
        success_rate REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        confidence_score REAL,
        num_tests INTEGER,
        date_from TEXT NOT NULL,
        date_to TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        UNIQUE(symbol, timeframe, rank, date_from, date_to)
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    log.info(f"OOS Database initialized at {OOS_DB_PATH}")

def save_best_oos_parameters_to_db(params, symbol, timeframe, start_date, end_date, summary, 
                                 confidence_score, top_n=1):
    """
    Save the best out-of-sample parameter combination to the database.
    
    Args:
        params (dict): Parameter dictionary with buy_fast, buy_slow, sell_fast, sell_slow
        symbol (str): Trading symbol
        timeframe (str): Data timeframe
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        summary (dict): Summary statistics dictionary
        confidence_score (float): Overall confidence score
        top_n (int): Number of top parameter combinations to save (usually 1 for OOS)
        
    Returns:
        None
    """
    # Initialize the database if it doesn't exist
    if not os.path.exists(OOS_DB_PATH):
        initialize_oos_database()
    
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    c = conn.cursor()
    
    # Format dates for database
    date_from = start_date.strftime('%Y-%m-%d')
    date_to = end_date.strftime('%Y-%m-%d')
    timestamp = datetime.now().isoformat()
    
    # Delete existing entries for this symbol, timeframe, date range
    c.execute('''
    DELETE FROM best_oos_parameters 
    WHERE symbol = ? AND timeframe = ? AND date_from = ? AND date_to = ?
    ''', (symbol, timeframe, date_from, date_to))
    
    # Extract parameters
    buy_fast = int(params['buy_fast'])
    buy_slow = int(params['buy_slow'])
    sell_fast = int(params['sell_fast'])
    sell_slow = int(params['sell_slow'])
    
    # Extract metrics from summary
    avg_return = float(summary.get('avg_return', 0))
    avg_alpha = float(summary.get('avg_alpha', 0))
    alpha_success_rate = float(summary.get('alpha_success_rate', 0))
    avg_theta = float(summary.get('avg_theta', 0))
    theta_success_rate = float(summary.get('theta_success_rate', 0))
    success_rate = float(summary.get('success_rate', 0))
    sharpe_ratio = float(summary.get('avg_sharpe', 0))
    max_drawdown = float(summary.get('avg_max_drawdown', 0))
    num_tests = int(summary.get('n_tests', 0))
        
    # Insert into database
    c.execute('''
    INSERT INTO best_oos_parameters 
    (symbol, timeframe, rank, buy_fast, buy_slow, sell_fast, sell_slow, 
     avg_return, avg_alpha, alpha_success_rate, avg_theta, theta_success_rate,
     success_rate, sharpe_ratio, max_drawdown, confidence_score, num_tests,
     date_from, date_to, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        symbol, timeframe, 1, buy_fast, buy_slow, sell_fast, sell_slow,
        avg_return, avg_alpha, alpha_success_rate, avg_theta, theta_success_rate,
        success_rate, sharpe_ratio, max_drawdown, confidence_score, num_tests,
        date_from, date_to, timestamp
    ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    log.info(f"Saved best OOS parameters to database for {symbol} {timeframe}")

def get_best_oos_parameters_from_db(symbol=None, timeframe=None):
    """
    Query the database for best out-of-sample parameters.
    
    Args:
        symbol (str, optional): Filter by symbol
        timeframe (str, optional): Filter by timeframe
        
    Returns:
        pd.DataFrame: DataFrame containing parameter combinations
    """
    # Check if database exists
    if not os.path.exists(OOS_DB_PATH):
        log.warning(f"OOS Database not found at {OOS_DB_PATH}")
        return pd.DataFrame()
    
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    
    # Build query based on filters
    query = "SELECT * FROM best_oos_parameters"
    params = []
    
    if symbol or timeframe:
        query += " WHERE"
        
        if symbol:
            query += " symbol = ?"
            params.append(symbol)
            
        if timeframe:
            if symbol:
                query += " AND"
            query += " timeframe = ?"
            params.append(timeframe)
    
    query += " ORDER BY symbol, timeframe, date_from DESC, date_to DESC, rank"
    
    # Execute query and load results into DataFrame
    df = pd.read_sql_query(query, conn, params=params)
    
    # Close connection
    conn.close()
    
    return df

def main():
    """Main function to run out-of-sample testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run out-of-sample tests for QMAC strategy')
    parser.add_argument('--display-confidence', action='store_true', help='Display current confidence summary')
    parser.add_argument('--n-stocks', type=int, default=DEFAULT_OOS_STOCKS, help='Number of stocks to test')
    parser.add_argument('--n-windows', type=int, default=MAX_OOS_WINDOWS, help='Number of time windows to test per stock')
    parser.add_argument('--window-length', type=int, default=DEFAULT_OOS_WINDOW_LENGTH, help='Length of each test window in days')
    parser.add_argument('--n-backtests', type=int, default=1, help='Number of backtest runs to execute')
    parser.add_argument('--top-params', type=int, default=DEFAULT_TOP_PARAMS, help='Number of top parameter sets to test')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for faster execution')
    parser.add_argument('--n-cores', type=int, default=None, help='Number of CPU cores to use (default: auto-detect)')
    parser.add_argument('--same-stocks', action='store_true', help='Use same stock selection for all parameter sets')
    parser.add_argument('--disable-early-termination', action='store_true', help='Disable early termination feature')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with minimal samples')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, help=f'Timeframe to use (default: {DEFAULT_TIMEFRAME})')
    args = parser.parse_args()
    
    # Check if early termination should be disabled via command line
    early_termination_enabled = EARLY_TERMINATION_ENABLED and not args.disable_early_termination
    
    if args.display_confidence:
        display_confidence_summary()
        return
    
    # If quick test is enabled, override parameters for a fast test
    if args.quick_test:
        args.n_stocks = 3
        args.n_windows = 2
        args.top_params = 2
        args.n_backtests = 1
        log.info("Running in quick test mode with minimal samples")
    
    # Use the timeframe from command line or config
    timeframe = args.timeframe
    log.info(f"Using timeframe: {timeframe}")
    
    # Get parameters from database instead of CSV files
    log.info(f"Getting best parameters from database for timeframe {timeframe}...")
    
    # Check if we have data for this timeframe
    available_timeframes = get_available_timeframes(symbol="SPY")
    if timeframe not in available_timeframes:
        log.warning(f"No data found for timeframe {timeframe}")
        if available_timeframes:
            log.info(f"Available timeframes: {', '.join(available_timeframes)}")
            timeframe = available_timeframes[0]
            log.info(f"Using timeframe {timeframe} instead")
        else:
            log.error("No timeframes available in database. Please run QMAC strategy optimization first.")
            return
    
    # Get parameters from database
    params_df = get_parameters_from_db(symbol="SPY", timeframe=timeframe)
    
    # Check if we got any results
    if params_df.empty:
        log.error(f"No parameters found in database for SPY with timeframe {timeframe}")
        return
    
    # Determine how many parameter sets to test
    num_param_sets = min(args.top_params, len(params_df))
    log.info(f"Testing top {num_param_sets} parameter sets from database:")
    
    # Create list of parameter sets to test
    param_sets = []
    for i in range(num_param_sets):
        row = params_df.iloc[i]
        params = {
            'buy_fast': int(row['buy_fast']),
            'buy_slow': int(row['buy_slow']),
            'sell_fast': int(row['sell_fast']),
            'sell_slow': int(row['sell_slow'])
        }
        param_sets.append(params)
        
        # Format return as percentage if it exists
        return_val = row.get('total_return', row.get('performance', 0))
        return_str = f"{return_val:.2%}" if isinstance(return_val, float) else str(return_val)
        
        print(f"Rank #{i+1}: Buy Fast: {params['buy_fast']}, Buy Slow: {params['buy_slow']}, " + 
              f"Sell Fast: {params['sell_fast']}, Sell Slow: {params['sell_slow']}")
        print(f"Total Return: {return_str}")
    
    # If same_stocks flag is set, generate tickers and periods just once
    shared_stocks = None
    shared_periods = None
    if args.same_stocks:
        # Get S&P 500 tickers
        log.info("Pre-selecting stocks for all parameter sets...")
        sp500_tickers = get_sp500_tickers()
        shared_stocks = random.sample(sp500_tickers, min(args.n_stocks, len(sp500_tickers)))
        log.info(f"Selected {len(shared_stocks)} stocks for all parameter sets")
        
        # Generate random periods
        start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
        end_date = datetime.now(pytz.UTC)
        log.info("Generating shared test periods...")
        shared_periods = generate_random_periods(start_date, end_date, args.n_windows, args.window_length)
        log.info(f"Generated {len(shared_periods)} time periods for testing")
    
    # Track the best performing parameter set out-of-sample
    best_oos_params = None
    best_oos_return = -float('inf')
    best_oos_confidence = -float('inf')
    best_oos_summary = None
    
    # Early termination message about status
    if early_termination_enabled:
        log.info(f"Early termination is ENABLED (threshold: {EARLY_TERMINATION_CONFIDENCE_THRESHOLD}, " +
                 f"min tests: {EARLY_TERMINATION_MIN_TESTS}, consecutive checks: {EARLY_TERMINATION_CONSECUTIVE_CHECKS})")
    else:
        log.info("Early termination is DISABLED")
    
    # Run tests for each parameter set
    i = 0
    while i < len(param_sets):
        params = param_sets[i]
        
        # Run multiple backtests if requested
        for backtest_num in range(args.n_backtests):
            # Determine which parameter set to use
            if len(param_sets) > 1:
                print(f"\nStarting backtest using parameter set #{i+1}...")
            else:
                print(f"\nStarting backtest #{backtest_num+1} using parameter set #{i+1}...")
                
            # Decide whether to use parallel or sequential execution
            if args.parallel:
                # Run with parallel execution
                results, summary, early_terminated = run_out_of_sample_test_parallel(
                    params=params,
                    start_date=datetime(2018, 1, 1, tzinfo=pytz.UTC),
                    end_date=datetime.now(pytz.UTC),
                    n_periods=args.n_windows,
                    period_length=args.window_length,
                    n_stocks=args.n_stocks,
                    timeframe=timeframe,
                    n_cores=args.n_cores
                )
            else:
                # Run with sequential execution
                results, summary, early_terminated = run_out_of_sample_test(
                    params=params,
                    start_date=datetime(2018, 1, 1, tzinfo=pytz.UTC),
                    end_date=datetime.now(pytz.UTC),
                    n_periods=args.n_windows,
                    period_length=args.window_length,
                    n_stocks=args.n_stocks,
                    timeframe=timeframe,
                    n_cores=args.n_cores,
                    selected_stocks=shared_stocks,
                    selected_periods=shared_periods
                )
            
            # Check if we need to skip to the next parameter set due to early termination
            if early_termination_enabled and early_terminated and backtest_num < args.n_backtests - 1:
                console.print("[yellow]Skipping remaining backtests for this parameter set due to low confidence")
                break  # Skip remaining backtests for this parameter set
            
            # Check if this is the best performing parameter set so far
            if summary and summary['avg_return'] > best_oos_return:
                best_oos_params = params
                best_oos_return = summary['avg_return']
                best_oos_summary = summary
                # Get the latest confidence from the tracker
                tracker = load_confidence_tracker()
                if tracker and 'confidence_metrics' in tracker and 'overall_confidence' in tracker['confidence_metrics']:
                    best_oos_confidence = tracker['confidence_metrics']['overall_confidence']
                    
        # Move to the next parameter set
        i += 1
    
    # Display confidence summary after all testing
    display_confidence_summary()
    
    # Output the best performing parameter set
    if best_oos_params:
        print("\n" + "="*80)
        print(f"BEST OUT-OF-SAMPLE PARAMETERS:")
        print(f"Buy Fast: {best_oos_params['buy_fast']}, Buy Slow: {best_oos_params['buy_slow']}, " + 
              f"Sell Fast: {best_oos_params['sell_fast']}, Sell Slow: {best_oos_params['sell_slow']}")
        print(f"Out-of-sample Return: {best_oos_return:.2%}")
        print(f"Confidence Score: {best_oos_confidence:.2%}")
        print("="*80)
        
        # Save best out-of-sample parameters to database
        try:
            start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
            end_date = datetime.now(pytz.UTC)
            save_best_oos_parameters_to_db(
                params=best_oos_params,
                symbol="SPY",  # Using SPY as the representative symbol
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                summary=best_oos_summary,
                confidence_score=best_oos_confidence
            )
            print(f"Best OOS parameters saved to database")
        except Exception as e:
            log.error(f"Error saving best OOS parameters to database: {e}")

if __name__ == "__main__":
    main() 