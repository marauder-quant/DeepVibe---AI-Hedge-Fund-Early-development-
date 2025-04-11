#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running out-of-sample tests for QMAC strategy.
"""

import os
import sys
import argparse
import logging
import random
from datetime import datetime
import pytz
from rich.console import Console
from rich.logging import RichHandler

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from other modules
from backtests.qmac_strategy.walk_forward_optimization.oos_config import (
    DEFAULT_OOS_STOCKS, MAX_OOS_WINDOWS, DEFAULT_OOS_WINDOW_LENGTH, 
    DEFAULT_TOP_PARAMS, EARLY_TERMINATION_ENABLED, EARLY_TERMINATION_MIN_TESTS,
    EARLY_TERMINATION_CONFIDENCE_THRESHOLD, EARLY_TERMINATION_CONSECUTIVE_CHECKS,
    DEFAULT_TIMEFRAME
)
from backtests.qmac_strategy.walk_forward_optimization.oos_core import (
    run_out_of_sample_test, run_out_of_sample_test_parallel
)
from backtests.qmac_strategy.walk_forward_optimization.oos_confidence import (
    display_confidence_summary
)
from backtests.qmac_strategy.walk_forward_optimization.oos_database import (
    save_best_oos_parameters_to_db
)
from backtests.qmac_strategy.src.qmac_db_query import (
    get_parameters_from_db, get_available_timeframes
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")
console = Console()

def main():
    """Main function to run out-of-sample testing."""
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
        from backtests.qmac_strategy.walk_forward_optimization.oos_utils import get_sp500_tickers, generate_random_periods
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
                from backtests.qmac_strategy.walk_forward_optimization.oos_confidence import load_confidence_tracker
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