#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quad Moving Average Crossover (QMAC) strategy main entry point.
This module provides command-line interface for running the QMAC strategy.
"""

import argparse
import os
import time
from datetime import datetime
import pytz
from dateutil.parser import parse

# Import configuration
from config import *

# Import from strategy modules
from lib.strategy_core import run_qmac_strategy
from lib.optimizer import analyze_window_combinations
from lib.visualization import plot_qmac_strategy, create_parameter_space_visualization, save_plots
from lib.utils import get_system_resources

def main():
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
    parser.add_argument('--no-opt', action='store_true',
                        help='Skip optimization and run with provided parameters')
    parser.add_argument('--buy-fast', type=int, default=None,
                        help='Buy fast window size (only used with --no-opt)')
    parser.add_argument('--buy-slow', type=int, default=None,
                        help='Buy slow window size (only used with --no-opt)')
    parser.add_argument('--sell-fast', type=int, default=None,
                        help='Sell fast window size (only used with --no-opt)')
    parser.add_argument('--sell-slow', type=int, default=None,
                        help='Sell slow window size (only used with --no-opt)')
    parser.add_argument('--max-memory-percent', type=int, default=80,
                        help='Maximum percentage of system memory to use (default: 80)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for processing (default: auto)')
    parser.add_argument('--adaptive-control', action='store_true', default=True,
                        help='Enable adaptive resource controls (default: True)')
    
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
    
    # Set environment variables for resource control
    if args.batch_size:
        os.environ["QMAC_BATCH_SIZE"] = str(args.batch_size)
        print(f"Setting custom batch size: {args.batch_size}")
    
    # Set environment variable for memory limit
    os.environ["QMAC_MAX_MEMORY_PERCENT"] = str(args.max_memory_percent)
    print(f"Setting memory limit to {args.max_memory_percent}% of system memory")
    
    # Set environment variable for adaptive control
    os.environ["QMAC_ADAPTIVE_CONTROL"] = "1" if args.adaptive_control else "0"
    print(f"Adaptive resource control is {'enabled' if args.adaptive_control else 'disabled'}")
    
    # Display system resources
    try:
        resources = get_system_resources()
        print("\nSystem Resources:")
        print(f"CPU: {resources['cpu_percent']}% used")
        print(f"Memory: {resources['memory_used_percent']}% used ({resources['memory_available_gb']:.2f} GB available)")
        print(f"Process memory: {resources['process_memory_gb']:.2f} GB\n")
    except Exception as e:
        print(f"Unable to analyze system resources: {e}\n")
    
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
    
    if args.no_opt:
        # Run with provided parameters
        if args.buy_fast is None or args.buy_slow is None or args.sell_fast is None or args.sell_slow is None:
            print("Error: When using --no-opt, you must provide all window parameters:")
            print("  --buy-fast, --buy-slow, --sell-fast, --sell-slow")
            return
        
        buy_fast = args.buy_fast
        buy_slow = args.buy_slow
        sell_fast = args.sell_fast
        sell_slow = args.sell_slow
        
        print(f"\nRunning QMAC strategy with provided parameters:")
        print(f"Buy signals: Fast MA = {buy_fast}, Slow MA = {buy_slow}")
        print(f"Sell signals: Fast MA = {sell_fast}, Slow MA = {sell_slow}")
        
        # Run strategy with provided parameters
        results = run_qmac_strategy(
            symbol, start_date, end_date,
            buy_fast_window=buy_fast,
            buy_slow_window=buy_slow,
            sell_fast_window=sell_fast,
            sell_slow_window=sell_slow,
            timeframe=timeframe,
            verbose=True
        )
        
        # Print strategy stats
        print("\nQMAC Strategy Performance:")
        stats = results['qmac_pf'].stats()
        print(stats)
        
        # Print hold strategy stats for comparison
        print("\nBuy & Hold Strategy Performance:")
        hold_stats = results['hold_pf'].stats()
        print(hold_stats)
        
        # Print performance comparison
        qmac_return = stats['Total Return [%]']
        hold_return = hold_stats['Total Return [%]']
        outperformance = qmac_return - hold_return
        
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"QMAC Return: {qmac_return:.2f}%")
        print(f"Buy & Hold Return: {hold_return:.2f}%")
        print(f"QMAC Outperformance: {outperformance:.2f}%")
        print(f"QMAC Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
        print(f"QMAC Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
        
        # Generate and save strategy plots
        strategy_figures = plot_qmac_strategy(results)
        
        # Save all plots
        timeframe_str = timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
        plot_dir = f'plots/{timeframe_str}_{symbol}_qmac_backtest_no_opt'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        save_plots(strategy_figures, symbol, start_date, end_date, output_dir=plot_dir)
        
    else:
        # Calculate total possible combinations
        from lib.utils import calculate_total_possible_combinations
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
            return
        
        # Analyze multiple window combinations
        print("\nAnalyzing window combinations...")
        window_results = analyze_window_combinations(
            symbol, start_date, end_date, min_window=min_window, max_window=max_window, 
            window_step=window_step, timeframe=timeframe, use_ray=use_ray, num_cpus=num_cpus,
            max_combinations=max_combinations, total_possible=total_combinations,
            save_to_db=True, top_n=TOP_N_PARAMS)
        
        # Check if we have optimal results directly
        if 'optimal_results' in window_results:
            optimal_results = window_results['optimal_results']
        else:
            # Run strategy with optimal parameters
            buy_fast, buy_slow, sell_fast, sell_slow = window_results['optimal_windows']
            optimal_perf = window_results['optimal_performance']
            
            print(f"\nOptimal window combination:")
            print(f"Buy signals: Fast MA = {buy_fast}, Slow MA = {buy_slow}")
            print(f"Sell signals: Fast MA = {sell_fast}, Slow MA = {sell_slow}")
            print(f"Optimal performance (total_return): {optimal_perf:.2%}")
            
            print(f"\nRunning QMAC strategy with optimal parameters")
            optimal_results = run_qmac_strategy(
                symbol, start_date, end_date,
                buy_fast_window=buy_fast,
                buy_slow_window=buy_slow,
                sell_fast_window=sell_fast,
                sell_slow_window=sell_slow,
                timeframe=timeframe,
                verbose=False
            )
        
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
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main() 