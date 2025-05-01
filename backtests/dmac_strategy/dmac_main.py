#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dual Moving Average Crossover (DMAC) strategy main entry point.
This module provides command-line interface for running the DMAC strategy.
"""

import os
import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import time
import pytz
from datetime import datetime
from dateutil.parser import parse
import pandas as pd

# Import configuration
from backtests.dmac_strategy.config import (
    DEFAULT_SYMBOL, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME,
    DEFAULT_MIN_WINDOW, DEFAULT_MAX_WINDOW, DEFAULT_WINDOW_STEP,
    MAX_COMBINATIONS, TOP_N_PARAMS, DEFAULT_PLOTS_DIR, DEFAULT_LOGS_DIR, 
    DEFAULT_EXPORTS_DIR, INTERACTIVE_PLOTS, SAVE_PNG_PLOTS
)

# Import from strategy modules
from backtests.dmac_strategy.dmac_strategy import run_dmac_strategy, analyze_window_combinations
from backtests.dmac_strategy.lib.visualization import plot_dmac_strategy, plot_heatmap, create_parameter_space_visualization, save_plots
from backtests.dmac_strategy.lib.database import initialize_database

# Define constants for output directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS_DIR = os.path.join(BASE_DIR, DEFAULT_LOGS_DIR)
PLOTS_DIR = os.path.join(BASE_DIR, DEFAULT_PLOTS_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, DEFAULT_EXPORTS_DIR)

def ensure_directory(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def setup_logging(symbol, timeframe, start_date, end_date):
    """Set up logging to file and console."""
    import logging
    
    ensure_directory(LOGS_DIR)
    
    # Create a timestamped log directory
    symbol_safe = symbol.replace('/', '_')
    date_range = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    log_dir = os.path.join(LOGS_DIR, f"dmac_{timeframe}_backtest", symbol_safe, date_range)
    ensure_directory(log_dir)
    
    # Set up log file
    log_file = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized for {symbol} backtest from {start_date} to {end_date}")
    return log_dir

def save_results(results, symbol, timeframe, start_date, end_date, fast_window, slow_window):
    """Save backtest results to a structured directory."""
    import logging
    
    # Create results directory structure
    symbol_safe = symbol.replace('/', '_')
    date_range = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    window_combo = f"fast{fast_window}_slow{slow_window}"
    
    results_dir = os.path.join(RESULTS_DIR, f"dmac_{timeframe}_backtest_results", symbol_safe, date_range)
    ensure_directory(results_dir)
    
    # Save key performance metrics
    dmac_stats = results['dmac_pf'].stats()
    hold_stats = results['hold_pf'].stats()
    
    with open(os.path.join(results_dir, f"performance_{window_combo}.txt"), 'w') as f:
        f.write(f"DMAC Strategy Performance Report\n")
        f.write(f"===============================\n\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Fast Window: {fast_window}\n")
        f.write(f"Slow Window: {slow_window}\n\n")
        
        f.write("DMAC Strategy Performance:\n")
        f.write(str(dmac_stats) + "\n\n")
        
        f.write("Hold Strategy Performance:\n")
        f.write(str(hold_stats) + "\n\n")
        
        f.write(f"Outperformance vs Hold: {(dmac_stats['Total Return [%]'] - hold_stats['Total Return [%]']):.2f}%\n")
    
    logging.info(f"Saved performance results to {results_dir}")
    return results_dir

def main():
    """Main function to run DMAC strategy with command-line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Dual Moving Average Crossover Strategy Backtest')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, 
                        help='Trading symbol (e.g., SPY)')
    
    # Optional arguments with defaults
    parser.add_argument('--start', type=str, default=DEFAULT_START_DATE, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=DEFAULT_END_DATE, 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--fast-window', type=int, default=30, 
                        help='Fast MA window size')
    parser.add_argument('--slow-window', type=int, default=80, 
                        help='Slow MA window size')
    parser.add_argument('--min-window', type=int, default=DEFAULT_MIN_WINDOW, 
                        help='Minimum window size for optimization')
    parser.add_argument('--max-window', type=int, default=DEFAULT_MAX_WINDOW, 
                        help='Maximum window size for optimization')
    parser.add_argument('--window-step', type=int, default=DEFAULT_WINDOW_STEP,
                        help='Step size between window values (higher = faster but less granular)')
    parser.add_argument('--max-combinations', type=int, default=MAX_COMBINATIONS,
                        help='Maximum window combinations to test (use -1 for all combinations)')
    parser.add_argument('--metric', type=str, default='total_return', 
                        help='Performance metric to optimize for')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, 
                        help='Timeframe for data (e.g., 1d, 1h, 15m)')
    parser.add_argument('--cash', type=float, default=100.0, 
                        help='Initial cash amount')
    parser.add_argument('--fees', type=float, default=0.0025, 
                        help='Fee percentage (e.g., 0.0025 for 0.25%%)')
    parser.add_argument('--slippage', type=float, default=0.0025, 
                        help='Slippage percentage')
    parser.add_argument('--optimize', action='store_true', 
                        help='Run window optimization')
    parser.add_argument('--no-opt', action='store_true',
                        help='Skip optimization and run with provided parameters')
    parser.add_argument('--save-to-db', action='store_true', default=True,
                        help='Save results to database')
    parser.add_argument('--calculate-only', action='store_true',
                        help='Only calculate possible combinations without running backtest')
    parser.add_argument('--save-plots', action='store_true', default=True,
                        help='Save plots to files')
    parser.add_argument('--output-dir', type=str, default=PLOTS_DIR, 
                        help='Base directory to save plots')
    parser.add_argument('--from-config', action='store_true',
                        help='Run using settings from config.py only')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed information')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                        help='Minimize output')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast mode with limited window range for quick testing')
    
    args = parser.parse_args()
    
    # If from_config flag is set, run the full backtest using config settings
    if args.from_config:
        return run_from_config()
    
    # Set up parameters from arguments
    symbol = args.symbol
    
    # Parse dates and ensure they are timezone-aware
    start_date = parse(args.start).replace(tzinfo=pytz.UTC)
    end_date = parse(args.end).replace(tzinfo=pytz.UTC)
    
    # Fast mode - use limited window range for quick testing
    if args.fast:
        print("Running in FAST MODE with limited window range")
        args.min_window = 5
        args.max_window = 40
        args.window_step = 5
    
    # Adjust window sizes based on timeframe for daily data
    if args.timeframe == '1d':
        if args.max_window == DEFAULT_MAX_WINDOW:  # If default was used
            args.max_window = 252  # Approximately 1 trading year
    
    # Set up logging
    log_dir = setup_logging(symbol, args.timeframe, start_date, end_date)
    
    # Start timer for total execution
    total_start_time = time.time()
    
    print(f"\nDMAC Strategy Parameters:")
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Timeframe: {args.timeframe}")
    
    # Initialize database if saving to DB
    if args.save_to_db:
        print("Initializing database...")
        initialize_database()
    
    if args.no_opt:
        # Run with provided fast and slow window parameters
        print(f"\nRunning DMAC strategy with provided parameters:")
        print(f"Fast MA = {args.fast_window}, Slow MA = {args.slow_window}")
        
        # Run a single strategy with specified windows
        single_results = run_dmac_strategy(
            symbol, 
            start_date, 
            end_date, 
            args.fast_window, 
            args.slow_window,
            init_cash=args.cash,
            fees=args.fees,
            slippage=args.slippage,
            timeframe=args.timeframe,
            verbose=args.verbose
        )
        
        # Print strategy stats
        dmac_stats = single_results['dmac_pf'].stats()
        print("\nDMAC Strategy Performance:")
        print(dmac_stats)
        
        # Print hold strategy stats for comparison
        hold_stats = single_results['hold_pf'].stats()
        print("\nHold Strategy Performance:")
        print(hold_stats)
        
        # Print performance comparison
        dmac_return = dmac_stats['Total Return [%]']
        hold_return = hold_stats['Total Return [%]']
        outperformance = dmac_return - hold_return
        
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"DMAC Return: {dmac_return:.2f}%")
        print(f"Buy & Hold Return: {hold_return:.2f}%")
        print(f"DMAC Outperformance: {outperformance:.2f}%")
        print(f"DMAC Max Drawdown: {dmac_stats['Max Drawdown [%]']:.2f}%")
        print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
        print(f"DMAC Sharpe Ratio: {dmac_stats['Sharpe Ratio']:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
        
        # Save results to organized directory
        save_results(
            single_results,
            symbol,
            args.timeframe,
            start_date,
            end_date,
            args.fast_window,
            args.slow_window
        )
        
        # Plot results
        figures = plot_dmac_strategy(single_results)
        
        # Save plots if requested
        if args.save_plots:
            timeframe_str = args.timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
            plot_dir = os.path.join(BASE_DIR, args.output_dir, f"{timeframe_str}_{symbol}_dmac_backtest_no_opt")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            save_plots(figures, symbol, start_date, end_date, output_dir=plot_dir, timeframe=args.timeframe)
            print(f"Saved plots to: {plot_dir}")
    
    else:
        # Calculate total possible combinations
        def calculate_total_combinations(min_window, max_window, window_step):
            return ((max_window - min_window) // window_step + 1) ** 2
        
        total_combinations = calculate_total_combinations(
            args.min_window, args.max_window, args.window_step)
        print(f"Window Range: {args.min_window} to {args.max_window} with step {args.window_step}")
        print(f"Total Possible Window Combinations: {total_combinations:,}")
        
        # If max_combinations is -1 (unlimited), set it to the total possible
        max_combinations = args.max_combinations
        if max_combinations < 0:
            max_combinations = total_combinations
            print(f"Running with ALL {total_combinations:,} possible window combinations")
        else:
            if max_combinations > total_combinations:
                max_combinations = total_combinations
                print(f"Adjusted to maximum possible {total_combinations:,} combinations")
            else:
                print(f"Testing {max_combinations:,} of {total_combinations:,} possible combinations")
        
        # If calculate-only flag is set, exit after displaying combination counts
        if args.calculate_only:
            print("\nCalculation complete. Exiting without running backtest.")
            print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
            return
        
        # Run window optimization if requested
        print("\nAnalyzing window combinations...")
        
        window_results = analyze_window_combinations(
            symbol, 
            start_date, 
            end_date, 
            min_window=args.min_window, 
            max_window=args.max_window,
            window_step=args.window_step,
            metric=args.metric,
            timeframe=args.timeframe,
            verbose=args.verbose
        )
        
        # Get best window combination
        best_windows = window_results.get('optimal_windows', (0, 0))
        best_fast, best_slow = best_windows
        best_performance = window_results.get('optimal_performance', 0)
        
        print(f"\nOptimal window combination:")
        print(f"Fast MA = {best_fast}, Slow MA = {best_slow}")
        print(f"Optimal performance ({args.metric}): {best_performance:.4f}")
        
        # Check if we have optimal results directly
        if 'optimal_results' in window_results:
            optimal_results = window_results['optimal_results']
        else:
            # Run strategy with optimal parameters
            print(f"\nRunning DMAC strategy with optimal parameters")
            optimal_results = run_dmac_strategy(
                symbol, 
                start_date, 
                end_date, 
                best_fast, 
                best_slow,
                init_cash=args.cash,
                fees=args.fees,
                slippage=args.slippage,
                timeframe=args.timeframe,
                verbose=args.verbose
            )
        
        # Print strategy stats
        dmac_stats = optimal_results['dmac_pf'].stats()
        print("\nOptimal DMAC Strategy Performance:")
        print(dmac_stats)
        
        # Print hold strategy stats for comparison
        hold_stats = optimal_results['hold_pf'].stats()
        print("\nHold Strategy Performance:")
        print(hold_stats)
        
        # Print performance comparison
        dmac_return = dmac_stats['Total Return [%]']
        hold_return = hold_stats['Total Return [%]']
        outperformance = dmac_return - hold_return
        
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"Optimal DMAC Return: {dmac_return:.2f}%")
        print(f"Buy & Hold Return: {hold_return:.2f}%")
        print(f"DMAC Outperformance: {outperformance:.2f}%")
        print(f"DMAC Max Drawdown: {dmac_stats['Max Drawdown [%]']:.2f}%")
        print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
        print(f"DMAC Sharpe Ratio: {dmac_stats['Sharpe Ratio']:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
        
        # Save results for optimal parameters
        save_results(
            optimal_results,
            symbol,
            args.timeframe,
            start_date,
            end_date,
            best_fast,
            best_slow
        )
        
        # Generate and save strategy plots
        strategy_figures = plot_dmac_strategy(optimal_results)
        
        # Plot and save heatmap if requested
        heatmap = plot_heatmap(window_results['dmac_perf_matrix'], args.metric)
        optimization_figures = {'heatmap': heatmap}
        
        # Create parameter space visualizations if available
        if 'performance_df' in window_results:
            param_space_figures = create_parameter_space_visualization(
                window_results['performance_df'], 
                symbol, 
                start_date, 
                end_date
            )
            # Add parameter space figures to optimization figures
            optimization_figures.update(param_space_figures)
        
        # Combine all figures
        all_figures = {**strategy_figures, **optimization_figures}
        
        # Save all plots if requested
        if args.save_plots:
            timeframe_str = args.timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
            plot_dir = os.path.join(BASE_DIR, args.output_dir, f"{timeframe_str}_{symbol}_dmac_backtest")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            save_plots(all_figures, symbol, start_date, end_date, output_dir=plot_dir, timeframe=args.timeframe)
            print(f"Saved plots to: {plot_dir}")
        
        # Save optimization results
        symbol_safe = symbol.replace('/', '_')
        date_range = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
        opt_results_dir = os.path.join(
            RESULTS_DIR, 
            f"dmac_{args.timeframe}_optimization_results",
            symbol_safe,
            date_range
        )
        ensure_directory(opt_results_dir)
        
        # Extract top combinations if available
        top_combinations = []
        if 'top_combinations' in window_results:
            top_combinations = window_results['top_combinations']
        elif isinstance(window_results.get('dmac_perf'), pd.Series):
            # If dmac_perf is a Series, extract top N values
            top_n = min(10, len(window_results['dmac_perf']))
            top_indices = window_results['dmac_perf'].nlargest(top_n).index
            top_combinations = [(idx, window_results['dmac_perf'][idx]) for idx in top_indices]
        
        # Save optimal parameters
        with open(os.path.join(opt_results_dir, f"optimization_results_{args.metric}.txt"), 'w') as f:
            f.write(f"DMAC Optimization Results\n")
            f.write(f"========================\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframe: {args.timeframe}\n")
            f.write(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Optimization Metric: {args.metric}\n\n")
            f.write(f"Best Fast Window: {best_fast}\n")
            f.write(f"Best Slow Window: {best_slow}\n")
            f.write(f"Best {args.metric}: {best_performance:.4f}\n\n")
            
            f.write("Top Window Combinations:\n")
            for i, combo in enumerate(top_combinations[:10]):
                if isinstance(combo, tuple) and len(combo) == 2:
                    (fast, slow), perf = combo
                    f.write(f"{i+1}. Fast: {fast}, Slow: {slow} - {args.metric}: {perf:.4f}\n")
        
        print(f"Saved optimization results to {opt_results_dir}")
    
    print("\nDone!")
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")

def run_from_config():
    """
    Run the full DMAC strategy backtest using config values.
    """
    # Start timing the execution
    start_time = time.time()
    
    # Parse dates from config and ensure they're timezone-aware
    start_date = parse(DEFAULT_START_DATE).replace(tzinfo=pytz.UTC)
    end_date = parse(DEFAULT_END_DATE).replace(tzinfo=pytz.UTC)

    # Print header with current config settings
    print("\n" + "="*80)
    print(f"DMAC STRATEGY BACKTEST")
    print("="*80)
    print(f"Symbol: {DEFAULT_SYMBOL}")
    print(f"Timeframe: {DEFAULT_TIMEFRAME}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Window Range: {DEFAULT_MIN_WINDOW} to {DEFAULT_MAX_WINDOW} (step: {DEFAULT_WINDOW_STEP})")
    print(f"Max Combinations: {MAX_COMBINATIONS}")
    print("="*80 + "\n")
    
    # Initialize database if it doesn't exist
    print("Initializing database...")
    initialize_database()
    
    # Run the optimization to find the best parameters
    print("\nRunning window optimization...")
    optimizer_results = analyze_window_combinations(
        DEFAULT_SYMBOL,
        start_date,
        end_date,
        min_window=DEFAULT_MIN_WINDOW,
        max_window=DEFAULT_MAX_WINDOW,
        window_step=DEFAULT_WINDOW_STEP,
        timeframe=DEFAULT_TIMEFRAME,
        max_combinations=MAX_COMBINATIONS,
        save_to_db=True,
        top_n=TOP_N_PARAMS,
        verbose=True
    )
    
    # Extract optimal windows and performance
    optimal_fast, optimal_slow = optimizer_results['optimal_windows']
    optimal_perf = optimizer_results['optimal_performance']
    optimal_results = optimizer_results['optimal_results']
    
    # Print optimization results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Optimal Window Combination: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
    print(f"Optimal Performance: {optimal_perf:.2f}%")
    
    # Print strategy performance
    dmac_stats = optimal_results['dmac_pf'].stats()
    hold_stats = optimal_results['hold_pf'].stats()
    
    # Calculate relative performance
    dmac_return = dmac_stats['Total Return [%]']
    hold_return = hold_stats['Total Return [%]']
    outperformance = dmac_return - hold_return
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"DMAC Strategy Return: {dmac_return:.2f}%")
    print(f"Buy & Hold Return: {hold_return:.2f}%")
    print(f"Outperformance: {outperformance:.2f}%")
    print(f"DMAC Sharpe Ratio: {dmac_stats['Sharpe Ratio']:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
    print(f"DMAC Max Drawdown: {dmac_stats['Max Drawdown [%]']:.2f}%")
    print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    
    # Create strategy visualizations
    strategy_figures = plot_dmac_strategy(optimal_results)
    
    # Create parameter space visualizations
    param_figures = create_parameter_space_visualization(
        optimizer_results['performance_df'],
        DEFAULT_SYMBOL,
        start_date,
        end_date
    )
    
    # Combine all figures
    all_figures = {**strategy_figures, **param_figures}
    
    # Create output directory with timeframe and symbol info
    timeframe_str = DEFAULT_TIMEFRAME.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
    output_dir = os.path.join(DEFAULT_PLOTS_DIR, f"{timeframe_str}_{DEFAULT_SYMBOL}_dmac_backtest")
    
    # Save all plots
    save_plots(all_figures, DEFAULT_SYMBOL, start_date, end_date, output_dir=output_dir, timeframe=DEFAULT_TIMEFRAME)
    print(f"Visualizations saved to directory: {output_dir}")
    
    # Print execution summary
    execution_time = time.time() - start_time
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Optimal Parameters: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
    print(f"Performance: {optimal_perf:.2f}%")
    print(f"Results saved in database and visualizations directory")
    print("="*80)
    print("\nBacktest Complete!")
    
    return optimal_results, optimizer_results

if __name__ == "__main__":
    main() 