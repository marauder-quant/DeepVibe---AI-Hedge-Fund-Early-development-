#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced DMAC Strategy Backtest Runner with Database Support
This script provides a command-line interface for running DMAC backtest strategies
with database storage and enhanced visualizations.
"""

import argparse
import pytz
import time
import os
import sys
from datetime import datetime
from dateutil.parser import parse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from local modules
from dmac_strategy import run_dmac_strategy
from lib.optimizer import analyze_window_combinations
from lib.visualization import plot_dmac_strategy, create_parameter_space_visualization, save_plots
from lib.database import get_parameters_from_db, list_database_contents
from config import (
    DEFAULT_SYMBOL, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TIMEFRAME,
    DEFAULT_MIN_WINDOW, DEFAULT_MAX_WINDOW, DEFAULT_WINDOW_STEP,
    TOP_N_PARAMS, MAX_COMBINATIONS, DEFAULT_PLOTS_DIR
)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='DMAC Strategy Backtest Runner with Database Support')
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
    parser.add_argument('--fast-window', type=int, default=None,
                       help='Fast MA window size (skip optimization if provided)')
    parser.add_argument('--slow-window', type=int, default=None,
                       help='Slow MA window size (skip optimization if provided)')
    parser.add_argument('--max-combinations', type=int, default=MAX_COMBINATIONS,
                       help='Maximum number of window combinations to test')
    parser.add_argument('--metric', type=str, default='total_return',
                       help='Performance metric to optimize for')
    parser.add_argument('--db-only', action='store_true',
                       help='Load optimal parameters from database without running backtest')
    parser.add_argument('--db-query', action='store_true',
                       help='Query and display database contents')
    parser.add_argument('--no-save-db', action='store_true',
                       help='Do not save results to the database')
    parser.add_argument('--top-n', type=int, default=TOP_N_PARAMS,
                       help='Number of top parameter combinations to save to database')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output plots (default: plots/TIMEFRAME_SYMBOL)')
    
    args = parser.parse_args()
    
    # Start timing the execution
    start_time = time.time()
    
    # Parse dates and ensure they are timezone-aware
    start_date = parse(args.start).replace(tzinfo=pytz.UTC)
    end_date = parse(args.end).replace(tzinfo=pytz.UTC)
    
    # Print header
    print(f"\n{'='*80}")
    print(f"DMAC Strategy Backtest - {args.symbol}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Timeframe: {args.timeframe}")
    print(f"{'='*80}\n")
    
    # If db-query flag is set, just query and display database contents
    if args.db_query:
        print("Database Query Mode")
        print(f"{'='*80}")
        list_database_contents()
        return
    
    # If db-only flag is set, load optimal parameters from database
    if args.db_only:
        print("Database-Only Mode: Loading optimal parameters from database")
        df = get_parameters_from_db(symbol=args.symbol, timeframe=args.timeframe)
        
        if df.empty:
            print(f"No data found in database for {args.symbol} with timeframe {args.timeframe}")
            print("Run optimization first or use --no-db-only to run backtest")
            return
        
        # Get parameters from rank 1 (best parameters)
        top_params = df[df['rank'] == 1].iloc[0]
        fast_window = int(top_params['fast_window'])
        slow_window = int(top_params['slow_window'])
        
        print(f"Loaded optimal parameters from database:")
        print(f"  Fast MA window: {fast_window}")
        print(f"  Slow MA window: {slow_window}")
        print(f"  Performance: {top_params['performance']:.2%}")
        
        # Run strategy with the loaded optimal parameters
        print(f"\nRunning DMAC strategy with database parameters")
        strategy_results = run_dmac_strategy(
            args.symbol, 
            start_date, 
            end_date, 
            fast_window, 
            slow_window,
            timeframe=args.timeframe,
            verbose=True
        )
        
        # Print strategy performance statistics
        print("\nStrategy Performance:")
        stats = strategy_results['dmac_pf'].stats()
        print(stats)
        
        # Plot and save results
        figures = plot_dmac_strategy(strategy_results)
        
        # Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timeframe_str = args.timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
            output_dir = os.path.join(DEFAULT_PLOTS_DIR, f"{timeframe_str}_{args.symbol}_dmac_backtest_db")
        
        save_plots(figures, args.symbol, start_date, end_date, output_dir=output_dir)
        
    # If fast-window and slow-window are provided, skip optimization
    elif args.fast_window is not None and args.slow_window is not None:
        fast_window = args.fast_window
        slow_window = args.slow_window
        
        print(f"Running DMAC strategy with provided parameters:")
        print(f"  Fast MA window: {fast_window}")
        print(f"  Slow MA window: {slow_window}")
        
        # Run strategy with the provided parameters
        strategy_results = run_dmac_strategy(
            args.symbol, 
            start_date, 
            end_date, 
            fast_window, 
            slow_window,
            timeframe=args.timeframe,
            verbose=True
        )
        
        # Print strategy performance statistics
        print("\nStrategy Performance:")
        stats = strategy_results['dmac_pf'].stats()
        print(stats)
        
        # Print hold strategy stats for comparison
        print("\nBuy & Hold Performance:")
        hold_stats = strategy_results['hold_pf'].stats()
        print(hold_stats)
        
        # Plot and save results
        figures = plot_dmac_strategy(strategy_results)
        
        # Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timeframe_str = args.timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
            output_dir = os.path.join(DEFAULT_PLOTS_DIR, f"{timeframe_str}_{args.symbol}_dmac_backtest_single")
        
        save_plots(figures, args.symbol, start_date, end_date, output_dir=output_dir)
        
    # Otherwise, run full optimization
    else:
        print("Running optimization to find best parameters")
        print(f"Window range: {args.min_window} to {args.max_window} with step {args.window_step}")
        print(f"Testing up to {args.max_combinations} combinations")
        print(f"Optimizing for: {args.metric}")
        
        # Run optimization
        opt_results = analyze_window_combinations(
            args.symbol, 
            start_date, 
            end_date, 
            min_window=args.min_window, 
            max_window=args.max_window, 
            window_step=args.window_step,
            metric=args.metric,
            timeframe=args.timeframe,
            max_combinations=args.max_combinations,
            save_to_db=not args.no_save_db,
            top_n=args.top_n,
            verbose=True
        )
        
        # Print optimal parameters
        optimal_fast, optimal_slow = opt_results['optimal_windows']
        optimal_perf = opt_results['optimal_performance']
        
        print(f"\nOptimal window combination: Fast MA = {optimal_fast}, Slow MA = {optimal_slow}")
        print(f"Optimal performance ({args.metric}): {optimal_perf:.2%}")
        
        # Print strategy stats for the optimal parameters
        print("\nStrategy Performance with Optimal Parameters:")
        stats = opt_results['optimal_results']['dmac_pf'].stats()
        print(stats)
        
        # Print hold strategy stats for comparison
        print("\nBuy & Hold Performance:")
        hold_stats = opt_results['optimal_results']['hold_pf'].stats()
        print(hold_stats)
        
        # Create performance comparison summary
        dmac_return = stats['Total Return [%]']
        hold_return = hold_stats['Total Return [%]']
        outperformance = dmac_return - hold_return
        
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"DMAC Return: {dmac_return:.2f}%")
        print(f"Buy & Hold Return: {hold_return:.2f}%")
        print(f"DMAC Outperformance: {outperformance:.2f}%")
        print(f"DMAC Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        print(f"Buy & Hold Max Drawdown: {hold_stats['Max Drawdown [%]']:.2f}%")
        print(f"DMAC Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {hold_stats['Sharpe Ratio']:.2f}")
        
        # Plot and save strategy results
        strategy_figures = plot_dmac_strategy(opt_results['optimal_results'])
        
        # Create parameter space visualizations
        param_space_figures = create_parameter_space_visualization(
            opt_results['performance_df'], 
            args.symbol, 
            start_date, 
            end_date
        )
        
        # Combine all figures
        all_figures = {**strategy_figures, **param_space_figures}
        
        # Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            timeframe_str = args.timeframe.replace('m', 'min').replace('d', 'day').replace('h', 'hour')
            output_dir = os.path.join(DEFAULT_PLOTS_DIR, f"{timeframe_str}_{args.symbol}_dmac_backtest_opt")
        
        # Save all plots
        save_plots(all_figures, args.symbol, start_date, end_date, output_dir=output_dir)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("Done!")

if __name__ == "__main__":
    main() 