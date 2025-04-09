#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run DMAC strategy backtests with command line arguments.
This allows for easy execution of backtests with different parameters.
"""

import argparse
import pytz
from datetime import datetime
import os
import sys
from pathlib import Path

# Import from the local module directly since we're in the same directory
from dmac_strategy import (
    run_dmac_strategy, 
    analyze_window_combinations, 
    plot_dmac_strategy, 
    plot_heatmap
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DMAC strategy backtest')
    
    # Required arguments
    parser.add_argument('symbol', type=str, help='Trading symbol (e.g., BTC/USD)')
    
    # Optional arguments with defaults
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2020-01-01', 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--fast-window', type=int, default=30, 
                        help='Fast MA window size')
    parser.add_argument('--slow-window', type=int, default=80, 
                        help='Slow MA window size')
    parser.add_argument('--min-window', type=int, default=2, 
                        help='Minimum window size for optimization')
    parser.add_argument('--max-window', type=int, default=100, 
                        help='Maximum window size for optimization')
    parser.add_argument('--metric', type=str, default='total_return', 
                        help='Performance metric to optimize for')
    parser.add_argument('--timeframe', type=str, default='1d', 
                        help='Timeframe for data (e.g., 1d, 1h, 15m)')
    parser.add_argument('--cash', type=float, default=100.0, 
                        help='Initial cash amount')
    parser.add_argument('--fees', type=float, default=0.0025, 
                        help='Fee percentage (e.g., 0.0025 for 0.25%%)')
    parser.add_argument('--slippage', type=float, default=0.0025, 
                        help='Slippage percentage')
    parser.add_argument('--optimize', action='store_true', 
                        help='Run window optimization')
    parser.add_argument('--save-plots', action='store_true', 
                        help='Save plots to files')
    parser.add_argument('--output-dir', type=str, default='plots', 
                        help='Directory to save plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed information')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                        help='Minimize output')
    
    return parser.parse_args()

def ensure_directory(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """Main function to run the backtest."""
    args = parse_args()
    
    # Parse dates and ensure they are timezone-aware
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    
    print(f"Running DMAC strategy for {args.symbol} from {start_date} to {end_date}")
    print(f"Fast window: {args.fast_window}, Slow window: {args.slow_window}")
    
    # Run a single strategy with specified windows
    single_results = run_dmac_strategy(
        args.symbol, 
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
    print("\nDMAC Strategy Performance:")
    print(single_results['dmac_pf'].stats())
    
    # Print hold strategy stats for comparison
    print("\nHold Strategy Performance:")
    print(single_results['hold_pf'].stats())
    
    # Plot results
    figures = plot_dmac_strategy(single_results)
    
    # Save plots if requested
    if args.save_plots:
        ensure_directory(args.output_dir)
        symbol_safe = args.symbol.replace('/', '_')
        date_range = f"{args.start}_to_{args.end}"
        window_combo = f"fast{args.fast_window}_slow{args.slow_window}"
        
        for name, fig in figures.items():
            filename = os.path.join(
                args.output_dir, 
                f"{symbol_safe}_{name}_{date_range}_{window_combo}.png"
            )
            fig.write_image(filename)
            print(f"Saved {name} plot to {filename}")
    
    # Run window optimization if requested
    if args.optimize:
        print("\nAnalyzing window combinations...")
        
        window_results = analyze_window_combinations(
            args.symbol, 
            start_date, 
            end_date, 
            min_window=args.min_window, 
            max_window=args.max_window,
            metric=args.metric,
            timeframe=args.timeframe,
            single_result=single_results,  # Reuse data from single strategy
            verbose=args.verbose
        )
        
        # Plot and save heatmap if requested
        print("\nGenerating heatmap...")
        heatmap = plot_heatmap(window_results['dmac_perf_matrix'], args.metric)
        
        if args.save_plots:
            filename = os.path.join(
                args.output_dir, 
                f"{symbol_safe}_heatmap_{date_range}.png"
            )
            print(f"Saving heatmap to {filename}...")
            try:
                heatmap.write_image(filename)
                print(f"Successfully saved heatmap to {filename}")
            except Exception as e:
                print(f"Error saving heatmap: {str(e)}")
                print("Attempting to save with heatmap.fig.write_image instead...")
                try:
                    heatmap.fig.write_image(filename)
                    print(f"Successfully saved heatmap with fig.write_image to {filename}")
                except Exception as e2:
                    print(f"Error with second attempt: {str(e2)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 