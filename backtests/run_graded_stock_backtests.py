#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graded Stock Walk-Forward Backtest Script

This script runs walk-forward optimization on stocks based on their grades,
using the DMAC strategy. It properly separates in-sample and out-of-sample 
testing to prevent overfitting, and stores the optimal parameters in a database.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import common utilities
from backtests.common import (
    fetch_market_data,
    apply_splits,
    get_split_config,
    SplitMethod,
    create_custom_split_config,
    save_figures,
    ensure_directory
)

# Import DMAC strategy
from backtests.dmac_strategy.dmac_strategy import (
    run_dmac_strategy,
    analyze_window_combinations,
    plot_dmac_strategy,
    plot_heatmap
)

# Add market_analysis directory to Python path
# Adjust the path depth (../) as needed depending on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'market_analysis')))

# Import necessary modules from market_analysis
# Assuming these modules and functions exist based on the request
try:
    from db_utils import update_stock_data, get_stocks_by_grade
    from economic_quadrant import determine_economic_quadrant
    # We don't need stock_grading directly here as db_utils provides the function
except ImportError as e:
    logging.error(f"Error importing market_analysis modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Parameter ranges for optimization
BUY_FAST_WINDOWS = np.arange(5, 55, 5)    # 5, 10, ..., 50
BUY_SLOW_WINDOWS = np.arange(10, 110, 10) # 10, 20, ..., 100

START_DATE = datetime(2018, 1, 1)
END_DATE = datetime.now()
TIMEFRAME = "1d"
DB_COLUMN_NAME = "best_params_dmac_daily"  # Name for the DB column

# Output directories
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Portfolio settings
PORTFOLIO_SETTINGS = {
    'direction': 'longonly',
    'fees': 0.001,
    'slippage': 0.001,
    'freq': TIMEFRAME,
    'init_cash': 10000,
}

# --- Data Splitting Configuration ---
# Use our improved data splitting configuration
WFO_CONFIG = create_custom_split_config(
    method=SplitMethod.WALK_FORWARD,
    n_splits=10,                    # Number of train/test splits
    is_window_len=365,              # Days in the in-sample window
    oos_window_len=100,             # Days in the out-of-sample window
    expand_is_window=False,         # Keep fixed IS window size
    overlap_windows=False,          # Non-overlapping windows
    min_is_samples=252,             # Minimum samples needed for IS window
    min_oos_samples=63,             # Minimum samples needed for OOS window
    enforce_minimum_samples=True    # Skip split if doesn't meet minimums
)

def run_optimization_for_stock(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = TIMEFRAME,
    split_config: Dict[str, Any] = None,
    fast_windows: np.ndarray = BUY_FAST_WINDOWS,
    slow_windows: np.ndarray = BUY_SLOW_WINDOWS,
    portfolio_settings: Dict[str, Any] = None,
    save_results: bool = True,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run walk-forward optimization for a single stock.
    
    Args:
        symbol: Stock symbol
        start_date: Backtest start date
        end_date: Backtest end date
        timeframe: Data timeframe
        split_config: Data splitting configuration
        fast_windows: Fast MA window sizes to test
        slow_windows: Slow MA window sizes to test
        portfolio_settings: Portfolio settings
        save_results: Whether to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with optimization results, or None if optimization failed
    """
    if split_config is None:
        split_config = WFO_CONFIG
        
    if portfolio_settings is None:
        portfolio_settings = PORTFOLIO_SETTINGS
    
    try:
        # 1. Fetch market data
        if verbose:
            logging.info(f"Fetching market data for {symbol}...")
        
        data_result = fetch_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            include_buffer=True,
            verbose=verbose
        )
        
        if not data_result or 'without_buffer' not in data_result:
            logging.warning(f"Could not fetch price data for {symbol}. Skipping.")
            return None
            
        price_data = data_result['without_buffer']
        
        # 2. Apply data splitting
        if verbose:
            logging.info(f"Splitting data for {symbol}...")
            
        min_required_samples = split_config['is_window_len'] + split_config['oos_window_len'] * split_config['n_splits']
        if len(price_data) < min_required_samples:
            logging.warning(
                f"Not enough data for {symbol} to perform {split_config['n_splits']} splits. "
                f"Need at least {min_required_samples} samples, but got {len(price_data)}. Skipping."
            )
            return None
        
        in_sample_data, out_sample_data = apply_splits(price_data, split_config)
        
        if verbose:
            logging.info(f"Data split complete for {symbol}. "
                        f"In-sample shape: {in_sample_data.shape}, "
                        f"Out-sample shape: {out_sample_data.shape}")
        
        # 3. Run in-sample optimization for each split
        if verbose:
            logging.info(f"Running in-sample parameter optimization for {symbol}...")
            
        all_in_sample_results = []
        
        for split_idx in range(split_config['n_splits']):
            # Get data for current split
            split_data = in_sample_data[split_idx]
            
            if verbose:
                logging.info(f"Processing split {split_idx+1}/{split_config['n_splits']}")
                
            # Test all parameter combinations
            window_results = []
            
            for fast_window in fast_windows:
                for slow_window in slow_windows:
                    # Skip invalid combinations
                    if fast_window >= slow_window:
                        continue
                        
                    # Run DMAC strategy with current parameters
                    try:
                        result = run_dmac_strategy(
                            symbol=symbol,
                            start_date=split_data.index[0],
                            end_date=split_data.index[-1],
                            fast_window=int(fast_window),
                            slow_window=int(slow_window),
                            timeframe=timeframe,
                            verbose=False  # Avoid too much output
                        )
                        
                        # Get performance metrics
                        dmac_pf = result['dmac_pf']
                        metrics = {
                            'total_return': dmac_pf.total_return(),
                            'sharpe_ratio': dmac_pf.sharpe_ratio(),
                            'sortino_ratio': dmac_pf.sortino_ratio(),
                            'max_drawdown': dmac_pf.max_drawdown(),
                            'win_rate': dmac_pf.trades.win_rate() if len(dmac_pf.trades) > 0 else 0,
                            'profit_factor': dmac_pf.trades.profit_factor() if len(dmac_pf.trades) > 0 else 0
                        }
                        
                        window_results.append({
                            'split': split_idx,
                            'fast_window': int(fast_window),
                            'slow_window': int(slow_window),
                            **metrics
                        })
                        
                    except Exception as e:
                        logging.warning(f"Error running DMAC with windows {fast_window}/{slow_window}: {e}")
            
            # Find best parameters for this split
            if window_results:
                # Convert to DataFrame for easier analysis
                split_df = pd.DataFrame(window_results)
                
                # Find best parameters based on Sharpe ratio
                best_idx = split_df['sharpe_ratio'].idxmax()
                best_params = split_df.iloc[best_idx].to_dict()
                
                all_in_sample_results.append(best_params)
                
                if verbose:
                    logging.info(f"Best in-sample parameters for split {split_idx+1}: "
                                f"Fast={best_params['fast_window']}, "
                                f"Slow={best_params['slow_window']} "
                                f"(Sharpe: {best_params['sharpe_ratio']:.2f})")
        
        # 4. Test best in-sample parameters on out-of-sample data
        if verbose:
            logging.info(f"Testing best parameters on out-of-sample data for {symbol}...")
            
        all_oos_results = []
        
        for split_idx, best_params in enumerate(all_in_sample_results):
            # Get out-of-sample data for current split
            split_data = out_sample_data[split_idx]
            
            # Run DMAC strategy with best parameters
            try:
                result = run_dmac_strategy(
                    symbol=symbol,
                    start_date=split_data.index[0],
                    end_date=split_data.index[-1],
                    fast_window=int(best_params['fast_window']),
                    slow_window=int(best_params['slow_window']),
                    timeframe=timeframe,
                    verbose=False
                )
                
                # Get performance metrics
                dmac_pf = result['dmac_pf']
                metrics = {
                    'total_return': dmac_pf.total_return(),
                    'sharpe_ratio': dmac_pf.sharpe_ratio(),
                    'sortino_ratio': dmac_pf.sortino_ratio(),
                    'max_drawdown': dmac_pf.max_drawdown(),
                    'win_rate': dmac_pf.trades.win_rate() if len(dmac_pf.trades) > 0 else 0,
                    'profit_factor': dmac_pf.trades.profit_factor() if len(dmac_pf.trades) > 0 else 0
                }
                
                oos_result = {
                    'split': split_idx,
                    'fast_window': int(best_params['fast_window']),
                    'slow_window': int(best_params['slow_window']),
                    **metrics
                }
                
                all_oos_results.append(oos_result)
                
                if verbose:
                    logging.info(f"Out-of-sample performance for split {split_idx+1}: "
                                f"Fast={oos_result['fast_window']}, "
                                f"Slow={oos_result['slow_window']} "
                                f"(Return: {oos_result['total_return']:.2%}, "
                                f"Sharpe: {oos_result['sharpe_ratio']:.2f})")
                
            except Exception as e:
                logging.warning(f"Error testing out-of-sample for split {split_idx+1}: {e}")
        
        # 5. Analyze OOS results to find the best average parameter set
        if verbose:
            logging.info(f"Analyzing out-of-sample results to find best average parameters...")
            
        if not all_oos_results:
            logging.warning(f"No out-of-sample results available for {symbol}. Skipping.")
            return None
            
        # Convert to DataFrame for easier analysis
        oos_df = pd.DataFrame(all_oos_results)
        
        # Group by parameter combination and calculate average metrics
        param_cols = ['fast_window', 'slow_window']
        avg_metrics = oos_df.groupby(param_cols)[
            ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        ].mean().reset_index()
        
        # Find best parameter set based on average Sharpe ratio
        best_idx = avg_metrics['sharpe_ratio'].idxmax()
        best_avg_params = avg_metrics.iloc[best_idx].to_dict()
        
        if verbose:
            logging.info(f"Best average out-of-sample parameters for {symbol}: "
                        f"Fast={int(best_avg_params['fast_window'])}, "
                        f"Slow={int(best_avg_params['slow_window'])} "
                        f"(Avg. Sharpe: {best_avg_params['sharpe_ratio']:.2f}, "
                        f"Avg. Return: {best_avg_params['total_return']:.2%})")
        
        # 6. Save results if requested
        if save_results:
            # Ensure directories exist
            ensure_directory(RESULTS_DIR)
            
            # Save best parameters
            params_df = pd.DataFrame([best_avg_params])
            params_df.to_csv(f"{RESULTS_DIR}/{symbol}_best_params.csv", index=False)
            
            # Save all in-sample results
            in_sample_df = pd.DataFrame(all_in_sample_results)
            in_sample_df.to_csv(f"{RESULTS_DIR}/{symbol}_in_sample_results.csv", index=False)
            
            # Save all out-of-sample results
            oos_df.to_csv(f"{RESULTS_DIR}/{symbol}_out_of_sample_results.csv", index=False)
            
            if verbose:
                logging.info(f"Results saved to {RESULTS_DIR} directory")
        
        # Return the final results
        return {
            'symbol': symbol,
            'best_params': {
                'fast_window': int(best_avg_params['fast_window']),
                'slow_window': int(best_avg_params['slow_window'])
            },
            'avg_metrics': {k: v for k, v in best_avg_params.items() if k not in param_cols},
            'in_sample_results': all_in_sample_results,
            'out_of_sample_results': all_oos_results
        }
        
    except Exception as e:
        logging.error(f"An error occurred processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_db_with_results(results: List[Dict[str, Any]], column_name: str = DB_COLUMN_NAME) -> Tuple[int, List[str]]:
    """
    Update the database with optimization results.
    
    Args:
        results: List of optimization results by symbol
        column_name: Name of database column to update
        
    Returns:
        Tuple of (count of successful updates, list of failed symbols)
    """
    update_count = 0
    failed_updates = []
    
    for result in results:
        if not result:
            continue
            
        symbol = result['symbol']
        params = result['best_params']
        
        # Format parameters for DB storage
        params_str = f"{params['fast_window']},{params['slow_window']}"
        
        try:
            success = update_stock_data(symbol, {column_name: params_str})
            if success:
                logging.info(f"Successfully updated {symbol} with best params: {params_str}")
                update_count += 1
            else:
                logging.warning(f"Failed to update database for {symbol}")
                failed_updates.append(symbol)
        except Exception as e:
            logging.error(f"Error updating database for {symbol}: {e}")
            failed_updates.append(symbol)
    
    return update_count, failed_updates


def main():
    """Main execution function."""
    logging.info("Starting graded stock walk-forward optimization process...")

    # 1. Get Current Economic Quadrant
    try:
        quadrant_info = determine_economic_quadrant()
        if not quadrant_info:
             logging.error("Failed to determine current economic quadrant.")
             return
        current_quadrant = quadrant_info[0]
        logging.info(f"Current economic quadrant: {current_quadrant}")
    except Exception as e:
        logging.error(f"Error determining economic quadrant: {e}")
        return

    # 2. Get Target Stocks (Using 'A' grade for now)
    target_grade = 'A'
    try:
        stocks_df = get_stocks_by_grade(target_grade)
        if stocks_df is None or stocks_df.empty:
            logging.warning(f"No stocks found with grade '{target_grade}'. Exiting.")
            return
        stocks_to_backtest = stocks_df['symbol'].unique().tolist()
        logging.info(f"Found {len(stocks_to_backtest)} unique stock(s) with grade '{target_grade}' to backtest: {stocks_to_backtest}")
    except Exception as e:
        logging.error(f"Error fetching stocks by grade: {e}")
        return

    results_list = []

    # 3. Process stocks (use only the first one for testing)
    # In production, you would process all stocks
    stocks_to_process = [stocks_to_backtest[0]]  # Just process the first stock for now
    logging.info(f"Processing stock: {stocks_to_process[0]}")
    
    for symbol in stocks_to_process:
        logging.info(f"Running optimization for {symbol}...")
        
        result = run_optimization_for_stock(
            symbol=symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            timeframe=TIMEFRAME,
            split_config=WFO_CONFIG,
            verbose=True
        )
        
        if result:
            results_list.append(result)
            
            # Plot the final results with best parameters
            try:
                # Fetch fresh data for the entire period
                data_result = fetch_market_data(
                    symbol=symbol,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    timeframe=TIMEFRAME
                )
                
                price_data = data_result['without_buffer'] if isinstance(data_result, dict) else data_result
                
                # Run strategy with best parameters
                best_params = result['best_params']
                strategy_result = run_dmac_strategy(
                    symbol=symbol,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    fast_window=best_params['fast_window'],
                    slow_window=best_params['slow_window'],
                    timeframe=TIMEFRAME
                )
                
                # Create plots
                ensure_directory(PLOTS_DIR)
                
                # Strategy plot
                figures = plot_dmac_strategy(strategy_result)
                
                # Save all figures
                save_figures(
                    figures=figures,
                    output_dir=PLOTS_DIR,
                    prefix=f"{symbol}_dmac",
                    formats=["png", "html"]
                )
                
                logging.info(f"Plots saved to {PLOTS_DIR} directory")
                
            except Exception as e:
                logging.error(f"Error creating plots for {symbol}: {e}")

    # 4. Update Database with results
    if results_list:
        logging.info("Updating database with optimization results...")
        update_count, failed_updates = update_db_with_results(results_list)
        
        logging.info(f"Database update process completed.")
        logging.info(f"Successfully updated {update_count} stock(s).")
        if failed_updates:
            logging.warning(f"Failed to update {len(failed_updates)} stock(s): {failed_updates}")
    else:
        logging.warning("No results to update in database.")

    logging.info("Graded stock walk-forward optimization process finished.")


if __name__ == "__main__":
    main() 