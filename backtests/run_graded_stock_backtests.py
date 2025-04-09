import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime
import logging


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

# Import necessary functions from cmcsa_4ma_cross_strategy
try:
    from outdated.four_ma_cross_strategy import (
        fetch_stock_data,
        roll_in_and_out_samples,
        simulate_all_params, # Finds best params per in-sample split
        simulate_best_params # Tests those params on out-of-sample splits
    )
except ImportError as e:
    logging.error(f"Error importing from four_ma_cross_strategy.py: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Define parameter ranges (adjust as needed)
# Keep these consistent with cmcsa_4ma_cross_strategy if desired
BUY_FAST_WINDOWS = np.arange(5, 55, 5) # Example: 5, 10, ..., 50
BUY_SLOW_WINDOWS = np.arange(10, 110, 10) # Example: 10, 20, ..., 100
SELL_FAST_WINDOWS = np.arange(5, 55, 5)
SELL_SLOW_WINDOWS = np.arange(10, 110, 10)

START_DATE = datetime(2018, 1, 1)
END_DATE = datetime.now()
TIMEFRAME = "1d"
DB_COLUMN_NAME = "best_params_4ma_daily" # Name for the DB column

# --- Walk-Forward Settings ---
# Define standard split settings (can be adjusted)
SPLIT_KWARGS = dict(
    n=10,                # Number of splits
    window_len=365,      # ~1 year in-sample window size (adjust based on data freq)
    set_lens=(100,),     # ~100 periods out-of-sample test size
    left_to_right=False  # Use for expanding window walk-forward
)

# --- Portfolio Settings ---
# Define standard portfolio settings
PF_KWARGS = dict(
    direction='longonly',
    fees=0.001,
    slippage=0.001,
    freq=TIMEFRAME,
    init_cash=10000,
)

# --- Backtesting Logic REMOVED ---
# The find_best_params_for_stock function is removed as we now use
# the walk-forward logic imported from cmcsa_4ma_cross_strategy.

# --- Main Execution Logic ---

def main():
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

    results_for_db = {} # Store final results: {symbol: "best_avg_oos_params_str"}

    # 3. Loop through stocks, run walk-forward optimization, find best avg OOS params
    total_stocks = len(stocks_to_backtest)
    # --- Modification: Process only the first stock --- 
    if not stocks_to_backtest:
        logging.warning("Stock list is empty, cannot process.")
        return
    stocks_to_process = [stocks_to_backtest[0]] # Get only the first stock
    logging.info(f"MODIFIED: Processing only the first stock found: {stocks_to_process[0]}")
    # --- End Modification ---
    
    # for i, symbol in enumerate(stocks_to_backtest):
    for i, symbol in enumerate(stocks_to_process): # Use the modified list
        # Use 0 for index since we only have one item, but keep total_stocks for context if needed
        logging.info(f"--- Processing symbol 1/1: {symbol} ---") 
        try:
            # Fetch data
            price_data = fetch_stock_data(symbol, TIMEFRAME, START_DATE, END_DATE)
            if price_data is None or price_data.empty:
                logging.warning(f"Could not fetch price data for {symbol}. Skipping.")
                continue
            if len(price_data) < SPLIT_KWARGS['window_len'] + SPLIT_KWARGS['set_lens'][0] * SPLIT_KWARGS['n']:
                logging.warning(f"Not enough data for {symbol} to perform {SPLIT_KWARGS['n']} splits with window {SPLIT_KWARGS['window_len']} and test {SPLIT_KWARGS['set_lens'][0]}. Skipping.")
                continue

            # Split data
            logging.info(f"Splitting data for {symbol}...")
            (in_price, _), (out_price, _) = roll_in_and_out_samples(price_data, **SPLIT_KWARGS)
            logging.info(f"Data split complete for {symbol}. In-sample shape: {in_price.shape}, Out-sample shape: {out_price.shape}")

            # Find best parameters for each *in-sample* split
            logging.info(f"Running in-sample parameter optimization for {symbol}...")
            # Note: simulate_all_params logs its own progress per split
            best_params_per_split_df = simulate_all_params(
                in_price,
                BUY_FAST_WINDOWS, BUY_SLOW_WINDOWS,
                SELL_FAST_WINDOWS, SELL_SLOW_WINDOWS,
                **PF_KWARGS
            )

            if best_params_per_split_df.empty:
                logging.warning(f"No best parameters found during in-sample optimization for {symbol}. Skipping.")
                continue
            logging.info(f"In-sample optimization complete for {symbol}. Found best params for {len(best_params_per_split_df)} splits.")
            # logging.debug(f"Best params per split for {symbol}:\n{best_params_per_split_df}") # Optional detailed log

            # Test the best parameters from each split on the corresponding *out-of-sample* data
            logging.info(f"Testing best parameters on out-of-sample data for {symbol}...")
            out_sample_results_df = simulate_best_params(out_price, best_params_per_split_df, **PF_KWARGS)

            if out_sample_results_df.empty:
                logging.warning(f"Out-of-sample simulation failed or produced no results for {symbol}. Skipping.")
                continue
            logging.info(f"Out-of-sample testing complete for {symbol}.")
            # logging.debug(f"Out-of-sample results for {symbol}:\n{out_sample_results_df}") # Optional detailed log

            # **** Analyze OOS results to find the best AVERAGE performing parameter set ****
            logging.info(f"Analyzing out-of-sample results for {symbol} to find best average parameters...")
            # Group by the parameters used in each OOS test and calculate the mean return
            param_cols = ['buy_fast', 'buy_slow', 'sell_fast', 'sell_slow']
            average_oos_returns = out_sample_results_df.groupby(param_cols)['return'].mean().reset_index()
            
            # Find the parameter set with the maximum average return
            best_avg_params_row = average_oos_returns.loc[average_oos_returns['return'].idxmax()]
            
            best_avg_params = (
                int(best_avg_params_row['buy_fast']),
                int(best_avg_params_row['buy_slow']),
                int(best_avg_params_row['sell_fast']),
                int(best_avg_params_row['sell_slow'])
            )
            best_avg_return = best_avg_params_row['return']

            logging.info(f"Best average OOS parameters for {symbol}: {best_avg_params} (Avg Return: {best_avg_return:.4f})")

            # Format parameters for DB storage
            params_str = ",".join(map(str, best_avg_params))
            results_for_db[symbol] = params_str

        except Exception as e:
            logging.error(f"An error occurred processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue # Move to the next symbol

    # 4. Update Database with the single best average OOS parameter set for each stock
    logging.info("Updating database with best average OOS parameters...")
    update_count = 0
    failed_updates = []
    for symbol, params_str in results_for_db.items():
        try:
            success = update_stock_data(symbol, {DB_COLUMN_NAME: params_str})
            if success:
                logging.info(f"Successfully updated {symbol} with best avg OOS params: {params_str}")
                update_count += 1
            else:
                failed_updates.append(symbol)
        except Exception as e:
            logging.error(f"Failed to update database for {symbol}: {e}")
            failed_updates.append(symbol)

    logging.info(f"Database update process completed.")
    logging.info(f"Successfully updated {update_count} stock(s).")
    if failed_updates:
        logging.warning(f"Failed to update {len(failed_updates)} stock(s): {failed_updates}")

    logging.info("Graded stock walk-forward optimization process finished.")

if __name__ == "__main__":
    main() 