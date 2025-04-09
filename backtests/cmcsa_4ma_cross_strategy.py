import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time # Added for timing
from tqdm.auto import tqdm # Added for progress bar

# Load environment variables
load_dotenv()

def fetch_stock_data(symbol, timeframe, start_date, end_date=None):
    """Fetch stock data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.now()
    
    # Use vectorbt's YFData to fetch data from Yahoo Finance
    data = vbt.YFData.download(
        symbol,
        start=start_date,
        end=end_date
    )
    
    # Get close prices
    close = data.get('Close')
    
    # If timeframe is not daily, we need to resample
    if timeframe != "1d":
        if timeframe == "4h":
            # For 4h timeframe we need to resample from daily to 4h
            # This is an approximation since Yahoo Finance doesn't provide intraday data easily
            # We'll just use daily data for the backtest
            pass  # No resampling, will use daily data as is
        else:
            # For other timeframes we could implement resampling here
            pass
    
    return close

def roll_in_and_out_samples(price, **kwargs):
    """Split price data into in-sample and out-of-sample periods"""
    return price.vbt.rolling_split(**kwargs)

def simulate_holding(price, **kwargs):
    """Simulate a simple buy and hold strategy"""
    pf = vbt.Portfolio.from_holding(price, **kwargs)
    return pf.total_return()

def simulate_all_params(price, buy_fast_windows, buy_slow_windows, sell_fast_windows, sell_slow_windows, **kwargs):
    """
    Simulate a 4 MA cross strategy, find best params for each split.
    Returns a DataFrame with the best parameters for each in-sample split.
    """
    # price is expected to be the output of vbt.rolling_split, having multiple columns (splits)
    n_splits = price.shape[1]
    print(f"Processing {n_splits} splits...")

    best_results_list = []

    # Generate all valid parameter combinations once
    param_grid = []
    for buy_fast in buy_fast_windows:
        for buy_slow in buy_slow_windows:
            if buy_fast >= buy_slow:
                continue
            for sell_fast in sell_fast_windows:
                for sell_slow in sell_slow_windows:
                    if sell_fast >= sell_slow:
                        continue
                    param_grid.append((int(buy_fast), int(buy_slow), int(sell_fast), int(sell_slow)))
    
    total_params = len(param_grid)
    print(f"Generated {total_params} valid parameter combinations to test per split.")

    # Iterate through each split (column in the price DataFrame)
    for i in tqdm(range(n_splits), desc="Processing Splits"):
        split_price = price.iloc[:, i]  # Get price data for the current split
        best_split_return = -np.inf
        best_split_params = None

        # Test all parameter combinations for this split
        inner_loop = tqdm(enumerate(param_grid), total=total_params, desc=f"Split {i+1}/{n_splits} Params", leave=False)
        for params_idx, params in inner_loop:
            buy_fast, buy_slow, sell_fast, sell_slow = params
            
            try:
                # Calculate MAs
                buy_fast_ma = vbt.MA.run(split_price, window=buy_fast, short_name=f'buy_fast_{buy_fast}')
                buy_slow_ma = vbt.MA.run(split_price, window=buy_slow, short_name=f'buy_slow_{buy_slow}')
                sell_fast_ma = vbt.MA.run(split_price, window=sell_fast, short_name=f'sell_fast_{sell_fast}')
                sell_slow_ma = vbt.MA.run(split_price, window=sell_slow, short_name=f'sell_slow_{sell_slow}')
                
                # Generate signals
                entries = buy_fast_ma.ma_crossed_above(buy_slow_ma)
                exits = sell_fast_ma.ma_crossed_below(sell_slow_ma)
                
                # Simulate portfolio
                pf = vbt.Portfolio.from_signals(split_price, entries, exits, **kwargs)
                current_return = pf.total_return()
                
                # Update best for this split
                if current_return > best_split_return:
                    best_split_return = current_return
                    best_split_params = params
                    
                # Clear memory
                del buy_fast_ma, buy_slow_ma, sell_fast_ma, sell_slow_ma, entries, exits, pf

            except Exception as e:
                print(f"  Error with params {params} on split {i}: {e}")
                # Optionally continue or break depending on how errors should be handled
                continue 
            
        # Store the best result for this split
        if best_split_params is not None:
            best_results_list.append({
                'split_idx': i,
                'buy_fast': best_split_params[0],
                'buy_slow': best_split_params[1],
                'sell_fast': best_split_params[2],
                'sell_slow': best_split_params[3],
                'return': best_split_return
            })
        else:
            print(f"Split {i+1} did not yield any valid results.")
            
        # Force garbage collection after each split processing
        import gc
        gc.collect()

    # Convert the list of best results into a DataFrame
    return pd.DataFrame(best_results_list)

def get_best_params(performance_df):
    """Get best parameters for each split based on total return"""
    # Find the index of the row with the maximum 'return' for each 'split_idx'
    idx = performance_df.groupby(['split_idx'])['return'].idxmax()
    # Select these rows from the original DataFrame
    best_params_df = performance_df.loc[idx]
    # Sort by split_idx just in case and reset index
    return best_params_df.sort_values('split_idx').reset_index(drop=True)

def simulate_best_params(price, best_params_df, **kwargs):
    """Simulate strategy using the best parameters for each split"""
    results = []
    
    # Iterate through each unique window in the out-of-sample data
    for split_idx in range(price.shape[1]):
        # Get corresponding price data for this window
        price_slice = price.iloc[:, split_idx]
        
        # Find the best parameters for this split index
        if split_idx < len(best_params_df):
            row = best_params_df.iloc[split_idx]
            buy_fast = int(row['buy_fast'])
            buy_slow = int(row['buy_slow'])
            sell_fast = int(row['sell_fast'])
            sell_slow = int(row['sell_slow'])
            
            # Calculate MAs and signals
            buy_fast_ma = vbt.MA.run(price_slice, window=buy_fast)
            buy_slow_ma = vbt.MA.run(price_slice, window=buy_slow)
            sell_fast_ma = vbt.MA.run(price_slice, window=sell_fast)
            sell_slow_ma = vbt.MA.run(price_slice, window=sell_slow)
            
            entries = buy_fast_ma.ma_crossed_above(buy_slow_ma)
            exits = sell_fast_ma.ma_crossed_below(sell_slow_ma)
            
            pf = vbt.Portfolio.from_signals(price_slice, entries, exits, **kwargs)
            
            results.append({
                'split_idx': split_idx,
                'return': pf.total_return(),
                'buy_fast': buy_fast,
                'buy_slow': buy_slow,
                'sell_fast': sell_fast,
                'sell_slow': sell_slow
            })
    
    return pd.DataFrame(results)

def run_backtest():
    # Settings
    symbol = "CMCSA"
    timeframe = "1d"  # Using daily data since Yahoo Finance doesn't easily provide 4h data
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    
    # Fetch historical data
    price = fetch_stock_data(symbol, timeframe, start_date, end_date)
    
    # Walk-forward optimization settings
    split_kwargs = dict(
        n=10,                # 10 windows instead of 20
        window_len=365,      # Each window is 1 year (trading days)
        set_lens=(100,),      # Reserve 100 days for test
        left_to_right=False  # More recent data has more importance
    )
    
    # Portfolio settings
    pf_kwargs = dict(
        direction='longonly',    # Long only trades
        fees=0.001,          # 0.1% trading fee
        slippage=0.001,      # 0.1% slippage
        freq=timeframe,      # Use the same frequency as the data
        init_cash=10000,     # Starting cash
    )
    
    # Parameter space to explore - reduced parameter space to save memory
    buy_fast_windows = np.arange(2, 252, 2)    # Buy signal MA windows
    buy_slow_windows = np.arange(2, 252, 2)    # Buy signal MA windows
    sell_fast_windows = np.arange(2, 252, 2)  # Sell signal MA windows
    sell_slow_windows = np.arange(2, 252, 2)  # Sell signal MA windows
    
    # Estimate total parameter combinations (Note: actual count depends on fast < slow constraints)
    potential_combinations = len(buy_fast_windows) * len(buy_slow_windows) * len(sell_fast_windows) * len(sell_slow_windows)
    print(f"Potential maximum parameter combinations before filtering: {potential_combinations}")
    
    # Split data into in-sample and out-of-sample
    (in_price, in_indexes), (out_price, out_indexes) = roll_in_and_out_samples(price, **split_kwargs)
    
    print(f"Shape of in-sample data: {in_price.shape}")
    print(f"Shape of out-of-sample data: {out_price.shape}")
    
    # Benchmark: Buy and hold
    in_hold_return = simulate_holding(in_price, **pf_kwargs)
    out_hold_return = simulate_holding(out_price, **pf_kwargs)
    
    print("\nBuy and Hold Returns:")
    print(f"In-sample: {in_hold_return.mean():.2%}")
    print(f"Out-of-sample: {out_hold_return.mean():.2%}")
    
    # Simulate all parameter combinations for in-sample and get best params per split
    print("\nRunning in-sample parameter optimization...")
    # This function now directly returns the best parameters found for each split
    best_params_df = simulate_all_params(in_price, buy_fast_windows, buy_slow_windows, sell_fast_windows, sell_slow_windows, **pf_kwargs)
    
    # The print statement for tested combinations per split is removed as we don't store all results anymore
    
    # The call to get_best_params is removed as simulate_all_params now returns the best directly
    
    print("\nBest Parameters Found for Each In-Sample Window:")
    # Ensure the columns exist before printing, handle case where no results found
    if not best_params_df.empty:
        print(best_params_df[['split_idx', 'buy_fast', 'buy_slow', 'sell_fast', 'sell_slow', 'return']])
    else:
        print("No best parameters found during in-sample optimization.")
        # Exit or handle this case as appropriate for your logic
        return # Example: exit if no params found

    # Test best parameters on out-of-sample data
    print("\nTesting best parameters on out-of-sample data...")
    out_sample_results = simulate_best_params(out_price, best_params_df, **pf_kwargs)
    
    # Analysis of results
    print("\nOut-of-Sample Performance with Best Parameters:")
    print(out_sample_results[['split_idx', 'buy_fast', 'buy_slow', 'sell_fast', 'sell_slow', 'return']])
    
    # Calculate average performance
    in_sample_avg_return = best_params_df['return'].mean()
    out_sample_avg_return = out_sample_results['return'].mean()
    
    print("\nSummary:")
    print(f"Average In-Sample Return: {in_sample_avg_return:.2%}")
    print(f"Average Out-of-Sample Return: {out_sample_avg_return:.2%}")
    print(f"Buy-and-Hold In-Sample Return: {in_hold_return.mean():.2%}")
    print(f"Buy-and-Hold Out-of-Sample Return: {out_hold_return.mean():.2%}")
    
    # Calculate win rate (% of windows where strategy beats buy and hold)
    in_hold_mean = in_hold_return.mean() if isinstance(in_hold_return, pd.Series) else in_hold_return
    out_hold_mean = out_hold_return.mean() if isinstance(out_hold_return, pd.Series) else out_hold_return
    
    in_sample_win_rate = (best_params_df['return'] > in_hold_mean).mean()
    out_sample_win_rate = (out_sample_results['return'] > out_hold_mean).mean()
    
    print(f"In-Sample Win Rate vs Buy-and-Hold: {in_sample_win_rate:.2%}")
    print(f"Out-of-Sample Win Rate vs Buy-and-Hold: {out_sample_win_rate:.2%}")
    
    # Rename variable used in visualization section for clarity
    best_params_for_vis = best_params_df 

    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        
        print("\nDebugging visualization...")
        print(f"Best params shape: {best_params_for_vis.shape}")
        print(f"Unique split_idx values: {best_params_for_vis['split_idx'].unique()}")
        
        # Visualization logic now uses best_params_for_vis which should have unique split_idx already
        # No need for drop_duplicates if simulate_all_params worked correctly
        best_params_unique = best_params_for_vis.copy() 
        
        # Ensure split_idx is numeric and sort
        best_params_unique['split_idx'] = pd.to_numeric(best_params_unique['split_idx'], errors='coerce')
        best_params_unique = best_params_unique.sort_values('split_idx')
        best_params_unique = best_params_unique.reset_index(drop=True)
        
        # Show the data we're plotting
        print("\nData for plotting (In-Sample Best Params):")
        print(best_params_unique[['split_idx', 'buy_fast', 'buy_slow', 'sell_fast', 'sell_slow', 'return']])
        
        # Plot of best parameters
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Use integer indexes for x-axis instead of split_idx values
        x_values = range(len(best_params_unique))
        
        axs[0, 0].plot(x_values, best_params_unique['buy_fast'], 'o-', label='Buy Fast MA')
        axs[0, 0].plot(x_values, best_params_unique['buy_slow'], 'o-', label='Buy Slow MA')
        axs[0, 0].set_title('Buy Signal Parameters')
        axs[0, 0].set_xlabel('Window Index')
        axs[0, 0].set_ylabel('MA Window Size')
        axs[0, 0].legend()
        
        axs[0, 1].plot(x_values, best_params_unique['sell_fast'], 'o-', label='Sell Fast MA')
        axs[0, 1].plot(x_values, best_params_unique['sell_slow'], 'o-', label='Sell Slow MA')
        axs[0, 1].set_title('Sell Signal Parameters')
        axs[0, 1].set_xlabel('Window Index')
        axs[0, 1].set_ylabel('MA Window Size')
        axs[0, 1].legend()
        
        # Plot returns
        axs[1, 0].plot(x_values, best_params_unique['return'], 'o-', label='In-Sample Return')
        axs[1, 0].axhline(y=in_hold_mean, color='r', linestyle='-', label='In-Sample Buy-Hold')
        axs[1, 0].set_title('In-Sample Returns')
        axs[1, 0].set_xlabel('Window Index')
        axs[1, 0].set_ylabel('Return')
        axs[1, 0].legend()
        
        # Use integer indexes for out-of-sample results as well
        out_sample_results = out_sample_results.sort_values('split_idx').reset_index(drop=True)
        axs[1, 1].plot(range(len(out_sample_results)), out_sample_results['return'], 'o-', label='Out-of-Sample Return')
        axs[1, 1].axhline(y=out_hold_mean, color='r', linestyle='-', label='Out-of-Sample Buy-Hold')
        axs[1, 1].set_title('Out-of-Sample Returns')
        axs[1, 1].set_xlabel('Window Index')
        axs[1, 1].set_ylabel('Return')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{symbol}_4ma_cross_results.png")
        print(f"\nResults visualization saved to {symbol}_4ma_cross_results.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return best_params_df, out_sample_results

if __name__ == "__main__":
    run_backtest() 