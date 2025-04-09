import os
import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import time # Added for timing
from tqdm.auto import tqdm # Added for progress bar
import numba as nb  # Import Numba for JIT compilation
import matplotlib.pyplot as plt  # Import matplotlib.pyplot at the top level
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Load environment variables
load_dotenv()

end_date = datetime.now()
start_date = end_date - timedelta(minutes=5*480)
ma_warmup_period = 250

def fetch_stock_data(symbol, timeframe, start_date, end_date, ma_warmup_period):
    """
    Fetch stock data from Alpaca with additional history for MA warmup
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to fetch
    timeframe : str
        Timeframe ("1d", "5m", etc.)
    start_date : datetime
        Start date for analysis
    end_date : datetime, optional
        End date (defaults to now)
    ma_warmup_period : int, optional
        Number of additional periods to fetch before start_date for MA calculation warmup
        
    Returns:
    --------
    pd.Series
        Close prices
    """
    
    # Calculate the adjusted start date with warmup period
    if timeframe == "5m":
        # For 5m data, add warmup_period bars (in minutes)
        warmup_start_date = start_date - timedelta(minutes=5 * ma_warmup_period)
    elif timeframe == "1h":
        # For hourly data
        warmup_start_date = start_date - timedelta(hours=ma_warmup_period)
    else:
        # For daily data and other timeframes
        warmup_start_date = start_date - timedelta(days=ma_warmup_period)
    
    # Initialize Alpaca data client
    alpaca_api_key = os.environ.get("alpaca_paper_key")
    alpaca_api_secret = os.environ.get("alpaca_paper_secret")
    
    # Create the StockHistoricalDataClient
    data_client = StockHistoricalDataClient(api_key=alpaca_api_key, secret_key=alpaca_api_secret)
    
    # Map the timeframe string to an Alpaca TimeFrame object
    if timeframe == "5m":
        alpaca_timeframe = TimeFrame(amount=5, unit=TimeFrameUnit.Minute)
    elif timeframe == "1h":
        alpaca_timeframe = TimeFrame(amount=1, unit=TimeFrameUnit.Hour)
    elif timeframe == "1d":
        alpaca_timeframe = TimeFrame.Day
    else:
        # Default to daily if timeframe not recognized
        print(f"Timeframe {timeframe} not recognized, defaulting to daily.")
        alpaca_timeframe = TimeFrame.Day
    
    # Create the request
    bars_request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_timeframe,
        start=warmup_start_date,
        end=end_date
    )
    
    try:
        # Get the bars
        bars_data = data_client.get_stock_bars(bars_request)
        
        # Convert to DataFrame
        if bars_data and hasattr(bars_data, 'df'):
            # Extract the dataframe from bars_data
            bars_df = bars_data.df
            
            # If we have multiple symbols (shouldn't happen), keep only our symbol
            if isinstance(bars_df.index, pd.MultiIndex):
                bars_df = bars_df.loc[symbol]
            
            # Get close prices as Series
            close = bars_df['close']
            
            print(f"Fetched {len(close)} periods of data with {ma_warmup_period} periods of MA warmup")
            return close
        else:
            raise ValueError(f"Empty or invalid response from Alpaca for {symbol}")
            
    except Exception as e:
        print(f"Error fetching data from Alpaca with {timeframe} timeframe: {e}")
        
        # Try with a different timeframe as fallback
        fallback_timeframe = None
        if timeframe == "5m":
            print("Falling back to 1h timeframe")
            fallback_timeframe = TimeFrame(amount=1, unit=TimeFrameUnit.Hour)
        elif timeframe == "1h":
            print("Falling back to daily timeframe")
            fallback_timeframe = TimeFrame.Day
        else:
            print("No appropriate fallback timeframe found, exiting")
            return None
        
        try:
            # Create a new request with the fallback timeframe
            fallback_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=fallback_timeframe,
                start=warmup_start_date,
                end=end_date
            )
            
            # Get the bars with fallback timeframe
            fallback_data = data_client.get_stock_bars(fallback_request)
            
            if fallback_data and hasattr(fallback_data, 'df'):
                # Extract the dataframe
                fallback_df = fallback_data.df
                
                # If we have multiple symbols, keep only our symbol
                if isinstance(fallback_df.index, pd.MultiIndex):
                    fallback_df = fallback_df.loc[symbol]
                
                # Get close prices as Series
                close = fallback_df['close']
                
                # Resample to the original timeframe if needed
                if timeframe == "5m" and fallback_timeframe == TimeFrame(amount=1, unit=TimeFrameUnit.Hour):
                    # Resample hourly data to 5-minute intervals (forward-fill)
                    resampled_df = fallback_df.resample('5min').ffill()
                    close = resampled_df['close']
                
                print(f"Fetched {len(close)} periods of data with fallback timeframe")
                return close
            else:
                print(f"Empty or invalid response from Alpaca with fallback timeframe")
                return None
                
        except Exception as fallback_error:
            print(f"Error fetching data with fallback timeframe: {fallback_error}")
            return None

def roll_in_and_out_samples(price, **kwargs):
    """Split price data into in-sample and out-of-sample periods"""
    return price.vbt.rolling_split(**kwargs)

def simulate_holding(price, **kwargs):
    """Simulate a simple buy and hold strategy"""
    pf = vbt.Portfolio.from_holding(price, **kwargs)
    return pf.total_return()

# Numba-optimized functions for MA calculation and crossing
@nb.njit
def calculate_ma(price_array, window):
    """
    Numba-optimized function to calculate moving average.
    """
    n = len(price_array)
    ma = np.empty(n)
    ma[:window] = np.nan  # Fill initial values with NaN
    
    # Calculate MA values
    for i in range(window, n):
        ma[i] = np.mean(price_array[i-window:i])
    
    return ma

@nb.njit
def ma_cross_above(fast_ma, slow_ma):
    """
    Numba-optimized function to detect when fast MA crosses above slow MA.
    """
    n = len(fast_ma)
    cross_above = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1])):
            # Fast was below or equal to slow in previous bar and now above
            cross_above[i] = (fast_ma[i-1] <= slow_ma[i-1]) and (fast_ma[i] > slow_ma[i])
    
    return cross_above

@nb.njit
def ma_cross_below(fast_ma, slow_ma):
    """
    Numba-optimized function to detect when fast MA crosses below slow MA.
    """
    n = len(fast_ma)
    cross_below = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n):
        if (not np.isnan(fast_ma[i]) and not np.isnan(slow_ma[i]) and
            not np.isnan(fast_ma[i-1]) and not np.isnan(slow_ma[i-1])):
            # Fast was above or equal to slow in previous bar and now below
            cross_below[i] = (fast_ma[i-1] >= slow_ma[i-1]) and (fast_ma[i] < slow_ma[i])
    
    return cross_below

@nb.njit
def calculate_returns(price_array, entries, exits):
    """
    Numba-optimized function to calculate returns based on entry/exit signals.
    This is a simplified return calculation and doesn't account for all the
    portfolio parameters that vectorbt handles.
    """
    n = len(price_array)
    position = False
    entry_price = 0.0
    total_return = 1.0
    
    for i in range(1, n):
        if not position and entries[i]:
            # Enter position
            position = True
            entry_price = price_array[i]
        elif position and exits[i]:
            # Exit position
            position = False
            exit_price = price_array[i]
            trade_return = exit_price / entry_price
            total_return *= trade_return
    
    # Close any remaining open position
    if position:
        exit_price = price_array[-1]
        trade_return = exit_price / entry_price
        total_return *= trade_return
    
    return total_return - 1.0  # Convert to percentage return

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
        
        # Get the maximum window size from parameters
        max_window_sizes = [p[1] for p in param_grid] + [p[3] for p in param_grid]  # Check slow windows
        max_window = max(max_window_sizes) if max_window_sizes else 0
        
        # Validate we have enough data
        if len(split_price) <= max_window:
            print(f"Warning: Split {i+1} has insufficient data points ({len(split_price)}) for max window size ({max_window})")
            # We'll continue anyway, but results may be unreliable
        
        # Convert price to numpy array for Numba optimization
        price_array = split_price.to_numpy()

        # Test all parameter combinations for this split with Numba optimization
        inner_loop = tqdm(enumerate(param_grid), total=total_params, desc=f"Split {i+1}/{n_splits} Params", leave=False)
        for params_idx, params in inner_loop:
            buy_fast, buy_slow, sell_fast, sell_slow = params
            
            try:
                # Calculate MAs using Numba optimized function
                buy_fast_ma = calculate_ma(price_array, buy_fast)
                buy_slow_ma = calculate_ma(price_array, buy_slow)
                sell_fast_ma = calculate_ma(price_array, sell_fast)
                sell_slow_ma = calculate_ma(price_array, sell_slow)
                
                # Generate signals using Numba optimized functions
                entries = ma_cross_above(buy_fast_ma, buy_slow_ma)
                exits = ma_cross_below(sell_fast_ma, sell_slow_ma)
                
                # Remove signals from warmup period (NaN values in the MAs)
                # The MA functions already put NaNs in the initial window values
                
                # Two options:
                # 1. Use pure Numba calculation for quick screening (faster but less accurate)
                # 2. Use vectorbt for final portfolio calculation (accurate but slower)
                
                # For quick parameter screening, use Numba calculation
                quick_return = calculate_returns(price_array, entries, exits)
                
                # If the quick calculation looks promising, use vectorbt for accurate calculation
                if quick_return > best_split_return:
                    # Convert numpy arrays back to Series for vectorbt
                    entries_series = pd.Series(entries, index=split_price.index)
                    exits_series = pd.Series(exits, index=split_price.index)
                    
                    # Simulate portfolio - skip warmup period automatically since entries/exits have NaNs
                    pf = vbt.Portfolio.from_signals(split_price, entries_series, exits_series, **kwargs)
                    current_return = pf.total_return()
                    
                    # Update best for this split
                    if current_return > best_split_return:
                        best_split_return = current_return
                        best_split_params = params

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

def simulate_best_params(in_price, out_price, best_params_df, **kwargs):
    """
    Simulate strategy using the best parameters for each split on out-of-sample data.
    Calculates MAs using combined in-sample and out-of-sample data for continuity
    but only trades on out-of-sample data.
    """
    results = []
    portfolios = {}  # Store portfolio objects for visualization
    
    # Iterate through each split index in the best_params_df
    for _, row in best_params_df.iterrows():
        split_idx = int(row['split_idx'])
        
        # Get corresponding price data for this window
        try:
            in_price_slice = in_price.iloc[:, split_idx]
            out_price_slice = out_price.iloc[:, split_idx]
        except IndexError:
            print(f"Warning: Split index {split_idx} out of bounds for price data. Skipping.")
            continue
            
        # Extract parameters
        buy_fast = int(row['buy_fast'])
        buy_slow = int(row['buy_slow'])
        sell_fast = int(row['sell_fast'])
        sell_slow = int(row['sell_slow'])
        
        # Determine the maximum window size to ensure proper warmup
        max_window = max(buy_fast, buy_slow, sell_fast, sell_slow)
        
        try:
            # CRITICAL STEP 1: Concatenate in-sample and out-of-sample data
            # to ensure proper MA calculation with sufficient history
            combined_price = pd.concat([in_price_slice, out_price_slice])
            combined_price = combined_price[~combined_price.index.duplicated(keep='last')]
            combined_price = combined_price.sort_index()
            
            # Ensure we have enough data for the largest MA window
            if len(combined_price) < max_window * 2:
                print(f"Warning: Not enough data for Split {split_idx} - need at least {max_window*2} points for MA calculation, got {len(combined_price)}")
                # We'll continue anyway, but results may be affected
            
            # CRITICAL STEP 2: Calculate MAs on the COMBINED data
            buy_fast_ma = vbt.MA.run(combined_price, window=buy_fast)
            buy_slow_ma = vbt.MA.run(combined_price, window=buy_slow)
            sell_fast_ma = vbt.MA.run(combined_price, window=sell_fast)
            sell_slow_ma = vbt.MA.run(combined_price, window=sell_slow)
            
            # CRITICAL STEP 3: Generate entry/exit signals on the combined data
            entries = buy_fast_ma.ma_crossed_above(buy_slow_ma)
            exits = sell_fast_ma.ma_crossed_below(sell_slow_ma)
            
            # CRITICAL STEP 4: Get only the signals that correspond to out-of-sample period
            # by filtering the signals Series to match the out_price_slice index
            entries_out = entries.loc[entries.index.isin(out_price_slice.index)]
            exits_out = exits.loc[exits.index.isin(out_price_slice.index)]
            
            # Now run the portfolio backtest on just the out-of-sample data
            pf = vbt.Portfolio.from_signals(
                out_price_slice, 
                entries_out, 
                exits_out,
                **kwargs
            )
            
            # Store the portfolio object for later visualization
            portfolios[split_idx] = pf
            
            # Store the results
            results.append({
                'split_idx': split_idx,
                'return': pf.total_return(),
                'buy_fast': buy_fast,
                'buy_slow': buy_slow,
                'sell_fast': sell_fast,
                'sell_slow': sell_slow
            })
            
        except Exception as e:
            print(f"Error in split {split_idx} with params ({buy_fast},{buy_slow},{sell_fast},{sell_slow}): {str(e)}")
            # Add the error data to results with NaN return
            results.append({
                'split_idx': split_idx,
                'return': np.nan,
                'buy_fast': buy_fast, 
                'buy_slow': buy_slow,
                'sell_fast': sell_fast,
                'sell_slow': sell_slow
            })
    
    # Create dataframe and sort by split_idx for readability
    results_df = pd.DataFrame(results).sort_values('split_idx').reset_index(drop=True)
    
    return results_df, portfolios

def display_portfolio_stats(pf, name="Strategy"):
    """Display key portfolio statistics"""
    print(f"\n{name} Portfolio Statistics:")
    
    # Basic metrics that should be available in most vectorbt versions
    print(f"Total Return: {pf.total_return():.2%}")
    
    # Other metrics with try/except to handle different vectorbt versions
    try:
        # Try the metrics with various possible method names
        # Annualized return
        if hasattr(pf, 'annual_return'):
            if callable(pf.annual_return):
                print(f"Annualized Return: {pf.annual_return():.2%}")
            else:
                print(f"Annualized Return: {pf.annual_return:.2%}")
        elif hasattr(pf, 'annualized_return'):
            if callable(pf.annualized_return):
                print(f"Annualized Return: {pf.annualized_return():.2%}")
            else:
                print(f"Annualized Return: {pf.annualized_return:.2%}")
            
        # Sharpe ratio
        if hasattr(pf, 'sharpe_ratio'):
            if callable(pf.sharpe_ratio):
                print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
            else:
                print(f"Sharpe Ratio: {pf.sharpe_ratio:.2f}")
        elif hasattr(pf, 'get_sharpe_ratio'):
            print(f"Sharpe Ratio: {pf.get_sharpe_ratio():.2f}")
            
        # Max drawdown
        if hasattr(pf, 'max_drawdown'):
            if callable(pf.max_drawdown):
                print(f"Max Drawdown: {pf.max_drawdown():.2%}")
            else:
                print(f"Max Drawdown: {pf.max_drawdown:.2%}")
        elif hasattr(pf, 'get_max_drawdown'):
            print(f"Max Drawdown: {pf.get_max_drawdown():.2%}")
            
        # Trade count and metrics
        if hasattr(pf, 'positions'):
            if hasattr(pf.positions, 'count'):
                if callable(pf.positions.count):
                    print(f"Number of Trades: {pf.positions.count()}")
                else:
                    print(f"Number of Trades: {pf.positions.count}")
                    
            if hasattr(pf.positions, 'win_rate'):
                if callable(pf.positions.win_rate):
                    print(f"Win Rate: {pf.positions.win_rate():.2%}")
                else:
                    print(f"Win Rate: {pf.positions.win_rate:.2%}")
                    
            if hasattr(pf.positions, 'avg_duration'):
                if callable(pf.positions.avg_duration):
                    avg_duration = pf.positions.avg_duration()
                else:
                    avg_duration = pf.positions.avg_duration
                    
                if avg_duration is not None:
                    print(f"Average Trade Duration: {avg_duration}")
        
        # Only try these if they might exist
        if hasattr(pf, 'sortino_ratio'):
            if callable(pf.sortino_ratio):
                print(f"Sortino Ratio: {pf.sortino_ratio():.2f}")
            else:
                print(f"Sortino Ratio: {pf.sortino_ratio:.2f}")
        elif hasattr(pf, 'get_sortino_ratio'):
            print(f"Sortino Ratio: {pf.get_sortino_ratio():.2f}")
            
        if hasattr(pf, 'calmar_ratio'):
            if callable(pf.calmar_ratio):
                print(f"Calmar Ratio: {pf.calmar_ratio():.2f}")
            else:
                print(f"Calmar Ratio: {pf.calmar_ratio:.2f}")
        elif hasattr(pf, 'get_calmar_ratio'):
            print(f"Calmar Ratio: {pf.get_calmar_ratio():.2f}")
    
    except (AttributeError, ValueError, TypeError) as e:
        print(f"Some metrics couldn't be calculated: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Return basic dict of stats
    try:
        if hasattr(pf, 'stats') and callable(pf.stats):
            return pf.stats()
        else:
            # Create a basic stats dictionary if stats() isn't available
            return {
                'total_return': pf.total_return()
            }
    except Exception as e:
        print(f"Error generating stats: {str(e)}")
        return {'total_return': pf.total_return()}

def visualize_ma_calculation(price, split_index, buy_fast, buy_slow, sell_fast, sell_slow, title=None):
    """
    Visualize price and MAs to confirm calculation from first candle of split
    
    Parameters:
    -----------
    price : pd.Series
        Price data for a specific split
    split_index : int
        Index of the split for labeling
    buy_fast, buy_slow, sell_fast, sell_slow : int
        MA window parameters to plot
    title : str, optional
        Title for the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    print(f"Starting visualization for split {split_index}...")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Convert price to numpy array for Numba optimization
    print(f"Converting price data to numpy array (length: {len(price)})...")
    price_array = price.to_numpy()
    
    # Calculate MAs using Numba optimized function
    print(f"Calculating MAs with windows: buy_fast={buy_fast}, buy_slow={buy_slow}, sell_fast={sell_fast}, sell_slow={sell_slow}...")
    buy_fast_ma = calculate_ma(price_array, buy_fast)
    buy_slow_ma = calculate_ma(price_array, buy_slow)
    sell_fast_ma = calculate_ma(price_array, sell_fast)
    sell_slow_ma = calculate_ma(price_array, sell_slow)
    
    # Create pandas Series for the MAs
    print("Creating DataFrame with price and MAs...")
    df = pd.DataFrame({
        'price': price,
        f'buy_fast_ma_{buy_fast}': buy_fast_ma,
        f'buy_slow_ma_{buy_slow}': buy_slow_ma,
        f'sell_fast_ma_{sell_fast}': sell_fast_ma,
        f'sell_slow_ma_{sell_slow}': sell_slow_ma
    }, index=price.index)
    
    # Add price trace
    print("Adding price trace to plot...")
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price'], name="QQQ Close", line=dict(color='black', width=1)),
        secondary_y=False,
    )
    
    # Add MA traces with different colors
    print("Adding MA traces to plot...")
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f'buy_fast_ma_{buy_fast}'], name=f"Buy Fast MA ({buy_fast})", 
                  line=dict(color='green', width=1.5)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f'buy_slow_ma_{buy_slow}'], name=f"Buy Slow MA ({buy_slow})", 
                  line=dict(color='blue', width=1.5)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f'sell_fast_ma_{sell_fast}'], name=f"Sell Fast MA ({sell_fast})", 
                  line=dict(color='orange', width=1.5, dash='dash')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df[f'sell_slow_ma_{sell_slow}'], name=f"Sell Slow MA ({sell_slow})", 
                  line=dict(color='red', width=1.5, dash='dash')),
        secondary_y=False,
    )
    
    # Add crossover points
    print("Calculating crossover points...")
    buy_signals = ma_cross_above(buy_fast_ma, buy_slow_ma)
    sell_signals = ma_cross_below(sell_fast_ma, sell_slow_ma)
    
    # Filter out NaN values and get indices where signals are True
    print("Filtering signal indices...")
    buy_indices = [i for i in range(len(buy_signals)) if buy_signals[i] and not np.isnan(price_array[i])]
    sell_indices = [i for i in range(len(sell_signals)) if sell_signals[i] and not np.isnan(price_array[i])]
    
    print(f"Found {len(buy_indices)} buy signals and {len(sell_indices)} sell signals")
    
    # Add buy signals as markers
    if buy_indices:
        print("Adding buy signals to plot...")
        fig.add_trace(
            go.Scatter(
                x=[price.index[i] for i in buy_indices],
                y=[price_array[i] for i in buy_indices],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signal'
            )
        )
    
    # Add sell signals as markers
    if sell_indices:
        print("Adding sell signals to plot...")
        fig.add_trace(
            go.Scatter(
                x=[price.index[i] for i in sell_indices],
                y=[price_array[i] for i in sell_indices],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signal'
            )
        )
    
    # Set title
    if title is None:
        title = f"QQQ Split {split_index} with Moving Averages"
    print("Updating layout...")
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Show the figure
    print("Displaying figure...")
    fig.show()
    
    # Save the figure to HTML file
    print("Saving to HTML file...")
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ma_visualization_split_{split_index}.html")
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    
    return fig

def run_backtest():
    # Settings
    symbol = "QQQ"
    timeframe = "5m"  # Changed from "1d" to "5m"
    
    # Calculate required data periods
    # For 5m data: a full trading day is roughly 78 bars (6.5 hours × 12 bars/hour)
    # Larger warmup period to ensure all MAs have sufficient data
    ma_warmup_period = 100
    
    # Set analysis period with explicit dates
    # Using more recent dates to stay within Yahoo Finance's 60-day limit for 5m data
    end_date = datetime(2024, 4, 5)  # April 5, 2024
    days_to_analyze = 30  # Using 30 days to stay well within the 60-day limit
    start_date = end_date - timedelta(days=days_to_analyze)  # Approx March 6, 2024
    
    print(f"Analyzing {timeframe} data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} with {ma_warmup_period} periods of MA warmup")
    
    # Fetch historical data with warmup period
    price = fetch_stock_data(symbol, timeframe, start_date, end_date, ma_warmup_period)
    
    # Walk-forward optimization settings
    split_kwargs = dict(
        n=5,              # Reduced from 10 to 3 windows for faster testing
        window_len=500,  # Reduced window length for faster testing
        set_lens=(250,),  # Reserve ~1 day for test
        left_to_right=False  # More recent data has more importance
    )
    
    # Portfolio settings
    pf_kwargs = dict(
        direction='longonly',  # Long only trades
        fees=0.001,         # 0.1% trading fee
        slippage=0.001,     # 0.1% slippage
        freq=timeframe,     # Use the same frequency as the data
        init_cash=10000,    # Starting cash
    )
    
    # Parameter space to explore - REDUCED FOR TESTING
    buy_fast_windows = np.arange(2,150,25)     # Reduced parameter space
    buy_slow_windows = np.arange(150,250,50)    # Reduced parameter space
    sell_fast_windows = np.arange(2,150,25)    # Reduced parameter space
    sell_slow_windows = np.arange(150,250,50)   # Reduced parameter space
    
    # Find maximum window size to ensure sufficient data
    max_window = max(
        np.max(buy_fast_windows), 
        np.max(buy_slow_windows), 
        np.max(sell_fast_windows), 
        np.max(sell_slow_windows)
    )
    
    print(f"Maximum window size in parameter space: {max_window}")
    if ma_warmup_period < max_window * 2:
        print(f"WARNING: MA warmup period ({ma_warmup_period}) is less than twice the maximum window size ({max_window*2})")
        print("Consider increasing ma_warmup_period for more accurate initial signals")

    # Ensure alpaca-py is installed
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
    except ImportError:
        print("alpaca-py package is required for data fetching. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "alpaca-py"])

    # Ensure numba is installed
    try:
        import numba
    except ImportError:
        print("numba package is required for optimization. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "numba"])

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
    
    # Precompile Numba functions with a small dummy input
    # This prevents the first real run from being slow due to compilation
    if len(in_price) > 0:
        dummy = np.random.random(100)
        _ = calculate_ma(dummy, 10)  # Precompile the MA calculation
        ma_fast = calculate_ma(dummy, 5)
        ma_slow = calculate_ma(dummy, 10)
        _ = ma_cross_above(ma_fast, ma_slow)  # Precompile cross detection
        _ = ma_cross_below(ma_fast, ma_slow)
        _ = calculate_returns(dummy, np.zeros(100, dtype=np.bool_), np.zeros(100, dtype=np.bool_))
        print("Numba JIT compilation completed")
    
    # This function now directly returns the best parameters found for each split
    best_params_df = simulate_all_params(in_price, buy_fast_windows, buy_slow_windows, sell_fast_windows, sell_slow_windows, **pf_kwargs)
    
    # The print statement for tested combinations per split is removed as we don't store all results anymore
    
    # The call to get_best_params is removed as simulate_all_params now returns the best directly
    
    print("\nBest Parameters Found for Each In-Sample Window:")
    # Ensure the columns exist before printing, handle case where no results found
    if not best_params_df.empty:
        print(best_params_df[['split_idx', 'buy_fast', 'buy_slow', 'sell_fast', 'sell_slow', 'return']])
        
        # Visualize the MA calculations for each split to confirm they start from the first candle
        print("\nGenerating visualization for each split to verify MA calculations...")
        for i, row in best_params_df.iterrows():
            split_idx = int(row['split_idx'])
            visualize_ma_calculation(
                in_price.iloc[:, split_idx], 
                split_idx,
                int(row['buy_fast']), 
                int(row['buy_slow']), 
                int(row['sell_fast']), 
                int(row['sell_slow']),
                f"Split {split_idx} In-Sample: MA Calculation Verification - QQQ"
            )
    else:
        print("No best parameters found during in-sample optimization.")
        # Exit or handle this case as appropriate for your logic
        return # Example: exit if no params found

    # Test best parameters on out-of-sample data
    print("\nTesting best parameters on out-of-sample data...")
    out_sample_results, portfolios = simulate_best_params(in_price, out_price, best_params_df, **pf_kwargs)
    
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
    
    # Display portfolio statistics for the best performing window
    if portfolios:
            # Find best performing split
        best_split_idx = out_sample_results.loc[out_sample_results['return'].idxmax(), 'split_idx']
        if best_split_idx in portfolios:
            best_pf = portfolios[best_split_idx]
            print(f"\nDetailed statistics for best performing window (Split {best_split_idx}):")
            stats = display_portfolio_stats(best_pf, f"Split {best_split_idx}")

    
    return best_params_df, out_sample_results, portfolios

if __name__ == "__main__":
    run_backtest() 