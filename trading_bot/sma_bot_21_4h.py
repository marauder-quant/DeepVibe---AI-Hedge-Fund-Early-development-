import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import sys
import importlib.util
from dotenv import load_dotenv
import config # Import the new config file
    
load_dotenv()

# Load API credentials from .env
ALPACA_API_KEY = os.environ.get('alpaca_paper_key')
ALPACA_SECRET_KEY = os.environ.get('alpaca_paper_secret')


# Initialize Alpaca client for trading
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Add market_analysis directory to path to import modules
# Get the project root directory (which contains trading_bot, market_analysis, etc.)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to sys.path
sys.path.insert(0, PROJECT_ROOT_DIR)

# Now we can import using absolute paths from the project root
from market_analysis.db_utils import get_latest_economic_quadrant, get_stocks_by_grade

# Trading parameters (now loaded from config)
# sma_period = 21  # 21-period SMA - Removed
# timeframe = "4h"  # 4-hour timeframe - Removed

# Set number of top stocks to trade per grade (now loaded from config)
# TOP_A_PLUS_STOCKS = 10 - Removed
# TOP_A_STOCKS = 50 - Removed
# TOP_B_STOCKS = 5 - Removed

def get_quadrant_based_allocation():
    """
    Get allocation based on current economic quadrant using config settings
    Returns dict with quadrant info and allocation percentages
    """
    quadrant_data = get_latest_economic_quadrant()
    
    if not quadrant_data:
        print(f"No economic quadrant data found. Using default quadrant: {config.DEFAULT_QUADRANT}")
        quadrant = config.DEFAULT_QUADRANT
        notes = 'Using default conservative allocation (no quadrant data)'
    else:
        quadrant = quadrant_data.get('quadrant', config.DEFAULT_QUADRANT)
        notes = quadrant_data.get('analysis_notes', 'No notes available')
        # Ensure the fetched quadrant exists in our config, otherwise use default
        if quadrant not in config.QUADRANT_ALLOCATIONS:
            print(f"Warning: Quadrant '{quadrant}' found in DB but not defined in config. Using default: {config.DEFAULT_QUADRANT}")
            quadrant = config.DEFAULT_QUADRANT
            notes = f"Database quadrant '{quadrant_data.get('quadrant')}' not in config. {notes}"
    
    # Get allocation from config based on the determined quadrant
    equity_allocation = config.QUADRANT_ALLOCATIONS.get(quadrant, config.QUADRANT_ALLOCATIONS[config.DEFAULT_QUADRANT])
    
    return {
        'quadrant': quadrant,
        'equity_allocation': equity_allocation,
        'notes': notes
    }

def get_stocks_by_current_quadrant():
    """
    Get stocks appropriate for the current economic quadrant based on config settings.
    Returns list of stock symbols to trade and the equity allocation.
    """
    quadrant_info = get_quadrant_based_allocation()
    quadrant = quadrant_info['quadrant']
    equity_allocation = quadrant_info['equity_allocation']
    print(f"Current economic quadrant: {quadrant}")
    print(f"Notes: {quadrant_info['notes']}")
    
    # Get the allowed grades for the current quadrant from config
    allowed_grades = config.QUADRANT_STOCK_GRADES.get(quadrant, config.QUADRANT_STOCK_GRADES[config.DEFAULT_QUADRANT])
    print(f"Allowed grades for quadrant {quadrant}: {', '.join(allowed_grades)}")
    
    selected_stocks = []
    
    # Fetch stocks for each allowed grade in the specified order
    for grade in allowed_grades:
        limit = config.GRADE_LIMIT_MAP.get(grade)
        if limit is None:
            print(f"Warning: Grade '{grade}' specified in QUADRANT_STOCK_GRADES for quadrant '{quadrant}' but has no limit defined in GRADE_LIMIT_MAP. Skipping grade.")
            continue
        if limit <= 0:
            continue # Skip grades with zero limit

        print(f"Fetching top {limit} stocks for grade {grade}...")
        stocks_df = get_stocks_by_grade(grade=grade, limit=limit)
        
        if not stocks_df.empty:
            new_symbols_added = 0
            for symbol in stocks_df['symbol'].unique():
                if symbol not in selected_stocks:
                    selected_stocks.append(symbol)
                    new_symbols_added += 1
            print(f"Added {new_symbols_added} new unique symbols from grade {grade}.")
        else:
             print(f"No stocks found for grade {grade}.")

    print(f"Selected {len(selected_stocks)} unique stocks based on quadrant '{quadrant}' and configured grades: {', '.join(allowed_grades)}")
    return selected_stocks, equity_allocation

def get_position_quantity(symbol):
    """Check if we have an existing position in the symbol"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == symbol:
                return int(position.qty)
        return 0
    except Exception as e:
        print(f"Error checking positions for {symbol}: {e}")
        return 0

def get_market_data(symbol):
    """Get market data using Yahoo Finance"""
    try:
        # Get data for the past 60 days with 4 hour intervals for 21-period SMA
        ticker = yf.Ticker(symbol)
        # Using 4-hour intervals for the 21-period SMA (from config)
        df = ticker.history(period="60d", interval=config.TIMEFRAME)
        
        if df.empty:
            print(f"No data returned from Yahoo Finance for {symbol}")
            return None, None, None
        
        # Calculate 21-period SMA (from config)
        sma = df['Close'].tail(config.SMA_PERIOD).mean()
        # Get current price
        current_price = df['Close'].iloc[-1]
        # Get latest bar timestamp
        latest_bar_time = df.index[-1]
        
        return current_price, sma, latest_bar_time
    except Exception as e:
        print(f"Error getting market data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def calculate_buy_quantity(symbol, current_price, equity_allocation):
    """Calculate number of shares to buy based on total equity allocation and trade size percentage."""
    try:
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        
        # Calculate the target value for this single trade based on total equity allocation and config percentage
        target_trade_value = portfolio_value * equity_allocation * config.TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION

        # Calculate quantity based on current price and target trade value
        if current_price > 0:
            quantity = int(target_trade_value / current_price) # Use floor division implicitly
        else:
            quantity = 0

        # Ensure we buy at least 1 share if the target trade value allows it,
        # otherwise set quantity to 0.
        if quantity <= 0:
            if target_trade_value >= current_price and current_price > 0:
                 print(f"Target trade value (${target_trade_value:.2f}) allows buying minimum 1 share of {symbol} at ${current_price:.2f}. Setting quantity to 1.")
                 quantity = 1
            else:
                 print(f"Target trade value (${target_trade_value:.2f}) not enough to buy 1 share of {symbol} at ${current_price:.2f} or price is zero. Setting quantity to 0.")
                 quantity = 0
        # No need for max(1, quantity) here as the above handles the minimum 1 share case.

        print(f"Calculated buy quantity for {symbol}: Portfolio=${portfolio_value:.2f}, EquityAlloc={equity_allocation:.2%}, TradeSizePerc={config.TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION:.2%}, TargetTradeValue=${target_trade_value:.2f}, Price=${current_price:.2f}, Quantity={quantity}")
        
        return quantity
    except Exception as e:
        print(f"Error calculating buy quantity for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # Return 0 on error
        return 0

def place_order(symbol, side, quantity):
    """Place a market order for a specific quantity."""
    # Don't place orders with zero quantity
    if quantity <= 0:
        print(f"Quantity is zero or less ({quantity}) for {symbol}. Skipping {side.name} order.")
        return None
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    
    try:
        order = trading_client.submit_order(order_data)
        print(f"{datetime.now()}: {side.name} order placed for {quantity} shares of {symbol}")
        return order
    except Exception as e:
        print(f"Error placing {side.name} order for {quantity} shares of {symbol}: {e}")
        return None

def run_strategy_for_symbol(symbol, last_bar_times, equity_allocation, force_check=False):
    """Run the SMA crossover strategy for a single symbol, calculating buy quantity on signal."""
    print(f"{datetime.now()}: Checking {symbol}...")
    
    # Check if we already have a position
    current_position = get_position_quantity(symbol)
    print(f"{symbol} current position: {current_position} shares")
    
    # Get current price and SMA using Yahoo Finance
    current_price, sma, latest_bar_time = get_market_data(symbol)
    
    if current_price is None or sma is None or latest_bar_time is None:
        print(f"Failed to get price data for {symbol}. Skipping.")
        return last_bar_times
    
    print(f"{symbol} current price: ${current_price:.2f}")
    print(f"{symbol} {config.SMA_PERIOD}-period SMA ({config.TIMEFRAME}): ${sma:.2f}") # Use config values
    print(f"Latest {config.TIMEFRAME} bar close time: {latest_bar_time}") # Use config value
    
    # Check if this is a new bar (or first run)
    is_new_bar = force_check or symbol not in last_bar_times or latest_bar_time > last_bar_times[symbol]
    
    # Store the latest bar time
    last_bar_times[symbol] = latest_bar_time
    
    # Only execute trading logic if we have a new bar
    if is_new_bar:
        print(f"New {config.TIMEFRAME} bar detected for {symbol} - executing trading logic") # Use config value
        # Execute trading logic
        if current_price > sma:
            # Price is above SMA - should be in long position
            if current_position <= 0:
                # Calculate buy quantity ONLY when a buy signal occurs for a stock we don't own
                buy_quantity = calculate_buy_quantity(symbol, current_price, equity_allocation)
                print(f"{symbol} price above SMA - Placing buy order for {buy_quantity} shares")
                place_order(symbol, OrderSide.BUY, buy_quantity)
            else:
                print(f"{symbol} already in long position - holding")
        else:
            # Price is below SMA - should be out of position
            if current_position > 0:
                print(f"{symbol} price below SMA - Selling entire position ({current_position} shares)")
                place_order(symbol, OrderSide.SELL, current_position)
            else:
                print(f"{symbol} already out of position - holding cash")
    else:
        print(f"No new {config.TIMEFRAME} bar for {symbol} - monitoring only") # Use config value
    
    print(f"Finished checking {symbol}")
    print("-" * 40)
    
    return last_bar_times

def run_strategy_for_all_symbols(last_bar_times=None, force_check=False):
    """Run the SMA strategy for all symbols based on market analysis"""
    # Get stocks based on current quadrant
    symbols, equity_allocation = get_stocks_by_current_quadrant()
    
    print(f"{datetime.now()}: Starting {config.SMA_PERIOD}-period SMA ({config.TIMEFRAME}) strategy") # Use config values
    print(f"Using {len(symbols)} stocks selected based on economic quadrant analysis")
    print(f"Equity allocation: {equity_allocation * 100:.1f}% of portfolio")
    print(f"Stocks: {', '.join(symbols)}")
    
    if last_bar_times is None:
        last_bar_times = {}
    
    # Iterate through symbols and run the strategy check
    for symbol in symbols:
        # Pass equity_allocation down; buy quantity is calculated inside run_strategy_for_symbol if needed
        last_bar_times = run_strategy_for_symbol(symbol, last_bar_times, equity_allocation, force_check)
    
    return last_bar_times

def calculate_next_4h_bar_time():
    """
    Calculate the time until the next 4-hour bar closes on Yahoo Finance
    4-hour bars typically close at 0:00, 4:00, 8:00, 12:00, 16:00, and 20:00 UTC
    Add 5 minutes buffer to ensure data is available
    """
    # Get current time in UTC
    now = datetime.now()
    current_hour = now.hour
    
    # Calculate the next 4-hour block
    next_4h_block = ((current_hour // 4) + 1) * 4
    
    if next_4h_block >= 24:
        # If we're in the last 4-hour block of the day, calculate time to midnight
        next_day = now.replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(days=1)
        seconds_to_wait = (next_day - now).total_seconds()
    else:
        # Calculate time to next 4-hour block plus 5 minutes buffer
        next_time = now.replace(hour=next_4h_block, minute=5, second=0, microsecond=0)
        seconds_to_wait = (next_time - now).total_seconds()
    
    # Format next bar time for display
    next_bar_time = now + timedelta(seconds=seconds_to_wait)
    
    return seconds_to_wait, next_bar_time

if __name__ == "__main__":
    # Check account status
    account = trading_client.get_account()
    print(f"Account status: {'ACTIVE' if account.status == 'ACTIVE' else 'INACTIVE'}")
    print(f"Buying power: ${float(account.buying_power):.2f}")
    
    # Get quadrant information
    quadrant_info = get_quadrant_based_allocation()
    print(f"Current economic quadrant: {quadrant_info['quadrant']}")
    print(f"Equity allocation: {quadrant_info['equity_allocation'] * 100:.1f}%")
    print(f"Notes: {quadrant_info['notes']}")
    
    print(f"Starting monitoring using {config.SMA_PERIOD}-period SMA on {config.TIMEFRAME} timeframe") # Use config values
    print(f"Bot will scan after each {config.TIMEFRAME} bar closes (with 5-minute buffer for data availability)") # Use config value
    print(f"Note: Trading will only occur when a new {config.TIMEFRAME} bar closes") # Use config value
    print("Press Ctrl+C to stop the bot")
    
    # Dictionary to store the latest bar time for each symbol
    last_bar_times = {}
    
    try:
        # Run strategy immediately on startup and force a check
        last_bar_times = run_strategy_for_all_symbols(force_check=True)
        
        while True:
            # Calculate time until next 4-hour bar closes (plus buffer)
            seconds_to_wait, next_check_time = calculate_next_4h_bar_time()
            
            hours = int(seconds_to_wait // 3600)
            minutes = int((seconds_to_wait % 3600) // 60)
            seconds = int(seconds_to_wait % 60)
            
            print(f"Next scan scheduled for: {next_check_time}")
            print(f"Waiting {hours}h {minutes}m {seconds}s until after next 4-hour bar closes...")
            time.sleep(seconds_to_wait)
            
            # After waiting, run the strategy for all symbols
            last_bar_times = run_strategy_for_all_symbols(last_bar_times)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Bot stopped. Final positions:")
        positions = trading_client.get_all_positions()
        for position in positions:
            print(f"{position.symbol}: {position.qty} shares at avg price ${float(position.avg_entry_price):.2f}")
            print(f"Current value: ${float(position.market_value):.2f}")
            print(f"P&L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc)*100:.2f}%)") 