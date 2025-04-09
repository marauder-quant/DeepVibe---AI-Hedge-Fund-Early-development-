import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import sys
import importlib.util
from dotenv import load_dotenv
import config # Import the new config file
from pytz import timezone
    
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
from market_analysis.db_utils import get_latest_economic_quadrant

# Bond ETF ticker symbols
BOND_TICKERS = ["TLT", "AGG"]

def is_regular_market_hours():
    """Check if we're in regular market hours (9:30 AM - 4:00 PM ET)"""
    now = datetime.now(timezone('America/New_York'))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close and now.weekday() < 5

def calculate_aggressive_limit_price(current_price, side):
    """
    Calculate an aggressive limit price for after-hours trading
    For buys: Add a small premium to increase fill probability
    For sells: Subtract a small discount to increase fill probability
    """
    if side == OrderSide.BUY:
        # For buys, we're willing to pay slightly more
        return current_price * (1 + config.AFTER_HOURS_SLIPPAGE_TOLERANCE)
    else:
        # For sells, we're willing to accept slightly less
        return current_price * (1 - config.AFTER_HOURS_SLIPPAGE_TOLERANCE)

def get_current_quadrant():
    """
    Get current economic quadrant information
    Returns dict with quadrant info
    """
    quadrant_data = get_latest_economic_quadrant()
    
    if not quadrant_data:
        print(f"No economic quadrant data found. Using default quadrant: {config.DEFAULT_QUADRANT}")
        quadrant = config.DEFAULT_QUADRANT
        notes = 'Using default conservative allocation (no quadrant data)'
    else:
        quadrant = quadrant_data.get('quadrant', config.DEFAULT_QUADRANT)
        notes = quadrant_data.get('analysis_notes', 'No notes available')
    
    return {
        'quadrant': quadrant,
        'notes': notes
    }

def is_quadrant_a():
    """Check if the current economic quadrant is 'A'"""
    quadrant_info = get_current_quadrant()
    quadrant = quadrant_info['quadrant']
    print(f"Current economic quadrant: {quadrant} (Notes: {quadrant_info['notes']})")
    return quadrant == 'A'

def get_bonds_allocation():
    """
    Get the bonds allocation based on the A quadrant hedge setting
    Returns the allocation percentage (0.0-1.0)
    """
    # Calculate the bonds allocation from config
    bonds_allocation = config.QUADRANT_A_HEDGE_ALLOCATIONS['BONDS']
    print(f"Total bonds allocation: {bonds_allocation:.2%} of portfolio")
    return bonds_allocation

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
        # Get data for the past 60 days with 4 hour intervals for SMA
        ticker = yf.Ticker(symbol)
        # Using intervals from config
        df = ticker.history(period="60d", interval=config.TIMEFRAME)
        
        if df.empty:
            print(f"No data returned from Yahoo Finance for {symbol}")
            return None, None, None
        
        # Calculate SMA (from config)
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

def calculate_buy_quantity(symbol, current_price):
    """Calculate number of shares to buy based on bonds allocation for each ETF"""
    try:
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        
        # Calculate the target value for bonds based on config
        bonds_allocation = get_bonds_allocation()
        # Each bond ETF gets 50% of the total bonds allocation
        per_etf_allocation = bonds_allocation / len(BOND_TICKERS)
        target_trade_value = portfolio_value * per_etf_allocation
        
        print(f"Portfolio value: ${portfolio_value:.2f}")
        print(f"Bonds allocation: {bonds_allocation:.2%}")
        print(f"Per ETF allocation: {per_etf_allocation:.2%}")
        print(f"Target trade value for {symbol}: ${target_trade_value:.2f}")

        # Calculate quantity based on current price and target trade value
        if current_price > 0:
            quantity = int(target_trade_value / current_price)
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

        print(f"Calculated buy quantity for {symbol}: {quantity} shares at ${current_price:.2f}")
        
        return quantity
    except Exception as e:
        print(f"Error calculating buy quantity for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # Return 0 on error
        return 0

def place_order(symbol, side, quantity, current_price):
    """Place either a market or limit order based on trading hours."""
    # Don't place orders with zero quantity
    if quantity <= 0:
        print(f"Quantity is zero or less ({quantity}) for {symbol}. Skipping {side.name} order.")
        return None
    
    try:
        if is_regular_market_hours():
            # During regular hours - use market orders
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            print(f"Market hours - Using market order for {side.name}")
        else:
            # After hours - use limit orders with aggressive pricing
            limit_price = calculate_aggressive_limit_price(current_price, side)
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                extended_hours=True
            )
            print(f"After hours - Using limit order for {side.name} at price ${limit_price:.2f}")
        
        order = trading_client.submit_order(order_data)
        print(f"{datetime.now()}: {side.name} order placed for {quantity} shares of {symbol}")
        return order
        
    except Exception as e:
        print(f"Error placing {side.name} order for {quantity} shares of {symbol}: {e}")
        return None

def run_bond_strategy(symbol, last_bar_time=None, force_check=False):
    """Run the SMA crossover strategy for bond ETFs, only trading during Quadrant A"""
    print(f"{datetime.now()}: Checking {symbol}...")
    
    # Check if we're in Quadrant A
    quadrant_a = is_quadrant_a()
    
    if not quadrant_a:
        print(f"Current quadrant is not A. Bond bot is in sleep mode, monitoring only.")
        # Even in sleep mode, we still check the position and sell if price is below SMA
    
    # Check if we already have a position
    current_position = get_position_quantity(symbol)
    print(f"{symbol} current position: {current_position} shares")
    
    # Get current price and SMA using Yahoo Finance
    current_price, sma, latest_bar_time = get_market_data(symbol)
    
    if current_price is None or sma is None or latest_bar_time is None:
        print(f"Failed to get price data for {symbol}. Skipping.")
        return last_bar_time
    
    print(f"{symbol} current price: ${current_price:.2f}")
    print(f"{symbol} {config.SMA_PERIOD}-period SMA ({config.TIMEFRAME}): ${sma:.2f}")
    print(f"Latest {config.TIMEFRAME} bar close time: {latest_bar_time}")
    
    # Check if this is a new bar (or first run)
    is_new_bar = force_check or last_bar_time is None or latest_bar_time > last_bar_time
    
    # Only execute trading logic if we have a new bar
    if is_new_bar:
        print(f"New {config.TIMEFRAME} bar detected for {symbol} - executing trading logic")
        
        # Execute trading logic
        if current_price > sma:
            # Price is above SMA - should be in long position
            if current_position <= 0 and quadrant_a:
                # Only buy if we're in Quadrant A
                buy_quantity = calculate_buy_quantity(symbol, current_price)
                print(f"{symbol} price above SMA & in Quadrant A - Placing buy order for {buy_quantity} shares")
                place_order(symbol, OrderSide.BUY, buy_quantity, current_price)
            elif current_position <= 0 and not quadrant_a:
                print(f"{symbol} price above SMA but not in Quadrant A - Not buying")
            else:  # current_position > 0
                print(f"{symbol} already in long position - holding")
        else:
            # Price is below SMA - should be out of position
            # Sell position if price drops below SMA, regardless of quadrant
            if current_position > 0:
                print(f"{symbol} price below SMA - Selling entire position ({current_position} shares)")
                place_order(symbol, OrderSide.SELL, current_position, current_price)
            else:
                print(f"{symbol} already out of position - holding cash")
    else:
        print(f"No new {config.TIMEFRAME} bar for {symbol} - monitoring only")
    
    print(f"Finished checking {symbol}")
    print("-" * 40)
    
    return latest_bar_time

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
    
    # Get quadrant information and bonds allocation
    quadrant_info = get_current_quadrant()
    bonds_allocation = get_bonds_allocation()
    
    print(f"=== BONDS TRADING BOT ===")
    print(f"Current economic quadrant: {quadrant_info['quadrant']}")
    print(f"Notes: {quadrant_info['notes']}")
    print(f"Total bonds allocation: {bonds_allocation:.2%} of portfolio")
    print(f"Per ETF allocation: {bonds_allocation/len(BOND_TICKERS):.2%} of portfolio")
    print(f"Bond trading active: {'Yes' if quadrant_info['quadrant'] == 'A' else 'No - Sleep mode'}")
    
    print(f"Starting monitoring using {config.SMA_PERIOD}-period SMA on {config.TIMEFRAME} timeframe")
    print(f"Bot will scan after each {config.TIMEFRAME} bar closes (with 5-minute buffer for data availability)")
    print(f"Note: Trading will only occur when a new {config.TIMEFRAME} bar closes")
    print(f"After-hours slippage tolerance: {config.AFTER_HOURS_SLIPPAGE_TOLERANCE * 100:.2f}%")
    print("Press Ctrl+C to stop the bot")
    
    # Variable to store the latest bar time for each ticker
    last_bar_times = {ticker: None for ticker in BOND_TICKERS}
    
    try:
        # Run strategy immediately on startup and force a check for each bond ETF
        for ticker in BOND_TICKERS:
            last_bar_times[ticker] = run_bond_strategy(ticker, force_check=True)
        
        while True:
            # Calculate time until next 4-hour bar closes (plus buffer)
            seconds_to_wait, next_check_time = calculate_next_4h_bar_time()
            
            hours = int(seconds_to_wait // 3600)
            minutes = int((seconds_to_wait % 3600) // 60)
            seconds = int(seconds_to_wait % 60)
            
            print(f"Next scan scheduled for: {next_check_time}")
            print(f"Waiting {hours}h {minutes}m {seconds}s until after next 4-hour bar closes...")
            time.sleep(seconds_to_wait)
            
            # After waiting, run the strategy for each bond ETF
            for ticker in BOND_TICKERS:
                last_bar_times[ticker] = run_bond_strategy(ticker, last_bar_times[ticker])
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Bonds bot stopped. Final positions:")
        for ticker in BOND_TICKERS:
            position_qty = get_position_quantity(ticker)
            if position_qty > 0:
                position = next((p for p in trading_client.get_all_positions() if p.symbol == ticker), None)
                if position:
                    print(f"{position.symbol}: {position.qty} shares at avg price ${float(position.avg_entry_price):.2f}")
                    print(f"Current value: ${float(position.market_value):.2f}")
                    print(f"P&L: ${float(position.unrealized_pl):.2f} ({float(position.unrealized_plpc)*100:.2f}%)")
            else:
                print(f"No position in {ticker}") 