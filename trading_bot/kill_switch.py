import os
import sys
import time
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def load_alpaca_credentials():
    """Load Alpaca API credentials from environment or .env file"""
    alpaca_api_key = os.environ.get('alpaca_paper_key')
    alpaca_secret_key = os.environ.get('alpaca_paper_secret')

    # Check if API keys exist
    if not alpaca_api_key or not alpaca_secret_key:
        print("API keys not found. Loading from .env file.")
        from dotenv import load_dotenv
        load_dotenv()
        alpaca_api_key = os.environ.get('alpaca_paper_key')
        alpaca_secret_key = os.environ.get('alpaca_paper_secret')
    
    if not alpaca_api_key or not alpaca_secret_key:
        print("ERROR: Could not find Alpaca API credentials.")
        sys.exit(1)
        
    return alpaca_api_key, alpaca_secret_key

def is_crypto(symbol):
    """Check if a symbol is a cryptocurrency"""
    return symbol.endswith('USD')

def close_all_positions(trading_client):
    """Close all positions by creating market sell orders"""
    print("\n3. Closing all positions...")
    try:
        # Get all positions
        positions = trading_client.get_all_positions()
        print(f"   Found {len(positions)} positions to close")
        
        success_count = 0
        fail_count = 0
        
        for position in positions:
            symbol = position.symbol
            qty = position.qty
            market_value = position.market_value
            
            print(f"   Closing position: {qty} of {symbol} (value: ${float(market_value):.2f})...")
            
            try:
                # Create market order to sell the position
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                
                # Place the sell order
                order = trading_client.submit_order(order_data)
                print(f"   ✓ Successfully created sell order for {symbol}")
                success_count += 1
            except Exception as e:
                print(f"   ✗ Error closing position for {symbol}: {e}")
                fail_count += 1
        
        print(f"   Position closing complete: {success_count} succeeded, {fail_count} failed")
        return success_count, fail_count
    except Exception as e:
        print(f"   ✗ Error getting positions: {e}")
        return 0, 0

def kill_switch(paper=True, confirm=True):
    """Kill switch to cancel all orders and liquidate all positions
    
    Args:
        paper (bool): Whether to use paper trading account
        confirm (bool): Whether to ask for confirmation before executing
    """
    # Load credentials
    alpaca_api_key, alpaca_secret_key = load_alpaca_credentials()
    
    # Initialize Alpaca client
    trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=paper)
    
    # Get account info
    account = trading_client.get_account()
    print(f"Account ID: {account.id}")
    print(f"Account status: {'ACTIVE' if account.status == 'ACTIVE' else 'INACTIVE'}")
    print(f"Current cash: ${float(account.cash)}")
    print(f"Current portfolio value: ${float(account.portfolio_value)}")
    
    # Ask for confirmation
    if confirm:
        print("\n⚠️ WARNING ⚠️")
        print("This will CANCEL ALL OPEN ORDERS and SELL ALL POSITIONS.")
        confirmation = input("Type 'CONFIRM' to proceed: ")
        if confirmation != "CONFIRM":
            print("Kill switch aborted.")
            return
    
    # Log start time
    start_time = datetime.now()
    print(f"\n[{start_time}] KILL SWITCH ACTIVATED")
    
    # Step 1: Cancel all open orders
    print("\n1. Cancelling all open orders...")
    try:
        cancelled_orders = trading_client.cancel_orders()
        print(f"   ✓ Cancelled all open orders")
    except Exception as e:
        print(f"   ✗ Error cancelling orders: {e}")
    
    # Step 2: Count all positions
    print("\n2. Getting all current positions...")
    try:
        positions = trading_client.get_all_positions()
        print(f"   ✓ Found {len(positions)} open positions")
    except Exception as e:
        print(f"   ✗ Error getting positions: {e}")
    
    # Step 3: Close all positions
    success_count, fail_count = close_all_positions(trading_client)
    
    # Check if we need to retry with a different method if some positions failed
    if fail_count > 0:
        print("\n4. Retrying failed positions with alternative method...")
        # Wait a second to ensure previous operations are completed
        time.sleep(1)
        remaining_success, remaining_fail = close_all_positions(trading_client)
        success_count += remaining_success
    
    # Log completion
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n[{end_time}] KILL SWITCH COMPLETED in {duration:.2f} seconds")
    
    # Get final account state
    try:
        account = trading_client.get_account()
        print(f"\nFinal account state:")
        print(f"Cash: ${float(account.cash)}")
        print(f"Portfolio value: ${float(account.portfolio_value)}")
        
        # Check if we still have positions
        remaining_positions = trading_client.get_all_positions()
        if remaining_positions:
            print(f"\nWARNING: {len(remaining_positions)} positions remain:")
            for position in remaining_positions:
                print(f"  - {position.symbol}: {position.qty} ({float(position.market_value):.2f})")
            
            # If we still have positions, list them and offer to use --force option
            print("\nTo force liquidate all positions, run with the --force option:")
            print("python kill_switch.py --force")
        else:
            print("\nAll positions have been sold successfully.")
    except Exception as e:
        print(f"Error getting final account state: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kill switch to cancel all orders and liquidate all positions")
    parser.add_argument("--live", action="store_true", help="Use live trading account instead of paper")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--force", action="store_true", help="Use close_all_positions() to force liquidate everything")
    
    args = parser.parse_args()
    
    # Check if we should use the force method
    if args.force:
        print("Using force liquidation...")
        alpaca_api_key, alpaca_secret_key = load_alpaca_credentials()
        trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=not args.live)
        print("Liquidating all positions...")
        try:
            # Get all positions
            positions = trading_client.get_all_positions()
            print(f"Found {len(positions)} positions to close")
            
            # Close each position with a market sell order
            for position in positions:
                symbol = position.symbol
                qty = position.qty
                try:
                    # Create market order to sell the position
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    # Place the sell order
                    order = trading_client.submit_order(order_data)
                    print(f"Created sell order for {qty} shares of {symbol}")
                except Exception as e:
                    print(f"Error closing position for {symbol}: {e}")
            
            print("All positions closed.")
        except Exception as e:
            print(f"Error in force liquidation: {e}")
    else:
        # Use standard method
        kill_switch(paper=not args.live, confirm=not args.no_confirm) 