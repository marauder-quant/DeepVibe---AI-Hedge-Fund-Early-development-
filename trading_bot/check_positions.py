import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.environ.get('alpaca_paper_key')
ALPACA_SECRET_KEY = os.environ.get('alpaca_paper_secret')

# Initialize Alpaca client for trading
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Get account info
account = trading_client.get_account()
print(f"Account ID: {account.id}")
print(f"Account status: {'ACTIVE' if account.status == 'ACTIVE' else 'INACTIVE'}")
print(f"Current cash: ${float(account.cash)}")
print(f"Current portfolio value: ${float(account.portfolio_value)}")

# Check positions
print("\nCurrent Positions:")
positions = trading_client.get_all_positions()
if positions:
    for position in positions:
        print(f"{position.symbol}: {position.qty} shares (${float(position.market_value):.2f})")
else:
    print("No open positions")

# Check orders
print("\nOpen Orders:")
orders = trading_client.get_orders()
if orders:
    for order in orders:
        print(f"{order.symbol}: {order.qty} shares - {order.side} - {order.status}")
else:
    print("No open orders") 