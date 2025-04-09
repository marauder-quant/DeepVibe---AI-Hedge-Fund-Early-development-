# config.py

# Trading parameters
SMA_PERIOD = 50
TIMEFRAME = "4h"  # Yahoo Finance interval (e.g., "1h", "4h", "1d")

# After-hours trading parameters
AFTER_HOURS_SLIPPAGE_TOLERANCE = 0.01  # 1% adjustment for after-hours limit orders

# Number of top stocks to select per grade
TOP_A_PLUS_STOCKS = 1000
TOP_A_STOCKS = 100
TOP_B_STOCKS = 50
TOP_C_STOCKS = 50
TOP_D_STOCKS = 50

# Maximum percentage of the per-stock allocation to use in a single buy order.
# 1.0 means use the full allocated amount. 0.5 means use up to half, etc.
# This helps manage risk concentration per position.
# TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION = 1.0 - OLD
# Percentage of the total allowed equity allocation (determined by quadrant) to use for a single buy trade.
# Example: If quadrant allows 50% equity, and this is 0.025 (2.5%), each buy trade will target
# a value of PortfolioValue * 0.50 * 0.025.
TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION = 0.025 # Default to 2.5%

# Quadrant Configuration
# Define equity allocation percentage for each quadrant
QUADRANT_ALLOCATIONS = {
    'A': 0.2,               # Inflation fighting - conservative
    'B': 0.5,               # Growth with inflation - balanced with growth bias
    'C': 0.5,               # Transition to growth - balanced with value bias
    'D': 1.0,               # Growth quadrant - aggressive
    'Unknown': 0.2,         # Default conservative allocation if quadrant data is missing
}

QUADRANT_A_HEDGE_ALLOCATIONS = {
    'Volatility': (1 - QUADRANT_ALLOCATIONS['A']) * 0.2,
    'Gold': (1 - QUADRANT_ALLOCATIONS['A']) * 0.2,
    'BONDS': (1 - QUADRANT_ALLOCATIONS['A']) * 0.6,
}

print(QUADRANT_A_HEDGE_ALLOCATIONS)

# Define which stock grades are allowed for selection in each quadrant.
# The order matters: Grades listed first are prioritized when selecting top stocks.
# Example: ['A+', 'A', 'B'] means select top A+, then top A, then top B.
QUADRANT_STOCK_GRADES = {
    'A': ['A+', 'A'],               # Conservative: Only A+ and A
    'B': ['B'],                      # Growth with inflation: Exclusively B grade stocks
    'C': ['C'],                      # Transition to growth: Exclusively C grade stocks
    'D': ['D'],                      # Growth quadrant: Exclusively D grade stocks
    'Unknown': ['A+'],              # Default conservative: Only A+
}

# Default values if quadrant data is missing from the database
DEFAULT_QUADRANT = 'Unknown'
# The allocation for DEFAULT_QUADRANT will be taken from QUADRANT_ALLOCATIONS

# Mapping from grade string to the number of stocks variable name
GRADE_LIMIT_MAP = {
    'A+': TOP_A_PLUS_STOCKS,
    'A': TOP_A_STOCKS,
    'B': TOP_B_STOCKS,
    'C': TOP_C_STOCKS,
    'D': TOP_D_STOCKS,
    # Add mappings for other grades if needed (F)
} 