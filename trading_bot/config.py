# config.py

# Trading parameters
SMA_PERIOD = 50
TIMEFRAME = "4h"  # Yahoo Finance interval (e.g., "1h", "4h", "1d")

# After-hours trading parameters
AFTER_HOURS_SLIPPAGE_TOLERANCE = 0.01  # 1% adjustment for after-hours limit orders

# Number of top stocks to select per grade
TOP_A_PLUS_STOCKS = 1000
TOP_A_STOCKS = 100
TOP_B_STOCKS = 50 # Note: B stocks might only be added in certain quadrants based on config below

# Maximum percentage of the per-stock allocation to use in a single buy order.
# 1.0 means use the full allocated amount. 0.5 means use up to half, etc.
# This helps manage risk concentration per position.
# TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION = 1.0 - OLD
# Percentage of the total allowed equity allocation (determined by quadrant) to use for a single buy trade.
# Example: If quadrant allows 50% equity, and this is 0.05 (5%), each buy trade will target
# a value of PortfolioValue * 0.50 * 0.05.
TRADE_SIZE_PERCENT_OF_EQUITY_ALLOCATION = 0.025 # Default to 5%

# Quadrant Configuration
# Define equity allocation percentage for each quadrant
QUADRANT_ALLOCATIONS = {
    'A': 0.2,               # Inflation fighting - conservative
    'B': 0.5,               # Growth with inflation - balanced with growth bias
    'B/C (prefer B)': 0.5,  # Transition (growth bias) - moderate
    'B/C (prefer C)': 0.5,  # Transition (value bias) - moderate
    'C': 0.7,               # Transition to growth - balanced with value bias
    'D': 1.0,               # Growth quadrant - aggressive
    'Unknown': 0.2,         # Default conservative allocation if quadrant data is missing
}

# Define which stock grades are allowed for selection in each quadrant.
# The order matters: Grades listed first are prioritized when selecting top stocks.
# Example: ['A+', 'A', 'B'] means select top A+, then top A, then top B.
QUADRANT_STOCK_GRADES = {
    'A': ['A+', 'A'],               # Conservative: Only A+ and A
    'B': ['A+', 'A', 'B'],         # Growth with inflation: A+, A, and some B
    'B/C (prefer B)': ['A+', 'A', 'B'], # Growth bias: Include B
    'B/C (prefer C)': ['A+', 'A'],      # Value bias: Stick to A+ and A
    'C': ['A+', 'A', 'B'],         # Balanced: Include A+, A, and B
    'D': ['A+', 'A', 'B'],               # Aggressive: Include B
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
    # Add mappings for other grades if needed (C, D, F)
} 