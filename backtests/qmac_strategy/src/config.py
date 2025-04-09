#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters for QMAC strategy.
"""

# Trading symbol
DEFAULT_SYMBOL = 'SPY'

# Date range for backtesting
DEFAULT_START_DATE = '2019-10-01'  # Format: YYYY-MM-DD
DEFAULT_END_DATE = '2020-01-01'    # Format: YYYY-MM-DD

# Timeframe settings
DEFAULT_TIMEFRAME = '30m'  # Options: '1d', '1h', '30m', '15m', '5m', etc.

# Window size range for optimization
DEFAULT_MIN_WINDOW = 2
DEFAULT_MAX_WINDOW = 252  # For daily timeframe, this is auto-adjusted to 252
DEFAULT_WINDOW_STEP = 1   # Step size between window values (higher = faster but less granular)

# Portfolio parameters
INITIAL_CASH = 100.0  # Initial capital
FEES = 0              # Alpaca is commission free
SLIPPAGE = 0.0025     # 0.25% slippage

# Optimization settings
MAX_COMBINATIONS = -1  # Maximum window combinations to test (set to -1 for unlimited)

# Frequency for Sharpe ratio and other metrics 
# 'd' for daily, 'h' for hourly, 'm' for minute, 'w' for weekly
METRICS_FREQUENCY = 'h' 