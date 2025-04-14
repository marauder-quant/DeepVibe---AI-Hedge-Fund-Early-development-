#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters for QMAC strategy.
"""

# Trading symbol
DEFAULT_SYMBOL = 'SPY'

# Date range for backtesting
DEFAULT_START_DATE = '2020-10-01'  # Format: YYYY-MM-DD
DEFAULT_END_DATE = '2020-10-02'    # Format: YYYY-MM-DD

# Timeframe settings
DEFAULT_TIMEFRAME = '2m'  # Options: '1d', '1h', '30m', '15m', '5m', etc.

# Window size range for optimization
DEFAULT_MIN_WINDOW = 2
DEFAULT_MAX_WINDOW = 252  # For daily timeframe, this is auto-adjusted to 252
DEFAULT_WINDOW_STEP = 5   # Step size between window values (higher = faster but less granular)

# Portfolio parameters
INITIAL_CASH = 100.0  # Initial capital
FEES = 0              # Alpaca is commission free
SLIPPAGE = 0.0025     # 0.25% slippage

# Optimization settings
MAX_COMBINATIONS = -1  # Maximum window combinations to test (set to -1 for unlimited)

# Out-of-sample testing settings
DEFAULT_OOS_STOCKS = 500      # Default number of stocks to test
MAX_OOS_WINDOWS = 50          # Maximum number of time windows to test per stock
TOP_N_PARAMS = 100              # Number of top parameter combinations to store in database
DEFAULT_OOS_WINDOW_LENGTH = 60  # Default length of each window in days
DEFAULT_TOP_PARAMS = 100  # Default number of top parameter sets to test

# Early termination settings
EARLY_TERMINATION_ENABLED = True  # Whether to enable early termination feature
EARLY_TERMINATION_MIN_TESTS = 30  # Minimum number of tests before checking for early termination
EARLY_TERMINATION_CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold below which to terminate testing
EARLY_TERMINATION_CONSECUTIVE_CHECKS = 3  # Number of consecutive low confidence checks needed to terminate

# Frequency for Sharpe ratio and other metrics 
# 'd' for daily, 'h' for hourly, 'm' for minute, 'w' for weekly
METRICS_FREQUENCY = 'h' 