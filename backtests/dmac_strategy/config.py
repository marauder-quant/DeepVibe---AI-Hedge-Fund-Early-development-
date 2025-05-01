#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters for DMAC strategy.
"""

# Trading symbol
DEFAULT_SYMBOL = 'SPY'

# Date range for backtesting
DEFAULT_START_DATE = '2021-01-01'  # Format: YYYY-MM-DD
DEFAULT_END_DATE = '2022-01-01'    # Format: YYYY-MM-DD

# Timeframe settings
DEFAULT_TIMEFRAME = '1d'  # Options: '1d', '1h', '30m', '15m', '5m', etc.

# Window size range for optimization
DEFAULT_MIN_WINDOW = 5
DEFAULT_MAX_WINDOW = 50  # For daily timeframe, this is auto-adjusted to 252
DEFAULT_WINDOW_STEP = 5   # Step size between window values (higher = faster but less granular)

# Portfolio parameters
INITIAL_CASH = 100000.0  # Initial capital
FEES = 0.0            # Alpaca is commission free
SLIPPAGE = 0.0025     # 0.25% slippage

# Optimization settings
MAX_COMBINATIONS = 50  # Maximum window combinations to test (set to -1 for all combinations)

# Database settings
TOP_N_PARAMS = 10  # Number of top parameter combinations to store in database
DB_PATH = 'db/dmac_parameters.db'  # Database path (relative to the strategy directory)

# Output settings
DEFAULT_PLOTS_DIR = 'plots'  # Directory to save plots
DEFAULT_EXPORTS_DIR = 'exports'  # Directory to save exports
DEFAULT_LOGS_DIR = 'logs'  # Directory to save logs
DEFAULT_CHECKPOINTS_DIR = 'checkpoints'  # Directory to save optimization checkpoints

# Frequency for Sharpe ratio and other metrics 
# 'd' for daily, 'h' for hourly, 'm' for minute, 'w' for weekly
METRICS_FREQUENCY = 'd'  # Changed to daily for more reliable metrics

# Visualization settings
INTERACTIVE_PLOTS = True  # Whether to generate interactive HTML plots
SAVE_PNG_PLOTS = True  # Whether to save static PNG plots
DEFAULT_PLOT_WIDTH = 1200  # Default plot width in pixels
DEFAULT_PLOT_HEIGHT = 800  # Default plot height in pixels 