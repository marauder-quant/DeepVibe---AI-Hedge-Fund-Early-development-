#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters for out-of-sample testing of QMAC strategy.
"""

# Trading settings
DEFAULT_TIMEFRAME = '30m'  # Options: '1d', '1h', '30m', '15m', '5m', etc.

# Out-of-sample testing settings
DEFAULT_OOS_STOCKS = 500      # Default number of stocks to test
MAX_OOS_WINDOWS = 50          # Maximum number of time windows to test per stock
DEFAULT_OOS_WINDOW_LENGTH = 60  # Default length of each window in days
DEFAULT_TOP_PARAMS = 100  # Default number of top parameter sets to test
TOP_N_PARAMS = 100      # Number of top parameter combinations to store in database

# Early termination settings
EARLY_TERMINATION_ENABLED = True  # Whether to enable early termination feature
EARLY_TERMINATION_MIN_TESTS = 30  # Minimum number of tests before checking for early termination
EARLY_TERMINATION_CONFIDENCE_THRESHOLD = 0.25  # Confidence threshold below which to terminate testing
EARLY_TERMINATION_CONSECUTIVE_CHECKS = 3  # Number of consecutive low confidence checks needed to terminate

# Database settings
OOS_DB_PATH = 'backtests/qmac_strategy/db/qmac_oos_parameters.db'

# Confidence tracking constants
CONFIDENCE_TRACKER_FILE = 'backtests/qmac_strategy/results/confidence_tracker.json' 