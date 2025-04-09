"""
Backtests package for trading strategies.

This package contains various backtesting implementations for trading strategies.
"""

# Re-export modules from the dmac_strategy package
from backtests.dmac_strategy.dmac_strategy import (
    run_dmac_strategy, 
    analyze_window_combinations, 
    plot_dmac_strategy, 
    plot_heatmap
)

__version__ = "0.1.0"
