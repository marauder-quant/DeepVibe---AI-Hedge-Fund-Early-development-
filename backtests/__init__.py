"""
Backtests package for trading strategies.

This package contains various backtesting implementations for trading strategies.
"""

from backtests.dmac_strategy.dmac_strategy import (
    run_dmac_strategy, 
    analyze_window_combinations, 
    plot_dmac_strategy, 
    plot_heatmap
)

# Export common utilities
from backtests.common import (
    fetch_market_data,
    apply_splits,
    plot_strategy_comparison,
    plot_ma_strategy,
    save_figures,
    ensure_directory,
    SplitMethod,
    get_split_config,
    create_custom_split_config,
    convert_to_vectorbt_params
)

# Provide version information
__version__ = "0.2.0"
