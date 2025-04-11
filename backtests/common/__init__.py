"""
Common utilities for backtesting strategies.
"""

# Export data utilities
from backtests.common.data_utils import (
    fetch_market_data,
    parse_timeframe,
    get_alpaca_timeframe,
    apply_splits
)

# Export visualization utilities
from backtests.common.visualization import (
    plot_strategy_comparison,
    plot_ma_strategy,
    plot_heatmap,
    save_figures,
    ensure_directory
)

# Export data splitting utilities
from backtests.common.data_splitting import (
    SplitMethod,
    get_split_config,
    create_custom_split_config,
    convert_to_vectorbt_params,
    DEFAULT_WFO_CONFIG,
    DEFAULT_CV_CONFIG,
    FAST_TEST_CONFIG,
    COMPREHENSIVE_CONFIG
)

__all__ = [
    'fetch_market_data',
    'parse_timeframe',
    'get_alpaca_timeframe',
    'apply_splits',
    'plot_strategy_comparison',
    'plot_ma_strategy',
    'plot_heatmap',
    'save_figures',
    'ensure_directory',
    'SplitMethod',
    'get_split_config',
    'create_custom_split_config',
    'convert_to_vectorbt_params',
    'DEFAULT_WFO_CONFIG',
    'DEFAULT_CV_CONFIG',
    'FAST_TEST_CONFIG',
    'COMPREHENSIVE_CONFIG'
] 