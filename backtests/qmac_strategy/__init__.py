"""
QMAC (Quad Moving Average Crossover) strategy package.

This package provides a more advanced implementation of moving average crossover 
strategy using four moving averages (two for buy signals, two for sell signals).
"""

# Import directly from src module to provide clean API
try:
    from backtests.qmac_strategy.src.qmac_strategy import (
        run_qmac_strategy,
        analyze_window_combinations,
        optimize_parameters,
        plot_qmac_strategy
    )
except ImportError:
    # Provide empty placeholders if imports fail
    def run_qmac_strategy(*args, **kwargs):
        raise NotImplementedError("QMAC strategy is not properly installed")
        
    def analyze_window_combinations(*args, **kwargs):
        raise NotImplementedError("QMAC strategy is not properly installed")
        
    def optimize_parameters(*args, **kwargs):
        raise NotImplementedError("QMAC strategy is not properly installed")
        
    def plot_qmac_strategy(*args, **kwargs):
        raise NotImplementedError("QMAC strategy is not properly installed")
