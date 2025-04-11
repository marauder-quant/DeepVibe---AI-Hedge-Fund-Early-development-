# Backtests Package

This package contains implementations of various trading strategy backtests with a focus on proper separation of in-sample and out-of-sample testing methodologies.

## Structure

```
backtests/
├── __init__.py               # Package initialization with exports
├── common/                   # Common utilities shared between strategies
│   ├── __init__.py           # Package initialization
│   ├── data_utils.py         # Data fetching and preprocessing utilities
│   ├── data_splitting.py     # Data splitting configurations (IS/OOS)
│   └── visualization.py      # Common visualization functions
├── dmac_strategy/            # Dual Moving Average Crossover strategy
│   ├── __init__.py
│   ├── dmac_strategy.py      # Core DMAC implementation
│   └── run_dmac_backtest.py  # CLI for running DMAC backtests
├── qmac_strategy/            # Quad Moving Average Crossover strategy
│   ├── __init__.py
│   ├── README.md             # QMAC-specific documentation
│   ├── requirements.txt      # QMAC-specific dependencies
│   ├── src/                  # Source code for QMAC
│   ├── plots/                # Generated QMAC visualizations
│   ├── results/              # QMAC backtest results
│   ├── db/                   # QMAC parameter databases
│   ├── checkpoints/          # Optimizer checkpoints
│   └── walk_forward_optimization/ # WFO implementation
├── plots/                    # Generated visualizations
├── results/                  # Backtest results
└── run_graded_stock_backtests.py # Script for running backtests on graded stocks
```

## Data Splitting Methodology

All strategies use a consistent data splitting methodology to properly separate in-sample (IS) and out-of-sample (OOS) data:

- **In-sample data**: Used for parameter optimization and strategy development
- **Out-of-sample data**: Used for validation and performance assessment

The package provides several predefined data splitting configurations:

- **Walk-Forward Optimization (WFO)**: Multiple sequential train/test splits to simulate real trading
- **Cross-Validation (CV)**: Multiple train/test splits for robust parameter selection
- **Fast Testing**: Minimal splits for quick experimentation
- **Comprehensive**: More splits for detailed analysis

## Getting Started

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r backtests/requirements.txt
   ```

2. Set up your Alpaca API credentials in a `.env` file:
   ```
   alpaca_paper_key=YOUR_ALPACA_API_KEY
   alpaca_paper_secret=YOUR_ALPACA_API_SECRET
   ```

### DMAC Strategy

The Dual Moving Average Crossover strategy is a simple trend-following strategy that uses two moving averages to generate entry and exit signals.

#### Basic Usage:

```bash
# Run a basic DMAC backtest
python -m backtests.dmac_strategy.run_dmac_backtest SPY --start 2022-01-01 --end 2023-01-01
```

#### All Available Options:

```bash
python -m backtests.dmac_strategy.run_dmac_backtest SYMBOL [options]

Arguments:
  SYMBOL                      Trading symbol (e.g., SPY, AAPL, BTC/USD)

Options:
  --start DATE                Start date in YYYY-MM-DD format (default: 2018-01-01)
  --end DATE                  End date in YYYY-MM-DD format (default: current date)
  --fast-window INT           Fast moving average window size (default: 30)
  --slow-window INT           Slow moving average window size (default: 80)
  --min-window INT            Minimum window size for optimization (default: 2)
  --max-window INT            Maximum window size for optimization (default: 100)
  --metric STR                Performance metric for optimization (default: total_return)
                              Options: total_return, sharpe_ratio, sortino_ratio, calmar_ratio
  --timeframe STR             Data timeframe (default: 1d)
                              Options: 1d, 1h, 30m, 15m, 5m, etc.
  --cash FLOAT                Initial cash amount (default: 100.0)
  --fees FLOAT                Fee percentage as decimal (default: 0.0025 = 0.25%)
  --slippage FLOAT            Slippage percentage as decimal (default: 0.0025 = 0.25%)
  --optimize                  Run parameter optimization to find best window sizes
  --save-plots                Save strategy plots to output directory
  --output-dir DIR            Directory to save plots (default: backtests/plots)
  --verbose                   Print detailed information during execution (default)
  --quiet                     Minimize output during execution
```

#### Example Commands:

```bash
# Run with specific window sizes
python -m backtests.dmac_strategy.run_dmac_backtest SPY --fast-window 20 --slow-window 50

# Run optimization with custom window range
python -m backtests.dmac_strategy.run_dmac_backtest SPY --optimize --min-window 10 --max-window 200

# Run on hourly data
python -m backtests.dmac_strategy.run_dmac_backtest SPY --timeframe 1h --start 2023-01-01 --end 2023-06-30

# Optimize for Sharpe ratio instead of total return
python -m backtests.dmac_strategy.run_dmac_backtest SPY --optimize --metric sharpe_ratio

# Run with custom trading costs
python -m backtests.dmac_strategy.run_dmac_backtest SPY --fees 0.001 --slippage 0.001
```

### QMAC Strategy

The Quad Moving Average Crossover strategy is an advanced version that uses four moving averages (two for buy signals, two for sell signals).

#### Basic Usage:

```bash
# Run a basic QMAC backtest
python -m backtests.qmac_strategy.src.qmac_main --symbol SPY --start 2022-01-01 --end 2023-01-01
```

#### All Available Options:

```bash
python -m backtests.qmac_strategy.src.qmac_main [options]

Options:
  --symbol STR                Trading symbol (default: SPY)
  --start DATE                Start date in YYYY-MM-DD format (default: 2018-01-01)
  --end DATE                  End date in YYYY-MM-DD format (default: current date)
  --timeframe STR             Data timeframe (default: 1d)
                              Options: 1d, 1h, 30m, 15m, 5m, etc.
  --fast                      Run in fast mode with reduced parameter combinations
  --no-opt                    Skip optimization and use provided parameters
  --buy-fast INT              Buy fast MA window size (default: 10)
  --buy-slow INT              Buy slow MA window size (default: 30)
  --sell-fast INT             Sell fast MA window size (default: 5)
  --sell-slow INT             Sell slow MA window size (default: 20)
  --calculate-only            Calculate total parameter combinations without running backtest
  --split-mode STR            Data splitting mode (default: wfo)
                              Options: wfo, cv, rolling, expanding
  --splits INT                Number of train/test splits (default: 10)
  --is-window INT             In-sample window length in periods (default: 365)
  --oos-window INT            Out-of-sample window length in periods (default: 100)
  --metric STR                Optimization metric (default: sharpe_ratio)
                              Options: total_return, sharpe_ratio, sortino_ratio, calmar_ratio
  --fee FLOAT                 Fee percentage as decimal (default: 0.001 = 0.1%)
  --slippage FLOAT            Slippage percentage as decimal (default: 0.001 = 0.1%)
  --save-db                   Save results to database
  --save-plots                Save plots to output directory
  --output-dir DIR            Output directory (default: backtests/qmac_strategy/plots)
  --verbose                   Print detailed logs
  --quiet                     Minimize output
```

#### Example Commands:

```bash
# Run in fast mode for quick testing
python -m backtests.qmac_strategy.src.qmac_main --fast

# Run with specific parameters (no optimization)
python -m backtests.qmac_strategy.src.qmac_main --no-opt --buy-fast 15 --buy-slow 45 --sell-fast 8 --sell-slow 24

# Run with custom split configuration
python -m backtests.qmac_strategy.src.qmac_main --split-mode cv --splits 5 --is-window 252 --oos-window 63

# Run on hourly data and save results
python -m backtests.qmac_strategy.src.qmac_main --symbol AAPL --timeframe 1h --save-db --save-plots
```

### Graded Stock Backtests

This script runs backtests on stocks based on their grade (from market analysis), using walk-forward optimization to find optimal parameters and storing results in a database.

#### Basic Usage:

```bash
# Run backtest on graded stocks
python -m backtests.run_graded_stock_backtests
```

#### Configuration:

The script is configured through constants at the top of the file. You can modify these parameters directly in the script:

```python
# These can be modified in run_graded_stock_backtests.py:

# Parameter ranges for optimization
BUY_FAST_WINDOWS = np.arange(5, 55, 5)    # 5, 10, ..., 50
BUY_SLOW_WINDOWS = np.arange(10, 110, 10) # 10, 20, ..., 100

START_DATE = datetime(2018, 1, 1)
END_DATE = datetime.now()
TIMEFRAME = "1d"

# Output directories
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Portfolio settings
PORTFOLIO_SETTINGS = {
    'direction': 'longonly',
    'fees': 0.001,
    'slippage': 0.001,
    'freq': TIMEFRAME,
    'init_cash': 10000,
}

# Data splitting configuration
WFO_CONFIG = create_custom_split_config(
    method=SplitMethod.WALK_FORWARD,
    n_splits=10,
    is_window_len=365,
    oos_window_len=100,
    # ...more parameters...
)
```

## Understanding Results

### Output Formats

The backtest results are saved in several formats:

1. **Console Output**: Summary statistics printed to the console
2. **CSV Files**: Detailed results saved in CSV format
3. **PNG/HTML Plots**: Visual representations of strategy performance

### Plots Explained

The strategies generate several types of plots:

1. **Strategy Plot**: Shows price, moving averages, and entry/exit points
2. **Value Comparison**: Compares strategy performance to buy-and-hold
3. **Trades Plot**: Visualizes individual trade performance
4. **Heatmap**: (For optimization) Shows performance across parameter combinations

### Key Metrics

When evaluating strategy performance, consider these key metrics:

- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Return adjusted for risk (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Calmar Ratio**: Return divided by max drawdown

## Advanced Usage

### Custom Data Splitting

You can create custom data splitting configurations:

```python
from backtests.common import create_custom_split_config, SplitMethod

# Create a custom configuration
custom_config = create_custom_split_config(
    method=SplitMethod.EXPANDING,
    n_splits=8,
    is_window_len=252,  # 1 year in trading days
    oos_window_len=63,  # 3 months in trading days
    expand_is_window=True,
    overlap_windows=False,
    min_is_samples=126,
    min_oos_samples=21
)

# Use this configuration in your backtest
result = run_optimization_for_stock(
    symbol="AAPL",
    start_date=start_date,
    end_date=end_date,
    split_config=custom_config
)
```

### Working with Different Data Sources

The package supports multiple data sources:

1. **Alpaca API** (primary): Uses your API credentials
2. **Yahoo Finance** (fallback): Used if Alpaca fails or is unavailable

To use other data sources, you can modify the `fetch_market_data` function in `common/data_utils.py`.

## Dependencies

Common dependencies for all backtests include:
- vectorbt
- alpaca-py
- pandas
- numpy
- plotly

For a complete list, see `requirements.txt`.

## Configuration

All strategies use environment variables for API keys and secrets:
- `alpaca_paper_key`: Your Alpaca API key
- `alpaca_paper_secret`: Your Alpaca API secret

Use a `.env` file in the project root or set these environment variables directly.

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your Alpaca API credentials
2. **Data Splitting Errors**: Ensure you have enough data for the requested splits
3. **Plotting Errors**: Make sure plotly and kaleido are installed
4. **Import Errors**: Check that all dependencies are installed

### Getting Help

For more information, refer to:
- VectorBT documentation: https://vectorbt.dev/
- Alpaca API documentation: https://alpaca.markets/docs/ 