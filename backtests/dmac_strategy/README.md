# DMAC Strategy with Database Support

This module provides an enhanced implementation of the Dual Moving Average Crossover (DMAC) strategy with database support for storing and retrieving optimal parameters across different timeframes and symbols.

## Features

- **Parameter Optimization**: Automatically test multiple combinations of fast and slow moving average window sizes to find the optimal parameters.
- **Database Storage**: Store the best parameters in a SQLite database for future use.
- **Enhanced Visualizations**: Generate interactive plots of strategy performance, parameter space, and trade statistics.
- **Multi-timeframe Support**: Run the strategy on different timeframes (1d, 1h, 30m, 15m, etc.).
- **Command-line Interface**: Easy-to-use command-line interface for running backtests and querying the database.

## Directory Structure

```
dmac_strategy/
├── lib/                 # Library modules
│   ├── database.py      # Database utilities
│   ├── optimizer.py     # Parameter optimization
│   ├── visualization.py # Plotting and visualization
│   └── __init__.py      # Package init
├── db/                  # Database storage
├── exports/             # CSV exports
├── plots/               # Generated plots
├── logs/                # Log files
├── checkpoints/         # Optimization checkpoints
├── results/             # Raw results data
├── dmac_strategy.py     # Core strategy implementation
├── run_dmac_backtest.py # Original backtest runner
├── run_dmac_strategy_db.py # Enhanced backtest runner with DB support
├── dmac_db_query.py     # Database query utility
├── export_dmac_db.py    # Database export utility
└── README.md            # This file
```

## Usage

### Running a Backtest with Optimization

```bash
python run_dmac_strategy_db.py --symbol SPY --start 2020-01-01 --end 2020-12-31 --timeframe 1d
```

This will:
1. Download historical data for SPY from 2020-01-01 to 2020-12-31
2. Test multiple window combinations
3. Find the optimal parameters
4. Store the best parameters in the database
5. Generate visualization plots

### Running with Specific Parameters

```bash
python run_dmac_strategy_db.py --symbol SPY --start 2020-01-01 --end 2020-12-31 --timeframe 1d --fast-window 10 --slow-window 30
```

This will run the strategy with the specified window sizes, skipping the optimization process.

### Loading Parameters from Database

```bash
python run_dmac_strategy_db.py --symbol SPY --start 2020-01-01 --end 2020-12-31 --timeframe 1d --db-only
```

This will load the optimal parameters from the database and run the strategy with those parameters.

### Querying the Database

```bash
python run_dmac_strategy_db.py --db-query
```

This will display a summary of the database contents, including available symbols and timeframes.

### Exporting Database Contents

```bash
python export_dmac_db.py --symbol SPY --timeframe 1d
```

This will export the parameters for SPY on the 1d timeframe to a CSV file in the exports directory.

## Command-line Arguments

### run_dmac_strategy_db.py

| Argument | Description | Default |
|----------|-------------|---------|
| --symbol | Trading symbol (e.g., SPY) | SPY |
| --start | Start date (YYYY-MM-DD) | 2020-01-01 |
| --end | End date (YYYY-MM-DD) | 2020-12-31 |
| --timeframe | Timeframe (e.g., 1d, 30m, 1h, 15m) | 1d |
| --min-window | Minimum window size | 2 |
| --max-window | Maximum window size | 252 |
| --window-step | Step size between window values | 1 |
| --fast-window | Fast MA window size (skip optimization if provided) | None |
| --slow-window | Slow MA window size (skip optimization if provided) | None |
| --max-combinations | Maximum number of window combinations to test | 1000 |
| --metric | Performance metric to optimize for | total_return |
| --db-only | Load optimal parameters from database | False |
| --db-query | Query and display database contents | False |
| --no-save-db | Do not save results to the database | False |
| --top-n | Number of top parameter combinations to save | 10 |
| --output-dir | Directory to save output plots | auto-generated |

### dmac_db_query.py

| Argument | Description | Default |
|----------|-------------|---------|
| --symbol | Symbol to query (e.g., SPY) | None |
| --timeframe | Timeframe to query (e.g., 1d, 1h, 30m) | None |
| --list | List database contents | False |
| --top | Number of top results to show | 10 |
| --format | Output format (pretty, csv, json) | pretty |
| --output | Output file (if not specified, prints to console) | None |

### export_dmac_db.py

| Argument | Description | Default |
|----------|-------------|---------|
| --output-dir | Directory to save the CSV files | exports/ |
| --symbol | Filter by symbol | None |
| --timeframe | Filter by timeframe | None |
| --no-format | Disable timestamp and descriptive names in output filename | False |
| --summary | Print database summary | False |

## Examples

### Finding Optimal Parameters for Multiple Timeframes

```bash
# Run on daily timeframe
python run_dmac_strategy_db.py --symbol SPY --timeframe 1d --start 2020-01-01 --end 2020-12-31

# Run on hourly timeframe
python run_dmac_strategy_db.py --symbol SPY --timeframe 1h --start 2020-01-01 --end 2020-12-31

# Run on 15-minute timeframe
python run_dmac_strategy_db.py --symbol SPY --timeframe 15m --start 2020-01-01 --end 2020-01-31
```

### Comparing Results Across Timeframes

```bash
# Export results to CSV
python export_dmac_db.py --symbol SPY

# View top parameters for each timeframe
python dmac_db_query.py --symbol SPY
```

## Database Structure

The strategy parameters are stored in a SQLite database located in the `db/` directory. The database contains the following table:

### best_parameters

| Column | Description |
|--------|-------------|
| id | Primary key |
| symbol | Trading symbol |
| timeframe | Data timeframe |
| rank | Parameter rank based on performance |
| fast_window | Fast MA window size |
| slow_window | Slow MA window size |
| performance | Strategy performance metric |
| total_return | Total return percentage |
| sharpe_ratio | Sharpe ratio |
| max_drawdown | Maximum drawdown percentage |
| num_trades | Number of trades |
| win_rate | Win rate percentage |
| date_from | Start date of backtest |
| date_to | End date of backtest |
| timestamp | Timestamp when the entry was created | 

## Configuration File

The DMAC strategy is highly configurable through the `config.py` file. This file contains various settings that control the behavior of the strategy, including:

- Default symbol and date ranges
- Timeframe settings
- Window size parameters for optimization
- Portfolio parameters like fees and slippage
- Database settings
- Visualization options

You can customize these settings by modifying the `config.py` file directly, or override them through command-line arguments in many cases.

Example `config.py` settings:

```python
# Trading symbol
DEFAULT_SYMBOL = 'SPY'

# Date range for backtesting
DEFAULT_START_DATE = '2020-01-01'  # Format: YYYY-MM-DD
DEFAULT_END_DATE = '2020-12-31'    # Format: YYYY-MM-DD

# Timeframe settings
DEFAULT_TIMEFRAME = '1d'  # Options: '1d', '1h', '30m', '15m', '5m', etc.

# Window size range for optimization
DEFAULT_MIN_WINDOW = 2
DEFAULT_MAX_WINDOW = 252  # For daily timeframe, this is auto-adjusted to 252
DEFAULT_WINDOW_STEP = 5   # Step size between window values

# Portfolio parameters
INITIAL_CASH = 100.0  # Initial capital
FEES = 0.0            # Alpaca is commission free
SLIPPAGE = 0.0025     # 0.25% slippage
```

## Simplified Execution

The DMAC strategy can now be run with a single command and no arguments required. All settings are configured in the `config.py` file:

```bash
python main.py
```

This will run a complete backtest using all settings from the configuration file, including:
- Optimizing window combinations
- Finding the best parameters
- Saving results to the database
- Generating visualizations
- Comparing to buy-and-hold strategy

This approach makes it very easy to run consistent backtests - simply update the settings in the config file and run the main script without any command-line arguments. 