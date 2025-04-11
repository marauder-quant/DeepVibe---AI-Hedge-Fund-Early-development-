# QMAC Strategy

Quad Moving Average Crossover (QMAC) strategy implementation using vectorbt and Alpaca data.

## Code Structure

The codebase has been organized into modular components for better maintainability:

- **src/config.py**: Configuration parameters and constants
- **src/qmac_main.py**: Main entry point for running the strategy
- **src/lib/strategy_core.py**: Core strategy implementation
- **src/lib/utils.py**: Utility functions and numba-optimized code
- **src/lib/optimizer.py**: Window optimization and parameter tuning
- **src/lib/database.py**: Database operations for storing and retrieving results
- **src/lib/visualization.py**: Plotting and visualization functions

## Usage

You can run the QMAC strategy with various configurations:

```bash
# Run with default parameters
python src/qmac_main.py

# Run with specific symbol and date range
python src/qmac_main.py --symbol "SPY" --start "2022-01-01" --end "2023-01-01"

# Run in fast mode for quick testing
python src/qmac_main.py --fast

# Run with specific timeframe
python src/qmac_main.py --timeframe "1h"

# Run with provided window parameters (skip optimization)
python src/qmac_main.py --no-opt --buy-fast 10 --buy-slow 30 --sell-fast 5 --sell-slow 20

# Calculate total possible combinations without running backtest
python src/qmac_main.py --calculate-only
```

## Data Storage

The strategy stores optimization results and best parameters in SQLite database:

- **db/qmac_parameters.db**: Database for storing optimal parameters

## Results

Generated plots and results are stored in:

- **plots/**: Directory for strategy plots and visualizations
- **results/**: Directory for CSV files with performance metrics
- **checkpoints/**: Directory for optimization checkpoints (allows resuming)

## Dependencies

Main dependencies include:

- vectorbt
- alpaca-py
- numpy/pandas
- numba
- ray (for distributed computing)
- plotly (for visualization)

## Market Data

The strategy can fetch data from:

1. Alpaca API (primary source)
2. Yahoo Finance (fallback)

Make sure to set up your Alpaca API keys in a `.env` file or environment variables:

```
alpaca_paper_key=YOUR_KEY
alpaca_paper_secret=YOUR_SECRET
```

## Features

- Quad Moving Average Crossover (two separate MA crossover systems)
- Comprehensive parameter optimization
- Distributed computing with Ray
- Performance visualization
- Database storage of optimal parameters
- Checkpointing for long-running optimizations 