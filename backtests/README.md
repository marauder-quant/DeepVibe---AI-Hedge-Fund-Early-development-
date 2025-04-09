# Backtests

This directory contains the backtesting framework for evaluating and optimizing trading strategies across multiple stocks.

## Functionality

The backtesting module uses walk-forward optimization to find robust strategy parameters for stocks identified by the market analysis component. It:

1.  **Fetches Target Stocks**: Retrieves stocks with specific grades (e.g., A+, A, B) from the market analysis database.
2.  **Applies Walk-Forward Optimization**: For each stock, it divides historical data into multiple in-sample/out-of-sample periods.
3.  **Optimizes Strategy Parameters**: For each in-sample period, it evaluates numerous parameter combinations.
4.  **Validates Out-of-Sample**: Tests the best parameters from each in-sample period on the corresponding out-of-sample period.
5.  **Finds Robust Parameters**: Selects parameter combinations that perform consistently well across all out-of-sample periods.
6.  **Saves Results**: Stores the optimal parameters back to the market analysis database for potential use by the trading bot.

## Files

*   **`run_graded_stock_backtests.py`**: The main script to execute the walk-forward backtests.
    *   It fetches stocks of a specified grade (e.g., 'A') from the `market_analysis.db`.
    *   It iterates through these stocks, running the walk-forward optimization using the strategy defined in `cmcsa_4ma_cross_strategy.py`.
    *   It saves the best average out-of-sample parameters back to the `stock_analysis` table in `market_analysis.db` (in the `best_params_4ma_daily` column).
*   **`cmcsa_4ma_cross_strategy.py`**: Defines the 4 Moving Average Crossover trading strategy logic using the `vectorbt` library. It includes the signal generation and parameter definition used by the backtester.

## Usage

*(Ensure you have completed the setup steps in the main project [README.md](../../README.md), including installing requirements from `requirements.txt`)*.

Commands should typically be run from the project's root directory (`/workspaces/dmac_strategy_research_alpaca/`).

1.  **Run Backtests for Specific Quadrant/Grade**: Execute the main backtesting script with desired parameters.
    ```bash
    # For Quadrant A stocks (value focus)
    python backtests/run_graded_stock_backtests.py --grade A+ --limit 10
    
    # For Quadrant B or D stocks (growth focus)
    python backtests/run_graded_stock_backtests.py --grade B --limit 10
    
    # For any quadrant, using multiple grades
    python backtests/run_graded_stock_backtests.py --grade "A+,A" --limit 20
    ```
    *   **Note**: This script can be computationally intensive and take a long time to run, especially if backtesting many stocks with wide parameter ranges.
    *   Results (the best parameters) are saved directly into the `market_analysis/data/market_analysis.db` database.

2.  **Verify Results**: After the backtester runs, you can use the `view_database.py` tool to see the parameters saved for each stock.
    ```bash
    python market_analysis/view_database.py --table stock_analysis
    ```
    Look for values populated in the `best_params_4ma_daily` column.

## Strategy Details (`cmcsa_4ma_cross_strategy.py`)

*   Uses four moving averages (typically two short-term, two long-term).
*   Generates buy signals when shorter MAs cross above longer MAs in a specific sequence.
*   Generates sell signals when shorter MAs cross below longer MAs.
*   The specific lookback periods for these four MAs are the parameters optimized by `run_graded_stock_backtests.py`.

## Adapting to Economic Quadrants

The backtesting framework can be used to find optimal parameters for stocks appropriate to each economic quadrant:

* **Quadrant A (Inflation Fighting)**: Focus on backtesting value and quality stocks (grades A+ and A)
* **Quadrant B (Growth with Inflation)**: Backtest a balanced mix with growth bias (grades A+, A, and B)
* **Quadrant C (Transition to Growth)**: Backtest a balanced mix with value bias (grades A+ and A)
* **Quadrant D (Growth)**: Focus on backtesting growth stocks (grades B and A)

The parameters saved to the database can then be used by the trading bot to tailor its strategy based on the current economic environment and stock characteristics.

## Dependencies

Requires packages listed in the main `requirements.txt`. Key dependencies include `vectorbt`, `pandas`, `numpy`, `yfinance`, `sqlite3`. 