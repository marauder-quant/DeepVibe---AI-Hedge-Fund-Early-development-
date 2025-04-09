# Market Analysis System

This directory contains the scripts and modules responsible for analyzing economic conditions (determining the market regime or "economic quadrant") and grading individual stocks based on fundamental metrics relevant to that regime.

The results are stored in a central SQLite database (`market_analysis/data/market_analysis.db`) which is used by other parts of the project (like the trading bot).

## Core Concepts

### Economic Quadrant Framework

This system categorizes the market environment based on trends in two key macroeconomic factors obtained from FRED (Federal Reserve Economic Data):

1.  **Federal Reserve Balance Sheet Trend**: Is the Fed expanding (increasing) or contracting (decreasing) its balance sheet? (Proxy for liquidity/quantitative easing or tightening).
2.  **Federal Funds Rate Level**: Are interest rates relatively high or low? (Proxy for the cost of borrowing).

Additionally, the system incorporates:
3.  **Jobs Data & Sentiment**: Non-farm payroll trends with forward-looking sentiment adjustments.
4.  **Consumer Spending & Sentiment**: Personal consumption trends with forward-looking sentiment adjustments.

These factors define the following quadrants, each suggesting different optimal investment styles:

| Quadrant Label  | Balance Sheet | Interest Rates | Typical Environment     | Suggested Focus    |
|-----------------|---------------|----------------|-----------------------|--------------------|
| `A`             | Decreasing    | High           | Inflation Fighting    | Value & Quality    |
| `B`             | Either        | Either         | Growth with Inflation | Balanced (Growth Bias) |
| `B/C (prefer B)`| Increasing    | High           | Transition (Growth Bias) | Balanced -> Growth |
| `B/C (prefer C)`| Decreasing    | Low            | Transition (Value Bias) | Balanced -> Value  |
| `C`             | Either        | Either         | Transition to Growth  | Balanced (Value Bias) |
| `D`             | Increasing    | Low            | Growth Promoting      | Growth             |

Market conditions are scored on a 0-100 scale, with the quadrant determined by score ranges:
- 0-24.9: Quadrant A (Inflation fighting)
- 25-49.9: Quadrant B (Growth with inflation)
- 50-74.9: Quadrant C (Transition to growth)
- 75-100: Quadrant D (Growth)

*(See `economic_quadrant.py` for implementation details)*.

### Sentiment Analysis

The system includes sentiment analysis for both jobs and consumer spending:

1. **Sentiment Calculation**: Compares current values to 2-year moving averages to determine sentiment.
2. **Sentiment Range**: Values from 0.1 (very bearish) to 2.0 (very bullish), with 1.0 being neutral.
3. **Score Adjustment**: Sentiment modifies the raw scores for both jobs and consumer spending:
   - Significantly bearish (<0.9): -1.0 point
   - Slightly bearish (0.9-1.0): -0.5 point
   - Neutral (1.0): No change
   - Slightly bullish (1.0-1.1): +0.5 point
   - Significantly bullish (>1.1): +1.0 point

This allows the system to incorporate forward-looking indicators into the economic quadrant determination.

### Stock Grading

Stocks (primarily S&P 500 components) are fetched using `yfinance` and evaluated based on fundamental metrics. The importance and target ranges for these metrics change depending on the current economic quadrant determined above. The goal is to identify stocks whose characteristics (e.g., value, growth, quality) align best with the prevailing market regime.

Key metrics considered include:
*   Revenue Growth
*   Earnings Growth
*   P/E Ratio
*   Debt-to-EBITDA
*   (Other metrics as defined in `stock_grading.py`)

Stocks are assigned an overall grade from F to A+ based on how well they match the ideal profile for the current quadrant.

*(See `stock_grading.py` for the specific grading logic and criteria per quadrant)*.

## Files & Scripts

*   **`run_full_analysis.py`**: The main script to run the entire analysis process. It determines the quadrant and then grades all S&P 500 stocks, saving results to the database.
*   **`analyze_all_sp500.py`**: Called by `run_full_analysis.py`. Handles fetching S&P 500 tickers and analyzing them in batches.
*   **`economic_quadrant.py`**: Contains the logic for fetching FRED data and determining the current economic quadrant.
*   **`stock_grading.py`**: Contains the logic for fetching stock data (`yfinance`) and applying the grading rules based on the quadrant.
*   **`db_utils.py`**: Provides functions for interacting with the SQLite database (saving analysis results, fetching graded stocks, retrieving quadrant info).
*   **`view_database.py`**: A command-line tool to inspect the contents of the database (view tables, filter by grade/stock, export to CSV).
*   **`generate_report.py`**: Creates summary reports and visualizations based on the latest analysis data in the database.
*   **`fred_api_setup.py`**: Utility to help configure the FRED API key.
*   **`requirements.txt`**: Python package dependencies specific to this module.
*   **`data/`**: Directory containing the SQLite database (`market_analysis.db`).
*   **`reports/`**: Directory where generated reports/visualizations are saved.

## Database (`data/market_analysis.db`)

This SQLite database stores the output of the analysis.

*   **`economic_quadrants` Table**: Stores the history of determined economic quadrants, including the date, quadrant label, underlying data values, and analysis notes.
*   **`stock_analysis` Table**: Stores the results of stock grading for each analysis run. Includes the stock symbol, name, sector, the overall grade, individual metric scores/values, the quadrant at the time of analysis, analysis date, and potentially backtested parameters (if populated by the `backtests` module).

*(See `db_utils.py` for precise schema and interaction functions)*.

## Usage

*(Ensure you have completed the setup steps in the main project [README.md](../../README.md), including installing requirements from `market_analysis/requirements.txt` and setting the `fred_api_key` in your `.env` file)*.

Commands should typically be run from the project's root directory (`/workspaces/dmac_strategy_research_alpaca/`).

1.  **Run the Full Analysis**: (Recommended periodically, e.g., weekly/monthly)
    ```bash
    python market_analysis/run_full_analysis.py
    ```
    This updates the economic quadrant and re-grades all S&P 500 stocks in the database.

2.  **View Database Results**: Check the latest analysis results.
    ```bash
    # View summary (latest quadrant, grade distribution)
    python market_analysis/view_database.py

    # View the full stock analysis table
    python market_analysis/view_database.py --table stock_analysis

    # View the economic quadrant history
    python market_analysis/view_database.py --table economic_quadrants

    # View stocks with grade 'A+' (most recent entry per stock)
    python market_analysis/view_database.py --grade "A+"

    # View details for AAPL (most recent entry)
    python market_analysis/view_database.py --stock AAPL

    # Export current database views to CSV files in market_analysis/data/export
    python market_analysis/view_database.py --export --output ./market_analysis/data/export
    ```

3.  **Generate Reports**: Create visualizations and summary reports.
    ```bash
    python market_analysis/generate_report.py
    ```
    *(Check the `reports/` directory for output)*.

## Dependencies

Requires packages listed in `market_analysis/requirements.txt`. Key dependencies include `pandas`, `yfinance`, `fredapi`, `sqlite3`. 