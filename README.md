
![original](https://github.com/user-attachments/assets/89e2332b-809e-4eb8-b54e-d96b858264bd)

# DeepVibe - AI Hedge Fund Framework & Trading Algorithm w/Alpaca Markets

This project provides a framework for analyzing market conditions, grading stocks, backtesting strategies, and executing automated trades via the Alpaca API.

## Overview

The project is divided into three main components, located in separate directories:

1.  **`market_analysis/`**: Contains tools to determine the current economic regime (based on growth, inflation trends, and sentiment indicators) and grade stocks based on fundamental factors suitable for that regime. See the [Market Analysis README](market_analysis/README.md) for details.
2.  **`backtests/`**: Includes scripts for backtesting trading strategies, specifically using walk-forward optimization to find robust parameters for strategies like the 4 Moving Average Crossover. See the [Backtests README](backtests/README.md) for details.
3.  **`trading_bot/`**: Houses the live trading bot that uses the insights from the market analysis (quadrant, selected stocks) and applies a defined trading strategy (e.g., SMA crossover) to execute trades using the Alpaca API. Supports both regular-hours and 24/7 after-hours trading. See the [Trading Bot README](trading_bot/README.md) for details.

## General Workflow

1.  **Run Market Analysis (Periodically)**: Execute scripts in `market_analysis/` (e.g., `run_full_analysis.py`) to update the economic quadrant and stock grades in the `market_analysis/data/market_analysis.db` database.
2.  **Run Backtester (Optional)**: Execute scripts in `backtests/` (e.g., `run_graded_stock_backtests.py`) to test strategies and optimize parameters for selected stocks based on the latest analysis.
3.  **Run Trading Bot (Continuously)**: Execute the main script in `trading_bot/` (e.g., `sma_bot_21_4h.py` or `sma_bot_21_4h_after_hours.py` for 24/7 trading) to perform live or paper trading based on the current analysis and configured strategy.

## Key Features

* **Economic Quadrant Analysis**: Automatically determines the current economic environment (Quadrants A through D) based on Federal Reserve data, jobs reports, consumer spending, and sentiment indicators.
* **Stock Grading & Selection**: Grades stocks based on fundamental metrics appropriate for the current economic quadrant.
* **After-Hours Trading**: Supports 24/7 trading with the after-hours enabled bot using adaptive limit orders.
* **Flexible Configuration**: Uses config files to allow easy adjustment of trading parameters, stock selection criteria, and allocation percentages.
* **Backtesting Framework**: Includes tools for optimizing strategy parameters using walk-forward validation.

## Setup Instructions

### Prerequisites

*   Python 3.8+
*   Alpaca Markets account (Paper or Live)
*   FRED API Key (for economic data in `market_analysis`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd dmac_strategy_research_alpaca
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Install requirements for the main project (including backtester and trading bot):
        ```bash
        pip install -r requirements.txt
        ```
    *   Install requirements specific to the market analysis module:
        ```bash
        pip install -r market_analysis/requirements.txt
        ```

4.  **Configure API Keys:**
    *   Create a `.env` file in the project root directory (`/workspaces/dmac_strategy_research_alpaca/`).
    *   Add your API keys to the `.env` file:
        ```dotenv
        # Alpaca Paper Trading Keys
        alpaca_paper_key=YOUR_ALPACA_PAPER_API_KEY
        alpaca_paper_secret=YOUR_ALPACA_PAPER_SECRET_KEY

        # FRED API Key (Required for Market Analysis)
        fred_api_key=YOUR_FRED_API_KEY

        # Optional: Alpaca Live Trading Keys
        # alpaca_live_key=YOUR_ALPACA_LIVE_API_KEY
        # alpaca_live_secret=YOUR_ALPACA_LIVE_SECRET_KEY
        ```
    *   **Important:** Ensure the `.env` file is added to your `.gitignore` to prevent accidentally committing your keys.

## Usage

Refer to the README files within each subdirectory for detailed usage instructions:

*   [Market Analysis (`market_analysis/README.md`)](market_analysis/README.md)
*   [Backtests (`backtests/README.md`)](backtests/README.md)
*   [Trading Bot (`trading_bot/README.md`)](trading_bot/README.md)

---

*Add any License information or Acknowledgements here.* 
