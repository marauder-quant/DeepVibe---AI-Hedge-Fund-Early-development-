![original](https://github.com/user-attachments/assets/89e2332b-809e-4eb8-b54e-d96b858264bd)

# DeepVibe - AI Hedge Fund Framework & Trading Algorithm w/Alpaca Markets

This project provides a framework for analyzing market conditions, grading stocks, backtesting strategies, and executing automated trades via the Alpaca API.

## Overview

The project is divided into three main components, located in separate directories:

1.  **`market_analysis/`**: Contains tools to determine the current economic regime (based on growth, inflation trends, and sentiment indicators) and grade stocks based on fundamental factors suitable for that regime. See the [Market Analysis README](market_analysis/README.md) for details.
2.  **`backtests/`**: Includes strategies and utilities for backtesting trading systems with proper in-sample/out-of-sample methodology. Features DMAC and QMAC strategies with comprehensive parameter optimization. See the [Backtests README](backtests/README.md) for details.
3.  **`trading_bot/`**: Houses the live trading bot that uses the insights from the market analysis (quadrant, selected stocks) and applies a defined trading strategy (e.g., SMA crossover) to execute trades using the Alpaca API. Supports both regular-hours and 24/7 after-hours trading. See the [Trading Bot README](trading_bot/README.md) for details.

## Complete Setup Guide

### Prerequisites

* Docker and Docker Compose (for containerized setup)
* Git
* Alpaca Markets account (Paper or Live)
* FRED API Key (for economic data in `market_analysis`)

### Python Dependencies

The project uses a comprehensive set of Python packages organized into categories:

1. **Core Dependencies**: Essential packages for data handling and environment management
2. **API Integrations**: Packages for interacting with financial and economic data APIs
3. **Data Processing & Analysis**: Tools for data manipulation and visualization
4. **Database**: Database connectivity and management
5. **Development Tools**: Code formatting, linting, and testing utilities
6. **Parallel Processing**: Tools for performance optimization
7. **CLI & Logging**: Command-line interface and logging utilities

All dependencies are specified in `requirements.txt` with exact version numbers to ensure reproducibility. The file is organized into sections with clear comments for easy maintenance.

#### System Dependencies

Some packages may require additional system libraries. On Ubuntu/Debian, install them with:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libpq-dev
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/marauder-quant/DeepVibe---AI-Hedge-Fund-Early-development-.git
cd DeepVibe---AI-Hedge-Fund-Early-development-
```

### Step 2: Setup Environment

#### Option A: Using Docker (Recommended)

1. Create a `docker-compose.yml` file in the project root:

```yaml
version: '3'
services:
  deepvibe:
    build: .
    volumes:
      - ./:/workspaces/dmac_strategy_research_alpaca
    environment:
      - alpaca_paper_key=${alpaca_paper_key}
      - alpaca_paper_secret=${alpaca_paper_secret}
      - fred_api_key=${fred_api_key}
      - alpaca_live_key=${alpaca_live_key}
      - alpaca_live_secret=${alpaca_live_secret}
    command: tail -f /dev/null
```

2. Create a `Dockerfile` in the project root:

```Dockerfile
FROM python:3.9-slim

WORKDIR /workspaces/dmac_strategy_research_alpaca

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY market_analysis/requirements.txt market_analysis/
COPY backtests/requirements.txt backtests/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r market_analysis/requirements.txt \
    && pip install --no-cache-dir -r backtests/requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]
```

3. Build and start the Docker container:

```bash
docker-compose up -d
```

4. Access the container shell:

```bash
docker-compose exec deepvibe bash
```

#### Option B: Using Local Python Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r market_analysis/requirements.txt
pip install -r backtests/requirements.txt
```

### Step 3: Configure API Keys

1. Create a `.env` file in the project root directory:
```bash
touch .env
```

2. Add your API keys to the `.env` file:
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

3. If using Docker, update the environment variables:
```bash
docker-compose down
docker-compose up -d
```

### Step 4: Database Setup

1. Initialize the market analysis database:
```bash
cd market_analysis
python init_database.py
```

2. Return to project root:
```bash
cd ..
```

## Detailed Workflow

### 1. Market Analysis (Run Weekly/Monthly)

This analyzes economic conditions and grades stocks based on the current regime:

```bash
cd market_analysis
python run_full_analysis.py
```

This will:
- Fetch and analyze economic data to determine the current quadrant (A-D)
- Screen stocks based on fundamental criteria appropriate for that quadrant
- Store results in the SQLite database at `market_analysis/data/market_analysis.db`

### 2. Backtesting (As Needed)

The backtesting system offers several strategies with robust parameter optimization:

#### Available Strategies:

1. **DMAC (Dual Moving Average Crossover)**: A simple trend-following strategy with two moving averages
2. **QMAC (Quad Moving Average Crossover)**: An advanced version using four moving averages for separate buy/sell signals

#### Running Backtests:

```bash
cd backtests

# DMAC Strategy Examples:
# Basic backtest on SPY with default settings
python -m backtests.dmac_strategy.run_dmac_backtest SPY

# Backtest with parameter optimization and date range
python -m backtests.dmac_strategy.run_dmac_backtest SPY --start 2021-01-01 --end 2023-01-01 --optimize --save-plots

# QMAC Strategy Examples:
# Fast mode backtest on SPY
python -m backtests.qmac_strategy.src.qmac_main --fast

# Backtest with custom parameters
python -m backtests.qmac_strategy.src.qmac_main --symbol AAPL --timeframe 1h --no-opt --buy-fast 15 --buy-slow 45 --sell-fast 8 --sell-slow 24

# Run backtests on graded stocks from the market analysis
python -m backtests.run_graded_stock_backtests
```

The backtesting process:
- Uses proper data splitting with in-sample (IS) and out-of-sample (OOS) validation
- Implements walk-forward optimization for robust parameter selection
- Generates performance visualizations and statistical reports
- Saves optimal parameters that can be used by the trading bot

#### Key Features:

- **Data Splitting Methodology**: Clear separation of training and testing data to prevent overfitting
- **Multiple Timeframes**: Support for daily, hourly, and intraday data
- **Performance Metrics**: Comprehensive metrics including Sharpe ratio, Sortino ratio, drawdown, and more
- **Visualization**: Interactive plots for strategy performance and parameter heatmaps

For full documentation of all available options, see the [Backtests README](backtests/README.md).

### 3. Live/Paper Trading (Continuous)

Start the trading bot with your chosen configuration:

```bash
cd trading_bot

# For regular market hours trading
python sma_bot_21_4h.py

# For 24/7 after-hours trading
python sma_bot_21_4h_after_hours.py
```

The bot will:
- Load the latest economic quadrant and stock grades from the database
- Apply your chosen trading strategy with optimized parameters
- Execute trades via the Alpaca API (paper or live)
- Log all activities and maintain position tracking

## Configuration Options

Each component has configuration files that control its behavior:

- **Market Analysis**: `market_analysis/config/analysis_config.json` - controls economic data sources, thresholds, and stock screening criteria
- **Backtests**: Configuration is available through command-line arguments or by directly editing strategy scripts
- **Trading Bot**: `trading_bot/config/bot_config.json` - controls trading strategy parameters, risk management, and allocation percentages

Customize these files to adjust the system to your preferences.

## Monitoring and Maintenance

### Logs

All components write logs to their respective `logs` directories:
- `market_analysis/logs/`
- `backtests/logs/`
- `trading_bot/logs/`

Review these logs regularly to monitor system performance and troubleshoot issues.

### Database Maintenance

The market analysis database may grow over time. Periodically clean up old data:

```bash
cd market_analysis
python cleanup_database.py
```

## Troubleshooting

### Common Issues

1. **API Connection Failures**:
   - Verify your API keys in the `.env` file
   - Check your internet connection
   - Confirm the Alpaca API status at status.alpaca.markets

2. **Database Errors**:
   - Ensure the database is properly initialized
   - Check file permissions
   - Verify SQL queries in the logs

3. **Trading Errors**:
   - Confirm sufficient buying power in your Alpaca account
   - Check for trading restrictions on specific securities
   - Verify market hours for your trading strategy

4. **Backtest Issues**:
   - Ensure sufficient historical data for the chosen date range and timeframe
   - Verify that required dependencies are installed
   - Check the date range has enough data for the specified window sizes

## Additional Resources

* [Alpaca API Documentation](https://alpaca.markets/docs/api-documentation/)
* [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
* Market Analysis: [market_analysis/README.md](market_analysis/README.md)
* Backtesting: [backtests/README.md](backtests/README.md)
* Trading Bot: [trading_bot/README.md](trading_bot/README.md)

---

*This project is provided for educational purposes only. Trading involves significant risk of loss and is not suitable for all investors.* 
