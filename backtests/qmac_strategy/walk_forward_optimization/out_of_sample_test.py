#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Out-of-sample testing for QMAC strategy.
This script tests the best parameters found from optimization on different stocks and time periods.
"""

import os
import sys
import json
import logging
import multiprocessing as mp
import random
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.logging import RichHandler
import warnings
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import QMAC strategy
try:
    from backtests.qmac_strategy.src.qmac_strategy import run_qmac_strategy
except ImportError:
    # If direct import fails, try relative import
    from src.qmac_strategy import run_qmac_strategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")
console = Console()

# Suppress warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    """Get list of S&P 500 tickers from Wikipedia."""
    try:
        # Try to get from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        print(f"Error getting S&P 500 tickers from Wikipedia: {e}")
        # Fallback to a sample list
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']

def generate_random_periods(start_date, end_date, n_periods=30, period_length=60):
    """Generate random 2-month periods within the date range."""
    periods = []
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    for _ in range(n_periods):
        # Generate random start date
        days_range = (end_date - start_date).days - period_length
        if days_range <= 0:
            continue
            
        random_start = start_date + timedelta(days=random.randint(0, days_range))
        random_end = random_start + timedelta(days=period_length)
        periods.append((random_start, random_end))
    
    return periods

def test_parameters_on_stock_period(params, symbol, start_date, end_date, timeframe='1d'):
    """Test QMAC parameters on a specific stock and period."""
    try:
        # Run strategy with given parameters
        result = run_qmac_strategy(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            buy_fast_window=params['buy_fast'],
            buy_slow_window=params['buy_slow'],
            sell_fast_window=params['sell_fast'],
            sell_slow_window=params['sell_slow'],
            timeframe=timeframe,
            verbose=False
        )
        
        # Get performance metrics
        pf = result['qmac_pf']
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.stats()['Sharpe Ratio'],
            'max_drawdown': pf.stats()['Max Drawdown [%]'] / 100,
            'n_trades': len(pf.trades)
        }
    except Exception as e:
        print(f"Error testing {symbol} from {start_date} to {end_date}: {e}")
        return None

def run_out_of_sample_test(params, start_date=None, end_date=None, n_periods=1, 
                          period_length=60, n_stocks=30, timeframe='1d', n_cores=None):
    """Run out-of-sample testing on random stocks and periods."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Get S&P 500 tickers
    log.info("Getting S&P 500 tickers...")
    sp500_tickers = get_sp500_tickers()
    selected_stocks = random.sample(sp500_tickers, min(n_stocks, len(sp500_tickers)))
    log.info(f"Selected {len(selected_stocks)} stocks for testing")
    
    # Generate random periods
    if start_date is None:
        start_date = datetime(2018, 1, 1, tzinfo=pytz.UTC)
    if end_date is None:
        end_date = datetime.now(pytz.UTC)
    
    log.info("Generating test periods...")
    periods = generate_random_periods(start_date, end_date, n_periods, period_length)
    
    # Prepare test cases
    test_cases = []
    for stock in selected_stocks:
        for period_start, period_end in periods:
            test_cases.append((params, stock, period_start, period_end, timeframe))
    
    total_tests = len(test_cases)
    log.info(f"Running {total_tests} tests using {n_cores} CPU cores...")
    
    # Run tests in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(test_parameters_on_stock_period, *case) for case in test_cases]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Testing...", total=total_tests)
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                progress.update(task, advance=1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        'n_tests': len(results),
        'avg_return': results_df['total_return'].mean(),
        'median_return': results_df['total_return'].median(),
        'std_return': results_df['total_return'].std(),
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_trades': results_df['n_trades'].mean(),
        'success_rate': (results_df['total_return'] > 0).mean()
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_results_{timestamp}.csv')
    summary_file = os.path.join('backtests/qmac_strategy/results', f'qmac_oos_summary_{timestamp}.json')
    
    results_df.to_csv(results_file, index=False)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Display summary
    summary_table = Table(title="Out-of-Sample Test Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    for key, value in summary.items():
        if isinstance(value, float):
            if key in ['avg_return', 'median_return', 'avg_max_drawdown', 'success_rate']:
                summary_table.add_row(key, f"{value:.2%}")
            else:
                summary_table.add_row(key, f"{value:.2f}")
        else:
            summary_table.add_row(key, str(value))
    
    console.print(summary_table)
    console.print(f"\nResults saved to {results_file}")
    console.print(f"Summary saved to {summary_file}")
    
    return results_df, summary

def main():
    """Main function to run out-of-sample testing."""
    # Read best parameters from CSV
    results_dir = 'backtests/qmac_strategy/results'
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and 'qmac_results' in f]
    if not csv_files:
        print("No results CSV files found!")
        return
    
    # Get the most recent 30m results file
    latest_file = max([f for f in csv_files if '30m' in f], 
                     key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    results_df = pd.read_csv(os.path.join(results_dir, latest_file))
    
    # Get the best parameters
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    best_params = {
        'buy_fast': int(best_row['buy_fast']),
        'buy_slow': int(best_row['buy_slow']),
        'sell_fast': int(best_row['sell_fast']),
        'sell_slow': int(best_row['sell_slow'])
    }
    
    print(f"\nBest parameters from {latest_file}:")
    print(f"Buy Fast: {best_params['buy_fast']}")
    print(f"Buy Slow: {best_params['buy_slow']}")
    print(f"Sell Fast: {best_params['sell_fast']}")
    print(f"Sell Slow: {best_params['sell_slow']}")
    print(f"Total Return: {best_row['total_return']:.2%}")
    
    # Run out-of-sample testing
    print("\nStarting out-of-sample testing...")
    run_out_of_sample_test(
        params=best_params,
        start_date=datetime(2018, 1, 1, tzinfo=pytz.UTC),
        end_date=datetime.now(pytz.UTC),
        n_periods=1,
        period_length=60,  # 2 months
        n_stocks=30,
        timeframe='30m',  # Match the timeframe of the best parameters
        n_cores=4
    )

if __name__ == "__main__":
    main() 