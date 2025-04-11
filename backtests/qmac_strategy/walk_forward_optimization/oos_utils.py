#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for QMAC strategy out-of-sample testing.
"""

import random
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pytz
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import logging

# Console setup for rich output
console = Console()
log = logging.getLogger("rich")

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
        # Import here to avoid circular imports
        from backtests.qmac_strategy.src.lib.strategy_core import run_qmac_strategy
        
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
        
        # Check if result contains required keys
        if 'qmac_pf' not in result:
            print(f"Missing portfolio data for {symbol} from {start_date} to {end_date}")
            return None
        
        # Get performance metrics from QMAC strategy
        qmac_pf = result['qmac_pf']
        total_return = qmac_pf.total_return()
        
        # Get buy and hold performance metrics
        if 'hold_pf' in result:
            # Use the hold portfolio provided by run_qmac_strategy
            hold_pf = result['hold_pf']
            buy_and_hold_return = hold_pf.total_return()
            buy_and_hold_drawdown = hold_pf.stats()['Max Drawdown [%]'] / 100
            print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (from strategy), drawdown: {buy_and_hold_drawdown:.2%}")
        else:
            # Fallback method using the price data directly
            try:
                if 'ohlcv' in result and len(result['ohlcv']) > 1:
                    # Calculate from price data
                    data = result['ohlcv']
                    first_price = data['Close'].iloc[0]
                    last_price = data['Close'].iloc[-1]
                    buy_and_hold_return = (last_price / first_price) - 1
                    
                    # Calculate drawdown for buy and hold
                    equity_curve = data['Close'] / first_price
                    peak = equity_curve.cummax()
                    drawdown = (equity_curve - peak) / peak
                    buy_and_hold_drawdown = abs(drawdown.min())
                    
                    print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (calculated from prices), drawdown: {buy_and_hold_drawdown:.2%}")
                else:
                    # No price data available, use a benchmark estimate
                    period_days = (end_date - start_date).days
                    buy_and_hold_return = 0.08 / 365 * period_days  # Annualized 8% return
                    buy_and_hold_drawdown = 0.03  # Default 3% drawdown
                    print(f"{symbol} buy & hold: {buy_and_hold_return:.2%} (estimated), drawdown: {buy_and_hold_drawdown:.2%}")
            except Exception as e:
                print(f"Error calculating buy & hold from prices for {symbol}: {e}")
                buy_and_hold_return = 0.01  # Default 1% return
                buy_and_hold_drawdown = 0.03  # Default 3% drawdown
        
        # Calculate alpha (strategy return - buy and hold return)
        alpha = total_return - buy_and_hold_return
        
        # Calculate theta (buy and hold drawdown - strategy drawdown)
        strategy_drawdown = qmac_pf.stats()['Max Drawdown [%]'] / 100
        theta = buy_and_hold_drawdown - strategy_drawdown
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'total_return': total_return,
            'sharpe_ratio': qmac_pf.stats()['Sharpe Ratio'],
            'max_drawdown': strategy_drawdown,
            'n_trades': len(qmac_pf.trades),
            'buy_and_hold_return': buy_and_hold_return,
            'buy_and_hold_drawdown': buy_and_hold_drawdown,
            'alpha': alpha,
            'theta': theta
        }
    except Exception as e:
        print(f"Error testing {symbol} from {start_date} to {end_date}: {e}")
        return None

def display_live_confidence_update(latest_result, test_num, current_summary, overall_confidence):
    """Display a live update of confidence metrics after each test."""
    # Create a mini table for the latest test result
    latest_table = Table(title=f"Test #{test_num} Result: {latest_result['symbol']}")
    latest_table.add_column("Metric", style="cyan")
    latest_table.add_column("Value", style="green")
    
    latest_table.add_row("Return", f"{latest_result['total_return']:.2%}")
    latest_table.add_row("Buy & Hold", f"{latest_result['buy_and_hold_return']:.2%}")
    latest_table.add_row("Alpha", f"{latest_result['alpha']:.2%}")
    latest_table.add_row("Drawdown", f"{latest_result['max_drawdown']:.2%}")
    latest_table.add_row("B&H Drawdown", f"{latest_result['buy_and_hold_drawdown']:.2%}")
    latest_table.add_row("Theta", f"{latest_result['theta']:.2%}")
    
    # Create a summary table
    summary_table = Table(title="Running Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Avg Return", f"{current_summary['avg_return']:.2%}")
    summary_table.add_row("Avg Alpha", f"{current_summary['avg_alpha']:.2%}")
    summary_table.add_row("Alpha Success", f"{current_summary['alpha_success_rate']:.2%}")
    summary_table.add_row("Avg Theta", f"{current_summary['avg_theta']:.2%}")
    summary_table.add_row("Theta Success", f"{current_summary['theta_success_rate']:.2%}")
    summary_table.add_row("Current Confidence", f"{overall_confidence:.2%}")
    
    # Choose color based on confidence level
    confidence_color = "[green]" if overall_confidence >= 0.7 else "[yellow]" if overall_confidence >= 0.4 else "[red]"
    
    # Display tables side by side
    console.print(Panel.fit(
        Columns([latest_table, summary_table]), 
        title=f"{confidence_color}Confidence Update after {test_num} tests"
    ))

def get_latest_in_sample_performance(timeframe):
    """Get the latest in-sample performance metrics."""
    import os
    import pandas as pd
    
    results_dir = 'backtests/qmac_strategy/results'
    # Look for in-sample result files with this timeframe
    csv_files = [f for f in os.listdir(results_dir) 
                if f.endswith('.csv') and f'qmac_results_SPY_{timeframe}' in f]
    
    if not csv_files:
        log.warning(f"No in-sample results found for timeframe {timeframe}")
        return {'total_return': 0}
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    results_df = pd.read_csv(os.path.join(results_dir, latest_file))
    
    # Get the best parameters
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    
    # Create result dict with available columns
    result = {'total_return': best_row['total_return']}
    
    # Add optional metrics if they exist in the dataframe
    for metric in ['sharpe_ratio', 'max_drawdown']:
        if metric in best_row:
            result[metric] = best_row[metric]
    
    return result 