"""
Analyze more S&P 500 stocks based on economic quadrant
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from economic_quadrant import determine_economic_quadrant
import time

def get_sp500_tickers():
    """
    Get list of S&P 500 tickers
    """
    # Use Wikipedia to get S&P 500 tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    
    # Clean tickers (remove dots and convert to uppercase)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    
    return tickers

def get_stock_data(tickers, start_idx=0, end_idx=100):
    """
    Get financial data for a list of tickers
    
    Parameters:
    - tickers: list of stock tickers
    - start_idx: starting index for processing
    - end_idx: ending index for processing
    
    Returns:
    - DataFrame with stock data
    """
    # Get the subset of tickers to process
    subset_tickers = tickers[start_idx:end_idx]
    print(f"Processing tickers {start_idx} to {end_idx-1} (total: {len(subset_tickers)})")
    
    # Get stock data
    stock_data = {}
    
    for ticker in subset_tickers:
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            
            # Get financials
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get basic info
            info = stock.info
            
            # Calculate metrics
            if not income_stmt.empty and not balance_sheet.empty:
                # Revenue Growth
                if 'Total Revenue' in income_stmt.index:
                    latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    prev_revenue = income_stmt.loc['Total Revenue'].iloc[1] if len(income_stmt.loc['Total Revenue']) > 1 else 0
                    revenue_growth = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else 0
                else:
                    revenue_growth = None
                
                # Earnings Growth
                if 'Net Income' in income_stmt.index:
                    latest_earnings = income_stmt.loc['Net Income'].iloc[0]
                    prev_earnings = income_stmt.loc['Net Income'].iloc[1] if len(income_stmt.loc['Net Income']) > 1 else 0
                    earnings_growth = (latest_earnings - prev_earnings) / prev_earnings if prev_earnings != 0 else 0
                else:
                    earnings_growth = None
                
                # P/E Ratio
                pe_ratio = info.get('forwardPE', None)
                
                # Debt to EBITDA
                if 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
                    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                    ebitda = income_stmt.loc['EBITDA'].iloc[0] if 'EBITDA' in income_stmt.index else 0
                    debt_to_ebitda = total_debt / ebitda if ebitda != 0 else float('inf')
                else:
                    debt_to_ebitda = None
                
                # Store data
                stock_data[ticker] = {
                    'name': info.get('shortName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth,
                    'pe_ratio': pe_ratio,
                    'debt_to_ebitda': debt_to_ebitda
                }
                
                print(f"Processed {ticker}: {stock_data[ticker]['name']}")
            else:
                print(f"Skipping {ticker}: Missing financial data")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(stock_data, orient='index')
    
    return df

def grade_stocks(stock_data, economic_quadrant):
    """
    Grade stocks based on economic quadrant
    
    Parameters:
    - stock_data: DataFrame with stock data
    - economic_quadrant: current economic quadrant (A, B/C, or D)
    
    Returns:
    - DataFrame with graded stocks
    """
    # Create a copy of the data
    graded_stocks = stock_data.copy()
    
    # Add grade columns
    graded_stocks['revenue_grade'] = None
    graded_stocks['earnings_grade'] = None
    graded_stocks['pe_grade'] = None
    graded_stocks['debt_grade'] = None
    graded_stocks['overall_grade'] = None
    
    # Grade based on criteria from the video
    for idx, stock in graded_stocks.iterrows():
        # Revenue Growth grading
        if stock['revenue_growth'] is not None:
            if stock['revenue_growth'] > 0.5:  # >50%
                graded_stocks.loc[idx, 'revenue_grade'] = 'D'
            elif stock['revenue_growth'] > 0.2:  # >20%
                graded_stocks.loc[idx, 'revenue_grade'] = 'C'
            elif stock['revenue_growth'] > 0.1:  # >10%
                graded_stocks.loc[idx, 'revenue_grade'] = 'B'
            elif stock['revenue_growth'] > 0.05:  # >5%
                graded_stocks.loc[idx, 'revenue_grade'] = 'A'
            else:
                graded_stocks.loc[idx, 'revenue_grade'] = 'F'
        
        # Earnings Growth grading
        if stock['earnings_growth'] is not None:
            if stock['earnings_growth'] > 0.1:  # >10%
                graded_stocks.loc[idx, 'earnings_grade'] = 'B/C'
            elif stock['earnings_growth'] > 0.05:  # >5%
                graded_stocks.loc[idx, 'earnings_grade'] = 'A'
            else:
                graded_stocks.loc[idx, 'earnings_grade'] = 'F'
        
        # P/E Ratio grading
        if stock['pe_ratio'] is not None and stock['pe_ratio'] > 0:
            if stock['pe_ratio'] <= 10:
                graded_stocks.loc[idx, 'pe_grade'] = 'A'
            elif stock['pe_ratio'] <= 20:
                graded_stocks.loc[idx, 'pe_grade'] = 'B'
            elif stock['pe_ratio'] <= 25:
                graded_stocks.loc[idx, 'pe_grade'] = 'C'
            else:
                graded_stocks.loc[idx, 'pe_grade'] = 'F'
        
        # Debt to EBITDA grading
        if stock['debt_to_ebitda'] is not None:
            if stock['debt_to_ebitda'] <= 1.0:
                graded_stocks.loc[idx, 'debt_grade'] = 'A'
            elif stock['debt_to_ebitda'] <= 2.0:
                graded_stocks.loc[idx, 'debt_grade'] = 'B'
            elif stock['debt_to_ebitda'] <= 5.0:
                graded_stocks.loc[idx, 'debt_grade'] = 'C'
            else:
                graded_stocks.loc[idx, 'debt_grade'] = 'D'
    
    # Calculate overall grade based on economic quadrant
    for idx, stock in graded_stocks.iterrows():
        grades = []
        if stock['revenue_grade'] is not None:
            grades.append(stock['revenue_grade'])
        if stock['earnings_grade'] is not None:
            grades.append(stock['earnings_grade'])
        if stock['pe_grade'] is not None:
            grades.append(stock['pe_grade'])
        if stock['debt_grade'] is not None:
            grades.append(stock['debt_grade'])
        
        if len(grades) > 0:
            # For quadrant A, prioritize stocks with A characteristics
            if economic_quadrant == 'A':
                # Count A grades
                a_count = grades.count('A')
                # Prioritize stocks with more A grades
                if a_count >= 3:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A+'
                elif a_count >= 2:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A'
                elif a_count >= 1:
                    graded_stocks.loc[idx, 'overall_grade'] = 'B'
                else:
                    graded_stocks.loc[idx, 'overall_grade'] = 'C'
            
            # For quadrant B/C (prefer B), prioritize stocks with B characteristics
            elif economic_quadrant == 'B/C (prefer B)':
                # Count B grades
                b_count = grades.count('B')
                # Prioritize stocks with more B grades
                if b_count >= 3:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A+'
                elif b_count >= 2:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A'
                elif b_count >= 1:
                    graded_stocks.loc[idx, 'overall_grade'] = 'B'
                else:
                    graded_stocks.loc[idx, 'overall_grade'] = 'C'
            
            # For quadrant B/C (prefer C), prioritize stocks with C characteristics
            elif economic_quadrant == 'B/C (prefer C)':
                # Count C grades
                c_count = grades.count('C')
                # Prioritize stocks with more C grades
                if c_count >= 3:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A+'
                elif c_count >= 2:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A'
                elif c_count >= 1:
                    graded_stocks.loc[idx, 'overall_grade'] = 'B'
                else:
                    graded_stocks.loc[idx, 'overall_grade'] = 'C'
            
            # For quadrant D, prioritize stocks with D characteristics
            elif economic_quadrant == 'D':
                # Count D grades
                d_count = grades.count('D')
                # Prioritize stocks with more D grades
                if d_count >= 2:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A+'
                elif d_count >= 1:
                    graded_stocks.loc[idx, 'overall_grade'] = 'A'
                # Also consider high growth (C grade for revenue)
                elif 'C' in grades:
                    graded_stocks.loc[idx, 'overall_grade'] = 'B'
                else:
                    graded_stocks.loc[idx, 'overall_grade'] = 'C'
    
    # Sort by overall grade
    grade_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'F': 4, None: 5}
    graded_stocks['grade_order'] = graded_stocks['overall_grade'].map(grade_order)
    graded_stocks = graded_stocks.sort_values('grade_order')
    graded_stocks = graded_stocks.drop('grade_order', axis=1)
    
    return graded_stocks

def create_visualization(graded_stocks, batch_num):
    """
    Create visualizations of the stock grading results
    
    Parameters:
    - graded_stocks: DataFrame with graded stocks
    - batch_num: batch number for file naming
    
    Returns:
    - List of visualization file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Sector distribution of top stocks
    top_stocks = graded_stocks[graded_stocks['overall_grade'].isin(['A+', 'A'])]
    
    if not top_stocks.empty:
        sector_counts = top_stocks['sector'].value_counts()
        
        plt.figure(figsize=(12, 8))
        sector_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Sector Distribution of Top-Graded Stocks (Batch {batch_num})')
        plt.xlabel('Sector')
        plt.ylabel('Number of Stocks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'visualizations/top_stocks_by_sector_batch{batch_num}.png')
        
    # 2. Grade distribution
    grade_counts = graded_stocks['overall_grade'].value_counts()
    
    plt.figure(figsize=(10, 6))
    grade_counts.plot(kind='bar', color='lightgreen')
    plt.title(f'Distribution of Overall Grades (Batch {batch_num})')
    plt.xlabel('Grade')
    plt.ylabel('Number of Stocks')
    plt.tight_layout()
    plt.savefig(f'visualizations/grade_distribution_batch{batch_num}.png')
    
    # 3. Debt to EBITDA vs PE Ratio scatter plot for top stocks
    if not top_stocks.empty and 'debt_to_ebitda' in top_stocks.columns and 'pe_ratio' in top_stocks.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        plot_data = top_stocks.dropna(subset=['debt_to_ebitda', 'pe_ratio'])
        
        if not plot_data.empty:
            plt.scatter(plot_data['debt_to_ebitda'], plot_data['pe_ratio'], alpha=0.7)
            
            # Add labels for each point
            for idx, row in plot_data.iterrows():
                plt.annotate(idx, (row['debt_to_ebitda'], row['pe_ratio']), 
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title(f'Debt to EBITDA vs P/E Ratio for Top-Graded Stocks (Batch {batch_num})')
            plt.xlabel('Debt to EBITDA')
            plt.ylabel('P/E Ratio')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'visualizations/debt_vs_pe_batch{batch_num}.png')
    
    # 4. Revenue Growth vs Earnings Growth scatter plot
    if not top_stocks.empty and 'revenue_growth' in top_stocks.columns and 'earnings_growth' in top_stocks.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        plot_data = top_stocks.dropna(subset=['revenue_growth', 'earnings_growth'])
        
        if not plot_data.empty:
            plt.scatter(plot_data['revenue_growth'], plot_data['earnings_growth'], alpha=0.7)
            
            # Add labels for each point
            for idx, row in plot_data.iterrows():
                plt.annotate(idx, (row['revenue_growth'], row['earnings_growth']), 
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title(f'Revenue Growth vs Earnings Growth for Top-Graded Stocks (Batch {batch_num})')
            plt.xlabel('Revenue Growth')
            plt.ylabel('Earnings Growth')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'visualizations/growth_metrics_batch{batch_num}.png')
    
    return [
        f'visualizations/top_stocks_by_sector_batch{batch_num}.png', 
        f'visualizations/grade_distribution_batch{batch_num}.png',
        f'visualizations/debt_vs_pe_batch{batch_num}.png',
        f'visualizations/growth_metrics_batch{batch_num}.png'
    ]

def analyze_more_stocks(start_idx=50, end_idx=150):
    """
    Analyze more S&P 500 stocks
    
    Parameters:
    - start_idx: starting index for processing
    - end_idx: ending index for processing
    
    Returns:
    - graded_stocks: DataFrame with graded stocks
    - top_stocks: DataFrame with top-graded stocks
    - visualization_paths: list of visualization file paths
    """
    start_time = time.time()
    
    print(f"Starting analysis of S&P 500 stocks from index {start_idx} to {end_idx}...")
    
    # Get current economic quadrant
    quadrant, balance_sheet_trend, interest_rate_level = determine_economic_quadrant()
    print(f"Current Economic Quadrant: {quadrant}")
    print(f"Balance Sheet Trend: {balance_sheet_trend}")
    print(f"Interest Rate Level: {interest_rate_level}")
    
    # Get S&P 500 tickers
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")
    
    # Get stock data for the specified range
    print(f"Getting stock data for tickers {start_idx} to {end_idx}...")
    stock_data = get_stock_data(tickers, start_idx, end_idx)
    
    # Grade stocks
    print("Grading stocks...")
    graded_stocks = grade_stocks(stock_data, quadrant)
    
    # Save results
    batch_num = f"{start_idx}_{end_idx}"
    graded_stocks.to_csv(f'graded_stocks_batch{batch_num}.csv')
    
    # Get top stocks
    top_stocks = graded_stocks[graded_stocks['overall_grade'].isin(['A+', 'A'])].head(20)
    
    # Save top stocks
    top_stocks.to_csv(f'top_stocks_batch{batch_num}.csv')
   
(Content truncated due to size limit. Use line ranges to read in chunks)