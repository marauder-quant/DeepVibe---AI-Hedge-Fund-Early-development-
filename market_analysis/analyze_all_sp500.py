"""
Unified S&P 500 Stock Analysis and Grading
This script analyzes all S&P 500 stocks at once, breaking them into batches internally
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from economic_quadrant import determine_economic_quadrant
from db_utils import save_stock_analysis, get_db_stats

# Define batch size for processing
BATCH_SIZE = 50

def get_sp500_tickers():
    """
    Get list of S&P 500 tickers
    """
    print("Getting S&P 500 tickers...")
    try:
        # Use Wikipedia to get S&P 500 tickers
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers (remove dots and convert to uppercase)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Found {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"Error getting S&P 500 tickers: {e}")
        return []

def get_stock_data(ticker):
    """
    Get financial data for a single ticker
    
    Parameters:
    - ticker: stock ticker symbol
    
    Returns:
    - dict with stock data or None if error
    """
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
            return {
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'pe_ratio': pe_ratio,
                'debt_to_ebitda': debt_to_ebitda,
                'price': info.get('regularMarketPrice', None),
                'volume': info.get('regularMarketVolume', None)
            }
        else:
            print(f"Skipping {ticker}: Missing financial data")
            return None
                
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def grade_stocks(stock_data, economic_quadrant):
    """
    Grade a stock based on economic quadrant and financial metrics
    
    Parameters:
    - stock_data: dict with stock data
    - economic_quadrant: current economic quadrant (A, B/C, or D)
    
    Returns:
    - dict with graded stock data including overall grade
    """
    graded_stock = stock_data.copy()
    
    # Initialize grades
    revenue_grade = None
    earnings_grade = None
    pe_grade = None
    debt_grade = None
    
    # Grade Revenue Growth
    if graded_stock['revenue_growth'] is not None:
        if graded_stock['revenue_growth'] > 0.5:  # >50%
            revenue_grade = 'D'
        elif graded_stock['revenue_growth'] > 0.2:  # >20%
            revenue_grade = 'C'
        elif graded_stock['revenue_growth'] > 0.1:  # >10%
            revenue_grade = 'B'
        elif graded_stock['revenue_growth'] > 0.05:  # >5%
            revenue_grade = 'A'
        else:
            revenue_grade = 'F'
    
    # Grade Earnings Growth
    if graded_stock['earnings_growth'] is not None:
        if graded_stock['earnings_growth'] > 0.1:  # >10%
            earnings_grade = 'B/C'
        elif graded_stock['earnings_growth'] > 0.05:  # >5%
            earnings_grade = 'A'
        else:
            earnings_grade = 'F'
    
    # Grade P/E Ratio
    if graded_stock['pe_ratio'] is not None and graded_stock['pe_ratio'] > 0:
        if graded_stock['pe_ratio'] <= 10:
            pe_grade = 'A'
        elif graded_stock['pe_ratio'] <= 20:
            pe_grade = 'B'
        elif graded_stock['pe_ratio'] <= 25:
            pe_grade = 'C'
        else:
            pe_grade = 'F'
    
    # Grade Debt to EBITDA
    if graded_stock['debt_to_ebitda'] is not None:
        if graded_stock['debt_to_ebitda'] <= 1.0:
            debt_grade = 'A'
        elif graded_stock['debt_to_ebitda'] <= 2.0:
            debt_grade = 'B'
        elif graded_stock['debt_to_ebitda'] <= 5.0:
            debt_grade = 'C'
        else:
            debt_grade = 'D'
    
    # Add grades to stock data
    graded_stock['revenue_grade'] = revenue_grade
    graded_stock['earnings_grade'] = earnings_grade
    graded_stock['pe_grade'] = pe_grade
    graded_stock['debt_grade'] = debt_grade
    
    # Calculate overall grade based on economic quadrant
    grades = []
    if revenue_grade:
        grades.append(revenue_grade)
    if earnings_grade:
        grades.append(earnings_grade)
    if pe_grade:
        grades.append(pe_grade)
    if debt_grade:
        grades.append(debt_grade)
    
    overall_grade = None
    
    if len(grades) > 0:
        # For quadrant A, prioritize stocks with A characteristics
        if economic_quadrant == 'A':
            # Count A grades
            a_count = grades.count('A')
            # Prioritize stocks with more A grades
            if a_count >= 3:
                overall_grade = 'A+'
            elif a_count >= 2:
                overall_grade = 'A'
            elif a_count >= 1:
                overall_grade = 'B'
            else:
                overall_grade = 'C'
        
        # For quadrant B/C (prefer B), prioritize stocks with B characteristics
        elif economic_quadrant == 'B/C (prefer B)':
            # Count B grades
            b_count = grades.count('B')
            # Prioritize stocks with more B grades
            if b_count >= 3:
                overall_grade = 'A+'
            elif b_count >= 2:
                overall_grade = 'A'
            elif b_count >= 1:
                overall_grade = 'B'
            else:
                overall_grade = 'C'
        
        # For quadrant B/C (prefer C), prioritize stocks with C characteristics
        elif economic_quadrant == 'B/C (prefer C)':
            # Count C grades
            c_count = grades.count('C')
            # Prioritize stocks with more C grades
            if c_count >= 3:
                overall_grade = 'A+'
            elif c_count >= 2:
                overall_grade = 'A'
            elif c_count >= 1:
                overall_grade = 'B'
            else:
                overall_grade = 'C'
        
        # For quadrant D, prioritize stocks with D characteristics
        elif economic_quadrant == 'D':
            # Count D grades
            d_count = grades.count('D')
            # Prioritize stocks with more D grades
            if d_count >= 2:
                overall_grade = 'A+'
            elif d_count >= 1:
                overall_grade = 'A'
            # Also consider high growth (C grade for revenue)
            elif 'C' in grades:
                overall_grade = 'B'
            else:
                overall_grade = 'C'
    
    graded_stock['overall_grade'] = overall_grade
    return graded_stock

def save_to_database(graded_stocks, batch_name="full_sp500"):
    """
    Save graded stocks to database
    """
    print(f"Saving {len(graded_stocks)} stocks to database...")
    count = 0
    
    for symbol, stock_data in graded_stocks.items():
        try:
            # Save to database
            save_stock_analysis(
                symbol=symbol,
                name=stock_data.get('name'),
                sector=stock_data.get('sector'),
                grade=stock_data.get('overall_grade'),
                price=stock_data.get('price'),
                volume=stock_data.get('volume'),
                momentum_score=None,  # Not calculated
                value_score=None,  # Not calculated
                growth_score=None,  # Not calculated
                quality_score=None,  # Not calculated
                revenue_growth=stock_data.get('revenue_growth'),
                earnings_growth=stock_data.get('earnings_growth'),
                pe_ratio=stock_data.get('pe_ratio'),
                debt_to_ebitda=stock_data.get('debt_to_ebitda'),
                batch_name=batch_name,
                quadrant=stock_data.get('quadrant'),
                notes=f"Revenue Grade: {stock_data.get('revenue_grade')}, Earnings Grade: {stock_data.get('earnings_grade')}, PE Grade: {stock_data.get('pe_grade')}, Debt Grade: {stock_data.get('debt_grade')}",
                json_data={
                    'revenue_grade': stock_data.get('revenue_grade'),
                    'earnings_grade': stock_data.get('earnings_grade'),
                    'pe_grade': stock_data.get('pe_grade'),
                    'debt_grade': stock_data.get('debt_grade')
                }
            )
            count += 1
        except Exception as e:
            print(f"Error saving {symbol} to database: {e}")
    
    print(f"Successfully saved {count} stocks to database")
    return count

def analyze_sp500():
    """
    Main function to analyze all S&P 500 stocks
    """
    start_time = time.time()
    print("Starting full S&P 500 analysis...")
    
    # Determine current economic quadrant
    quadrant, balance_sheet_trend, interest_rate_level = determine_economic_quadrant()
    print(f"Current Economic Quadrant: {quadrant}")
    print(f"Balance Sheet Trend: {balance_sheet_trend}")
    print(f"Interest Rate Level: {interest_rate_level}")
    
    # Get all S&P 500 tickers
    tickers = get_sp500_tickers()
    if not tickers:
        print("Failed to retrieve S&P 500 tickers. Exiting.")
        return None
    
    # Process tickers in batches
    all_graded_stocks = {}
    total_processed = 0
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch_start = i
        batch_end = min(i + BATCH_SIZE, len(tickers))
        batch_tickers = tickers[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(tickers)+BATCH_SIZE-1)//BATCH_SIZE}: tickers {batch_start} to {batch_end-1}")
        
        batch_stocks = {}
        for ticker in batch_tickers:
            print(f"Processing {ticker}...")
            stock_data = get_stock_data(ticker)
            if stock_data:
                # Add quadrant information
                stock_data['quadrant'] = quadrant
                # Grade the stock
                graded_stock = grade_stocks(stock_data, quadrant)
                batch_stocks[ticker] = graded_stock
                total_processed += 1
                print(f"Graded {ticker} ({stock_data['name']}): {graded_stock['overall_grade']}")
            
            # Small delay to avoid API rate limits
            time.sleep(0.1)
        
        # Save batch to database
        save_to_database(batch_stocks, batch_name=f"batch_{batch_start}_{batch_end}")
        
        # Add to overall results
        all_graded_stocks.update(batch_stocks)
        
        print(f"Processed {len(batch_stocks)} stocks in batch {batch_start//BATCH_SIZE + 1} ({total_processed} total)")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nAnalysis complete in {execution_time:.2f} seconds")
    print(f"Processed {total_processed} stocks out of {len(tickers)} total S&P 500 stocks")
    
    # Get database statistics
    db_stats = get_db_stats()
    print("\nDatabase Statistics:")
    print(f"Total stock analyses: {db_stats['stock_analysis_count']}")
    print(f"Total economic quadrant analyses: {db_stats['economic_quadrant_count']}")
    print(f"Analysis batches: {', '.join(db_stats['batches'])}")
    
    # Create grade distribution summary
    grade_distribution = db_stats['grade_distribution']
    print("\nGrade Distribution:")
    for grade, count in grade_distribution.items():
        print(f"{grade}: {count} stocks")
    
    return all_graded_stocks

if __name__ == "__main__":
    # Run the analysis
    all_graded_stocks = analyze_sp500()
    
    # Print a summary of top-graded stocks
    if all_graded_stocks:
        top_stocks = {symbol: data for symbol, data in all_graded_stocks.items() 
                     if data['overall_grade'] in ['A+', 'A']}
        
        print(f"\nFound {len(top_stocks)} top-graded stocks (A+ or A):")
        for symbol, data in top_stocks.items():
            print(f"{symbol} ({data['name']}): {data['overall_grade']} - {data['sector']}") 