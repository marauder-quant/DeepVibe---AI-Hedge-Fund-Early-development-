"""
S&P 500 Stock Grading based on Economic Quadrant
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from economic_quadrant import determine_economic_quadrant
# Import database functions
from db_utils import save_stock_analysis

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

def get_stock_data(tickers, period="2y"):
    """
    Get financial data for a list of tickers
    
    Parameters:
    - tickers: list of stock tickers
    - period: time period for data (default: 2 years)
    
    Returns:
    - DataFrame with stock data
    """
    # Use a smaller subset for testing
    if len(tickers) > 50:
        print(f"Using first 50 tickers for testing...")
        test_tickers = tickers[:50]
    else:
        test_tickers = tickers
    
    # Get stock data
    stock_data = {}
    
    for ticker in test_tickers:
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
    Grade stocks based on financial metrics and the current economic quadrant
    
    Parameters:
    - stock_data: DataFrame with stock data
    - economic_quadrant: current economic quadrant (A, B, C, or D)
    
    Returns:
    - graded_stocks: DataFrame with graded stock data
    """
    # Create a copy of the stock data DataFrame
    graded_stocks = stock_data.copy()
    
    # Initialize columns for grades
    graded_stocks['revenue_grade'] = None
    graded_stocks['earnings_grade'] = None
    graded_stocks['pe_grade'] = None
    graded_stocks['debt_grade'] = None
    graded_stocks['overall_grade'] = None
    
    # Grade each stock
    for idx, stock in graded_stocks.iterrows():
        # Revenue Growth Grade
        if pd.notnull(stock['revenue_growth']):
            if stock['revenue_growth'] > 0.2:  # >20%
                graded_stocks.loc[idx, 'revenue_grade'] = 'A'
            elif stock['revenue_growth'] > 0.1:  # >10%
                graded_stocks.loc[idx, 'revenue_grade'] = 'B'
            elif stock['revenue_growth'] > 0.05:  # >5%
                graded_stocks.loc[idx, 'revenue_grade'] = 'C'
            elif stock['revenue_growth'] > 0:  # >0%
                graded_stocks.loc[idx, 'revenue_grade'] = 'D'
            else:  # negative
                graded_stocks.loc[idx, 'revenue_grade'] = 'F'
        
        # Earnings Growth Grade
        if pd.notnull(stock['earnings_growth']):
            if stock['earnings_growth'] > 0.2:  # >20%
                graded_stocks.loc[idx, 'earnings_grade'] = 'A'
            elif stock['earnings_growth'] > 0.1:  # >10%
                graded_stocks.loc[idx, 'earnings_grade'] = 'B'
            elif stock['earnings_growth'] > 0.05:  # >5%
                graded_stocks.loc[idx, 'earnings_grade'] = 'C'
            elif stock['earnings_growth'] > 0:  # >0%
                graded_stocks.loc[idx, 'earnings_grade'] = 'D'
            else:  # negative
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
            
            # For quadrant B, prioritize stocks with B characteristics
            elif economic_quadrant == 'B':
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
            
            # For quadrant C, prioritize stocks with C characteristics
            elif economic_quadrant == 'C':
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

def grade_stock(symbol, data=None):
    """
    Grade individual stock based on financial metrics
    """
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        
        # Get financials
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get price history
        if data is None:
            data = stock.history(period="2y")
        
        # Get basic info
        info = stock.info
        
        # Initialize scores dictionary
        all_scores = {}
        
        # Skip if missing critical data
        if income_stmt.empty or balance_sheet.empty or data.empty:
            print(f"Skipping {symbol}: Missing financial data")
            return None, None
        
        # Calculate metrics
        # 1. Momentum Score
        momentum_score = 0
        
        # Price Trend (positive slope = 1, negative = 0)
        recent_prices = data['Close'].tail(100)
        if not recent_prices.empty:
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices.values, 1)
            momentum_score += 1 if slope > 0 else 0
        
        # Moving Average (price > 200ma = 1, else 0)
        if len(data) > 200:
            ma_200 = data['Close'].rolling(window=200).mean()
            current_price = data['Close'].iloc[-1]
            current_ma = ma_200.iloc[-1]
            momentum_score += 1 if current_price > current_ma else 0
        
        # Normalize to 0-100
        momentum_score = momentum_score * 50
        all_scores['momentum'] = momentum_score
        
        # 2. Value Score
        value_score = 0
        
        # P/E Ratio Score (lower is better)
        pe_ratio = info.get('forwardPE', None)
        if pe_ratio is not None and pe_ratio > 0:
            if pe_ratio <= 10:
                value_score += 2
            elif pe_ratio <= 20:
                value_score += 1
        
        # Debt to EBITDA Score (lower is better)
        if 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            ebitda = income_stmt.loc['EBITDA'].iloc[0]
            debt_to_ebitda = total_debt / ebitda if ebitda != 0 else float('inf')
            
            if debt_to_ebitda <= 1.0:
                value_score += 2
            elif debt_to_ebitda <= 2.0:
                value_score += 1
        
        # Normalize to 0-100
        value_score = value_score * 25
        all_scores['value'] = value_score
        
        # 3. Growth Score
        growth_score = 0
        
        # Revenue Growth
        if 'Total Revenue' in income_stmt.index:
            latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
            prev_revenue = income_stmt.loc['Total Revenue'].iloc[1] if len(income_stmt.loc['Total Revenue']) > 1 else 0
            revenue_growth = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else 0
            
            if revenue_growth > 0.2:  # >20%
                growth_score += 2
            elif revenue_growth > 0.1:  # >10%
                growth_score += 1
        
        # Earnings Growth
        if 'Net Income' in income_stmt.index:
            latest_earnings = income_stmt.loc['Net Income'].iloc[0]
            prev_earnings = income_stmt.loc['Net Income'].iloc[1] if len(income_stmt.loc['Net Income']) > 1 else 0
            earnings_growth = (latest_earnings - prev_earnings) / prev_earnings if prev_earnings != 0 else 0
            
            if earnings_growth > 0.1:  # >10%
                growth_score += 2
            elif earnings_growth > 0.05:  # >5%
                growth_score += 1
        
        # Normalize to 0-100
        growth_score = growth_score * 25
        all_scores['growth'] = growth_score
        
        # 4. Quality Score
        quality_score = 0
        
        # Return on Equity
        if 'Return On Equity' in info:
            roe = info['Return On Equity']
            if roe > 0.2:  # >20%
                quality_score += 2
            elif roe > 0.1:  # >10%
                quality_score += 1
        
        # Profit Margin
        if 'profitMargins' in info:
            profit_margin = info['profitMargins']
            if profit_margin > 0.2:  # >20%
                quality_score += 2
            elif profit_margin > 0.1:  # >10%
                quality_score += 1
        
        # Normalize to 0-100
        quality_score = quality_score * 25
        all_scores['quality'] = quality_score
        
        # Calculate overall score
        overall_score = momentum_score * 0.1 + value_score * 0.3 + growth_score * 0.3 + quality_score * 0.3
        all_scores['overall'] = overall_score
        
        # Assign grade based on overall score
        if overall_score >= 80:
            grade = 'A+'
        elif overall_score >= 70:
            grade = 'A'
        elif overall_score >= 60:
            grade = 'B'
        elif overall_score >= 50:
            grade = 'C'
        elif overall_score >= 40:
            grade = 'D'
        else:
            grade = 'F'
            
        # Get current economic quadrant
        current_quadrant, _, _ = determine_economic_quadrant()
        
        # Save results to database
        try:
            # Extract key metrics for database storage
            latest_price = data['Close'].iloc[-1] if not data.empty and 'Close' in data else None
            latest_volume = data['Volume'].iloc[-1] if not data.empty and 'Volume' in data else None
            
            pe_ratio = info.get('forwardPE', None)
            
            # Calculate debt to EBITDA
            if 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                debt_to_ebitda = total_debt / ebitda if ebitda != 0 else None
            else:
                debt_to_ebitda = None
            
            # Calculate revenue growth
            if 'Total Revenue' in income_stmt.index and len(income_stmt.loc['Total Revenue']) > 1:
                latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                prev_revenue = income_stmt.loc['Total Revenue'].iloc[1]
                revenue_growth = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else None
            else:
                revenue_growth = None
            
            # Calculate earnings growth
            if 'Net Income' in income_stmt.index and len(income_stmt.loc['Net Income']) > 1:
                latest_earnings = income_stmt.loc['Net Income'].iloc[0]
                prev_earnings = income_stmt.loc['Net Income'].iloc[1]
                earnings_growth = (latest_earnings - prev_earnings) / prev_earnings if prev_earnings != 0 else None
            else:
                earnings_growth = None
            
            # Save to database
            save_stock_analysis(
                symbol=symbol,
                name=info.get('shortName', symbol),
                sector=info.get('sector', 'Unknown'),
                grade=grade,
                price=latest_price,
                volume=latest_volume,
                momentum_score=momentum_score,
                value_score=value_score,
                growth_score=growth_score,
                quality_score=quality_score,
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                pe_ratio=pe_ratio,
                debt_to_ebitda=debt_to_ebitda,
                batch_name="individual_analysis",
                quadrant=current_quadrant,
                notes=f"Stock {symbol} analyzed in quadrant {current_quadrant}",
                json_data=all_scores
            )
        except Exception as e:
            print(f"Error saving {symbol} to database: {e}")
        
        return grade, all_scores
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None, None

def analyze_stocks_for_quadrant():
    """
    Analyze all S&P 500 stocks based on current economic quadrant
    
    Returns:
    - graded_stocks: DataFrame with all graded stocks
    - top_stocks: DataFrame with top-graded stocks
    """
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    
    # Get current economic quadrant
    quadrant, balance_sheet_trend, interest_rate_level = determine_economic_quadrant()
    print(f"Current Economic Quadrant: {quadrant}")
    print(f"Balance Sheet Trend: {balance_sheet_trend}")
    print(f"Interest Rate Level: {interest_rate_level}")
    
    # Get stock data
    stock_data = get_stock_data(tickers)
    
    # Grade stocks
    graded_stocks = grade_stocks(stock_data, quadrant)
    
    # Filter top stocks (A+ and A grades)
    top_stocks = graded_stocks[graded_stocks['overall_grade'].isin(['A+', 'A'])]
    
    # Save results to CSV
    graded_stocks.to_csv('graded_stocks.csv')
    top_stocks.to_csv('top_stocks.csv')
    
    # Save results to database in batch
    try:
        for idx, stock in graded_stocks.iterrows():
            save_stock_analysis(
                symbol=idx,
                name=stock.get('name', idx),
                sector=stock.get('sector', 'Unknown'),
                grade=stock.get('overall_grade', None),
                price=None,  # Price data not available in this context
                volume=None,  # Volume data not available in this context
                momentum_score=None,
                value_score=None,
                growth_score=None,
                quality_score=None,
                revenue_growth=stock.get('revenue_growth', None),
                earnings_growth=stock.get('earnings_growth', None),
                pe_ratio=stock.get('pe_ratio', None),
                debt_to_ebitda=stock.get('debt_to_ebitda', None),
                batch_name="full_analysis",
                quadrant=quadrant,
                notes=f"Stock {idx} analyzed in batch for quadrant {quadrant}",
                json_data=stock.to_dict()
            )
    except Exception as e:
        print(f"Error saving batch analysis to database: {e}")
    
    return graded_stocks, top_stocks

if __name__ == "__main__":
    print("Analyzing S&P 500 stocks based on current economic quadrant...")
    graded_stocks, top_stocks = analyze_stocks_for_quadrant()
    
    print("\nTop 20 Stocks:")
    print(top_stocks[['name', 'sector', 'revenue_growth', 'earnings_growth', 'pe_ratio', 'debt_to_ebitda', 'overall_grade']])
