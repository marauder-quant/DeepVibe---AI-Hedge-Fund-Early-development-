"""
Unified Russell 2000 Stock Analysis and Grading
This script analyzes all Russell 2000 stocks at once, breaking them into batches internally
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from economic_quadrant import determine_economic_quadrant
from db_utils import save_stock_analysis, get_db_stats
import io # Add io for handling in-memory text
from tqdm.auto import tqdm # Add tqdm for progress bars

# Define batch size for processing
BATCH_SIZE = 50

def get_russell2k_tickers():
    """
    Get list of Russell 2000 tickers by downloading holdings from the iShares IWM ETF.
    Note: The download URL structure or data format might change over time.
    """
    print("Getting Russell 2000 tickers from iShares IWM holdings...")
    
    # Attempt URL without specific date, hoping it defaults to latest
    url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    # Alternative: Use a known recent date if the above fails, e.g. &asOfDate=YYYYMMDD
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    try:
        print(f"Fetching holdings data from {url}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (like 404)
        
        # Use pandas to read the CSV directly from the response content
        # The number of rows to skip might change, typical iShares CSVs have ~9 header rows
        # We also specify 'Ticker' as the column to keep
        csv_content = response.content
        data = pd.read_csv(io.BytesIO(csv_content), skiprows=9, encoding='utf-8', usecols=['Ticker'], keep_default_na=True)
        
        # Clean the data
        tickers = data['Ticker'].dropna().astype(str).tolist()
        
        # Filter out potential non-stock entries sometimes found in holdings (e.g., cash, '-','USD')
        cleaned_tickers = [ticker for ticker in tickers if ticker.strip() and ticker not in ['-', 'USD'] and not ticker.endswith('%') and not 'CASH' in ticker]
        
        # Further cleaning for yfinance compatibility (e.g., BRK.B -> BRK-B)
        final_tickers = [ticker.replace('.', '-') for ticker in cleaned_tickers]
        
        if not final_tickers:
             raise ValueError("No valid tickers extracted from the downloaded CSV. Check skiprows or CSV format.")

        print(f"Found {len(final_tickers)} Russell 2000 tickers from IWM holdings.")
        return final_tickers

    except requests.exceptions.RequestException as e:
        print(f"HTTP Error getting IWM holdings: {e}")
        print("Please check the iShares URL or your network connection.")
        return []
    except pd.errors.ParserError as e:
         print(f"Error parsing CSV data: {e}")
         print("The CSV format might have changed. Consider inspecting the downloaded file or adjusting 'skiprows'.")
         return []
    except KeyError as e:
         print(f"Error finding column {e} in CSV.")
         print("The CSV column names might have changed. Expected 'Ticker'.")
         return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching/processing tickers: {e}")
        return []

def get_stock_data(ticker):
    """
    Get financial data for a single ticker using yfinance.
    (Identical to S&P 500 version)
    
    Parameters:
    - ticker: stock ticker symbol
    
    Returns:
    - dict with stock data or None if error
    """
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        info = stock.info
        
        if not income_stmt.empty and not balance_sheet.empty:
            if 'Total Revenue' in income_stmt.index and len(income_stmt.loc['Total Revenue']) > 1:
                latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                prev_revenue = income_stmt.loc['Total Revenue'].iloc[1]
                revenue_growth = (latest_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else 0
            else: revenue_growth = None
            
            if 'Net Income' in income_stmt.index and len(income_stmt.loc['Net Income']) > 1:
                latest_earnings = income_stmt.loc['Net Income'].iloc[0]
                prev_earnings = income_stmt.loc['Net Income'].iloc[1]
                earnings_growth = (latest_earnings - prev_earnings) / prev_earnings if prev_earnings != 0 else 0
            else: earnings_growth = None
            
            pe_ratio = info.get('forwardPE', None)
            
            if 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                ebitda = income_stmt.loc['EBITDA'].iloc[0] if 'EBITDA' in income_stmt.index else 0
                debt_to_ebitda = total_debt / ebitda if ebitda != 0 else float('inf')
            else: debt_to_ebitda = None
            
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
            print(f"Skipping {ticker}: Missing financial data (income or balance sheet)")
            return None
                
    except Exception as e:
        # More specific error catching for yfinance issues might be useful
        if "No data found" in str(e) or "No fundamentals data found" in str(e):
             print(f"Skipping {ticker}: No data found via yfinance.")
        else:
             print(f"Error processing {ticker} with yfinance: {e}")
        return None

def grade_stocks(stock_data, economic_quadrant):
    """
    Grade a stock based on economic quadrant and financial metrics
    
    Parameters:
    - stock_data: dict with stock data
    - economic_quadrant: current economic quadrant (A, B, C, or D)
    
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
    try:
        if graded_stock['revenue_growth'] is not None:
            if graded_stock['revenue_growth'] > 0.5:
                revenue_grade = 'A'
            elif graded_stock['revenue_growth'] > 0.2:
                revenue_grade = 'B'
            elif graded_stock['revenue_growth'] > 0.1:
                revenue_grade = 'C'
            elif graded_stock['revenue_growth'] > 0.05:
                revenue_grade = 'D'
            else:
                revenue_grade = 'F'
    except (KeyError, TypeError) as e:
        pass
    
    # Grade Earnings Growth
    try:
        if graded_stock['earnings_growth'] is not None:
            if graded_stock['earnings_growth'] > 0.2:
                earnings_grade = 'A'
            elif graded_stock['earnings_growth'] > 0.1:
                earnings_grade = 'B'
            elif graded_stock['earnings_growth'] > 0.05:
                earnings_grade = 'C'
            elif graded_stock['earnings_growth'] > 0:
                earnings_grade = 'D'
            else:
                earnings_grade = 'F'
    except (KeyError, TypeError) as e:
        pass
    
    if graded_stock['pe_ratio'] is not None and graded_stock['pe_ratio'] > 0:
        if graded_stock['pe_ratio'] <= 10: pe_grade = 'A'
        elif graded_stock['pe_ratio'] <= 20: pe_grade = 'B'
        elif graded_stock['pe_ratio'] <= 25: pe_grade = 'C'
        else: pe_grade = 'F'
    
    if graded_stock['debt_to_ebitda'] is not None:
        if graded_stock['debt_to_ebitda'] <= 1.0: debt_grade = 'A'
        elif graded_stock['debt_to_ebitda'] <= 2.0: debt_grade = 'B'
        elif graded_stock['debt_to_ebitda'] <= 5.0: debt_grade = 'C'
        else: debt_grade = 'D'
    
    graded_stock['revenue_grade'] = revenue_grade
    graded_stock['earnings_grade'] = earnings_grade
    graded_stock['pe_grade'] = pe_grade
    graded_stock['debt_grade'] = debt_grade
    
    grades = [g for g in [revenue_grade, earnings_grade, pe_grade, debt_grade] if g]
    overall_grade = None
    
    if grades:
        if economic_quadrant == 'A':
            a_count = grades.count('A')
            if a_count >= 3: overall_grade = 'A+'
            elif a_count >= 2: overall_grade = 'A'
            elif a_count >= 1: overall_grade = 'B'
            else: overall_grade = 'C'
        elif economic_quadrant == 'B':
            b_count = grades.count('B')
            if b_count >= 3: overall_grade = 'A+'
            elif b_count >= 2: overall_grade = 'A'
            elif b_count >= 1: overall_grade = 'B'
            else: overall_grade = 'C'
        elif economic_quadrant == 'C':
            c_count = grades.count('C')
            if c_count >= 3: overall_grade = 'A+'
            elif c_count >= 2: overall_grade = 'A'
            elif c_count >= 1: overall_grade = 'B'
            else: overall_grade = 'C'
        elif economic_quadrant == 'D':
            d_count = grades.count('D')
            if d_count >= 2: overall_grade = 'A+'
            elif d_count >= 1: overall_grade = 'A'
            elif 'C' in grades: overall_grade = 'B'
            else: overall_grade = 'C'
            
    graded_stock['overall_grade'] = overall_grade
    return graded_stock

def save_to_database(graded_stocks, batch_name="full_russell2k"): # Default batch name updated
    """
    Save graded stocks to database
    """
    print(f"Saving {len(graded_stocks)} stocks to database (batch: {batch_name})...")
    count = 0
    
    for symbol, stock_data in graded_stocks.items():
        try:
            save_stock_analysis(
                symbol=symbol,
                name=stock_data.get('name'),
                sector=stock_data.get('sector'),
                grade=stock_data.get('overall_grade'),
                price=stock_data.get('price'),
                volume=stock_data.get('volume'),
                momentum_score=None, value_score=None, growth_score=None, quality_score=None, # Not calculated
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
    
    print(f"Successfully saved {count} stocks from batch {batch_name} to database")
    return count

def analyze_russell2k():
    """
    Main function to analyze all Russell 2000 stocks
    """
    start_time = time.time()
    print("Starting full Russell 2000 analysis...")

    quadrant, balance_sheet_trend, interest_rate_level, details = determine_economic_quadrant()
    print(f"Current Economic Quadrant: {quadrant}")
    print(f"Balance Sheet Trend: {balance_sheet_trend}")
    print(f"Interest Rate Level: {interest_rate_level}")

    tickers = get_russell2k_tickers()
    if not tickers:
        print("Failed to retrieve Russell 2000 tickers. Exiting.")
        return None

    all_graded_stocks = {}
    total_processed = 0 # Track total attempts across all batches
    total_tickers = len(tickers)
    num_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
    analysis_start_time = time.time() # Use a separate timer for the analysis loop

    # Removed overall progress bar creation

    for i in range(0, total_tickers, BATCH_SIZE):
        batch_start_index = i
        batch_end_index = min(i + BATCH_SIZE, total_tickers)
        batch_tickers = tickers[batch_start_index:batch_end_index]
        current_batch_num = batch_start_index // BATCH_SIZE + 1
        batch_identifier = f"russell2k_batch_{batch_start_index}_{batch_end_index}"

        # Restore batch start print
        print(f"\nProcessing batch {current_batch_num}/{num_batches}: tickers {batch_start_index} to {batch_end_index-1}")

        batch_stocks = {}
        batch_start_time = time.time()

        # Re-add per-batch progress bar
        batch_progress = tqdm(batch_tickers, desc=f"Batch {current_batch_num}/{num_batches}", leave=False, unit="ticker")
        
        # Iterate with per-batch bar
        for ticker in batch_progress:
            graded_stock = None
            stock_data = get_stock_data(ticker)
            if stock_data:
                stock_data['quadrant'] = quadrant
                graded_stock = grade_stocks(stock_data, quadrant)
                if graded_stock:
                    batch_stocks[ticker] = graded_stock
                    # Print individual stock grade here
                    print(f"  Graded: {ticker:<8} ({stock_data.get('name', 'N/A'):<30}) - Grade: {graded_stock.get('overall_grade', 'N/A')}")
                else:
                    # Optional: print if grading itself failed
                    # print(f"  Skipping {ticker}: Failed to grade.")
                    pass
            # else: # get_stock_data already prints errors/skips
            #     pass

            # Rate limiting
            time.sleep(0.5) # Increased delay to 0.5 seconds

            # Removed overall_progress.update()
            total_processed += 1 # Still increment total processed count

        # --- Batch finished ---
        batch_execution_time = time.time() - batch_start_time
        saved_count = save_to_database(batch_stocks, batch_name=batch_identifier)
        all_graded_stocks.update(batch_stocks)

        # Restore calculation based on total_processed
        elapsed_time = time.time() - analysis_start_time
        percentage_complete = (total_processed / total_tickers) * 100 if total_tickers > 0 else 0

        estimated_remaining_time_str = "Calculating..."
        # Use total_processed for ETA calculation
        if total_processed > 0 and elapsed_time > 0:
            time_per_ticker = elapsed_time / total_processed
            estimated_total_time = time_per_ticker * total_tickers
            estimated_remaining_time_sec = estimated_total_time - elapsed_time
            if estimated_remaining_time_sec > 0:
                mins, secs = divmod(int(estimated_remaining_time_sec), 60)
                hours, mins = divmod(mins, 60)
                if hours > 0: estimated_remaining_time_str = f"{hours}h {mins}m {secs}s"
                elif mins > 0: estimated_remaining_time_str = f"{mins}m {secs}s"
                else: estimated_remaining_time_str = f"{secs}s"
            else:
                estimated_remaining_time_str = "Almost done!"
        
        # Restore original end-of-batch summary print
        print(f"Finished batch {current_batch_num}/{num_batches}. "
              f"Saved {saved_count}/{len(batch_stocks)} stocks from this batch ({batch_execution_time:.2f}s). "
              f"Overall Progress: {total_processed}/{total_tickers} tickers attempted ({percentage_complete:.1f}%). "
              f"Est. time remaining: {estimated_remaining_time_str}")

    # --- All batches finished ---
    # Removed overall_progress.close()
    total_execution_time = time.time() - analysis_start_time
    print(f"\nRussell 2000 Analysis complete in {total_execution_time:.2f} seconds") # Use total_execution_time
    print(f"Attempted to process {total_tickers} tickers. Successfully retrieved data for and graded {len(all_graded_stocks)} stocks.")
    
    # Get database statistics (might need adjustment in db_utils if batch names are used differently)
    try:
        db_stats = get_db_stats(index_filter='russell2k') # Assuming get_db_stats can filter by index/batch pattern
        print("Database Statistics (Russell 2000 related):")
        print(f"Total stock analyses: {db_stats.get('stock_analysis_count', 'N/A')}")
        # print(f"Total economic quadrant analyses: {db_stats.get('economic_quadrant_count', 'N/A')}") # Might not be relevant here
        print(f"Analysis batches found: {', '.join(db_stats.get('batches', ['N/A']))}")
        
        grade_distribution = db_stats.get('grade_distribution', {})
        print("Grade Distribution (Russell 2000):")
        if grade_distribution:
            for grade, count in grade_distribution.items():
                print(f"{grade}: {count} stocks")
        else:
            print("No grade distribution data found for Russell 2000 batches.")

    except Exception as e:
        print(f"Could not retrieve detailed database statistics: {e}")
        print("Showing overall grade summary from current run:")
        # Fallback summary from memory if DB stats fail
        if all_graded_stocks:
            grades = [data['overall_grade'] for data in all_graded_stocks.values() if data.get('overall_grade')]
            grade_counts = {grade: grades.count(grade) for grade in sorted(list(set(grades)))}
            print("Grade Distribution (from this run):")
            for grade, count in grade_counts.items():
                print(f"{grade}: {count} stocks")


    return all_graded_stocks

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import requests
        from bs4 import BeautifulSoup
        import yfinance
        import pandas
        from tqdm.auto import tqdm # Ensure tqdm is checked here too
    except ImportError as e:
        print(f"Missing required library: {e}. Please install it (e.g., pip install requests beautifulsoup4 yfinance pandas tqdm)")
        exit(1)

    # Run the analysis for Russell 2000
    russell_graded_stocks = analyze_russell2k()
    
    # Print a summary of top-graded stocks
    if russell_graded_stocks:
        top_stocks = {symbol: data for symbol, data in russell_graded_stocks.items() 
                     if data.get('overall_grade') in ['A+', 'A']}
        
        print(f"Found {len(top_stocks)} top-graded Russell 2000 stocks (A+ or A) in this run:")
        # Limit printing details if too many
        count = 0
        for symbol, data in top_stocks.items():
            print(f"{symbol} ({data.get('name', 'N/A')}): {data['overall_grade']} - Sector: {data.get('sector', 'N/A')}") 
            count += 1
            if count >= 20: # Print details for max 20 top stocks
                print("... (and potentially more)")
                break

    print("Russell 2000 analysis script finished.") 