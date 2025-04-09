"""
Database Viewer for Market Analysis
View and export stock analysis database content
"""
import os
import pandas as pd
import sqlite3
from datetime import datetime
from db_utils import ensure_db_exists, get_db_stats

# Define database directory and path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'market_analysis.db')

def get_all_stocks():
    """
    Retrieve all stock analysis data from the database
    """
    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get all stock analysis data, selecting most recent entry for each symbol
    query = """
    SELECT s1.*
    FROM stock_analysis s1
    JOIN (
        SELECT symbol, MAX(analysis_date) as max_date
        FROM stock_analysis
        GROUP BY symbol
    ) s2
    ON s1.symbol = s2.symbol AND s1.analysis_date = s2.max_date
    ORDER BY grade ASC, symbol ASC
    """
    
    # Get data as DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    return df

def get_all_quadrants():
    """
    Retrieve all economic quadrant analyses from the database
    """
    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get all quadrant data
    query = """
    SELECT *
    FROM economic_quadrants
    ORDER BY analysis_date DESC
    """
    
    # Get data as DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    return df

def get_stocks_by_grade(grade=None):
    """
    Get stocks filtered by grade
    """
    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    if grade:
        query = """
        SELECT s1.*
        FROM stock_analysis s1
        JOIN (
            SELECT symbol, MAX(analysis_date) as max_date
            FROM stock_analysis
            GROUP BY symbol
        ) s2
        ON s1.symbol = s2.symbol AND s1.analysis_date = s2.max_date
        WHERE s1.grade = ?
        ORDER BY symbol ASC
        """
        df = pd.read_sql_query(query, conn, params=(grade,))
    else:
        # Get distribution of all grades
        query = """
        SELECT s1.grade, COUNT(*) as count
        FROM stock_analysis s1
        JOIN (
            SELECT symbol, MAX(analysis_date) as max_date
            FROM stock_analysis
            GROUP BY symbol
        ) s2
        ON s1.symbol = s2.symbol AND s1.analysis_date = s2.max_date
        GROUP BY s1.grade
        ORDER BY s1.grade ASC
        """
        df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    return df

def get_top_stocks():
    """
    Get only A+ and A graded stocks
    """
    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get top graded stocks
    query = """
    SELECT s1.*
    FROM stock_analysis s1
    JOIN (
        SELECT symbol, MAX(analysis_date) as max_date
        FROM stock_analysis
        GROUP BY symbol
    ) s2
    ON s1.symbol = s2.symbol AND s1.analysis_date = s2.max_date
    WHERE s1.grade IN ('A+', 'A')
    ORDER BY s1.grade ASC, symbol ASC
    """
    
    # Get data as DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    return df

def export_to_csv(output_dir='.'):
    """
    Export database content to CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get all data
    all_stocks = get_all_stocks()
    top_stocks = get_top_stocks()
    all_quadrants = get_all_quadrants()
    
    # Export to CSV
    all_stocks.to_csv(os.path.join(output_dir, f'all_stocks_{timestamp}.csv'), index=False)
    top_stocks.to_csv(os.path.join(output_dir, f'top_stocks_{timestamp}.csv'), index=False)
    all_quadrants.to_csv(os.path.join(output_dir, f'all_quadrants_{timestamp}.csv'), index=False)
    
    print(f"Exported data to {output_dir}:")
    print(f"  - all_stocks_{timestamp}.csv ({len(all_stocks)} records)")
    print(f"  - top_stocks_{timestamp}.csv ({len(top_stocks)} records)")
    print(f"  - all_quadrants_{timestamp}.csv ({len(all_quadrants)} records)")
    
    return {
        'all_stocks': os.path.join(output_dir, f'all_stocks_{timestamp}.csv'),
        'top_stocks': os.path.join(output_dir, f'top_stocks_{timestamp}.csv'),
        'all_quadrants': os.path.join(output_dir, f'all_quadrants_{timestamp}.csv')
    }

def print_summary():
    """
    Print a summary of the database content
    """
    # Get database statistics
    db_stats = get_db_stats()
    
    print("\n=== DATABASE SUMMARY ===")
    print(f"Total historical stock analyses: {db_stats['stock_analysis_count']}")
    
    # Get count of unique stocks (most recent analyses only)
    all_stocks = get_all_stocks()
    print(f"Unique stocks in database: {len(all_stocks)}")
    
    print(f"Total economic quadrant analyses: {db_stats['economic_quadrant_count']}")
    
    # Print grade distribution
    print("\nGrade Distribution (most recent analysis per stock):")
    unique_grade_distribution = all_stocks['grade'].value_counts().to_dict()
    for grade in ['A+', 'A', 'B', 'C', 'F']:
        if grade in unique_grade_distribution:
            print(f"  {grade}: {unique_grade_distribution[grade]} stocks")
    
    # Get latest economic quadrant
    all_quadrants = get_all_quadrants()
    if not all_quadrants.empty:
        latest_quadrant = all_quadrants.iloc[0]
        print(f"\nLatest Economic Quadrant: {latest_quadrant['quadrant']}")
        print(f"Analysis Date: {latest_quadrant['analysis_date']}")
        print(f"Balance Sheet Trend: {latest_quadrant['balance_sheet_trend']}")
        print(f"Interest Rate Level: {latest_quadrant['interest_rate_level']}")
    
    # Get top stocks
    top_stocks = get_top_stocks()
    print(f"\nTop Stocks (A+ and A grades): {len(top_stocks)} stocks")
    
    if not top_stocks.empty:
        # Group by grade and sector
        grade_sector_count = top_stocks.groupby(['grade', 'sector']).size().reset_index(name='count')
        
        # Print A+ stocks
        a_plus_stocks = top_stocks[top_stocks['grade'] == 'A+']
        if not a_plus_stocks.empty:
            print("\nA+ Grade Stocks:")
            for idx, row in a_plus_stocks.iterrows():
                print(f"  {row['symbol']} ({row['name']}): {row['sector']}")
        
        # --- Add check for A grade stocks and the new column --- 
        a_stocks = top_stocks[top_stocks['grade'] == 'A']
        if not a_stocks.empty:
            print("\nA Grade Stocks (Params Check):")
            # Check if the column exists before trying to access it
            if 'best_params_4ma_daily' in a_stocks.columns:
                for idx, row in a_stocks.head(10).iterrows(): # Print first 10 A stocks
                    params = row['best_params_4ma_daily']
                    print(f"  {row['symbol']}: Params={params if params else 'Not set'}")
            else:
                print("  'best_params_4ma_daily' column not found in the DataFrame.")
        # --- End check ---

        # Print sector distribution
        print("\nSector Distribution of Top Stocks:")
        for sector, count in top_stocks.groupby('sector').size().items():
            print(f"  {sector}: {count} stocks")
    
    return top_stocks

def print_detailed_stock_info(symbol):
    """
    Print detailed information for a specific stock
    """
    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get stock data
    query = """
    SELECT *
    FROM stock_analysis
    WHERE symbol = ?
    ORDER BY analysis_date DESC
    LIMIT 1
    """
    
    # Get data as DataFrame
    df = pd.read_sql_query(query, conn, params=(symbol,))
    
    # Close connection
    conn.close()
    
    if df.empty:
        print(f"No data found for symbol {symbol}")
        return None
    
    # Get stock info
    stock = df.iloc[0]
    
    print(f"\n=== DETAILED STOCK INFORMATION: {symbol} ===")
    print(f"Name: {stock['name']}")
    print(f"Sector: {stock['sector']}")
    print(f"Grade: {stock['grade']}")
    print(f"Analysis Date: {stock['analysis_date']}")
    
    # Financial metrics
    if 'price' in stock and stock['price'] is not None:
        print(f"Price: ${stock['price']:.2f}")
    
    if 'revenue_growth' in stock and stock['revenue_growth'] is not None:
        print(f"Revenue Growth: {stock['revenue_growth']*100:.2f}%")
    
    if 'earnings_growth' in stock and stock['earnings_growth'] is not None:
        print(f"Earnings Growth: {stock['earnings_growth']*100:.2f}%")
    
    if 'pe_ratio' in stock and stock['pe_ratio'] is not None:
        print(f"P/E Ratio: {stock['pe_ratio']:.2f}")
    
    if 'debt_to_ebitda' in stock and stock['debt_to_ebitda'] is not None:
        print(f"Debt to EBITDA: {stock['debt_to_ebitda']:.2f}")
    
    # Analysis notes
    if 'analysis_notes' in stock and stock['analysis_notes'] is not None:
        print(f"\nAnalysis Notes: {stock['analysis_notes']}")
    
    return stock

def view_full_table(table_name):
    """
    Retrieve and print the full contents of a specified table.
    """
    allowed_tables = ['stock_analysis', 'economic_quadrants']
    if table_name not in allowed_tables:
        print(f"Error: Invalid table name. Choose from: {allowed_tables}")
        return

    # Ensure database exists
    ensure_db_exists()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        print(f"\n=== FULL TABLE VIEW: {table_name} ===")
        # Get all data from the specified table
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print(f"Table '{table_name}' is empty.")
        else:
            # Print the entire DataFrame without truncation
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.to_string())
            print(f"\nTotal records in {table_name}: {len(df)}")
            
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
    finally:
        # Close connection
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View and export market analysis database content")
    parser.add_argument('--export', action='store_true', help='Export database to CSV files')
    parser.add_argument('--output', type=str, default='.', help='Output directory for CSV exports')
    parser.add_argument('--stock', type=str, help='Get detailed information for a specific stock')
    parser.add_argument('--grade', type=str, help='Filter stocks by grade (A+, A, B, C, F)')
    parser.add_argument('--table', type=str, choices=['stock_analysis', 'economic_quadrants'], 
                        help='View the full contents of a specific table (stock_analysis or economic_quadrants)')
    
    args = parser.parse_args()
    
    if args.stock:
        # Print detailed information for a specific stock
        print_detailed_stock_info(args.stock.upper())
    elif args.grade:
        # Filter stocks by grade
        stocks = get_stocks_by_grade(args.grade)
        if isinstance(stocks, pd.DataFrame) and not stocks.empty:
            print(f"\n=== STOCKS WITH GRADE {args.grade} ===")
            for idx, stock in stocks.iterrows():
                print(f"{stock['symbol']}: {stock['name']} ({stock['sector']})")
            print(f"\nTotal: {len(stocks)} stocks with grade {args.grade}")
        else:
            print(f"No stocks found with grade {args.grade}")
    elif args.export:
        # Export database to CSV
        export_to_csv(args.output)
    elif args.table:
        view_full_table(args.table)
    else:
        # Print summary
        print_summary() 