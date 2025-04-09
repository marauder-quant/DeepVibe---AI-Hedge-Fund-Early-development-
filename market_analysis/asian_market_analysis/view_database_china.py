"""
Database Viewer for China Market Analysis
View and export China stock analysis and economic quadrant database content
"""
import os
import pandas as pd
import sqlite3
import json
from datetime import datetime
import sys

# --- Adjust sys.path to allow direct script execution --- 
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End path adjustment ---

# Import from the china-specific db utils (now using direct import)
try:
    from db_utils_china import ensure_china_db_exists, get_china_db_stats
except ImportError as e:
    print(f"Error importing db_utils_china: {e}")
    print("Ensure db_utils_china.py is in the same directory as this script.")
    sys.exit(1)

# Define database directory and path for China data
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'china_market_analysis.db')

def get_all_china_stocks():
    """
    Retrieve all China stock analysis data from the database
    """
    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Get all stock analysis data, selecting most recent entry for each symbol
    query = """
    SELECT s1.*
    FROM china_stock_analysis s1
    JOIN (
        SELECT symbol, MAX(analysis_date) as max_date
        FROM china_stock_analysis
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

def get_all_china_quadrants():
    """
    Retrieve all China economic quadrant analyses from the database
    """
    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Get all quadrant data
    query = """
    SELECT *
    FROM china_economic_quadrants
    ORDER BY analysis_date DESC
    """

    # Get data as DataFrame
    df = pd.read_sql_query(query, conn)

    # Close connection
    conn.close()

    return df

def get_china_stocks_by_grade(grade=None):
    """
    Get China stocks filtered by grade
    """
    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    if grade:
        query = """
        SELECT s1.*
        FROM china_stock_analysis s1
        JOIN (
            SELECT symbol, MAX(analysis_date) as max_date
            FROM china_stock_analysis
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
        FROM china_stock_analysis s1
        JOIN (
            SELECT symbol, MAX(analysis_date) as max_date
            FROM china_stock_analysis
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

def get_top_china_stocks():
    """
    Get only A+ and A graded China stocks
    """
    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Get top graded stocks
    query = """
    SELECT s1.*
    FROM china_stock_analysis s1
    JOIN (
        SELECT symbol, MAX(analysis_date) as max_date
        FROM china_stock_analysis
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

def export_china_db_to_csv(output_dir='.'):
    """
    Export China database content to CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get all data
    all_stocks = get_all_china_stocks()
    top_stocks = get_top_china_stocks()
    all_quadrants = get_all_china_quadrants()

    # Export to CSV
    all_stocks.to_csv(os.path.join(output_dir, f'all_china_stocks_{timestamp}.csv'), index=False)
    top_stocks.to_csv(os.path.join(output_dir, f'top_china_stocks_{timestamp}.csv'), index=False)
    all_quadrants.to_csv(os.path.join(output_dir, f'all_china_quadrants_{timestamp}.csv'), index=False)

    print(f"Exported China data to {output_dir}:")
    print(f"  - all_china_stocks_{timestamp}.csv ({len(all_stocks)} records)")
    print(f"  - top_china_stocks_{timestamp}.csv ({len(top_stocks)} records)")
    print(f"  - all_china_quadrants_{timestamp}.csv ({len(all_quadrants)} records)")

    return {
        'all_china_stocks': os.path.join(output_dir, f'all_china_stocks_{timestamp}.csv'),
        'top_china_stocks': os.path.join(output_dir, f'top_china_stocks_{timestamp}.csv'),
        'all_china_quadrants': os.path.join(output_dir, f'all_china_quadrants_{timestamp}.csv')
    }

def print_china_db_summary():
    """
    Print a summary of the China database content
    """
    # Get database statistics
    db_stats = get_china_db_stats()

    print("\n=== CHINA DATABASE SUMMARY ===")
    print(f"Total historical China stock analyses: {db_stats['china_stock_analysis_count']}")

    # Get count of unique stocks (most recent analyses only)
    all_stocks = get_all_china_stocks()
    print(f"Unique China stocks in database: {len(all_stocks)}")

    print(f"Total China economic quadrant analyses: {db_stats['china_economic_quadrant_count']}")

    # Print grade distribution for China stocks
    print("\nChina Stock Grade Distribution (most recent analysis per stock):")
    if not all_stocks.empty:
        unique_grade_distribution = all_stocks['grade'].value_counts().to_dict()
        for grade in ['A+', 'A', 'B', 'C', 'F']:
            if grade in unique_grade_distribution:
                print(f"  {grade}: {unique_grade_distribution[grade]} stocks")
    else:
        print("  No stock data found.")

    # Get latest economic quadrant for China
    all_quadrants = get_all_china_quadrants()
    if not all_quadrants.empty:
        latest_quadrant = all_quadrants.iloc[0]
        print(f"\nLatest China Economic Quadrant: {latest_quadrant['quadrant']}")
        print(f"Analysis Date: {latest_quadrant['analysis_date']}")
        # Display data from the new structure (CMS Percentile based)
        print(f"Notes: {latest_quadrant['analysis_notes']}")

        # --- Display data from NEW specific columns --- 
        bs_val = latest_quadrant.get('latest_total_reserves_ex_gold_value')
        ir_val = latest_quadrant.get('latest_interbank_rate_3m_value')
        z_bs = latest_quadrant.get('z_score_total_reserves_ex_gold')
        z_ir = latest_quadrant.get('z_score_interbank_rate_3m')

        bs_val_str = f"{bs_val:.2f}" if bs_val is not None else 'N/A' # Assuming billion for BS
        ir_val_str = f"{ir_val:.2f}%" if ir_val is not None else 'N/A' # Assuming percent for IR
        z_bs_str = f"{z_bs:.2f}" if z_bs is not None else 'N/A'
        z_ir_str = f"{z_ir:.2f}" if z_ir is not None else 'N/A'

        print(f"Latest Total Reserves ex Gold: {bs_val_str}")
        print(f"Latest Interbank Rate (3M): {ir_val_str}")
        print(f"Z-Score (Total Reserves ex Gold): {z_bs_str}")
        print(f"Z-Score (Interbank Rate 3M): {z_ir_str}")
        # --- End display from NEW specific columns ---

        # Optionally, display CMS/Percentile from JSON as fallback/comparison
        if latest_quadrant['json_data']:
            try:
                details = json.loads(latest_quadrant['json_data'])
                cms_val = details.get('current_cms')
                cms_pctl = details.get('cms_percentile')
                cms_str = f"{cms_val:.2f}" if cms_val is not None else 'N/A'
                cms_pctl_str = f"{cms_pctl:.1f}%" if cms_pctl is not None else 'N/A'
                print(f"CMS (2-Factor, from JSON): {cms_str}")
                print(f"CMS Percentile (vs 5yr, from JSON): {cms_pctl_str}")
                # Z-scores from JSON (already displayed from direct columns, maybe comment out)
                # print(f"Z-Score (Balance Sheet, from JSON): {details['last_z_scores'].get('balance_sheet_cn'):.2f}" if details.get('last_z_scores') and details['last_z_scores'].get('balance_sheet_cn') is not None else "Z-Score (BS, JSON): N/A")
                # print(f"Z-Score (Interest Rate, from JSON): {details['last_z_scores'].get('interest_rate_cn'):.2f}" if details.get('last_z_scores') and details['last_z_scores'].get('interest_rate_cn') is not None else "Z-Score (IR, JSON): N/A")
            except (json.JSONDecodeError, TypeError, KeyError):
                print("Could not parse details from json_data.")

    # Get top China stocks
    top_stocks = get_top_china_stocks()
    print(f"\nTop China Stocks (A+ and A grades): {len(top_stocks)} stocks")

    if not top_stocks.empty:
        # Group by grade and sector
        grade_sector_count = top_stocks.groupby(['grade', 'sector']).size().reset_index(name='count')

        # Print A+ stocks
        a_plus_stocks = top_stocks[top_stocks['grade'] == 'A+']
        if not a_plus_stocks.empty:
            print("\nA+ Grade China Stocks:")
            for idx, row in a_plus_stocks.iterrows():
                print(f"  {row['symbol']} ({row['name']}): {row['sector']}")

        # Print A stocks
        a_stocks = top_stocks[top_stocks['grade'] == 'A']
        if not a_stocks.empty:
            print("\nA Grade China Stocks:")
            for idx, row in a_stocks.iterrows():
                print(f"  {row['symbol']} ({row['name']}): {row['sector']}")

        # Print sector distribution
        print("\nSector Distribution of Top China Stocks:")
        for sector, count in top_stocks.groupby('sector').size().items():
            print(f"  {sector}: {count} stocks")

    return top_stocks

def print_detailed_china_stock_info(symbol):
    """
    Print detailed information for a specific China stock
    """
    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Get stock data
    query = """
    SELECT *
    FROM china_stock_analysis
    WHERE symbol = ?
    ORDER BY analysis_date DESC
    LIMIT 1
    """

    # Get data as DataFrame
    df = pd.read_sql_query(query, conn, params=(symbol,))

    # Close connection
    conn.close()

    if df.empty:
        print(f"No data found for China stock symbol {symbol}")
        return None

    # Get stock info
    stock = df.iloc[0]

    print(f"\n=== DETAILED CHINA STOCK INFORMATION: {symbol} ===")
    print(f"Name: {stock['name']}")
    print(f"Sector: {stock['sector']}")
    print(f"Grade: {stock['grade']}")
    print(f"Analysis Date: {stock['analysis_date']}")

    # Financial metrics
    if 'price' in stock and stock['price'] is not None:
        print(f"Price: ${stock['price']:.2f}") # Assuming price is in USD, adjust if needed

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

def view_full_china_table(table_name):
    """
    Retrieve and print the full contents of a specified China table.
    """
    allowed_tables = ['china_stock_analysis', 'china_economic_quadrants']
    if table_name not in allowed_tables:
        print(f"Error: Invalid table name. Choose from: {allowed_tables}")
        return

    # Ensure database exists
    ensure_china_db_exists()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    try:
        print(f"\n=== FULL TABLE VIEW (China DB): {table_name} ===")
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
        print(f"Error reading table {table_name} from China DB: {e}")
    finally:
        # Close connection
        conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View and export China market analysis database content")
    parser.add_argument('--export', action='store_true', help='Export China database to CSV files')
    parser.add_argument('--output', type=str, default='.', help='Output directory for CSV exports')
    parser.add_argument('--stock', type=str, help='Get detailed information for a specific China stock')
    parser.add_argument('--grade', type=str, help='Filter China stocks by grade (A+, A, B, C, F)')
    parser.add_argument('--table', type=str, choices=['china_stock_analysis', 'china_economic_quadrants'],
                        help='View the full contents of a specific China table')

    args = parser.parse_args()

    if args.stock:
        # Print detailed information for a specific stock
        print_detailed_china_stock_info(args.stock.upper())
    elif args.grade:
        # Filter stocks by grade
        stocks = get_china_stocks_by_grade(args.grade)
        if isinstance(stocks, pd.DataFrame) and not stocks.empty:
            print(f"\n=== CHINA STOCKS WITH GRADE {args.grade} ===")
            for idx, stock in stocks.iterrows():
                print(f"{stock['symbol']}: {stock['name']} ({stock['sector']})")
            print(f"\nTotal: {len(stocks)} stocks with grade {args.grade}")
        else:
            print(f"No China stocks found with grade {args.grade}")
    elif args.export:
        # Export database to CSV
        export_china_db_to_csv(args.output)
    elif args.table:
        view_full_china_table(args.table)
    else:
        # Print summary
        print_china_db_summary() 