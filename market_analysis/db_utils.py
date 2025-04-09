"""
Database utilities for storing market analysis results.
"""
import os
import sqlite3
import pandas as pd
import json
from datetime import datetime

# Define database directory and path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'market_analysis.db')

def ensure_db_exists():
    """Ensure database directory and file exist"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    
    # Initialize database if it doesn't exist
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create stock analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        name TEXT,
        sector TEXT,
        grade TEXT,
        analysis_date TEXT,
        price REAL,
        volume REAL,
        momentum_score REAL,
        value_score REAL,
        growth_score REAL,
        quality_score REAL,
        revenue_growth REAL,
        earnings_growth REAL,
        pe_ratio REAL,
        debt_to_ebitda REAL,
        analysis_batch TEXT,
        quadrant TEXT,
        analysis_notes TEXT,
        json_data TEXT
    )
    ''')
    
    # Create economic quadrant analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS economic_quadrants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_date TEXT,
        quadrant TEXT,
        balance_sheet_trend TEXT,
        interest_rate_level TEXT,
        balance_sheet_value REAL,
        interest_rate_value REAL,
        analysis_notes TEXT,
        json_data TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    
    return True

def save_stock_analysis(symbol, name=None, sector=None, grade=None, price=None, 
                        volume=None, momentum_score=None, value_score=None, 
                        growth_score=None, quality_score=None, revenue_growth=None,
                        earnings_growth=None, pe_ratio=None, debt_to_ebitda=None,
                        batch_name="general", quadrant=None, notes=None, json_data=None):
    """Save stock analysis results to database"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert JSON data to string if it's a dict
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)
    
    cursor.execute('''
    INSERT INTO stock_analysis 
    (symbol, name, sector, grade, analysis_date, price, volume, 
     momentum_score, value_score, growth_score, quality_score,
     revenue_growth, earnings_growth, pe_ratio, debt_to_ebitda,
     analysis_batch, quadrant, analysis_notes, json_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, name, sector, grade, analysis_date, price, volume, 
          momentum_score, value_score, growth_score, quality_score,
          revenue_growth, earnings_growth, pe_ratio, debt_to_ebitda,
          batch_name, quadrant, notes, json_data))
    
    conn.commit()
    conn.close()
    
    return True

def save_economic_quadrant(quadrant, balance_sheet_trend, interest_rate_level, 
                          balance_sheet_value=None, interest_rate_value=None, 
                          notes=None, json_data=None):
    """Save economic quadrant analysis to database"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert JSON data to string if it's a dict
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)
    
    cursor.execute('''
    INSERT INTO economic_quadrants 
    (analysis_date, quadrant, balance_sheet_trend, interest_rate_level,
     balance_sheet_value, interest_rate_value, analysis_notes, json_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (analysis_date, quadrant, balance_sheet_trend, interest_rate_level,
          balance_sheet_value, interest_rate_value, notes, json_data))
    
    conn.commit()
    conn.close()
    
    return True

def get_latest_economic_quadrant():
    """Get the most recent economic quadrant analysis"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM economic_quadrants 
    ORDER BY analysis_date DESC 
    LIMIT 1
    ''')
    
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        # Convert result to dictionary
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, result))
    else:
        return None

def get_stocks_by_grade(grade=None, limit=100):
    """Get stocks with a specific grade or all stocks if grade is None"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if grade:
        cursor.execute(f'''
        SELECT * FROM stock_analysis 
        WHERE grade = ? 
        ORDER BY analysis_date DESC 
        LIMIT {limit}
        ''', (grade,))
    else:
        cursor.execute(f'''
        SELECT * FROM stock_analysis 
        ORDER BY analysis_date DESC 
        LIMIT {limit}
        ''')
    
    results = cursor.fetchall()
    
    # Convert results to DataFrame
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(results, columns=columns)
    
    conn.close()
    
    return df

def get_stocks_by_batch(batch_name, limit=500):
    """Get stocks from a specific analysis batch"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f'''
    SELECT * FROM stock_analysis 
    WHERE analysis_batch = ? 
    ORDER BY analysis_date DESC 
    LIMIT {limit}
    ''', (batch_name,))
    
    results = cursor.fetchall()
    
    # Convert results to DataFrame
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(results, columns=columns)
    
    conn.close()
    
    return df

def get_economic_quadrant_history(limit=10):
    """Get historical economic quadrant analyses"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    
    df = pd.read_sql_query(f'''
    SELECT * FROM economic_quadrants 
    ORDER BY analysis_date DESC 
    LIMIT {limit}
    ''', conn)
    
    conn.close()
    
    return df

def get_db_stats():
    """Get database statistics"""
    ensure_db_exists()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Count records in stock_analysis table
    cursor.execute('SELECT COUNT(*) FROM stock_analysis')
    stock_count = cursor.fetchone()[0]
    
    # Count records in economic_quadrants table
    cursor.execute('SELECT COUNT(*) FROM economic_quadrants')
    quadrant_count = cursor.fetchone()[0]
    
    # Get unique batch names
    cursor.execute('SELECT DISTINCT analysis_batch FROM stock_analysis')
    batches = [row[0] for row in cursor.fetchall()]
    
    # Get grade distribution
    cursor.execute('''
    SELECT grade, COUNT(*) as count 
    FROM stock_analysis 
    GROUP BY grade 
    ORDER BY count DESC
    ''')
    grade_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        'stock_analysis_count': stock_count,
        'economic_quadrant_count': quadrant_count,
        'batches': batches,
        'grade_distribution': grade_dist
    }

def add_column_if_not_exists(table_name, column_name, column_type="TEXT"):
    """Add a column to a table if it doesn't exist"""
    ensure_db_exists()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        
        if column_name not in columns:
            print(f"Adding column '{column_name}' to table '{table_name}'...")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            conn.commit()
            print(f"Column '{column_name}' added successfully.")
        # else:
        #     print(f"Column '{column_name}' already exists in table '{table_name}'.")
            
    except sqlite3.Error as e:
        print(f"Database error when adding column {column_name}: {e}")
    finally:
        conn.close()

def update_stock_data(symbol, update_data):
    """
    Update the most recent stock analysis record for a given symbol.
    Args:
        symbol (str): The stock symbol to update.
        update_data (dict): A dictionary where keys are column names and values are the new values.
    Returns:
        bool: True if update was successful, False otherwise.
    """
    ensure_db_exists()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if not update_data:
        print(f"No update data provided for {symbol}.")
        conn.close()
        return False
        
    # Ensure all required columns exist, especially custom ones like best_params_4ma_daily
    for column_name in update_data.keys():
        add_column_if_not_exists('stock_analysis', column_name) # Add column if it doesn't exist

    # Construct the SET part of the SQL query
    set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
    values = list(update_data.values())
    
    try:
        # Find the ID of the most recent record for the symbol
        cursor.execute('''
        SELECT id FROM stock_analysis 
        WHERE symbol = ? 
        ORDER BY analysis_date DESC 
        LIMIT 1
        ''', (symbol,))
        
        result = cursor.fetchone()
        
        if not result:
            print(f"No record found for symbol {symbol} to update.")
            conn.close()
            return False
            
        record_id = result[0]
        
        # Update the record using its ID
        query = f"UPDATE stock_analysis SET {set_clause} WHERE id = ?"
        values.append(record_id) # Add the ID to the values list for the WHERE clause
        
        cursor.execute(query, values)
        conn.commit()
        
        # Check if the update was successful
        if cursor.rowcount > 0:
            # print(f"Successfully updated record for {symbol}.") # Optional: logging handled in calling script
            return True
        else:
            print(f"No rows were updated for {symbol} (ID: {record_id}). Check if data is different.")
            return False
            
    except sqlite3.Error as e:
        print(f"Database error updating {symbol}: {e}")
        conn.rollback() # Rollback changes on error
        return False
    finally:
        conn.close() 