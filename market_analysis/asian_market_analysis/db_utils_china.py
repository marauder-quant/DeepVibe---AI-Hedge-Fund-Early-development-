"""
Database utilities for storing China market analysis results.
"""
import os
import sqlite3
import pandas as pd
import json
from datetime import datetime

# Define database directory and path for China data
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'china_market_analysis.db')

def ensure_china_db_exists():
    """Ensure China database directory and file exist"""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # Initialize database if it doesn't exist
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create stock analysis table (for potential future China stock data)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS china_stock_analysis (
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
    """)

    # Create economic quadrant analysis table for China
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS china_economic_quadrants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_date TEXT,
        country TEXT DEFAULT 'China',
        quadrant TEXT,
        balance_sheet_trend TEXT, -- Kept, but will be optional
        interest_rate_level TEXT, -- Kept, but will be optional
        balance_sheet_value REAL, -- Original field, kept for potential compatibility
        interest_rate_value REAL, -- Original field, kept for potential compatibility
        -- Use new specific column names
        latest_total_reserves_ex_gold_value REAL, -- Specific name for BS
        latest_interbank_rate_3m_value REAL,      -- Specific name for IR
        z_score_total_reserves_ex_gold REAL,      -- Specific name for BS Z-score
        z_score_interbank_rate_3m REAL,           -- Specific name for IR Z-score
        analysis_notes TEXT,
        json_data TEXT -- Stores detailed results like CMS, percentile, z-scores etc.
    )
    """)

    # --- Add column logic for existing databases (optional but good practice) ---
    # Check and add new columns if they don't exist in an older version of the DB
    table_info = cursor.execute("PRAGMA table_info(china_economic_quadrants)").fetchall()
    existing_columns = [col[1] for col in table_info]
    new_columns = {
        'latest_total_reserves_ex_gold_value': 'REAL',
        'latest_interbank_rate_3m_value': 'REAL',
        'z_score_total_reserves_ex_gold': 'REAL',
        'z_score_interbank_rate_3m': 'REAL'
    }
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            print(f"Adding column '{col_name}' to table 'china_economic_quadrants'...")
            cursor.execute(f"ALTER TABLE china_economic_quadrants ADD COLUMN {col_name} {col_type}")
            print(f"Column '{col_name}' added successfully.")
    # --- End add column logic ---

    conn.commit()
    conn.close()

    return True

def save_china_stock_analysis(symbol, name=None, sector=None, grade=None, price=None,
                        volume=None, momentum_score=None, value_score=None,
                        growth_score=None, quality_score=None, revenue_growth=None,
                        earnings_growth=None, pe_ratio=None, debt_to_ebitda=None,
                        batch_name="general_china", quadrant=None, notes=None, json_data=None):
    """Save China stock analysis results to database"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Convert JSON data to string if it's a dict
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)

    cursor.execute("""
    INSERT INTO china_stock_analysis
    (symbol, name, sector, grade, analysis_date, price, volume,
     momentum_score, value_score, growth_score, quality_score,
     revenue_growth, earnings_growth, pe_ratio, debt_to_ebitda,
     analysis_batch, quadrant, analysis_notes, json_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, name, sector, grade, analysis_date, price, volume,
          momentum_score, value_score, growth_score, quality_score,
          revenue_growth, earnings_growth, pe_ratio, debt_to_ebitda,
          batch_name, quadrant, notes, json_data))

    conn.commit()
    conn.close()

    return True

# Modified function for China Quadrant data based on asian_economic_quadrants.py output
def save_china_economic_quadrant(quadrant,
                          balance_sheet_value=None, interest_rate_value=None,
                          notes=None, json_data=None,
                          balance_sheet_trend=None, interest_rate_level=None,
                          # Use new specific parameter names
                          latest_total_reserves_ex_gold_value=None,
                          latest_interbank_rate_3m_value=None,
                          z_score_total_reserves_ex_gold=None,
                          z_score_interbank_rate_3m=None
                          ):
    """Save China economic quadrant analysis to database, using specific indicator names for columns."""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    country = "China" # Hardcoded for this function

    # Convert JSON data to string if it's a dict
    if isinstance(json_data, dict):
        json_data = json.dumps(json_data)

    cursor.execute("""
    INSERT INTO china_economic_quadrants
    (analysis_date, country, quadrant, balance_sheet_trend, interest_rate_level,
     balance_sheet_value, interest_rate_value, analysis_notes, json_data,
     -- Use new specific column names in INSERT
     latest_total_reserves_ex_gold_value, latest_interbank_rate_3m_value,
     z_score_total_reserves_ex_gold, z_score_interbank_rate_3m
     )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (analysis_date, country, quadrant, balance_sheet_trend, interest_rate_level,
          balance_sheet_value, interest_rate_value, notes, json_data,
          # Pass values for new specific columns
          latest_total_reserves_ex_gold_value, latest_interbank_rate_3m_value,
          z_score_total_reserves_ex_gold, z_score_interbank_rate_3m
          ))

    conn.commit()
    conn.close()
    print(f"Saved China Economic Quadrant {quadrant} to {DB_PATH}")

    return True


# --- Getter functions adapted for China DB ---

def get_latest_china_economic_quadrant():
    """Get the most recent China economic quadrant analysis"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM china_economic_quadrants
    ORDER BY analysis_date DESC
    LIMIT 1
    """)

    result = cursor.fetchone()

    conn.close()

    if result:
        # Convert result to dictionary
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, result))
    else:
        return None

def get_china_stocks_by_grade(grade=None, limit=100):
    """Get China stocks with a specific grade or all stocks if grade is None"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if grade:
        cursor.execute(f"""
        SELECT * FROM china_stock_analysis
        WHERE grade = ?
        ORDER BY analysis_date DESC
        LIMIT {limit}
        """, (grade,))
    else:
        cursor.execute(f"""
        SELECT * FROM china_stock_analysis
        ORDER BY analysis_date DESC
        LIMIT {limit}
        """)

    results = cursor.fetchall()

    # Convert results to DataFrame
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(results, columns=columns)

    conn.close()

    return df

def get_china_stocks_by_batch(batch_name, limit=500):
    """Get China stocks from a specific analysis batch"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
    SELECT * FROM china_stock_analysis
    WHERE analysis_batch = ?
    ORDER BY analysis_date DESC
    LIMIT {limit}
    """, (batch_name,))

    results = cursor.fetchall()

    # Convert results to DataFrame
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(results, columns=columns)

    conn.close()

    return df

def get_china_economic_quadrant_history(limit=10):
    """Get historical China economic quadrant analyses"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(f"""
    SELECT * FROM china_economic_quadrants
    ORDER BY analysis_date DESC
    LIMIT {limit}
    """, conn)

    conn.close()

    return df

def get_china_db_stats():
    """Get China database statistics"""
    ensure_china_db_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count records in china_stock_analysis table
    cursor.execute('SELECT COUNT(*) FROM china_stock_analysis')
    stock_count = cursor.fetchone()[0]

    # Count records in china_economic_quadrants table
    cursor.execute('SELECT COUNT(*) FROM china_economic_quadrants')
    quadrant_count = cursor.fetchone()[0]

    # Get unique batch names from china_stock_analysis
    cursor.execute('SELECT DISTINCT analysis_batch FROM china_stock_analysis')
    batches = [row[0] for row in cursor.fetchall()]

    # Get grade distribution from china_stock_analysis
    cursor.execute("""
    SELECT grade, COUNT(*) as count
    FROM china_stock_analysis
    GROUP BY grade
    ORDER BY count DESC
    """)
    grade_dist = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    return {
        'china_stock_analysis_count': stock_count,
        'china_economic_quadrant_count': quadrant_count,
        'china_batches': batches,
        'china_grade_distribution': grade_dist
    }

# Note: add_column_if_not_exists and update_stock_data adapted for china tables
def add_china_column_if_not_exists(table_name, column_name, column_type="TEXT"):
    """Add a column to a China table if it doesn't exist"""
    ensure_china_db_exists()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    allowed_tables = ['china_stock_analysis', 'china_economic_quadrants']
    if table_name not in allowed_tables:
        print(f"Error: Invalid table name '{table_name}' for China DB.")
        conn.close()
        return

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
        print(f"Database error when adding column {column_name} to {table_name}: {e}")
    finally:
        conn.close()

def update_china_stock_data(symbol, update_data):
    """
    Update the most recent China stock analysis record for a given symbol.
    Args:
        symbol (str): The stock symbol to update.
        update_data (dict): A dictionary where keys are column names and values are the new values.
    Returns:
        bool: True if update was successful, False otherwise.
    """
    ensure_china_db_exists()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if not update_data:
        print(f"No update data provided for China stock {symbol}.")
        conn.close()
        return False

    # Ensure all required columns exist in china_stock_analysis
    for column_name in update_data.keys():
        add_china_column_if_not_exists('china_stock_analysis', column_name)

    # Construct the SET part of the SQL query
    set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
    values = list(update_data.values())

    try:
        # Find the ID of the most recent record for the symbol
        cursor.execute("""
        SELECT id FROM china_stock_analysis
        WHERE symbol = ?
        ORDER BY analysis_date DESC
        LIMIT 1
        """, (symbol,))

        result = cursor.fetchone()

        if not result:
            print(f"No record found for China stock symbol {symbol} to update.")
            conn.close()
            return False

        record_id = result[0]

        # Update the record using its ID
        query = f"UPDATE china_stock_analysis SET {set_clause} WHERE id = ?"
        values.append(record_id) # Add the ID to the values list for the WHERE clause

        cursor.execute(query, values)
        conn.commit()

        # Check if the update was successful
        if cursor.rowcount > 0:
            # print(f"Successfully updated record for China stock {symbol}.")
            return True
        else:
            print(f"No rows were updated for China stock {symbol} (ID: {record_id}). Check if data is different.")
            return False

    except sqlite3.Error as e:
        print(f"Database error updating China stock {symbol}: {e}")
        conn.rollback() # Rollback changes on error
        return False
    finally:
        conn.close() 