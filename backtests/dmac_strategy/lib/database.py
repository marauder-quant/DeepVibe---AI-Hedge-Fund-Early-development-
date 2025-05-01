#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database utilities for the DMAC strategy.
This module manages parameter storage and retrieval.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MIN_WINDOW, DEFAULT_MAX_WINDOW, DEFAULT_WINDOW_STEP, TOP_N_PARAMS, DB_PATH

# Database settings - use absolute path based on the relative path from config
DB_FULL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DB_PATH)

def initialize_database():
    """
    Initialize the SQLite database for storing best parameters.
    
    Returns:
        None
    """
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(DB_FULL_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    # Connect to database
    conn = sqlite3.connect(DB_FULL_PATH)
    c = conn.cursor()
    
    # Create table for parameter combinations if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS best_parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        rank INTEGER NOT NULL,
        fast_window INTEGER NOT NULL,
        slow_window INTEGER NOT NULL,
        performance REAL NOT NULL,
        total_return REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        num_trades INTEGER,
        win_rate REAL,
        date_from TEXT NOT NULL,
        date_to TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        UNIQUE(symbol, timeframe, rank, date_from, date_to)
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {DB_FULL_PATH}")

def save_top_parameters_to_db(symbol, timeframe, start_date, end_date, performance_df, 
                             optimal_results=None, top_n=TOP_N_PARAMS):
    """
    Save the top N parameter combinations to the database.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Data timeframe
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        performance_df (DataFrame): DataFrame containing performance data for all combinations
        optimal_results (dict, optional): Results from optimal parameter run
        top_n (int): Number of top parameter combinations to save
        
    Returns:
        None
    """
    # Initialize the database if it doesn't exist
    if not os.path.exists(DB_FULL_PATH):
        initialize_database()
    
    # Connect to database
    conn = sqlite3.connect(DB_FULL_PATH)
    c = conn.cursor()
    
    # Get top N parameter combinations
    # Sort by 'total_return' by default, but allow other metrics
    top_params = performance_df.nlargest(top_n, 'total_return')
    
    # Format dates for database
    date_from = start_date.strftime('%Y-%m-%d')
    date_to = end_date.strftime('%Y-%m-%d')
    timestamp = datetime.now().isoformat()
    
    # Delete existing entries for this symbol, timeframe, date range
    c.execute('''
    DELETE FROM best_parameters 
    WHERE symbol = ? AND timeframe = ? AND date_from = ? AND date_to = ?
    ''', (symbol, timeframe, date_from, date_to))
    
    # Insert top parameter combinations
    for rank, (_, row) in enumerate(top_params.iterrows(), 1):
        # Extract parameters
        fast_window = int(row['fast_window'])
        slow_window = int(row['slow_window'])
        performance = float(row['total_return'])
        
        # Default values for additional metrics
        total_return = None
        sharpe_ratio = None
        max_drawdown = None
        num_trades = None
        win_rate = None
        
        # If this is the top-ranked combination and we have optimal results, extract detailed metrics
        if rank == 1 and optimal_results is not None:
            try:
                stats = optimal_results['dmac_pf'].stats()
                total_return = stats['Total Return [%]'] / 100.0  # Convert from percentage
                sharpe_ratio = stats['Sharpe Ratio']
                max_drawdown = stats['Max Drawdown [%]'] / 100.0  # Convert from percentage
                num_trades = len(optimal_results['dmac_pf'].trades)
                win_rate = stats['Win Rate [%]'] / 100.0 if 'Win Rate [%]' in stats else None
            except Exception as e:
                print(f"Warning: Could not extract detailed metrics: {e}")
        
        # Insert into database
        c.execute('''
        INSERT INTO best_parameters 
        (symbol, timeframe, rank, fast_window, slow_window, 
         performance, total_return, sharpe_ratio, max_drawdown, num_trades, win_rate,
         date_from, date_to, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, timeframe, rank, fast_window, slow_window,
            performance, total_return, sharpe_ratio, max_drawdown, num_trades, win_rate,
            date_from, date_to, timestamp
        ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Saved top {top_n} parameter combinations to database for {symbol} {timeframe}")

def get_parameters_from_db(symbol=None, timeframe=None, top_n=TOP_N_PARAMS):
    """
    Query the database for saved parameter combinations.
    
    Args:
        symbol (str, optional): Filter by symbol
        timeframe (str, optional): Filter by timeframe
        top_n (int): Maximum number of combinations to return per group
        
    Returns:
        pd.DataFrame: DataFrame containing parameter combinations
    """
    # Check if database exists
    if not os.path.exists(DB_FULL_PATH):
        print(f"Database not found at {DB_FULL_PATH}")
        return pd.DataFrame()
    
    # Connect to database
    conn = sqlite3.connect(DB_FULL_PATH)
    
    # Build query based on filters
    query = "SELECT * FROM best_parameters"
    params = []
    
    if symbol or timeframe:
        query += " WHERE"
        
        if symbol:
            query += " symbol = ?"
            params.append(symbol)
            
        if timeframe:
            if symbol:
                query += " AND"
            query += " timeframe = ?"
            params.append(timeframe)
    
    query += " ORDER BY symbol, timeframe, date_from DESC, date_to DESC, rank"
    
    # Execute query and load results into DataFrame
    df = pd.read_sql_query(query, conn, params=params)
    
    # Close connection
    conn.close()
    
    # Pivot the data to show different timeframes as columns with specific parameter values
    if timeframe is None and symbol is not None:
        # Get only the latest parameters for each timeframe
        latest_params = df.sort_values(['timeframe', 'date_from', 'date_to', 'rank'], 
                                      ascending=[True, False, False, True])
        
        # Get unique timeframes
        timeframes = latest_params['timeframe'].unique()
        
        # Create a structured DataFrame with specific columns for each parameter and timeframe
        result_data = []
        
        for i in range(top_n):
            row_data = {'rank': i+1}
            
            # Add data for each timeframe
            for tf in timeframes:
                tf_data = latest_params[latest_params['timeframe'] == tf]
                
                # Check if we have data for this rank
                if i < len(tf_data):
                    param_row = tf_data.iloc[i]
                    row_data[f'{tf}_fast'] = param_row['fast_window']
                    row_data[f'{tf}_slow'] = param_row['slow_window']
                    row_data[f'{tf}_return'] = param_row['performance']
                    
                    # Add additional metrics if available
                    if param_row['total_return'] is not None:
                        row_data[f'{tf}_total_return'] = param_row['total_return']
                    if param_row['sharpe_ratio'] is not None:
                        row_data[f'{tf}_sharpe'] = param_row['sharpe_ratio']
                    if param_row['max_drawdown'] is not None:
                        row_data[f'{tf}_drawdown'] = param_row['max_drawdown']
                    if param_row['win_rate'] is not None:
                        row_data[f'{tf}_win_rate'] = param_row['win_rate']
                
            # Only add rows that have some data
            if len(row_data) > 1:  # More than just the rank
                result_data.append(row_data)
        
        # Create DataFrame
        return pd.DataFrame(result_data)
    
    return df

def get_available_symbols():
    """
    Get a list of all symbols available in the database.
    
    Returns:
        list: List of unique symbols
    """
    if not os.path.exists(DB_FULL_PATH):
        print(f"Database not found at {DB_FULL_PATH}")
        return []
    
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT symbol FROM best_parameters")
    symbols = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return symbols

def get_available_timeframes(symbol=None):
    """
    Get a list of all timeframes available in the database.
    
    Args:
        symbol (str, optional): Filter by symbol
        
    Returns:
        list: List of unique timeframes
    """
    if not os.path.exists(DB_FULL_PATH):
        print(f"Database not found at {DB_FULL_PATH}")
        return []
    
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    
    if symbol:
        cursor.execute("SELECT DISTINCT timeframe FROM best_parameters WHERE symbol = ?", (symbol,))
    else:
        cursor.execute("SELECT DISTINCT timeframe FROM best_parameters")
    
    timeframes = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return timeframes

def list_database_contents():
    """
    Print a summary of the database contents.
    
    Returns:
        None
    """
    symbols = get_available_symbols()
    
    if not symbols:
        print("No data found in database.")
        return
    
    print(f"Database found at {DB_FULL_PATH}")
    print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    
    for symbol in symbols:
        timeframes = get_available_timeframes(symbol)
        print(f"\n{symbol} parameters found for {len(timeframes)} timeframes: {', '.join(timeframes)}")
        
        # Get the most recent date range for each timeframe
        conn = sqlite3.connect(DB_FULL_PATH)
        cursor = conn.cursor()
        
        for tf in timeframes:
            cursor.execute("""
                SELECT date_from, date_to, COUNT(*) as count
                FROM best_parameters 
                WHERE symbol = ? AND timeframe = ?
                GROUP BY date_from, date_to
                ORDER BY date_from DESC, date_to DESC
                LIMIT 1
            """, (symbol, tf))
            
            result = cursor.fetchone()
            if result:
                date_from, date_to, count = result
                print(f"  {tf}: {count} parameter sets from {date_from} to {date_to}")
        
        conn.close() 