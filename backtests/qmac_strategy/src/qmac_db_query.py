#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QMAC Database Query Tool.
This script provides a way to query the stored best parameters from the QMAC strategy database.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import sqlite3

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# Database settings
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db', 'qmac_parameters.db')

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
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return pd.DataFrame()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
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
                    row_data[f'{tf}_buy_fast'] = param_row['buy_fast']
                    row_data[f'{tf}_buy_slow'] = param_row['buy_slow']
                    row_data[f'{tf}_sell_fast'] = param_row['sell_fast']
                    row_data[f'{tf}_sell_slow'] = param_row['sell_slow']
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
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return []
    
    conn = sqlite3.connect(DB_PATH)
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
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return []
    
    conn = sqlite3.connect(DB_PATH)
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
    
    print(f"Database found at {DB_PATH}")
    print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    
    for symbol in symbols:
        timeframes = get_available_timeframes(symbol)
        print(f"\n{symbol} parameters found for {len(timeframes)} timeframes: {', '.join(timeframes)}")
        
        # Get the most recent date range for each timeframe
        conn = sqlite3.connect(DB_PATH)
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

def format_parameter_output(df, format_type='pretty'):
    """
    Format the parameter output.
    
    Args:
        df (DataFrame): DataFrame containing parameters
        format_type (str): Output format ('pretty', 'csv', or 'json')
        
    Returns:
        str: Formatted output
    """
    if df.empty:
        return "No parameters found."
    
    if format_type == 'pretty':
        # Set pandas display options for better readability
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        return str(df)
    
    elif format_type == 'csv':
        return df.to_csv(index=False)
    
    elif format_type == 'json':
        return df.to_json(orient='records', indent=2)
    
    return str(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QMAC Database Query Tool')
    parser.add_argument('--symbol', type=str, help='Symbol to query (e.g., SPY)')
    parser.add_argument('--timeframe', type=str, help='Timeframe to query (e.g., 1d, 1h, 30m)')
    parser.add_argument('--list', action='store_true', help='List database contents')
    parser.add_argument('--top', type=int, default=TOP_N_PARAMS, help='Number of top results to show')
    parser.add_argument('--format', type=str, choices=['pretty', 'csv', 'json'], default='pretty',
                        help='Output format')
    parser.add_argument('--output', type=str, help='Output file (if not specified, prints to console)')
    
    args = parser.parse_args()
    
    # Display header
    print("QMAC Strategy Database Query Tool")
    print("=================================")
    
    if args.list:
        # List database contents
        list_database_contents()
    elif args.symbol:
        # Query specific symbol
        df = get_parameters_from_db(symbol=args.symbol, timeframe=args.timeframe, top_n=args.top)
        
        # Format output
        output = format_parameter_output(df, args.format)
        
        # Output to file or console
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print("\nResults:")
            print(output)
    else:
        # No specific query, list available symbols
        symbols = get_available_symbols()
        
        if symbols:
            print(f"Available symbols: {', '.join(symbols)}")
            print("\nUse --symbol to query a specific symbol or --list for detailed information.")
        else:
            print("No data found in database.") 