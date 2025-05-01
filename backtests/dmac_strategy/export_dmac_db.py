#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMAC Database Export Utility
Exports parameter data from the DMAC database to CSV format.
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime
import sys
import argparse

# Import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.database import get_parameters_from_db, DB_FULL_PATH
from config import DEFAULT_EXPORTS_DIR

def export_dmac_db_to_csv(output_dir=None, symbol=None, timeframe=None, format_output=True):
    """
    Export DMAC database content to CSV files
    
    Args:
        output_dir (str, optional): Directory to save the CSV files. If None, creates 'exports' dir
        symbol (str, optional): Filter by symbol
        timeframe (str, optional): Filter by timeframe
        format_output (bool): Whether to format the output with timestamps and descriptive names
        
    Returns:
        str: Path to the saved CSV file
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_EXPORTS_DIR)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get parameters from database with optional filtering
    df = get_parameters_from_db(symbol=symbol, timeframe=timeframe)
    
    if df.empty:
        print(f"No data found in database at {DB_FULL_PATH}")
        if symbol or timeframe:
            filter_msg = []
            if symbol:
                filter_msg.append(f"symbol='{symbol}'")
            if timeframe:
                filter_msg.append(f"timeframe='{timeframe}'")
            print(f"Filters applied: {', '.join(filter_msg)}")
        return None
    
    # Generate filename
    if format_output:
        # Format timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create descriptive filename parts
        if symbol:
            symbol_part = symbol
        elif not df.empty and 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            if len(symbols) == 1:
                symbol_part = symbols[0]
            else:
                symbol_part = f"{len(symbols)}symbols"
        else:
            symbol_part = "all"
        
        if timeframe:
            timeframe_part = timeframe
        elif not df.empty and 'timeframe' in df.columns:
            timeframes = df['timeframe'].unique()
            if len(timeframes) == 1:
                timeframe_part = timeframes[0]
            else:
                timeframe_part = f"{len(timeframes)}timeframes"
        else:
            timeframe_part = "all"
        
        filename = f'dmac_{symbol_part}_{timeframe_part}_{timestamp}.csv'
    else:
        filename = 'dmac_parameters.csv'
    
    # Save to CSV
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    
    print(f"Exported {len(df)} records to {output_file}")
    return output_file

def query_database_direct(query, params=None):
    """
    Execute a direct SQL query on the database
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        
    Returns:
        pd.DataFrame: Query results
    """
    if not os.path.exists(DB_FULL_PATH):
        print(f"Database not found at {DB_FULL_PATH}")
        return pd.DataFrame()
    
    # Connect to database
    conn = sqlite3.connect(DB_FULL_PATH)
    
    # Execute query
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    return df

def print_database_summary():
    """
    Print a summary of the DMAC database content
    """
    if not os.path.exists(DB_FULL_PATH):
        print(f"Database not found at {DB_FULL_PATH}")
        return
    
    # Connect to database
    conn = sqlite3.connect(DB_FULL_PATH)
    cursor = conn.cursor()
    
    # Get table information
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\n=== DMAC DATABASE SUMMARY ===")
    print(f"Database path: {DB_FULL_PATH}")
    print(f"Tables: {', '.join([t[0] for t in tables])}")
    
    # Get record counts
    print("\nRecord counts:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count:,} records")
    
    # Get symbol and timeframe statistics
    if 'best_parameters' in [t[0] for t in tables]:
        # Get unique symbols
        cursor.execute("SELECT DISTINCT symbol FROM best_parameters")
        symbols = cursor.fetchall()
        print(f"\nUnique symbols: {len(symbols)}")
        
        # Get unique timeframes
        cursor.execute("SELECT DISTINCT timeframe FROM best_parameters")
        timeframes = cursor.fetchall()
        print(f"Unique timeframes: {len(timeframes)}")
        if timeframes:
            print(f"Available timeframes: {', '.join([t[0] for t in timeframes])}")
        
        # Get top performing parameter sets
        cursor.execute("""
            SELECT symbol, timeframe, MAX(performance) as max_perf
            FROM best_parameters
            WHERE rank = 1
            GROUP BY symbol, timeframe
            ORDER BY max_perf DESC
            LIMIT 5
        """)
        top_params = cursor.fetchall()
        
        if top_params:
            print("\nTop 5 performing parameter sets:")
            for tp in top_params:
                print(f"  {tp[0]} ({tp[1]}): {tp[2]:.2%}")
    
    # Close connection
    conn.close()

def main():
    """
    Main entry point for the script when run from command line
    """
    parser = argparse.ArgumentParser(description='DMAC Database Export Utility')
    
    # Add command line arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the CSV files')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Filter by symbol')
    parser.add_argument('--timeframe', type=str, default=None,
                        help='Filter by timeframe')
    parser.add_argument('--no-format', action='store_true',
                        help='Disable timestamp and descriptive names in output filename')
    parser.add_argument('--summary', action='store_true',
                        help='Print database summary')
    
    args = parser.parse_args()
    
    # Print database summary if requested
    if args.summary:
        print_database_summary()
    
    # Export to CSV
    export_dmac_db_to_csv(
        output_dir=args.output_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        format_output=not args.no_format
    )

if __name__ == "__main__":
    main() 