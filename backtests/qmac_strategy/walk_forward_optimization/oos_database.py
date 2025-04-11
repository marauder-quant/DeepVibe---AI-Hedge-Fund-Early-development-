#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database operations for QMAC strategy out-of-sample testing.
"""

import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime

# Import configuration
from backtests.qmac_strategy.walk_forward_optimization.oos_config import OOS_DB_PATH

# Set up logging
log = logging.getLogger("rich")

def initialize_oos_database():
    """
    Initialize the SQLite database for storing best out-of-sample parameters.
    
    Returns:
        None
    """
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(OOS_DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    c = conn.cursor()
    
    # Create table for OOS parameter combinations if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS best_oos_parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        rank INTEGER NOT NULL,
        buy_fast INTEGER NOT NULL,
        buy_slow INTEGER NOT NULL,
        sell_fast INTEGER NOT NULL,
        sell_slow INTEGER NOT NULL,
        avg_return REAL NOT NULL,
        avg_alpha REAL,
        alpha_success_rate REAL,
        avg_theta REAL,
        theta_success_rate REAL,
        success_rate REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        confidence_score REAL,
        num_tests INTEGER,
        date_from TEXT NOT NULL,
        date_to TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        UNIQUE(symbol, timeframe, rank, date_from, date_to)
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    log.info(f"OOS Database initialized at {OOS_DB_PATH}")

def save_best_oos_parameters_to_db(params, symbol, timeframe, start_date, end_date, summary, 
                                 confidence_score, top_n=1):
    """
    Save the best out-of-sample parameter combination to the database.
    
    Args:
        params (dict): Parameter dictionary with buy_fast, buy_slow, sell_fast, sell_slow
        symbol (str): Trading symbol
        timeframe (str): Data timeframe
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        summary (dict): Summary statistics dictionary
        confidence_score (float): Overall confidence score
        top_n (int): Number of top parameter combinations to save (usually 1 for OOS)
        
    Returns:
        None
    """
    # Initialize the database if it doesn't exist
    if not os.path.exists(OOS_DB_PATH):
        initialize_oos_database()
    
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    c = conn.cursor()
    
    # Format dates for database
    date_from = start_date.strftime('%Y-%m-%d')
    date_to = end_date.strftime('%Y-%m-%d')
    timestamp = datetime.now().isoformat()
    
    # Delete existing entries for this symbol, timeframe, date range
    c.execute('''
    DELETE FROM best_oos_parameters 
    WHERE symbol = ? AND timeframe = ? AND date_from = ? AND date_to = ?
    ''', (symbol, timeframe, date_from, date_to))
    
    # Extract parameters
    buy_fast = int(params['buy_fast'])
    buy_slow = int(params['buy_slow'])
    sell_fast = int(params['sell_fast'])
    sell_slow = int(params['sell_slow'])
    
    # Extract metrics from summary
    avg_return = float(summary.get('avg_return', 0))
    avg_alpha = float(summary.get('avg_alpha', 0))
    alpha_success_rate = float(summary.get('alpha_success_rate', 0))
    avg_theta = float(summary.get('avg_theta', 0))
    theta_success_rate = float(summary.get('theta_success_rate', 0))
    success_rate = float(summary.get('success_rate', 0))
    sharpe_ratio = float(summary.get('avg_sharpe', 0))
    max_drawdown = float(summary.get('avg_max_drawdown', 0))
    num_tests = int(summary.get('n_tests', 0))
        
    # Insert into database
    c.execute('''
    INSERT INTO best_oos_parameters 
    (symbol, timeframe, rank, buy_fast, buy_slow, sell_fast, sell_slow, 
     avg_return, avg_alpha, alpha_success_rate, avg_theta, theta_success_rate,
     success_rate, sharpe_ratio, max_drawdown, confidence_score, num_tests,
     date_from, date_to, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        symbol, timeframe, 1, buy_fast, buy_slow, sell_fast, sell_slow,
        avg_return, avg_alpha, alpha_success_rate, avg_theta, theta_success_rate,
        success_rate, sharpe_ratio, max_drawdown, confidence_score, num_tests,
        date_from, date_to, timestamp
    ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    log.info(f"Saved best OOS parameters to database for {symbol} {timeframe}")

def get_best_oos_parameters_from_db(symbol=None, timeframe=None):
    """
    Query the database for best out-of-sample parameters.
    
    Args:
        symbol (str, optional): Filter by symbol
        timeframe (str, optional): Filter by timeframe
        
    Returns:
        pd.DataFrame: DataFrame containing parameter combinations
    """
    # Check if database exists
    if not os.path.exists(OOS_DB_PATH):
        log.warning(f"OOS Database not found at {OOS_DB_PATH}")
        return pd.DataFrame()
    
    # Connect to database
    conn = sqlite3.connect(OOS_DB_PATH)
    
    # Build query based on filters
    query = "SELECT * FROM best_oos_parameters"
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
    
    return df 