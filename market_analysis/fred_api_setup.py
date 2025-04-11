"""
FRED API Setup and Economic Quadrant Analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get FRED API key from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in environment variables. Please add it to your .env file.")

fred = Fred(api_key=FRED_API_KEY)

def test_fred_connection():
    """Test the FRED API connection by retrieving a simple series"""
    try:
        # Get Federal Funds Rate data
        ffr = fred.get_series('FEDFUNDS')
        print(f"Successfully connected to FRED API")
        print(f"Federal Funds Rate (last 5 values):")
        print(ffr.tail(5))
        return True
    except Exception as e:
        print(f"Error connecting to FRED API: {e}")
        return False

if __name__ == "__main__":
    print("Testing FRED API connection...")
    test_fred_connection()
