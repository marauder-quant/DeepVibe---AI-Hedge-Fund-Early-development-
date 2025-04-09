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

# Set up FRED API with the provided key
FRED_API_KEY = "69d56106bb7eb53d5117414d9d6e0b9e"
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
