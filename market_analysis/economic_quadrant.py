"""
US Economic Quadrant Determination using FRED data with simplified determination
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
from dotenv import load_dotenv
# Import database functions
from db_utils import save_economic_quadrant

# Load environment variables from .env file
load_dotenv()

# Get FRED API key from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in environment variables. Please add it to your .env file.")

fred = Fred(api_key=FRED_API_KEY)

def get_fed_balance_sheet(start_date=None, end_date=None):
    """
    Get Federal Reserve Balance Sheet data
    Using WALCL (Total Assets) as the indicator
    """
    # WALCL = Total Assets (Less Eliminations from Consolidation)
    balance_sheet = fred.get_series('WALCL', start_date=start_date, end_date=end_date)
    return balance_sheet

def get_interest_rates(start_date=None, end_date=None):
    """
    Get Federal Funds Rate data
    """
    interest_rates = fred.get_series('FEDFUNDS', start_date=start_date, end_date=end_date)
    return interest_rates

def determine_balance_sheet_trend(balance_sheet):
    """
    Determine if the balance sheet is increasing or decreasing based on recent trend
    
    Parameters:
    - balance_sheet: pandas Series with balance sheet data
    
    Returns:
    - trend: 'Increasing' or 'Decreasing'
    - details: dictionary with detailed metrics
    """
    if balance_sheet.empty:
        return 'Unavailable', {}
        
    # Calculate short-term slope (3 months)
    short_term = balance_sheet.tail(3)
    x_short = np.arange(len(short_term))
    y_short = short_term.values
    short_slope, _ = np.polyfit(x_short, y_short, 1)
    
    # Calculate medium-term slope (6 months)
    medium_term = balance_sheet.tail(6)
    x_medium = np.arange(len(medium_term))
    y_medium = medium_term.values
    medium_slope, _ = np.polyfit(x_medium, y_medium, 1)
    
    # Get current value
    current_value = balance_sheet.iloc[-1]
    
    # Calculate normalized slopes
    short_slope_pct = (short_slope * 3 / current_value) * 100
    medium_slope_pct = (medium_slope * 6 / current_value) * 100
    
    # Combined score (positive means increasing, negative means decreasing)
    trend_score = short_slope_pct + medium_slope_pct
    
    trend = 'Increasing' if trend_score > 0 else 'Decreasing'
    
    details = {
        'current_value': current_value,
        'short_term_slope_pct': short_slope_pct,
        'medium_term_slope_pct': medium_slope_pct,
        'total_score': trend_score
    }
    
    return trend, details

def determine_interest_rate_level(interest_rates):
    """
    Determine if interest rates are high or low based on fixed thresholds
    
    Parameters:
    - interest_rates: pandas Series with interest rate data
    
    Returns:
    - level: 'High' or 'Low'
    - details: dictionary with detailed metrics
    """
    if interest_rates.empty:
        return 'Unavailable', {}
        
    # Get current interest rate
    current_value = interest_rates.iloc[-1]
    
    # Simple threshold-based determination
    # Below 2.5% = Low, Above 2.5% = High
    level = 'Low' if current_value < 2.5 else 'High'
    
    details = {
        'current_value': current_value,
        'threshold': 2.5
    }
    
    return level, details

def determine_economic_quadrant():
    """
    Determine the current US economic quadrant based on
    balance sheet trend and interest rate level only
    
    Returns:
    - quadrant: A, B, C, or D
    - details: dictionary with detailed calculation metrics
    """
    # Get data for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Get economic data
    balance_sheet = get_fed_balance_sheet(start_date, end_date)
    interest_rates = get_interest_rates(start_date, end_date)
    
    # Determine balance sheet trend and interest rate level
    balance_sheet_trend, bs_details = determine_balance_sheet_trend(balance_sheet)
    interest_rate_level, ir_details = determine_interest_rate_level(interest_rates)
    
    # Determine quadrant based on the two factors
    if balance_sheet_trend == 'Increasing' and interest_rate_level == 'Low':
        quadrant = 'D'  # Growth focus
        description = 'Growth quadrant - focus on growth stocks'
    elif balance_sheet_trend == 'Increasing' and interest_rate_level == 'High':
        quadrant = 'B'  # Growth with inflation
        description = 'Growth with inflation quadrant - balanced approach with growth bias'
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'Low':
        quadrant = 'C'  # Transition to growth
        description = 'Transition to growth quadrant - balanced approach with value bias'
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'High':
        quadrant = 'A'  # Inflation fighting
        description = 'Inflation fighting quadrant - focus on value stocks'
    else:
        quadrant = 'Unknown'
        description = 'Unknown economic quadrant due to unavailable data'
    
    # Create a note describing the quadrant
    note = f"{description} - Balance Sheet: {balance_sheet_trend}, Interest Rates: {interest_rate_level}"
    
    # Convert time series data to JSON-serializable format
    balance_sheet_dict = {}
    interest_rate_dict = {}
    
    for date, value in balance_sheet.tail(30).items():
        balance_sheet_dict[date.strftime('%Y-%m-%d')] = value
        
    for date, value in interest_rates.tail(30).items():
        interest_rate_dict[date.strftime('%Y-%m-%d')] = value
    
    # Save values for database
    balance_sheet_value = balance_sheet.iloc[-1] if not balance_sheet.empty else None
    interest_rate_value = interest_rates.iloc[-1] if not interest_rates.empty else None
    
    # Save to database
    save_economic_quadrant(
        quadrant=quadrant,
        balance_sheet_trend=balance_sheet_trend,
        interest_rate_level=interest_rate_level,
        balance_sheet_value=balance_sheet_value,
        interest_rate_value=interest_rate_value,
        notes=note,
        json_data={
            'balance_sheet_data': balance_sheet_dict,
            'interest_rate_data': interest_rate_dict,
            'balance_sheet_details': bs_details,
            'interest_rate_details': ir_details
        }
    )
    
    details = {
        'balance_sheet_details': bs_details,
        'interest_rate_details': ir_details,
        'quadrant_description': description
    }
    
    return quadrant, balance_sheet_trend, interest_rate_level, details

def plot_economic_data(balance_sheet, interest_rates, bs_details=None, ir_details=None, quadrant=None):
    """
    Plot balance sheet and interest rate data with simplified visualization
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot balance sheet
    axs[0].plot(balance_sheet.index, balance_sheet.values, 'b-', label='Balance Sheet')
    
    axs[0].set_title('Federal Reserve Balance Sheet (WALCL)')
    axs[0].set_ylabel('Billions of Dollars')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot interest rates
    axs[1].plot(interest_rates.index, interest_rates.values, 'r-', label='Interest Rate')
    
    # Add threshold line for interest rates
    if ir_details and 'threshold' in ir_details:
        axs[1].axhline(y=ir_details['threshold'], color='g', linestyle='--', 
                        label=f'Threshold ({ir_details["threshold"]}%)')
    
    axs[1].set_title('Federal Funds Rate')
    axs[1].set_ylabel('Percent')
    axs[1].grid(True)
    axs[1].legend()
    
    # Add quadrant information if available
    if quadrant is not None:
        fig.suptitle(f"US Economic Indicators Analysis - Quadrant {quadrant}",
                     fontsize=16)
    
    plt.tight_layout()
    PLOTS_DIR = 'plots' # Use root plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'economic_data.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Economic quadrant plot saved to {plot_path}") # Using print instead of logger for now
    return plot_path

def analyze_us_economy():
    """
    Analyze the US economy and determine the current quadrant based on
    simplified balance sheet and interest rate criteria
    
    Returns:
    - results: dictionary with economic analysis results
    """
    # Get data for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Get economic data
    balance_sheet = get_fed_balance_sheet(start_date, end_date)
    interest_rates = get_interest_rates(start_date, end_date)
    
    # Determine quadrant
    quadrant, balance_sheet_trend, interest_rate_level, details = determine_economic_quadrant()
    
    # Plot data with simplified visualization
    plot_path = plot_economic_data(
        balance_sheet, 
        interest_rates,
        details['balance_sheet_details'], 
        details['interest_rate_details'],
        quadrant
    )
    
    # Prepare results
    results = {
        'quadrant': quadrant,
        'balance_sheet_trend': balance_sheet_trend,
        'interest_rate_level': interest_rate_level,
        'plot_path': plot_path,
        'balance_sheet_latest': balance_sheet.iloc[-1] if not balance_sheet.empty else None,
        'interest_rate_latest': interest_rates.iloc[-1] if not interest_rates.empty else None,
        'quadrant_description': details['quadrant_description']
    }
    
    return results

if __name__ == "__main__":
    print("Analyzing US economy to determine current quadrant...")
    
    # Run simplified analysis
    results = analyze_us_economy()
    
    print(f"\nCurrent US Economic Quadrant: {results['quadrant']}")
    print(f"Balance Sheet Trend: {results['balance_sheet_trend']}")
    print(f"Interest Rate Level: {results['interest_rate_level']}")
    
    if results['balance_sheet_latest'] is not None:
        print(f"Latest Balance Sheet Value: ${results['balance_sheet_latest']:.2f} billion")
    
    if results['interest_rate_latest'] is not None:
        print(f"Latest Interest Rate: {results['interest_rate_latest']:.2f}%")
    
    print(f"\nQuadrant Description: {results['quadrant_description']}")
    print(f"\nPlot saved to: {results['plot_path']}")
