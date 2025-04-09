"""
US Economic Quadrant Determination with Advanced Classification Model
Includes Composite Macro Score (CMS) using multiple economic indicators
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
import json

# Import database functions
from db_utils import save_economic_quadrant, add_column_if_not_exists

# Set up FRED API with the provided key
FRED_API_KEY = "69d56106bb7eb53d5117414d9d6e0b9e"
fred = Fred(api_key=FRED_API_KEY)

# Define FRED series IDs for economic indicators
SERIES_IDS = {
    'balance_sheet': 'WALCL',       # Federal Reserve Assets
    'interest_rates': 'FEDFUNDS',   # Effective Federal Funds Rate
    'jobs': 'PAYEMS',               # Nonfarm Payrolls
    'spending': 'PCE',              # Personal Consumption Expenditures
    'inflation': 'CPIAUCSL',        # Consumer Price Index
    'tariffs': None                 # Disable tariffs for now
}

# Define weights for composite macro score
CMS_WEIGHTS = {
    'jobs': 0.4,
    'spending': 0.4, 
    'inflation': 0.2
    # Removed tariffs from the weights
}

def get_fred_data(series_id, start_date=None, end_date=None):
    """
    Get data from FRED for a specific series
    Handle potential errors gracefully
    """
    if series_id is None:
        return pd.Series()
        
    try:
        return fred.get_series(series_id, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"Error fetching {series_id}: {str(e)}")
        # Return empty series if data fetching fails
        return pd.Series()

def compute_zscore(series, window=60):
    """
    Calculate Z-score normalization for a time series
    
    Parameters:
    - series: pandas Series with time series data
    - window: number of periods for rolling window (default: 60)
    
    Returns:
    - Z-scores: (Current Value - Historical Mean) / Standard Deviation
    """
    if series.empty:
        return pd.Series()
        
    # Calculate rolling mean and standard deviation
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    # Calculate Z-scores
    z_scores = (series - mean) / std
    
    return z_scores

def determine_balance_sheet_trend(balance_sheet):
    """
    Determine if the balance sheet is increasing or decreasing
    
    Parameters:
    - balance_sheet: pandas Series with balance sheet data
    
    Returns:
    - trend: 'Increasing' or 'Decreasing'
    - details: dictionary with detailed metrics
    """
    if balance_sheet.empty:
        return 'Unavailable', {}
    
    # Calculate 3-month moving average rate of change
    pct_change = balance_sheet.pct_change(periods=3)
    
    # Get current trend
    current_change = pct_change.iloc[-1]
    
    # Determine trend
    trend = 'Increasing' if current_change > 0 else 'Decreasing'
    
    details = {
        'current_value': balance_sheet.iloc[-1],
        'pct_change_3m': current_change,
        'raw_data': balance_sheet.tail(12).to_dict()
    }
    
    return trend, details

def determine_interest_rate_level(interest_rates):
    """
    Determine if interest rates are high or low based on historical median
    
    Parameters:
    - interest_rates: pandas Series with interest rate data
    
    Returns:
    - level: 'High' or 'Low'
    - details: dictionary with detailed metrics
    """
    if interest_rates.empty:
        return 'Unavailable', {}
    
    # Calculate 5-year (60-month) historical median
    historical_median = interest_rates.rolling(window=60).median().iloc[-1]
    
    # Get current interest rate
    current_value = interest_rates.iloc[-1]
    
    # Determine level compared to historical median
    level = 'High' if current_value > historical_median else 'Low'
    
    details = {
        'current_value': current_value,
        'historical_median': historical_median,
        'raw_data': interest_rates.tail(12).to_dict()
    }
    
    return level, details

def calculate_composite_macro_score(z_scores, weights=CMS_WEIGHTS):
    """
    Calculate the Composite Macro Score (CMS) using weighted Z-scores
    
    Parameters:
    - z_scores: dictionary with Z-scores for each indicator
    - weights: dictionary with weights for each indicator
    
    Returns:
    - cms: weighted average of Z-scores
    - details: dictionary with detailed calculation
    """
    # Check if we have at least some data to work with
    valid_indicators = []
    for indicator in weights.keys():
        if indicator in z_scores and not z_scores[indicator].empty and len(z_scores[indicator]) > 0:
            valid_indicators.append(indicator)
    
    if not valid_indicators:
        return None, {}
    
    # Calculate weighted sum of Z-scores with available data
    cms = 0
    total_weight = 0
    component_scores = {}
    
    for indicator in valid_indicators:
        weight = weights[indicator]
        if not pd.isna(z_scores[indicator].iloc[-1]):
            z_value = z_scores[indicator].iloc[-1]
            contribution = weight * z_value
            cms += contribution
            total_weight += weight
            
            component_scores[indicator] = {
                'zscore': z_value,
                'weight': weight,
                'contribution': contribution
            }
    
    # Normalize by total weight if not all indicators are available
    if total_weight > 0 and total_weight < 1.0:
        cms = cms / total_weight
    
    # Prepare detailed calculation
    details = {
        'cms': cms,
        'component_scores': component_scores,
        'total_weight': total_weight,
        'valid_indicators': valid_indicators
    }
    
    return cms, details

def determine_regime_strength(cms, historical_cms):
    """
    Determine the strength of the current regime based on historical percentile
    
    Parameters:
    - cms: current Composite Macro Score
    - historical_cms: pandas Series with historical CMS values
    
    Returns:
    - percentile: percentile rank (0-100) of current CMS
    - strength: qualitative description ('Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong')
    """
    if historical_cms.empty or pd.isna(cms) or cms is None:
        return None, 'Unknown'
    
    # Handle edge case with too few data points
    if len(historical_cms) < 5:
        # Not enough historical data for percentile calculation
        if cms > 1.0:
            strength = 'Strong'
        elif cms < -1.0:
            strength = 'Weak'
        else:
            strength = 'Neutral'
        return 50, strength  # Default to middle percentile
    
    # Calculate percentile rank of current CMS
    percentile = 100 * sum(historical_cms < cms) / len(historical_cms)
    
    # Assign strength category
    if percentile < 20:
        strength = 'Very Weak'
    elif percentile < 40:
        strength = 'Weak'
    elif percentile < 60:
        strength = 'Neutral'
    elif percentile < 80:
        strength = 'Strong'
    else:
        strength = 'Very Strong'
    
    return percentile, strength

def determine_advanced_economic_quadrant():
    """
    Determine the current US economic quadrant with advanced indicators
    
    Returns:
    - quadrant: A, B, C, or D
    - details: dictionary with detailed calculation metrics
    """
    # Get data for the last 5 years (need sufficient history for Z-scores)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch data from FRED
    data = {}
    for indicator, series_id in SERIES_IDS.items():
        if series_id is not None:
            data[indicator] = get_fred_data(series_id, start_date, end_date)
        else:
            data[indicator] = pd.Series()  # Empty series for disabled indicators
    
    # Calculate derived metrics only if we have the base data
    if not data['jobs'].empty:
        # Jobs growth (month-over-month percentage change)
        data['jobs_growth'] = data['jobs'].pct_change(periods=1)
    else:
        data['jobs_growth'] = pd.Series()

    if not data['inflation'].empty:
        # Inflation (year-over-year percentage change)
        data['inflation_yoy'] = data['inflation'].pct_change(periods=12)
    else:
        data['inflation_yoy'] = pd.Series()
    
    # Calculate Z-scores for key indicators
    z_scores = {}
    for indicator, source in [
        ('jobs', 'jobs_growth'),
        ('spending', 'spending'),
        ('inflation', 'inflation_yoy')
    ]:
        if source in data and not data[source].empty and len(data[source]) > 5:
            z_scores[indicator] = compute_zscore(data[source])
        else:
            z_scores[indicator] = pd.Series()  # Empty series if data is unavailable
    
    # Determine balance sheet trend and interest rate level
    balance_sheet_trend, bs_details = determine_balance_sheet_trend(data['balance_sheet'])
    interest_rate_level, ir_details = determine_interest_rate_level(data['interest_rates'])
    
    # Calculate Composite Macro Score
    cms, cms_details = calculate_composite_macro_score(z_scores)
    
    # Calculate historical CMS values for regime strength determination
    historical_cms = pd.Series()
    historical_cms_values = []
    for i in range(60):  # Use last 60 data points for historical comparison
        if i >= len(z_scores['jobs']) - 1:
            break
            
        point_z_scores = {
            indicator: z_series.iloc[-(i+1)] 
            for indicator, z_series in z_scores.items() 
            if len(z_series) > i and not pd.isna(z_series.iloc[-(i+1)])
        }
        
        # Skip if missing data
        if len(point_z_scores) < len(CMS_WEIGHTS):
            continue
            
        # Calculate CMS for this historical point
        point_cms, _ = calculate_composite_macro_score({
            indicator: pd.Series([value]) for indicator, value in point_z_scores.items()
        })
        
        if point_cms is not None:
            historical_cms_values.append(point_cms)

    historical_cms = pd.Series(historical_cms_values)
    
    # Determine regime strength
    cms_percentile, regime_strength = determine_regime_strength(cms, historical_cms)
    
    # Determine quadrant based on balance sheet trend and interest rate level
    if balance_sheet_trend == 'Increasing' and interest_rate_level == 'Low':
        quadrant = 'D'  # Most bullish (Growth focus)
        description = 'Growth quadrant - Optimal for risk assets (equities, crypto)'
    elif balance_sheet_trend == 'Increasing' and interest_rate_level == 'High':
        quadrant = 'B'  # Growth with inflation
        description = 'Growth with inflation quadrant - Balanced approach with growth bias'
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'Low':
        quadrant = 'C'  # Transition to growth
        description = 'Transition to growth quadrant - Balanced approach with value bias'
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'High':
        quadrant = 'A'  # Inflation fighting
        description = 'Inflation fighting quadrant - Defensive positioning (bonds, cash)'
    else:
        quadrant = 'Unknown'
        description = 'Unknown economic quadrant due to unavailable data'
    
    # Create a note describing the quadrant and CMS
    note = f"{description} - Balance Sheet: {balance_sheet_trend}, Interest Rates: {interest_rate_level}, Regime Strength: {regime_strength} (CMS: {cms:.2f})"
    
    # Convert time series data to JSON-serializable format
    balance_sheet_dict = {}
    interest_rate_dict = {}
    
    for date, value in data['balance_sheet'].tail(30).items():
        date_str = date.strftime('%Y-%m-%d')
        balance_sheet_dict[date_str] = value
        
    for date, value in data['interest_rates'].tail(30).items():
        date_str = date.strftime('%Y-%m-%d')
        interest_rate_dict[date_str] = value
    
    # Process detailed data for JSON
    bs_details_json = process_dict_for_json(bs_details)
    ir_details_json = process_dict_for_json(ir_details)

    # Create JSON data for all indicators
    indicator_data = {}
    for indicator in ['jobs_growth', 'spending', 'inflation_yoy']:
        if indicator in data and not data[indicator].empty:
            indicator_dict = {}
            for date, value in data[indicator].tail(30).items():
                date_str = date.strftime('%Y-%m-%d')
                indicator_dict[date_str] = value
            indicator_data[indicator] = indicator_dict

    # Process z-scores for JSON
    z_scores_json = {}
    for indicator, z_series in z_scores.items():
        if not z_series.empty and len(z_series) > 0:
            z_scores_dict = {}
            for date, value in z_series.tail(5).items():
                if not pd.isna(value):  # Skip NaN values
                    date_str = date.strftime('%Y-%m-%d')
                    z_scores_dict[date_str] = value
            z_scores_json[indicator] = z_scores_dict

    # Prepare JSON data for storage
    json_data = {
        'balance_sheet_data': balance_sheet_dict,
        'interest_rate_data': interest_rate_dict,
        'balance_sheet_details': bs_details_json,
        'interest_rate_details': ir_details_json,
        'cms': cms,
        'cms_details': cms_details,
        'regime_strength': regime_strength,
        'cms_percentile': cms_percentile,
        'z_scores': z_scores_json,
        'indicators': indicator_data
    }
    
    # Save values for database
    balance_sheet_value = data['balance_sheet'].iloc[-1] if not data['balance_sheet'].empty else None
    interest_rate_value = data['interest_rates'].iloc[-1] if not data['interest_rates'].empty else None
    
    # Update database schema if needed
    add_column_if_not_exists('economic_quadrants', 'composite_macro_score', 'REAL')
    add_column_if_not_exists('economic_quadrants', 'regime_strength', 'TEXT')
    add_column_if_not_exists('economic_quadrants', 'cms_percentile', 'REAL')
    
    # Create a custom save function that includes the new fields
    def save_advanced_economic_quadrant(
        quadrant, balance_sheet_trend, interest_rate_level, 
        balance_sheet_value=None, interest_rate_value=None,
        composite_macro_score=None, regime_strength=None, cms_percentile=None,
        notes=None, json_data=None):
        """Save advanced economic quadrant analysis to database"""
        from db_utils import ensure_db_exists
        import sqlite3
        from datetime import datetime
        import json as json_lib
        
        ensure_db_exists()
        
        conn = sqlite3.connect('data/market_analysis.db')
        cursor = conn.cursor()
        
        analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert JSON data to string if it's a dict
        if isinstance(json_data, dict):
            json_data = json_lib.dumps(json_data)
        
        cursor.execute('''
        INSERT INTO economic_quadrants 
        (analysis_date, quadrant, balance_sheet_trend, interest_rate_level,
         balance_sheet_value, interest_rate_value, composite_macro_score,
         regime_strength, cms_percentile, analysis_notes, json_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (analysis_date, quadrant, balance_sheet_trend, interest_rate_level,
              balance_sheet_value, interest_rate_value, composite_macro_score,
              regime_strength, cms_percentile, notes, json_data))
        
        conn.commit()
        conn.close()
        
        return True

    # Save to database using the enhanced function
    save_advanced_economic_quadrant(
        quadrant=quadrant,
        balance_sheet_trend=balance_sheet_trend,
        interest_rate_level=interest_rate_level,
        balance_sheet_value=balance_sheet_value,
        interest_rate_value=interest_rate_value,
        composite_macro_score=cms,
        regime_strength=regime_strength,
        cms_percentile=cms_percentile,
        notes=note,
        json_data=json_data
    )
    
    details = {
        'balance_sheet_details': bs_details,
        'interest_rate_details': ir_details,
        'cms': cms,
        'cms_details': cms_details,
        'regime_strength': regime_strength,
        'quadrant_description': description
    }
    
    return quadrant, balance_sheet_trend, interest_rate_level, cms, regime_strength, details

def plot_advanced_economic_data(data, quadrant=None, cms=None, regime_strength=None):
    """
    Plot economic data with advanced visualization including CMS
    
    Parameters:
    - data: dictionary with economic data
    - quadrant: current economic quadrant
    - cms: composite macro score
    - regime_strength: strength of the current regime
    
    Returns:
    - plot_path: path to the saved plot
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Balance Sheet
    axs[0, 0].plot(data['balance_sheet'].index, data['balance_sheet'].values, 'b-', label='Balance Sheet')
    axs[0, 0].set_title('Federal Reserve Balance Sheet (WALCL)')
    axs[0, 0].set_ylabel('Billions of Dollars')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Plot 2: Interest Rates
    axs[0, 1].plot(data['interest_rates'].index, data['interest_rates'].values, 'r-', label='Interest Rate')
    
    # Add median line for interest rates
    median = data['interest_rates'].rolling(window=60).median()
    axs[0, 1].plot(median.index, median.values, 'g--', label='5-Year Median')
    
    axs[0, 1].set_title('Federal Funds Rate')
    axs[0, 1].set_ylabel('Percent')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Plot 3: Jobs Growth
    axs[1, 0].plot(data['jobs_growth'].index, data['jobs_growth'].values * 100, 'purple', label='Jobs Growth (MoM)')
    axs[1, 0].set_title('Nonfarm Payrolls (MoM % Change)')
    axs[1, 0].set_ylabel('Percent')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Plot 4: Consumer Spending
    axs[1, 1].plot(data['spending'].index, data['spending'].values, 'orange', label='Consumer Spending')
    axs[1, 1].set_title('Personal Consumption Expenditures')
    axs[1, 1].set_ylabel('Billions of Dollars')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Plot 5: Inflation
    axs[2, 0].plot(data['inflation_yoy'].index, data['inflation_yoy'].values * 100, 'brown', label='Inflation (YoY)')
    axs[2, 0].set_title('Consumer Price Index (YoY % Change)')
    axs[2, 0].set_ylabel('Percent')
    axs[2, 0].grid(True)
    axs[2, 0].legend()
    
    # Plot 6: Tariffs
    axs[2, 1].plot(data['tariffs'].index, data['tariffs'].values, 'teal', label='Customs Duties')
    axs[2, 1].set_title('Customs Duties')
    axs[2, 1].set_ylabel('Millions of Dollars')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    
    # Add quadrant information
    if quadrant is not None:
        fig.suptitle(f"Economic Regime Analysis - Quadrant {quadrant} | CMS: {cms:.2f} | Strength: {regime_strength}",
                     fontsize=16)
    
    plt.tight_layout()
    
    # Add quadrant descriptions
    figtext_x = 0.5
    figtext_y = 0.01
    quadrant_descriptions = {
        'A': 'Quadrant A: Decreasing Balance Sheet, High Rates - Defensive positioning',
        'B': 'Quadrant B: Increasing Balance Sheet, High Rates - Balanced with growth bias',
        'C': 'Quadrant C: Decreasing Balance Sheet, Low Rates - Balanced with value bias',
        'D': 'Quadrant D: Increasing Balance Sheet, Low Rates - Optimal for risk assets'
    }
    
    if quadrant in quadrant_descriptions:
        plt.figtext(figtext_x, figtext_y, quadrant_descriptions[quadrant], 
                   ha='center', fontsize=12, bbox={'facecolor':'lightgrey', 'alpha':0.5})
    
    # Save the plot
    plot_path = 'advanced_economic_data.png'
    plt.savefig(plot_path)
    
    return plot_path

def analyze_us_economy_advanced():
    """
    Analyze the US economy with advanced metrics and determine the current quadrant
    
    Returns:
    - results: dictionary with advanced economic analysis results
    """
    # Get data for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Get economic data
    data = {}
    for indicator, series_id in SERIES_IDS.items():
        data[indicator] = get_fred_data(series_id, start_date, end_date)
    
    # Calculate derived metrics
    data['jobs_growth'] = data['jobs'].pct_change(periods=1)
    data['inflation_yoy'] = data['inflation'].pct_change(periods=12)
    
    # Determine quadrant with advanced metrics
    quadrant, balance_sheet_trend, interest_rate_level, cms, regime_strength, details = determine_advanced_economic_quadrant()
    
    # Plot data with advanced visualization
    plot_path = plot_advanced_economic_data(
        data=data,
        quadrant=quadrant,
        cms=cms,
        regime_strength=regime_strength
    )
    
    # Prepare results
    results = {
        'quadrant': quadrant,
        'balance_sheet_trend': balance_sheet_trend,
        'interest_rate_level': interest_rate_level,
        'composite_macro_score': cms,
        'regime_strength': regime_strength,
        'plot_path': plot_path,
        'balance_sheet_latest': data['balance_sheet'].iloc[-1] if not data['balance_sheet'].empty else None,
        'interest_rate_latest': data['interest_rates'].iloc[-1] if not data['interest_rates'].empty else None,
        'quadrant_description': details['quadrant_description'],
        'cms_details': details['cms_details'] if 'cms_details' in details else None
    }
    
    return results

# Helper function to convert dates in dictionaries for JSON serialization
def convert_timestamps_for_json(data):
    """Convert timestamps to strings for JSON serialization"""
    if isinstance(data, dict):
        return {k: convert_timestamps_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps_for_json(item) for item in data]
    elif isinstance(data, pd.Timestamp):
        return data.strftime('%Y-%m-%d')
    else:
        return data

# Update the function to convert dates in raw_data dictionaries
def process_dict_for_json(input_dict):
    """Process dictionary to make it JSON serializable"""
    if input_dict is None:
        return None
        
    result = {}
    for key, value in input_dict.items():
        if isinstance(key, pd.Timestamp):
            key = key.strftime('%Y-%m-%d')
            
        if isinstance(value, dict):
            result[key] = process_dict_for_json(value)
        elif isinstance(value, pd.Timestamp):
            result[key] = value.strftime('%Y-%m-%d')
        else:
            result[key] = value
            
    return result

if __name__ == "__main__":
    print("Analyzing US economy with advanced metrics to determine current quadrant...")
    
    # Run advanced analysis
    results = analyze_us_economy_advanced()
    
    print(f"\nCurrent US Economic Quadrant: {results['quadrant']}")
    print(f"Balance Sheet Trend: {results['balance_sheet_trend']}")
    print(f"Interest Rate Level: {results['interest_rate_level']}")
    print(f"Composite Macro Score (CMS): {results['composite_macro_score']:.2f}")
    print(f"Regime Strength: {results['regime_strength']}")
    
    if results['balance_sheet_latest'] is not None:
        print(f"Latest Balance Sheet Value: ${results['balance_sheet_latest']:.2f} billion")
    
    if results['interest_rate_latest'] is not None:
        print(f"Latest Interest Rate: {results['interest_rate_latest']:.2f}%")
    
    print(f"\nQuadrant Description: {results['quadrant_description']}")
    print(f"\nPlot saved to: {results['plot_path']}")
