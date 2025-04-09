"""
US Economic Quadrant Determination using FRED data with point scoring system
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
# Import database functions
from db_utils import save_economic_quadrant

# Set up FRED API with the provided key
FRED_API_KEY = "69d56106bb7eb53d5117414d9d6e0b9e"
fred = Fred(api_key=FRED_API_KEY)

# Define default sentiment multipliers (can be overridden when calling functions)
# Range: 0.1-0.99 for bearish, 1.0 for neutral, 1.01-2.0 for bullish
DEFAULT_JOBS_SENTIMENT = 1.0  # Neutral by default
DEFAULT_CONSUMER_SPENDING_SENTIMENT = 1.0  # Neutral by default

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

def get_jobs_data(start_date=None, end_date=None):
    """
    Get non-farm payroll employment data
    Using PAYEMS (All Employees: Total Nonfarm Payrolls)
    """
    jobs_data = fred.get_series('PAYEMS', start_date=start_date, end_date=end_date)
    return jobs_data

def get_consumer_spending(start_date=None, end_date=None):
    """
    Get consumer spending data
    Using PCE (Personal Consumption Expenditures)
    """
    consumer_spending = fred.get_series('PCE', start_date=start_date, end_date=end_date)
    return consumer_spending

def get_consumer_sentiment(start_date=None, end_date=None):
    """
    Get University of Michigan Consumer Sentiment data from FRED
    Using UMCSENT series
    
    Parameters:
    - start_date: start date for data retrieval
    - end_date: end date for data retrieval
    
    Returns:
    - sentiment_data: pandas Series with consumer sentiment index
    """
    sentiment_data = fred.get_series('UMCSENT', start_date=start_date, end_date=end_date)
    return sentiment_data

def get_job_sentiment(start_date=None, end_date=None):
    """
    Get job market sentiment data from FRED
    Using ICSA (Initial Claims for Unemployment Insurance) as a proxy for job market health
    Note: Lower values are better for job market health, so we invert this indicator
    
    Parameters:
    - start_date: start date for data retrieval
    - end_date: end date for data retrieval
    
    Returns:
    - job_sentiment_data: pandas Series with job market sentiment proxy
    """
    # Initial Claims is a weekly series of new unemployment claims
    # Higher values indicate worse job market conditions
    initial_claims = fred.get_series('ICSA', start_date=start_date, end_date=end_date)
    
    # Since this is a weekly series, and we want to use it as a sentiment indicator,
    # we'll take the 4-week moving average to smooth it out
    if not initial_claims.empty:
        initial_claims = initial_claims.rolling(window=4).mean().dropna()
    
    return initial_claims

def calculate_sentiment_multiplier(sentiment_data, baseline=None, sensitivity=1.0):
    """
    Calculate a sentiment multiplier based on sentiment data
    
    Parameters:
    - sentiment_data: pandas Series with sentiment index data
    - baseline: the neutral sentiment level (if None, uses 2-year moving average)
    - sensitivity: how strongly sentiment affects the multiplier (default: 1.0)
    
    Returns:
    - multiplier: sentiment multiplier between 0.1 and 2.0
    - latest_sentiment: the latest sentiment value used for calculation
    - baseline_value: the baseline used for calculation
    """
    if sentiment_data.empty:
        return 1.0, None, None  # Neutral if no data
    
    # Get the latest sentiment value
    latest_sentiment = sentiment_data.iloc[-1]
    
    # If baseline is None, calculate 2-year moving average
    if baseline is None:
        # Use 24 months as 2 years if we have enough data
        window_size = min(24, len(sentiment_data))
        baseline_value = sentiment_data.rolling(window=window_size).mean().iloc[-1]
    else:
        baseline_value = baseline
    
    # Calculate deviation from baseline as a percentage
    deviation_pct = (latest_sentiment - baseline_value) / baseline_value
    
    # Apply sensitivity factor
    adjusted_deviation = deviation_pct * sensitivity
    
    # Calculate multiplier (1.0 is neutral)
    # Values above baseline create multipliers > 1.0
    # Values below baseline create multipliers < 1.0
    multiplier = 1.0 + adjusted_deviation
    
    # Constrain the multiplier between 0.1 and 2.0
    multiplier = max(min(multiplier, 2.0), 0.1)
    
    return multiplier, latest_sentiment, baseline_value

def calculate_balance_sheet_score(balance_sheet):
    """
    Calculate a score for the balance sheet trend based on:
    1. Current value vs 2-year moving average
    2. Short-term slope (3 months)
    3. Medium-term slope (6 months)
    
    Parameters:
    - balance_sheet: pandas Series with balance sheet data
    
    Returns:
    - score: numeric value indicating balance sheet trend strength
    - details: dictionary with detailed metrics
    """
    # Calculate moving average
    moving_avg = balance_sheet.rolling(window=24).mean()
    
    # Calculate percent deviation from moving average
    current_value = balance_sheet.iloc[-1]
    current_ma = moving_avg.iloc[-1]
    percent_deviation = ((current_value - current_ma) / current_ma) * 100
    
    # Calculate short-term slope (3 months)
    short_term = balance_sheet.tail(3)
    x_short = np.arange(len(short_term))
    y_short = short_term.values
    short_slope, _ = np.polyfit(x_short, y_short, 1)
    
    # Normalize short-term slope to a percentage of current value
    short_slope_pct = (short_slope * 3 / current_value) * 100
    
    # Calculate medium-term slope (6 months)
    medium_term = balance_sheet.tail(6)
    x_medium = np.arange(len(medium_term))
    y_medium = medium_term.values
    medium_slope, _ = np.polyfit(x_medium, y_medium, 1)
    
    # Normalize medium-term slope to a percentage of current value
    medium_slope_pct = (medium_slope * 6 / current_value) * 100
    
    # Calculate score (positive score means increasing, negative means decreasing)
    score = percent_deviation + short_slope_pct + medium_slope_pct
    
    details = {
        'current_value': current_value,
        'moving_avg': current_ma,
        'percent_deviation': percent_deviation,
        'short_term_slope_pct': short_slope_pct,
        'medium_term_slope_pct': medium_slope_pct,
        'total_score': score
    }
    
    return score, details

def calculate_interest_rate_score(interest_rates):
    """
    Calculate a score for interest rates based on absolute thresholds:
    - Below 2.5% = Accommodative (stimulative, good for market)
    - 2.5-3.5% = Neutral
    - Above 3.5% = Restrictive (bad for market)
    
    Parameters:
    - interest_rates: pandas Series with interest rate data
    
    Returns:
    - score: numeric value indicating interest rate level
    - details: dictionary with detailed metrics
    """
    # Get current interest rate
    current_value = interest_rates.iloc[-1]
    
    # Calculate score based on absolute thresholds
    if current_value < 2.5:
        # Accommodative (good for market) - score from 10 to 20 (lower is better)
        base_score = -20 + (current_value / 2.5) * 10
    elif current_value <= 3.5:
        # Neutral - score from -10 to 10
        base_score = -10 + (current_value - 2.5) * 20
    else:
        # Restrictive (bad for market) - score from 10 to 20 or higher
        base_score = 10 + (current_value - 3.5) * 10
    
    # Calculate short-term trend (3 months)
    short_term = interest_rates.tail(3)
    x_short = np.arange(len(short_term))
    y_short = short_term.values
    short_slope, _ = np.polyfit(x_short, y_short, 1)
    
    # Normalize short-term slope (different approach for interest rates)
    short_slope_pct = short_slope * 12  # Annualized rate change
    
    # Calculate medium-term trend (6 months)
    medium_term = interest_rates.tail(6)
    x_medium = np.arange(len(medium_term))
    y_medium = medium_term.values
    medium_slope, _ = np.polyfit(x_medium, y_medium, 1)
    
    # Normalize medium-term slope
    medium_slope_pct = medium_slope * 12  # Annualized rate change
    
    # Incorporate trends into the score
    # Rising rates are worse for the market, falling rates are better
    trend_score = short_slope_pct + medium_slope_pct
    
    # Calculate total score (base score adjusted by trend)
    score = base_score + trend_score
    
    # Determine the rate regime based on thresholds
    if current_value < 2.5:
        rate_regime = "Accommodative"
    elif current_value <= 3.5:
        rate_regime = "Neutral"
    else:
        rate_regime = "Restrictive"
    
    details = {
        'current_value': current_value,
        'base_score': base_score,
        'trend_score': trend_score,
        'total_score': score,
        'rate_regime': rate_regime
    }
    
    return score, details

def calculate_jobs_score(jobs_data):
    """
    Calculate a score for jobs growth/shrinking based on:
    1. 3-month rate of change
    2. 6-month rate of change
    3. 12-month rate of change
    
    Parameters:
    - jobs_data: pandas Series with jobs data
    
    Returns:
    - score: numeric value indicating jobs trend (-10 to +10)
    - details: dictionary with detailed metrics
    """
    # Ensure we have enough data
    if len(jobs_data) < 12:
        raise ValueError("Not enough data to calculate jobs score")
    
    # Get current value
    current_value = jobs_data.iloc[-1]
    
    # Calculate 3-month change
    three_month_ago = jobs_data.iloc[-3]
    three_month_change_pct = ((current_value - three_month_ago) / three_month_ago) * 100
    
    # Calculate 6-month change
    six_month_ago = jobs_data.iloc[-6]
    six_month_change_pct = ((current_value - six_month_ago) / six_month_ago) * 100
    
    # Calculate 12-month change
    twelve_month_ago = jobs_data.iloc[-12]
    twelve_month_change_pct = ((current_value - twelve_month_ago) / twelve_month_ago) * 100
    
    # Calculate trend score
    # Weight the changes: 3-month (40%), 6-month (30%), 12-month (30%)
    trend_score = (0.4 * three_month_change_pct * 4) + (0.3 * six_month_change_pct * 2) + (0.3 * twelve_month_change_pct)
    
    # Cap the score within -10 to +10 range
    score = max(min(trend_score, 10), -10)
    
    # Determine trend description
    if score > 2:
        trend = "Strong Growth"
    elif score > 0:
        trend = "Moderate Growth"
    elif score > -2:
        trend = "Stagnant"
    else:
        trend = "Shrinking"
    
    details = {
        'current_value': current_value,
        'three_month_change_pct': three_month_change_pct,
        'six_month_change_pct': six_month_change_pct,
        'twelve_month_change_pct': twelve_month_change_pct,
        'total_score': score,
        'trend': trend
    }
    
    return score, details

def calculate_consumer_spending_score(consumer_spending):
    """
    Calculate a score for consumer spending based on:
    1. 3-month rate of change
    2. 6-month rate of change
    
    Parameters:
    - consumer_spending: pandas Series with consumer spending data
    
    Returns:
    - score: numeric value indicating consumer spending trend (-10 to +10)
    - details: dictionary with detailed metrics
    """
    # Ensure we have enough data
    if len(consumer_spending) < 6:
        raise ValueError("Not enough data to calculate consumer spending score")
    
    # Get current value
    current_value = consumer_spending.iloc[-1]
    
    # Calculate 3-month change
    three_month_ago = consumer_spending.iloc[-3]
    three_month_change_pct = ((current_value - three_month_ago) / three_month_ago) * 100
    
    # Calculate 6-month change
    six_month_ago = consumer_spending.iloc[-6]
    six_month_change_pct = ((current_value - six_month_ago) / six_month_ago) * 100
    
    # Calculate trend score
    # Weight the changes: 3-month (60%), 6-month (40%)
    trend_score = (0.6 * three_month_change_pct * 3) + (0.4 * six_month_change_pct * 1.5)
    
    # Cap the score within -10 to +10 range
    score = max(min(trend_score, 10), -10)
    
    # Determine trend description
    if score > 2:
        trend = "Strong Increase"
    elif score > 0:
        trend = "Moderate Increase"
    elif score > -2:
        trend = "Flat"
    else:
        trend = "Decreasing"
    
    details = {
        'current_value': current_value,
        'three_month_change_pct': three_month_change_pct,
        'six_month_change_pct': six_month_change_pct,
        'total_score': score,
        'trend': trend
    }
    
    return score, details

def get_sentiment_description(sentiment_value):
    """
    Convert a sentiment multiplier to a descriptive string
    
    Parameters:
    - sentiment_value: sentiment multiplier (0.1-2.0)
    
    Returns:
    - description: text description of the sentiment
    """
    if sentiment_value < 0.5:
        return "Very Bearish"
    elif sentiment_value < 0.8:
        return "Bearish"
    elif sentiment_value < 0.95:
        return "Slightly Bearish"
    elif sentiment_value <= 1.05:
        return "Neutral"
    elif sentiment_value < 1.3:
        return "Slightly Bullish"
    elif sentiment_value < 1.7:
        return "Bullish"
    else:
        return "Very Bullish"

def calculate_market_score(balance_sheet_score, interest_rate_score, jobs_score=0, consumer_spending_score=0,
                          jobs_sentiment=DEFAULT_JOBS_SENTIMENT, consumer_spending_sentiment=DEFAULT_CONSUMER_SPENDING_SENTIMENT):
    """
    Calculate a unified market score based on all economic indicators:
    - Balance sheet trend
    - Interest rate level
    - Jobs growth/shrinking (with sentiment trend modifier)
    - Consumer spending trend (with sentiment trend modifier)
    
    Parameters:
    - balance_sheet_score: score representing balance sheet trend
    - interest_rate_score: score representing interest rate level
    - jobs_score: score representing jobs trend
    - consumer_spending_score: score representing consumer spending trend
    - jobs_sentiment: value (0.1-2.0) reflecting sentiment on jobs
    - consumer_spending_sentiment: value (0.1-2.0) reflecting sentiment on spending
    
    Returns:
    - market_score: a score between 0 and 100 indicating overall market conditions
    - components: dictionary with individual score components
    """
    # Normalize balance sheet score (typically ranges from -10 to +10)
    # A positive balance sheet score (expansion) is good for the market
    normalized_bs_score = max(min(balance_sheet_score, 10), -10)  # Cap at -10 to 10
    bs_contribution = (normalized_bs_score + 10) * 1.25  # Convert to 0-25 scale
    
    # Normalize interest rate score (typically ranges from -20 to +20)
    # A negative interest rate score (lower rates) is good for the market
    normalized_ir_score = max(min(interest_rate_score, 20), -20)  # Cap at -20 to 20
    ir_contribution = (20 - normalized_ir_score) * 0.625  # Convert to 0-25 scale
    
    # Simplify sentiment handling - just use sentiment as a simple modifier
    # Validate and constrain jobs sentiment value
    jobs_sentiment = max(min(jobs_sentiment, 2.0), 0.1)
    
    # Apply a simpler sentiment adjustment
    # If sentiment is neutral (1.0), no change
    # If sentiment is bearish (<1.0), slightly reduce the score
    # If sentiment is bullish (>1.0), slightly increase the score
    adjusted_jobs_score = jobs_score
    
    # Apply sentiment trend as a simple modifier (+/- 1 point maximum)
    if jobs_sentiment < 0.9:  # Significantly bearish
        adjusted_jobs_score -= 1
    elif jobs_sentiment < 1.0:  # Slightly bearish
        adjusted_jobs_score -= 0.5
    elif jobs_sentiment > 1.1:  # Significantly bullish
        adjusted_jobs_score += 1
    elif jobs_sentiment > 1.0:  # Slightly bullish
        adjusted_jobs_score += 0.5
    # If sentiment is 1.0 (neutral), no adjustment
    
    # Cap the adjusted score
    adjusted_jobs_score = max(min(adjusted_jobs_score, 10), -10)
    
    # Convert to 0-25 scale
    normalized_jobs_score = adjusted_jobs_score  # Already capped at -10 to 10
    jobs_contribution = (normalized_jobs_score + 10) * 1.25  # Convert to 0-25 scale
    
    # Do the same simplified approach for consumer spending
    # Validate and constrain consumer spending sentiment
    consumer_spending_sentiment = max(min(consumer_spending_sentiment, 2.0), 0.1)
    
    # Apply a simpler sentiment adjustment
    adjusted_cs_score = consumer_spending_score
    
    # Apply sentiment trend as a simple modifier (+/- 1 point maximum)
    if consumer_spending_sentiment < 0.9:  # Significantly bearish
        adjusted_cs_score -= 1
    elif consumer_spending_sentiment < 1.0:  # Slightly bearish
        adjusted_cs_score -= 0.5
    elif consumer_spending_sentiment > 1.1:  # Significantly bullish
        adjusted_cs_score += 1
    elif consumer_spending_sentiment > 1.0:  # Slightly bullish
        adjusted_cs_score += 0.5
    # If sentiment is 1.0 (neutral), no adjustment
    
    # Cap the adjusted score
    adjusted_cs_score = max(min(adjusted_cs_score, 10), -10)
    
    # Convert to 0-25 scale
    normalized_cs_score = adjusted_cs_score  # Already capped at -10 to 10
    cs_contribution = (normalized_cs_score + 10) * 1.25  # Convert to 0-25 scale
    
    # Calculate total market score (0-100)
    market_score = bs_contribution + ir_contribution + jobs_contribution + cs_contribution
    
    # Create components dictionary for detailed breakdown
    components = {
        'balance_sheet_contribution': bs_contribution,
        'interest_rate_contribution': ir_contribution,
        'jobs_contribution': jobs_contribution,
        'consumer_spending_contribution': cs_contribution,
        'jobs_score_raw': jobs_score,
        'jobs_score_adjusted': adjusted_jobs_score,
        'jobs_sentiment': jobs_sentiment,
        'consumer_spending_score_raw': consumer_spending_score,
        'consumer_spending_score_adjusted': adjusted_cs_score,
        'consumer_spending_sentiment': consumer_spending_sentiment
    }
    
    return market_score, components

def determine_quadrant_from_market_score(market_score):
    """
    Determine the economic quadrant based on the calculated market score
    
    Parameters:
    - market_score: unified market score (0-100)
    
    Returns:
    - quadrant: A, B, C, or D
    - quadrant_description: text description of the quadrant
    """
    # Define quadrant ranges
    if market_score < 25:
        quadrant = 'A'
        description = 'Inflation fighting quadrant - focus on value stocks'
    elif market_score < 50:
        quadrant = 'B'
        description = 'Growth with inflation quadrant - balanced approach with growth bias'
    elif market_score < 75:
        quadrant = 'C'
        description = 'Transition to growth quadrant - balanced approach with value bias'
    else:
        quadrant = 'D'
        description = 'Growth quadrant - focus on growth stocks'
    
    return quadrant, description

def determine_economic_quadrant(jobs_sentiment=DEFAULT_JOBS_SENTIMENT, 
                              consumer_spending_sentiment=DEFAULT_CONSUMER_SPENDING_SENTIMENT):
    """
    Determine the current US economic quadrant based on
    balance sheet trend, interest rate level, jobs data, and consumer spending
    using a unified scoring system, with sentiment adjustments for jobs and consumer spending
    
    Parameters:
    - jobs_sentiment: multiplier (0.1-2.0) for jobs sentiment (default=1.0 neutral)
    - consumer_spending_sentiment: multiplier (0.1-2.0) for consumer spending sentiment (default=1.0 neutral)
    
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
    jobs_data = get_jobs_data(start_date, end_date)
    consumer_spending = get_consumer_spending(start_date, end_date)
    
    # Calculate scores
    balance_sheet_score, bs_details = calculate_balance_sheet_score(balance_sheet)
    interest_rate_score, ir_details = calculate_interest_rate_score(interest_rates)
    jobs_score, jobs_details = calculate_jobs_score(jobs_data)
    consumer_spending_score, cs_details = calculate_consumer_spending_score(consumer_spending)
    
    # Calculate unified market score with sentiment adjustments
    market_score, score_components = calculate_market_score(
        balance_sheet_score, 
        interest_rate_score,
        jobs_score,
        consumer_spending_score,
        jobs_sentiment,
        consumer_spending_sentiment
    )
    
    # Determine quadrant
    quadrant, quadrant_description = determine_quadrant_from_market_score(market_score)
    
    # Create sentiment descriptions
    jobs_sentiment_desc = get_sentiment_description(jobs_sentiment)
    cs_sentiment_desc = get_sentiment_description(consumer_spending_sentiment)
    
    # Create a note describing the quadrant
    rate_regime = ir_details['rate_regime']
    jobs_trend = jobs_details['trend']
    cs_trend = cs_details['trend']
    note = (f"{quadrant_description} - Interest rates: {rate_regime}, "
            f"Jobs: {jobs_trend} (Sentiment: {jobs_sentiment_desc}), "
            f"Consumer Spending: {cs_trend} (Sentiment: {cs_sentiment_desc})")
    
    # Balance sheet trend for database compatibility
    balance_sheet_trend = 'Increasing' if balance_sheet_score > 0 else 'Decreasing'
    
    # Interest rate level for database compatibility - now based on fixed thresholds
    interest_rate_level = ir_details['rate_regime']
    
    # Save to database
    balance_sheet_value = balance_sheet.iloc[-1] if not balance_sheet.empty else None
    interest_rate_value = interest_rates.iloc[-1] if not interest_rates.empty else None
    jobs_value = jobs_data.iloc[-1] if not jobs_data.empty else None
    consumer_spending_value = consumer_spending.iloc[-1] if not consumer_spending.empty else None
    
    # Convert time series data to JSON-serializable format
    balance_sheet_dict = {}
    interest_rate_dict = {}
    jobs_dict = {}
    consumer_spending_dict = {}
    
    for date, value in balance_sheet.tail(30).items():
        balance_sheet_dict[date.strftime('%Y-%m-%d')] = value
        
    for date, value in interest_rates.tail(30).items():
        interest_rate_dict[date.strftime('%Y-%m-%d')] = value
        
    for date, value in jobs_data.tail(30).items():
        jobs_dict[date.strftime('%Y-%m-%d')] = value
        
    for date, value in consumer_spending.tail(30).items():
        consumer_spending_dict[date.strftime('%Y-%m-%d')] = value
    
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
            'jobs_data': jobs_dict,
            'consumer_spending_data': consumer_spending_dict,
            'balance_sheet_score': bs_details,
            'interest_rate_score': ir_details,
            'jobs_score': jobs_details,
            'consumer_spending_score': cs_details,
            'score_components': score_components,
            'market_score': market_score,
            'jobs_sentiment': jobs_sentiment,
            'consumer_spending_sentiment': consumer_spending_sentiment
        }
    )
    
    details = {
        'balance_sheet_details': bs_details,
        'interest_rate_details': ir_details,
        'jobs_details': jobs_details,
        'consumer_spending_details': cs_details,
        'score_components': score_components,
        'market_score': market_score,
        'jobs_sentiment': jobs_sentiment,
        'jobs_sentiment_desc': jobs_sentiment_desc,
        'consumer_spending_sentiment': consumer_spending_sentiment,
        'consumer_spending_sentiment_desc': cs_sentiment_desc
    }
    
    return quadrant, balance_sheet_trend, interest_rate_level, details

def plot_economic_data(balance_sheet, interest_rates, jobs_data, consumer_spending, 
                      bs_details=None, ir_details=None, jobs_details=None, cs_details=None, 
                      market_score=None, jobs_sentiment=DEFAULT_JOBS_SENTIMENT, 
                      consumer_spending_sentiment=DEFAULT_CONSUMER_SPENDING_SENTIMENT):
    """
    Plot all economic data indicators with enhanced visualization
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 20))
    
    # Plot balance sheet
    axs[0].plot(balance_sheet.index, balance_sheet.values, 'b-', label='Balance Sheet')
    
    # Add moving average if details are available
    if bs_details and 'moving_avg' in bs_details:
        # We need to calculate the full moving average for plotting
        ma_series = balance_sheet.rolling(window=24).mean()
        axs[0].plot(ma_series.index, ma_series.values, 'g--', label='2-Year Moving Avg')
    
    axs[0].set_title('Federal Reserve Balance Sheet (WALCL)')
    axs[0].set_ylabel('Billions of Dollars')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot interest rates
    axs[1].plot(interest_rates.index, interest_rates.values, 'r-', label='Interest Rate')
    
    # Add moving average if details are available
    if ir_details and 'moving_avg' in ir_details:
        # We need to calculate the full moving average for plotting
        ma_series = interest_rates.rolling(window=24).mean()
        axs[1].plot(ma_series.index, ma_series.values, 'g--', label='2-Year Moving Avg')
    
    axs[1].set_title('Federal Funds Rate')
    axs[1].set_ylabel('Percent')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot jobs data - use different color if sentiment is not neutral
    job_line_color = 'g-'
    if jobs_sentiment < 0.95:  # Bearish
        job_line_color = 'r-'
    elif jobs_sentiment > 1.05:  # Bullish
        job_line_color = 'b-'
        
    axs[2].plot(jobs_data.index, jobs_data.values, job_line_color, label='Jobs (Non-Farm Payrolls)')
    
    # Add 12-month moving average
    ma_series = jobs_data.rolling(window=12).mean()
    axs[2].plot(ma_series.index, ma_series.values, 'g--', label='12-Month Moving Avg')
    
    # Add sentiment annotation to the jobs plot
    if jobs_sentiment != 1.0:
        jobs_sentiment_desc = get_sentiment_description(jobs_sentiment)
        axs[2].annotate(f'Sentiment: {jobs_sentiment_desc} ({jobs_sentiment:.2f})',
                       xy=(0.02, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", alpha=0.3))
    
    axs[2].set_title('Employment: Non-Farm Payrolls (PAYEMS)')
    axs[2].set_ylabel('Thousands of Persons')
    axs[2].grid(True)
    axs[2].legend()
    
    # Plot consumer spending - use different color if sentiment is not neutral
    cs_line_color = 'c-'
    if consumer_spending_sentiment < 0.95:  # Bearish
        cs_line_color = 'r-'
    elif consumer_spending_sentiment > 1.05:  # Bullish
        cs_line_color = 'b-'
        
    axs[3].plot(consumer_spending.index, consumer_spending.values, cs_line_color, label='Consumer Spending')
    
    # Add 12-month moving average
    ma_series = consumer_spending.rolling(window=12).mean()
    axs[3].plot(ma_series.index, ma_series.values, 'c--', label='12-Month Moving Avg')
    
    # Add sentiment annotation to the consumer spending plot
    if consumer_spending_sentiment != 1.0:
        cs_sentiment_desc = get_sentiment_description(consumer_spending_sentiment)
        axs[3].annotate(f'Sentiment: {cs_sentiment_desc} ({consumer_spending_sentiment:.2f})',
                       xy=(0.02, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", alpha=0.3))
    
    axs[3].set_title('Personal Consumption Expenditures (PCE)')
    axs[3].set_ylabel('Billions of Dollars')
    axs[3].grid(True)
    axs[3].legend()
    
    # Add additional information if available
    if bs_details and ir_details and jobs_details and cs_details and market_score is not None:
        quadrant, _ = determine_quadrant_from_market_score(market_score)
        jobs_sentiment_desc = get_sentiment_description(jobs_sentiment)
        cs_sentiment_desc = get_sentiment_description(consumer_spending_sentiment)
        
        fig.suptitle(f"US Economic Indicators Analysis\n"
                     f"Market Score: {market_score:.1f}/100 (Quadrant {quadrant})\n"
                     f"Balance Sheet: {bs_details['total_score']:.2f}, Interest Rate: {ir_details['total_score']:.2f}\n"
                     f"Jobs: {jobs_details['total_score']:.2f} ({jobs_details['trend']}, Sentiment: {jobs_sentiment_desc}), "
                     f"Consumer Spending: {cs_details['total_score']:.2f} ({cs_details['trend']}, Sentiment: {cs_sentiment_desc})",
                     fontsize=16)
    
    plt.tight_layout()
    plt.savefig('economic_data.png')
    
    return 'economic_data.png'

def analyze_us_economy(jobs_sentiment=DEFAULT_JOBS_SENTIMENT,
                    consumer_spending_sentiment=DEFAULT_CONSUMER_SPENDING_SENTIMENT):
    """
    Analyze the US economy and determine the current quadrant
    
    Parameters:
    - jobs_sentiment: multiplier (0.1-2.0) for jobs sentiment (default=1.0 neutral)
    - consumer_spending_sentiment: multiplier (0.1-2.0) for consumer spending sentiment (default=1.0 neutral)
    
    Returns:
    - results: dictionary with economic analysis results
    """
    # Get data for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Get economic data
    balance_sheet = get_fed_balance_sheet(start_date, end_date)
    interest_rates = get_interest_rates(start_date, end_date)
    jobs_data = get_jobs_data(start_date, end_date)
    consumer_spending = get_consumer_spending(start_date, end_date)
    
    # Determine quadrant with sentiment adjustments
    quadrant, balance_sheet_trend, interest_rate_level, details = determine_economic_quadrant(
        jobs_sentiment, consumer_spending_sentiment
    )
    
    # Also get sentiment data for further analysis if needed
    consumer_sentiment_data = get_consumer_sentiment(start_date, end_date)
    job_claims_data = get_job_sentiment(start_date, end_date)
    
    # If sentiment values were provided as parameters, we use those
    # Otherwise, we calculate them based on the latest data using moving averages
    consumer_sentiment_desc = get_sentiment_description(consumer_spending_sentiment)
    jobs_sentiment_desc = get_sentiment_description(jobs_sentiment)
    
    # Calculate baselines for the UI (will be used in reporting)
    baseline_cs = None
    baseline_jobs = None
    
    if consumer_sentiment_data is not None and not consumer_sentiment_data.empty:
        window_size = min(24, len(consumer_sentiment_data))
        baseline_cs = consumer_sentiment_data.rolling(window=window_size).mean().iloc[-1]
    
    if job_claims_data is not None and not job_claims_data.empty:
        window_size = min(24, len(job_claims_data))
        baseline_jobs = job_claims_data.rolling(window=window_size).mean().iloc[-1]
    
    # Plot data with enhanced visualization
    plot_path = plot_economic_data(
        balance_sheet, 
        interest_rates,
        jobs_data,
        consumer_spending,
        details['balance_sheet_details'], 
        details['interest_rate_details'],
        details['jobs_details'],
        details['consumer_spending_details'],
        details['market_score'],
        jobs_sentiment,
        consumer_spending_sentiment
    )
    
    # Get raw and adjusted scores for jobs and consumer spending
    jobs_score_raw = details['jobs_details']['total_score']
    jobs_score_adjusted = details['score_components']['jobs_score_adjusted']
    cs_score_raw = details['consumer_spending_details']['total_score']
    cs_score_adjusted = details['score_components']['consumer_spending_score_adjusted']
    
    # Prepare results
    results = {
        'quadrant': quadrant,
        'balance_sheet_trend': balance_sheet_trend,
        'interest_rate_level': interest_rate_level,
        'plot_path': plot_path,
        'balance_sheet_latest': balance_sheet.iloc[-1],
        'interest_rate_latest': interest_rates.iloc[-1],
        'jobs_latest': jobs_data.iloc[-1],
        'consumer_spending_latest': consumer_spending.iloc[-1],
        'balance_sheet_score': details['balance_sheet_details']['total_score'],
        'interest_rate_score': details['interest_rate_details']['total_score'],
        'jobs_score_raw': jobs_score_raw,
        'jobs_score_adjusted': jobs_score_adjusted,
        'consumer_spending_score_raw': cs_score_raw,
        'consumer_spending_score_adjusted': cs_score_adjusted,
        'market_score': details['market_score'],
        'jobs_trend': details['jobs_details']['trend'],
        'consumer_spending_trend': details['consumer_spending_details']['trend'],
        'jobs_sentiment': jobs_sentiment,
        'jobs_sentiment_desc': jobs_sentiment_desc,
        'consumer_spending_sentiment': consumer_spending_sentiment,
        'consumer_spending_sentiment_desc': consumer_sentiment_desc,
        'baseline_cs': baseline_cs,
        'baseline_jobs': baseline_jobs
    }
    
    return results

if __name__ == "__main__":
    print("Analyzing US economy to determine current quadrant...")
    
    # Get data for sentiment analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)
    
    # Get consumer sentiment data (University of Michigan)
    consumer_sentiment_data = get_consumer_sentiment(start_date, end_date)
    
    # Get job market sentiment data (Initial Claims - inverse indicator)
    job_claims_data = get_job_sentiment(start_date, end_date)
    
    # Calculate consumer sentiment multiplier using 2-year moving average as baseline
    # Use a sensitivity of 3.0 to amplify the effect
    consumer_spending_sentiment, latest_cs_sentiment, baseline_cs = calculate_sentiment_multiplier(
        consumer_sentiment_data, baseline=None, sensitivity=3.0
    )
    
    # Print sentiment data
    print("\n--- Sentiment Analysis ---")
    if latest_cs_sentiment is not None:
        print(f"Latest UMich Consumer Sentiment: {latest_cs_sentiment:.1f}")
        print(f"Baseline (2-year moving avg): {baseline_cs:.1f}")
        print(f"% Difference from baseline: {((latest_cs_sentiment - baseline_cs) / baseline_cs * 100):.1f}%")
        print(f"Consumer Sentiment Value: {consumer_spending_sentiment:.2f}")
        print(f"Consumer Sentiment Interpretation: {get_sentiment_description(consumer_spending_sentiment)}")
        
        # Show simplified effect on score
        sentiment_effect = 0
        if consumer_spending_sentiment < 0.9:
            sentiment_effect = -1
            print("Effect on consumer spending score: -1.0 (Significantly bearish)")
        elif consumer_spending_sentiment < 1.0:
            sentiment_effect = -0.5
            print("Effect on consumer spending score: -0.5 (Slightly bearish)")
        elif consumer_spending_sentiment > 1.1:
            sentiment_effect = 1
            print("Effect on consumer spending score: +1.0 (Significantly bullish)")
        elif consumer_spending_sentiment > 1.0:
            sentiment_effect = 0.5
            print("Effect on consumer spending score: +0.5 (Slightly bullish)")
        else:
            print("Effect on consumer spending score: 0 (Neutral)")
    
    # For job claims data (ICSA), calculate job sentiment using 2-year moving average as baseline
    if not job_claims_data.empty:
        # Get the latest claims value
        latest_claims = job_claims_data.iloc[-1]
        
        # Calculate 2-year moving average as baseline
        window_size = min(24, len(job_claims_data))
        baseline_claims = job_claims_data.rolling(window=window_size).mean().iloc[-1]
        
        # Calculate deviation percentage from baseline
        deviation_pct = (latest_claims - baseline_claims) / baseline_claims
        
        # Invert the relationship since higher claims are negative
        # For job claims, lower values are better (fewer unemployment claims)
        jobs_sentiment = 1.0 - (deviation_pct * 2.0)
        
        # Constrain between 0.1 and 2.0
        jobs_sentiment = max(min(jobs_sentiment, 2.0), 0.1)
        
        print(f"Latest Initial Unemployment Claims (4-week avg): {latest_claims:.0f}")
        print(f"Baseline Claims (2-year moving avg): {baseline_claims:.0f}")
        print(f"% Difference from baseline: {((latest_claims - baseline_claims) / baseline_claims * 100):.1f}%")
        print(f"Job Sentiment Value: {jobs_sentiment:.2f}")
        print(f"Job Sentiment Interpretation: {get_sentiment_description(jobs_sentiment)}")
        
        # Show simplified effect on score
        sentiment_effect = 0
        if jobs_sentiment < 0.9:
            sentiment_effect = -1
            print("Effect on jobs score: -1.0 (Significantly bearish)")
        elif jobs_sentiment < 1.0:
            sentiment_effect = -0.5
            print("Effect on jobs score: -0.5 (Slightly bearish)")
        elif jobs_sentiment > 1.1:
            sentiment_effect = 1
            print("Effect on jobs score: +1.0 (Significantly bullish)")
        elif jobs_sentiment > 1.0:
            sentiment_effect = 0.5
            print("Effect on jobs score: +0.5 (Slightly bullish)")
        else:
            print("Effect on jobs score: 0 (Neutral)")
    else:
        # Default fallback if data isn't available
        jobs_sentiment = 0.5  # Very bearish default due to tariff concerns
        print("Using default job sentiment value: 0.5 (Very Bearish)")
        print("Effect on jobs score: -1.0 (Significantly bearish)")
    
    # Use fixed sentiment values if FRED data is not available
    if latest_cs_sentiment is None:
        consumer_spending_sentiment = 0.6  # Bearish default
        print("Using default consumer sentiment value: 0.6 (Bearish)")
        print("Effect on consumer spending score: -1.0 (Significantly bearish)")
    
    print("\n--- Economic Analysis ---")
    # Run analysis with sentiment adjustments
    results = analyze_us_economy(jobs_sentiment, consumer_spending_sentiment)
    
    print(f"\nCurrent US Economic Quadrant: {results['quadrant']}")
    print(f"Market Score: {results['market_score']:.1f}/100")
    print(f"Balance Sheet Trend: {results['balance_sheet_trend']} (Score: {results['balance_sheet_score']:.2f})")
    
    # Get rate regime description directly from results
    rate_regime = results['interest_rate_level']
    print(f"Interest Rate Level: {rate_regime} (Score: {results['interest_rate_score']:.2f})")
    
    # Print jobs and consumer spending information with sentiment adjustments
    print(f"Jobs Trend: {results['jobs_trend']} (Raw Score: {results['jobs_score_raw']:.2f})")
    print(f"Jobs Sentiment: {results['jobs_sentiment_desc']} ({results['jobs_sentiment']:.2f})")
    print(f"Jobs Adjusted Score: {results['jobs_score_adjusted']:.2f}")
    
    print(f"Consumer Spending Trend: {results['consumer_spending_trend']} (Raw Score: {results['consumer_spending_score_raw']:.2f})")
    print(f"Consumer Spending Sentiment: {results['consumer_spending_sentiment_desc']} ({results['consumer_spending_sentiment']:.2f})")
    print(f"Consumer Spending Adjusted Score: {results['consumer_spending_score_adjusted']:.2f}")
    
    print(f"Latest Balance Sheet Value: ${results['balance_sheet_latest']:.2f} billion")
    print(f"Latest Interest Rate: {results['interest_rate_latest']:.2f}%")
    print(f"Latest Non-Farm Payrolls: {results['jobs_latest']:.0f} thousand jobs")
    print(f"Latest Consumer Spending: ${results['consumer_spending_latest']:.2f} billion")
    
    # Get quadrant description
    _, quadrant_description = determine_quadrant_from_market_score(results['market_score'])
    print(f"\nQuadrant Description: {quadrant_description}")
    
    print(f"\nPlot saved to: {results['plot_path']}")
