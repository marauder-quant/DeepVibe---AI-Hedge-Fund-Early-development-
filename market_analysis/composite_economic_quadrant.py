"""
US Economic Quadrant Determination using FRED data based on 5-Factor CMS Percentile
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
from dotenv import load_dotenv

# Import database functions
try:
    from db_utils import save_economic_quadrant # Assuming this table/function is still relevant for storing results
except ImportError:
    print("Warning: db_utils not found. Database saving will be skipped.")
    def save_economic_quadrant(*args, **kwargs):
        print("Skipping database save.")
        pass

# Load environment variables from .env file
load_dotenv()

# Get FRED API key from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in environment variables. Please add it to your .env file.")

fred = Fred(api_key=FRED_API_KEY)

# --- Data Fetching Functions ---

def fetch_fred_data(series_id, start_date=None, end_date=None):
    """Fetch data for a given FRED series ID."""
    try:
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        series = series.ffill()
        series = series.dropna()
        return series
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.Series(dtype=float)

# --- Indicator Calculation Functions ---

def calculate_balance_sheet_trend(balance_sheet):
    """Determine balance sheet trend based on 3-month rate of change."""
    if balance_sheet.empty or len(balance_sheet) < 4:
        return 'Unavailable', None
    roc_3m = balance_sheet.pct_change(periods=3).iloc[-1]
    trend = 'Increasing' if roc_3m > 0 else 'Decreasing'
    return trend, roc_3m

def calculate_interest_rate_level(interest_rates, window_years=5):
    """Determine interest rate level based on comparison to historical median."""
    if interest_rates.empty or len(interest_rates) < 2:
        return 'Unavailable', None, None
    window_days = window_years * 365
    if len(interest_rates) < window_days / 30: # Approximate check
         print(f"Warning: Insufficient data for {window_years}-year median calculation. Using available data.")
         median_rate = interest_rates.median()
    else:
         # Ensure index is datetime before rolling
         if not pd.api.types.is_datetime64_any_dtype(interest_rates.index):
             interest_rates.index = pd.to_datetime(interest_rates.index)
         median_rate = interest_rates.rolling(window=f'{window_days}D', min_periods=int(window_days / 60)).median().iloc[-1] # Use time window

    current_rate = interest_rates.iloc[-1]

    if pd.isna(median_rate):
         return 'Unavailable', current_rate, None

    level = 'High' if current_rate > median_rate else 'Low'
    return level, current_rate, median_rate

def compute_zscore(series, window_years=5):
    """Calculate the Z-score using a rolling window."""
    # Check if series is None or empty before proceeding
    if series is None or series.empty:
        print(f"Warning: Cannot compute Z-score for an empty or None series.")
        # Return an empty series with the same name if possible, or just an empty series
        return pd.Series(dtype=float, name=getattr(series, 'name', None))

    window_size = window_years * 12 # Assuming monthly data for window size calculation
    min_periods_required = int(window_size * 0.6) # 60% of window

    if len(series) < min_periods_required:
        print(f"Warning: Insufficient data ({len(series)} points) for {window_years}-year Z-score for {series.name}. Min periods: {min_periods_required}. Returning empty Z-score series.")
        return pd.Series(dtype=float, name=series.name)
    elif len(series) < window_size:
         print(f"Warning: Less than {window_years} years of data for Z-score for {series.name}. Using rolling window with available data.")
         mean = series.rolling(window=window_size, min_periods=min_periods_required).mean()
         std = series.rolling(window=window_size, min_periods=min_periods_required).std()
    else:
        mean = series.rolling(window=window_size, min_periods=min_periods_required).mean()
        std = series.rolling(window=window_size, min_periods=min_periods_required).std()

    # Avoid division by zero or near-zero std dev
    std = std.replace(0, np.nan)

    z_scores = (series - mean) / std
    z_scores = z_scores.dropna() # Drop NaNs resulting from std=0 or insufficient periods early on

    # Cap Z-scores at +/- 3 to handle outliers
    z_scores = z_scores.clip(lower=-3, upper=3)
    return z_scores

def calculate_cms(z_scores_dict, default_weights):
    """Calculate the Composite Macro Score using available Z-scores from a dictionary."""
    available_indicators = {k: v for k, v in z_scores_dict.items() if v is not None and not v.empty}

    if not available_indicators:
        print("Error: No valid Z-score series available for CMS calculation.")
        return pd.Series(dtype=float), {}, {}

    # Align available series by date index, using outer join and forward fill
    df = pd.concat(available_indicators.values(), axis=1, join='outer', keys=available_indicators.keys())
    df = df.ffill() # Forward fill first
    # Drop rows where any of the required indicators specified in weights are still NaN
    required_indicators = list(default_weights.keys())
    df = df.dropna(subset=required_indicators)

    if df.empty:
        print("Error: DataFrame empty after aligning Z-scores and dropping NaNs for required indicators.")
        return pd.Series(dtype=float), {}, {}

    # Use the default weights as provided (assuming they sum to 1)
    used_weights = default_weights

    # Calculate CMS
    cms = pd.Series(0.0, index=df.index)
    calculated_indicators = []
    for indicator, weight in used_weights.items():
        if indicator in df.columns:
            # Invert interest rate and inflation contribution:
            # Lower rates/inflation -> higher CMS
            # Higher rates/inflation -> lower CMS
            if 'interest_rate' in indicator or 'inflation' in indicator:
                 cms -= df[indicator].astype(float) * weight # Subtract if interest rate or inflation
            else:
                 cms += df[indicator].astype(float) * weight # Add otherwise (Jobs, Spending, Balance Sheet)
            calculated_indicators.append(indicator)
        else:
             print(f"Warning: Indicator '{indicator}' specified in weights not found in aligned data.")

    if cms.empty:
        print("Warning: CMS calculation resulted in an empty series.")
        last_z_scores = {}
    else:
        last_z_scores = df.iloc[-1].to_dict()

    # Updated print statement
    print(f"CMS calculated using indicators: {calculated_indicators} with weights: {used_weights} (Inflation & Interest rate inverted)")

    return cms, last_z_scores, used_weights

# --- Quadrant Determination based on CMS Percentile ---

def determine_economic_quadrant(start_date, end_date):
    """
    Determine the current US economic quadrant based on the historical percentile
    of the 5-Factor Composite Macro Score.
    """
    print("Fetching economic data...")
    # Fetch data (5 factors)
    balance_sheet = fetch_fred_data('WALCL', start_date, end_date)
    interest_rates = fetch_fred_data('FEDFUNDS', start_date, end_date)
    jobs_raw = fetch_fred_data('PAYEMS', start_date, end_date)
    spending_raw = fetch_fred_data('PCE', start_date, end_date)
    inflation_raw = fetch_fred_data('CPIAUCSL', start_date, end_date)

    # --- Preprocess Data ---
    print("Preprocessing data...")
    balance_sheet_monthly = balance_sheet.resample('ME').last()
    interest_rates_monthly = interest_rates.resample('ME').last()

    jobs = jobs_raw.pct_change(periods=1) * 100
    spending = spending_raw.pct_change(periods=1) * 100
    inflation = inflation_raw.pct_change(periods=12) * 100

    # Name series for Z-score warnings and dictionary keys
    jobs.name = 'jobs'
    spending.name = 'spending'
    inflation.name = 'inflation'
    balance_sheet_monthly.name = 'balance_sheet'
    interest_rates_monthly.name = 'interest_rate'

    # --- Calculate Trend/Level for DB logging (Not for Quadrant Determination) ---
    print("Calculating legacy Trend/Level for DB logging...")
    bs_trend_for_db, _ = calculate_balance_sheet_trend(balance_sheet_monthly)
    ir_level_for_db, _, _ = calculate_interest_rate_level(interest_rates_monthly, window_years=5)

    # --- Calculate Z-Scores and CMS ---
    print("Calculating Z-Scores and CMS...")
    z_scores_data = {
        'jobs': compute_zscore(jobs, window_years=5),
        'spending': compute_zscore(spending, window_years=5),
        'inflation': compute_zscore(inflation, window_years=5),
        'balance_sheet': compute_zscore(balance_sheet_monthly, window_years=5),
        'interest_rate': compute_zscore(interest_rates_monthly, window_years=5)
    }

    # Define weights (equal 0.20)
    five_factor_weights = {
        'jobs': 0.20, 'spending': 0.20, 'inflation': 0.20,
        'balance_sheet': 0.20, 'interest_rate': 0.20
    }
    cms, last_z_scores, actual_weights = calculate_cms(z_scores_data, five_factor_weights)

    # --- Determine Quadrant from CMS Percentile ---
    print("Determining quadrant based on CMS percentile...")
    cms_percentile = None
    current_cms = None
    current_quadrant = 'Unknown'
    description = "Quadrant could not be determined."

    if not cms.empty:
        current_cms = cms.iloc[-1]
        # Use a longer history for percentile calculation if possible (e.g., full history calculated)
        # Or stick to 5 years as before
        historical_cms = cms.last('5Y')
        if not historical_cms.empty and not pd.isna(current_cms):
             # Ensure enough historical data points for meaningful percentile
             if len(historical_cms.dropna()) > 10: # Need some minimum points
                  cms_percentile = percentileofscore(historical_cms.dropna(), current_cms)

                  if cms_percentile < 25:
                      current_quadrant = 'A'
                      description = f"Quadrant A (CMS Pctl {cms_percentile:.1f}%): Weakest macro momentum."
                  elif cms_percentile < 50:
                      current_quadrant = 'B'
                      description = f"Quadrant B (CMS Pctl {cms_percentile:.1f}%): Below average macro momentum."
                  elif cms_percentile < 75:
                      current_quadrant = 'C'
                      description = f"Quadrant C (CMS Pctl {cms_percentile:.1f}%): Above average macro momentum."
                  else:
                      current_quadrant = 'D'
                      description = f"Quadrant D (CMS Pctl {cms_percentile:.1f}%): Strongest macro momentum."
             else:
                  description = "Insufficient historical CMS data for reliable percentile calculation."
        else:
            description = "Could not calculate current CMS or percentile."
    else:
        description = "CMS calculation failed."

    # --- Prepare Details & Save ---
    print("Finalizing results...")
    details = {
        'quadrant_determination_method': 'CMS Percentile (5-Factor)',
        'current_cms': current_cms,
        'cms_percentile': cms_percentile,
        'cms_weights_used': actual_weights,
        'last_z_scores': last_z_scores,
        'quadrant_description': description,
        # Keep latest raw values for context if needed
        'latest_balance_sheet': balance_sheet_monthly.iloc[-1] if not balance_sheet_monthly.empty else None,
        'latest_interest_rate': interest_rates_monthly.iloc[-1] if not interest_rates_monthly.empty else None,
        'latest_jobs_mom_pct': jobs.iloc[-1] if not jobs.empty else None,
        'latest_spending_mom_pct': spending.iloc[-1] if not spending.empty else None,
        'latest_inflation_yoy_pct': inflation.iloc[-1] if not inflation.empty else None,
        # Add legacy values for DB context if needed in json_data too
        'legacy_balance_sheet_trend': bs_trend_for_db,
        'legacy_interest_rate_level': ir_level_for_db
    }

    # Prepare data for saving
    serializable_details = {}
    for k, v in details.items():
        if isinstance(v, pd.Timestamp):
            serializable_details[k] = v.strftime('%Y-%m-%d')
        elif isinstance(v, (np.datetime64, np.timedelta64)):
             serializable_details[k] = str(v)
        elif isinstance(v, (np.int64, np.float64)):
             serializable_details[k] = v.item() if not pd.isna(v) else None
        elif isinstance(v, dict):
             serializable_details[k] = {ik: (iv.item() if isinstance(iv, (np.int64, np.float64)) and not pd.isna(iv)
                                           else str(iv) if isinstance(iv, (np.datetime64, np.timedelta64))
                                           else iv.strftime('%Y-%m-%d') if isinstance(iv, pd.Timestamp)
                                           else None if pd.isna(iv)
                                           else iv)
                                       for ik, iv in v.items()}
        elif pd.isna(v):
             serializable_details[k] = None
        else:
             serializable_details[k] = v

    # Save to database (adjust function/table if needed for new structure)
    # Note: The original `economic_quadrants` table had BS Trend/IR Level columns.
    # You might want a new table or just store everything in json_data.
    # Assuming save_economic_quadrant can handle this or we modify it.
    save_economic_quadrant(
        quadrant=current_quadrant,
        balance_sheet_trend=bs_trend_for_db,
        interest_rate_level=ir_level_for_db,
        balance_sheet_value=details['latest_balance_sheet'],
        interest_rate_value=details['latest_interest_rate'],
        notes=description,
        json_data=serializable_details
    )

    # Return raw data for plotting
    all_data = {
        # Raw data still useful for plotting context
        'balance_sheet_monthly': balance_sheet_monthly,
        'interest_rates_monthly': interest_rates_monthly,
        'jobs': jobs,
        'spending': spending,
        'inflation': inflation,
        'cms': cms,
        'z_scores': z_scores_data
    }

    return current_quadrant, details, all_data

# --- Plotting Function ---
def plot_composite_economic_data(quadrant, details, all_data):
    """
    Plot the raw time series for the 5 core economic indicators.
    Saves the plot as 'plots/economic_data.png'.
    """
    print("Generating plot of raw economic indicators...")

    # Access raw data (ensure keys match those in all_data dictionary)
    jobs_pct_change = all_data.get('jobs')
    spending_pct_change = all_data.get('spending')
    inflation_yoy_pct = all_data.get('inflation')
    balance_sheet_level = all_data.get('balance_sheet_monthly')
    interest_rate_level = all_data.get('interest_rates_monthly')
    z_scores_data = all_data.get('z_scores')
    cms = all_data.get('cms')

    # Create 7 subplots (7 rows, 1 column)
    fig, axs = plt.subplots(7, 1, figsize=(12, 24), sharex=True) # Increased plots, adjusted figsize

    # --- Plotting Each Indicator ---

    # 1. Jobs MoM % Change
    if jobs_pct_change is not None and not jobs_pct_change.empty:
        axs[0].plot(jobs_pct_change.index, jobs_pct_change.values, label='Jobs (MoM % Change)', color='blue')
        axs[0].set_title('Jobs (Nonfarm Payrolls MoM % Change)')
        axs[0].set_ylabel('% Change')
        axs[0].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[0].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[0].set_title('Jobs Data Unavailable')

    # 2. Spending MoM % Change
    if spending_pct_change is not None and not spending_pct_change.empty:
        axs[1].plot(spending_pct_change.index, spending_pct_change.values, label='Spending (MoM % Change)', color='green')
        axs[1].set_title('Personal Consumption Expenditures (PCE MoM % Change)')
        axs[1].set_ylabel('% Change')
        axs[1].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[1].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[1].set_title('Spending Data Unavailable')

    # 3. Inflation YoY % Change
    if inflation_yoy_pct is not None and not inflation_yoy_pct.empty:
        axs[2].plot(inflation_yoy_pct.index, inflation_yoy_pct.values, label='Inflation (YoY % Change)', color='red')
        axs[2].set_title('Consumer Price Index (CPI YoY % Change)')
        axs[2].set_ylabel('% Change')
        axs[2].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[2].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[2].set_title('Inflation Data Unavailable')

    # 4. Fed Balance Sheet Level
    if balance_sheet_level is not None and not balance_sheet_level.empty:
        # Convert millions/billions if necessary for better scale, assuming it's in Millions
        axs[3].plot(balance_sheet_level.index, balance_sheet_level.values / 1000, label='Fed Balance Sheet (Trillions USD)', color='purple') # Assuming WALCL is Millions USD
        axs[3].set_title('Federal Reserve Balance Sheet Level')
        axs[3].set_ylabel('Trillions USD')
        axs[3].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[3].set_title('Balance Sheet Data Unavailable')

    # 5. Interest Rate Level
    if interest_rate_level is not None and not interest_rate_level.empty:
        axs[4].plot(interest_rate_level.index, interest_rate_level.values, label='Fed Funds Rate (%)', color='orange')
        axs[4].set_title('Effective Federal Funds Rate')
        axs[4].set_ylabel('Percent (%)')
        axs[4].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[4].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[4].set_title('Interest Rate Data Unavailable')

    # 6. Z-Scores Plot
    if z_scores_data:
        for name, z_series in z_scores_data.items():
             if z_series is not None and not z_series.empty:
                 axs[5].plot(z_series.index, z_series.values, label=name.replace('_', ' ').title())
        axs[5].set_title('Component Z-Scores (5yr Rolling)')
        axs[5].set_ylabel('Z-Score')
        axs[5].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[5].grid(True, linestyle=':', alpha=0.7)
        axs[5].legend(loc='best')
    else:
         axs[5].set_title('Z-Score Data Unavailable')

    # 7. Composite Macro Score (CMS) Plot
    if cms is not None and not cms.empty:
        axs[6].plot(cms.index, cms.values, label='CMS', color='black')
        axs[6].set_title('Composite Macro Score (CMS)')
        axs[6].set_ylabel('CMS Value')
        axs[6].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[6].grid(True, linestyle=':', alpha=0.7)
    else:
        axs[6].set_title('CMS Data Unavailable')

    # Adjust overall layout
    fig.suptitle(f"Core Economic Indicators & CMS - Quadrant: {quadrant}", fontsize=16, y=1.0) # Adjusted y and title
    plt.tight_layout(rect=[0, 0, 1, 0.99]) # Adjust layout rectangle slightly

    # Plot saving - overwrite economic_data.png in plots/
    PLOTS_DIR = 'plots' # Use root plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(PLOTS_DIR, 'economic_data.png') # Static filename
    plt.savefig(plot_filename)
    print(f"Plot saved to: {plot_filename}")
    plt.close(fig)
    return plot_filename

# --- Main Execution ---
if __name__ == "__main__":
    print("Analyzing US economy using 5-Factor CMS Percentile Quadrant Methodology...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365) # 10 years for history

    try:
        # Run analysis
        current_quadrant, details, all_data = determine_economic_quadrant(start_date, end_date)

        # Print results
        print("\n--- CMS Percentile Economic Quadrant Analysis Results ---")
        print(f"Analysis Date: {end_date.strftime('%Y-%m-%d')}")
        print(f"Determined Quadrant: {current_quadrant}")
        print(f"Description: {details['quadrant_description']}")
        print("-" * 20)
        cms_str = f"{details.get('current_cms'):.2f}" if isinstance(details.get('current_cms'), (int, float)) else 'N/A'
        cms_perc_str = f"{details.get('cms_percentile'):.1f}%" if isinstance(details.get('cms_percentile'), (int, float)) else 'N/A'
        print(f"Composite Macro Score (CMS): {cms_str}")
        print(f"CMS Percentile (vs last 5yr): {cms_perc_str}")
        print("Contributing Z-Scores (Latest Values):")
        if details.get('last_z_scores'):
            for k, v in details['last_z_scores'].items():
                 if k in details.get('cms_weights_used', {}):
                     z_score_str = f"{v:.2f}" if isinstance(v, (int, float)) else 'N/A'
                     print(f"  - {k}: {z_score_str}")
        else:
             print("  - N/A (CMS calculation failed or no Z-scores available)")
        print(f"CMS Indicator Weights Used: {details.get('cms_weights_used', 'N/A')}")
        print("-" * 20)

        # Generate plot
        plot_path = plot_composite_economic_data(current_quadrant, details, all_data)
        print(f"Analysis complete. Plot saved to {plot_path}")

    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        import traceback
        traceback.print_exc() 