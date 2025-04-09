"""
China Economic Quadrant Determination using FRED data based on 2-Factor CMS Percentile
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from datetime import datetime, timedelta
from scipy.stats import percentileofscore

# --- Adjust sys.path to allow direct script execution --- 
import sys
import os
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End path adjustment ---

# Import China-specific database function (now using direct import)
try:
    from db_utils_china import save_china_economic_quadrant
except ImportError as e:
    print(f"Warning: db_utils_china.py not found or import failed: {e}. Database saving will be skipped.")
    def save_china_economic_quadrant(*args, **kwargs):
        print("Skipping database save (dummy function).")
        pass

# Set up FRED API with the provided key
FRED_API_KEY = "69d56106bb7eb53d5117414d9d6e0b9e" # Replace with your key if needed
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

# --- Indicator Calculation Functions (Keep Z-score, remove others if not directly used) ---

def compute_zscore(series, window_years=5):
    """Calculate the Z-score using a rolling window."""
    if series is None or series.empty:
        print(f"Warning: Cannot compute Z-score for an empty or None series ({getattr(series, 'name', 'Unnamed')}).")
        return pd.Series(dtype=float, name=getattr(series, 'name', None))

    # Data frequency might not be monthly for all series, adjust logic if needed
    # Assume monthly for window calculation for now
    window_size = window_years * 12
    min_periods_required = int(window_size * 0.6) # 60% of window

    if len(series) < min_periods_required:
        print(f"Warning: Insufficient data ({len(series)} points) for {window_years}-year Z-score for {series.name}. Min periods: {min_periods_required}. Returning empty Z-score series.")
        return pd.Series(dtype=float, name=series.name)
    # The rolling calculation handles cases where len(series) < window_size but >= min_periods_required
    # else: # No need for elif len(series) < window_size:
    mean = series.rolling(window=window_size, min_periods=min_periods_required).mean()
    std = series.rolling(window=window_size, min_periods=min_periods_required).std()

    # Avoid division by zero or near-zero std dev
    std = std.replace(0, np.nan).ffill().bfill() # Fill NaNs from std=0 if possible

    z_scores = (series - mean) / std
    z_scores = z_scores.dropna() # Drop NaNs resulting from initial periods or std=0 issues

    # Cap Z-scores at +/- 3 to handle outliers
    z_scores = z_scores.clip(lower=-3, upper=3)
    return z_scores

def calculate_cms(z_scores_dict, weights):
    """Calculate the Composite Macro Score using available Z-scores from a dictionary."""
    available_indicators = {k: v for k, v in z_scores_dict.items() if v is not None and not v.empty}

    if not available_indicators:
        print("Error: No valid Z-score series available for CMS calculation.")
        return pd.Series(dtype=float), {}, {}

    # Align available series by date index, using outer join and forward fill
    df = pd.concat(available_indicators.values(), axis=1, join='outer', keys=available_indicators.keys())
    df = df.ffill() # Forward fill first
    # Drop rows where any of the required indicators specified in weights are still NaN
    required_indicators = list(weights.keys())
    # Drop only if *all* required indicators are NaN for that row after ffill
    df = df.dropna(subset=required_indicators, how='all')
    # Then drop rows where *any* required indicator is still NaN (handles edge cases after ffill)
    df = df.dropna(subset=required_indicators, how='any')


    if df.empty:
        print("Error: DataFrame empty after aligning Z-scores and dropping NaNs for required indicators.")
        return pd.Series(dtype=float), {}, {}

    # Use the provided weights
    used_weights = weights

    # Calculate CMS
    cms = pd.Series(0.0, index=df.index)
    calculated_indicators = []
    for indicator, weight in used_weights.items():
        if indicator in df.columns:
            # Invert interest rate contribution: lower rates -> higher CMS
            if 'interest_rate' in indicator:
                 cms -= df[indicator].astype(float) * weight # Subtract if interest rate
            else:
                 cms += df[indicator].astype(float) * weight # Add otherwise
            calculated_indicators.append(indicator)
        else:
             print(f"Warning: Indicator '{indicator}' specified in weights not found in aligned data.")

    if cms.empty:
        print("Warning: CMS calculation resulted in an empty series.")
        last_z_scores = {}
    else:
        last_z_scores = df.iloc[-1].to_dict()

    print(f"CMS calculated using indicators: {calculated_indicators} with weights: {used_weights} (Interest rate inverted)")

    return cms, last_z_scores, used_weights

# --- Quadrant Determination based on CMS Percentile ---

def determine_chinese_economic_quadrant(start_date, end_date):
    """
    Determine the current China economic quadrant based on the historical percentile
    of the 2-Factor Composite Macro Score (Balance Sheet, Interest Rates).
    """
    print("Fetching China economic data (Balance Sheet, Interest Rates)...")
    # Fetch data (2 factors for China)
    # Balance Sheet: Total Reserve Assets excluding Gold for China
    balance_sheet_raw = fetch_fred_data('TRESEGCNM052N', start_date, end_date)
    # Interest Rate: Interbank Rate, 3-Month Treasury Bill Rate for China
    interest_rates_raw = fetch_fred_data('IR3TIB01CNM156N', start_date, end_date)

    # --- Preprocess Data ---
    print("Preprocessing data...")
    # Resample to monthly, taking the last observation of the month
    # Use monthly frequency for consistency in Z-score window calculation
    balance_sheet = balance_sheet_raw.resample('ME').last()
    interest_rates = interest_rates_raw.resample('ME').last()

    # Name series for Z-score warnings and dictionary keys
    balance_sheet.name = 'balance_sheet_cn'
    interest_rates.name = 'interest_rate_cn'

    # --- Calculate Z-Scores and CMS ---
    print("Calculating Z-Scores and CMS...")
    z_scores_data = {
        balance_sheet.name: compute_zscore(balance_sheet, window_years=5),
        interest_rates.name: compute_zscore(interest_rates, window_years=5)
    }

    # Define weights (50% each)
    two_factor_weights = {
        balance_sheet.name: 0.50,
        interest_rates.name: 0.50
    }
    cms, last_z_scores, actual_weights = calculate_cms(z_scores_data, two_factor_weights)

    # --- Determine Quadrant from CMS Percentile ---
    print("Determining quadrant based on CMS percentile...")
    cms_percentile = None
    current_cms = None
    current_quadrant = 'Unknown'
    description = "Quadrant could not be determined."

    if not cms.empty:
        current_cms = cms.iloc[-1]
        # Use 5 years history for percentile calculation
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
                  description = f"Insufficient historical CMS data ({len(historical_cms.dropna())} points) for reliable percentile calculation."
        elif pd.isna(current_cms):
            description = "Current CMS value is NaN."
        else: # historical_cms is empty
            description = "Could not calculate historical CMS for percentile."

    else:
        description = "CMS calculation failed or resulted in empty series."

    # --- Prepare Details & Save ---
    print("Finalizing results...")
    details = {
        'country': 'China',
        'quadrant_determination_method': 'CMS Percentile (2-Factor: BS, IR)',
        'current_cms': current_cms,
        'cms_percentile': cms_percentile,
        'cms_weights_used': actual_weights,
        'last_z_scores': last_z_scores,
        'quadrant_description': description,
        # Keep latest raw values for context if needed
        'latest_balance_sheet': balance_sheet.iloc[-1] if not balance_sheet.empty else None,
        'latest_interest_rate': interest_rates.iloc[-1] if not interest_rates.empty else None,
        'fred_balance_sheet_id': 'TRESEGCNM052N',
        'fred_interest_rate_id': 'IR3TIB01CNM156N'
    }

    # Prepare data for saving (handle potential NaNs and numpy types)
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

    # --- Extract values for new DB columns ---
    latest_bs_val = details.get('latest_balance_sheet')
    latest_ir_val = details.get('latest_interest_rate')
    z_bs_val = details.get('last_z_scores', {}).get('balance_sheet_cn')
    z_ir_val = details.get('last_z_scores', {}).get('interest_rate_cn')

    # Save to China database
    save_china_economic_quadrant(
        quadrant=current_quadrant,
        # Original values passed (kept for potential backward compatibility or other uses)
        balance_sheet_value=latest_bs_val,
        interest_rate_value=latest_ir_val,
        notes=description,
        json_data=serializable_details, # Contains all details including country, CMS, etc.
        # Pass values using the new specific parameter names
        latest_total_reserves_ex_gold_value=latest_bs_val,
        latest_interbank_rate_3m_value=latest_ir_val,
        z_score_total_reserves_ex_gold=z_bs_val,
        z_score_interbank_rate_3m=z_ir_val
        # balance_sheet_trend and interest_rate_level are not calculated here, so we omit them (they default to None)
    )

    # Return raw data for plotting
    all_data = {
        'balance_sheet': balance_sheet,
        'interest_rates': interest_rates,
        'cms': cms,
        'z_scores': z_scores_data
    }

    return current_quadrant, details, all_data

# --- Plotting Function ---
def plot_china_economic_data(quadrant, details, all_data):
    """
    Plot Raw Data (Balance Sheet, Interest Rate), Z-Scores, and CMS for China.
    Saves the plot as 'plots/chinese_economic_data.png'.
    """
    print("Generating China economic plot...")

    # Access raw and derived data
    balance_sheet = all_data.get('balance_sheet')
    interest_rates = all_data.get('interest_rates')
    z_scores_to_plot = all_data.get('z_scores', {})
    cms_series = all_data.get('cms')

    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True) # Increased plots to 4, adjusted figsize

    # --- Plotting ---

    # 1. Raw Balance Sheet
    if balance_sheet is not None and not balance_sheet.empty:
        # Determine units (assuming millions USD from FRED ID TRESEGCNM052N description)
        axs[0].plot(balance_sheet.index, balance_sheet.values / 1e6, label='Total Reserves (Billions USD)', color='blue') # Convert Millions to Billions
        axs[0].set_title('China Total Reserves excluding Gold')
        axs[0].set_ylabel('Billions USD')
        axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[0].legend(loc='best')
    else:
        axs[0].set_title('Balance Sheet Data Unavailable')

    # 2. Raw Interest Rate
    if interest_rates is not None and not interest_rates.empty:
        axs[1].plot(interest_rates.index, interest_rates.values, label='Interbank Rate 3M (%)', color='red')
        axs[1].set_title('China Interbank Rate (3-Month Treasury Bill)')
        axs[1].set_ylabel('Percent (%)')
        axs[1].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[1].legend(loc='best')
    else:
        axs[1].set_title('Interest Rate Data Unavailable')

    # 3. Z-scores (shifted to axs[2])
    weights = details.get('cms_weights_used', {})
    z_colors = {'balance_sheet_cn': 'purple', 'interest_rate_cn': 'orange'}
    plotted_labels = []

    for key, z_series in z_scores_to_plot.items():
         # Only plot if it's one of the expected keys and has data
         if key in weights and z_series is not None and not z_series.empty:
             weight_str = f"{weights[key]:.2f}"
             # Make label more descriptive for China
             label_base = key.replace('_cn', '').replace('_', ' ').title()
             label = f'Z-{label_base} (China) (w={weight_str})'
             axs[2].plot(z_series.index, z_series.values, label=label, color=z_colors.get(key, 'grey'), alpha=0.8)
             plotted_labels.append(label)

    if plotted_labels: # Only add legend and title if something was plotted
        axs[2].axhline(0, color='grey', linestyle='--', linewidth=1)
        axs[2].set_title('Contributing Z-Scores (China: Balance Sheet, Interest Rate)')
        axs[2].set_ylabel('Z-Score')
        axs[2].grid(True, linestyle=':', alpha=0.7)
        axs[2].legend(loc='best') # Changed to best fit location
    else:
         axs[2].set_title('Z-Scores (No data available)')

    # 4. Composite Macro Score (shifted to axs[3])
    if cms_series is not None and not cms_series.empty:
        axs[3].plot(cms_series.index, cms_series.values, 'k-', label='Composite Macro Score (CMS - China)')
    axs[3].axhline(0, color='grey', linestyle='--', linewidth=1)
    cms_val_str = f"{details.get('current_cms'):.2f}" if isinstance(details.get('current_cms'), (int, float)) else 'N/A'
    cms_perc_str = f"{details.get('cms_percentile'):.1f}%" if isinstance(details.get('cms_percentile'), (int, float)) else 'N/A'
    axs[3].set_title(f"China Composite Macro Score (2-Factor) - Current: {cms_val_str} ({cms_perc_str} percentile)")
    axs[3].set_ylabel('CMS Score')
    axs[3].grid(True, linestyle=':', alpha=0.7)
    if cms_series is not None and not cms_series.empty: # Add legend only if CMS plotted
        axs[3].legend(loc='best')

    # Adjust overall layout
    fig.suptitle(f"China Economic Data & CMS Analysis - Quadrant: {quadrant}", fontsize=16, y=1.0) # Adjusted title and y
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout rectangle

    # Plot saving - use static filename
    PLOTS_DIR = 'plots'
    # Ensure plots directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_filename = os.path.join(PLOTS_DIR, 'chinese_economic_data.png') # Static filename
    plt.savefig(plot_filename)
    print(f"Plot saved to: {plot_filename}")
    plt.close(fig)
    return plot_filename

# --- Main Execution ---
if __name__ == "__main__":
    print("Analyzing China economy using 2-Factor CMS Percentile Quadrant Methodology...")

    end_date = datetime.now()
    # Fetch more history if available for better Z-score/percentile calculation
    start_date = end_date - timedelta(days=15*365) # 15 years history

    try:
        # Run analysis
        current_quadrant, details, all_data = determine_chinese_economic_quadrant(start_date, end_date)

        # Print results
        print("\n--- China CMS Percentile Economic Quadrant Analysis Results ---")
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
                 # Check if the key is one of the factors used
                 if k in details.get('cms_weights_used', {}):
                     z_score_str = f"{v:.2f}" if isinstance(v, (int, float, np.float64)) and not pd.isna(v) else 'N/A'
                     label_base = k.replace('_cn', '').replace('_', ' ').title()
                     print(f"  - {label_base}: {z_score_str}")
        else:
             print("  - N/A (CMS calculation failed or no Z-scores available)")
        print(f"CMS Indicator Weights Used: {details.get('cms_weights_used', 'N/A')}")
        print(f"FRED IDs Used: BS={details.get('fred_balance_sheet_id', 'N/A')}, IR={details.get('fred_interest_rate_id', 'N/A')}")
        print("-" * 20)

        # Generate plot
        if all_data and not all(d is None or d.empty for d in [all_data.get('cms'), all_data.get('balance_sheet'), all_data.get('interest_rates')]):
             plot_path = plot_china_economic_data(current_quadrant, details, all_data)
             print(f"Analysis complete. Plot saved to {plot_path}")
        else:
             print("Skipping plot generation due to missing data.")


    except Exception as e:
        print("\nAn error occurred during China analysis: {e}")
        import traceback
        traceback.print_exc()
