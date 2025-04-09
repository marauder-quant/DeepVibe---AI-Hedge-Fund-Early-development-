"""
Asian Economic Quadrant Determination using IMF data (via sdmx1 library) for selected economies
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
# import requests # No longer needed, using sdmx1
import sdmx # Import the sdmx library
# from sdmx.errors import RequestError as SdmxRequestError # Import specific error type
import requests # Need this for the exception type

# Assuming db_utils is in the parent directory or accessible via PYTHONPATH
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from db_utils import save_economic_quadrant # Temporarily disable DB saving


# --- IMF Data Configuration (using sdmx1) ---
# Note: sdmx1 uses a slightly different key structure.
# We still need the Indicator codes, but the key in the query combines them.
# Ensure you have run: pip install sdmx1
IMF_DATASET_ID = 'IFS' # International Financial Statistics
ASIAN_ECONOMIES = {
    # Country codes used by IMF (e.g., JP, CN, KR, IN)
    'Japan': {
        'country_code': 'JP',
        'asset_indicator_code': 'FASMB_XDC',  # Central bank assets (Monetary Base?)
        'rate_indicator_code': 'FPOLM_PA',     # Policy rate
        'rate_threshold': 1.0
    },
    'China': {
        'country_code': 'CN',
        'asset_indicator_code': 'FASMB_XDC',
        'rate_indicator_code': 'FPOLM_PA',
        'rate_threshold': 3.0
    },
    'South Korea': {
        'country_code': 'KR',
        'asset_indicator_code': 'FASMB_XDC',
        'rate_indicator_code': 'FPOLM_PA',
        'rate_threshold': 2.0
    },
    'India': {
        'country_code': 'IN',
        'asset_indicator_code': 'FASMB_XDC',
        'rate_indicator_code': 'FPOLM_PA',
        'rate_threshold': 5.0
    }
}
# ------------------------------------------ #

def get_imf_data_sdmx(country_code, indicator_code, start_year, frequency='M'):
    """
    Fetch data from IMF using the sdmx1 library.

    Args:
        country_code (str): IMF country code (e.g., 'JP').
        indicator_code (str): IMF indicator code (e.g., 'FASMB_XDC').
        start_year (int): Start year for data retrieval.
        frequency (str): Data frequency ('M' for monthly, 'Q' for quarterly). Defaults to 'M'.

    Returns:
        pd.Series: Time series data with DateTimeIndex, or empty Series on failure.
    """
    if not country_code or not indicator_code:
        print(f"Warning: Missing country ({country_code}) or indicator ({indicator_code}) code.")
        return pd.Series(dtype=float)

    # Construct the key for the sdmx query
    # Format: [Freq].[Ref_Area].[Indicator].[Counterpart?].[Other Dims?]
    # Example key structure for IFS monthly data: M.JP.FASMB_XDC
    # The library might handle dimensions differently, let's try the direct key format first.
    series_key = f'{frequency}.{country_code}.{indicator_code}'
    print(f"Attempting to fetch IMF data for key: {series_key} starting from {start_year}")

    try:
        # Create an SDMX client for the IMF source
        imf = sdmx.Client("IMF")

        # Prepare parameters for the query
        params = {'startPeriod': str(start_year)}

        # Fetch data message
        # Use flow_ref=IMF_DATASET_ID to specify the dataset (e.g., 'IFS')
        # Use resource_id=series_key for the specific series
        # Updated based on potential sdmx1 structure: use key=series_key directly
        # --- Attempt 2: Pass dataset ID as first arg --- #
        # data_msg = imf.data(resource_id=IMF_DATASET_ID, key=series_key, params=params)
        data_msg = imf.data(IMF_DATASET_ID, key=series_key, params=params)

        # Convert the data message to a pandas Series
        # .to_pandas() often returns a DataFrame with multi-index.
        # We need to extract the specific series, assuming the key uniquely identifies it.
        data_pd = data_msg.to_pandas()

        if isinstance(data_pd, pd.DataFrame):
            # If it's a DataFrame, try to select the series.
            # The columns might represent different series if the key wasn't specific enough,
            # or index levels might represent dimensions.
            if data_pd.empty:
                 print(f"Warning: Received empty DataFrame for key {series_key}.")
                 return pd.Series(dtype=float)
            # Assuming the first column is the data if multiple series match the key pattern
            ts = data_pd.iloc[:, 0]
        elif isinstance(data_pd, pd.Series):
            # If it's already a Series, use it directly
            ts = data_pd
        else:
             print(f"Warning: Unexpected data type returned by to_pandas() for key {series_key}: {type(data_pd)}")
             return pd.Series(dtype=float)

        # Ensure the index is datetime
        ts.index = pd.to_datetime(ts.index)
        ts = ts.sort_index()
        # Convert values to numeric, coercing errors
        ts = pd.to_numeric(ts, errors='coerce')
        # Drop NaNs that might result from coercion or be present in original data
        ts = ts.dropna()

        if ts.empty:
            print(f"Warning: No valid numeric data found after processing for key {series_key}.")
            return pd.Series(dtype=float)

        print(f"Successfully fetched and parsed {len(ts)} data points for key {series_key}.")
        return ts

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors from the underlying requests library
        print(f"HTTP Error for key {series_key}: {e}")
        # Check if the error response has details
        if e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            # print(f"Response text: {e.response.text[:500]}...") # Careful printing large responses
        return pd.Series(dtype=float)
    except Exception as e:
        # Catch other potential errors during processing
        print(f"An unexpected error occurred processing IMF data via sdmx1 for key {series_key}: {e}")
        import traceback
        # traceback.print_exc() # Uncomment for detailed traceback during debugging
        return pd.Series(dtype=float)


# Keep existing trend/level/quadrant determination functions
# They work on pandas Series, regardless of the source

def determine_balance_sheet_trend(balance_sheet, window=12):
    """
    Determine if the balance sheet is increasing or decreasing using linear regression slope.
    Handles monthly or potentially quarterly data by adjusting the window interpretation.
    Returns 'Increasing', 'Decreasing', or 'Unavailable' if data is insufficient.
    """
    if balance_sheet.empty or len(balance_sheet) < 2:
        return 'Unavailable'

    # Use the last 'window' number of available data points
    # Ensure window size doesn't exceed available data points
    actual_window = min(window, len(balance_sheet))
    if actual_window < 2: # Need at least 2 points for regression
         return 'Unavailable'

    recent_data = balance_sheet.tail(actual_window)

    # Calculate the linear regression slope
    x = np.arange(len(recent_data))
    y = recent_data.values
    # Check for NaN/inf values that might remain (should be handled by dropna in get_imf_data_sdmx)
    if not np.all(np.isfinite(y)):
         print(f"Warning: Non-finite values found in balance sheet data for trend calculation (should have been dropped).")
         # Attempt to drop again just in case
         recent_data = recent_data.dropna()
         x = np.arange(len(recent_data))
         y = recent_data.values
         if len(y) < 2 or not np.all(np.isfinite(y)):
              return 'Unavailable'

    try:
         slope, _ = np.polyfit(x, y, 1)
         # Add a small tolerance for near-zero slopes if desired
         # tolerance = 1e-9
         # if abs(slope) < tolerance:
         #     return 'Flat' # Or handle as Increasing/Decreasing based on sign
         if slope > 0:
             return 'Increasing'
         else:
             return 'Decreasing'
    except Exception as e:
         print(f"Error calculating trend slope: {e}")
         return 'Unavailable'


def determine_interest_rate_level(interest_rates, threshold=3.0):
    """
    Determine if interest rates are high or low based on the latest available rate.
    Returns 'High', 'Low', or 'Unavailable' if data is insufficient.
    """
    if interest_rates.empty:
        return 'Unavailable'

    # Data should already be cleaned by get_imf_data_sdmx, get the last value
    current_rate_value = interest_rates.iloc[-1]

    # Check if the retrieved value is somehow still NaN (shouldn't happen)
    if pd.isna(current_rate_value):
         print("Warning: NaN value encountered in rate level determination (should have been dropped).")
         return 'Unavailable'

    if current_rate_value > threshold:
        return 'High'
    else:
        return 'Low'

def determine_economic_quadrant(balance_sheet_trend, interest_rate_level):
    """
    Determine the economic quadrant based on balance sheet trend and interest rate level.
    Uses the same logic as the US quadrant script for consistency.
    Returns quadrant ('A', 'B/C (prefer B)', 'B/C (prefer C)', 'D') or 'Unknown'.
    """
    if balance_sheet_trend == 'Unavailable' or interest_rate_level == 'Unavailable':
        return 'Unknown'

    if balance_sheet_trend == 'Increasing' and interest_rate_level == 'Low':
        quadrant = 'D' # Growth focus
    elif balance_sheet_trend == 'Increasing' and interest_rate_level == 'High':
        quadrant = 'B/C (prefer B)' # Transition, lean growth
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'Low':
        quadrant = 'B/C (prefer C)' # Transition, lean value
    elif balance_sheet_trend == 'Decreasing' and interest_rate_level == 'High':
        quadrant = 'A' # Inflation fighting, value focus
    else:
        quadrant = 'Unknown' # Should not happen if inputs are valid

    return quadrant

# Database saving function remains commented out
# def save_asian_economic_quadrant(...): ...

def plot_economic_data(balance_sheet, interest_rates, country_name, asset_label, rate_label, output_dir='plots'):
    """
    Plot balance sheet and interest rate data for a specific country.
    Saves the plot to a file named {country_name}_economic_data_imf.png.
    Returns the path to the saved plot or None if plotting fails or data is insufficient.
    """
    if balance_sheet.empty and interest_rates.empty:
        print(f"No data available to plot for {country_name}.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{country_name} Economic Indicators (IMF Data via sdmx1)', fontsize=16) # Updated title

    ax1 = axes[0]
    ax2 = axes[1]
    color1 = 'tab:blue'
    color2 = 'tab:red'

    # Plot balance sheet if available
    if not balance_sheet.empty:
        ax1.plot(balance_sheet.index, balance_sheet.values, color=color1, label=asset_label)
        ax1.set_ylabel(f'{asset_label} (Units vary)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(f'Central Bank Assets Proxy ({asset_label})')
        ax1.grid(True)
    else:
        ax1.text(0.5, 0.5, 'Asset Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title(f'Central Bank Assets Proxy ({asset_label} - Unavailable)')

    # Plot interest rates if available
    if not interest_rates.empty:
        ax2.plot(interest_rates.index, interest_rates.values, color=color2, label=rate_label)
        ax2.set_ylabel(f'{rate_label} (%)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_title(f'Policy Interest Rate ({rate_label})')
        ax2.grid(True)
        ax2.set_xlabel('Date')
    else:
        ax2.text(0.5, 0.5, 'Interest Rate Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title(f'Policy Interest Rate ({rate_label} - Unavailable)')
        ax2.set_xlabel('Date')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{country_name.replace(' ', '_')}_economic_data_imf_sdmx.png" # Added _sdmx
    plot_path = os.path.join(output_dir, plot_filename)

    try:
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Plot saved for {country_name} at {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Error saving plot for {country_name}: {e}")
        plt.close(fig)
        return None


def analyze_asian_economy(country_name, config):
    """
    Analyze a specific Asian economy using IMF data (via sdmx1) to determine its current quadrant.

    Parameters:
    - country_name: Name of the country (e.g., 'Japan')
    - config: Dictionary containing IMF 'country_code', 'asset_indicator_code', 'rate_indicator_code', 'rate_threshold'

    Returns:
    - Dictionary with analysis results, or None if analysis fails.
    """
    print(f"--- Analyzing {country_name} using IMF Data (sdmx1) ---")
    country_code = config.get('country_code')
    asset_indicator_code = config.get('asset_indicator_code')
    rate_indicator_code = config.get('rate_indicator_code')
    rate_threshold = config.get('rate_threshold', 3.0)

    if not country_code:
        print(f"Error: Missing country_code in config for {country_name}")
        return None

    # Get data for the last ~5 years (adjust as needed)
    end_date = datetime.now()
    start_year = end_date.year - 5
    # Use Monthly frequency by default for better trend analysis
    frequency = 'M'

    # Fetch data from IMF using sdmx1
    print(f"Fetching Asset Data for {country_name} ({frequency}.{country_code}.{asset_indicator_code})")
    balance_sheet = get_imf_data_sdmx(country_code, asset_indicator_code, start_year, frequency)

    print(f"Fetching Rate Data for {country_name} ({frequency}.{country_code}.{rate_indicator_code})")
    interest_rates = get_imf_data_sdmx(country_code, rate_indicator_code, start_year, frequency)

    if balance_sheet.empty and interest_rates.empty:
        print(f"Insufficient IMF data (via sdmx1) to analyze {country_name}.")
        return None

    # Determine trends and levels
    balance_sheet_trend = determine_balance_sheet_trend(balance_sheet, window=12)
    interest_rate_level = determine_interest_rate_level(interest_rates, rate_threshold)

    # Determine quadrant
    quadrant = determine_economic_quadrant(balance_sheet_trend, interest_rate_level)

    # Get latest values (use .iloc[-1] as data should be cleaned Series)
    balance_sheet_value = balance_sheet.iloc[-1] if not balance_sheet.empty else None
    interest_rate_value = interest_rates.iloc[-1] if not interest_rates.empty else None

    # Construct full indicator strings for notes/labels
    asset_indicator_full = f'{frequency}.{country_code}.{asset_indicator_code}'
    rate_indicator_full = f'{frequency}.{country_code}.{rate_indicator_code}'

    # Create note
    notes = f"Quadrant: {quadrant}, Assets Trend ({asset_indicator_full}): {balance_sheet_trend}, Rate Level ({rate_indicator_full}): {interest_rate_level} (Threshold: {rate_threshold}%)"
    if balance_sheet_trend == 'Unavailable':
        notes += f" (Asset data ({asset_indicator_full}) might be missing or insufficient for trend)"
    if interest_rate_level == 'Unavailable':
        notes += f" (Rate data ({rate_indicator_full}) might be missing or insufficient)"

    # Prepare JSON data (Temporarily disabled saving)
    balance_sheet_dict = {d.strftime('%Y-%m-%d'): v for d, v in balance_sheet.tail(30).items()} if not balance_sheet.empty else {}
    interest_rate_dict = {d.strftime('%Y-%m-%d'): v for d, v in interest_rates.tail(30).items()} if not interest_rates.empty else {}

    json_data_payload = json.dumps({
        'balance_sheet_data': balance_sheet_dict,
        'interest_rate_data': interest_rate_dict,
        'asset_indicator': asset_indicator_full, # Store full key used
        'rate_indicator': rate_indicator_full,   # Store full key used
        'rate_threshold': rate_threshold
    })

    # Save to database (Temporarily disabled)
    # save_asian_economic_quadrant(...)

    # Plot data
    plot_path = plot_economic_data(balance_sheet, interest_rates, country_name, asset_indicator_full, rate_indicator_full)

    # Prepare results dictionary
    results = {
        'country': country_name,
        'quadrant': quadrant,
        'balance_sheet_trend': balance_sheet_trend,
        'interest_rate_level': interest_rate_level,
        'balance_sheet_latest': balance_sheet_value,
        'interest_rate_latest': interest_rate_value,
        'asset_indicator': asset_indicator_full,
        'rate_indicator': rate_indicator_full,
        'plot_path': plot_path,
        'notes': notes
    }

    print(f"Analysis complete for {country_name}.")
    print(f"  Quadrant: {quadrant}")
    print(f"  Assets Trend ({asset_indicator_full}): {balance_sheet_trend}")
    print(f"  Rate Level ({rate_indicator_full}): {interest_rate_level} (Threshold: {rate_threshold}%)")
    print(f"  Latest Assets Value: {balance_sheet_value if balance_sheet_value is not None else 'N/A'}")
    print(f"  Latest Rate Value: {interest_rate_value if interest_rate_value is not None else 'N/A'}%")
    if plot_path:
        print(f"  Plot saved to: {plot_path}")
    print("---")

    return results


if __name__ == "__main__":
    print("Starting Asian Economic Quadrant Analysis using IMF Data (sdmx1)...")
    print("Ensure you have installed sdmx1: pip install sdmx1")
    all_results = {}

    for country, config in ASIAN_ECONOMIES.items():
        analysis_result = analyze_asian_economy(country, config)
        if analysis_result:
            all_results[country] = analysis_result

    print("\n--- IMF Analysis Summary (sdmx1) ---")
    if not all_results:
         print("No successful analysis results to display. Check API calls and indicator codes.")
    else:
         for country, result in all_results.items():
             print(f"{country}: Quadrant {result['quadrant']} (Assets Trend: {result['balance_sheet_trend']} [{result['asset_indicator']}], Rate Level: {result['interest_rate_level']} [{result['rate_indicator']}])")

    print("\nAnalysis finished.")


