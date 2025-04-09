"""
Plots historical 5-Factor CMS score against SPY price, with background
shading indicating the CMS percentile-based quadrant.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from fredapi import Fred
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
import yfinance as yf
import sqlite3

# --- Configuration ---
FRED_API_KEY = "69d56106bb7eb53d5117414d9d6e0b9e" # Replace with your key if needed
fred = Fred(api_key=FRED_API_KEY)

# Plotting period
YEARS_HISTORY = 10
end_date = datetime.now()
start_date = end_date - timedelta(days=YEARS_HISTORY * 365 + 180) # Fetch extra for rolling windows

# Ticker for S&P 500
SPY_TICKER = 'SPY'

# Z-Score and Percentile rolling window
WINDOW_YEARS = 5

# Quadrant Percentile Thresholds
QUADRANT_THRESHOLDS = {
    'A': (0, 25),
    'B': (25, 50),
    'C': (50, 75),
    'D': (75, 100)
}
QUADRANT_COLORS = {
    'A': 'lightcoral',   # Weakest
    'B': 'lightsalmon',
    'C': 'lightgreen',
    'D': 'lightblue'     # Strongest
}

# Output file name
PLOTS_DIR = 'plots' # Use root plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)
PLOT_FILENAME = os.path.join(PLOTS_DIR, 'cms_spy_history_quadrants.png')

# --- Helper Functions (Adapted from composite_economic_quadrant.py) ---

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

def compute_zscore(series, window_years=5):
    """Calculate the Z-score using a rolling window."""
    if series is None or series.empty: return pd.Series(dtype=float, name=getattr(series, 'name', None))
    window_size = window_years * 12
    min_periods_required = int(window_size * 0.6)
    if len(series) < min_periods_required: return pd.Series(dtype=float, name=series.name)

    mean = series.rolling(window=window_size, min_periods=min_periods_required).mean()
    std = series.rolling(window=window_size, min_periods=min_periods_required).std()
    std = std.replace(0, np.nan)
    z_scores = ((series - mean) / std).dropna()
    return z_scores.clip(lower=-3, upper=3)

def calculate_cms(z_scores_dict, default_weights):
    """Calculate the Composite Macro Score using available Z-scores."""
    available_indicators = {k: v for k, v in z_scores_dict.items() if v is not None and not v.empty}
    if not available_indicators: return pd.Series(dtype=float), {}, {}
    df = pd.concat(available_indicators.values(), axis=1, join='outer', keys=available_indicators.keys())
    df = df.ffill()
    required_indicators = list(default_weights.keys())
    df = df.dropna(subset=required_indicators)
    if df.empty: return pd.Series(dtype=float), {}, {}
    used_weights = default_weights
    cms = pd.Series(0.0, index=df.index)
    for indicator, weight in used_weights.items():
        if indicator in df.columns: cms += df[indicator] * weight
    last_z = df.iloc[-1].to_dict() if not df.empty else {}
    return cms, last_z, used_weights

# Function to calculate rolling percentile
def rolling_percentile(series, window_size):
    """ Calculates the rolling percentile of the last value in the window. """
    return series.rolling(window=window_size, min_periods=int(window_size*0.6)).apply(lambda x: percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) > 1 else np.nan, raw=False)

# --- Main Calculation Logic ---

def generate_historical_plot(start_date, end_date):
    """Generates the historical CMS vs SPY plot with quadrant backgrounds."""
    print("Fetching FRED data...")
    balance_sheet = fetch_fred_data('WALCL', start_date, end_date)
    interest_rates = fetch_fred_data('FEDFUNDS', start_date, end_date)
    jobs_raw = fetch_fred_data('PAYEMS', start_date, end_date)
    spending_raw = fetch_fred_data('PCE', start_date, end_date)
    inflation_raw = fetch_fred_data('CPIAUCSL', start_date, end_date)

    print("Fetching SPY data...")
    try:
        # Use auto_adjust=True (default) and expect 'Close' column
        spy_data = yf.download(SPY_TICKER, start=start_date, end=end_date)
        if spy_data.empty:
            raise ValueError("SPY data fetch returned empty DataFrame.")
        # Use 'Close' column which should contain adjusted prices when auto_adjust=True
        spy_monthly = spy_data['Close'].resample('ME').last() # Monthly end closing price
    except KeyError:
        # Fallback in case 'Close' isn't present (unexpected with auto_adjust=True)
        print("Warning: 'Close' column not found in yfinance data. Trying 'Adj Close'...")
        try:
             spy_monthly = spy_data['Adj Close'].resample('ME').last()
        except KeyError:
              print("Error: Neither 'Close' nor 'Adj Close' found in SPY data.")
              return None
    except Exception as e:
        print(f"Error fetching SPY data: {e}")
        return None

    print("Preprocessing economic data...")
    balance_sheet_monthly = balance_sheet.resample('ME').last()
    interest_rates_monthly = interest_rates.resample('ME').last()
    jobs = jobs_raw.pct_change(periods=1) * 100
    spending = spending_raw.pct_change(periods=1) * 100
    inflation = inflation_raw.pct_change(periods=12) * 100

    jobs.name = 'jobs'
    spending.name = 'spending'
    inflation.name = 'inflation'
    balance_sheet_monthly.name = 'balance_sheet'
    interest_rates_monthly.name = 'interest_rate'

    print("Calculating Z-Scores and CMS...")
    z_scores_data = {
        'jobs': compute_zscore(jobs, window_years=WINDOW_YEARS),
        'spending': compute_zscore(spending, window_years=WINDOW_YEARS),
        'inflation': compute_zscore(inflation, window_years=WINDOW_YEARS),
        'balance_sheet': compute_zscore(balance_sheet_monthly, window_years=WINDOW_YEARS),
        'interest_rate': compute_zscore(interest_rates_monthly, window_years=WINDOW_YEARS)
    }
    five_factor_weights = {'jobs': 0.20, 'spending': 0.20, 'inflation': 0.20, 'balance_sheet': 0.20, 'interest_rate': 0.20}
    cms, _, _ = calculate_cms(z_scores_data, five_factor_weights)

    if cms.empty:
        print("CMS calculation resulted in empty series. Cannot proceed.")
        return None

    print("Calculating historical CMS percentiles...")
    cms_percentiles = rolling_percentile(cms, window_size=WINDOW_YEARS * 12) # 5 year rolling percentile

    # Align data for plotting
    plot_data = pd.concat([cms, cms_percentiles, spy_monthly], axis=1, join='inner')
    plot_data.columns = ['CMS', 'CMS_Percentile', 'SPY']
    plot_data = plot_data.dropna() # Drop periods where percentile couldn't be calculated

    if plot_data.empty:
        print("No overlapping data found after alignment. Cannot plot.")
        return None

    print("Generating plot...")
    fig, ax1 = plt.subplots(figsize=(18, 9))

    # Plot CMS
    color_cms = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('CMS Score', color=color_cms)
    ax1.plot(plot_data.index, plot_data['CMS'], color=color_cms, label='5-Factor CMS', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_cms)
    ax1.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    # Plot SPY on secondary axis
    ax2 = ax1.twinx()
    color_spy = 'tab:red'
    ax2.set_ylabel('SPY Price (Adj Close)', color=color_spy)
    ax2.plot(plot_data.index, plot_data['SPY'], color=color_spy, label='SPY Price', linewidth=1.5, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_spy)
    ax2.grid(False) # Avoid double grid

    # Add Quadrant Background Shading based on Percentile
    last_quadrant = None
    block_start_date = plot_data.index[0]

    for date, percentile in plot_data['CMS_Percentile'].items():
        current_quadrant = None
        for quad, (lower, upper) in QUADRANT_THRESHOLDS.items():
            if lower <= percentile < upper:
                current_quadrant = quad
                break
        # Handle 100th percentile falling into the last bucket
        if percentile == 100 and 'D' in QUADRANT_THRESHOLDS:
             current_quadrant = 'D'

        if current_quadrant is None: # Skip if percentile is NaN or outside 0-100
             continue

        if last_quadrant is None: # First iteration
            last_quadrant = current_quadrant
            block_start_date = date
            continue

        if current_quadrant != last_quadrant:
            # End of the previous block, draw the span
            ax1.axvspan(block_start_date, date, facecolor=QUADRANT_COLORS[last_quadrant], alpha=0.2)
            # Start of the new block
            last_quadrant = current_quadrant
            block_start_date = date

    # Draw the last block after the loop finishes
    if last_quadrant is not None:
        ax1.axvspan(block_start_date, plot_data.index[-1], facecolor=QUADRANT_COLORS[last_quadrant], alpha=0.2)

    # Create legend for background colors
    patches = [mpatches.Patch(color=color, label=f'Quadrant {quad} ({l}%-{u}%)')
               for quad, (l, u) in QUADRANT_THRESHOLDS.items()
               for color in [QUADRANT_COLORS[quad]]]

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + patches, labels1 + labels2 + [p.get_label() for p in patches], loc='upper left')

    plt.title(f'Historical 5-Factor CMS vs. SPY ({YEARS_HISTORY} Years) with CMS Percentile Quadrants')
    fig.tight_layout()

    # Save plot
    plt.savefig(PLOT_FILENAME)
    print(f"Plot saved to: {PLOT_FILENAME}")
    plt.close(fig) # Close the plot figure to free memory
    return PLOT_FILENAME

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating historical CMS vs SPY plot for the last {YEARS_HISTORY} years...")
    generate_historical_plot(start_date, end_date) 