"""
Generate a summary report from the China market analysis database
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# --- Adjust sys.path to allow direct script execution --- 
import sys
import os
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End path adjustment ---

# Import China-specific DB utils (now using direct import)
try:
    from db_utils_china import (
        get_latest_china_economic_quadrant,
        get_china_stocks_by_grade,
        get_china_economic_quadrant_history,
        get_china_db_stats
    )
except ImportError as e:
    print(f"Error importing db_utils_china: {e}")
    print("Ensure db_utils_china.py is in the same directory as this script.")
    # Provide dummy functions or exit if DB access is critical
    sys.exit(1) # Exit if DB functions are essential

# Keep sys import if used later, remove if not needed elsewhere
# import sys # Already imported above

def create_china_report_directory():
    """Create a directory for the China report"""
    # Report directory within asian_market_analysis
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # Create a timestamped directory for this report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'china_report_{timestamp}')
    os.makedirs(report_path)

    return report_path

def generate_china_economic_summary(report_path):
    """Generate a summary of the China economic situation"""
    print("Generating China economic summary...")

    # Get the latest China economic quadrant
    latest_quadrant = get_latest_china_economic_quadrant()
    if not latest_quadrant:
        print("No China economic quadrant data found in the database.")
        return None, None # Return None for quadrant and details

    # Get China economic quadrant history
    quadrant_history = get_china_economic_quadrant_history(limit=10)

    # Extract details from the JSON data (CMS, percentile, Z-scores etc.)
    details = {}
    latest_cms_value = None
    latest_cms_percentile = None
    last_z_scores = {}

    if latest_quadrant.get('json_data'):
        try:
            details = json.loads(latest_quadrant['json_data'])
            latest_cms_value = details.get('current_cms')
            latest_cms_percentile = details.get('cms_percentile')
            last_z_scores = details.get('last_z_scores', {})
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Could not parse json_data for latest quadrant: {e}")
            details = {} # Reset details if parsing fails

    # Create a markdown report
    with open(os.path.join(report_path, 'china_economic_summary.md'), 'w') as f:
        f.write("# China Economic Conditions Summary (CMS Percentile Method)\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Current Economic Quadrant (China)\n\n")
        f.write(f"**Quadrant**: {latest_quadrant['quadrant']}\n\n")
        f.write(f"**Analysis Date**: {latest_quadrant['analysis_date']}\n\n")
        f.write(f"**Description**: {latest_quadrant['analysis_notes']}\n\n")

        # Display CMS and Z-score details if available
        if details:
            cms_val_str = f"{latest_cms_value:.2f}" if latest_cms_value is not None else 'N/A'
            cms_perc_str = f"{latest_cms_percentile:.1f}%" if latest_cms_percentile is not None else 'N/A'
            z_bs = last_z_scores.get('balance_sheet_cn')
            z_ir = last_z_scores.get('interest_rate_cn')
            z_bs_str = f"{z_bs:.2f}" if z_bs is not None else 'N/A'
            z_ir_str = f"{z_ir:.2f}" if z_ir is not None else 'N/A'

            f.write(f"**Composite Macro Score (CMS)**: {cms_val_str}\n")
            f.write(f"**CMS Percentile (vs. 5yr)**: {cms_perc_str}\n")
            f.write(f"**Z-Score (Balance Sheet)**: {z_bs_str}\n")
            f.write(f"**Z-Score (Interest Rate)**: {z_ir_str}\n\n")
            f.write(f"**Calculation Method**: {details.get('quadrant_determination_method', 'N/A')}\n")
            f.write(f"**Weights Used**: {details.get('cms_weights_used', 'N/A')}\n")
            f.write(f"**FRED IDs Used**: BS={details.get('fred_balance_sheet_id', 'N/A')}, IR={details.get('fred_interest_rate_id', 'N/A')}\n\n")

        f.write("## Economic Quadrant History (China)\n\n")

        # Updated history table for China including specific values
        f.write("| Date | Quad | Notes | Latest BS Val | Latest IR Val | Z-Score BS | Z-Score IR |\n")
        f.write("|------|------|-------|---------------|---------------|------------|------------|\n")

        for _, row in quadrant_history.iterrows():
            date = row['analysis_date']
            quadrant = row['quadrant']
            notes = row['analysis_notes'] or '-'
            
            # Get values from the direct columns
            bs_val = row.get('latest_total_reserves_ex_gold_value')
            ir_val = row.get('latest_interbank_rate_3m_value')
            z_bs = row.get('z_score_total_reserves_ex_gold')
            z_ir = row.get('z_score_interbank_rate_3m')

            bs_val_str = f"{bs_val:.2f}" if bs_val is not None else 'N/A'
            ir_val_str = f"{ir_val:.2f}%" if ir_val is not None else 'N/A'
            z_bs_str = f"{z_bs:.2f}" if z_bs is not None else 'N/A'
            z_ir_str = f"{z_ir:.2f}" if z_ir is not None else 'N/A'

            f.write(f"| {date} | {quadrant} | {notes} | {bs_val_str} | {ir_val_str} | {z_bs_str} | {z_ir_str} |\n")

    # Create a visualization of quadrant history if there's enough data
    if not quadrant_history.empty and len(quadrant_history) > 1:
        try:
            plt.figure(figsize=(10, 6))

            # Convert quadrants to numeric values for plotting
            quadrant_map = {
                'A': 1,
                'B': 2,
                'C': 3,
                'D': 4
            }

            # Ensure analysis_date is datetime type for plotting
            quadrant_history['analysis_date_dt'] = pd.to_datetime(quadrant_history['analysis_date'])

            # Make sure we handle all quadrant values properly
            quadrant_history['quadrant_numeric'] = quadrant_history['quadrant'].map(
                lambda q: quadrant_map.get(q, 2.5) # Default to middle if unknown
            )

            # Plot quadrant over time
            plt.plot(quadrant_history['analysis_date_dt'], quadrant_history['quadrant_numeric'], 'o-', linewidth=2, color='red')
            plt.yticks([1, 2, 3, 4], ['A', 'B', 'C', 'D'])
            plt.xlabel('Date')
            plt.ylabel('Economic Quadrant (China)')
            plt.title('China Economic Quadrant History (CMS Percentile Method)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(report_path, 'china_quadrant_history.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved China quadrant history plot to {plot_filename}")
        except Exception as e:
            print(f"Error generating China quadrant history plot: {e}")

    # Optionally, plot CMS history if available in json_data
    cms_history_values = []
    cms_history_dates = []
    for _, row in quadrant_history.iterrows():
        if pd.notnull(row['json_data']):
            try:
                row_details = json.loads(row['json_data'])
                cms_val = row_details.get('current_cms')
                if cms_val is not None:
                    cms_history_values.append(cms_val)
                    cms_history_dates.append(pd.to_datetime(row['analysis_date']))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue # Skip if json_data is invalid or missing keys

    if cms_history_values and len(cms_history_values) > 1:
        try:
            # Sort by date just in case
            cms_data = pd.DataFrame({'date': cms_history_dates, 'cms': cms_history_values}).sort_values(by='date')

            plt.figure(figsize=(10, 6))
            plt.plot(cms_data['date'], cms_data['cms'], 'o-', linewidth=2, color='purple')
            plt.axhline(0, color='grey', linestyle='--', linewidth=1)
            plt.xlabel('Date')
            plt.ylabel('CMS Score (China 2-Factor)')
            plt.title('China Composite Macro Score (CMS) History')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_filename = os.path.join(report_path, 'china_cms_history.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved China CMS history plot to {plot_filename}")
        except Exception as e:
            print(f"Error generating China CMS history plot: {e}")

    return latest_quadrant, details

def generate_china_stock_summary(report_path, current_quadrant_details=None):
    """Generate a summary of China stock analysis results (if available)"""
    print("Generating China stock analysis summary...")

    # Get stock statistics
    stats = get_china_db_stats()

    # Get top stocks (grade A+ and A) - Use china specific function
    top_stocks = get_china_stocks_by_grade('A+')
    if not top_stocks.empty:
        top_stocks = pd.concat([top_stocks, get_china_stocks_by_grade('A')])
    else:
        # Handle case where get_china_stocks_by_grade('A+') returned empty
        a_grade_stocks = get_china_stocks_by_grade('A')
        if not a_grade_stocks.empty:
            top_stocks = a_grade_stocks
        # If both are empty, top_stocks remains empty

    # Create a markdown report
    with open(os.path.join(report_path, 'china_stock_summary.md'), 'w') as f:
        f.write("# China Stock Analysis Summary\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Database Statistics (China)\n\n")
        f.write(f"Total stock analyses: {stats['china_stock_analysis_count']}\n")
        f.write(f"Analysis batches: {', '.join(stats['china_batches']) if stats['china_batches'] else 'None'}\n\n")

        f.write("## Grade Distribution (China Stocks)\n\n")
        if stats['china_grade_distribution']:
            for grade, count in stats['china_grade_distribution'].items():
                if grade:  # Skip None values
                    f.write(f"- **{grade}**: {count} stocks\n")
        else:
            f.write("No stock grade data available.\n")

        f.write("\n## Top-Rated China Stocks (Example - if data exists)\n\n")

        if current_quadrant_details and 'quadrant' in current_quadrant_details:
            # Make sure current_quadrant_details is the actual quadrant dict, not None
            quadrant_value = current_quadrant_details.get('quadrant', 'Unknown') # Safely get quadrant
            f.write(f"Current China Economic Quadrant: **{quadrant_value}**\n\n")

        if not top_stocks.empty:
            f.write("| Symbol | Name | Sector | Grade | P/E Ratio | Debt/EBITDA | Rev Growth | Earn Growth |\n")
            f.write("|--------|------|--------|-------|-----------|-------------|------------|-------------|\n")

            for _, stock in top_stocks.iterrows():
                symbol = stock['symbol']
                name = stock['name'] or ''
                sector = stock['sector'] or ''
                grade = stock['grade'] or ''
                pe = f"{stock['pe_ratio']:.2f}" if pd.notnull(stock['pe_ratio']) else 'N/A'
                debt = f"{stock['debt_to_ebitda']:.2f}" if pd.notnull(stock['debt_to_ebitda']) else 'N/A'
                rev_growth = f"{stock['revenue_growth']*100:.1f}%" if pd.notnull(stock['revenue_growth']) else 'N/A'
                earn_growth = f"{stock['earnings_growth']*100:.1f}%" if pd.notnull(stock['earnings_growth']) else 'N/A'

                f.write(f"| {symbol} | {name} | {sector} | {grade} | {pe} | {debt} | {rev_growth} | {earn_growth} |\n")
        else:
            f.write("No top-rated (A+/A) China stock data found in the database.\n")

    # Create a visualization of grade distribution for China stocks
    if stats['china_grade_distribution']:
        try:
            plt.figure(figsize=(10, 6))

            grades = []
            counts = []
            for grade, count in stats['china_grade_distribution'].items():
                if grade:  # Skip None values
                    grades.append(grade)
                    counts.append(count)

            # Create bar chart
            plt.bar(grades, counts, color='coral')
            plt.xlabel('Grade')
            plt.ylabel('Number of China Stocks')
            plt.title('China Stock Grade Distribution')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(report_path, 'china_grade_distribution.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved China grade distribution plot to {plot_filename}")
        except Exception as e:
            print(f"Error generating China grade distribution plot: {e}")

    # Create a sector distribution for top China stocks
    if not top_stocks.empty:
        try:
            plt.figure(figsize=(12, 8))

            sector_counts = top_stocks['sector'].value_counts()

            # Create bar chart
            sector_counts.plot(kind='bar', color='lightcoral')
            plt.title('Sector Distribution of Top-Graded China Stocks')
            plt.xlabel('Sector')
            plt.ylabel('Number of Stocks')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()

            # Save the plot
            plot_filename = os.path.join(report_path, 'china_top_stocks_by_sector.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved China top stocks sector plot to {plot_filename}")
        except Exception as e:
            print(f"Error generating China top stocks sector plot: {e}")

def generate_full_china_report():
    """Generate a complete China market analysis report"""
    print("Generating China market analysis report...")

    # Create report directory
    report_path = create_china_report_directory()
    print(f"China report will be saved to: {report_path}")

    # Generate economic summary for China
    current_quadrant, current_details = generate_china_economic_summary(report_path)

    # Generate stock summary for China (pass quadrant details)
    # Pass the dictionary containing quadrant info itself
    generate_china_stock_summary(report_path, current_quadrant_details=current_quadrant)

    # Create index file
    with open(os.path.join(report_path, 'index.md'), 'w') as f:
        f.write("# China Market Analysis Report\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Contents\n\n")
        f.write("1. [China Economic Conditions Summary](china_economic_summary.md)\n")
        f.write("2. [China Stock Analysis Summary](china_stock_summary.md)\n\n")

        if current_quadrant:
            analysis_notes = current_quadrant.get('analysis_notes', 'N/A') # Safely get notes
            f.write(f"Current China Economic Quadrant: **{current_quadrant['quadrant']}** ({analysis_notes})\n\n")
            
            # --- Add specific values from DB columns to index.md --- 
            bs_val = current_quadrant.get('latest_total_reserves_ex_gold_value')
            ir_val = current_quadrant.get('latest_interbank_rate_3m_value')
            z_bs = current_quadrant.get('z_score_total_reserves_ex_gold')
            z_ir = current_quadrant.get('z_score_interbank_rate_3m')

            bs_val_str = f"{bs_val:.2f}" if bs_val is not None else 'N/A'
            ir_val_str = f"{ir_val:.2f}%" if ir_val is not None else 'N/A'
            z_bs_str = f"{z_bs:.2f}" if z_bs is not None else 'N/A'
            z_ir_str = f"{z_ir:.2f}" if z_ir is not None else 'N/A'
            
            f.write(f"**Latest Total Reserves ex Gold**: {bs_val_str}\n")
            f.write(f"**Latest Interbank Rate (3M)**: {ir_val_str}\n")
            f.write(f"**Z-Score (Total Reserves ex Gold)**: {z_bs_str}\n")
            f.write(f"**Z-Score (Interbank Rate 3M)**: {z_ir_str}\n\n")
            # --- End adding specific values --- 
            
            if current_details: # Display CMS/Percentile from JSON as well
                 cms_val_str = f"{current_details.get('current_cms'):.2f}" if current_details.get('current_cms') is not None else 'N/A'
                 cms_perc_str = f"{current_details.get('cms_percentile'):.1f}%" if current_details.get('cms_percentile') is not None else 'N/A'
                 f.write(f"**CMS (2-Factor)**: {cms_val_str} ({cms_perc_str} percentile vs 5yr)\n\n")

        f.write("### Visualizations\n\n")

        if os.path.exists(os.path.join(report_path, 'china_quadrant_history.png')):
            f.write("![China Economic Quadrant History](china_quadrant_history.png)\n\n")

        if os.path.exists(os.path.join(report_path, 'china_cms_history.png')):
            f.write("![China CMS History](china_cms_history.png)\n\n")

        if os.path.exists(os.path.join(report_path, 'china_grade_distribution.png')):
            f.write("![China Stock Grade Distribution](china_grade_distribution.png)\n\n")

        if os.path.exists(os.path.join(report_path, 'china_top_stocks_by_sector.png')):
            f.write("![China Top Stocks by Sector](china_top_stocks_by_sector.png)\n\n")

    print(f"China report generation complete. Open {os.path.join(report_path, 'index.md')} to view the report.")
    return report_path

if __name__ == "__main__":
    report_path = generate_full_china_report()
    print(f"China report saved to: {report_path}") 