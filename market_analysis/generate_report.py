"""
Generate a summary report from the market analysis database
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from db_utils import (
    get_latest_economic_quadrant,
    get_stocks_by_grade,
    get_economic_quadrant_history,
    get_db_stats
)
import importlib
import sys

def create_report_directory():
    """Create a directory for the report"""
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Create a timestamped directory for this report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'report_{timestamp}')
    os.makedirs(report_path)
    
    return report_path

def generate_economic_summary(report_path):
    """Generate a summary of the economic situation"""
    print("Generating economic summary...")
    
    # Get the latest economic quadrant
    latest_quadrant = get_latest_economic_quadrant()
    if not latest_quadrant:
        print("No economic quadrant data found in the database.")
        return
    
    # Get economic quadrant history
    quadrant_history = get_economic_quadrant_history(limit=10)
    
    # Extract sentiment data if available in the JSON data
    sentiment_data = {}
    if latest_quadrant.get('json_data'):
        try:
            json_data = json.loads(latest_quadrant['json_data'])
            
            # Check for sentiment data
            score_components = json_data.get('score_components', {})
            if score_components:
                sentiment_data = {
                    'jobs_sentiment': score_components.get('jobs_sentiment'),
                    'consumer_spending_sentiment': score_components.get('consumer_spending_sentiment'),
                    'jobs_score_raw': score_components.get('jobs_score_raw'),
                    'jobs_score_adjusted': score_components.get('jobs_score_adjusted'),
                    'consumer_spending_score_raw': score_components.get('consumer_spending_score_raw'),
                    'consumer_spending_score_adjusted': score_components.get('consumer_spending_score_adjusted')
                }
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Create a markdown report
    with open(os.path.join(report_path, 'economic_summary.md'), 'w') as f:
        f.write("# Economic Conditions Summary\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Current Economic Quadrant\n\n")
        f.write(f"**Quadrant**: {latest_quadrant['quadrant']}\n\n")
        f.write(f"**Balance Sheet Trend**: {latest_quadrant['balance_sheet_trend']}\n\n")
        f.write(f"**Interest Rate Level**: {latest_quadrant['interest_rate_level']}\n\n")
        f.write(f"**Analysis Date**: {latest_quadrant['analysis_date']}\n\n")
        f.write(f"**Notes**: {latest_quadrant['analysis_notes']}\n\n")
        
        # Add sentiment data if available
        if sentiment_data:
            f.write("## Sentiment Analysis\n\n")
            
            f.write("### Job Market Sentiment\n\n")
            if sentiment_data.get('jobs_sentiment'):
                js_val = sentiment_data['jobs_sentiment']
                # Import get_sentiment_description function to get sentiment description
                try:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from economic_quadrant import get_sentiment_description
                    js_desc = get_sentiment_description(js_val)
                    f.write(f"**Job Sentiment**: {js_desc} ({js_val:.2f})\n\n")
                except ImportError:
                    f.write(f"**Job Sentiment Multiplier**: {js_val:.2f}\n\n")
                
                # Add raw vs adjusted scores
                if sentiment_data.get('jobs_score_raw') is not None and sentiment_data.get('jobs_score_adjusted') is not None:
                    f.write(f"**Raw Jobs Score**: {sentiment_data['jobs_score_raw']:.2f}\n\n")
                    f.write(f"**Adjusted Jobs Score**: {sentiment_data['jobs_score_adjusted']:.2f}\n\n")
                    
                    # Calculate impact percentage
                    if sentiment_data['jobs_score_raw'] != 0:
                        impact_pct = ((sentiment_data['jobs_score_adjusted'] - sentiment_data['jobs_score_raw']) / 
                                     abs(sentiment_data['jobs_score_raw'])) * 100
                        impact_direction = "increased" if impact_pct > 0 else "decreased"
                        f.write(f"Sentiment {impact_direction} jobs score by {abs(impact_pct):.1f}%\n\n")
            
            f.write("### Consumer Spending Sentiment\n\n")
            if sentiment_data.get('consumer_spending_sentiment'):
                cs_val = sentiment_data['consumer_spending_sentiment']
                # Try to get sentiment description
                try:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from economic_quadrant import get_sentiment_description
                    cs_desc = get_sentiment_description(cs_val)
                    f.write(f"**Consumer Spending Sentiment**: {cs_desc} ({cs_val:.2f})\n\n")
                except ImportError:
                    f.write(f"**Consumer Spending Sentiment Multiplier**: {cs_val:.2f}\n\n")
                
                # Add raw vs adjusted scores
                if sentiment_data.get('consumer_spending_score_raw') is not None and sentiment_data.get('consumer_spending_score_adjusted') is not None:
                    f.write(f"**Raw Consumer Spending Score**: {sentiment_data['consumer_spending_score_raw']:.2f}\n\n")
                    f.write(f"**Adjusted Consumer Spending Score**: {sentiment_data['consumer_spending_score_adjusted']:.2f}\n\n")
                    
                    # Calculate impact percentage
                    if sentiment_data['consumer_spending_score_raw'] != 0:
                        impact_pct = ((sentiment_data['consumer_spending_score_adjusted'] - sentiment_data['consumer_spending_score_raw']) / 
                                     abs(sentiment_data['consumer_spending_score_raw'])) * 100
                        impact_direction = "increased" if impact_pct > 0 else "decreased"
                        f.write(f"Sentiment {impact_direction} consumer spending score by {abs(impact_pct):.1f}%\n\n")
        
        # Add additional data if available in JSON format
        if latest_quadrant.get('json_data'):
            try:
                json_data = json.loads(latest_quadrant['json_data'])
                
                # Check for latest indicator values
                if 'balance_sheet_data' in json_data and json_data['balance_sheet_data']:
                    f.write("## Latest Economic Indicators\n\n")
                    
                    # Add balance sheet data
                    if len(json_data['balance_sheet_data']) > 0:
                        last_date = list(json_data['balance_sheet_data'].keys())[-1]
                        last_value = json_data['balance_sheet_data'][last_date]
                        f.write(f"**Fed Balance Sheet** ({last_date}): ${last_value:.2f} billion\n\n")
                    
                    # Add interest rate data
                    if 'interest_rate_data' in json_data and len(json_data['interest_rate_data']) > 0:
                        last_date = list(json_data['interest_rate_data'].keys())[-1]
                        last_value = json_data['interest_rate_data'][last_date]
                        f.write(f"**Federal Funds Rate** ({last_date}): {last_value:.2f}%\n\n")
                    
                    # Add jobs data
                    if 'jobs_data' in json_data and len(json_data['jobs_data']) > 0:
                        last_date = list(json_data['jobs_data'].keys())[-1]
                        last_value = json_data['jobs_data'][last_date]
                        f.write(f"**Non-Farm Payrolls** ({last_date}): {last_value:.0f} thousand jobs\n\n")
                    
                    # Add consumer spending data
                    if 'consumer_spending_data' in json_data and len(json_data['consumer_spending_data']) > 0:
                        last_date = list(json_data['consumer_spending_data'].keys())[-1]
                        last_value = json_data['consumer_spending_data'][last_date]
                        f.write(f"**Consumer Spending** ({last_date}): ${last_value:.2f} billion\n\n")
                    
                    # Add unemployment claims data if available
                    if 'unemployment_claims' in json_data:
                        claims_data = json_data.get('unemployment_claims', {})
                        if claims_data and len(claims_data) > 0:
                            last_date = list(claims_data.keys())[-1]
                            last_value = claims_data[last_date]
                            f.write(f"**Initial Unemployment Claims** ({last_date}): {last_value:.0f} claims\n\n")
                    
                    # Add consumer sentiment data if available
                    if 'consumer_sentiment' in json_data:
                        sentiment_data = json_data.get('consumer_sentiment', {})
                        if sentiment_data and len(sentiment_data) > 0:
                            last_date = list(sentiment_data.keys())[-1]
                            last_value = sentiment_data[last_date]
                            f.write(f"**UMich Consumer Sentiment** ({last_date}): {last_value:.1f}\n\n")
            except (json.JSONDecodeError, TypeError):
                pass
        
        f.write("## Economic Quadrant History\n\n")
        
        # Enhanced table with jobs, consumer spending, and sentiment data
        f.write("| Date | Quad | Balance Sheet | Int Rates | Jobs Trend | Jobs Sentiment | Spending Trend | Spending Sentiment |\n")
        f.write("|------|------|---------------|-----------|------------|----------------|----------------|-------------------|\n")
        
        # Import the get_sentiment_description function if needed
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from economic_quadrant import get_sentiment_description
            sentiment_desc_available = True
        except ImportError:
            sentiment_desc_available = False
        
        for _, row in quadrant_history.iterrows():
            date = row['analysis_date']
            quadrant = row['quadrant']
            balance = row['balance_sheet_trend']
            rates = row['interest_rate_level']
            
            # Try to extract the jobs and consumer spending data from JSON
            jobs_trend = "-"
            jobs_sentiment = "-"
            spending_trend = "-"
            spending_sentiment = "-"
            
            if pd.notnull(row['json_data']):
                try:
                    json_data = json.loads(row['json_data'])
                    
                    # Extract jobs data
                    if 'jobs_score' in json_data:
                        jobs_details = json_data.get('jobs_score', {})
                        if isinstance(jobs_details, dict) and 'trend' in jobs_details:
                            jobs_trend = jobs_details['trend']
                    
                    # Extract consumer spending data
                    if 'consumer_spending_score' in json_data:
                        cs_details = json_data.get('consumer_spending_score', {})
                        if isinstance(cs_details, dict) and 'trend' in cs_details:
                            spending_trend = cs_details['trend']
                    
                    # Extract sentiment data
                    if 'score_components' in json_data:
                        components = json_data.get('score_components', {})
                        
                        # Jobs sentiment
                        if 'jobs_sentiment' in components:
                            js_val = components['jobs_sentiment']
                            if sentiment_desc_available:
                                js_desc = get_sentiment_description(js_val)
                                jobs_sentiment = f"{js_desc} ({js_val:.2f})"
                            else:
                                jobs_sentiment = f"{js_val:.2f}"
                        
                        # Consumer spending sentiment
                        if 'consumer_spending_sentiment' in components:
                            cs_val = components['consumer_spending_sentiment']
                            if sentiment_desc_available:
                                cs_desc = get_sentiment_description(cs_val)
                                spending_sentiment = f"{cs_desc} ({cs_val:.2f})"
                            else:
                                spending_sentiment = f"{cs_val:.2f}"
                except (json.JSONDecodeError, TypeError):
                    pass
            
            f.write(f"| {date} | {quadrant} | {balance} | {rates} | {jobs_trend} | {jobs_sentiment} | {spending_trend} | {spending_sentiment} |\n")
    
    # Create a visualization of quadrant history if there's enough data
    if len(quadrant_history) > 1:
        plt.figure(figsize=(10, 6))
        
        # Convert quadrants to numeric values for plotting
        quadrant_map = {
            'A': 1, 
            'B': 2,
            'C': 3, 
            'D': 4,
            'B/C (prefer B)': 2.5, 
            'B/C (prefer C)': 2.7
        }
        
        # Make sure we handle all quadrant values properly
        quadrant_history['quadrant_numeric'] = quadrant_history['quadrant'].map(
            lambda q: quadrant_map.get(q, 2.5) # Default to middle if unknown
        )
        
        # Plot quadrant over time
        plt.plot(quadrant_history['analysis_date'], quadrant_history['quadrant_numeric'], 'o-', linewidth=2)
        plt.yticks([1, 2, 3, 4], ['A', 'B', 'C', 'D'])
        plt.xlabel('Date')
        plt.ylabel('Economic Quadrant')
        plt.title('Economic Quadrant History')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(report_path, 'quadrant_history.png'))
    
    # Create sentiment visualization if we have the data
    if sentiment_data:
        try:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart for sentiment multipliers
            labels = ['Jobs', 'Consumer Spending']
            sentiment_values = [
                sentiment_data.get('jobs_sentiment', 1.0),
                sentiment_data.get('consumer_spending_sentiment', 1.0)
            ]
            
            # Calculate colors - red for bearish, green for bullish, yellow for neutral
            colors = []
            for val in sentiment_values:
                if val < 0.95:  # Bearish
                    colors.append('salmon')
                elif val <= 1.05:  # Neutral
                    colors.append('khaki')
                else:  # Bullish
                    colors.append('lightgreen')
            
            plt.bar(labels, sentiment_values, color=colors)
            plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
            plt.text(-0.1, 1.01, 'Neutral', fontsize=8, color='gray')
            
            plt.xlabel('Economic Factor')
            plt.ylabel('Sentiment Multiplier')
            plt.title('Sentiment Analysis')
            plt.ylim(0, max(2.0, max(sentiment_values) * 1.1))  # Set y-axis upper limit
            
            # Add sentiment descriptions to the bars
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from economic_quadrant import get_sentiment_description
                
                for i, val in enumerate(sentiment_values):
                    desc = get_sentiment_description(val)
                    plt.text(i, val + 0.05, desc, ha='center', fontsize=9)
            except ImportError:
                pass
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_path, 'sentiment_analysis.png'))
            
            # Create a before/after chart for sentiment impact
            if all(k in sentiment_data for k in ['jobs_score_raw', 'jobs_score_adjusted', 
                                               'consumer_spending_score_raw', 'consumer_spending_score_adjusted']):
                plt.figure(figsize=(10, 6))
                
                # Data to plot
                labels = ['Jobs', 'Consumer Spending']
                raw_scores = [sentiment_data['jobs_score_raw'], sentiment_data['consumer_spending_score_raw']]
                adjusted_scores = [sentiment_data['jobs_score_adjusted'], sentiment_data['consumer_spending_score_adjusted']]
                
                x = range(len(labels))
                width = 0.35
                
                # Plot bars
                plt.bar([i - width/2 for i in x], raw_scores, width, label='Raw Score', color='skyblue')
                plt.bar([i + width/2 for i in x], adjusted_scores, width, label='Sentiment Adjusted', color='lightgreen')
                
                # Add labels and legend
                plt.xlabel('Economic Factor')
                plt.ylabel('Score (-10 to +10 scale)')
                plt.title('Impact of Sentiment on Economic Scores')
                plt.xticks(x, labels)
                plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(report_path, 'sentiment_impact.png'))
        except Exception as e:
            print(f"Error creating sentiment visualization: {e}")
    
    return latest_quadrant

def generate_stock_summary(report_path, current_quadrant=None):
    """Generate a summary of stock analysis results"""
    print("Generating stock analysis summary...")
    
    # Get stock statistics
    stats = get_db_stats()
    
    # Get top stocks (grade A+ and A)
    top_stocks = get_stocks_by_grade('A+')
    top_stocks = pd.concat([top_stocks, get_stocks_by_grade('A')])
    
    # Create a markdown report
    with open(os.path.join(report_path, 'stock_summary.md'), 'w') as f:
        f.write("# Stock Analysis Summary\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Database Statistics\n\n")
        f.write(f"Total stock analyses: {stats['stock_analysis_count']}\n\n")
        f.write(f"Analysis batches: {', '.join(stats['batches']) if stats['batches'] else 'None'}\n\n")
        
        f.write("## Grade Distribution\n\n")
        for grade, count in stats['grade_distribution'].items():
            if grade:  # Skip None values
                f.write(f"- **{grade}**: {count} stocks\n")
        
        f.write("\n## Top-Rated Stocks\n\n")
        
        if current_quadrant:
            f.write(f"Current Economic Quadrant: **{current_quadrant['quadrant']}**\n\n")
        
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
    
    # Create a visualization of grade distribution
    if stats['grade_distribution']:
        plt.figure(figsize=(10, 6))
        
        grades = []
        counts = []
        for grade, count in stats['grade_distribution'].items():
            if grade:  # Skip None values
                grades.append(grade)
                counts.append(count)
        
        # Create bar chart
        plt.bar(grades, counts, color='skyblue')
        plt.xlabel('Grade')
        plt.ylabel('Number of Stocks')
        plt.title('Stock Grade Distribution')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(report_path, 'grade_distribution.png'))
    
    # Create a sector distribution for top stocks
    if not top_stocks.empty:
        plt.figure(figsize=(12, 8))
        
        sector_counts = top_stocks['sector'].value_counts()
        
        # Create bar chart
        sector_counts.plot(kind='bar', color='lightgreen')
        plt.title('Sector Distribution of Top-Graded Stocks')
        plt.xlabel('Sector')
        plt.ylabel('Number of Stocks')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(report_path, 'top_stocks_by_sector.png'))

def generate_full_report():
    """Generate a complete market analysis report"""
    print("Generating market analysis report...")
    
    # Create report directory
    report_path = create_report_directory()
    print(f"Report will be saved to: {report_path}")
    
    # Generate economic summary
    current_quadrant = generate_economic_summary(report_path)
    
    # Generate stock summary
    generate_stock_summary(report_path, current_quadrant)
    
    # Create index file
    with open(os.path.join(report_path, 'index.md'), 'w') as f:
        f.write("# Market Analysis Report\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Contents\n\n")
        f.write("1. [Economic Conditions Summary](economic_summary.md)\n")
        f.write("2. [Stock Analysis Summary](stock_summary.md)\n\n")
        
        if current_quadrant:
            f.write(f"Current Economic Quadrant: **{current_quadrant['quadrant']}**\n\n")
            
            # Updated quadrant descriptions
            quadrant_descriptions = {
                'A': 'Inflation fighting quadrant - focus on value stocks',
                'B': 'Growth with inflation quadrant - balanced approach with growth bias',
                'C': 'Transition to growth quadrant - balanced approach with value bias',
                'D': 'Growth quadrant - focus on growth stocks',
                'B/C (prefer B)': 'Transition quadrant (prefer growth stocks)',
                'B/C (prefer C)': 'Transition quadrant (prefer value stocks)'
            }
            
            description = quadrant_descriptions.get(current_quadrant['quadrant'], 'Unknown quadrant')
            f.write(f"**{description}**\n\n")
            
            # Include sentiment information if available in current_quadrant
            try:
                json_data = json.loads(current_quadrant['json_data'])
                if 'score_components' in json_data:
                    f.write("### Sentiment Analysis\n\n")
                    
                    components = json_data['score_components']
                    
                    if 'jobs_sentiment' in components:
                        try:
                            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                            from economic_quadrant import get_sentiment_description
                            js_desc = get_sentiment_description(components['jobs_sentiment'])
                            f.write(f"**Jobs Sentiment**: {js_desc} ({components['jobs_sentiment']:.2f})\n\n")
                        except ImportError:
                            f.write(f"**Jobs Sentiment**: {components['jobs_sentiment']:.2f}\n\n")
                    
                    if 'consumer_spending_sentiment' in components:
                        try:
                            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                            from economic_quadrant import get_sentiment_description
                            cs_desc = get_sentiment_description(components['consumer_spending_sentiment'])
                            f.write(f"**Consumer Spending Sentiment**: {cs_desc} ({components['consumer_spending_sentiment']:.2f})\n\n")
                        except ImportError:
                            f.write(f"**Consumer Spending Sentiment**: {components['consumer_spending_sentiment']:.2f}\n\n")
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        
        f.write("### Visualizations\n\n")
        
        if os.path.exists(os.path.join(report_path, 'sentiment_analysis.png')):
            f.write("![Sentiment Analysis](sentiment_analysis.png)\n\n")
            
        if os.path.exists(os.path.join(report_path, 'sentiment_impact.png')):
            f.write("![Sentiment Impact on Economic Scores](sentiment_impact.png)\n\n")
        
        if os.path.exists(os.path.join(report_path, 'quadrant_history.png')):
            f.write("![Economic Quadrant History](quadrant_history.png)\n\n")
        
        if os.path.exists(os.path.join(report_path, 'grade_distribution.png')):
            f.write("![Stock Grade Distribution](grade_distribution.png)\n\n")
        
        if os.path.exists(os.path.join(report_path, 'top_stocks_by_sector.png')):
            f.write("![Top Stocks by Sector](top_stocks_by_sector.png)\n\n")
    
    print(f"Report generation complete. Open {os.path.join(report_path, 'index.md')} to view the report.")
    return report_path

if __name__ == "__main__":
    report_path = generate_full_report()
    print(f"Report saved to: {report_path}") 