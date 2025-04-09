"""
Run full S&P 500 stock analysis based on economic quadrant
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_grading import analyze_stocks_for_quadrant
from economic_quadrant import determine_economic_quadrant
import time
# Import database utilities
from db_utils import ensure_db_exists, get_db_stats

def create_visualization(graded_stocks):
    """
    Create visualizations of the stock grading results
    """
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Sector distribution of top stocks
    top_stocks = graded_stocks[graded_stocks['overall_grade'].isin(['A+', 'A'])]
    
    if not top_stocks.empty:
        sector_counts = top_stocks['sector'].value_counts()
        
        plt.figure(figsize=(12, 8))
        sector_counts.plot(kind='bar', color='skyblue')
        plt.title('Sector Distribution of Top-Graded Stocks')
        plt.xlabel('Sector')
        plt.ylabel('Number of Stocks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualizations/top_stocks_by_sector.png')
        
    # 2. Grade distribution
    grade_counts = graded_stocks['overall_grade'].value_counts()
    
    plt.figure(figsize=(10, 6))
    grade_counts.plot(kind='bar', color='lightgreen')
    plt.title('Distribution of Overall Grades')
    plt.xlabel('Grade')
    plt.ylabel('Number of Stocks')
    plt.tight_layout()
    plt.savefig('visualizations/grade_distribution.png')
    
    # 3. Debt to EBITDA vs PE Ratio scatter plot for top stocks
    if not top_stocks.empty and 'debt_to_ebitda' in top_stocks.columns and 'pe_ratio' in top_stocks.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        plot_data = top_stocks.dropna(subset=['debt_to_ebitda', 'pe_ratio'])
        
        if not plot_data.empty:
            plt.scatter(plot_data['debt_to_ebitda'], plot_data['pe_ratio'], alpha=0.7)
            
            # Add labels for each point
            for idx, row in plot_data.iterrows():
                plt.annotate(idx, (row['debt_to_ebitda'], row['pe_ratio']), 
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title('Debt to EBITDA vs P/E Ratio for Top-Graded Stocks')
            plt.xlabel('Debt to EBITDA')
            plt.ylabel('P/E Ratio')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('visualizations/debt_vs_pe.png')
    
    # 4. Revenue Growth vs Earnings Growth scatter plot
    if not top_stocks.empty and 'revenue_growth' in top_stocks.columns and 'earnings_growth' in top_stocks.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        plot_data = top_stocks.dropna(subset=['revenue_growth', 'earnings_growth'])
        
        if not plot_data.empty:
            plt.scatter(plot_data['revenue_growth'], plot_data['earnings_growth'], alpha=0.7)
            
            # Add labels for each point
            for idx, row in plot_data.iterrows():
                plt.annotate(idx, (row['revenue_growth'], row['earnings_growth']), 
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title('Revenue Growth vs Earnings Growth for Top-Graded Stocks')
            plt.xlabel('Revenue Growth')
            plt.ylabel('Earnings Growth')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/growth_metrics.png')
    
    return ['visualizations/top_stocks_by_sector.png', 
            'visualizations/grade_distribution.png',
            'visualizations/debt_vs_pe.png',
            'visualizations/growth_metrics.png']

def run_full_analysis():
    """
    Run the full S&P 500 stock analysis
    """
    start_time = time.time()
    
    print("Starting full S&P 500 analysis...")
    print("This may take some time as we process all 500+ stocks.")
    
    # Ensure database exists and is initialized
    print("Initializing database...")
    ensure_db_exists()
    
    # Get current economic quadrant
    quadrant, balance_sheet_trend, interest_rate_level = determine_economic_quadrant()
    print(f"Current Economic Quadrant: {quadrant}")
    print(f"Balance Sheet Trend: {balance_sheet_trend}")
    print(f"Interest Rate Level: {interest_rate_level}")
    
    # Run the full analysis
    graded_stocks, top_stocks = analyze_stocks_for_quadrant()
    
    # Create visualizations
    visualization_paths = create_visualization(graded_stocks)
    
    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    
    # Get database stats
    db_stats = get_db_stats()
    
    print(f"\nAnalysis complete in {runtime:.2f} seconds.")
    print(f"Found {len(top_stocks)} top-graded stocks out of {len(graded_stocks)} total stocks analyzed.")
    print(f"Results saved to 'graded_stocks.csv', 'top_stocks.csv', and database")
    print(f"Visualizations saved to: {', '.join(visualization_paths)}")
    print("\nDatabase Statistics:")
    print(f"Total stock analyses: {db_stats['stock_analysis_count']}")
    print(f"Total economic quadrant analyses: {db_stats['economic_quadrant_count']}")
    print(f"Analysis batches: {', '.join(db_stats['batches']) if db_stats['batches'] else 'None'}")
    
    return graded_stocks, top_stocks, visualization_paths

if __name__ == "__main__":
    graded_stocks, top_stocks, visualization_paths = run_full_analysis()
    
    print("\nTop Stocks:")
    print(top_stocks[['name', 'sector', 'revenue_growth', 'earnings_growth', 'pe_ratio', 'debt_to_ebitda', 'overall_grade']])
