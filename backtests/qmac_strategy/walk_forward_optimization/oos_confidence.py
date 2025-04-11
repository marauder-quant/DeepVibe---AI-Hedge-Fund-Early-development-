#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Confidence tracking and reporting for QMAC strategy out-of-sample testing.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rich.console import Console
from rich.table import Table
import logging

# Import configuration
from backtests.qmac_strategy.walk_forward_optimization.oos_config import CONFIDENCE_TRACKER_FILE
from backtests.qmac_strategy.walk_forward_optimization.oos_utils import get_latest_in_sample_performance

# Console setup for rich output
console = Console()
log = logging.getLogger("rich")

def load_confidence_tracker():
    """Load the confidence tracking data from file or initialize if not exists."""
    if os.path.exists(CONFIDENCE_TRACKER_FILE):
        with open(CONFIDENCE_TRACKER_FILE, 'r') as f:
            return json.load(f)
    else:
        # Initialize new confidence tracker
        return {
            'tracker_version': '1.0',
            'tests': [],
            'confidence_metrics': {
                'alpha_quality_score': 0,
                'overall_confidence': 0
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def save_confidence_tracker(data):
    """Save the confidence tracking data to file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(CONFIDENCE_TRACKER_FILE), exist_ok=True)
    with open(CONFIDENCE_TRACKER_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def calculate_alpha_quality(results_df):
    """
    Calculate alpha quality score based on alpha distribution.
    Higher values indicate strategy adds more value over buy-and-hold.
    """
    # If no results, return zero
    if len(results_df) == 0:
        return 0.0
    
    # Calculate percentage of positive alphas
    alpha_success_rate = (results_df['alpha'] > 0).mean()
    
    # Calculate average alpha
    mean_alpha = results_df['alpha'].mean()
    
    # Calculate alpha t-statistic
    if len(results_df) > 1:
        alpha_t_stat = stats.ttest_1samp(results_df['alpha'], 0)[0]
        # Normalize t-stat to [0, 1] range, cap at 3 (very significant)
        normalized_t = min(abs(alpha_t_stat) / 3, 1.0) if alpha_t_stat > 0 else 0.0
    else:
        normalized_t = 0.0
    
    # Combine metrics: 60% on success rate, 30% on mean alpha, 10% on t-stat
    alpha_quality = (
        0.6 * alpha_success_rate + 
        0.3 * min(max(mean_alpha * 3, 0), 1) +  # Scale mean alpha, cap at 0-1
        0.1 * normalized_t
    )
    
    return alpha_quality

def calculate_statistical_confidence(returns):
    """Calculate statistical confidence using t-test."""
    # One-sample t-test against zero
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    
    # If t-statistic is positive, we're testing if mean is > 0
    # If negative, we don't have evidence strategy works
    if t_stat <= 0:
        return 1.0  # worst p-value
    
    return p_value

def calculate_avg_alpha_quality(tests):
    """Calculate average alpha quality from all tests."""
    if not tests:
        return 0.0
    
    alphas = []
    for test in tests:
        if 'confidence_metrics' in test and 'alpha_quality' in test['confidence_metrics']:
            alphas.append(test['confidence_metrics']['alpha_quality'])
    
    if not alphas:
        return 0.0
        
    return sum(alphas) / len(alphas)

def update_confidence_after_batch(params, results_df, timestamp, timeframe):
    """Update the confidence metrics after processing a batch of results."""
    # Calculate current summary statistics
    current_summary = {
        'n_tests': len(results_df),
        'avg_return': results_df['total_return'].mean(),
        'median_return': results_df['total_return'].median(),
        'std_return': results_df['total_return'].std() if len(results_df) > 1 else 0,
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_trades': results_df['n_trades'].mean(),
        'success_rate': (results_df['total_return'] > 0).mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean()
    }
    
    # Get in-sample performance data
    in_sample_data = get_latest_in_sample_performance(timeframe)
    
    # Calculate confidence metric - only using alpha_quality now
    alpha_quality = calculate_alpha_quality(results_df)
    overall_confidence = alpha_quality  # Directly use alpha_quality as the overall confidence
    
    # Create a mini table for batch update
    batch_table = Table(title=f"Batch Update: {len(results_df)} Tests Completed")
    batch_table.add_column("Metric", style="cyan")
    batch_table.add_column("Value", style="green")
    
    batch_table.add_row("Avg Return", f"{current_summary['avg_return']:.2%}")
    batch_table.add_row("Avg Alpha", f"{current_summary['avg_alpha']:.2%}")
    batch_table.add_row("Alpha Success", f"{current_summary['alpha_success_rate']:.2%}")
    batch_table.add_row("Avg Theta", f"{current_summary['avg_theta']:.2%}")
    batch_table.add_row("Current Confidence", f"{overall_confidence:.2%}")
    
    console.print(batch_table)
    
    return current_summary, overall_confidence

def update_confidence_after_single_test(params, result, test_num, results_so_far, timestamp, timeframe):
    """Update confidence tracker after a single test for live updates."""
    log.info(f"Updating confidence after test {test_num}...")
    
    # Convert results so far to DataFrame
    results_df = pd.DataFrame(results_so_far)
    
    # Calculate current summary statistics
    current_summary = {
        'n_tests': len(results_df),
        'avg_return': results_df['total_return'].mean(),
        'median_return': results_df['total_return'].median(),
        'std_return': results_df['total_return'].std() if len(results_df) > 1 else 0,
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_trades': results_df['n_trades'].mean(),
        'success_rate': (results_df['total_return'] > 0).mean(),
        'avg_alpha': results_df['alpha'].mean(),
        'alpha_success_rate': (results_df['alpha'] > 0).mean(),
        'avg_theta': results_df['theta'].mean(),
        'theta_success_rate': (results_df['theta'] > 0).mean()
    }
    
    # Get in-sample performance data
    in_sample_data = get_latest_in_sample_performance(timeframe)
    
    # Calculate confidence metric - only using alpha_quality now
    alpha_quality = calculate_alpha_quality(results_df)
    overall_confidence = alpha_quality  # Directly use alpha_quality as the overall confidence
    
    # Load existing tracker
    tracker = load_confidence_tracker()
    
    # Create test entry for current state
    test_entry = {
        'timestamp': f"{timestamp}_test{test_num}",
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': params,
        'timeframe': timeframe,
        'in_sample_return': in_sample_data.get('total_return', 0),
        'out_of_sample_return': current_summary['avg_return'],
        'return_delta': current_summary['avg_return'] - in_sample_data.get('total_return', 0),
        'success_rate': current_summary['success_rate'],
        'n_tests': current_summary['n_tests'],
        'avg_alpha': current_summary['avg_alpha'],
        'alpha_success_rate': current_summary['alpha_success_rate'],
        'avg_theta': current_summary['avg_theta'],
        'theta_success_rate': current_summary['theta_success_rate'],
        'confidence_metrics': {
            'alpha_quality': alpha_quality,
            'p_value': calculate_statistical_confidence(results_df['total_return']) if len(results_df) > 1 else 1.0
        }
    }
    
    # If there are previous entries from this batch, remove them
    # to avoid cluttering the tracker with intermediate results
    if 'tests' in tracker:
        tracker['tests'] = [t for t in tracker['tests'] 
                           if 'timestamp' in t and not t['timestamp'].startswith(timestamp)]
    else:
        tracker['tests'] = []
    
    # Add current test entry
    tracker['tests'].append(test_entry)
    
    # Calculate alpha quality score across all tests
    alpha_quality_score = calculate_avg_alpha_quality(tracker['tests'])
    
    # Update overall confidence metrics - use alpha_quality_score as the overall confidence
    tracker['confidence_metrics'] = {
        'alpha_quality_score': alpha_quality_score,
        'overall_confidence': alpha_quality_score  # Use alpha_quality_score as overall_confidence
    }
    tracker['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated tracker
    save_confidence_tracker(tracker)
    
    # Return current summary and overall confidence
    return current_summary, overall_confidence

def update_confidence_tracker(params, summary, results_df, timestamp, timeframe):
    """Update the confidence tracker with new test results."""
    log.info("Updating confidence tracker...")
    
    # Load existing tracker data
    tracker = load_confidence_tracker()
    
    # Calculate confidence metric
    confidence = summary['alpha_success_rate'] * 0.5 + summary['theta_success_rate'] * 0.3 + (results_df['total_return'] > 0).mean() * 0.2
    
    # Create new test entry with only the requested fields
    test_entry = {
        'timestamp': timestamp,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date_from': results_df['start_date'].min().strftime('%Y-%m-%d') if 'start_date' in results_df.columns else None,
        'date_to': results_df['end_date'].max().strftime('%Y-%m-%d') if 'end_date' in results_df.columns else None,
        'out_of_sample_return': summary['avg_return'],
        'avg_alpha': summary['avg_alpha'],
        'alpha_success_rate': summary['alpha_success_rate'],
        'avg_theta': summary['avg_theta'],
        'theta_success_rate': summary['theta_success_rate'],
        'confidence': confidence
    }
    
    # Add test to tracker
    tracker['tests'].append(test_entry)
    
    # Update overall confidence metrics
    tracker['confidence_metrics'] = {
        'overall_confidence': confidence
    }
    tracker['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated tracker
    save_confidence_tracker(tracker)
    
    # Generate confidence report
    generate_confidence_report(tracker, timestamp)
    
    log.info(f"Confidence tracker updated. Overall confidence score: {confidence:.2%}")
    
    return confidence

def generate_confidence_report(tracker, timestamp):
    """Generate and save a visual confidence report."""
    report_file = os.path.join('backtests/qmac_strategy/results', f'confidence_report_{timestamp}.png')
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Extract data from tracker
    tests = tracker['tests']
    if len(tests) < 2:  # Need at least 2 tests for meaningful plots
        log.info("Not enough data to generate confidence report yet")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([{
        'date': test['date'],
        'start_date': test.get('date_from', test['date']),
        'end_date': test.get('date_to', test['date']),
        'avg_return': test['out_of_sample_return'],
        'avg_alpha': test.get('avg_alpha', 0),
        'alpha_success': test.get('alpha_success_rate', 0),
        'avg_theta': test.get('avg_theta', 0),
        'theta_success': test.get('theta_success_rate', 0),
        'confidence': test.get('confidence_metrics', {}).get('alpha_quality', 0) * 0.5 + 
                      test.get('theta_success_rate', 0) * 0.3 + 
                      test.get('success_rate', 0) * 0.2
    } for test in tests])
    df['date'] = pd.to_datetime(df['date'])
    
    # Configure the plot - use a modern style that's available in all matplotlib versions
    try:
        plt.style.use('ggplot')
    except:
        # Fallback to default style if ggplot is not available
        plt.style.use('default')
    
    # Plot 1: Average Return and Alpha
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['avg_return'], 'g-', label='Avg Return')
    plt.plot(df['date'], df['avg_alpha'], 'r-', label='Avg Alpha')
    plt.title('Performance Metrics Over Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Plot 2: Alpha and Theta Success Rates
    plt.subplot(2, 2, 2)
    plt.plot(df['date'], df['alpha_success'], 'b-', label='Alpha Success')
    plt.plot(df['date'], df['theta_success'], 'y-', label='Theta Success')
    plt.title('Success Rates Over Time')
    plt.ylabel('Rate (0-1)')
    plt.legend()
    plt.tight_layout()
    
    # Plot 3: Average Theta
    plt.subplot(2, 2, 3)
    plt.plot(df['date'], df['avg_theta'], 'm-', label='Avg Theta')
    plt.title('Theta Performance Over Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    
    # Plot 4: Confidence
    plt.subplot(2, 2, 4)
    plt.plot(df['date'], df['confidence'], 'g-', label='Confidence')
    plt.title('Strategy Confidence Over Time')
    plt.ylabel('Confidence (0-1)')
    plt.legend()
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle(f'QMAC Strategy Performance Summary - {datetime.now().strftime("%Y-%m-%d")}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(report_file)
    log.info(f"Confidence report saved to {report_file}")
    plt.close()

def display_confidence_summary():
    """Display a summary of the current confidence in the strategy."""
    if not os.path.exists(CONFIDENCE_TRACKER_FILE):
        console.print("[yellow]No confidence tracking data available yet. Run out-of-sample tests to generate data.")
        return
    
    tracker = load_confidence_tracker()
    
    # Create a rich table
    table = Table(title="QMAC Strategy Confidence Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add confidence metrics
    metrics = tracker['confidence_metrics']
    table.add_row("Overall Confidence", f"{metrics.get('overall_confidence', 0):.2%}")
    
    # Add alpha metrics if available
    if 'alpha_quality_score' in metrics:
        table.add_row("Alpha Quality", f"{metrics['alpha_quality_score']:.2%}")
    
    # Add latest test info if available
    if tracker['tests']:
        latest = tracker['tests'][-1]
        table.add_row("Latest Test Date", latest['date'])
        table.add_row("In-sample Return", f"{latest['in_sample_return']:.2%}")
        table.add_row("Out-of-sample Return", f"{latest['out_of_sample_return']:.2%}")
        
        # Add alpha and theta info if available
        if 'avg_alpha' in latest:
            table.add_row("Alpha", f"{latest['avg_alpha']:.2%}")
            table.add_row("Alpha Success Rate", f"{latest['alpha_success_rate']:.2%}")
        
        if 'avg_theta' in latest:
            table.add_row("Theta", f"{latest['avg_theta']:.2%}")
            table.add_row("Theta Success Rate", f"{latest['theta_success_rate']:.2%}")
        
        table.add_row("Success Rate", f"{latest['success_rate']:.2%}")
    
    table.add_row("Total Tests Run", str(len(tracker['tests'])))
    table.add_row("Last Updated", tracker['last_updated'])
    
    console.print(table)
    
    # If confidence is high, print a success message
    if metrics.get('overall_confidence', 0) >= 0.7:
        console.print("[green]High confidence in strategy parameters!")
    elif metrics.get('overall_confidence', 0) >= 0.4:
        console.print("[yellow]Moderate confidence in strategy parameters.")
    else:
        console.print("[red]Low confidence in strategy parameters. Consider re-optimizing.") 