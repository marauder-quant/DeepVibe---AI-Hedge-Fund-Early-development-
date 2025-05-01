#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for DMAC strategy.
This module provides functions for plotting strategy results and parameter space visualizations.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import logging

# Import from parent module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_PLOTS_DIR, INTERACTIVE_PLOTS, SAVE_PNG_PLOTS, DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT, DEFAULT_TIMEFRAME

def plot_dmac_strategy(results):
    """
    Plot the DMAC strategy results.
    
    Args:
        results (dict): Results from run_dmac_strategy
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Plot the OHLC data with MA lines and entry/exit points
    fig = results['ohlcv']['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
    fig = results['fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Fast MA'), fig=fig)
    fig = results['slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Slow MA'), fig=fig)
    fig = results['dmac_entries'].vbt.signals.plot_as_entry_markers(results['ohlcv']['Open'], fig=fig)
    fig = results['dmac_exits'].vbt.signals.plot_as_exit_markers(results['ohlcv']['Open'], fig=fig)
    
    # Update figure layout
    fig.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    # Plot equity comparison
    value_fig = results['dmac_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (DMAC)'))
    results['hold_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=value_fig)
    value_fig.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    # Plot trades
    trades_fig = results['dmac_pf'].trades.plot()
    trades_fig.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    # Plot returns
    returns_fig = results['dmac_pf'].returns().vbt.plot(
        trace_kwargs=dict(name='Returns (DMAC)'),
        title='Strategy Returns'
    )
    results['hold_pf'].returns().vbt.plot(
        trace_kwargs=dict(name='Returns (Hold)'),
        fig=returns_fig
    )
    returns_fig.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    # Plot drawdown
    drawdown_fig = results['dmac_pf'].drawdown().vbt.plot(
        trace_kwargs=dict(name='Drawdown (DMAC)'),
        title='Strategy Drawdown'
    )
    results['hold_pf'].drawdown().vbt.plot(
        trace_kwargs=dict(name='Drawdown (Hold)'),
        fig=drawdown_fig
    )
    drawdown_fig.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    return {
        'strategy_fig': fig,
        'value_fig': value_fig,
        'trades_fig': trades_fig,
        'returns_fig': returns_fig,
        'drawdown_fig': drawdown_fig
    }

def plot_heatmap(perf_matrix, metric='total_return'):
    """
    Plot a heatmap of strategy performance across window combinations.
    
    Args:
        perf_matrix (DataFrame): Performance matrix
        metric (str): Performance metric
        
    Returns:
        Figure: Heatmap figure
    """
    heatmap = perf_matrix.vbt.heatmap(
        xaxis_title='Slow window', 
        yaxis_title='Fast window',
        title=f'{metric} by window combination')
    
    heatmap.update_layout(width=DEFAULT_PLOT_WIDTH, height=DEFAULT_PLOT_HEIGHT)
    
    return heatmap

def create_parameter_space_visualization(performance_df, symbol, start_date, end_date):
    """
    Create visualizations of the parameter space.
    
    Args:
        performance_df (DataFrame): DataFrame with window combinations and performance
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        
    Returns:
        dict: Dictionary containing the created figures
    """
    figures = {}
    
    # Create 3D scatter plot of parameter space
    if len(performance_df) > 0:
        # Create 3D scatter plot
        fig_3d = px.scatter_3d(
            performance_df, 
            x='fast_window', 
            y='slow_window', 
            z='total_return',
            color='total_return',
            title=f'DMAC Parameter Space - {symbol} ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # Improve layout
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Fast Window',
                yaxis_title='Slow Window',
                zaxis_title='Total Return',
                xaxis=dict(type='linear'),
                yaxis=dict(type='linear'),
                zaxis=dict(tickformat='.1%')
            ),
            coloraxis_colorbar=dict(
                title='Return',
                tickformat='.1%'
            ),
            width=DEFAULT_PLOT_WIDTH,
            height=DEFAULT_PLOT_HEIGHT
        )
        
        figures['param_3d_fig'] = fig_3d
        
        # Create 2D heatmap of parameter space
        if len(performance_df) > 10:
            # Pivot the data to create a matrix
            try:
                # Create a pivot table for heatmap
                heatmap_data = performance_df.pivot_table(
                    index='fast_window', 
                    columns='slow_window', 
                    values='total_return'
                )
                
                # Create heatmap
                heatmap_fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Slow Window", y="Fast Window", color="Total Return"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title=f'DMAC Parameters Heatmap - {symbol}'
                )
                
                heatmap_fig.update_layout(
                    coloraxis_colorbar=dict(
                        title='Return',
                        tickformat='.1%'
                    ),
                    width=DEFAULT_PLOT_WIDTH,
                    height=DEFAULT_PLOT_HEIGHT
                )
                
                figures['param_heatmap_fig'] = heatmap_fig
            except Exception as e:
                print(f"Error creating heatmap: {e}")
        
        # Create bar chart of top performing parameters
        if len(performance_df) > 1:
            # Get top 10 parameter combinations
            top_params = performance_df.nlargest(min(10, len(performance_df)), 'total_return')
            
            # Create labels for parameter combinations
            top_params['param_label'] = top_params.apply(
                lambda row: f"Fast:{int(row['fast_window'])}, Slow:{int(row['slow_window'])}", 
                axis=1
            )
            
            # Create bar chart
            bar_fig = px.bar(
                top_params, 
                x='param_label', 
                y='total_return',
                title=f'Top DMAC Parameters - {symbol}',
                labels={'param_label': 'Parameter Combination', 'total_return': 'Total Return'},
                color='total_return',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            bar_fig.update_layout(
                xaxis_title='Parameter Combination',
                yaxis_title='Total Return',
                yaxis_tickformat='.1%',
                coloraxis_colorbar=dict(
                    title='Return',
                    tickformat='.1%'
                ),
                width=DEFAULT_PLOT_WIDTH,
                height=DEFAULT_PLOT_HEIGHT
            )
            
            figures['top_params_fig'] = bar_fig
    
    return figures

def save_plots(figures, symbol, start_date, end_date, output_dir=DEFAULT_PLOTS_DIR, timeframe=DEFAULT_TIMEFRAME):
    """
    Save all plots to the specified directory with a structured organization.
    
    Args:
        figures (dict): Dictionary of figures to save
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        output_dir (str): Base directory to save plots to
        timeframe (str): Timeframe of the data
        
    Returns:
        str: Path to the directory where plots were saved
    """
    # Create safe symbol name for filenames
    symbol_safe = symbol.replace('/', '_')
    
    # Format dates for directory and filenames
    date_dir = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    date_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
    
    # Create structured directory path
    plots_dir = os.path.join(output_dir, f"dmac_{timeframe}_backtest_plots", symbol_safe, date_dir)
    
    # Ensure output directory exists
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logging.info(f"Created directory: {plots_dir}")
    
    logging.info(f"\nSaving plots to {plots_dir}")
    
    saved_files = []
    
    # Save each figure as PNG if enabled
    if SAVE_PNG_PLOTS:
        for name, fig in figures.items():
            filename = os.path.join(plots_dir, f"{symbol_safe}_{name}_{date_str}.png")
            try:
                fig.write_image(filename)
                logging.info(f"Saved {name} plot to {filename}")
                saved_files.append(filename)
            except Exception as e:
                logging.error(f"Error saving {name} plot: {str(e)}")
            
    # Generate HTML versions for interactive viewing if enabled
    if INTERACTIVE_PLOTS:
        html_dir = os.path.join(plots_dir, "interactive")
        if not os.path.exists(html_dir):
            os.makedirs(html_dir)
            
        for name, fig in figures.items():
            html_filename = os.path.join(html_dir, f"{symbol_safe}_{name}_{date_str}.html")
            try:
                fig.write_html(html_filename)
                logging.info(f"Saved interactive {name} plot to {html_filename}")
                saved_files.append(html_filename)
            except Exception as e:
                logging.error(f"Error saving interactive {name} plot: {str(e)}")
                
    # Create a summary file
    summary_file = os.path.join(plots_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"DMAC Backtest Plots Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Timeframe: {timeframe}\n")
        f.write(f"Date Range: {date_str}\n\n")
        f.write(f"Generated plots:\n")
        for name in figures.keys():
            f.write(f"- {name}\n")
        
        f.write(f"\nSaved files:\n")
        for file in saved_files:
            f.write(f"- {os.path.basename(file)}\n")
            
    return plots_dir 