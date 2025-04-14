#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for the QMAC strategy.
This module contains functions for plotting strategy results and parameter space.
"""

import os
import plotly.graph_objects as go
import pandas as pd

def plot_qmac_strategy(results):
    """
    Plot the QMAC strategy results.
    
    Args:
        results (dict): Results from run_qmac_strategy
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Plot the OHLC data with MA lines and entry/exit points
    fig = results['ohlcv']['Open'].vbt.plot(trace_kwargs=dict(name='Price'))
    fig = results['buy_fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Buy Fast MA'), fig=fig)
    fig = results['buy_slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Buy Slow MA'), fig=fig)
    fig = results['sell_fast_ma'].ma.vbt.plot(trace_kwargs=dict(name='Sell Fast MA'), fig=fig)
    fig = results['sell_slow_ma'].ma.vbt.plot(trace_kwargs=dict(name='Sell Slow MA'), fig=fig)
    fig = results['qmac_entries'].vbt.signals.plot_as_entry_markers(results['ohlcv']['Open'], fig=fig)
    fig = results['qmac_exits'].vbt.signals.plot_as_exit_markers(results['ohlcv']['Open'], fig=fig)
    
    # Plot equity comparison
    value_fig = results['qmac_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (QMAC)'))
    results['hold_pf'].value().vbt.plot(trace_kwargs=dict(name='Value (Hold)'), fig=value_fig)
    
    # Plot trades
    trades_fig = results['qmac_pf'].trades.plot()
    
    return {
        'strategy_fig': fig,
        'value_fig': value_fig,
        'trades_fig': trades_fig
    }

def create_parameter_space_visualization(performance_df, symbol, start_date, end_date):
    """
    Create visualizations of the 4D parameter space.
    
    Args:
        performance_df (DataFrame): DataFrame containing performance data for all combinations
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        
    Returns:
        dict: Dictionary containing the created figures
    """
    # Calculate ratios for visualization
    performance_df['buy_ratio'] = performance_df['buy_fast'] / performance_df['buy_slow']
    performance_df['sell_ratio'] = performance_df['sell_fast'] / performance_df['sell_slow']
    
    # Create 2D heatmaps for buy and sell parameters
    buy_heatmap = go.Figure(data=go.Heatmap(
        z=performance_df.groupby(['buy_fast', 'buy_slow'])['total_return'].mean().unstack(),
        x=performance_df['buy_slow'].unique(),
        y=performance_df['buy_fast'].unique(),
        colorscale='RdYlGn',
        colorbar=dict(title='Average Return')
    ))
    buy_heatmap.update_layout(
        title=f'Buy Parameters Performance Heatmap - {symbol}',
        xaxis_title='Buy Slow Window',
        yaxis_title='Buy Fast Window'
    )
    
    sell_heatmap = go.Figure(data=go.Heatmap(
        z=performance_df.groupby(['sell_fast', 'sell_slow'])['total_return'].mean().unstack(),
        x=performance_df['sell_slow'].unique(),
        y=performance_df['sell_fast'].unique(),
        colorscale='RdYlGn',
        colorbar=dict(title='Average Return')
    ))
    sell_heatmap.update_layout(
        title=f'Sell Parameters Performance Heatmap - {symbol}',
        xaxis_title='Sell Slow Window',
        yaxis_title='Sell Fast Window'
    )
    
    # Create 3D surface plot
    surface_plot = go.Figure(data=go.Surface(
        z=performance_df.groupby(['buy_ratio', 'sell_ratio'])['total_return'].mean().unstack(),
        x=performance_df['sell_ratio'].unique(),
        y=performance_df['buy_ratio'].unique(),
        colorscale='RdYlGn'
    ))
    surface_plot.update_layout(
        title=f'Performance Surface - {symbol}',
        scene=dict(
            xaxis_title='Sell Ratio',
            yaxis_title='Buy Ratio',
            zaxis_title='Return'
        )
    )
    
    # Create parallel coordinates plot
    top_performers = performance_df.nlargest(100, 'total_return')
    parallel_plot = go.Figure(data=go.Parcoords(
        line=dict(
            color=top_performers['total_return'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Return')
        ),
        dimensions=[
            dict(label='Buy Fast', values=top_performers['buy_fast']),
            dict(label='Buy Slow', values=top_performers['buy_slow']),
            dict(label='Sell Fast', values=top_performers['sell_fast']),
            dict(label='Sell Slow', values=top_performers['sell_slow']),
            dict(label='Return', values=top_performers['total_return'])
        ]
    ))
    parallel_plot.update_layout(
        title=f'Top 100 Performers - {symbol}'
    )
    
    # Create scatter plot of top performers
    scatter_plot = go.Figure(data=go.Scatter(
        x=top_performers['buy_ratio'],
        y=top_performers['sell_ratio'],
        mode='markers',
        marker=dict(
            size=10,
            color=top_performers['total_return'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Return')
        ),
        text=[f"Buy: {row['buy_fast']}/{row['buy_slow']}<br>Sell: {row['sell_fast']}/{row['sell_slow']}<br>Return: {row['total_return']:.2%}" 
              for _, row in top_performers.iterrows()]
    ))
    scatter_plot.update_layout(
        title=f'Top 100 Performers by Ratio - {symbol}',
        xaxis_title='Buy Ratio',
        yaxis_title='Sell Ratio'
    )
    
    return {
        'buy_heatmap': buy_heatmap,
        'sell_heatmap': sell_heatmap,
        'surface_plot': surface_plot,
        'parallel_plot': parallel_plot,
        'scatter_plot': scatter_plot
    }

def save_plots(figures, symbol, start_date, end_date, output_dir='plots'):
    """
    Save all plots to the specified directory.
    
    Args:
        figures (dict): Dictionary of figures to save
        symbol (str): Trading symbol
        start_date (datetime): Start date of backtest
        end_date (datetime): End date of backtest
        output_dir (str): Directory to save plots to
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Format dates for filenames
    date_str = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
    
    # Create safe symbol name for filenames
    symbol_safe = symbol.replace('/', '_')
    
    # Save each figure
    for name, fig in figures.items():
        filename = os.path.join(output_dir, f"{symbol_safe}_{name}_{date_str}.html")
        try:
            fig.write_html(filename)
            print(f"Saved {name} plot to {filename}")
        except Exception as e:
            print(f"Error saving {name} plot: {str(e)}") 