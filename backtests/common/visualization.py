"""
Visualization utilities for backtesting strategies.

This module provides common visualization functions for strategy results
that can be shared across different strategy implementations.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists.
    
    Args:
        directory: Path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_strategy_comparison(
    strategy_pf: Any,
    benchmark_pf: Any,
    title: str = "Strategy vs Benchmark Performance",
    legend_names: Tuple[str, str] = ("Strategy", "Benchmark"),
    include_drawdown: bool = True,
    include_trades: bool = True,
    figsize: Tuple[int, int] = (1200, 800)
) -> Dict[str, go.Figure]:
    """
    Plot a comparison between strategy and benchmark performance.
    
    Args:
        strategy_pf: Strategy portfolio object (vectorbt Portfolio)
        benchmark_pf: Benchmark portfolio object (vectorbt Portfolio)
        title: Plot title
        legend_names: Names for strategy and benchmark in legend
        include_drawdown: Whether to include drawdown subplot
        include_trades: Whether to include trade markers
        figsize: Figure size as (width, height)
        
    Returns:
        Dictionary of plotly figures
    """
    figures = {}
    
    # 1. Value comparison plot
    if include_drawdown:
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[title, "Drawdown"],
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
        fig.update_layout(title=title)
    
    # Add strategy value line
    strategy_value = strategy_pf.value()
    fig.add_trace(
        go.Scatter(
            x=strategy_value.index,
            y=strategy_value.values,
            name=legend_names[0],
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add benchmark value line
    benchmark_value = benchmark_pf.value()
    fig.add_trace(
        go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            name=legend_names[1],
            line=dict(color='gray', width=2, dash='dot')
        ),
        row=1, col=1
    )
    
    # Add trade markers if requested
    if include_trades:
        # Entry markers
        entries = strategy_pf.trades.records[strategy_pf.trades.records['side'] == 0]
        if len(entries) > 0:
            entry_times = entries['entry_time'].tolist()
            entry_values = [strategy_value.loc[time] for time in entry_times]
            fig.add_trace(
                go.Scatter(
                    x=entry_times,
                    y=entry_values,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Entries'
                ),
                row=1, col=1
            )
        
        # Exit markers
        exits = strategy_pf.trades.records[strategy_pf.trades.records['side'] == 0]
        if len(exits) > 0:
            exit_times = exits['exit_time'].tolist()
            exit_values = [strategy_value.loc[time] for time in exit_times]
            fig.add_trace(
                go.Scatter(
                    x=exit_times,
                    y=exit_values,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Exits'
                ),
                row=1, col=1
            )
    
    # Add drawdown subplot if requested
    if include_drawdown:
        # Strategy drawdown
        strategy_dd = strategy_pf.drawdown()
        fig.add_trace(
            go.Scatter(
                x=strategy_dd.index,
                y=strategy_dd.values,
                name=f"{legend_names[0]} Drawdown",
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ),
            row=2, col=1
        )
        
        # Benchmark drawdown
        benchmark_dd = benchmark_pf.drawdown()
        fig.add_trace(
            go.Scatter(
                x=benchmark_dd.index,
                y=benchmark_dd.values,
                name=f"{legend_names[1]} Drawdown",
                line=dict(color='gray', width=2, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(128, 128, 128, 0.1)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        yaxis2_title="Drawdown (%)",
        template="plotly_white"
    )
    
    figures['value_comparison'] = fig
    
    # 2. Create a standalone trades plot
    trades_fig = strategy_pf.trades.plot()
    figures['trades'] = trades_fig
    
    # 3. Create a performance metrics table
    metrics_fig = go.Figure(data=[
        go.Table(
            header=dict(
                values=['Metric', legend_names[0], legend_names[1]],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                     'Win Rate (%)', 'Profit Factor', 'Avg. Trade (%)'],
                    [
                        f"{strategy_pf.total_return() * 100:.2f}%",
                        f"{strategy_pf.sharpe_ratio():.2f}",
                        f"{strategy_pf.max_drawdown() * 100:.2f}%",
                        f"{strategy_pf.trades.win_rate() * 100:.2f}%",
                        f"{strategy_pf.trades.profit_factor():.2f}",
                        f"{strategy_pf.trades.avg_profit() * 100:.2f}%"
                    ],
                    [
                        f"{benchmark_pf.total_return() * 100:.2f}%",
                        f"{benchmark_pf.sharpe_ratio():.2f}",
                        f"{benchmark_pf.max_drawdown() * 100:.2f}%",
                        "N/A",
                        "N/A",
                        "N/A"
                    ]
                ],
                fill_color='lavender',
                align='left'
            )
        )
    ])
    
    metrics_fig.update_layout(
        title="Performance Metrics",
        width=800
    )
    
    figures['metrics'] = metrics_fig
    
    return figures


def plot_ma_strategy(
    ohlc_data: pd.DataFrame,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    title: str = "Moving Average Strategy",
    ma_names: Tuple[str, str] = ("Fast MA", "Slow MA"),
    figsize: Tuple[int, int] = (1200, 600)
) -> go.Figure:
    """
    Plot a moving average strategy with price data and crossover signals.
    
    Args:
        ohlc_data: OHLC price data
        fast_ma: Fast moving average series
        slow_ma: Slow moving average series
        entries: Entry signal series (boolean)
        exits: Exit signal series (boolean)
        title: Plot title
        ma_names: Names for fast and slow MA in legend
        figsize: Figure size as (width, height)
        
    Returns:
        Plotly figure
    """
    # Create figure with price data
    fig = go.Figure()
    
    # Add candlestick or line chart based on available data
    if all(col in ohlc_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(
            go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data['Open'],
                high=ohlc_data['High'],
                low=ohlc_data['Low'],
                close=ohlc_data['Close'],
                name="Price"
            )
        )
    else:
        # Fallback to line chart if OHLC data not available
        fig.add_trace(
            go.Scatter(
                x=ohlc_data.index,
                y=ohlc_data['Close'],
                name="Price",
                line=dict(color='black', width=1)
            )
        )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=fast_ma.index,
            y=fast_ma.values,
            name=ma_names[0],
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=slow_ma.index,
            y=slow_ma.values,
            name=ma_names[1],
            line=dict(color='red', width=2)
        )
    )
    
    # Add entry/exit markers
    if entries is not None:
        entry_points = ohlc_data.index[entries]
        entry_values = ohlc_data.loc[entry_points, 'Low']
        
        fig.add_trace(
            go.Scatter(
                x=entry_points,
                y=entry_values * 0.99,  # Slightly below the low price
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='Entries'
            )
        )
    
    if exits is not None:
        exit_points = ohlc_data.index[exits]
        exit_values = ohlc_data.loc[exit_points, 'High']
        
        fig.add_trace(
            go.Scatter(
                x=exit_points,
                y=exit_values * 1.01,  # Slightly above the high price
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='Exits'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig


def plot_heatmap(
    perf_matrix: pd.DataFrame,
    metric: str = "total_return",
    title: str = None,
    colorscale: str = "RdYlGn",
    figsize: Tuple[int, int] = (800, 600)
) -> go.Figure:
    """
    Plot a heatmap of strategy performance across parameter combinations.
    
    Args:
        perf_matrix: Performance matrix (DataFrame)
        metric: Performance metric name
        title: Plot title (defaults to metric name)
        colorscale: Plotly colorscale
        figsize: Figure size as (width, height)
        
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"{metric.replace('_', ' ').title()} by Parameter Combination"
    
    # Create heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=perf_matrix.values,
        x=perf_matrix.columns,
        y=perf_matrix.index,
        colorscale=colorscale,
        colorbar=dict(title=metric.replace('_', ' ').title())
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=perf_matrix.columns.name,
        yaxis_title=perf_matrix.index.name,
        width=figsize[0],
        height=figsize[1],
        template="plotly_white"
    )
    
    return fig


def save_figures(
    figures: Dict[str, go.Figure],
    output_dir: str,
    prefix: str = "",
    formats: List[str] = ["png", "html"]
) -> Dict[str, str]:
    """
    Save multiple figures to files.
    
    Args:
        figures: Dictionary of figures {name: figure}
        output_dir: Directory to save figures
        prefix: Prefix for filenames
        formats: List of formats to save (png, html, pdf, svg, jpeg)
        
    Returns:
        Dictionary mapping figure names to saved file paths
    """
    ensure_directory(output_dir)
    saved_paths = {}
    
    for name, fig in figures.items():
        for fmt in formats:
            if prefix:
                filename = f"{prefix}_{name}.{fmt}"
            else:
                filename = f"{name}.{fmt}"
                
            filepath = os.path.join(output_dir, filename)
            
            if fmt == 'html':
                fig.write_html(filepath)
            else:
                fig.write_image(filepath)
            
            saved_paths[f"{name}_{fmt}"] = filepath
    
    return saved_paths