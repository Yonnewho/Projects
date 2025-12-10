"""
Visualization module for plotting stock data and simulations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_real_vs_simulated(real_data, times, simulated_paths, stats=None):
    """
    Create two separate interactive Plotly charts: one for real data, one for simulations.
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with 'date' and 'close' columns
    times : np.ndarray
        Time array from GBM simulation
    simulated_paths : np.ndarray
        Array of shape (M, N+1) with simulated prices
    stats : dict, optional
        Statistics dict from get_path_statistics()
        
    Returns:
    --------
    tuple of (fig_real, fig_sim)
        Two separate interactive charts
    """
    from plotly.subplots import make_subplots
    
    # CHART 1: Real Data Only
    fig_real = go.Figure()
    
    if real_data is not None and len(real_data) > 0:
        fig_real.add_trace(go.Scatter(
            x=real_data['date'],
            y=real_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='<b>Historical Data</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add min/max price bands
        min_price = real_data['close'].min()
        max_price = real_data['close'].max()
        fig_real.add_hline(y=min_price, line_dash='dot', line_color='red', line_width=1,
                          annotation_text=f'Low: ${min_price:.2f}', annotation_position='right')
        fig_real.add_hline(y=max_price, line_dash='dot', line_color='green', line_width=1,
                          annotation_text=f'High: ${max_price:.2f}', annotation_position='right')
    
    fig_real.update_layout(
        title='📊 Real Historical Stock Data',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        showlegend=True,
        font=dict(size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # CHART 2: GBM Simulations Only
    fig_sim = go.Figure()
    
    # Convert times to dates (relative to last real date if available)
    if real_data is not None and len(real_data) > 0:
        last_date = real_data['date'].iloc[-1]
        sim_dates = pd.date_range(start=last_date, periods=len(times)+1, freq='B')[1:]
    else:
        sim_dates = np.arange(len(times))
    
    # Add confidence intervals FIRST (so they appear behind)
    if stats is not None:
        # Add the confidence interval as a shaded region
        fig_sim.add_trace(go.Scatter(
            x=sim_dates,
            y=stats['percentile_95'],
            mode='lines',
            name='95th Percentile',
            line=dict(color='rgba(255, 0, 0, 0)', width=0),
            showlegend=True,
            hovertemplate='<b>95th Percentile</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig_sim.add_trace(go.Scatter(
            x=sim_dates,
            y=stats['percentile_5'],
            mode='lines',
            name='5th Percentile',
            line=dict(color='rgba(255, 0, 0, 0)', width=0),
            fill='tonexty',
            fillcolor='rgba(200, 100, 100, 0.3)',
            hovertemplate='<b>5th Percentile</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add simulated paths (with better visibility)
    num_paths = min(simulated_paths.shape[0], 200)
    
    for i in range(num_paths):
        opacity = 0.12 + (0.08 * (i % 3))  # Vary opacity slightly
        fig_sim.add_trace(go.Scatter(
            x=sim_dates,
            y=simulated_paths[i, :],
            mode='lines',
            name=f'Path {i+1}',
            line=dict(color=f'rgba(100, 150, 200, {opacity})', width=0.8),
            hovertemplate='<b>Path {}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>'.format(i+1),
            showlegend=(i == 0),  # Only show legend for first path
            legendgroup='paths'
        ))
    
    # Add mean path with bold styling
    if stats is not None:
        fig_sim.add_trace(go.Scatter(
            x=sim_dates,
            y=stats['mean'],
            mode='lines',
            name='Mean Path',
            line=dict(color='#2E7D32', width=3),
            hovertemplate='<b>Mean Path</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    fig_sim.update_layout(
        title='🎲 GBM Simulations (Future Price Paths)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        font=dict(size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig_real, fig_sim


def plot_distribution(final_prices):
    """
    Create a histogram of final simulated prices.
    
    Parameters:
    -----------
    final_prices : np.ndarray
        Array of final prices from all simulations
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram chart
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=final_prices,
            nbinsx=60,
            name='Final Price Distribution',
            marker=dict(color='rgba(100, 150, 255, 0.8)'),
            hovertemplate='Price Range: $%{x:.2f}<br>Frequency: %{y}<extra></extra>'
        )
    ])
    
    # Add vertical lines for mean and percentiles
    mean = np.mean(final_prices)
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    
    fig.add_vline(x=mean, line_dash='solid', line_color='#2E7D32', line_width=3,
                  annotation_text=f'Mean: ${mean:.2f}', 
                  annotation_position='top right',
                  annotation_bgcolor='#2E7D32',
                  annotation_font_color='white')
    
    fig.add_vline(x=p5, line_dash='dash', line_color='#C62828', line_width=2,
                  annotation_text=f'5th %ile: ${p5:.2f}', 
                  annotation_position='top left',
                  annotation_bgcolor='#C62828',
                  annotation_font_color='white')
    
    fig.add_vline(x=p95, line_dash='dash', line_color='#C62828', line_width=2,
                  annotation_text=f'95th %ile: ${p95:.2f}', 
                  annotation_position='top right',
                  annotation_bgcolor='#C62828',
                  annotation_font_color='white')
    
    fig.update_layout(
        title='📊 Distribution of Final Prices (All Simulations)',
        xaxis_title='Price ($)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=550,
        showlegend=False,
        font=dict(size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def plot_real_returns_distribution(real_data):
    """
    Create a histogram of real daily returns.
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with 'close' column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Histogram chart
    """
    returns = np.diff(np.log(real_data['close'].values)) * 100  # Convert to percentage
    
    fig = go.Figure(data=[
        go.Histogram(
            x=returns,
            nbinsx=40,
            name='Daily Returns',
            marker=dict(color='rgba(0, 100, 200, 0.8)'),
            hovertemplate='Return Range (%): %{x:.3f}%<br>Frequency: %{y}<extra></extra>'
        )
    ])
    
    # Add mean line
    mean_return = np.mean(returns)
    fig.add_vline(x=mean_return, line_dash='solid', line_color='#2E7D32', line_width=3,
                  annotation_text=f'Mean: {mean_return:.3f}%',
                  annotation_position='top right',
                  annotation_bgcolor='#2E7D32',
                  annotation_font_color='white')
    
    fig.update_layout(
        title='📉 Distribution of Real Daily Returns',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=550,
        showlegend=False,
        font=dict(size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig
