"""
Statistics module for calculating and summarizing data.
"""

import numpy as np
import pandas as pd


def calculate_summary_stats(prices):
    """
    Calculate summary statistics for a price series.
    
    Parameters:
    -----------
    prices : np.ndarray or pd.Series
        Array of prices
        
    Returns:
    --------
    dict
        Dictionary with statistics:
        - mean, std, min, max, median
        - quantiles (25%, 75%)
    """
    prices = np.asarray(prices)
    
    stats = {
        'mean': np.mean(prices),
        'std': np.std(prices),
        'min': np.min(prices),
        'max': np.max(prices),
        'median': np.median(prices),
        'q25': np.percentile(prices, 25),
        'q75': np.percentile(prices, 75),
    }
    
    return stats


def calculate_returns_stats(prices):
    """
    Calculate statistics on returns from a price series.
    
    Parameters:
    -----------
    prices : np.ndarray or pd.Series
        Array of prices
        
    Returns:
    --------
    dict
        Dictionary with return statistics:
        - daily_mean, daily_std
        - total_return, annualized_return
        - sharpe_ratio (assuming 0% risk-free rate)
    """
    prices = np.asarray(prices)
    
    # Calculate daily log returns
    returns = np.diff(np.log(prices))
    
    daily_mean = np.mean(returns)
    daily_std = np.std(returns)
    
    # Total return from first to last price
    total_return = (prices[-1] - prices[0]) / prices[0]
    
    # Annualized metrics (252 trading days)
    annualized_return = daily_mean * 252
    annualized_volatility = daily_std * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    stats = {
        'daily_mean': daily_mean,
        'daily_std': daily_std,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
    }
    
    return stats


def compare_distributions(real_final_prices, simulated_final_prices):
    """
    Compare distributions of real vs simulated final prices.
    
    Parameters:
    -----------
    real_final_prices : float or list
        Final price(s) from real data
    simulated_final_prices : np.ndarray
        Final prices from all simulations
        
    Returns:
    --------
    dict
        Dictionary with comparison metrics
    """
    if isinstance(real_final_prices, (list, np.ndarray)):
        real_final = real_final_prices[-1] if len(real_final_prices) > 0 else 0
    else:
        real_final = real_final_prices
    
    sim_mean = np.mean(simulated_final_prices)
    sim_std = np.std(simulated_final_prices)
    
    # Calculate probability of final price being above real final price
    prob_above = np.mean(simulated_final_prices >= real_final)
    
    comparison = {
        'real_final_price': real_final,
        'simulated_mean_final_price': sim_mean,
        'simulated_std_final_price': sim_std,
        'probability_above_real': prob_above,
        'mean_difference': sim_mean - real_final,
        'mean_difference_pct': (sim_mean - real_final) / real_final * 100 if real_final > 0 else 0,
    }
    
    return comparison


def create_summary_dataframe(real_stats, simulated_stats):
    """
    Create a comparison DataFrame for display in Streamlit.
    
    Parameters:
    -----------
    real_stats : dict
        Statistics from real data
    simulated_stats : dict
        Statistics from simulated data
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    data = {
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Median', '25th %ile', '75th %ile'],
        'Real Data': [
            real_stats.get('mean', 0),
            real_stats.get('std', 0),
            real_stats.get('min', 0),
            real_stats.get('max', 0),
            real_stats.get('median', 0),
            real_stats.get('q25', 0),
            real_stats.get('q75', 0),
        ],
        'Simulated Data': [
            simulated_stats.get('mean', 0),
            simulated_stats.get('std', 0),
            simulated_stats.get('min', 0),
            simulated_stats.get('max', 0),
            simulated_stats.get('median', 0),
            simulated_stats.get('q25', 0),
            simulated_stats.get('q75', 0),
        ]
    }
    
    df = pd.DataFrame(data)
    return df
