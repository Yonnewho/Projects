"""
Geometric Brownian Motion (GBM) simulator for stock price paths.
"""

import numpy as np
import pandas as pd


def simulate_gbm(S0, mu, sigma, T, N, M):
    """
    Simulate stock price paths using Geometric Brownian Motion.
    
    The GBM model is:
        dS = mu * S * dt + sigma * S * dW
    
    Parameters:
    -----------
    S0 : float
        Initial stock price (starting price)
    mu : float
        Drift coefficient (annualized expected return, e.g., 0.1 for 10%)
    sigma : float
        Volatility (annualized standard deviation, e.g., 0.2 for 20%)
    T : int
        Time horizon in days
    N : int
        Number of time steps
    M : int
        Number of simulated paths
        
    Returns:
    --------
    tuple
        (times, paths) where:
        - times: array of time points (length N+1)
        - paths: array of shape (M, N+1) containing simulated prices
    """
    # Create time grid
    dt = T / N
    times = np.linspace(0, T, N + 1)
    
    # Initialize price matrix
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    
    # Generate random normal variables
    # Use shape (M, N) for M paths with N steps each
    Z = np.random.standard_normal((M, N))
    
    # Simulate paths
    for t in range(1, N + 1):
        # Vectorized GBM formula:
        # S(t) = S(t-1) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    
    return times, paths


def get_path_statistics(paths):
    """
    Calculate statistics across simulated paths.
    
    Parameters:
    -----------
    paths : np.ndarray
        Array of shape (M, N+1) containing simulated prices
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'mean': mean path across all simulations
        - 'std': standard deviation of final prices
        - 'percentile_5': 5th percentile path
        - 'percentile_95': 95th percentile path
        - 'min': minimum final price
        - 'max': maximum final price
        - 'median': median final price
    """
    final_prices = paths[:, -1]
    
    stats = {
        'mean': np.mean(paths, axis=0),
        'std': np.std(final_prices),
        'percentile_5': np.percentile(paths, 5, axis=0),
        'percentile_95': np.percentile(paths, 95, axis=0),
        'min': np.min(final_prices),
        'max': np.max(final_prices),
        'median': np.median(final_prices),
    }
    
    return stats


def create_simulated_dataframe(times, paths, initial_date):
    """
    Convert simulated paths to a DataFrame format for easier export/analysis.
    
    Parameters:
    -----------
    times : np.ndarray
        Time array from GBM simulation
    paths : np.ndarray
        Paths array from GBM simulation (shape M x N+1)
    initial_date : pd.Timestamp
        Starting date for the simulation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'simulation_X' for each path
    """
    # Convert to business days from initial date
    date_range = pd.date_range(start=initial_date, periods=len(times), freq='B')
    
    # Create DataFrame with each path as a column
    df = pd.DataFrame(
        paths.T,  # Transpose to have time on rows
        columns=[f'simulation_{i}' for i in range(paths.shape[0])],
        index=date_range
    )
    
    return df


def export_simulations_to_csv(times, paths, initial_date, filepath):
    """
    Export simulated paths to CSV file.
    
    Parameters:
    -----------
    times : np.ndarray
        Time array from GBM simulation
    paths : np.ndarray
        Paths array from GBM simulation
    initial_date : pd.Timestamp
        Starting date for the simulation
    filepath : str
        Output file path
    """
    df = create_simulated_dataframe(times, paths, initial_date)
    df.to_csv(filepath)
    return filepath
