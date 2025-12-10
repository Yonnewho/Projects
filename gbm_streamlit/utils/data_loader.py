"""
Data loader module for handling stock data import and validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_csv_data(uploaded_file):
    """
    Load stock data from an uploaded CSV file.
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        The uploaded CSV file from Streamlit file uploader
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns 'Date' and 'Close', or None if invalid
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        
        required_cols = ['date', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, "CSV must contain 'Date' and 'Close' columns."
        
        # Remove rows where all values are identical to column names (header duplication)
        df = df[~df.astype(str).apply(lambda x: (x == x.index).any(), axis=1)]
        
        # Parse date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'])
        
        # Ensure close prices are numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 2:
            return None, "Need at least 2 data points."
        
        return df[['date', 'close']], None
        
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def calculate_returns(prices):
    """
    Calculate daily log returns from price series.
    
    Parameters:
    -----------
    prices : np.ndarray or pd.Series
        Array of closing prices
        
    Returns:
    --------
    np.ndarray
        Array of daily log returns
    """
    prices = np.asarray(prices)
    returns = np.diff(np.log(prices))
    return returns


def estimate_parameters(prices):
    """
    Estimate drift and volatility from historical prices.
    
    Parameters:
    -----------
    prices : np.ndarray or pd.Series
        Array of closing prices
        
    Returns:
    --------
    tuple
        (drift, volatility) estimated from daily returns
    """
    returns = calculate_returns(prices)
    
    # Annualize parameters (252 trading days per year)
    drift = np.mean(returns) * 252
    volatility = np.std(returns) * np.sqrt(252)
    
    return drift, volatility


def load_builtin_data(filename):
    """
    Load built-in stock data from the data folder.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file in the data folder
        
    Returns:
    --------
    pd.DataFrame or tuple
        DataFrame with 'date' and 'close' columns, or (None, error_message)
    """
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        return None, f"File not found: {filename}"
    
    try:
        with open(filepath, 'r') as f:
            return load_csv_data(f)
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def validate_data(df):
    """
    Validate stock data integrity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' and 'close' columns
        
    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    if df is None or len(df) < 2:
        return False, "Need at least 2 data points."
    
    if df['close'].min() <= 0:
        return False, "Stock prices must be positive."
    
    if df['close'].isna().any() or df['date'].isna().any():
        return False, "Data contains NaN values."
    
    return True, None
