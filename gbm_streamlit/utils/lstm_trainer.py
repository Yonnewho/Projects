"""
LSTM Neural Network trainer for stock price prediction.

This module provides functions to:
1. Prepare time series data for LSTM training
2. Build and train LSTM models
3. Make predictions on future prices
4. Evaluate model performance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')


def prepare_lstm_data(prices, lookback_window=60, test_split=0.2):
    """
    Prepare time series data for LSTM training.
    
    Args:
        prices: Array of stock prices
        lookback_window: Number of previous days to use for prediction
        test_split: Fraction of data to use for testing (0.2 = 20%)
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scaler, split_idx)
    """
    # Normalize prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_prices) - lookback_window):
        X.append(scaled_prices[i:i + lookback_window, 0])
        y.append(scaled_prices[i + lookback_window, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split into train and test
    split_idx = int(len(X) * (1 - test_split))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler, split_idx


def build_lstm_model(input_shape, layers=[64, 32], dropout_rate=0.2):
    """
    Build an LSTM neural network model.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        layers: List of LSTM units for each layer
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=layers[0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for units in layers[1:]:
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(LSTM(units=layers[-1] if len(layers) > 1 else layers[0]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    
    # Compile
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(prices, lookback_window=60, epochs=50, batch_size=32, 
                     layers=[64, 32], dropout_rate=0.2, verbose=0):
    """
    Train LSTM model on price data.
    
    Args:
        prices: Array of stock prices
        lookback_window: Number of previous days for prediction
        epochs: Number of training epochs
        batch_size: Batch size for training
        layers: List of LSTM units per layer
        dropout_rate: Dropout rate
        verbose: Verbosity level (0 or 1)
    
    Returns:
        Tuple of (model, scaler, history, metrics)
    """
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, split_idx = prepare_lstm_data(
        prices, lookback_window
    )
    
    # Build model
    model = build_lstm_model(
        input_shape=(X_train.shape[1], 1),
        layers=layers,
        dropout_rate=dropout_rate
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=verbose,
        shuffle=False
    )
    
    # Evaluate on test set
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return model, scaler, history, metrics


def predict_future_prices(model, last_sequence, scaler, num_days, lookback_window=60):
    """
    Predict future stock prices using trained LSTM model.
    
    Args:
        model: Trained Keras LSTM model
        last_sequence: Last price sequence from training data
        scaler: MinMaxScaler fitted on original data
        num_days: Number of days to predict into the future
        lookback_window: Number of lookback steps used in model
    
    Returns:
        Array of predicted prices (denormalized)
    """
    predictions = []
    current_sequence = last_sequence.reshape((lookback_window, 1))
    
    for _ in range(num_days):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape((1, lookback_window, 1))
        
        # Predict next value
        next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence with new prediction
        current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
    
    # Denormalize predictions
    predictions = np.array(predictions).reshape(-1, 1)
    denormalized_predictions = scaler.inverse_transform(predictions)
    
    return denormalized_predictions.flatten()


def get_last_lookback_sequence(prices, lookback_window=60):
    """
    Get the last lookback_window prices, normalized.
    
    Args:
        prices: Array of historical prices
        lookback_window: Number of days to include
    
    Returns:
        Tuple of (normalized_sequence, scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    
    last_sequence = scaled_prices[-lookback_window:, 0]
    
    return last_sequence, scaler


def evaluate_predictions(actual_prices, predicted_prices):
    """
    Evaluate prediction accuracy.
    
    Args:
        actual_prices: Array of actual prices
        predicted_prices: Array of predicted prices
    
    Returns:
        Dictionary of evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    r2 = r2_score(actual_prices, predicted_prices)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
