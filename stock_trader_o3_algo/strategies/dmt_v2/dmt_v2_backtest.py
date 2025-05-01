#!/usr/bin/env python3
"""
DMT v2 Backtest - Implementation of the transformer-based DMT backtest.

This module provides functionality to run backtests for the DMT v2 strategy,
train the transformer models, and evaluate performance.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from .dmt_v2_model import (
    Config, VolatilityEstimator, RegimeClassifier, 
    PredictionModel, StrategyLayer, Backtester, 
    loss_function, create_feature_matrix, EnsembleModel
)

from ..tri_shot.tri_shot_features import fetch_data_from_date


def prepare_data(prices: pd.DataFrame, window_size: int = 20, include_vix: bool = True) -> Tuple:
    """
    Prepare data for the DMT v2 model.
    
    Args:
        prices: DataFrame with price data for QQQ
        window_size: Lookback window for features
        include_vix: Whether to include VIX data if available
        
    Returns:
        X_tensor: Feature tensors
        y_tensor: Target tensors
        ret_tensor: Return tensors
        dates: Dates for the data points
        df: Processed DataFrame with all features
        vix_index: Index of VIX feature in X_tensor, -1 if not available
    """
    # Make a copy to avoid modifying the original DataFrame
    df = prices.copy()
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Momentum/Trend Indicators
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_10d'] = df['Close'].pct_change(10)
    df['ret_20d'] = df['Close'].pct_change(20)
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Volatility & Range Indicators
    df['vol_5d'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
    df['vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # ATR (14) - Enhanced calculation for dynamic stop-losses
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['atr14'] = df['true_range'].rolling(window=14).mean()
    df['tr_pct_close'] = df['true_range'] / df['Close']
    df['atr_pct'] = df['atr14'] / df['Close']  # ATR as percentage of price
    
    # Mean-reversion / Extremes
    sma20 = df['Close'].rolling(window=20).mean()
    sma50 = df['Close'].rolling(window=50).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Distance from moving averages as z-scores
    df['sma20_dist'] = (df['Close'] - sma20) / (df['Close'].rolling(window=20).std())
    df['sma50_dist'] = (df['Close'] - sma50) / (df['Close'].rolling(window=50).std())
    df['ema20_dist'] = (df['Close'] - ema20) / (df['Close'].rolling(window=20).std())
    
    # Bollinger Band z-score
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_zscore'] = (df['Close'] - sma20) / (2 * bb_std)
    
    # Volume Indicators
    df['vol_zscore'] = (df['Volume'] - df['Volume'].rolling(window=20).mean()) / df['Volume'].rolling(window=20).std()
    
    # Add VIX data if available (from the prices DataFrame if '^VIX' column exists)
    vix_index = -1  # Default to -1 if VIX not available
    if include_vix and '^VIX' in df.columns:
        df['vix'] = df['^VIX']
        df['vix_zscore'] = (df['vix'] - df['vix'].rolling(window=20).mean()) / df['vix'].rolling(window=20).std()
        df['vix_ratio'] = df['vix'] / df['vix'].rolling(window=10).mean()
        df['vix_ma_cross'] = df['vix'] - df['vix'].rolling(window=10).mean()
    elif include_vix:
        # Try to fetch VIX data if it wasn't included
        try:
            import yfinance as yf
            vix_data = yf.download('^VIX', start=df.index[0], end=df.index[-1])
            if not vix_data.empty:
                vix_data = vix_data.reindex(df.index, method='ffill')
                df['vix'] = vix_data['Close']
                df['vix_zscore'] = (df['vix'] - df['vix'].rolling(window=20).mean()) / df['vix'].rolling(window=20).std()
                df['vix_ratio'] = df['vix'] / df['vix'].rolling(window=10).mean()
                df['vix_ma_cross'] = df['vix'] - df['vix'].rolling(window=10).mean()
        except:
            # If VIX data can't be fetched, continue without it
            print("Could not fetch VIX data, continuing without it.")
    
    # Check for NaN values and drop them
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Standardize features
    feature_cols = ['returns', 'log_returns', 
                   'ret_5d', 'ret_10d', 'ret_20d',
                   'macd', 'macd_signal', 'macd_hist',
                   'rsi14', 'stoch_k', 'stoch_d',
                   'vol_5d', 'vol_20d', 'atr14', 'tr_pct_close', 'atr_pct',
                   'sma20_dist', 'sma50_dist', 'ema20_dist', 'bb_zscore',
                   'vol_zscore']
    
    # Add VIX features if available
    if 'vix' in df.columns:
        feature_cols.extend(['vix', 'vix_zscore', 'vix_ratio', 'vix_ma_cross'])
        vix_index = len(feature_cols) - 4  # Index of 'vix' in feature_cols
    
    # Create X, y DataFrames
    X = df[feature_cols].copy()
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X[col] = (X[col] - mean) / std
    
    y = np.sign(df['returns'].shift(-1))
    ret_future = df['returns'].shift(-1)
    
    # Convert to numpy arrays
    X_np = X.values
    y_np = y.values
    ret_np = ret_future.values
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    ret_tensor = torch.tensor(ret_np, dtype=torch.float32)
    
    return X_tensor, y_tensor, ret_tensor, df.index, df, vix_index


def create_sequence_data(X: torch.Tensor, seq_len: int = 10) -> torch.Tensor:
    """Create sequence data for transformer input.
    
    Args:
        X: Feature tensor (n_samples, n_features)
        seq_len: Sequence length for transformer
        
    Returns:
        Sequence data (n_samples-seq_len+1, seq_len, n_features)
    """
    n_samples, n_features = X.shape
    seq_data = []
    
    for i in range(n_samples - seq_len + 1):
        seq_data.append(X[i:i+seq_len])
    
    return torch.stack(seq_data)


def create_multi_timeframe_sequences(X: torch.Tensor, short_len: int = 5, med_len: int = 15, long_len: int = 60) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create multi-timeframe sequence data for transformer input.
    
    Args:
        X: Feature tensor (n_samples, n_features)
        short_len: Length of short-term sequence
        med_len: Length of medium-term sequence
        long_len: Length of long-term sequence
        
    Returns:
        Multi-timeframe sequence data (X_short, X_med, X_long)
    """
    n_samples, n_features = X.shape
    
    # Create short sequence
    X_short_seq = []
    for i in range(short_len, n_samples + 1):
        X_short_seq.append(X[i-short_len:i])
    X_short = torch.stack(X_short_seq)
    
    # Create medium sequence
    X_med_seq = []
    for i in range(med_len, n_samples + 1):
        X_med_seq.append(X[i-med_len:i])
    X_med = torch.stack(X_med_seq)
    
    # Create long sequence (using stride to reduce dimensionality)
    stride = 4
    effective_long_len = long_len // stride
    X_long_seq = []
    
    for i in range(effective_long_len * stride, n_samples + 1, stride):
        indices = [i - j * stride for j in range(effective_long_len, 0, -1)]
        X_long_seq.append(X[indices])
    
    if len(X_long_seq) > 0:
        X_long = torch.stack(X_long_seq)
    else:
        # Fallback if long sequence can't be created
        X_long = X_med
    
    # Ensure all sequences have the same batch dimension
    min_samples = min(X_short.size(0), X_med.size(0), X_long.size(0))
    X_short = X_short[-min_samples:]
    X_med = X_med[-min_samples:]
    X_long = X_long[-min_samples:]
    
    return X_short, X_med, X_long


def initialize_vol_estimator(returns: torch.Tensor, window_size: int = 20) -> Tuple[VolatilityEstimator, torch.Tensor]:
    """Initialize volatility estimator with historical volatility.
    
    Args:
        returns: Return tensor
        window_size: Window size for volatility calculation
        
    Returns:
        vol_estimator: Initialized volatility estimator
        init_sigma: Initial volatility estimate
    """
    # Create volatility estimator
    vol_estimator = VolatilityEstimator()
    
    # Calculate initial volatility using simple historical approach
    annual_factor = np.sqrt(252)
    rolling_std = returns.rolling(window=window_size).std()
    init_sigma = rolling_std.iloc[-1] * annual_factor
    
    # Convert to tensor
    init_sigma_tensor = torch.tensor(init_sigma, dtype=torch.float32)
    
    return vol_estimator, init_sigma_tensor


class EnsembleModel(nn.Module):
    """Ensemble of prediction models for improved robustness."""
    
    def __init__(self, feature_dim, n_models=3, hidden_dims=[96, 128, 64], seq_len=15):
        """Initialize the ensemble model.
        
        Args:
            feature_dim: Input feature dimension
            n_models: Number of models in the ensemble
            hidden_dims: List of hidden dimensions for each model
            seq_len: Sequence length for transformers
        """
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList()
        
        # Create diverse models with different architectures
        for i in range(n_models):
            # Vary architecture parameters slightly for diversity
            n_heads = 4 + i * 2  # 4, 6, 8 heads
            n_layers = 4 + i * 2  # 4, 6, 8 layers
            hidden_dim = hidden_dims[i % len(hidden_dims)]
            dropout = 0.1 + (i * 0.05)  # 0.1, 0.15, 0.2 dropout
            
            model = PredictionModel(
                in_dim=feature_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                seq_len=seq_len,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout
            )
            self.models.append(model)
    
    def forward(self, X_short, X_med, X_long):
        """Forward pass through the ensemble.
        
        Args:
            X_short: Short-term sequence data
            X_med: Medium-term sequence data
            X_long: Long-term sequence data
            
        Returns:
            mean_pred: Mean prediction across models
            lower_bound: Lower confidence bound (10th percentile)
            upper_bound: Upper confidence bound (90th percentile)
            uncertainty: Prediction uncertainty (std dev)
        """
        all_preds = []
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                pred, _, _ = model(X_short, X_med, X_long)
                all_preds.append(pred)
        
        # Stack predictions
        pred_stack = torch.stack(all_preds, dim=0)
        
        # Calculate mean prediction
        mean_pred = pred_stack.mean(dim=0)
        
        # Calculate uncertainty (standard deviation)
        uncertainty = pred_stack.std(dim=0)
        
        # Calculate confidence bounds (10th and 90th percentiles)
        pred_sorted, _ = torch.sort(pred_stack, dim=0)
        lower_idx = max(0, int(0.1 * self.n_models) - 1)
        upper_idx = min(self.n_models - 1, int(0.9 * self.n_models))
        
        # Handle edge cases
        if self.n_models <= 3:
            lower_bound = pred_sorted[0]
            upper_bound = pred_sorted[-1]
        else:
            lower_bound = pred_sorted[lower_idx]
            upper_bound = pred_sorted[upper_idx]
        
        return mean_pred, lower_bound, upper_bound, uncertainty


def train_model(model, train_loader, optimizer, epochs, device, scheduler=None):
    """
    Train a PyTorch model with the given data loaders
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): PyTorch optimizer
        epochs (int): Number of training epochs
        device (torch.device): Device to train on
        scheduler (lr_scheduler, optional): Learning rate scheduler
        
    Returns:
        list: Training losses
    """
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            p_t, sigma_t, regime_logits = model(x_batch)
            
            # Calculate loss
            loss = model.loss_fn(p_t, sigma_t, y_batch)
            
            # Add L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2) ** 2
            loss += 1e-5 * l2_reg
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def run_dmt_v2_backtest(data, initial_capital=10000.0, n_epochs=100, target_annual_vol=0.35,
                      max_position_size=2.0, neutral_zone=0.025, plot=True, use_ensemble=True, 
                      use_dynamic_stops=True, max_drawdown_threshold=0.2, learning_rate=0.015):
    """
    Run a backtest of the DMT v2 strategy with enhanced features.
    
    Args:
        data (pandas.DataFrame): OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        initial_capital (float): Initial capital for the backtest
        n_epochs (int): Number of training epochs
        target_annual_vol (float): Target annualized volatility
        max_position_size (float): Maximum position size as a multiple of capital
        neutral_zone (float): Neutral zone size for position calculation
        plot (bool): Whether to plot the backtest results
        use_ensemble (bool): Whether to use ensemble modeling
        use_dynamic_stops (bool): Whether to use dynamic stop-loss levels
        max_drawdown_threshold (float): Maximum drawdown allowed before reducing exposure
        learning_rate (float): Learning rate for training
        
    Returns:
        tuple: (results_df, metrics_dict)
    """
    # Create a copy of the data to avoid modifying the original
    data = data.copy()
    
    # Check if data is valid and has all required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain all of {required_cols}")
    
    # Save original column case
    rename_map = {}
    for col in data.columns:
        if col.lower() in [c.lower() for c in required_cols]:
            orig_col = next(c for c in required_cols if c.lower() == col.lower())
            rename_map[col] = orig_col
    
    # Standardize column names if needed
    if rename_map:
        data = data.rename(columns=rename_map)
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate additional features that will enhance model performance
    print("Preparing enhanced feature set...")
    
    # Price-derived features
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Volatility metrics
    data['atr'] = calculate_atr(data)
    data['realized_vol'] = data['returns'].rolling(21).std() * np.sqrt(252)
    
    # Momentum indicators
    for period in [5, 10, 21, 63]:
        data[f'mom_{period}'] = data['Close'].pct_change(period)
    
    # Mean reversion indicators
    for period in [5, 10, 21]:
        data[f'mean_rev_{period}'] = (data['Close'] - data['Close'].rolling(period).mean()) / data['Close'].rolling(period).std()
    
    # Try to get VIX data or estimate market volatility if unavailable
    try:
        start_date = data.index[0] - pd.Timedelta(days=10)
        end_date = data.index[-1]
        
        # If download fails, we'll use a synthetic VIX based on realized volatility
        vix_data = None
        
        # Calculate a synthetic VIX based on rolling volatility
        data['synthetic_vix'] = data['realized_vol'] * 100
        
        # Impute any remaining missing values with forward/backward fill
        data['synthetic_vix'] = data['synthetic_vix'].fillna(method='ffill').fillna(method='bfill')
        data['vix_feature'] = data['synthetic_vix']
        
        print("Using synthetic VIX derived from realized volatility.")
    except Exception as e:
        print(f"Error fetching VIX data: {e}. Using synthetic VIX.")
        data['vix_feature'] = data['realized_vol'] * 100
    
    # Calculate ratios and additional technical indicators to enrich the feature set
    data['high_low_ratio'] = data['High'] / data['Low']
    data['close_open_ratio'] = data['Close'] / data['Open']
    
    # Gap indicators
    data['overnight_gap'] = (data['Open'] / data['Close'].shift(1)) - 1
    
    # Additional volume indicators
    data['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(10).mean()
    
    # Forward returns for train/test target
    data['target'] = data['returns'].shift(-1)
    
    # Drop rows with NaN due to feature calculations
    data = data.dropna()
    
    # Select features for modeling
    feature_cols = [
        'returns', 'log_returns', 
        'atr', 'realized_vol',
        'mom_5', 'mom_10', 'mom_21', 'mom_63',
        'mean_rev_5', 'mean_rev_10', 'mean_rev_21',
        'high_low_ratio', 'close_open_ratio', 'overnight_gap',
        'volume_ma_ratio', 'vix_feature'
    ]
    
    # Create time-series sequences for the model
    # We'll use a sequence length of 15 days (3 weeks) to capture patterns
    seq_len = 15
    X, y, _ = prepare_sequences(data, feature_cols, seq_len=seq_len)
    
    # Split data for training and validation
    # Use 80% of data for training, 20% for validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create PyTorch datasets and dataloaders
    batch_size = 64
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration for the models
    config = Config(
        eps=1e-8,
        tau_max=target_annual_vol,
        max_pos=max_position_size,
        neutral_zone=neutral_zone,
        lr=learning_rate,
        seq_len=seq_len,
        max_drawdown_threshold=max_drawdown_threshold,
        use_ensemble=use_ensemble,
        use_dynamic_stops=use_dynamic_stops
    )
    
    # Create models
    in_dim = X_train.shape[2]  # Number of features
    
    if use_ensemble:
        print("Using enhanced ensemble of models...")
        model = EnsembleModel(in_dim=in_dim, device=device)
    else:
        print("Using single model...")
        model = PredictionModel(
            in_dim=in_dim,
            hidden_dim=128,
            transformer_dim=96,
            n_heads=8,
            n_layers=6,
            dropout=0.1
        ).to(device)
    
    # Create optimizer with weight decay for regularization
    if use_ensemble:
        # Create a parameter list from all models
        all_params = []
        for m in model.models:
            all_params.extend(list(m.parameters()))
        optimizer = torch.optim.AdamW(all_params, lr=config.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.lr,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs
    )
    
    # Train models
    print(f"Training model for {n_epochs} epochs...")
    if use_ensemble:
        losses = model.train(train_loader, optimizer, n_epochs, scheduler)
    else:
        losses = train_model(model, train_loader, optimizer, n_epochs, device, scheduler)
    
    # Run backtest
    print("Running backtest...")
    results_df, metrics = backtest_strategy(
        data.iloc[seq_len:],  # Skip the first seq_len rows used for sequences
        model,
        feature_cols,
        seq_len,
        initial_capital,
        config,
        device
    )
    
    # Add baseline equity columns (buy and hold)
    initial_price = data['Close'].iloc[seq_len]
    final_price = data['Close'].iloc[-1]
    daily_returns = data['Close'].pct_change()
    
    # Calculate buy and hold performance
    buy_hold_equity = []
    baseline_capital = initial_capital
    
    for i in range(len(results_df)):
        idx = results_df.index[i]
        if i > 0:
            # Update based on daily return
            r = data.loc[idx, 'returns']
            baseline_capital *= (1 + r)
        buy_hold_equity.append(baseline_capital)
    
    results_df['baseline_equity'] = buy_hold_equity
    results_df['buy_hold_equity'] = buy_hold_equity
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Avg Profit/Loss Ratio: {metrics['avg_profit_loss_ratio']:.2f}")
    
    # Regime analysis
    if 'regime_probs' in metrics:
        print("\nAverage Regime Probabilities:")
        for i, prob in enumerate(metrics['regime_probs']):
            regime_name = ['Bull', 'Neutral', 'Bear'][i]
            print(f"{regime_name} Market: {prob:.2%}")
    
    # Generate plot
    if plot:
        plot_backtest_results(results_df, metrics)
    
    # Save results to CSV
    os.makedirs('tri_shot_data', exist_ok=True)
    results_file = os.path.join('tri_shot_data', 'dmt_v2_backtest_results.csv')
    results_df.to_csv(results_file)
    print(f"Results saved to {results_file}")
    
    return results_df, metrics

def prepare_sequences(data, feature_cols, seq_len=15, target_col='target'):
    """
    Create sequence data for time-series modeling
    
    Args:
        data (pandas.DataFrame): DataFrame with features and target
        feature_cols (list): Feature column names
        seq_len (int): Sequence length
        target_col (str): Target column name
        
    Returns:
        tuple: (X, y, df) - sequences, targets, and original data
    """
    # Extract features and target
    features = data[feature_cols].values
    target = data[target_col].values
    
    # Create sequences
    X = []
    y = []
    
    # For each possible sequence of length seq_len
    for i in range(len(data) - seq_len):
        # Get sequence of features
        seq = features[i:i+seq_len]
        # Get target (next day's return)
        target_val = target[i+seq_len-1]
        
        X.append(seq)
        y.append(target_val)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    return X, y, data.iloc[seq_len-1:-1]

def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def backtest_strategy(data, model, feature_cols, seq_len, initial_capital, config, device):
    """
    Backtest the DMT v2 strategy on historical data
    
    Args:
        data (pandas.DataFrame): OHLCV data with features
        model: Prediction model (single or ensemble)
        feature_cols (list): Feature column names
        seq_len (int): Sequence length for model input
        initial_capital (float): Initial capital
        config (Config): Strategy configuration
        device (torch.device): PyTorch device
        
    Returns:
        tuple: (results_df, metrics_dict)
    """
    # Initialize results
    dates = data.index
    equity_curve = [initial_capital]
    positions = [0.0]
    daily_returns = [0.0]
    trades = []
    current_position = 0.0
    
    # Initialize regime tracking
    regime_counts = np.zeros(3)  # [bull, neutral, bear]
    
    # Set model to evaluation mode
    if hasattr(model, 'models'):  # Ensemble model
        for m in model.models:
            m.eval()
    else:
        model.eval()
    
    # Track stop-loss levels
    stop_loss = None
    
    # Loop through data day by day
    for i in range(seq_len, len(data)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        # Get feature sequence up to current day (excluding current day's return)
        feature_sequence = data[feature_cols].iloc[i-seq_len:i].values
        
        # Convert to PyTorch tensor
        x = torch.from_numpy(feature_sequence).float().unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            if hasattr(model, 'predict'):  # Ensemble model
                p_t, sigma_t, regime_logits = model.predict(x)
            else:
                p_t, sigma_t, regime_logits = model(x)
        
        # Get regime probabilities
        regime_probs = F.softmax(regime_logits, dim=1).squeeze().cpu().numpy()
        regime_counts += regime_probs
        
        # Extract ATR for stop-loss calculation
        current_atr = data['atr'].iloc[i]
        
        # Create equity tensor for drawdown calculation
        equity_tensor = torch.tensor([equity_curve], dtype=torch.float32).to(device)
        
        # Extract uncertainty (for ensemble)
        uncertainty = None
        if hasattr(model, 'predict') and config.use_ensemble:
            # Use the variance between model predictions as uncertainty
            all_preds = []
            for m in model.models:
                with torch.no_grad():
                    pred, _, _ = m(x)
                    all_preds.append(pred.item())
            
            # Calculate standard deviation of predictions
            if len(all_preds) > 1:
                uncertainty = torch.tensor([[np.std(all_preds)]], dtype=torch.float32).to(device)
        
        # Create strategy layer if not using an internal one
        if not hasattr(model, 'strategy'):
            strategy = StrategyLayer(config).to(device)
        
        # Calculate new position
        if hasattr(model, 'strategy'):
            # Use model's internal strategy layer
            new_position = model.strategy(
                p_t, sigma_t, regime_logits, 
                equity_curve=equity_tensor,
                atr=torch.tensor([[current_atr]], dtype=torch.float32).to(device),
                uncertainty=uncertainty
            ).item()
        else:
            # Use our strategy layer
            new_position = strategy(
                p_t, sigma_t, regime_logits, 
                equity_curve=equity_tensor,
                atr=torch.tensor([[current_atr]], dtype=torch.float32).to(device),
                uncertainty=uncertainty
            ).item()
        
        # Apply dynamic stop-loss if enabled
        if config.use_dynamic_stops:
            current_price = data['Close'].iloc[i]
            
            # Set stop-loss when entering a position
            if abs(new_position) > 0.1 and abs(current_position) < 0.1:
                if new_position > 0:  # Long position
                    stop_loss = current_price - (current_atr * config.stop_loss_atr_multiple)
                else:  # Short position
                    stop_loss = current_price + (current_atr * config.stop_loss_atr_multiple)
            
            # Check if stop-loss has been hit
            elif stop_loss is not None:
                if (new_position > 0 and data['Low'].iloc[i] <= stop_loss) or \
                   (new_position < 0 and data['High'].iloc[i] >= stop_loss):
                    # Stop-loss hit - close position
                    new_position = 0
                    trades.append({
                        'exit_date': current_date,
                        'reason': 'stop_loss',
                        'price': stop_loss
                    })
                    stop_loss = None
            
            # Reset stop-loss if position closed or flipped
            if abs(new_position) < 0.1 or (new_position * current_position < 0):
                stop_loss = None
        
        # Record when a trade is opened or closed
        if abs(new_position - current_position) > 0.1:
            if abs(current_position) < 0.1 and abs(new_position) > 0.1:
                # Opening a new position
                trades.append({
                    'entry_date': current_date,
                    'direction': 'long' if new_position > 0 else 'short',
                    'size': abs(new_position),
                    'price': data['Close'].iloc[i]
                })
            elif abs(current_position) > 0.1 and abs(new_position) < 0.1:
                # Closing a position
                trades.append({
                    'exit_date': current_date,
                    'reason': 'signal',
                    'price': data['Close'].iloc[i]
                })
        
        # Calculate daily return based on position
        daily_return = current_position * data['returns'].iloc[i]
        
        # Update equity
        new_equity = equity_curve[-1] * (1 + daily_return)
        
        # Record results
        equity_curve.append(new_equity)
        positions.append(new_position)
        daily_returns.append(daily_return)
        
        # Update position for next day
        current_position = new_position
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'dmt_v2_equity': equity_curve[1:],  # Skip initial equity
        'position': positions[1:],          # Skip initial position
        'daily_return': daily_returns[1:]   # Skip initial return
    }, index=dates[seq_len:])
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results_df, initial_capital)
    
    # Add regime probabilities
    if len(regime_counts) > 0:
        metrics['regime_probs'] = regime_counts / len(data[seq_len:])
    
    # Add trade statistics
    if trades:
        metrics['trades'] = trades
        metrics['trade_count'] = len([t for t in trades if 'entry_date' in t])
    
    return results_df, metrics

def calculate_performance_metrics(results_df, initial_capital):
    """
    Calculate performance metrics from backtest results
    
    Args:
        results_df (pandas.DataFrame): Backtest results
        initial_capital (float): Initial capital
        
    Returns:
        dict: Performance metrics
    """
    equity = results_df['dmt_v2_equity'].values
    daily_returns = results_df['daily_return'].values
    
    # Calculate returns
    total_return = (equity[-1] / initial_capital) - 1
    
    # Calculate days and annualized return
    days = len(results_df)
    years = days / 252  # Assuming 252 trading days per year
    cagr = (equity[-1] / initial_capital) ** (1 / years) - 1
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_drawdown = np.min(drawdown)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_daily = 0.02 / 252
    excess_returns = daily_returns - risk_free_daily
    sharpe_ratio = (np.mean(excess_returns) / np.std(daily_returns)) * np.sqrt(252)
    
    # Calculate Sortino ratio (downside risk only)
    downside_returns = np.where(daily_returns < 0, daily_returns, 0)
    sortino_ratio = (np.mean(excess_returns) / np.std(downside_returns)) * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Calculate win rate and profit/loss ratio
    wins = np.sum(daily_returns > 0)
    losses = np.sum(daily_returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    avg_win = np.mean(daily_returns[daily_returns > 0]) if np.any(daily_returns > 0) else 0
    avg_loss = abs(np.mean(daily_returns[daily_returns < 0])) if np.any(daily_returns < 0) else 1
    avg_profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Calculate maximum consecutive wins/losses
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    
    for ret in daily_returns:
        if ret > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        elif ret < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_loss_streak = min(max_loss_streak, current_streak)
        else:
            current_streak = 0
    
    max_win_streak = max_win_streak
    max_loss_streak = abs(max_loss_streak)
    
    return {
        'initial_capital': initial_capital,
        'final_equity': equity[-1],
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_profit_loss_ratio': avg_profit_loss_ratio,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak
    }

def plot_backtest_results(results_df, metrics):
    """
    Plot backtest results with enhanced visualization
    
    Args:
        results_df (pandas.DataFrame): Backtest results
        metrics (dict): Performance metrics
    """
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot equity curves
    ax1.plot(results_df.index, results_df['dmt_v2_equity'], label='DMT v2 Strategy')
    ax1.plot(results_df.index, results_df['buy_hold_equity'], label='Buy & Hold', linestyle='--')
    
    # Add metrics to the plot
    ax1.set_title('DMT v2 Backtest Results', fontsize=14)
    ax1.text(0.01, 0.95, f"Total Return: {metrics['total_return']:.2%}", transform=ax1.transAxes)
    ax1.text(0.01, 0.90, f"CAGR: {metrics['cagr']:.2%}", transform=ax1.transAxes)
    ax1.text(0.01, 0.85, f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}", transform=ax1.transAxes)
    ax1.text(0.01, 0.80, f"Max Drawdown: {metrics['max_drawdown']:.2%}", transform=ax1.transAxes)
    ax1.text(0.01, 0.75, f"Win Rate: {metrics['win_rate']:.2%}", transform=ax1.transAxes)
    
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Plot positions
    ax2.plot(results_df.index, results_df['position'], color='purple')
    ax2.fill_between(results_df.index, results_df['position'], 0, where=results_df['position']>0, color='green', alpha=0.3)
    ax2.fill_between(results_df.index, results_df['position'], 0, where=results_df['position']<0, color='red', alpha=0.3)
    ax2.set_ylabel('Position Size', fontsize=12)
    ax2.grid(True)
    
    # Plot drawdown
    equity = results_df['dmt_v2_equity'].values
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    
    ax3.fill_between(results_df.index, drawdown, 0, color='red', alpha=0.3)
    ax3.set_ylabel('Drawdown', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('tri_shot_data', exist_ok=True)
    plt.savefig(os.path.join('tri_shot_data', 'dmt_v2_backtest.png'))
    plt.close()
