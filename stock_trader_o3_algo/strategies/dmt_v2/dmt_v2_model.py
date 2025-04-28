#!/usr/bin/env python3
"""
DMT v2 Model - Transformer-Enhanced, Regime-Adaptive Model Components.

This module implements the core neural network components of the DMT v2 strategy:
- Configuration class for hyperparameters
- VolatilityEstimator - EGARCH-like neural volatility forecaster
- RegimeClassifier - identifies market regimes
- PredictionModel - transformer-based price/return prediction
- StrategyLayer - converts predictions to positions with adaptive controls
"""

import math
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


@dataclass
class Config:
    """Configuration parameters for DMT v2 strategy."""
    # Model architecture
    d_model: int = 64            # Transformer embedding dimension
    nhead: int = 4               # Number of attention heads
    nlayers: int = 4             # Number of transformer layers
    dropout: float = 0.1         # Dropout rate
    n_regimes: int = 3           # Number of market regimes to identify

    # Strategy hyperparameters
    tau_max: float = 0.30        # Maximum annualized vol target
    max_pos: float = 1.0         # Hard leverage cap
    k0: float = 50.0             # Initial sigmoid steepness

    # Loss weights
    lambda_sharpe: float = 0.10  # Weight for Sharpe ratio in objective
    lambda_draw: float = 0.05    # Weight for drawdown penalty
    lambda_turn: float = 0.002   # Weight for turnover penalty

    # Transaction costs & numerical stability
    trans_cost: float = 2.5e-4   # Round-trip transaction cost
    eps: float = 1e-8            # Small epsilon for numerical stability


class VolatilityEstimator(nn.Module):
    """Neural EGARCH(1,1)-style volatility forecaster.
    
    Implements a differentiable volatility estimator inspired by EGARCH models:
    log(σ²_t) = ω + α·|ε_{t-1}| + γ·ε_{t-1} + β·log(σ²_{t-1})
    
    Args:
        hidden (int): Size of hidden layer in optional nonlinear component
    """
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        # Initialize with reasonable EGARCH parameters
        self.omega = nn.Parameter(torch.tensor(-9.0))  # bias so σ≈20% initially
        self.alpha = nn.Parameter(torch.tensor(0.10))
        self.beta = nn.Parameter(torch.tensor(0.85))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, ret: torch.Tensor, prev_sigma: torch.Tensor) -> torch.Tensor:
        """Estimate next-step volatility.
        
        Args:
            ret: Return tensor of shape (batch_size,)
            prev_sigma: Previous volatility estimate of shape (batch_size,)
            
        Returns:
            New volatility estimate of shape (batch_size,)
        """
        # Standardized return (like innovation term in EGARCH)
        eps = ret / (prev_sigma + 1e-8)
        
        # EGARCH(1,1) update
        log_sigma2 = self.omega + self.alpha * eps.abs() + self.gamma * eps + self.beta * prev_sigma.log()
        
        # Convert back to volatility (standard deviation)
        sigma = log_sigma2.mul(0.5).exp()  # √exp(log σ²)
        
        return sigma.clamp(min=1e-4)  # Ensure positive volatility


class RegimeClassifier(nn.Module):
    """Classify market regime (e.g., 0: low-vol, 1: high-vol trend, 2: choppy).
    
    Args:
        in_dim (int): Input feature dimension
        n_regimes (int): Number of distinct market regimes to identify
    """
    def __init__(self, in_dim: int, n_regimes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, n_regimes)
        )

    def forward(self, x):
        """Classify current market regime.
        
        Args:
            x: Features tensor of shape (batch_size, in_dim)
            
        Returns:
            Regime logits of shape (batch_size, n_regimes)
        """
        return self.net(x)  # Outputs logits (not softmaxed)


class PredictionModel(nn.Module):
    """Transformer encoder predicting p_t, q_lo_t, q_hi_t.
    
    Args:
        in_dim (int): Input feature dimension
        config (Config): Model configuration
    """
    def __init__(self, in_dim: int, config: Config):
        super().__init__()
        self.config = config
        
        # Project input features to transformer dimension
        self.input_proj = nn.Linear(in_dim, config.d_model)
        
        # Create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.nlayers)
        
        # Output heads
        self.class_head = nn.Linear(config.d_model, 1)    # Probability head
        self.quant_head = nn.Linear(config.d_model, 2)    # 10% & 90% quantile head

    def forward(self, x, mask=None):
        """Forward pass of the prediction model.
        
        Args:
            x: Input features tensor of shape (batch_size, seq_len, in_dim)
            mask: Optional attention mask
            
        Returns:
            p: Probability predictions (batch_size,)
            q_lo: 10th percentile return predictions (batch_size,)
            q_hi: 90th percentile return predictions (batch_size,)
        """
        # Project input to transformer dimension
        z = self.input_proj(x)
        
        # Apply transformer
        h = self.transformer(z, mask=mask)
        
        # Extract last timestep representation
        h_last = h[:, -1]
        
        # Get outputs from the different heads
        p = torch.sigmoid(self.class_head(h_last)).squeeze(-1)  # (batch_size,)
        q = self.quant_head(h_last)  # (batch_size, 2)
        q_lo = q[:, 0]  # 10th percentile
        q_hi = q[:, 1]  # 90th percentile
        
        return p, q_lo, q_hi


class StrategyLayer(nn.Module):
    """Convert model outputs & vol estimate to differentiable position_t.
    
    Implements adaptive neutral zone and target volatility based on regime classification.
    
    Args:
        n_regimes (int): Number of market regimes
        config (Config): Strategy configuration
    """
    def __init__(self, n_regimes: int = 3, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config
        
        # Learnable long/short thresholds (initialized near 0.55 / 0.45)
        self.theta_L = nn.Parameter(torch.tensor(0.55))
        self.theta_S = nn.Parameter(torch.tensor(0.45))
        self.log_k = nn.Parameter(torch.log(torch.tensor(config.k0)))
        
        # Adaptive neutral-zone and target vol weights (per regime)
        self.nz_lin = nn.Linear(n_regimes, 1)
        self.tau_lin = nn.Linear(n_regimes, 1)
        
        # Initialize to sensible defaults
        with torch.no_grad():
            # Initialize neutral zone to be wider in regime 2 (choppy) than in regime 1 (trending)
            self.nz_lin.weight.data = torch.tensor([[0.01, 0.0, 0.05]])
            self.nz_lin.bias.data.fill_(0.02)  # Base neutral zone size
            
            # Initialize target vol to be higher in regime 1 (trending) than in regime 2 (choppy)
            self.tau_lin.weight.data = torch.tensor([[0.0, 0.5, -0.5]])
            self.tau_lin.bias.data.fill_(0.0)  # Start with mid-level target vol

    def forward(self, p_t, sigma_t, regime_logits):
        """Compute position size with adaptive controls.
        
        Args:
            p_t: Probability predictions (batch_size,)
            sigma_t: Volatility estimates (batch_size,)
            regime_logits: Regime classification logits (batch_size, n_regimes)
            
        Returns:
            position_t: Position sizing (batch_size,)
        """
        # Clamp thresholds to a reasonable range [0.3, 0.7]
        theta_L = self.theta_L.clamp(0.3, 0.7)
        theta_S = self.theta_S.clamp(0.3, 0.7)
        
        # Compute adaptive neutral zone and target vol based on regime
        nz_t = F.softplus(self.nz_lin(regime_logits).squeeze(-1))  # Ensures nz_t ≥ 0
        tau_t = torch.sigmoid(self.tau_lin(regime_logits).squeeze(-1)) * self.config.tau_max
        
        # Get steepness factor
        k = torch.exp(self.log_k)
        
        # Compute masks for long and short positions using thresholds directly
        long_mask = (p_t > theta_L).float()
        short_mask = (p_t < theta_S).float()
        
        # Compute signal strengths using clamped thresholds
        long_sig = torch.sigmoid(k * (p_t - theta_L)) * long_mask
        short_sig = torch.sigmoid(k * (theta_S - p_t)) * short_mask
        
        # Normalize signals
        norm_sum = long_sig + short_sig + self.config.eps
        pos_dir = (long_sig / norm_sum) - (short_sig / norm_sum)
        
        # Apply volatility scaling
        dyn_size = (tau_t / (sigma_t + self.config.eps)).clamp(0, self.config.max_pos)
        position_t = pos_dir * dyn_size
        
        return position_t


class Backtester(nn.Module):
    """End-to-end differentiable backtester for DMT v2 strategy.
    
    Args:
        strategy (StrategyLayer): DMT strategy layer
        config (Config): Configuration parameters
    """
    def __init__(self, strategy: StrategyLayer, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        self.strategy = strategy
        self.config = config

    def forward(self, p_series, sigma_series, regime_series, r_series):
        """Run backtest in a differentiable way.
        
        Args:
            p_series: Probability predictions tensor (batch_size, time_steps)
            sigma_series: Volatility estimates tensor (batch_size, time_steps)
            regime_series: Regime logits tensor (batch_size, time_steps, n_regimes)
            r_series: Returns tensor (batch_size, time_steps)
            
        Returns:
            log_eq: Log equity curves (batch_size, time_steps)
            rets: Net returns (batch_size, time_steps)
            turn: Turnover (batch_size, time_steps)
        """
        batch, T = p_series.shape
        equity = p_series.new_ones(batch)  # Initial equity = 1.0
        prev_pos = p_series.new_zeros(batch)  # Initial position = 0
        log_equity = []
        daily_rets = []
        turnover = []
        
        # Step through time
        for t in range(T):
            p_t = p_series[:, t]
            sigma_t = sigma_series[:, t]
            reg_t = regime_series[:, t]
            r_t = r_series[:, t]
            
            # Get position for this step
            pos_t = self.strategy(p_t, sigma_t, reg_t)
            
            # Calculate transaction costs
            t_cost = self.config.trans_cost * (pos_t - prev_pos).abs()
            
            # Calculate net return
            net_ret = pos_t * r_t - t_cost
            
            # Update equity
            equity = equity * (1.0 + net_ret)
            
            # Store values
            log_equity.append(equity.log())
            daily_rets.append(net_ret)
            turnover.append((pos_t - prev_pos).abs())
            
            # Update previous position
            prev_pos = pos_t
            
        # Stack results
        log_eq = torch.stack(log_equity, dim=1)
        rets = torch.stack(daily_rets, dim=1)
        turn = torch.stack(turnover, dim=1)
        
        return log_eq, rets, turn


def loss_function(log_eq, rets, turn, config: Config = None):
    """Composite loss function for DMT v2.
    
    Maximizes:
        log return + λ_S·Sharpe - λ_D·MaxDrawdown - λ_T·Turnover
    
    Args:
        log_eq: Log equity curves (batch_size, time_steps)
        rets: Net returns (batch_size, time_steps)
        turn: Turnover (batch_size, time_steps)
        config: Configuration parameters
        
    Returns:
        Loss value to minimize
    """
    if config is None:
        config = Config()
    
    # Final log return
    log_ret = log_eq[:, -1] - log_eq[:, 0]
    
    # Sharpe ratio (annualized)
    mean_r = rets.mean(dim=1)
    std_r = rets.std(dim=1) + config.eps
    sharpe = mean_r / std_r * math.sqrt(252)
    
    # Max drawdown
    cum = log_eq.exp()
    running_max, _ = torch.cummax(cum, dim=1)
    dd = (cum / running_max - 1.0).min(dim=1).values.abs()
    
    # Total turnover
    tot_turn = turn.sum(dim=1)
    
    # Composite objective
    obj = (log_ret + 
           config.lambda_sharpe * sharpe - 
           config.lambda_draw * dd - 
           config.lambda_turn * tot_turn)
    
    return -obj.mean()  # Minimize negative objective


# Feature creation utilities
def create_feature_matrix(prices_df, window_size=20, handle_nans='drop'):
    """Create feature matrix from price DataFrame.
    
    Args:
        prices_df: DataFrame with price data
        window_size: Lookback window for features
        handle_nans: How to handle NaN values - 'drop', 'fill_zeros', or 'fill_means'
        
    Returns:
        Feature DataFrame and target labels
    """
    # Create basic features
    df = prices_df.copy()
    
    # Handle different DataFrame structures
    price_col = 'Close' if 'Close' in df.columns else 'QQQ' if 'QQQ' in df.columns else df.columns[0]
    
    # Returns at different horizons
    df['ret_1d'] = df[price_col].pct_change()
    df['ret_5d'] = df[price_col].pct_change(5)
    df['ret_10d'] = df[price_col].pct_change(10)
    df['ret_20d'] = df[price_col].pct_change(20)
    
    # Moving averages
    df['ma_5'] = df[price_col].rolling(5).mean()
    df['ma_10'] = df[price_col].rolling(10).mean()
    df['ma_20'] = df[price_col].rolling(20).mean()
    df['ma_50'] = df[price_col].rolling(50).mean()
    df['ma_200'] = df[price_col].rolling(200).mean()
    
    # MA ratios
    df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
    df['ma_ratio_10_20'] = df['ma_10'] / df['ma_20']
    df['ma_ratio_50_200'] = df['ma_50'] / df['ma_200']
    
    # Volatility features
    df['vol_20'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['vol_50'] = df['ret_1d'].rolling(50).std() * np.sqrt(252)
    # Volatility ratio (short-term vs long-term)
    df['vol_ratio'] = df['ret_1d'].rolling(10).std() / df['ret_1d'].rolling(30).std()
    
    # RSI
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # Add shorter and longer RSI periods
    avg_gain_short = gain.rolling(7).mean()
    avg_loss_short = loss.rolling(7).mean()
    rs_short = avg_gain_short / avg_loss_short.replace(0, 1e-8)
    df['rsi_7'] = 100 - (100 / (1 + rs_short))
    
    # Bollinger Bands
    df['bb_upper'] = df['ma_20'] + 2 * df[price_col].rolling(20).std()
    df['bb_lower'] = df['ma_20'] - 2 * df[price_col].rolling(20).std()
    df['bb_pct'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Momentum indicators
    df['mom_10'] = df[price_col] / df[price_col].shift(10) - 1
    df['mom_20'] = df[price_col] / df[price_col].shift(20) - 1
    
    # Mean reversion signals
    df['zscore_5d'] = (df[price_col] - df[price_col].rolling(5).mean()) / df[price_col].rolling(5).std()
    df['zscore_10d'] = (df[price_col] - df[price_col].rolling(10).mean()) / df[price_col].rolling(10).std()
    
    # Target: binary label for positive return
    df['target'] = (df['ret_1d'].shift(-1) > 0).astype(float)
    
    # Create feature columns
    feature_cols = [
        'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'ma_ratio_5_20', 'ma_ratio_10_20', 'ma_ratio_50_200',
        'vol_20', 'vol_50', 'vol_ratio',
        'rsi_7', 'rsi_14', 'bb_pct',
        'macd', 'macd_signal', 'macd_hist',
        'mom_10', 'mom_20',
        'zscore_5d', 'zscore_10d'
    ]
    
    # Handle missing values according to the specified method
    if handle_nans == 'drop':
        df = df.dropna()
    elif handle_nans == 'fill_zeros':
        msg = f"Filled NaN values with zeros in {df[feature_cols].isna().sum().sum()} feature cells"
        print(msg)
        df[feature_cols] = df[feature_cols].fillna(0)
    elif handle_nans == 'fill_means':
        # Calculate the mean for each feature
        feature_means = df[feature_cols].mean()
        # Fill NaN values with the mean of each feature
        filled_count = df[feature_cols].isna().sum().sum()
        df[feature_cols] = df[feature_cols].fillna(feature_means)
        print(f"Filled NaN values with feature means in {filled_count} features")
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, df
