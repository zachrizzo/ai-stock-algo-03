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


class Config:
    """Configuration for DMT v2 strategy."""
    
    def __init__(self, 
                 eps=1e-8, 
                 tau_max=0.35, 
                 max_pos=2.0, 
                 neutral_zone=0.03,
                 lr=0.015,
                 seq_len=15):
        self.eps = eps
        self.tau_max = tau_max  # Maximum target volatility
        self.max_pos = max_pos  # Maximum position size
        self.neutral_zone = neutral_zone  # Neutral zone size
        self.lr = lr  # Learning rate
        self.seq_len = seq_len  # Sequence length


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
    """Market regime classifier for DMT v2 strategy."""
    
    def __init__(self, in_dim, hidden_dim=64, n_regimes=3):
        """Initialize the regime classifier.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_regimes: Number of regimes to classify (typically 3: bull, bear, neutral)
        """
        super().__init__()
        
        # MLP regime classifier
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_regimes)
        )
    
    def forward(self, x):
        """Forward pass of the regime classifier.
        
        Args:
            x: Input tensor (batch_size, feature_dim)
            
        Returns:
            logits: Regime logits (batch_size, n_regimes)
        """
        return self.model(x)


class PredictionModel(nn.Module):
    """Transformer-based prediction model for DMT v2 with multi-timeframe support."""
    
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int = 96, 
                 out_dim: int = 1, 
                 seq_len: int = 15, 
                 n_heads: int = 6,
                 n_layers: int = 5,
                 dropout: float = 0.1):
        """Initialize the prediction model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension for transformer
            out_dim: Output dimension (typically 1)
            seq_len: Sequence length for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Make sure each transformer's hidden dimension is divisible by the number of heads
        # For 96 hidden_dim and 6 heads, we use 32 dims per timeframe (32*3=96, 32 is divisible by 2 heads)
        frames_hidden = 32  # Per-timeframe hidden dimension
        per_frame_heads = 2  # Heads per timeframe, must evenly divide frames_hidden
        
        # Project input features to transformer dimension for each timeframe
        self.short_proj = nn.Linear(in_dim, frames_hidden)
        self.med_proj = nn.Linear(in_dim, frames_hidden)
        self.long_proj = nn.Linear(in_dim, frames_hidden)
        
        self.short_norm = nn.LayerNorm(frames_hidden)
        self.med_norm = nn.LayerNorm(frames_hidden)
        self.long_norm = nn.LayerNorm(frames_hidden)
        
        # Dropout layers
        self.input_dropout = nn.Dropout(dropout)
        
        # Transformer encoder for each timeframe
        short_encoder_layer = nn.TransformerEncoderLayer(
            d_model=frames_hidden,
            nhead=per_frame_heads,
            dim_feedforward=frames_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        med_encoder_layer = nn.TransformerEncoderLayer(
            d_model=frames_hidden,
            nhead=per_frame_heads,
            dim_feedforward=frames_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        long_encoder_layer = nn.TransformerEncoderLayer(
            d_model=frames_hidden,
            nhead=per_frame_heads,
            dim_feedforward=frames_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.short_transformer = nn.TransformerEncoder(short_encoder_layer, n_layers // 2)
        self.med_transformer = nn.TransformerEncoder(med_encoder_layer, n_layers)
        self.long_transformer = nn.TransformerEncoder(long_encoder_layer, n_layers // 2)
        
        # Output projection after concatenating transformer outputs (3 * frames_hidden = combined dimension)
        self.output_linear = nn.Linear(frames_hidden * 3, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        # Output heads
        self.class_head = nn.Linear(hidden_dim, out_dim)
        self.quant_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x_short, x_med=None, x_long=None, mask=None):
        """Forward pass of the prediction model.
        
        Args:
            x_short: Short-term input tensor (batch_size, short_seq_len, in_dim)
            x_med: Medium-term input tensor (batch_size, med_seq_len, in_dim)
            x_long: Long-term input tensor (batch_size, long_seq_len, in_dim)
            mask: Optional attention mask
            
        Returns:
            p_t: Probability predictions
            q_lo: Lower quantile predictions
            q_hi: Upper quantile predictions
        """
        # Process short timeframe
        short_out = self.short_proj(x_short)
        short_out = self.short_norm(short_out)
        short_out = self.input_dropout(short_out)
        short_out = self.short_transformer(short_out, mask=mask)
        short_feat = short_out[:, -1, :]  # Use final timestep features
        
        # If medium timeframe is provided
        if x_med is not None:
            med_out = self.med_proj(x_med)
            med_out = self.med_norm(med_out)
            med_out = self.input_dropout(med_out)
            med_out = self.med_transformer(med_out, mask=mask)
            med_feat = med_out[:, -1, :]
        else:
            med_feat = short_feat  # Fallback
            
        # If long timeframe is provided
        if x_long is not None:
            long_out = self.long_proj(x_long)
            long_out = self.long_norm(long_out)
            long_out = self.input_dropout(long_out)
            long_out = self.long_transformer(long_out, mask=mask)
            long_feat = long_out[:, -1, :]
        else:
            long_feat = med_feat  # Fallback
            
        # Concatenate features from all timeframes
        combined_feat = torch.cat([short_feat, med_feat, long_feat], dim=1)
        
        # Project to single vector and apply norm/dropout
        output = self.output_linear(combined_feat)
        output = self.output_norm(output)
        output = self.output_dropout(output)
        
        # Get probability prediction
        logits = self.class_head(output)
        p_t = torch.sigmoid(logits).squeeze(-1)
        
        # Get quantile predictions
        q_t = self.quant_head(output)
        q_lo, q_hi = q_t.split(1, dim=-1)
        q_lo = q_lo.squeeze(-1)
        q_hi = q_hi.squeeze(-1)
        
        return p_t, q_lo, q_hi


class StrategyLayer(nn.Module):
    """Strategy layer for DMT v2, transforms predictions into positions."""
    
    def __init__(self, config, n_regimes=3):
        """Initialize the strategy layer.
        
        Args:
            config: Configuration instance with strategy parameters
            n_regimes: Number of market regimes to consider
        """
        super().__init__()
        
        self.config = config
        
        # Learnable long/short thresholds (initialized to proven values)
        self.theta_L = nn.Parameter(torch.tensor(0.55))  # Standard long threshold
        self.theta_S = nn.Parameter(torch.tensor(0.45))  # Standard short threshold
        self.log_k = nn.Parameter(torch.log(torch.tensor(50.0)))
        
        # Adaptive neutral-zone and target vol weights (per regime)
        self.nz_lin = nn.Linear(n_regimes, 1)
        self.tau_lin = nn.Linear(n_regimes, 1)
        self.max_pos_lin = nn.Linear(n_regimes, 1)
        
        # Initialize with the high-performing values
        self.nz_lin.bias.data.fill_(float(config.neutral_zone))
        self.tau_lin.bias.data.fill_(0.5)  # Middle of sigmoid range
        self.max_pos_lin.bias.data.fill_(0.5)  # Middle of sigmoid range
        
        # Initialize regime weights according to the original high-performing version
        # Regime 0: Bull, Regime 1: Neutral, Regime 2: Bear
        self.nz_lin.weight.data = torch.tensor([[0.01, 0.03, 0.05]])  # Wider neutral zone in bear markets
        self.tau_lin.weight.data = torch.tensor([[0.2, 0.0, -0.2]])   # Higher target vol in bull markets
        self.max_pos_lin.weight.data = torch.tensor([[0.3, 0.0, -0.2]])  # Higher max pos in bull markets

    def forward(self, p_t, sigma_t, regime_logits):
        """Compute position size with adaptive controls.
        
        Args:
            p_t: Probability predictions (batch_size,)
            sigma_t: Volatility estimates (batch_size,)
            regime_logits: Regime classification logits (batch_size, n_regimes)
            
        Returns:
            position_t: Position sizing (batch_size,)
        """
        # Clamp thresholds to reasonable range (0.3-0.7)
        theta_L = self.theta_L.clamp(0.3, 0.7)
        theta_S = self.theta_S.clamp(0.3, 0.7)
        
        # Ensure regime_logits is properly formatted for linear layers
        if regime_logits.dim() == 1:
            regime_logits = regime_logits.unsqueeze(0)  # Add batch dimension if missing
        
        # Apply softmax to get regime probabilities
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Slightly smaller neutral zone (less aggressive than before)
        nz_t = F.softplus(self.nz_lin(regime_probs).squeeze(-1)) * 0.8
        
        # Moderately higher target vol (less aggressive)
        tau_t = (torch.sigmoid(self.tau_lin(regime_probs)).squeeze(-1) * 0.3 + 0.7) * self.config.tau_max
        
        # Dynamic max position based on regime - calculate max position as scalar
        max_pos_scalar = self.config.max_pos
        max_pos_modifier = (torch.sigmoid(self.max_pos_lin(regime_probs)).squeeze(-1) * 0.4 + 0.7).mean().item()
        effective_max_pos = max_pos_scalar * max_pos_modifier
        
        # Get steepness factor - slightly increased for sharper transitions
        k = torch.exp(self.log_k) * 1.1
        
        # Compute masks for long and short positions using thresholds directly
        # Slightly narrower threshold gap for more time in market
        gap_adjust = 0.015  # Less aggressive threshold adjustment
        long_mask = (p_t > (theta_L - gap_adjust)).float()
        short_mask = (p_t < (theta_S + gap_adjust)).float()
        
        # Compute signal strengths using clamped thresholds and increased steepness
        long_sig = torch.sigmoid(k * (p_t - theta_L)) * long_mask
        short_sig = torch.sigmoid(k * (theta_S - p_t)) * short_mask
        
        # Normalize signals
        norm_sum = long_sig + short_sig + self.config.eps
        pos_dir = (long_sig / norm_sum) - (short_sig / norm_sum)
        
        # Ensure tau_t matches sigma_t's shape
        if tau_t.shape != sigma_t.shape:
            if tau_t.dim() == 0 and sigma_t.dim() == 1:
                # tau_t is scalar, sigma_t is vector
                tau_t = tau_t.expand_as(sigma_t)
            elif tau_t.dim() == 1 and sigma_t.dim() == 1 and tau_t.shape[0] != sigma_t.shape[0]:
                # Both are vectors but with different lengths
                if tau_t.shape[0] == 1:
                    tau_t = tau_t.expand_as(sigma_t)
                else:
                    # Use the mean of tau_t as a scalar
                    tau_t = torch.mean(tau_t).expand_as(sigma_t)
        
        # Apply volatility scaling with scalar max position size
        dyn_size = (tau_t / (sigma_t + self.config.eps)).clamp(0, effective_max_pos)
        
        # Apply position scaling
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
            t_cost = self.config.eps * (pos_t - prev_pos).abs()
            
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
           0.10 * sharpe - 
           0.05 * dd - 
           0.002 * tot_turn)
    
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
