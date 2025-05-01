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
                 neutral_zone=0.025,  
                 lr=0.015,
                 seq_len=15,
                 max_drawdown_threshold=0.2,  
                 risk_scaling_factor=0.6,  
                 uncertainty_threshold=0.25,  
                 use_ensemble=True,
                 use_dynamic_stops=True,
                 stop_loss_atr_multiple=2.5,  
                 use_regime_detection=True,
                 regime_smoothing_window=3):  
        self.eps = eps
        self.tau_max = tau_max
        self.max_pos = max_pos
        self.neutral_zone = neutral_zone
        self.lr = lr
        self.seq_len = seq_len
        self.max_drawdown_threshold = max_drawdown_threshold
        self.risk_scaling_factor = risk_scaling_factor
        self.uncertainty_threshold = uncertainty_threshold
        self.use_ensemble = use_ensemble
        self.use_dynamic_stops = use_dynamic_stops
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.use_regime_detection = use_regime_detection
        self.regime_smoothing_window = regime_smoothing_window


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
    """Market regime classifier for DMT v2 strategy with enhanced indicators."""
    
    def __init__(self, in_dim, hidden_dim=96, n_regimes=3, smoothing_window=3):
        """Initialize the regime classifier.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            n_regimes: Number of regimes to classify (typically 3: bull, bear, neutral)
            smoothing_window: Window size for regime probability smoothing
        """
        super().__init__()
        
        self.smoothing_window = smoothing_window
        self.n_regimes = n_regimes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Enhanced VIX processor with increased capacity
        self.vix_processor = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        # Increased combined dimension
        self.regime_head = nn.Linear(hidden_dim // 2 + 12, n_regimes)
        self.regime_history = []
        
        # Add a momentum layer to better identify trend direction
        self.momentum_processor = nn.Sequential(
            nn.Linear(5, 16),  # Process 5 recent returns
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
    def forward(self, x, vix=None):
        """Forward pass of the regime classifier.
        
        Args:
            x: Input tensor (batch_size, feature_dim)
            vix: VIX feature tensor (batch_size, 1)
            
        Returns:
            logits: Regime logits (batch_size, n_regimes)
        """
        # Process the main features
        features = self.feature_extractor(x)
        
        # Process VIX data if available
        if vix is not None:
            vix_features = self.vix_processor(vix)
            features = torch.cat([features, vix_features], dim=1)
        
        # Calculate regime probabilities
        regime_logits = self.regime_head(features)
        regime_probs = F.softmax(regime_logits, dim=1)
        
        # Apply smoothing if we have history
        if len(self.regime_history) > 0:
            # Calculate exponential moving average weight for new data
            # More aggressive smoothing with recent regime having more weight
            alpha = 0.6  # Increased from default value to be more responsive
            
            # Get previous regime probabilities
            prev_probs = self.regime_history[-1]
            
            # Apply exponential smoothing
            smoothed_probs = alpha * regime_probs + (1 - alpha) * prev_probs
            
            # Normalize to ensure they sum to 1
            smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True)
            regime_probs = smoothed_probs
        
        # Store regime probabilities for future smoothing
        self.regime_history.append(regime_probs.detach())
        
        # Only keep the most recent smoothing_window entries
        if len(self.regime_history) > self.smoothing_window:
            self.regime_history.pop(0)
            
        return regime_logits
        
    def get_regime_probs(self):
        # Return the most recent smoothed regime probabilities
        if len(self.regime_history) > 0:
            return self.regime_history[-1]
        else:
            return torch.ones(1, self.n_regimes) / self.n_regimes


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
    
    def loss_fn(self, p_t, sigma_t, y):
        """
        Calculate loss for training the model.
        
        Args:
            p_t: Probability predictions (batch_size, 1)
            sigma_t: Volatility estimates (batch_size, 1)
            y: Target values (batch_size, 1)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Convert p_t from probability to expected return
        expected_return = (p_t - 0.5) * 2 * 0.02  # Scale to ±2% range
        
        # Mean squared error on returns
        mse_loss = torch.mean((expected_return - y) ** 2)
        
        # Volatility calibration loss (encourage sigma_t to match actual squared error)
        squared_error = (expected_return - y) ** 2
        vol_loss = torch.mean((sigma_t - squared_error) ** 2)
        
        # Combine losses with weighting
        combined_loss = mse_loss + 0.2 * vol_loss
        
        return combined_loss


class EnsembleModel:
    def __init__(self, in_dim, n_models=4, device='cpu'):
        self.models = []
        self.device = device
        
        # Create models with different architectures and initialization
        # Model 1: Standard configuration but larger
        self.models.append(PredictionModel(
            in_dim=in_dim,
            hidden_dim=128,  # Increased from default
            transformer_dim=96,  # Increased from default
            n_heads=8,  # Increased from 6
            n_layers=6,  # Increased from 5
            dropout=0.1
        ).to(device))
        
        # Model 2: Deeper architecture with more regularization
        self.models.append(PredictionModel(
            in_dim=in_dim,
            hidden_dim=144,
            transformer_dim=102,
            n_heads=6,
            n_layers=8,  # More layers
            dropout=0.2  # More dropout
        ).to(device))
        
        # Model 3: Wider architecture with less regularization
        self.models.append(PredictionModel(
            in_dim=in_dim,
            hidden_dim=160,
            transformer_dim=120,
            n_heads=10,
            n_layers=4,  # Fewer layers but wider
            dropout=0.05  # Less dropout
        ).to(device))
        
        # Model 4: Hybrid architecture with different activation functions
        hybrid_model = PredictionModel(
            in_dim=in_dim,
            hidden_dim=136,
            transformer_dim=112,
            n_heads=7,
            n_layers=5,
            dropout=0.15
        ).to(device)
        
        # Modify the hybrid model to use GELU activations
        for name, module in hybrid_model.named_modules():
            if isinstance(module, nn.ReLU):
                if hasattr(nn, 'GELU'):
                    relu_parent = get_parent_module(hybrid_model, name)
                    setattr(relu_parent, name.split('.')[-1], nn.GELU())
        
        self.models.append(hybrid_model)
        
        # Assign different weights to each model - give more weight to larger models
        self.model_weights = [0.25, 0.25, 0.25, 0.25]
        
    def train(self, train_data, optimizer, epochs, scheduler=None):
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for model in self.models:
                model.train()
                total_loss = 0
                
                for x_batch, y_batch in train_data:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    p_t, sigma_t, regime_logits = model(x_batch)
                    
                    # Calculate loss
                    loss = model.loss_fn(p_t, sigma_t, y_batch)
                    
                    # Add L2 regularization to prevent overfitting
                    l2_reg = 0
                    for param in model.parameters():
                        l2_reg += torch.norm(param, 2) ** 2
                    loss += 1e-5 * l2_reg
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if scheduler:
                    scheduler.step()
                
                avg_loss = total_loss / len(train_data)
                epoch_losses.append(avg_loss)
            
            # Calculate average loss across all models
            avg_ensemble_loss = sum(epoch_losses) / len(self.models)
            losses.append(avg_ensemble_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_ensemble_loss:.6f}")
        
        return losses
        
    def predict(self, x):
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
        
        p_all = []
        sigma_all = []
        regime_all = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                p_t, sigma_t, regime_logits = model(x)
                
                # Apply model weight
                weight = self.model_weights[i]
                p_all.append(p_t * weight)
                sigma_all.append(sigma_t * weight)
                regime_all.append(regime_logits * weight)
        
        # Combine predictions from all models
        p_ensemble = sum(p_all)
        sigma_ensemble = sum(sigma_all)
        regime_ensemble = sum(regime_all)
        
        # Calculate prediction uncertainty (variance between models)
        p_variance = torch.zeros_like(p_ensemble)
        for p in p_all:
            p_variance += (p - p_ensemble) ** 2
        p_variance /= len(self.models)
        
        # Adjust sigma based on model disagreement
        sigma_ensemble = torch.sqrt(sigma_ensemble ** 2 + p_variance)
        
        return p_ensemble, sigma_ensemble, regime_ensemble


class StrategyLayer(nn.Module):
    """Strategy layer for DMT v2, transforms predictions into positions with enhanced risk management."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize neural network layers for position sizing
        self.nz_lin = nn.Linear(3, 1)
        
        # Enhanced position sizing networks with larger capacity
        self.regime_position_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        
        # Uncertainty handling network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Volatility adjustment network
        self.volatility_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Initialize with conservative values
        with torch.no_grad():
            # Start with a wider neutral zone
            self.nz_lin.weight.fill_(0.5)
            self.nz_lin.bias.fill_(0.5)
    
    def forward(self, p_t, sigma_t, regime_logits, equity_curve=None, atr=None, uncertainty=None):
        # Calculate position size with adaptive controls and risk management
        regime_probs = F.softmax(regime_logits, dim=1)
        
        # Calculate neutral zone - dynamically set based on regime
        neutral_zone = torch.sigmoid(self.nz_lin(regime_probs)) * self.config.neutral_zone * 2
        
        # Base position scaling from prediction
        raw_pos = torch.zeros_like(p_t)
        
        # For values above neutral zone
        above_mask = p_t > neutral_zone
        below_mask = p_t < -neutral_zone
        
        # More aggressive scaling function with reduced neutral zone
        raw_pos[above_mask] = ((p_t[above_mask] - neutral_zone[above_mask]) / 
                              (self.config.tau_max - neutral_zone[above_mask])) * self.config.max_pos
        
        raw_pos[below_mask] = ((p_t[below_mask] + neutral_zone[below_mask]) / 
                              (self.config.tau_max - neutral_zone[below_mask])) * -self.config.max_pos
        
        # Apply regime-based position sizing
        # More aggressive in bullish regimes, more conservative in bearish
        # Regime weights: [bullish, neutral, bearish]
        regime_pos_scaling = self.regime_position_net(regime_probs)
        regime_pos_scaling = 1.0 + 0.5 * torch.tanh(regime_pos_scaling)  # Range: [0.5, 1.5]
        
        # Apply uncertainty-based scaling if available
        uncertainty_scaling = torch.ones_like(p_t)
        if uncertainty is not None:
            uncertainty_relative = uncertainty / sigma_t
            uncertainty_scaling = self.uncertainty_net(uncertainty_relative)
            
            # Scale positions down when uncertainty is high relative to predicted volatility
            # More aggressive scaling curve - linearly reduce from 1.0 to 0.5 instead of 0.0
            uncertainty_scaling = 0.5 + 0.5 * (1.0 - torch.clamp(
                uncertainty_relative / self.config.uncertainty_threshold, 0.0, 1.0))
        
        # Apply drawdown protection if equity curve is provided
        drawdown_scaling = torch.ones_like(p_t)
        if equity_curve is not None and len(equity_curve) > 1:
            # Calculate current drawdown
            peak = torch.maximum(equity_curve[0], torch.max(equity_curve))
            current = equity_curve[-1]
            drawdown = (current / peak) - 1.0
            
            # Apply graduated scaling based on drawdown severity
            # New: More graduated scaling that reduces position more gently
            if drawdown < 0:
                # Convert to positive number for easier comparison
                drawdown_pct = -drawdown
                
                # Define drawdown thresholds and corresponding scaling factors
                # More granular scale with less aggressive reductions
                if drawdown_pct < self.config.max_drawdown_threshold * 0.5:
                    # Minor drawdown - apply minimal scaling
                    dd_scale = 1.0 - (drawdown_pct / (self.config.max_drawdown_threshold * 2))
                    drawdown_scaling = torch.ones_like(p_t) * max(dd_scale, 0.9)
                elif drawdown_pct < self.config.max_drawdown_threshold:
                    # Moderate drawdown - apply graduated scaling
                    dd_scale = 0.9 - 0.3 * ((drawdown_pct - self.config.max_drawdown_threshold * 0.5) / 
                                          (self.config.max_drawdown_threshold * 0.5))
                    drawdown_scaling = torch.ones_like(p_t) * max(dd_scale, 0.6)
                else:
                    # Severe drawdown - apply maximum scaling reduction
                    drawdown_scaling = torch.ones_like(p_t) * self.config.risk_scaling_factor
        
        # Apply volatility-based scaling if ATR is provided
        volatility_scaling = torch.ones_like(p_t)
        if atr is not None:
            # Calculate relative ATR (volatility)
            relative_atr = atr / equity_curve[-1]
            
            # Scale positions based on volatility
            volatility_scaling = self.volatility_net(relative_atr.unsqueeze(-1))
            
            # For low volatility, allow increased position size
            # For high volatility, reduce position size
            volatility_scaling = 1.5 - volatility_scaling
        
        # Combine all scaling factors - ensure they work together harmoniously
        combined_scaling = (regime_pos_scaling * uncertainty_scaling * 
                           drawdown_scaling * volatility_scaling)
        
        # Apply combined scaling to raw position
        position = raw_pos * combined_scaling
        
        # Add position caps to prevent excessive leverage
        position = torch.clamp(position, -self.config.max_pos, self.config.max_pos)
        
        return position


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


# Helper function to get parent module for modifying activations
def get_parent_module(model, name):
    """Get the parent module of a nested module specified by name"""
    if '.' not in name:
        return model
    
    parent_name = '.'.join(name.split('.')[:-1])
    names = parent_name.split('.')
    module = model
    
    for n in names:
        module = getattr(module, n)
    
    return module
