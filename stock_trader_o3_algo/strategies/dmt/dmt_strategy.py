#!/usr/bin/env python3
"""
Differentiable Market Twin (DMT) Strategy

This module implements a differentiable version of the Tri-Shot strategy,
allowing for gradient-based optimization of strategy parameters through
backpropagation with the Market Twin model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import datetime as dt

# Local imports
from dmt_model import MarketTwinLSTM, GumbelSoftmax


class DifferentiableTriShot(nn.Module):
    """
    A differentiable version of the Tri-Shot strategy.
    
    This class wraps the Tri-Shot strategy logic in a fully differentiable
    PyTorch module, allowing for gradient-based optimization of strategy
    parameters through backpropagation.
    """
    
    def __init__(self,
                 feature_dim: int,
                 initial_capital: float = 500.0,
                 transaction_cost: float = 0.0003,  # 3 bps per round trip
                 device: str = 'cpu'):
        """
        Initialize the differentiable strategy.
        
        Args:
            feature_dim: Dimension of the feature vector
            initial_capital: Initial capital for the strategy
            transaction_cost: Transaction cost rate (as decimal)
            device: PyTorch device to use
        """
        super(DifferentiableTriShot, self).__init__()
        
        self.feature_dim = feature_dim
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.device = device
        
        # Differentiable parameters for the strategy
        
        # 1. Signal threshold parameters - learnable
        self.long_threshold = nn.Parameter(torch.tensor(0.52, device=device))
        self.short_threshold = nn.Parameter(torch.tensor(0.48, device=device))
        self.neutral_band_size = nn.Parameter(torch.tensor(0.04, device=device))  # Size of neutral zone
        
        # 2. Feature importance weights for predictions - learnable
        self.feature_weights = nn.Parameter(torch.ones(feature_dim, device=device) / feature_dim)
        
        # 3. Position sizing parameters - learnable
        self.base_position_size = nn.Parameter(torch.tensor(0.80, device=device))  # Base position size
        self.conviction_scalar = nn.Parameter(torch.tensor(0.50, device=device))   # How much to scale by conviction
        self.vix_impact = nn.Parameter(torch.tensor(0.20, device=device))         # How much VIX affects sizing
        
        # 4. Opportunistic entry parameters - learnable
        self.vix_collapse_threshold = nn.Parameter(torch.tensor(0.15, device=device))
        self.vix_spike_threshold = nn.Parameter(torch.tensor(0.15, device=device))
        
        # 5. Other parameters
        # PDT constraint is enforced externally
        
        # Gumbel Softmax for differentiable discrete decisions
        self.gumbel_softmax = GumbelSoftmax(temperature=1.0)
        
        # For position decision
        self.position_logits = nn.Linear(feature_dim, 3)  # 3 positions: Long/Short/Cash
        
    def forward(self, features, initial_state=None):
        """
        Forward pass for the strategy.
        
        Args:
            features: Tensor of features (batch_size, seq_len, feature_dim)
            initial_state: Optional initial state
            
        Returns:
            Dictionary with position actions and computed metrics
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Initialize equity curve at initial_capital
        equity = torch.ones(batch_size, seq_len + 1, device=self.device) * self.initial_capital
        positions = torch.zeros(batch_size, seq_len + 1, 3, device=self.device)  # Long/Short/Cash
        positions[:, 0, 2] = 1.0  # Start in cash
        
        # Extract market features
        price_momentum = features[:, :, 0]  # Assuming first feature is price_momentum
        vix = features[:, :, 1]             # Assuming second feature is VIX
        vix_change = features[:, :, 2]      # Assuming third feature is VIX change
        
        # Generate predictions using weighted features
        weighted_features = features * self.feature_weights.unsqueeze(0).unsqueeze(0)
        predictions = torch.sigmoid(weighted_features.sum(dim=2))  # Naive predictor
        
        # Process each time step
        for t in range(seq_len):
            # Calculate signal strength
            signal_strength = torch.abs(predictions[:, t] - 0.5) / 0.5
            
            # Detect VIX conditions
            vix_collapse = (vix_change[:, t] < -self.vix_collapse_threshold) & (vix[:, t] < 30.0)
            vix_spike = (vix_change[:, t] > self.vix_spike_threshold) & (vix[:, t] > 20.0)
            
            # Position sizing with conviction and VIX conditions
            position_size = self.base_position_size + self.conviction_scalar * signal_strength
            position_size = torch.clamp(position_size, 0.2, 1.0)  # Limit position size
            
            # Adjust for VIX conditions
            position_size = torch.where(
                vix_collapse | vix_spike,
                position_size * (1.0 + self.vix_impact),
                position_size
            )
            
            # Generate position logits
            long_condition = (predictions[:, t] >= self.long_threshold) & (price_momentum[:, t] > 0)
            short_condition = (predictions[:, t] <= self.short_threshold) & (price_momentum[:, t] < 0)
            
            # Add opportunistic conditions
            opportunistic_long = vix_collapse
            opportunistic_short = vix_spike
            
            # Final long/short conditions
            long_position = long_condition | opportunistic_long
            short_position = short_condition | opportunistic_short
            
            # Create position logits
            logits = torch.zeros(batch_size, 3, device=self.device)
            logits[:, 0] = torch.where(long_position, torch.tensor(1.0, device=self.device), torch.tensor(-10.0, device=self.device))
            logits[:, 1] = torch.where(short_position, torch.tensor(1.0, device=self.device), torch.tensor(-10.0, device=self.device))
            # Cash is default when neither long nor short
            logits[:, 2] = torch.where(~long_position & ~short_position, torch.tensor(1.0, device=self.device), torch.tensor(-10.0, device=self.device))
            
            # Use Gumbel-Softmax for differentiable position selection
            position_probs = self.gumbel_softmax(logits)
            
            # Store position
            positions[:, t+1] = position_probs
            
            # Calculate returns (simplified here, would be replaced with market simulator)
            market_return = features[:, t, 3]  # Assuming 4th feature is returns
            
            # Apply position sizing to actionable positions (first 2 columns)
            sized_positions = positions[:, t+1].clone()
            sized_positions[:, 0:2] *= position_size.unsqueeze(1)
            
            # Calculate strategy return
            strategy_return = (
                sized_positions[:, 0] * market_return +   # Long return
                sized_positions[:, 1] * -market_return    # Short return
            )
            
            # Transaction costs on position changes
            position_change = (positions[:, t+1] - positions[:, t]).abs().sum(dim=1)
            transaction_cost = position_change * self.transaction_cost
            
            # Update equity
            equity[:, t+1] = equity[:, t] * (1.0 + strategy_return - transaction_cost)
        
        return {
            'equity': equity,
            'positions': positions,
            'predictions': predictions,
            'position_sizes': position_size
        }
    
    def compute_metrics(self, equity, risk_free_rate=0.0):
        """
        Compute performance metrics for the strategy.
        
        Args:
            equity: Equity curve tensor
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of performance metrics
        """
        # Terminal value
        terminal_value = equity[:, -1]
        
        # Total return
        total_return = terminal_value / equity[:, 0] - 1.0
        
        # Calculate returns for metrics
        returns = equity[:, 1:] / equity[:, :-1] - 1.0
        
        # Volatility (annualized)
        volatility = returns.std(dim=1) * torch.sqrt(torch.tensor(252.0, device=self.device))
        
        # Sharpe ratio (annualized)
        mean_return = returns.mean(dim=1)
        excess_return = mean_return - risk_free_rate / 252.0  # Daily risk-free rate
        sharpe = excess_return * torch.sqrt(torch.tensor(252.0, device=self.device)) / volatility
        
        # Maximum drawdown
        cum_returns = torch.cumprod(1.0 + returns, dim=1)
        running_max = torch.maximum.accumulate(cum_returns, dim=1)
        drawdowns = running_max - cum_returns
        max_drawdown = drawdowns.max(dim=1)[0] / running_max.max(dim=1)[0]
        
        # Calmar ratio
        # CAGR = (terminal_value / equity[:, 0]) ** (252.0 / returns.shape[1]) - 1.0
        cagr = torch.pow(terminal_value / equity[:, 0], 252.0 / returns.shape[1]) - 1.0
        calmar = cagr / max_drawdown
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'calmar': calmar
        }
    
    def loss_function(self, equity, target_sharpe=3.0, target_calmar=3.0, target_drawdown=0.20):
        """
        Compute loss function for optimization.
        
        Args:
            equity: Equity curve tensor
            target_sharpe: Target Sharpe ratio
            target_calmar: Target Calmar ratio
            target_drawdown: Target maximum drawdown
            
        Returns:
            Loss tensor
        """
        metrics = self.compute_metrics(equity)
        
        # Compute components of the loss
        sharpe_loss = torch.clamp(target_sharpe - metrics['sharpe'], min=0.0)
        calmar_loss = torch.clamp(target_calmar - metrics['calmar'], min=0.0)
        drawdown_loss = torch.clamp(metrics['max_drawdown'] - target_drawdown, min=0.0)
        
        # Combined loss
        loss = sharpe_loss + 0.5 * calmar_loss + 10.0 * drawdown_loss
        
        return loss.mean()
