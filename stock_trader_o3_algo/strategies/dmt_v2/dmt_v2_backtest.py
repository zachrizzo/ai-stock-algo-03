#!/usr/bin/env python3
"""
DMT v2 Backtest - Implementation of the transformer-based DMT backtest.

This module provides functionality to run backtests for the DMT v2 strategy,
train the transformer models, and evaluate performance.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Optional, List, Union

from .dmt_v2_model import (
    Config, VolatilityEstimator, RegimeClassifier, 
    PredictionModel, StrategyLayer, Backtester, 
    loss_function, create_feature_matrix
)

from ..tri_shot.tri_shot_features import fetch_data_from_date


def prepare_data(prices: pd.DataFrame, window_size: int = 20) -> Tuple:
    """Prepare data for the DMT v2 model.
    
    Args:
        prices: DataFrame with price data for QQQ
        window_size: Lookback window for features
        
    Returns:
        X_tensor: Feature tensors
        y_tensor: Target tensors
        ret_tensor: Return tensors
        dates: Dates for the data points
    """
    # Get features and targets with improved NaN handling
    X, y, df = create_feature_matrix(prices, window_size, handle_nans='fill_means')
    
    # Get returns for PnL calculation
    returns = df['ret_1d'].values
    
    # Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ret_tensor = torch.tensor(returns, dtype=torch.float32)
    
    # Get dates
    dates = df.index
    
    return X_tensor, y_tensor, ret_tensor, dates, df


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


def run_dmt_v2_backtest(
    prices: pd.DataFrame,
    initial_capital: float = 500.0,
    n_epochs: int = 150,
    learning_rate: float = 0.015,
    device: str = 'cpu',
    seq_len: int = 15,
    neutral_zone: float = 0.05,
    target_annual_vol: float = 0.20,
    vol_window: int = 20,
    max_position_size: float = 1.0,
    plot: bool = True
) -> pd.DataFrame:
    """Run DMT v2 backtest with transformer model.
    
    Args:
        prices: Price data for QQQ
        initial_capital: Starting capital
        n_epochs: Optimization epochs
        learning_rate: Learning rate for optimization
        device: Torch device
        seq_len: Sequence length for transformer
        neutral_zone: Base neutral zone size
        target_annual_vol: Target annual volatility
        vol_window: Lookback window for volatility
        max_position_size: Maximum position size
        plot: Whether to plot results
        
    Returns:
        Results DataFrame
    """
    print(f"=== Running Advanced DMT v2 Backtest with Transformer ===")
    print(f"Target Vol: {target_annual_vol:.1%}, Window: {vol_window}, Max Size: {max_position_size:.1%}, Neutral Zone: {neutral_zone:.2f}")
    print(f"Transformer sequence length: {seq_len}, Learning rate: {learning_rate}")
    
    # Prepare data with enhanced feature set
    print("Preparing features and data...")
    X_tensor, y_tensor, ret_tensor, dates, df = prepare_data(prices, vol_window)
    
    # Create sequence data for transformer with longer sequence
    print("Creating sequence data for transformer...")
    X_seq = create_sequence_data(X_tensor, seq_len)
    
    # Adjust targets and returns to match sequence data
    y_tensor = y_tensor[seq_len-1:]
    ret_tensor = ret_tensor[seq_len-1:]
    dates = dates[seq_len-1:]
    
    # Move data to device
    X_seq = X_seq.to(device)
    y_tensor = y_tensor.to(device)
    ret_tensor = ret_tensor.to(device)
    
    # Set up configuration with slightly modified hyperparameters
    config = Config(
        d_model=96,
        nhead=6,
        nlayers=5,
        dropout=0.15,
        n_regimes=3,
        tau_max=target_annual_vol,
        max_pos=max_position_size,
        k0=50.0,
        lambda_sharpe=0.15,
        lambda_draw=0.08,
        lambda_turn=0.002
    )
    
    # Create models
    print(f"Creating and initializing DMT v2 model components...")
    in_dim = X_tensor.shape[1]
    
    # Prediction model (transformer)
    pred_model = PredictionModel(in_dim, config).to(device)
    
    # Regime classifier
    regime_classifier = RegimeClassifier(in_dim, config.n_regimes).to(device)
    
    # Strategy layer
    strategy_layer = StrategyLayer(config.n_regimes, config).to(device)
    
    # Initialize neutral zone bias based on input parameter
    with torch.no_grad():
        strategy_layer.nz_lin.bias.data.fill_(neutral_zone)
    
    # Backtester
    backtester = Backtester(strategy_layer, config).to(device)
    
    # Optimization
    params = list(pred_model.parameters()) + \
             list(regime_classifier.parameters()) + \
             list(strategy_layer.parameters())
    
    # Lists to store training progress
    losses = []
    sharpes = []
    log_returns = []
    
    # Training loop with improved initialization
    print(f"Optimizing DMT v2 strategy for {n_epochs} epochs...")
    
    # Higher random initialization to break symmetry
    with torch.no_grad():
        # Initialize with wider separation to break symmetry
        strategy_layer.theta_L.data = torch.tensor(0.70)
        strategy_layer.theta_S.data = torch.tensor(0.30)
        
        # Initialize regime classifier weights with some random values
        for param in regime_classifier.parameters():
            param.data = param.data + torch.randn_like(param.data) * 0.1
            
        # Initialize transformer weights with some random values
        for param in pred_model.parameters():
            if len(param.shape) > 1:  # Only randomize matrices, not biases
                param.data = param.data + torch.randn_like(param.data) * 0.1
    
    # Use higher learning rate for faster convergence and separate optimizers
    # Higher rate for threshold parameters, lower for neural network
    threshold_params = [strategy_layer.theta_L, strategy_layer.theta_S]
    nn_params = list(pred_model.parameters()) + list(regime_classifier.parameters())
    
    # Use AdamW optimizer with weight decay for better regularization
    optimizers = [
        optim.AdamW(threshold_params, lr=learning_rate * 25.0, weight_decay=0.01),
        optim.AdamW(nn_params, lr=learning_rate * 3.0, weight_decay=0.01)
    ]
    
    # Learning rate schedulers for better convergence
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=n_epochs),
        optim.lr_scheduler.CosineAnnealingLR(optimizers[1], T_max=n_epochs)
    ]
    
    for epoch in range(n_epochs):
        # Reset gradients
        for opt in optimizers:
            opt.zero_grad()
        
        # Forward pass through prediction model
        p_t, q_lo, q_hi = pred_model(X_seq)
        
        # Forward pass through regime classifier
        regime_logits = regime_classifier(X_seq[:, -1])
        
        # Estimate volatility (simplified for backtest)
        sigma_t = torch.ones_like(p_t) * target_annual_vol
        
        # Expand regime_logits to match batch size
        regime_logits_batch = regime_logits.unsqueeze(1).expand(-1, p_t.shape[0] // regime_logits.shape[0], -1)
        regime_logits_batch = regime_logits_batch.reshape(-1, config.n_regimes)
        
        # Run backtest
        log_eq, rets, turn = backtester(
            p_t.unsqueeze(0), sigma_t.unsqueeze(0),
            regime_logits_batch.unsqueeze(0), ret_tensor.unsqueeze(0)
        )
        
        # Compute loss
        loss = loss_function(log_eq, rets, turn, config)
        
        # Add a small regularization term to encourage parameter changes
        param_reg = 0.01 * (strategy_layer.theta_L - 0.55).pow(2) + 0.01 * (strategy_layer.theta_S - 0.45).pow(2)
        loss = loss - param_reg  # Subtract to encourage movement away from initialization
        
        # Backward pass and optimization
        loss.backward()
        for opt in optimizers:
            opt.step()

        # --- Add parameter clamping after optimizer step ---
        with torch.no_grad():
            strategy_layer.theta_L.clamp_(0.3, 0.7)
            strategy_layer.theta_S.clamp_(0.3, 0.7)
        # --------------------------------------------------
        
        # Step the schedulers
        for scheduler in schedulers:
            scheduler.step()

        # Record metrics
        losses.append(loss.item())
        
        # Calculate Sharpe for monitoring
        mean_ret = rets.mean(dim=1)
        std_ret = rets.std(dim=1) + config.eps
        sharpe = (mean_ret / std_ret * np.sqrt(252)).item()
        sharpes.append(sharpe)
        
        # Calculate log return for monitoring
        log_ret = (log_eq[:, -1] - log_eq[:, 0]).item()
        log_returns.append(log_ret)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Sharpe: {sharpe:.2f}, LogReturn: {log_ret:.4f}")
            print(f"  Parameters - Long: {strategy_layer.theta_L.item():.3f}, Short: {strategy_layer.theta_S.item():.3f}")
            
            # Log prediction statistics
            p_t_min = p_t.min().item()
            p_t_max = p_t.max().item()
            p_t_mean = p_t.mean().item()
            print(f"    Predictions (p_t) - Min: {p_t_min:.3f}, Max: {p_t_max:.3f}, Mean: {p_t_mean:.3f}")
    
    # Final forward pass for evaluation
    print("\nRunning final evaluation with optimized DMT v2 parameters...")
    with torch.no_grad():
        # Get predictions
        p_t, q_lo, q_hi = pred_model(X_seq)
        
        # Get regime classifications
        regime_logits = regime_classifier(X_seq[:, -1])
        
        # Use constant volatility for simplicity in the demo
        sigma_t = torch.ones_like(p_t) * target_annual_vol
        
        # Prepare regime logits for batch
        regime_logits_batch = regime_logits.unsqueeze(1).expand(-1, p_t.shape[0] // regime_logits.shape[0], -1)
        regime_logits_batch = regime_logits_batch.reshape(-1, config.n_regimes)
        
        # Get positions
        positions = [0.0]  # Start with zero position before first trade
        for i in range(len(p_t)):
            pos = strategy_layer(
                p_t[i:i+1], sigma_t[i:i+1], regime_logits_batch[i:i+1]
            ).item()
            positions.append(pos)  # Add position AFTER getting prediction
        
        # Calculate equity curve manually for demonstration
        equity = [initial_capital]
        for i in range(len(p_t)):
            ret = ret_tensor[i].item()
            pos = positions[i+1]  # Use position from positions (shifted by 1)
            
            # Transaction cost (if not the first position)
            tc = config.trans_cost * abs(pos - positions[i]) if i > 0 else config.trans_cost * abs(pos)
            
            # Calculate new equity
            new_equity = equity[-1] * (1 + pos * ret - tc)
            equity.append(new_equity)
        
        # Create baseline (equal size) for comparison
        baseline_pos = [0.0]  # Start with zero position before first trade
        for i in range(len(p_t)):
            pos = 1.0 if p_t[i].item() > 0.5 else -1.0 if p_t[i].item() < 0.5 else 0.0
            baseline_pos.append(pos)
            
        baseline_equity = [initial_capital]
        for i in range(len(ret_tensor)):
            ret = ret_tensor[i].item()
            pos = baseline_pos[i+1]  # Use position from baseline_pos (shifted by 1)
            
            # Transaction cost (if not the first position)
            tc = config.trans_cost * abs(pos - baseline_pos[i]) if i > 0 else config.trans_cost * abs(pos)
            
            # Calculate new equity
            new_equity = baseline_equity[-1] * (1 + pos * ret - tc)
            baseline_equity.append(new_equity)
        
        # Calculate buy and hold equity
        buy_hold = [initial_capital]
        for ret in ret_tensor:
            new_equity = buy_hold[-1] * (1 + ret.item())
            buy_hold.append(new_equity)
    
    # Compute performance metrics
    # DMT v2
    dmt_returns = np.diff(equity) / equity[:-1]
    dmt_ann_return = np.mean(dmt_returns) * 252
    dmt_ann_vol = np.std(dmt_returns) * np.sqrt(252)
    dmt_sharpe = dmt_ann_return / dmt_ann_vol if dmt_ann_vol != 0 else 0
    dmt_max_dd = max(1 - equity[i] / max(equity[:i+1]) for i in range(len(equity)))
    dmt_total_return = (equity[-1] / equity[0] - 1) * 100
    
    # Baseline
    baseline_returns = np.diff(baseline_equity) / baseline_equity[:-1]
    baseline_ann_return = np.mean(baseline_returns) * 252
    baseline_ann_vol = np.std(baseline_returns) * np.sqrt(252)
    baseline_sharpe = baseline_ann_return / baseline_ann_vol if baseline_ann_vol != 0 else 0
    baseline_max_dd = max(1 - baseline_equity[i] / max(baseline_equity[:i+1]) for i in range(len(baseline_equity)))
    baseline_total_return = (baseline_equity[-1] / baseline_equity[0] - 1) * 100
    
    # Print performance
    print("\n=== Performance Comparison ===")
    print(f"DMT v2 Strategy (Optimized):")
    print(f"  Final Value:    ${equity[-1]:.2f}")
    print(f"  Total Return:   {dmt_total_return:.2f}%")
    print(f"  Annualized:     {dmt_ann_return*100:.2f}%")
    print(f"  Volatility:     {dmt_ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio:   {dmt_sharpe:.2f}")
    print(f"  Max Drawdown:   {dmt_max_dd*100:.2f}%")
    
    print(f"\nBaseline Strategy (Fixed Size):")
    print(f"  Final Value:    ${baseline_equity[-1]:.2f}")
    print(f"  Total Return:   {baseline_total_return:.2f}%")
    print(f"  Annualized:     {baseline_ann_return*100:.2f}%")
    print(f"  Volatility:     {baseline_ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio:   {baseline_sharpe:.2f}")
    print(f"  Max Drawdown:   {baseline_max_dd*100:.2f}%")
    
    print(f"\nOptimized Parameters:")
    print(f"  Long Threshold: {strategy_layer.theta_L.item():.3f}")
    print(f"  Short Threshold: {strategy_layer.theta_S.item():.3f}")
    
    # Create results DataFrame (make sure lengths match)
    results = pd.DataFrame(index=dates)
    results['date'] = dates
    
    # Adjust equity arrays if needed (they should have the same length now)
    equity_series = pd.Series(equity[1:], index=dates)
    baseline_series = pd.Series(baseline_equity[1:], index=dates)
    buy_hold_series = pd.Series(buy_hold[1:], index=dates)
    position_series = pd.Series(positions[1:], index=dates)
    
    # Assign to results
    results['dmt_v2_equity'] = equity_series
    results['baseline_equity'] = baseline_series
    results['buy_hold_equity'] = buy_hold_series
    results['position'] = position_series
    
    # Plot results if requested
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Equity curves
        plt.subplot(3, 1, 1)
        plt.plot(results.index, results['dmt_v2_equity'], label='DMT v2 Strategy')
        plt.plot(results.index, results['baseline_equity'], label='Baseline Strategy')
        plt.plot(results.index, results['buy_hold_equity'], label='Buy & Hold')
        plt.title('Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Position allocation
        plt.subplot(3, 1, 2)
        plt.plot(results.index, position_series, label='DMT v2 Position', color='purple')
        plt.title('DMT v2 Position Allocation')
        plt.xlabel('Date')
        plt.ylabel('Position Size')
        plt.ylim(-1.05, 1.05)
        plt.legend()
        plt.grid(True)
        
        # Optimization progress
        plt.subplot(3, 1, 3)
        plt.plot(range(len(losses)), losses, label='Loss')
        plt.plot(range(len(sharpes)), [-s for s in sharpes], label='-Sharpe')
        plt.title('Optimization Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / -Sharpe')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), "tri_shot_data"), exist_ok=True)
        
        # Save plot
        plot_path = os.path.join(os.getcwd(), "tri_shot_data", "dmt_v2_backtest.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save results
        results_path = os.path.join(os.getcwd(), "tri_shot_data", "dmt_v2_backtest_results.csv")
        results.to_csv(results_path, index=False)
        
        print(f"\nPlot saved to {plot_path}")
        print(f"Results saved to {results_path}")
    
    return results
