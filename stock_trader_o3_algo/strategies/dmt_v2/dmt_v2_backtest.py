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
    neutral_zone: float = 0.03,
    target_annual_vol: float = 0.25,
    vol_window: int = 20,
    max_position_size: float = 1.5,
    start_date: datetime = None,
    end_date: datetime = None,
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
        start_date: Start date for the backtest
        end_date: End date for the backtest
        plot: Whether to plot results
        
    Returns:
        Results DataFrame
    """
    print(f"=== Running Enhanced DMT v2 Backtest with Transformer and Balanced Parameters ===")
    print(f"Target Vol: {target_annual_vol:.1%}, Window: {vol_window}, Max Size: {max_position_size:.1%}, Neutral Zone: {neutral_zone:.2f}")
    print(f"Transformer sequence length: {seq_len}, Learning rate: {learning_rate}")
    
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime(2025, 1, 31)
        
    # Ensure dates are in datetime format
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Filter prices by date
    prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
    
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
    
    # Set up configuration with balanced hyperparameters
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
    
    # Train model
    print(f"Optimizing DMT v2 strategy for {n_epochs} epochs...")
    
    # More aggressive learning rate schedule
    initial_lr = learning_rate 
    min_lr = initial_lr * 0.1
    
    # Initialize model with better starting weights
    strategy_layer.theta_L.data = torch.tensor(0.55, device=device)
    strategy_layer.theta_S.data = torch.tensor(0.45, device=device)
    
    # Stabilize training with improved optimizer settings
    optimizer = torch.optim.Adam([
        {'params': pred_model.parameters(), 'lr': learning_rate},
        {'params': regime_classifier.parameters(), 'lr': learning_rate * 1.5},
        {'params': strategy_layer.parameters(), 'lr': learning_rate * 2.0},
    ], weight_decay=0.0001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=min_lr
    )
    
    # Improved training loop with early stopping
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state_pred = None
    best_model_state_regime = None
    best_model_state_strategy = None
    
    T = X_seq.shape[0]
    sigma_est = torch.ones(T, device=device) * target_annual_vol
    
    for epoch in range(1, n_epochs + 1):
        # Forward pass
        print(f"Debug - Preparing tensors for training:")
        print(f"X_seq shape: {X_seq.shape}")
        print(f"ret_tensor shape: {ret_tensor.shape}")
        
        # Create tensors with correct lengths
        p_t, q_lo, q_hi = pred_model(X_seq)
        
        # Fixed: Use X_seq[:, -1, :] to get the final element of each sequence for regime classifier
        # This reduces the dimensions to [432, 20]
        final_inputs = X_seq[:, -1, :]
        regime_logits = regime_classifier(final_inputs)
        
        # Debug tensor shapes
        print(f"Debug - Generated predictions:")
        print(f"p_t shape: {p_t.shape}")
        print(f"final_inputs shape: {final_inputs.shape}")
        print(f"regime_logits shape: {regime_logits.shape}")
        print(f"sigma_est shape: {sigma_est.shape}")
        
        # Initialize sigma directly with correct size to match p_t
        sigma_est = torch.ones_like(p_t) * target_annual_vol
        print(f"Debug - After reshaping:")
        print(f"p_t shape: {p_t.shape}")
        print(f"regime_logits shape: {regime_logits.shape}")
        print(f"sigma_est shape: {sigma_est.shape}")
        
        # Forward pass through strategy layer
        pos_t = strategy_layer(p_t, sigma_est, regime_logits)
        
        # Debug tensor shapes before return calculation
        print(f"Debug - Before returns calculation:")
        print(f"pos_t shape: {pos_t.shape}")
        print(f"ret_tensor shape: {ret_tensor.shape}")
        
        # Align tensors properly for returns calculation
        # We need to make sure pos_t and ret_tensor have the same dimensions
        # Using pos_t[:-1] * ret_tensor[1:] would cause us to miss the first return
        # Instead, we'll use pos_t to compute returns on the next day's price change
        aligned_pos = pos_t[:-1]  # Remove the last position since we don't have a return for it
        aligned_ret = ret_tensor[1:]  # Remove the first return since we don't have a position for it yet
        
        print(f"Debug - After alignment:")
        print(f"aligned_pos shape: {aligned_pos.shape}")
        print(f"aligned_ret shape: {aligned_ret.shape}")
        
        # Calculate log returns using the aligned tensors
        log_rets = aligned_pos * aligned_ret
        
        # Calculate portfolio equity curve (cumulative log returns)
        log_eq = torch.zeros(T)
        log_eq[1:] = torch.cumsum(log_rets, dim=0)
        
        # Calculate Sharpe ratio (mean return / std of returns)
        mu = log_rets.mean()
        sigma = log_rets.std() + 1e-6  # Add small epsilon to avoid division by zero
        sharpe = mu / sigma * np.sqrt(252.0)  # Annualize
        
        # Loss is negative Sharpe (we want to maximize Sharpe)
        loss = -sharpe
        
        # Add parameter regularization term to encourage parameter movement
        param_reg = 0.0001 * (torch.abs(strategy_layer.theta_L - 0.5) + torch.abs(strategy_layer.theta_S - 0.5))
        loss = loss - param_reg  # Subtract to encourage movement away from initialization
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        # Collect all parameters for clipping
        all_params = list(pred_model.parameters()) + list(regime_classifier.parameters()) + list(strategy_layer.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        
        optimizer.step()
        
        # Update learning rate
        scheduler.step(loss.item())
        
        # Print progress
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}, Sharpe: {-loss.item():.2f}, LogReturn: {log_eq[-1].item():.4f}")
            print(f"  Parameters - Long: {strategy_layer.theta_L.item():.3f}, Short: {strategy_layer.theta_S.item():.3f}")
            
            # Print prediction statistics for debugging
            print(f"    Predictions (p_t) - Min: {p_t.min().item():.3f}, Max: {p_t.max().item():.3f}, Mean: {p_t.mean().item():.3f}")
            print(f"    Regime logits - Mean: {regime_logits.mean(dim=0)}")
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model states
            best_model_state_pred = {k: v.clone() for k, v in pred_model.state_dict().items()}
            best_model_state_regime = {k: v.clone() for k, v in regime_classifier.state_dict().items()}
            best_model_state_strategy = {k: v.clone() for k, v in strategy_layer.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                # Restore best model states
                pred_model.load_state_dict(best_model_state_pred)
                regime_classifier.load_state_dict(best_model_state_regime)
                strategy_layer.load_state_dict(best_model_state_strategy)
                break
        
        # Record metrics
        losses.append(loss.item())
        
        # Calculate Sharpe for monitoring
        sharpe = -loss.item()  # Since loss is -Sharpe
        sharpes.append(sharpe)
        
        # Calculate log return for monitoring
        log_ret = log_eq[-1].item()
        log_returns.append(log_ret)
    
    # Final forward pass for evaluation
    print("\nRunning final evaluation with optimized DMT v2 parameters...")
    
    with torch.no_grad():
        # Get predictions
        p_t, q_lo, q_hi = pred_model(X_seq)
        
        # Get regime classifications
        final_inputs = X_seq[:, -1, :]
        regime_logits = regime_classifier(final_inputs)
        
        # Use constant volatility for simplicity in the demo
        sigma_t = torch.ones_like(p_t) * target_annual_vol
        
        # Get positions directly from the strategy layer
        positions_tensor = strategy_layer(p_t, sigma_t, regime_logits)
        
        # Convert to list for backtesting
        positions = positions_tensor.cpu().numpy().tolist()
    
    # Convert tensors to numpy arrays
    dates_np = dates
    returns_np = ret_tensor.cpu().numpy()
    
    # Initialize arrays for equity curves
    equity = [initial_capital]  # DMT v2 strategy
    baseline_equity = [initial_capital]  # Baseline long-only strategy
    buy_hold = [initial_capital]  # Buy and hold
    
    # Run backtest with optimized parameters
    for i in range(len(returns_np)):
        # DMT v2 strategy (position is already determined from the optimization)
        ret_dmt_v2 = returns_np[i] * positions[i] 
        equity.append(equity[-1] * (1 + ret_dmt_v2))
        
        # Baseline strategy (always fully invested)
        ret_baseline = returns_np[i] * 1.0  # Assumes always fully invested
        baseline_equity.append(baseline_equity[-1] * (1 + ret_baseline))
        
        # Buy and hold
        buy_hold.append(buy_hold[-1] * (1 + returns_np[i]))
    
    # Calculate performance metrics
    # DMT v2 Strategy
    total_return_dmt_v2 = equity[-1] / equity[0] - 1
    
    # Calculate days and annualized return
    days = (dates_np[-1] - dates_np[0]).days
    years = max(days / 365.25, 0.1)  # Avoid division by zero
    cagr_dmt_v2 = (equity[-1] / equity[0]) ** (1 / years) - 1
    
    # Calculate volatility and drawdown
    daily_returns_dmt_v2 = [(equity[i] / equity[i-1]) - 1 for i in range(1, len(equity))]
    vol_dmt_v2 = np.std(daily_returns_dmt_v2) * np.sqrt(252)
    
    # Calculate max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity) / peak - 1)
    max_dd_dmt_v2 = drawdown.min()
    
    # Calculate Sharpe ratio
    excess_return = cagr_dmt_v2 - 0.02  # Assuming 2% risk-free rate
    sharpe_dmt_v2 = excess_return / vol_dmt_v2 if vol_dmt_v2 > 0 else 0
    
    # Baseline Strategy
    total_return_baseline = baseline_equity[-1] / baseline_equity[0] - 1
    cagr_baseline = (baseline_equity[-1] / baseline_equity[0]) ** (1 / years) - 1
    
    daily_returns_baseline = [(baseline_equity[i] / baseline_equity[i-1]) - 1 for i in range(1, len(baseline_equity))]
    vol_baseline = np.std(daily_returns_baseline) * np.sqrt(252)
    
    peak_baseline = np.maximum.accumulate(baseline_equity)
    drawdown_baseline = (np.array(baseline_equity) / peak_baseline - 1)
    max_dd_baseline = drawdown_baseline.min()
    
    excess_return_baseline = cagr_baseline - 0.02
    sharpe_baseline = excess_return_baseline / vol_baseline if vol_baseline > 0 else 0
    
    # Print results
    print("\n=== Performance Comparison ===")
    print("DMT v2 Strategy (Optimized):")
    print(f"  Final Value:    ${equity[-1]:.2f}")
    print(f"  Total Return:   {total_return_dmt_v2:.2%}")
    print(f"  Annualized:     {cagr_dmt_v2:.2%}")
    print(f"  Volatility:     {vol_dmt_v2:.2%}")
    print(f"  Sharpe Ratio:   {sharpe_dmt_v2:.2f}")
    print(f"  Max Drawdown:   {max_dd_dmt_v2:.2%}")
    
    print("\nBaseline Strategy (Fixed Size):")
    print(f"  Final Value:    ${baseline_equity[-1]:.2f}")
    print(f"  Total Return:   {total_return_baseline:.2%}")
    print(f"  Annualized:     {cagr_baseline:.2%}")
    print(f"  Volatility:     {vol_baseline:.2%}")
    print(f"  Sharpe Ratio:   {sharpe_baseline:.2f}")
    print(f"  Max Drawdown:   {max_dd_baseline:.2%}")
    
    print("\nOptimized Parameters:")
    print(f"  Long Threshold: {strategy_layer.theta_L.item():.3f}")
    print(f"  Short Threshold: {strategy_layer.theta_S.item():.3f}")
    
    # Create results DataFrame
    results = pd.DataFrame(index=dates_np)
    
    # Adjust equity arrays if needed (they should have the same length now)
    equity_series = pd.Series(equity[1:], index=dates_np)
    baseline_series = pd.Series(baseline_equity[1:], index=dates_np)
    buy_hold_series = pd.Series(buy_hold[1:], index=dates_np)
    position_series = pd.Series(positions, index=dates_np)
    
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
    
    # Compute performance metrics
    equity_curve = results['dmt_v2_equity']
    performance_metrics = {
        'Strategy': 'DMT_v2',
        'Initial Value': equity_curve.iloc[0],
        'Final Value': equity_curve.iloc[-1],
        'Total Return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        'Period': f"{results.index.min().strftime('%Y-%m-%d')} to {results.index.max().strftime('%Y-%m-%d')}",
        'Trading Days': len(results),
        'Equity Curve': equity_curve,
    }
    
    # Calculate years for CAGR
    days = (results.index.max() - results.index.min()).days
    years = max(days / 365.25, 0.1)  # Avoid division by zero
    performance_metrics['CAGR'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    
    # Calculate volatility
    returns_pct = equity_curve.pct_change().dropna()
    performance_metrics['Volatility'] = returns_pct.std() * np.sqrt(252)
    
    # Calculate drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak - 1)
    performance_metrics['Max Drawdown'] = drawdown.min()
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02  # Assumed risk-free rate
    performance_metrics['Sharpe Ratio'] = (performance_metrics['CAGR'] - risk_free_rate) / performance_metrics['Volatility']
    
    return results, performance_metrics
