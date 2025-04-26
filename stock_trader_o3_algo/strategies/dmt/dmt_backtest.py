#!/usr/bin/env python3
"""
Differentiable Market Twin (DMT) Backtest

This module implements backtesting functionality for the DMT strategy.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import datetime as dt

# Local imports
from .dmt_model import MarketTwinLSTM, load_market_twin
from .dmt_strategy import DifferentiableTriShot
from stock_trader_o3_algo.strategies.tri_shot import tri_shot_features as tsf


class SimpleVectorizedDMT(nn.Module):
    """
    A simplified fully vectorized differentiable version of the Tri-Shot strategy.
    """
    
    def __init__(self, initial_capital=500.0, device='cpu'):
        """
        Initialize the differentiable strategy.
        
        Args:
            initial_capital: Starting capital
            device: Torch device
        """
        super(SimpleVectorizedDMT, self).__init__()
        
        self.initial_capital = initial_capital
        self.device = device
        
        # Learnable strategy parameters
        self.long_threshold = nn.Parameter(torch.tensor(0.52, device=device))
        self.short_threshold = nn.Parameter(torch.tensor(0.48, device=device))
        self.position_size = nn.Parameter(torch.tensor(0.80, device=device))
        
        # Fixed parameters
        self.transaction_cost = 0.0003  # 3 basis points
        
    def forward(self, probabilities, returns):
        """
        Fully vectorized forward pass through the strategy.
        
        Args:
            probabilities: Prediction probabilities (seq_len,)
            returns: Asset returns (seq_len,)
            
        Returns:
            Dictionary with equity curve and positions
        """
        batch_size = 1
        seq_len = returns.shape[0]
        
        # Create differentiable position signals using sigmoid
        long_signals = torch.sigmoid((probabilities - self.long_threshold) * 20.0)
        short_signals = torch.sigmoid((self.short_threshold - probabilities) * 20.0)
        
        # Convert to position sizes with learned parameter
        long_positions = long_signals * self.position_size
        short_positions = short_signals * self.position_size
        
        # Ensure positions are mutually exclusive by soft competition
        total = long_positions + short_positions
        # Soft normalization with max capping at self.position_size
        scale_factor = torch.min(
            torch.ones_like(total),
            self.position_size / (total + 1e-8)
        )
        
        long_positions = long_positions * scale_factor
        short_positions = short_positions * scale_factor
        cash_positions = 1.0 - long_positions - short_positions
        
        # Stack positions
        all_positions = torch.stack([long_positions, short_positions, cash_positions], dim=1)
        
        # Calculate position changes (prepend initial position of all cash)
        init_position = torch.zeros((1, 3), device=self.device)
        init_position[0, 2] = 1.0  # Start with all cash
        prev_positions = torch.cat([init_position, all_positions[:-1]], dim=0)
        position_changes = torch.abs(all_positions - prev_positions).sum(dim=1)
        
        # Calculate transaction costs
        costs = position_changes * self.transaction_cost
        
        # Calculate strategy returns
        strategy_returns = (
            long_positions * returns + 
            short_positions * (-returns)
        )
        
        # Apply costs
        net_returns = strategy_returns - costs
        
        # Calculate cumulative returns (starting from 1.0)
        cumulative_returns = torch.cumprod(1.0 + net_returns, dim=0)
        
        # Calculate equity curve
        equity = self.initial_capital * torch.cat([torch.ones(1, device=self.device), cumulative_returns])
        
        return {
            'equity': equity,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'cash_positions': cash_positions,
            'returns': net_returns
        }
    
    def compute_metrics(self, outputs):
        """
        Compute performance metrics.
        
        Args:
            outputs: Output dictionary from forward pass
            
        Returns:
            Dictionary of performance metrics
        """
        equity = outputs['equity']
        returns = outputs['returns']
        
        # Total return
        total_return = (equity[-1] / equity[0] - 1.0).item()
        
        # Daily metrics
        mean_return = returns.mean().item()
        std_return = returns.std().item() + 1e-8  # Avoid division by zero
        
        # Annualized metrics
        annual_return = mean_return * 252
        annual_vol = std_return * np.sqrt(252)
        sharpe = annual_return / annual_vol
        
        # Max drawdown - fixed implementation
        cumulative = torch.cumprod(1.0 + returns, dim=0)
        # Calculate running maximum (peak)
        peak = torch.zeros_like(cumulative)
        peak[0] = cumulative[0]
        for i in range(1, len(cumulative)):
            peak[i] = torch.max(peak[i-1], cumulative[i])
        
        drawdown = (peak - cumulative) / peak
        max_drawdown = drawdown.max().item()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe.item() if isinstance(sharpe, torch.Tensor) else sharpe,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'annual_vol': annual_vol
        }


def run_dmt_backtest(prices: pd.DataFrame,
                     initial_capital: float = 500.0,
                     n_epochs: int = 100,
                     learning_rate: float = 0.01,
                     device: str = 'cpu'):
    """
    Run DMT backtest with simplified implementation.
    
    Args:
        prices: Price data
        initial_capital: Starting capital
        n_epochs: Optimization epochs
        learning_rate: Learning rate for optimization
        device: Torch device
        
    Returns:
        Results DataFrame
    """
    print(f"=== Running Simplified DMT Backtest ===")
    
    # Create features for traditional model
    print("Preparing features and training traditional model...")
    X, y = tsf.make_feature_matrix(prices, "QQQ")
    
    # Create traditional backtest results
    traditional_model = tsf.train_model(prices)
    probabilities = traditional_model.predict_proba(X)[:, 1]
    
    # Calculate returns series matching X
    returns = prices["QQQ"].pct_change().iloc[len(prices) - len(X):].values
    
    # Convert to tensors
    prob_tensor = torch.tensor(probabilities, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    
    # Create DMT strategy
    print(f"Creating and optimizing DMT strategy for {n_epochs} epochs...")
    dmt_strategy = SimpleVectorizedDMT(initial_capital=initial_capital, device=device)
    optimizer = torch.optim.Adam(dmt_strategy.parameters(), lr=learning_rate)
    
    # Lists to store training progress
    losses = []
    metrics = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        optimizer.zero_grad()
        outputs = dmt_strategy(prob_tensor, returns_tensor)
        equity = outputs['equity']
        
        # Calculate Sharpe ratio (our objective to maximize)
        # We'll use the net returns directly from the output
        returns_for_sharpe = outputs['returns']
        mean_return = returns_for_sharpe.mean()
        std_return = returns_for_sharpe.std() + 1e-8  # Avoid division by zero
        sharpe = mean_return / std_return * torch.sqrt(torch.tensor(252.0, device=device))
        
        # Loss is negative Sharpe (we want to maximize Sharpe)
        loss = -sharpe
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Record loss and metrics
        losses.append(loss.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Calculate metrics
            current_metrics = dmt_strategy.compute_metrics(outputs)
            metrics.append(current_metrics)
            
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Sharpe: {current_metrics['sharpe']:.2f}")
            print(f"  Parameters - Long: {dmt_strategy.long_threshold.item():.3f}, "
                  f"Short: {dmt_strategy.short_threshold.item():.3f}, "
                  f"Size: {dmt_strategy.position_size.item():.2f}")
    
    # Final forward pass for the optimized model
    print("\nRunning backtest with optimized DMT parameters...")
    with torch.no_grad():
        dmt_outputs = dmt_strategy(prob_tensor, returns_tensor)
        dmt_equity = dmt_outputs['equity'].cpu().numpy()
        dmt_long = dmt_outputs['long_positions'].cpu().numpy()
        dmt_short = dmt_outputs['short_positions'].cpu().numpy()
        dmt_cash = dmt_outputs['cash_positions'].cpu().numpy()
        
        # Run traditional strategy for comparison
        trad_strategy = SimpleVectorizedDMT(initial_capital=initial_capital, device=device)
        trad_outputs = trad_strategy(prob_tensor, returns_tensor)
        trad_equity = trad_outputs['equity'].cpu().numpy()
    
    # Calculate buy & hold equity
    dates = X.index
    price_series = prices["QQQ"].reindex(dates)
    buy_hold = initial_capital * price_series / price_series.iloc[0]
    
    # Create results DataFrame
    results = pd.DataFrame(index=dates)
    results['date'] = dates
    results['dmt_equity'] = dmt_equity[1:]  # Skip initial equity
    results['trad_equity'] = trad_equity[1:]
    results['buy_hold_equity'] = buy_hold.values
    results['long_position'] = dmt_long
    results['short_position'] = dmt_short
    results['cash_position'] = dmt_cash
    
    # Calculate final metrics
    final_dmt_metrics = dmt_strategy.compute_metrics(dmt_outputs)
    final_trad_metrics = trad_strategy.compute_metrics(trad_outputs)
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"DMT Strategy (Optimized):")
    print(f"  Final Value:    ${dmt_equity[-1]:.2f}")
    print(f"  Total Return:   {final_dmt_metrics['total_return']:.2%}")
    print(f"  Annualized:     {final_dmt_metrics['annual_return']:.2%}")
    print(f"  Volatility:     {final_dmt_metrics['annual_vol']:.2%}")
    print(f"  Sharpe Ratio:   {final_dmt_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:   {final_dmt_metrics['max_drawdown']:.2%}")
    
    print(f"\nTraditional Strategy:")
    print(f"  Final Value:    ${trad_equity[-1]:.2f}")
    print(f"  Total Return:   {final_trad_metrics['total_return']:.2%}")
    print(f"  Annualized:     {final_trad_metrics['annual_return']:.2%}")
    print(f"  Volatility:     {final_trad_metrics['annual_vol']:.2%}")
    print(f"  Sharpe Ratio:   {final_trad_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:   {final_trad_metrics['max_drawdown']:.2%}")
    
    print(f"\nOptimized Parameters:")
    print(f"  Long Threshold: {dmt_strategy.long_threshold.item():.3f}")
    print(f"  Short Threshold: {dmt_strategy.short_threshold.item():.3f}")
    print(f"  Position Size:  {dmt_strategy.position_size.item():.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Equity curves
    plt.subplot(3, 1, 1)
    plt.plot(dates, dmt_equity[1:], label='DMT Strategy (Optimized)')
    plt.plot(dates, trad_equity[1:], label='Traditional Strategy')
    plt.plot(dates, buy_hold, label='Buy & Hold')
    plt.title('Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    # Position allocation
    plt.subplot(3, 1, 2)
    plt.fill_between(dates, dmt_long, label='Long', alpha=0.7, color='green')
    plt.fill_between(dates, dmt_short, label='Short', alpha=0.7, color='red')
    plt.fill_between(dates, dmt_cash, label='Cash', alpha=0.7, color='blue')
    plt.title('DMT Position Allocation')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    plt.legend()
    plt.grid(True)
    
    # Optimization progress
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(losses)), losses)
    plt.title('Optimization Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative Sharpe)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "tri_shot_data"), exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(os.getcwd(), "tri_shot_data", "dmt_backtest_vectorized.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Ensure the first equity value matches the initial capital exactly
    if 'dmt_equity' in results.columns and not results.empty:
        results['dmt_equity'].iloc[0] = initial_capital
        
    # Save results CSV
    results_path = os.path.join(os.getcwd(), "tri_shot_data", "dmt_backtest_results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"\nPlot saved to {plot_path}")
    print(f"Results saved to {results_path}")
    
    return results
