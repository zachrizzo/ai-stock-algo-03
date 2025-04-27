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
from stock_trader_o3_algo.strategies.tri_shot.tri_shot_features import make_feature_matrix, train_model 

class SimpleVectorizedDMT(nn.Module):
    """
    A simplified fully vectorized differentiable version of the Tri-Shot strategy.
    """
    
    def __init__(self, initial_capital=500.0, device='cpu', 
                 target_annual_vol=0.20, vol_window=20, max_position_size=1.0,
                 neutral_zone: float = 0.05):
        """
        Initialize the differentiable strategy.
        
        Args:
            initial_capital: Starting capital
            device: Torch device
            target_annual_vol: Target annualized volatility for position sizing
            vol_window: Lookback window for volatility calculation
            max_position_size: Maximum allowed position size (fraction of capital)
            neutral_zone: Zone around 0.5 where trades are avoided
        """
        super(SimpleVectorizedDMT, self).__init__()
        
        self.initial_capital = initial_capital
        self.device = device
        
        # Learnable strategy parameters
        self.long_threshold = nn.Parameter(torch.tensor(0.52, device=device))
        self.short_threshold = nn.Parameter(torch.tensor(0.48, device=device))
        
        # Fixed parameters
        self.transaction_cost = 0.0003  # 3 basis points
        self.target_annual_vol = target_annual_vol
        self.vol_window = vol_window
        self.max_position_size = max_position_size
        self.sqrt_252 = np.sqrt(252) # Precompute for efficiency
        self.neutral_zone = neutral_zone
        
    def forward(self, probabilities, returns, raw_returns_series):
        """
        Fully vectorized forward pass through the strategy.
        
        Args:
            probabilities: Prediction probabilities (seq_len,)
            returns: Asset returns (seq_len,) used for PnL calculation
            raw_returns_series: Pandas Series of asset returns matching probabilities index, used for volatility calculation
            
        Returns:
            Dictionary with equity curve and positions
        """
        
        # --- Volatility Scaling Calculation ---
        rolling_vol_daily = raw_returns_series.rolling(window=self.vol_window, min_periods=self.vol_window).std()
        rolling_vol_annual = rolling_vol_daily * self.sqrt_252
        # Fill initial NaNs (from rolling window) and potential zeros
        rolling_vol_annual = rolling_vol_annual.fillna(method='bfill').fillna(self.target_annual_vol) 
        rolling_vol_annual[rolling_vol_annual < 1e-6] = self.target_annual_vol # Replace near-zero vol with target
        
        rolling_vol_tensor = torch.tensor(rolling_vol_annual.values, dtype=torch.float32, device=self.device)
        
        # Calculate dynamic position size based on target volatility
        dynamic_position_size = self.target_annual_vol / (rolling_vol_tensor + 1e-8) # Add epsilon for safety
        dynamic_position_size = torch.clamp(dynamic_position_size, 0.0, self.max_position_size)
        # --- End Volatility Scaling ---
        
        # Determine positions based on thresholds using differentiable sigmoid approximation
        k = 100.0 # Steepness factor for sigmoid - adjust as needed
        
        # Impose neutral zone around 0.5 to avoid trades when edge is weak
        long_mask = (probabilities > 0.5 + self.neutral_zone).float()
        short_mask = (probabilities < 0.5 - self.neutral_zone).float()
        
        long_signal = torch.sigmoid(k * (probabilities - self.long_threshold)) * long_mask
        short_signal = torch.sigmoid(k * (self.short_threshold - probabilities)) * short_mask  # Reversed for short side
        
        # Normalize signals and calculate net position
        # This combines signals differentiably, scaling by dynamic_position_size
        total_signal = long_signal + short_signal + 1e-8 # Add epsilon for stability
        norm_long = long_signal / total_signal
        norm_short = short_signal / total_signal
        positions = (norm_long - norm_short) * dynamic_position_size # Target position for *next* day
        
        # --- Equity Curve Calculation (Differentiable) ---
        
        # Shift positions by 1 day to represent position held *during* the return period `t`
        positions_held = torch.zeros_like(positions) # Position held during day t
        positions_held[1:] = positions[:-1] # Today's held position was decided yesterday
        # positions_held[0] is 0, meaning no position on the first day the return is calculated
        
        # Calculate position changes required to *enter* the positions_held[t]
        position_changes = torch.zeros_like(positions) # Cost incurred on day t for position held[t]
        position_changes[1:] = torch.abs(positions_held[1:] - positions_held[:-1])
        position_changes[0] = torch.abs(positions_held[0]) # Cost for initial position (if any, here it's 0)
        
        # Calculate daily net returns factor
        # Return[t] is earned on position_held[t]
        # Cost[t] is incurred for changing into position_held[t]
        daily_net_returns = positions_held * returns - position_changes * self.transaction_cost
        daily_return_factors = 1.0 + daily_net_returns
        
        # Calculate cumulative equity using cumprod
        initial_factor = torch.tensor([1.0], device=self.device)
        all_factors = torch.cat((initial_factor, daily_return_factors))
        cumulative_factors = torch.cumprod(all_factors, dim=0)
        equity = self.initial_capital * cumulative_factors # Equity curve (seq_len + 1,)
        
        # --- End Equity Curve Calculation ---

        # Calculate cash position (as fraction of equity) based on position held during the day
        cash_frac = 1.0 - torch.abs(positions_held)
        cash_positions = cash_frac # Cash fraction during day t (shape: seq_len,)

        return {
            'equity': equity, # Full equity curve (seq_len + 1,)
            'returns': daily_net_returns, # Daily net returns (seq_len,)
            'long_positions': torch.clamp(positions_held, min=0.0), # Long part of held position
            'short_positions': torch.clamp(positions_held, max=0.0).abs(), # Short part of held position
            'cash_positions': cash_positions, # Cash part of held position
            'positions': positions_held # Actual signed position held during day t
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
                     device: str = 'cpu',
                     target_annual_vol: float = 0.20, # Added param
                     vol_window: int = 20,           # Added param
                     max_position_size: float = 1.0, # Added param
                     neutral_zone: float = 0.05):    # Added param
    """
    Run DMT backtest with simplified implementation.
    
    Args:
        prices: Price data
        initial_capital: Starting capital
        n_epochs: Optimization epochs
        learning_rate: Learning rate for optimization
        device: Torch device
        target_annual_vol: Target annual volatility for position sizing
        vol_window: Lookback window for volatility calculation
        max_position_size: Maximum allowed position size (fraction of capital)
        neutral_zone: Zone around 0.5 where trades are avoided
        
    Returns:
        Results DataFrame
    """
    print(f"=== Running Simplified DMT Backtest with Vol Scaling ===")
    print(f"Target Vol: {target_annual_vol:.1%}, Window: {vol_window}, Max Size: {max_position_size:.1%}, Neutral Zone: {neutral_zone:.2f}")
    
    # Create features for traditional model
    print("Preparing features and training traditional model...")
    X, y = make_feature_matrix(prices, "QQQ")
    
    # Create traditional backtest results
    traditional_model = train_model(prices)
    probabilities = traditional_model.predict_proba(X)[:, 1]
    
    # Calculate returns series matching X (used for PnL in forward pass)
    returns = prices["QQQ"].pct_change().iloc[len(prices) - len(X):].values
    # Get raw returns series (Pandas) matching X for vol calculation
    raw_returns_series = prices["QQQ"].pct_change().iloc[len(prices) - len(X):]
    
    # Convert to tensors
    prob_tensor = torch.tensor(probabilities, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    
    # Create DMT strategy
    print(f"Creating and optimizing DMT strategy for {n_epochs} epochs...")
    dmt_strategy = SimpleVectorizedDMT(initial_capital=initial_capital, device=device,
                                     target_annual_vol=target_annual_vol,
                                     vol_window=vol_window,
                                     max_position_size=max_position_size,
                                     neutral_zone=neutral_zone)
    # Optimize only thresholds now
    optimizer = torch.optim.Adam([dmt_strategy.long_threshold, dmt_strategy.short_threshold], 
                                 lr=learning_rate)
    
    # Lists to store training progress
    losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        optimizer.zero_grad()
        # Pass raw returns series for vol scaling
        outputs = dmt_strategy(prob_tensor, returns_tensor, raw_returns_series)
        equity = outputs['equity']
        
        # Calculate Sharpe ratio (our objective to maximize)
        # We'll use the net returns directly from the output
        returns_for_sharpe = outputs['returns']
        mean_return = returns_for_sharpe.mean()
        std_return = returns_for_sharpe.std() + 1e-8  # Avoid division by zero
        sharpe = mean_return / std_return * torch.sqrt(torch.tensor(252.0, device=device))
        
        # Calculate log return for final equity growth objective
        log_return = torch.log(equity[-1] / equity[0])
        
        # Composite loss: maximize log return and Sharpe (higher is better)
        sharpe_weight = 0.1
        loss = -(log_return + sharpe_weight * sharpe)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Record loss and metrics
        losses.append(loss.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Calculate metrics
            current_metrics = dmt_strategy.compute_metrics(outputs)
            
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Sharpe: {current_metrics['sharpe']:.2f}")
            print(f"  Parameters - Long: {dmt_strategy.long_threshold.item():.3f}, "
                  f"Short: {dmt_strategy.short_threshold.item():.3f}")
    
    # Final forward pass for the optimized model
    print("\nRunning backtest with optimized DMT parameters...")
    with torch.no_grad():
        dmt_outputs = dmt_strategy(prob_tensor, returns_tensor, raw_returns_series)
        dmt_equity = dmt_outputs['equity'].cpu().numpy()
        dmt_long = dmt_outputs['long_positions'].cpu().numpy()
        dmt_short = dmt_outputs['short_positions'].cpu().numpy()
        dmt_cash = dmt_outputs['cash_positions'].cpu().numpy()
        dmt_positions = dmt_outputs['positions'].cpu().numpy() # Get actual signed positions
        
        # Run traditional strategy for comparison
        # Instantiate with parameters that effectively disable vol scaling for baseline
        trad_strategy = SimpleVectorizedDMT(initial_capital=initial_capital, device=device,
                                          target_annual_vol=1.0, # Set target high relative to typical market vol
                                          vol_window=vol_window, # Use same window 
                                          max_position_size=1.0,
                                          neutral_zone=neutral_zone)
        # Use fixed thresholds for trad comparison
        trad_strategy.long_threshold.data.fill_(0.52)  # Example fixed threshold
        trad_strategy.short_threshold.data.fill_(0.48) # Example fixed threshold
        
        # Run the forward pass for the traditional strategy
        # Use the same inputs as the optimized strategy
        trad_outputs = trad_strategy(prob_tensor, returns_tensor, raw_returns_series)
        trad_equity = trad_outputs['equity'].cpu().numpy()
        # We don't need to recalculate equity manually anymore
        
        # trad_outputs_for_metrics = {'equity': trad_eq, 'returns': trad_eq[1:] / trad_eq[:-1] - 1.0}
    
    # Calculate buy & hold equity
    dates = X.index
    price_series = prices["QQQ"].reindex(dates)
    buy_hold = initial_capital * price_series / price_series.iloc[0]
    
    # Create results DataFrame
    results = pd.DataFrame(index=dates)
    results['date'] = dates
    results['dmt_equity'] = dmt_equity[1:]  # Skip initial equity
    results['trad_equity'] = trad_equity[1:] # Trad baseline
    results['buy_hold_equity'] = buy_hold.values
    results['position'] = dmt_positions # Save actual signed position
    # results['long_position'] = dmt_long # Replaced by actual position
    # results['short_position'] = dmt_short # Replaced by actual position
    # results['cash_position'] = dmt_cash # Cash implied by 1 - abs(position)
    
    # Calculate final metrics
    final_dmt_metrics = dmt_strategy.compute_metrics(dmt_outputs)
    # Use the full outputs from the trad_strategy forward pass for metrics
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
    
    print(f"\nTraditional Strategy (Fixed Size Baseline):")
    print(f"  Final Value:    ${trad_equity[-1]:.2f}")
    print(f"  Total Return:   {final_trad_metrics['total_return']:.2%}")
    print(f"  Annualized:     {final_trad_metrics['annual_return']:.2%}")
    print(f"  Volatility:     {final_trad_metrics['annual_vol']:.2%}")
    print(f"  Sharpe Ratio:   {final_trad_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown:   {final_trad_metrics['max_drawdown']:.2%}")
    
    print(f"\nOptimized Parameters:")
    print(f"  Long Threshold: {dmt_strategy.long_threshold.item():.3f}")
    print(f"  Short Threshold: {dmt_strategy.short_threshold.item():.3f}")
    
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
    plt.plot(dates, dmt_positions, label='DMT Position (Vol Scaled)', color='purple')
    plt.title('DMT Position Allocation (Vol Scaled)')
    plt.xlabel('Date')
    plt.ylabel('Position Size (Fraction of Capital)')
    plt.ylim(-1.05, 1.05) # Keep scale consistent
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
        pass # No change needed here, equity[0] is initial_capital
        
    # Save results CSV
    results_path = os.path.join(os.getcwd(), "tri_shot_data", "dmt_backtest_results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"\nPlot saved to {plot_path}")
    print(f"Results saved to {results_path}")
    
    return results
