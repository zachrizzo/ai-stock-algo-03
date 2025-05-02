#!/usr/bin/env python3
"""
TurboDMT_v2 Risk Management
===========================
Advanced risk management and position sizing system with dynamic leverage adjustment,
regime-dependent position sizing, and drawdown protection mechanisms.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class RiskParameters:
    """Risk management parameters for TurboDMT_v2"""
    # Basic position sizing
    max_position_size: float = 3.0
    min_position_size: float = 0.1
    target_vol: float = 0.35
    neutral_zone: float = 0.02
    
    # Drawdown protection
    max_drawdown_threshold: float = 0.15  # 15% max drawdown limit
    drawdown_reduction_factor: float = 0.7  # Reduce position by 70% at max drawdown
    recovery_factor: float = 0.5  # Factor controlling recovery speed
    
    # Volatility scaling
    vol_scaling_factor: float = 1.0  # Base scaling factor
    vol_cap: float = 0.40  # Cap on annualized volatility (40%)
    vol_floor: float = 0.05  # Floor on annualized volatility (5%)
    
    # Confidence scaling
    confidence_scaling_power: float = 1.5  # Power applied to confidence score
    min_confidence_threshold: float = 0.55  # Minimum confidence required
    
    # Regime adjustment
    bull_regime_multiplier: float = 1.2  # Increase positions in bull regimes
    bear_regime_multiplier: float = 0.7  # Decrease positions in bear regimes
    
    # Stop-loss / Take-profit
    use_dynamic_stops: bool = True
    stop_loss_atr_multiple: float = 2.5  # Stop at 2.5x ATR
    take_profit_atr_multiple: float = 5.0  # Take profit at 5x ATR
    trailing_stop_activation: float = 0.03  # Activate trailing stop after 3% gain
    trailing_stop_distance: float = 0.02  # Trail by 2%
    
    # Risk limits
    max_concentration: float = 0.25  # Max allocation to single position (25%)
    max_correlation_exposure: float = 0.4  # Max exposure to correlated assets
    stress_test_threshold: float = 0.2  # Maximum allowable loss in stress test
    
    # VaR/CVaR limits
    var_confidence_level: float = 0.95  # VaR confidence level (95%)
    var_limit: float = 0.03  # 3% daily VaR limit
    cvar_limit: float = 0.05  # 5% daily CVaR limit


class DynamicRiskManager:
    """Advanced risk management system with dynamic adjustment based on market conditions"""
    
    def __init__(
        self, 
        params: RiskParameters = None, 
        use_pytorch: bool = True
    ):
        self.params = params if params is not None else RiskParameters()
        self.use_pytorch = use_pytorch
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
        self.position_history = []
        self.regime_history = []
        self.vol_history = []
        self.confidence_history = []
        self.stop_losses = {}
        self.take_profits = {}
        self.trailing_stops = {}
    
    def calculate_position_size(
        self,
        prediction: float,  # From 0 to 1, mapped to -1 to 1 for direction
        volatility: float,  # Annualized volatility estimate
        confidence: float,  # Prediction confidence or certainty, from 0 to 1
        regime_probs: List[float],  # Probabilities for [bull, neutral, bear] regimes
        current_equity: float,  # Current equity value
        current_position: float = 0.0,  # Current position size (for smoothing)
        atr: Optional[float] = None,  # Average True Range for stop-loss calculation
        correlation_matrix: Optional[pd.DataFrame] = None,  # Asset correlation matrix
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate the optimal position size based on prediction, risk factors, and market regime
        
        Returns:
            Dict containing:
                - position_size: Final position size (-3.0 to +3.0)
                - raw_signal: Raw signal before adjustments
                - volatility_adjustment: Adjustment factor from volatility scaling
                - drawdown_adjustment: Adjustment factor from drawdown protection
                - regime_adjustment: Adjustment factor from market regime
                - confidence_adjustment: Adjustment factor from prediction confidence
                - stop_loss: Optional stop-loss price
                - take_profit: Optional take-profit price
        """
        # Convert prediction from [0, 1] to [-1, 1]
        signal = (prediction - 0.5) * 2
        
        # Apply neutral zone
        if abs(signal) < self.params.neutral_zone:
            signal = 0.0
        
        # Calculate base position size with volatility targeting
        # Base formula: position = signal * target_vol / actual_vol
        target_vol = self.params.target_vol
        
        # Cap and floor volatility
        vol_adjusted = np.clip(volatility, self.params.vol_floor, self.params.vol_cap)
        
        # Volatility scaling factor
        vol_scaling = target_vol / vol_adjusted if vol_adjusted > 0 else 0
        vol_adjustment = min(vol_scaling * self.params.vol_scaling_factor, 2.0)  # Cap scaling
        
        # Confidence adjustment
        if confidence < self.params.min_confidence_threshold:
            confidence_adjustment = 0.0  # No position if below threshold
        else:
            # Scale confidence factor non-linearly
            normalized_confidence = (confidence - self.params.min_confidence_threshold) / (1 - self.params.min_confidence_threshold)
            confidence_adjustment = normalized_confidence ** self.params.confidence_scaling_power
        
        # Update drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_drawdown = 1.0 - (current_equity / self.peak_equity)
        
        # Drawdown adjustment factor (reduce positions in drawdowns)
        max_dd = self.params.max_drawdown_threshold
        if self.current_drawdown <= 0:
            drawdown_adjustment = 1.0
        elif self.current_drawdown >= max_dd:
            drawdown_adjustment = self.params.drawdown_reduction_factor
        else:
            # Linear reduction as drawdown increases
            drawdown_adjustment = 1.0 - (1.0 - self.params.drawdown_reduction_factor) * (self.current_drawdown / max_dd)
        
        # Recovery scaling (reduce more significantly in early recovery)
        if current_equity < self.peak_equity and len(self.position_history) > 0:
            if self.position_history[-1] * signal < 0:  # Position direction change during drawdown
                recovery_adjustment = self.params.recovery_factor
                drawdown_adjustment *= recovery_adjustment
        
        # Market regime adjustment
        # regime_probs is [bull, neutral, bear]
        bull_prob, neutral_prob, bear_prob = regime_probs
        
        # Calculate weighted regime multiplier
        regime_adjustment = (
            bull_prob * self.params.bull_regime_multiplier +
            neutral_prob * 1.0 +
            bear_prob * self.params.bear_regime_multiplier
        )
        
        # Calculate raw position size
        raw_position = signal * vol_adjustment * confidence_adjustment * regime_adjustment * drawdown_adjustment
        
        # Apply maximum position constraint
        position_size = np.clip(raw_position, -self.params.max_position_size, self.params.max_position_size)
        
        # Calculate stop-loss and take-profit levels if ATR is provided
        stop_loss = None
        take_profit = None
        trailing_stop = None
        
        if atr is not None and self.params.use_dynamic_stops and abs(position_size) > 0:
            # Current price is assumed to be close price from the latest data
            current_price = kwargs.get('current_price', 100.0)
            
            if position_size > 0:  # Long position
                stop_loss = current_price - (atr * self.params.stop_loss_atr_multiple)
                take_profit = current_price + (atr * self.params.take_profit_atr_multiple)
            elif position_size < 0:  # Short position
                stop_loss = current_price + (atr * self.params.stop_loss_atr_multiple)
                take_profit = current_price - (atr * self.params.take_profit_atr_multiple)
        
        # Store position and adjustments in history
        self.position_history.append(position_size)
        self.regime_history.append(regime_probs)
        self.vol_history.append(volatility)
        self.confidence_history.append(confidence)
        
        # Return position size and all adjustment factors
        return {
            'position_size': position_size,
            'raw_signal': signal,
            'volatility_adjustment': vol_adjustment,
            'drawdown_adjustment': drawdown_adjustment,
            'regime_adjustment': regime_adjustment,
            'confidence_adjustment': confidence_adjustment,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'current_drawdown': self.current_drawdown,
        }
    
    def update_stops(
        self,
        symbol: str,
        current_price: float,
        position_size: float,
        atr: float
    ) -> Dict[str, float]:
        """
        Update stop-loss, take-profit, and trailing stop levels for an active position
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_size: Current position size
            atr: Average True Range
            
        Returns:
            Updated stop levels
        """
        if abs(position_size) <= 0 or not self.params.use_dynamic_stops:
            # Clear stops if position is closed
            self.stop_losses.pop(symbol, None)
            self.take_profits.pop(symbol, None)
            self.trailing_stops.pop(symbol, None)
            return {'stop_loss': None, 'take_profit': None, 'trailing_stop': None}
        
        # Get existing stop levels
        stop_loss = self.stop_losses.get(symbol, None)
        take_profit = self.take_profits.get(symbol, None)
        trailing_stop = self.trailing_stops.get(symbol, None)
        
        # Initialize stops if not already set
        if stop_loss is None:
            if position_size > 0:  # Long position
                stop_loss = current_price - (atr * self.params.stop_loss_atr_multiple)
            else:  # Short position
                stop_loss = current_price + (atr * self.params.stop_loss_atr_multiple)
            
            self.stop_losses[symbol] = stop_loss
        
        if take_profit is None:
            if position_size > 0:  # Long position
                take_profit = current_price + (atr * self.params.take_profit_atr_multiple)
            else:  # Short position
                take_profit = current_price - (atr * self.params.take_profit_atr_multiple)
                
            self.take_profits[symbol] = take_profit
        
        # Update trailing stop if needed
        activation_threshold = self.params.trailing_stop_activation
        
        if position_size > 0:  # Long position
            profit_pct = (current_price / self.stop_losses.get(symbol, current_price)) - 1
            
            # Activate trailing stop once profit exceeds threshold
            if profit_pct > activation_threshold:
                new_trailing_stop = current_price * (1 - self.params.trailing_stop_distance)
                
                # Update trailing stop only if it's higher than the existing one
                if trailing_stop is None or new_trailing_stop > trailing_stop:
                    trailing_stop = new_trailing_stop
                    self.trailing_stops[symbol] = trailing_stop
                    
                    # Also update stop-loss if trailing stop is higher
                    if trailing_stop > self.stop_losses.get(symbol, 0):
                        self.stop_losses[symbol] = trailing_stop
        
        elif position_size < 0:  # Short position
            profit_pct = 1 - (current_price / self.stop_losses.get(symbol, current_price))
            
            # Activate trailing stop once profit exceeds threshold
            if profit_pct > activation_threshold:
                new_trailing_stop = current_price * (1 + self.params.trailing_stop_distance)
                
                # Update trailing stop only if it's lower than the existing one
                if trailing_stop is None or new_trailing_stop < trailing_stop:
                    trailing_stop = new_trailing_stop
                    self.trailing_stops[symbol] = trailing_stop
                    
                    # Also update stop-loss if trailing stop is lower
                    if trailing_stop < self.stop_losses.get(symbol, float('inf')):
                        self.stop_losses[symbol] = trailing_stop
        
        return {
            'stop_loss': self.stop_losses.get(symbol, None),
            'take_profit': self.take_profits.get(symbol, None),
            'trailing_stop': self.trailing_stops.get(symbol, None),
        }
    
    def calculate_var_cvar(
        self,
        returns: np.ndarray,
        position_size: float,
        confidence_level: float = None
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        Args:
            returns: Historical return series
            position_size: Current position size
            confidence_level: VaR confidence level (default: from parameters)
            
        Returns:
            Tuple of (VaR, CVaR) as positive percentages
        """
        if confidence_level is None:
            confidence_level = self.params.var_confidence_level
        
        # Scale returns by position size
        scaled_returns = returns * abs(position_size)
        
        # For short positions, negate the returns
        if position_size < 0:
            scaled_returns = -scaled_returns
        
        # Calculate VaR
        var = np.percentile(scaled_returns, 100 * (1 - confidence_level))
        
        # Calculate CVaR (expected shortfall)
        cvar = scaled_returns[scaled_returns <= var].mean()
        
        # Return as positive percentages
        return -var, -cvar
    
    def stress_test(
        self,
        position_size: float,
        volatility: float,
        stress_scenarios: List[float] = None
    ) -> Dict[str, float]:
        """
        Perform stress testing on the current position
        
        Args:
            position_size: Current position size
            volatility: Current volatility estimate
            stress_scenarios: List of market move scenarios in percentage terms
                              (default: [-10%, -7%, -5%, -3%, 3%, 5%, 7%, 10%])
            
        Returns:
            Dictionary with stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = [-0.10, -0.07, -0.05, -0.03, 0.03, 0.05, 0.07, 0.10]
        
        results = {}
        
        # Calculate potential P&L for each scenario
        for scenario in stress_scenarios:
            # For long positions, positive market moves = positive P&L
            # For short positions, positive market moves = negative P&L
            pnl = scenario * position_size
            results[f"{scenario:.1%}"] = pnl
        
        # Calculate worst-case scenario
        if position_size > 0:
            worst_case = min(stress_scenarios) * position_size
        else:
            worst_case = max(stress_scenarios) * (-position_size)
        
        results["worst_case"] = worst_case
        
        # Calculate VaR-like stress metric (2-sigma event)
        var_stress = 2 * volatility * abs(position_size)
        if position_size != 0:
            var_stress *= (-1 if position_size > 0 else 1)
        
        results["var_stress"] = var_stress
        
        return results
    
    def check_risk_limits(
        self,
        position_size: float,
        var: float,
        cvar: float,
        correlation_exposure: Optional[float] = None,
        stress_test_result: Optional[float] = None
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if the position exceeds any risk limits
        
        Args:
            position_size: Current position size
            var: Value at Risk
            cvar: Conditional Value at Risk
            correlation_exposure: Exposure to correlated assets
            stress_test_result: Worst-case stress test result
            
        Returns:
            Tuple of (overall_ok, detailed_limits) where detailed_limits is a dict
            mapping limit names to boolean values (True = within limit)
        """
        limits = {}
        
        # Check position concentration limit
        limits["concentration"] = abs(position_size) <= self.params.max_concentration
        
        # Check VaR limit
        limits["var"] = var <= self.params.var_limit
        
        # Check CVaR limit
        limits["cvar"] = cvar <= self.params.cvar_limit
        
        # Check correlation exposure if provided
        if correlation_exposure is not None:
            limits["correlation"] = correlation_exposure <= self.params.max_correlation_exposure
        
        # Check stress test result if provided
        if stress_test_result is not None:
            limits["stress_test"] = abs(stress_test_result) <= self.params.stress_test_threshold
        
        # Overall check - all limits must be satisfied
        overall_ok = all(limits.values())
        
        return overall_ok, limits
    
    def adjust_for_correlation(
        self,
        position_size: float,
        correlation_matrix: pd.DataFrame,
        current_positions: Dict[str, float],
        symbol: str
    ) -> float:
        """
        Adjust position size based on correlation with existing positions
        
        Args:
            position_size: Proposed position size
            correlation_matrix: Asset correlation matrix
            current_positions: Dictionary mapping symbols to current position sizes
            symbol: Current trading symbol
            
        Returns:
            Adjusted position size
        """
        # If no other positions or no correlation data, return original size
        if not current_positions or correlation_matrix is None:
            return position_size
        
        # Skip if symbol not in correlation matrix
        if symbol not in correlation_matrix.columns:
            return position_size
        
        # Calculate weighted correlation exposure
        total_correlation = 0
        total_exposure = 0
        
        for other_symbol, other_size in current_positions.items():
            # Skip if other symbol not in correlation matrix or is the same symbol
            if other_symbol not in correlation_matrix.columns or other_symbol == symbol:
                continue
            
            # Get correlation between the two assets
            correlation = correlation_matrix.loc[symbol, other_symbol]
            
            # Add to weighted correlation sum
            total_correlation += abs(correlation * other_size)
            total_exposure += abs(other_size)
        
        # Calculate average correlation
        avg_correlation = total_correlation / total_exposure if total_exposure > 0 else 0
        
        # Calculate correlation adjustment factor (reduce position for high correlation)
        correlation_factor = 1.0 - (avg_correlation * 0.5)  # Reduce by up to 50% for perfect correlation
        
        # Apply adjustment factor
        adjusted_size = position_size * correlation_factor
        
        return adjusted_size


class RiskManagerTorch(nn.Module):
    """PyTorch implementation of risk management for use within neural network models"""
    
    def __init__(self, params: RiskParameters = None):
        super().__init__()
        self.params = params if params is not None else RiskParameters()
        
        # Learnable parameters
        self.max_pos = nn.Parameter(torch.tensor(self.params.max_position_size))
        self.vol_scaling = nn.Parameter(torch.tensor(self.params.vol_scaling_factor))
        self.regime_bull = nn.Parameter(torch.tensor(self.params.bull_regime_multiplier))
        self.regime_bear = nn.Parameter(torch.tensor(self.params.bear_regime_multiplier))
        self.drawdown_factor = nn.Parameter(torch.tensor(self.params.drawdown_reduction_factor))
        self.neutral_zone = nn.Parameter(torch.tensor(self.params.neutral_zone))
    
    def forward(
        self,
        predictions: torch.Tensor,  # (batch_size, 1)
        volatility: torch.Tensor,  # (batch_size, 1)
        confidence: torch.Tensor,  # (batch_size, 1)
        regime_logits: torch.Tensor,  # (batch_size, 3)
        equity_curve: Optional[torch.Tensor] = None,  # (batch_size, 1)
    ) -> torch.Tensor:
        """
        Calculate position sizes based on model outputs
        
        Args:
            predictions: Trading signals (0-1)
            volatility: Volatility predictions
            confidence: Prediction confidences (0-1)
            regime_logits: Logits for market regimes (bullish, neutral, bearish)
            equity_curve: Optional equity curve for drawdown calculation
            
        Returns:
            Position sizes (-max_pos to +max_pos)
        """
        batch_size = predictions.shape[0]
        
        # Convert prediction from [0, 1] to [-1, 1]
        signal = (predictions - 0.5) * 2.0
        
        # Apply neutral zone
        neutral_mask = torch.abs(signal) < self.neutral_zone
        signal = torch.where(neutral_mask, torch.zeros_like(signal), signal)
        
        # Get target volatility and cap volatility
        target_vol = torch.tensor(self.params.target_vol, device=signal.device)
        vol_capped = torch.clamp(volatility, 
                                min=self.params.vol_floor, 
                                max=self.params.vol_cap)
        
        # Volatility scaling
        vol_scaling = torch.where(
            vol_capped > 0,
            target_vol / vol_capped,
            torch.zeros_like(vol_capped)
        ) * self.vol_scaling
        vol_scaling = torch.clamp(vol_scaling, max=2.0)
        
        # Confidence scaling
        confidence_threshold = torch.tensor(self.params.min_confidence_threshold, 
                                         device=signal.device)
        normalized_confidence = torch.clamp(
            (confidence - confidence_threshold) / (1.0 - confidence_threshold),
            min=0.0
        )
        confidence_scaling = normalized_confidence ** self.params.confidence_scaling_power
        
        # Regime scaling
        regime_probs = F.softmax(regime_logits, dim=1)
        bull_prob, neutral_prob, bear_prob = regime_probs[:, 0], regime_probs[:, 1], regime_probs[:, 2]
        
        regime_scaling = (
            bull_prob * self.regime_bull +
            neutral_prob * 1.0 +
            bear_prob * self.regime_bear
        ).unsqueeze(1)
        
        # Drawdown adjustment (if equity curve provided)
        drawdown_scaling = torch.ones_like(signal)
        if equity_curve is not None:
            # Calculate drawdown
            peak_equity = torch.cummax(equity_curve, dim=0)[0]
            drawdown = 1.0 - (equity_curve / peak_equity)
            
            # Apply drawdown scaling
            max_dd = torch.tensor(self.params.max_drawdown_threshold, device=signal.device)
            drawdown_ratio = torch.clamp(drawdown / max_dd, max=1.0)
            drawdown_scaling = 1.0 - (1.0 - self.drawdown_factor) * drawdown_ratio
        
        # Calculate raw position size
        raw_position = signal * vol_scaling * confidence_scaling * regime_scaling * drawdown_scaling
        
        # Clip to max position
        position_size = torch.clamp(raw_position, -self.max_pos, self.max_pos)
        
        return position_size


if __name__ == "__main__":
    # Simple test
    params = RiskParameters(
        max_position_size=3.0,
        target_vol=0.35,
        neutral_zone=0.02,
        max_drawdown_threshold=0.15
    )
    
    risk_manager = DynamicRiskManager(params)
    
    # Test position sizing
    result = risk_manager.calculate_position_size(
        prediction=0.75,  # Bullish prediction
        volatility=0.20,  # 20% annualized volatility
        confidence=0.80,  # 80% confidence
        regime_probs=[0.7, 0.2, 0.1],  # Mostly bullish regime
        current_equity=1.0,
        atr=0.1,  # ATR is 0.1
        current_price=100.0,
    )
    
    print("Position size calculation result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Test stress testing
    stress_test = risk_manager.stress_test(
        position_size=result['position_size'],
        volatility=0.20
    )
    
    print("\nStress test results:")
    for k, v in stress_test.items():
        print(f"  {k}: {v:.4f}")
    
    # Test VaR/CVaR calculation
    returns = np.random.normal(0.0005, 0.01, 1000)  # Generate random returns
    var, cvar = risk_manager.calculate_var_cvar(
        returns=returns,
        position_size=result['position_size']
    )
    
    print(f"\nVaR (95%): {var:.4f}")
    print(f"CVaR (95%): {cvar:.4f}")
    
    # Test PyTorch implementation
    torch_manager = RiskManagerTorch(params)
    
    # Create tensor inputs
    predictions = torch.tensor([[0.75]])
    volatility = torch.tensor([[0.20]])
    confidence = torch.tensor([[0.80]])
    regime_logits = torch.tensor([[1.0, 0.0, -1.0]])  # Bullish
    
    # Get position size
    position_size = torch_manager(predictions, volatility, confidence, regime_logits)
    print(f"\nPyTorch position size: {position_size.item():.4f}")
