#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMT_v2 Strategy Module
======================
Core implementation of the Differentiable Market Twin (DMT) v2 strategy.
This module provides the primary signal generation, position sizing, and
strategy execution logic for the DMT_v2 family of strategies.

Models:
    - Original DMT_v2: Conservative momentum-based strategy
    - Enhanced DMT_v2: More aggressive with dynamic position sizing
    - TurboDMT_v2: Full-featured with market regime detection and short capabilities
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional

from .market_regime import detect_market_regime


class DMT_v2_Strategy:
    """
    Implementation of the DMT_v2 family of strategies.
    
    This strategy uses a combination of momentum signals, market regime detection,
    volatility targeting, and adaptive position sizing to generate trading signals.
    
    Versions:
        - original: Conservative momentum-based strategy
        - enhanced: More aggressive with dynamic position sizing
        - turbo: Full-featured with market regime detection and short capabilities
    """
    
    def __init__(self, 
                 version: str = "turbo", 
                 asset_type: str = "equity",
                 lookback_period: int = 252,
                 initial_capital: float = 10000.0):
        """
        Initialize the DMT_v2 strategy.
        
        Args:
            version: Strategy version ('original', 'enhanced', or 'turbo')
            asset_type: Type of asset ('equity' or 'crypto')
            lookback_period: Lookback period for calculations
            initial_capital: Starting capital for the simulation
        """
        self.version = version.lower()
        self.asset_type = asset_type.lower()
        self.lookback_period = lookback_period
        self.initial_capital = initial_capital
        
        # Set strategy parameters based on version and asset type
        self._set_parameters()
        
        # Initialize regime performance tracking
        self.regime_performance = {
            'Bull': {'returns': [], 'count': 0},
            'Bear': {'returns': [], 'count': 0},
            'Neutral': {'returns': [], 'count': 0}
        }
        
        # Market bias trackers
        self.bull_bias = 0.0
        self.bear_bias = 0.0
        self.adaptive_kicker = 1.0  # Multiplier for adaptive leverage
        
    def _set_parameters(self):
        """Set strategy parameters based on version and asset type."""
        # Default parameter values
        self.allow_shorting = False
        self.adaptive_signal = False
        self.lookback_window = 60  # Days for bias calculation
        
        # Base parameters by version
        if self.version == "original":
            self.neutral_zone = 0.05
            self.core_long_bias = 0.0
            
            if self.asset_type == "equity":
                self.target_vol = 0.25
                self.max_position = 1.0
            elif self.asset_type == "crypto":
                self.target_vol = 0.40
                self.max_position = 1.5
                
        elif self.version == "enhanced":
            self.neutral_zone = 0.03
            self.core_long_bias = 0.0
            
            if self.asset_type == "equity":
                self.target_vol = 0.35
                self.max_position = 2.0
            elif self.asset_type == "crypto":
                self.target_vol = 0.50
                self.max_position = 2.0
                
        elif self.version == "turbo":
            self.allow_shorting = True
            self.adaptive_signal = True
            self.neutral_zone = 0.0  # No neutral zone - always invested
            self.core_long_bias = 0.4  # Permanent long bias to never fall behind index
            
            if self.asset_type == "equity":
                self.target_vol = 0.40
                self.max_position = 3.0
            elif self.asset_type == "crypto":
                self.target_vol = 0.60
                self.max_position = 2.5
                
        else:
            raise ValueError(f"Unknown strategy version: {self.version}")
        
        # Volatility scaling limits
        if self.asset_type == "equity":
            self.min_vol_scalar = 0.3
            self.max_vol_scalar = 3.5
        elif self.asset_type == "crypto":
            self.min_vol_scalar = 0.2
            self.max_vol_scalar = 2.5
        
    def run_backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Run a backtest of the DMT_v2 strategy on historical data.
        
        Args:
            data: Historical price data (OHLCV)
            
        Returns:
            DataFrame with backtest results and activity rate
        """
        # Initialize results DataFrame
        results = pd.DataFrame(index=data.index)
        results['Close'] = data['Close']
        results['returns'] = data['Close'].pct_change()
        results['signal'] = 0.0
        results['position'] = 0.0
        results['equity'] = self.initial_capital
        results['cash'] = self.initial_capital
        results['regime'] = 'Neutral'
        results['bull_confidence'] = 0.0
        results['bear_confidence'] = 0.0
        results['neutral_confidence'] = 0.0
        results['market_state'] = None
        results['adaptive_kicker'] = 1.0
        
        # Buy and hold strategy for comparison
        results['buy_hold_equity'] = self.initial_capital * (1 + results['returns']).cumprod().fillna(1)
        
        # Minimum history needed for calculations
        min_history = max(self.lookback_period, 20)
        
        # Loop through each day and calculate signals
        for i in range(min_history, len(data)):
            # Get history up to this point
            history = data.iloc[:i+1]
            
            # Get current price and position
            current_price = history['Close'].iloc[-1]
            current_position = results['position'].iloc[i-1]
            current_equity = results['equity'].iloc[i-1]
            
            # For enhanced versions, detect market regime
            regime = 'Neutral'
            regime_confidences = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 1.0}
            
            if self.version in ["enhanced", "turbo"]:
                regime, regime_confidences = detect_market_regime(history, lookback_period=40)
                
                # Store regime detection results
                results.loc[results.index[i], 'regime'] = regime
                results.loc[results.index[i], 'bull_confidence'] = regime_confidences['Bull']
                results.loc[results.index[i], 'bear_confidence'] = regime_confidences['Bear']
                results.loc[results.index[i], 'neutral_confidence'] = regime_confidences['Neutral']
                results.loc[results.index[i], 'market_state'] = regime
                
                # For backward compatibility with bear market columns
                if regime == 'Bear':
                    results.loc[results.index[i], 'bear_market'] = True
                else:
                    results.loc[results.index[i], 'bear_market'] = False
            
            # Calculate statistical properties
            returns = history['Close'].pct_change().dropna()
            
            # Calculate average return and volatility
            avg_return = returns.mean()
            volatility = returns.std()
            
            # Adjust for annualization (assuming daily data)
            annual_avg_return = avg_return * 252
            annual_volatility = volatility * np.sqrt(252)
            
            # Check for breakout (if TurboDMT_v2)
            breakout_boost = 0.0
            if self.version == "turbo":
                # Check if we're at a 20-day high
                if i >= 20 and current_price >= history['Close'].iloc[-20:].max():
                    breakout_boost = 0.3
            
            # Check performance vs buy & hold over last 30 days
            if i >= 30 and self.version == "turbo":
                recent_strategy_return = results['equity'].iloc[i-30:i].pct_change().sum()
                recent_bh_return = results['buy_hold_equity'].iloc[i-30:i].pct_change().sum()
                
                # If we're lagging buy & hold by >1%, boost our signals
                if recent_strategy_return < recent_bh_return - 0.01:
                    adaptive_kicker = 1.2  # Boost signals by 20%
                else:
                    adaptive_kicker = 1.0
                    
                results.loc[results.index[i], 'adaptive_kicker'] = adaptive_kicker
            else:
                adaptive_kicker = 1.0
            
            # Calculate signal
            if i > min_history:  # Skip the first day after min_history
                signal = self._calculate_signal(history, regime, regime_confidences, 
                                              breakout_boost, adaptive_kicker)
            else:
                # Not enough history, no signal
                signal = 0.0
            
            # Store signal and position
            results.loc[results.index[i], 'signal'] = signal
            results.loc[results.index[i], 'position'] = signal
            
            # Calculate returns for this time step
            day_return = results['returns'].iloc[i]
            position_return = current_position * day_return
            
            # Update equity
            new_equity = current_equity * (1 + position_return)
            results.loc[results.index[i], 'equity'] = new_equity
            
            # Update cash (not used in this simplified model but kept for future enhancements)
            results.loc[results.index[i], 'cash'] = new_equity
            
            # Update regime performance tracking for adaptive signal
            if i > self.lookback_window and self.version == "turbo":
                self._update_regime_performance(results, data, i)
        
        # Calculate daily strategy returns
        results['strategy_returns'] = results['equity'].pct_change()
        
        # Additional statistics
        rolling_vol = results['strategy_returns'].rolling(window=20).std() * np.sqrt(252)
        results['rolling_volatility'] = rolling_vol
        
        # Calculate activity rate
        activity_rate = np.sum(np.abs(results['position'].diff()) > 0.01) / len(results)
        
        print(f"  Average position size: {results['position'].abs().mean():.2f}")
        
        return results, activity_rate
    
    def _calculate_signal(self, 
                         history: pd.DataFrame, 
                         regime: str, 
                         regime_confidences: Dict[str, float],
                         breakout_boost: float = 0.0,
                         adaptive_kicker: float = 1.0) -> float:
        """
        Calculate the trading signal based on market conditions.
        
        Args:
            history: Historical price data up to current point
            regime: Current market regime ('Bull', 'Bear', or 'Neutral')
            regime_confidences: Confidence scores for each regime
            breakout_boost: Additional signal boost for breakout conditions
            adaptive_kicker: Multiplier for signals when we're lagging buy & hold
            
        Returns:
            float: The trading signal (-max_position to +max_position)
        """
        # Calculate momentum metrics for different time frames
        momentum_5d = history['Close'].iloc[-1] / history['Close'].iloc[-6] - 1
        momentum_10d = history['Close'].iloc[-1] / history['Close'].iloc[-11] - 1
        momentum_20d = history['Close'].iloc[-1] / history['Close'].iloc[-21] - 1
        momentum_50d = history['Close'].iloc[-1] / history['Close'].iloc[-51] - 1
        
        # Base signal calculation
        signal = 0.0
        
        # Regime-specific signal calculation
        if self.version == "turbo":
            if regime == 'Bull':
                # In bull regimes, emphasize short-term momentum
                signal = 0.45 * momentum_5d + 0.30 * momentum_10d + 0.15 * momentum_20d + 0.10 * momentum_50d
                
                # Apply bull bias
                signal = max(0, signal) * (1 + self.bull_bias) + min(0, signal) * (1 - self.bull_bias)
                
            elif regime == 'Bear':
                # In bear regimes, emphasize medium-term momentum and consider inverse signals
                raw_signal = 0.15 * momentum_5d + 0.25 * momentum_10d + 0.40 * momentum_20d + 0.20 * momentum_50d
                
                # Blend normal and inverse signal based on bear confidence
                inverse_weight = min(0.8, regime_confidences['Bear'] * 1.2)
                normal_weight = 1.0 - inverse_weight
                
                signal = (raw_signal * normal_weight) + (-raw_signal * inverse_weight)
                
                # Apply bear bias
                signal = max(0, signal) * (1 - self.bear_bias) + min(0, signal) * (1 + self.bear_bias)
                
            else:  # Neutral regime
                # In neutral regimes, use a balanced approach
                signal = 0.25 * momentum_5d + 0.25 * momentum_10d + 0.25 * momentum_20d + 0.25 * momentum_50d
                
            # Add breakout boost if applicable
            signal += breakout_boost
                
        else:
            # For original and enhanced versions, use standard weighting
            if self.version == "original":
                signal = 0.6 * momentum_10d + 0.4 * momentum_50d
            else:  # enhanced
                signal = 0.4 * momentum_5d + 0.3 * momentum_10d + 0.2 * momentum_20d + 0.1 * momentum_50d
        
        # Apply adaptive kicker if we're lagging buy & hold
        signal *= adaptive_kicker
        
        # Add core long bias (only for turbo)
        if self.version == "turbo":
            signal += self.core_long_bias
        
        # Normalize signal to a -1 to 1 scale based on historical distribution
        if len(history) > 60:
            # Get historical momentum
            hist_momentum = pd.Series([
                history['Close'].iloc[j] / history['Close'].iloc[j-10] - 1
                for j in range(max(51, len(history)-100), len(history)) if j >= 10
            ])
            
            # Calculate scaling factor based on 90th percentile
            if not hist_momentum.empty:
                pos_scale = max(0.02, np.percentile(hist_momentum, 90)) if np.percentile(hist_momentum, 90) > 0 else 0.02
                neg_scale = min(-0.02, np.percentile(hist_momentum, 10)) if np.percentile(hist_momentum, 10) < 0 else -0.02
                
                # Normalize signal based on historical distribution
                if signal > 0:
                    signal = min(1.0, signal / pos_scale)
                elif signal < 0:
                    signal = max(-1.0, signal / neg_scale)
        
        # Apply neutral zone
        if abs(signal) < self.neutral_zone:
            signal = 0
        
        # Scale signal by target volatility / actual volatility
        volatility = history['Close'].pct_change().std() * np.sqrt(252)
        if volatility > 0:
            vol_scalar = self.target_vol / volatility
            
            # Limit how much we can scale the signal
            vol_scalar = min(self.max_vol_scalar, max(self.min_vol_scalar, vol_scalar))
            
            # Apply the scaling
            signal *= vol_scalar
        
        # Apply regime-specific position sizing (for turbo version)
        if self.version == "turbo":
            # In bull regimes, be more aggressive with long positions
            if regime == 'Bull':
                bull_max = self.max_position * (1 + 0.6 * regime_confidences['Bull'])
                bear_max = self.max_position * 0.3  # Very limited shorts in bull regimes
                
                if signal > 0:
                    signal = min(signal, bull_max)
                else:
                    signal = max(signal, -bear_max)
                    
            # In bear regimes, be more aggressive with short positions
            elif regime == 'Bear':
                bull_max = self.max_position * 0.3  # Minimal longs allowed in bear regimes
                bear_max = self.max_position * (1 + 0.6 * regime_confidences['Bear'])
                
                if signal > 0:
                    signal = min(signal, bull_max)
                else:
                    signal = max(signal, -bear_max)
                    
            # In neutral regimes, use standard position sizing
            else:
                neutral_cap = self.max_position * 0.7
                signal = max(min(signal, neutral_cap), -neutral_cap)
        else:
            # For original and enhanced versions, use standard position sizing
            signal = max(min(signal, self.max_position), -self.max_position)
            
        return signal
    
    def _update_regime_performance(self, results: pd.DataFrame, data: pd.DataFrame, current_idx: int):
        """
        Update regime performance tracking for adaptive signal adjustment.
        
        Args:
            results: Backtest results dataframe
            data: Historical price data
            current_idx: Current index in the simulation
        """
        # Calculate recent performance in each regime
        for j in range(current_idx - self.lookback_window, current_idx):
            day_regime = results['regime'].iloc[j]
            if j+1 < len(data):  # Ensure we don't go out of bounds
                next_return = data['Close'].iloc[j+1] / data['Close'].iloc[j] - 1
                
                if day_regime in self.regime_performance:
                    # Keep only the most recent 20 returns
                    if len(self.regime_performance[day_regime]['returns']) >= 20:
                        self.regime_performance[day_regime]['returns'].pop(0)
                    
                    self.regime_performance[day_regime]['returns'].append(next_return)
                    self.regime_performance[day_regime]['count'] += 1
        
        # Calculate average performance by regime
        bull_perf = np.mean(self.regime_performance['Bull']['returns']) if self.regime_performance['Bull']['returns'] else 0
        bear_perf = np.mean(self.regime_performance['Bear']['returns']) if self.regime_performance['Bear']['returns'] else 0
        
        # Update bias based on recent performance
        self.bull_bias = max(0, min(0.5, bull_perf * 20))  # Cap at 0.5 (50% bias)
        self.bear_bias = max(0, min(0.5, -bear_perf * 20))  # Cap at 0.5 (50% bias)
