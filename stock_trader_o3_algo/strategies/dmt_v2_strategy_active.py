#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMT_v2_Strategy_Active Module
=============================
Modified version of the DMT_v2 strategy optimized for active paper trading.
This version reduces the neutral zone and modifies the market regime detection
to work better with limited historical data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
import logging

# Import original strategy components
from .dmt_v2_strategy import DMT_v2_Strategy

logger = logging.getLogger(__name__)

class DMT_v2_Strategy_Active(DMT_v2_Strategy):
    """
    Active version of the DMT_v2 strategy optimized for paper trading.
    
    This strategy uses the same core logic as DMT_v2 but with modifications
    to make it more active in paper trading environments:
    1. Reduced neutral zone
    2. Modified market regime detection for limited data
    3. More aggressive position sizing
    """
    
    def __init__(self, 
                 version: str = "turbo", 
                 asset_type: str = "equity",
                 lookback_period: int = 252,
                 initial_capital: float = 10000.0):
        """
        Initialize the active DMT_v2 strategy.
        
        Args:
            version: Strategy version ('original', 'enhanced', or 'turbo')
            asset_type: Type of asset ('equity' or 'crypto')
            lookback_period: Lookback period for calculations
            initial_capital: Starting capital for the simulation
        """
        # Initialize the parent class
        super().__init__(version, asset_type, lookback_period, initial_capital)
        
        # Override parameters to make the strategy more active
        self._set_active_parameters()
        
        logger.info(f"Initialized DMT_v2_Strategy_Active with version={version}, asset_type={asset_type}")
        
    def _set_active_parameters(self):
        """Set more aggressive parameters for active trading."""
        # Reduce neutral zone to trigger more trades
        self.neutral_zone = 0.01  # Reduced from 0.03/0.05
        
        # Increase position sizing
        if self.version == "original":
            self.max_position *= 1.5
        elif self.version == "enhanced":
            self.max_position *= 1.2
        
        # Add a small long bias for crypto
        if self.asset_type == "crypto":
            self.core_long_bias = 0.1
            
        logger.info(f"Active parameters set: neutral_zone={self.neutral_zone}, max_position={self.max_position}")
        
    def detect_market_regime_with_limited_data(self, data: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Modified market regime detection that works with limited historical data.
        
        Args:
            data: Historical price data
            
        Returns:
            Tuple of (regime, confidence_scores)
        """
        # Default to neutral with low confidence
        regime = 'Neutral'
        confidence = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 1.0}
        
        # Need at least 20 bars for minimal analysis
        if len(data) < 20:
            return regime, confidence
        
        current_price = data['Close'].iloc[-1]
        
        # 1. Check recent momentum (5-day)
        recent_returns = data['Close'].pct_change().iloc[-5:].mean() * 252
        
        # 2. Check shorter-term moving averages
        ma_short = data['Close'].rolling(window=min(20, len(data)//2)).mean().iloc[-1]
        ma_medium = data['Close'].rolling(window=min(50, len(data)-5)).mean().iloc[-1] if len(data) > 10 else ma_short
        
        # 3. Simple trend detection
        is_uptrend = current_price > ma_short > ma_medium
        is_downtrend = current_price < ma_short < ma_medium
        
        # Determine regime based on simple rules
        if is_uptrend and recent_returns > 0.05:
            regime = 'Bull'
            confidence = {'Bull': 0.7, 'Bear': 0.0, 'Neutral': 0.3}
        elif is_downtrend and recent_returns < -0.05:
            regime = 'Bear'
            confidence = {'Bull': 0.0, 'Bear': 0.7, 'Neutral': 0.3}
        elif recent_returns > 0.10:  # Strong positive momentum
            regime = 'Bull'
            confidence = {'Bull': 0.6, 'Bear': 0.0, 'Neutral': 0.4}
        elif recent_returns < -0.10:  # Strong negative momentum
            regime = 'Bear'
            confidence = {'Bull': 0.0, 'Bear': 0.6, 'Neutral': 0.4}
        else:
            # More nuanced neutral
            if recent_returns > 0:
                confidence = {'Bull': 0.4, 'Bear': 0.0, 'Neutral': 0.6}
            else:
                confidence = {'Bull': 0.0, 'Bear': 0.4, 'Neutral': 0.6}
                
        logger.info(f"Market regime with limited data: {regime} (Bull: {confidence['Bull']:.2f}, Bear: {confidence['Bear']:.2f}, Neutral: {confidence['Neutral']:.2f})")
        return regime, confidence
        
    def run_backtest(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Run a backtest of the active DMT_v2 strategy on historical data.
        
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
        min_history = max(20, min(self.lookback_period, len(data)//2))
        
        # Loop through each day and calculate signals
        for i in range(min_history, len(data)):
            # Get history up to this point
            history = data.iloc[:i+1]
            
            # Get current price and position
            current_price = history['Close'].iloc[-1]
            current_position = results['position'].iloc[i-1]
            current_equity = results['equity'].iloc[i-1]
            
            # Use our modified market regime detection for limited data
            regime, regime_confidences = self.detect_market_regime_with_limited_data(history)
                
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
            
            # Calculate signal
            if i > min_history:  # Skip the first day after min_history
                signal = self._calculate_signal(history, regime, regime_confidences, 
                                             breakout_boost=0.0, adaptive_kicker=1.0)
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
        Modified for more active trading.
        
        Args:
            history: Historical price data up to current point
            regime: Current market regime ('Bull', 'Bear', or 'Neutral')
            regime_confidences: Confidence scores for each regime
            breakout_boost: Additional signal boost for breakout conditions
            adaptive_kicker: Multiplier for signals when we're lagging buy & hold
            
        Returns:
            float: The trading signal (-max_position to +max_position)
        """
        # Get the base signal from the parent class
        signal = super()._calculate_signal(history, regime, regime_confidences, 
                                        breakout_boost, adaptive_kicker)
        
        # Make the signal more aggressive
        if signal > 0:
            signal = signal * 1.2  # Boost positive signals by 20%
        elif signal < 0:
            signal = signal * 1.2  # Boost negative signals by 20%
            
        # Apply the reduced neutral zone (already set in _set_active_parameters)
        if abs(signal) < self.neutral_zone:
            signal = 0
            
        # Ensure we don't exceed max position
        signal = max(min(signal, self.max_position), -self.max_position)
            
        return signal
