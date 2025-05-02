#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Detection Module
=============================
This module provides functions for detecting market regimes (Bull, Bear, or Neutral)
based on various technical and statistical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union


def detect_market_regime(data: pd.DataFrame, lookback_period: int = 40) -> Tuple[str, Dict[str, float]]:
    """
    Enhanced function to detect market regime: Bull, Bear, or Neutral.
    Returns regime type and confidence scores.
    
    Parameters:
    - data: DataFrame with price data (must contain at least Close, and ideally High/Low/Volume)
    - lookback_period: Period for calculations
    
    Returns:
    - regime: 'Bull', 'Bear', or 'Neutral'
    - confidence: Dict with confidence scores for each regime (0-1)
    """
    if len(data) < max(lookback_period, 200):
        return 'Neutral', {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 1.0}
    
    current_price = data['Close'].iloc[-1]
    
    # 1. Check medium-term price action (percent change over lookback period)
    if lookback_period <= len(data):
        period_return = (current_price / data['Close'].iloc[-lookback_period] - 1)
    else:
        period_return = 0
    
    # 2. Check moving averages
    ma_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.0}
    
    if len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
        
        # Bullish MA signals
        price_above_ma200 = current_price > ma200
        price_above_ma50 = current_price > ma50
        ma50_above_ma200 = ma50 > ma200
        ma20_above_ma50 = ma20 > ma50
        
        # Bearish MA signals
        price_below_ma200 = current_price < ma200
        price_below_ma50 = current_price < ma50
        ma50_below_ma200 = ma50 < ma200
        ma20_below_ma50 = ma20 < ma50
        
        # Calculate MA scores
        if price_above_ma200 and ma50_above_ma200 and ma20_above_ma50:
            ma_scores['Bull'] = 1.0
        elif price_below_ma200 and ma50_below_ma200 and ma20_below_ma50:
            ma_scores['Bear'] = 1.0
        elif price_above_ma50 and ma20_above_ma50:
            ma_scores['Bull'] = 0.7
            ma_scores['Neutral'] = 0.3
        elif price_below_ma50 and ma20_below_ma50:
            ma_scores['Bear'] = 0.7
            ma_scores['Neutral'] = 0.3
        else:
            ma_scores['Neutral'] = 0.8
            if price_above_ma50:
                ma_scores['Bull'] = 0.2
            else:
                ma_scores['Bear'] = 0.2
    else:
        ma_scores['Neutral'] = 1.0
    
    # 3. Check volatility
    vol_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.0}
    
    recent_volatility = data['Close'].pct_change().iloc[-20:].std() * np.sqrt(252)
    longer_volatility = data['Close'].pct_change().iloc[-60:].std() * np.sqrt(252)
    volatility_ratio = recent_volatility / longer_volatility if longer_volatility > 0 else 1.0
    
    if volatility_ratio > 1.3:  # Significantly increasing volatility - bearish
        vol_scores['Bear'] = 0.7
        vol_scores['Neutral'] = 0.3
    elif volatility_ratio < 0.8:  # Decreasing volatility - bullish
        vol_scores['Bull'] = 0.7
        vol_scores['Neutral'] = 0.3
    else:  # Stable volatility - neutral
        vol_scores['Neutral'] = 0.6
        vol_scores['Bull'] = 0.2
        vol_scores['Bear'] = 0.2
    
    # 4. Check volume analysis
    volume_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.5}
    
    if 'Volume' in data.columns:
        # Calculate average volume on up and down days
        up_days = data[data['Close'] > data['Close'].shift(1)]
        down_days = data[data['Close'] < data['Close'].shift(1)]
        
        recent_up_volume = up_days['Volume'].iloc[-20:].mean() if len(up_days) > 0 else 0
        recent_down_volume = down_days['Volume'].iloc[-20:].mean() if len(down_days) > 0 else 0
        
        volume_ratio = recent_down_volume / recent_up_volume if recent_up_volume > 0 else 1.0
        
        if volume_ratio > 1.2:  # Higher volume on down days - bearish
            volume_scores['Bear'] = 0.7
            volume_scores['Neutral'] = 0.3
        elif volume_ratio < 0.8:  # Higher volume on up days - bullish
            volume_scores['Bull'] = 0.7
            volume_scores['Neutral'] = 0.3
    
    # 5. Check recent momentum
    momentum_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.0}
    
    recent_returns = data['Close'].pct_change().iloc[-5:].mean() * 252
    if recent_returns > 0.15:  # Strong positive momentum
        momentum_scores['Bull'] = 0.8
        momentum_scores['Neutral'] = 0.2
    elif recent_returns < -0.15:  # Strong negative momentum
        momentum_scores['Bear'] = 0.8
        momentum_scores['Neutral'] = 0.2
    elif recent_returns > 0.05:  # Moderate positive momentum
        momentum_scores['Bull'] = 0.6
        momentum_scores['Neutral'] = 0.4
    elif recent_returns < -0.05:  # Moderate negative momentum
        momentum_scores['Bear'] = 0.6
        momentum_scores['Neutral'] = 0.4
    else:  # Weak momentum
        momentum_scores['Neutral'] = 0.7
        if recent_returns > 0:
            momentum_scores['Bull'] = 0.3
        else:
            momentum_scores['Bear'] = 0.3
    
    # 6. Check distance from recent high/low
    range_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.0}
    
    highest_price = data['Close'].iloc[-60:].max()
    lowest_price = data['Close'].iloc[-60:].min()
    distance_from_high = (current_price / highest_price - 1)
    distance_from_low = (current_price / lowest_price - 1)
    price_position = (current_price - lowest_price) / (highest_price - lowest_price) if (highest_price - lowest_price) > 0 else 0.5
    
    if price_position > 0.8:  # Near recent highs - bullish
        range_scores['Bull'] = 0.7
        range_scores['Neutral'] = 0.3
    elif price_position < 0.2:  # Near recent lows - bearish
        range_scores['Bear'] = 0.7
        range_scores['Neutral'] = 0.3
    else:  # Middle of range - neutral
        range_scores['Neutral'] = 0.6
        if price_position > 0.5:
            range_scores['Bull'] = 0.4
        else:
            range_scores['Bear'] = 0.4
    
    # 7. Check trend strength using ADX-like calculation
    trend_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.5}
    
    if len(data) >= 30 and 'High' in data.columns and 'Low' in data.columns:
        # Calculate directional movement
        plus_dm = np.maximum(0, data['High'].diff())
        minus_dm = np.maximum(0, -data['Low'].diff())
        
        # Simple smoothing
        plus_di = plus_dm.rolling(14).mean()
        minus_di = minus_dm.rolling(14).mean()
        
        # Latest values
        latest_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
        latest_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
        
        # Determine trend strength
        if latest_plus_di > 1.5 * latest_minus_di:  # Strong uptrend
            trend_scores['Bull'] = 0.8
            trend_scores['Neutral'] = 0.2
        elif latest_minus_di > 1.5 * latest_plus_di:  # Strong downtrend
            trend_scores['Bear'] = 0.8
            trend_scores['Neutral'] = 0.2
        elif latest_plus_di > latest_minus_di:  # Moderate uptrend
            trend_scores['Bull'] = 0.6
            trend_scores['Neutral'] = 0.4
        elif latest_minus_di > latest_plus_di:  # Moderate downtrend
            trend_scores['Bear'] = 0.6
            trend_scores['Neutral'] = 0.4
    
    # 8. Crypto-specific indicators (if applicable)
    crypto_scores = {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 0.5}
    
    # Combine all factors with weights
    weights = {
        'ma': 0.25,         # Moving averages (most reliable long-term)
        'vol': 0.05,        # Volatility
        'volume': 0.05,     # Volume analysis
        'momentum': 0.20,   # Recent momentum (reliable short-term)
        'range': 0.15,      # Range position
        'trend': 0.30,      # Trend strength (most reliable medium-term)
        'crypto': 0.00      # Crypto-specific (only if applicable)
    }
    
    # Calculate final scores
    final_scores = {
        'Bull': (ma_scores['Bull'] * weights['ma'] + 
                vol_scores['Bull'] * weights['vol'] + 
                volume_scores['Bull'] * weights['volume'] + 
                momentum_scores['Bull'] * weights['momentum'] + 
                range_scores['Bull'] * weights['range'] + 
                trend_scores['Bull'] * weights['trend'] +
                crypto_scores['Bull'] * weights['crypto']),
        
        'Bear': (ma_scores['Bear'] * weights['ma'] + 
                vol_scores['Bear'] * weights['vol'] + 
                volume_scores['Bear'] * weights['volume'] + 
                momentum_scores['Bear'] * weights['momentum'] + 
                range_scores['Bear'] * weights['range'] + 
                trend_scores['Bear'] * weights['trend'] +
                crypto_scores['Bear'] * weights['crypto']),
        
        'Neutral': (ma_scores['Neutral'] * weights['ma'] + 
                   vol_scores['Neutral'] * weights['vol'] + 
                   volume_scores['Neutral'] * weights['volume'] + 
                   momentum_scores['Neutral'] * weights['momentum'] + 
                   range_scores['Neutral'] * weights['range'] + 
                   trend_scores['Neutral'] * weights['trend'] +
                   crypto_scores['Neutral'] * weights['crypto'])
    }
    
    # Determine market regime based on highest score
    regime = max(final_scores, key=final_scores.get)
    
    # Ensure scores sum to 1
    score_sum = sum(final_scores.values())
    normalized_scores = {k: v/score_sum for k, v in final_scores.items()}
    
    return regime, normalized_scores


def detect_bear_market(data: pd.DataFrame, lookback_period: int = 40, threshold: float = -0.10) -> Tuple[bool, float]:
    """
    Legacy function to detect bear markets based on drawdown from recent highs.
    Included for backward compatibility.
    
    Parameters:
    - data: DataFrame with price data
    - lookback_period: Period over which to check drawdown
    - threshold: Drawdown threshold for bear market detection
    
    Returns:
    - is_bear_market: Boolean indicating whether we're in a bear market
    - confidence: Confidence level (0-1) of the detection
    """
    if len(data) < lookback_period:
        return False, 0.0
    
    # Calculate drawdown from recent high
    recent_high = data['Close'].rolling(window=lookback_period).max().iloc[-1]
    current_price = data['Close'].iloc[-1]
    drawdown = current_price / recent_high - 1
    
    # Calculate moving averages for trend confirmation
    if len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        price_below_ma200 = current_price < ma200
        ma50_below_ma200 = ma50 < ma200
    else:
        price_below_ma200 = False
        ma50_below_ma200 = False
    
    # Determine if we're in a bear market
    is_bear_market = (drawdown < threshold) or (price_below_ma200 and ma50_below_ma200)
    
    # Calculate confidence level
    if is_bear_market:
        # Normalize confidence based on drawdown severity
        # More negative drawdown = higher confidence
        confidence_from_drawdown = min(1.0, abs(drawdown / threshold))
        
        # Factor in moving average confirmation
        ma_confirmation = 0.0
        if price_below_ma200 and ma50_below_ma200:
            ma_confirmation = 1.0
        elif price_below_ma200:
            ma_confirmation = 0.5
        
        # Combine factors for final confidence score
        confidence = 0.6 * confidence_from_drawdown + 0.4 * ma_confirmation
        
        # Additional confidence boost from extreme volatility
        recent_volatility = data['Close'].pct_change().iloc[-20:].std() * np.sqrt(252)
        typical_volatility = data['Close'].pct_change().iloc[-252:-20].std() * np.sqrt(252) if len(data) > 252 else recent_volatility
        if recent_volatility > 1.5 * typical_volatility:
            confidence = min(1.0, confidence + 0.2)
    else:
        confidence = 0.0
    
    return is_bear_market, confidence


# Extended regime detection functions - for future development

def detect_crypto_specific_regime(data: pd.DataFrame) -> Dict[str, float]:
    """
    Placeholder for crypto-specific regime detection.
    This could incorporate on-chain metrics, exchange flows, etc.
    
    Parameters:
    - data: DataFrame with crypto price/volume data and potentially on-chain metrics
    
    Returns:
    - Dict with confidence scores for Bull, Bear, and Neutral regimes
    """
    # This is a placeholder for future development
    return {'Bull': 0.0, 'Bear': 0.0, 'Neutral': 1.0}


def detect_sector_rotation(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Placeholder for sector rotation detection.
    This could help identify which sectors are leading/lagging.
    
    Parameters:
    - data_dict: Dictionary of DataFrames with sector ETF price data
    
    Returns:
    - Dict with sector scores (-1 to 1 scale, negative = bearish)
    """
    # This is a placeholder for future development
    return {}
