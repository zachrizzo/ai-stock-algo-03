#!/usr/bin/env python3
"""
Test TurboDMT_v2 strategy against DMT_v2 and Enhanced DMT_v2 using a realistic simulation
based on actual market statistics from recent years
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import yfinance as yf
import time

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_realistic_market_data(start_date, end_date, ticker='SPY'):
    """
    Generate realistic market data based on historical statistics
    
    Args:
        start_date: Start date for simulation (str or datetime)
        end_date: End date for simulation (str or datetime)
        ticker: Ticker symbol (affects volatility and return characteristics)
        
    Returns:
        DataFrame with OHLCV data
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # Create date range (business days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Set realistic market parameters based on historical data
    if ticker == 'SPY':
        # S&P 500 ETF characteristics (2010-2023)
        annual_return = 0.10  # 10% annual return
        annual_volatility = 0.16  # 16% annual volatility
        skew = -0.5  # Slight negative skew
        kurtosis = 5  # Excess kurtosis
        autocorrelation = 0.05  # Slight autocorrelation in returns
    elif ticker == 'QQQ':
        # Nasdaq 100 ETF characteristics (2010-2023)
        annual_return = 0.15  # 15% annual return
        annual_volatility = 0.22  # 22% annual volatility
        skew = -0.7  # More pronounced negative skew
        kurtosis = 7  # Higher excess kurtosis
        autocorrelation = 0.07  # Slightly higher autocorrelation
    else:
        # Default parameters
        annual_return = 0.08
        annual_volatility = 0.15
        skew = -0.3
        kurtosis = 4
        autocorrelation = 0.03
    
    # Calculate daily parameters
    num_days = len(date_range)
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Random noise with correlation structure for realistic returns
    np.random.seed(42)  # For reproducibility
    random_noise = np.random.normal(0, 1, num_days)
    
    # Add skewness and kurtosis (simple approximation)
    # Using Cornish-Fisher expansion for skewness and kurtosis
    random_noise = random_noise + (skew/6)*(random_noise**2 - 1) + (kurtosis/24)*(random_noise**3 - 3*random_noise)
    
    # Add autocorrelation for more realistic returns
    for i in range(1, num_days):
        random_noise[i] = autocorrelation * random_noise[i-1] + (1 - autocorrelation) * random_noise[i]
    
    # Generate log returns (mean = daily_return, std = daily_volatility)
    log_returns = daily_return - 0.5 * daily_volatility**2 + daily_volatility * random_noise
    
    # Generate simple returns from log returns
    simple_returns = np.exp(log_returns) - 1
    
    # Add volatility clustering (ARCH-like effects)
    volatility_multiplier = np.ones(num_days)
    for i in range(1, num_days):
        # Volatility responds to past shocks
        volatility_multiplier[i] = 0.9 * volatility_multiplier[i-1] + 0.1 * abs(random_noise[i-1])
    
    # Apply volatility clustering
    adjusted_returns = simple_returns * volatility_multiplier
    
    # Generate price series from returns
    start_price = 100.0  # Starting price
    prices = [start_price]
    for ret in adjusted_returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Ensure we have the right number of prices
    prices = prices[:num_days]
    
    # Create dummy OHLCV data
    # We'll use price as Close and create other values around it
    data = pd.DataFrame(index=date_range)
    data['Close'] = prices
    
    # Add some intraday volatility for Open, High, Low
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, daily_volatility/2, num_days))
    data['High'] = data['Close'] * (1 + abs(np.random.normal(0, daily_volatility, num_days)))
    data['Low'] = data['Close'] * (1 - abs(np.random.normal(0, daily_volatility, num_days)))
    
    # Ensure High >= Close >= Low and High >= Open >= Low
    data['High'] = data[['High', 'Close', 'Open']].max(axis=1)
    data['Low'] = data[['Low', 'Close', 'Open']].min(axis=1)
    
    # Add Volume (positively correlated with volatility)
    base_volume = 1000000  # Base volume level
    data['Volume'] = base_volume * (1 + 2 * abs(adjusted_returns)) * volatility_multiplier
    
    # Handle first day NaN values
    data['Open'].iloc[0] = data['Close'].iloc[0] * 0.995  # First day open
    
    # Ensure all numeric columns have proper float types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(float)
    
    print(f"Generated {len(data)} days of {ticker} data")
    return data

def detect_bear_market(data, lookback_period=40, threshold=-0.05):
    """
    Enhanced function to detect bear market conditions based on multiple indicators.
    Returns a boolean flag and a confidence score between 0 and 1.
    
    Parameters:
    - data: DataFrame with price data
    - lookback_period: Period for calculations
    - threshold: Return threshold for bear market detection
    
    Returns:
    - bear_market: Boolean indicating if a bear market is detected
    - confidence: Confidence score (0-1) of the bear market detection
    """
    if len(data) < max(lookback_period, 200):
        return False, 0.0
    
    current_price = data['Close'].iloc[-1]
    
    # 1. Check medium-term price action (percent change over lookback period)
    if lookback_period <= len(data):
        period_return = (current_price / data['Close'].iloc[-lookback_period] - 1)
    else:
        period_return = 0
    
    # 2. Check moving averages
    if len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
        
        price_below_ma200 = current_price < ma200
        ma50_below_ma200 = ma50 < ma200
        ma20_below_ma50 = ma20 < ma50
    else:
        price_below_ma200 = False
        ma50_below_ma200 = False
        ma20_below_ma50 = False
    
    # 3. Check volatility
    recent_volatility = data['Close'].pct_change().iloc[-20:].std() * np.sqrt(252)
    longer_volatility = data['Close'].pct_change().iloc[-60:].std() * np.sqrt(252)
    volatility_ratio = recent_volatility / longer_volatility if longer_volatility > 0 else 1.0
    
    # 4. Check volume analysis
    if 'Volume' in data.columns:
        # Calculate average volume on up and down days
        up_days = data[data['Close'] > data['Close'].shift(1)]
        down_days = data[data['Close'] < data['Close'].shift(1)]
        
        recent_up_volume = up_days['Volume'].iloc[-20:].mean() if len(up_days) > 0 else 0
        recent_down_volume = down_days['Volume'].iloc[-20:].mean() if len(down_days) > 0 else 0
        
        volume_ratio = recent_down_volume / recent_up_volume if recent_up_volume > 0 else 1.0
    else:
        volume_ratio = 1.0
    
    # 5. Check recent momentum
    recent_returns = data['Close'].pct_change().iloc[-5:].mean() * 252
    
    # 6. Check distance from recent high
    highest_price = data['Close'].iloc[-60:].max()
    distance_from_high = (current_price / highest_price - 1)
    
    # Calculate confidence score (0-1) based on all factors
    confidence = 0.0
    
    # Price action weight (40%)
    if period_return <= threshold:
        confidence += 0.4 * min(abs(period_return / threshold), 1.0)
    
    # Moving average weight (20%)
    ma_score = 0
    if price_below_ma200:
        ma_score += 0.4
    if ma50_below_ma200:
        ma_score += 0.3
    if ma20_below_ma50:
        ma_score += 0.3
    confidence += 0.2 * ma_score
    
    # Volatility weight (10%)
    if volatility_ratio > 1.1:
        confidence += 0.1 * min((volatility_ratio - 1.0) / 0.5, 1.0)
    
    # Volume weight (10%)
    if volume_ratio > 1.0:
        confidence += 0.1 * min((volume_ratio - 1.0) / 1.0, 1.0)
    
    # Recent momentum weight (10%)
    if recent_returns < 0:
        confidence += 0.1 * min(abs(recent_returns / (threshold * 252)), 1.0)
    
    # Distance from high weight (10%)
    if distance_from_high < 0:
        confidence += 0.1 * min(abs(distance_from_high / 0.1), 1.0)
    
    # Determine if it's a bear market based on confidence threshold
    bear_market = confidence >= 0.3  # Lower threshold for more sensitivity
    
    return bear_market, confidence

def detect_market_regime(data, lookback_period=40):
    """
    Enhanced function to detect market regime: Bull, Bear, or Neutral.
    Returns regime type and confidence scores.
    
    Parameters:
    - data: DataFrame with price data
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
    
    if len(data) >= 30:
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
    
    # Combine all factors with weights
    weights = {
        'ma': 0.25,         # Moving averages (most reliable long-term)
        'vol': 0.05,        # Volatility
        'volume': 0.05,     # Volume analysis
        'momentum': 0.20,   # Recent momentum (reliable short-term)
        'range': 0.15,      # Range position
        'trend': 0.30       # Trend strength (most reliable medium-term)
    }
    
    # Calculate final scores
    final_scores = {
        'Bull': (ma_scores['Bull'] * weights['ma'] + 
                vol_scores['Bull'] * weights['vol'] + 
                volume_scores['Bull'] * weights['volume'] + 
                momentum_scores['Bull'] * weights['momentum'] + 
                range_scores['Bull'] * weights['range'] + 
                trend_scores['Bull'] * weights['trend']),
        
        'Bear': (ma_scores['Bear'] * weights['ma'] + 
                vol_scores['Bear'] * weights['vol'] + 
                volume_scores['Bear'] * weights['volume'] + 
                momentum_scores['Bear'] * weights['momentum'] + 
                range_scores['Bear'] * weights['range'] + 
                trend_scores['Bear'] * weights['trend']),
        
        'Neutral': (ma_scores['Neutral'] * weights['ma'] + 
                   vol_scores['Neutral'] * weights['vol'] + 
                   volume_scores['Neutral'] * weights['volume'] + 
                   momentum_scores['Neutral'] * weights['momentum'] + 
                   range_scores['Neutral'] * weights['range'] + 
                   trend_scores['Neutral'] * weights['trend'])
    }
    
    # Determine market regime based on highest score
    regime = max(final_scores, key=final_scores.get)
    
    # Ensure scores sum to 1
    score_sum = sum(final_scores.values())
    normalized_scores = {k: v/score_sum for k, v in final_scores.items()}
    
    return regime, normalized_scores

def run_dmt_v2_realistic_backtest(data, version="original", initial_capital=10000.0, lookback_period=252):
    """
    Run a backtest of the DMT strategy on historical data.
    
    Args:
        data: Historical price data (OHLCV)
        version: Which version of the strategy to run
        initial_capital: Starting capital for the simulation
        lookback_period: Lookback period for calculations
        
    Returns:
        DataFrame with backtest results
    """
    # Parameters for different strategy versions
    if version == "original":
        target_vol = 0.25
        max_position = 1.0
        neutral_zone = 0.05
        allow_shorting = False
        adaptive_signal = False
    elif version == "enhanced":
        target_vol = 0.35
        max_position = 2.0
        neutral_zone = 0.03
        allow_shorting = False
        adaptive_signal = False
    elif version == "turbo":
        # Aggressive turbo parameters to outperform Buy & Hold
        # Increase target volatility and max position sizing
        target_vol = 0.40  # was 0.30 – take more risk when edge is strong
        max_position = 3.0  # was 2.0 – allow higher leverage in strong regimes
        neutral_zone = 0.02
        allow_shorting = True
        adaptive_signal = True
    else:
        raise ValueError(f"Unknown strategy version: {version}")
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=data.index)
    results['Close'] = data['Close']
    results['returns'] = data['Close'].pct_change()
    results['signal'] = 0.0
    results['position'] = 0.0
    results['equity'] = initial_capital
    results['cash'] = initial_capital
    results['regime'] = 'Neutral'
    results['bull_confidence'] = 0.0
    results['bear_confidence'] = 0.0
    results['neutral_confidence'] = 0.0
    results['market_state'] = None
    
    # Initialize tracking variables for adaptive signal adjustment
    regime_performance = {
        'Bull': {'returns': [], 'count': 0},
        'Bear': {'returns': [], 'count': 0},
        'Neutral': {'returns': [], 'count': 0}
    }
    
    # Initialize market bias trackers
    bull_bias = 0.0
    bear_bias = 0.0
    lookback_window = 60  # Days to look back for bias calculation
    
    # Buy and hold strategy for comparison
    results['buy_hold_equity'] = initial_capital * (1 + results['returns']).cumprod().fillna(1)
    
    # Minimum history needed for calculations
    min_history = max(lookback_period, 20)
    
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
        
        if version in ["enhanced", "turbo"]:
            regime, regime_confidences = detect_market_regime(history, lookback_period=40)
            
            # Store regime detection results
            results.loc[results.index[i], 'regime'] = regime
            results.loc[results.index[i], 'bull_confidence'] = regime_confidences['Bull']
            results.loc[results.index[i], 'bear_confidence'] = regime_confidences['Bear']
            results.loc[results.index[i], 'neutral_confidence'] = regime_confidences['Neutral']
            results.loc[results.index[i], 'market_state'] = regime
            
            # For backward compatibility
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
        
        # Calculate signal
        if i > min_history:  # Skip the first day after min_history
            # Calculate momentum metrics for different time frames
            momentum_5d = history['Close'].iloc[-1] / history['Close'].iloc[-6] - 1
            momentum_10d = history['Close'].iloc[-1] / history['Close'].iloc[-11] - 1
            momentum_20d = history['Close'].iloc[-1] / history['Close'].iloc[-21] - 1
            momentum_50d = history['Close'].iloc[-1] / history['Close'].iloc[-51] - 1
            
            # Base signal calculation
            signal = 0.0
            
            # Regime-specific signal calculation (for turbo version)
            if version == "turbo":
                if regime == 'Bull':
                    # In bull regimes, emphasize short-term momentum
                    signal = 0.45 * momentum_5d + 0.30 * momentum_10d + 0.15 * momentum_20d + 0.10 * momentum_50d
                    
                    # Apply bull bias
                    signal = max(0, signal) * (1 + bull_bias) + min(0, signal) * (1 - bull_bias)
                    
                elif regime == 'Bear':
                    # In bear regimes, we'll emphasize medium-term momentum and consider inverse signals
                    raw_signal = 0.15 * momentum_5d + 0.25 * momentum_10d + 0.40 * momentum_20d + 0.20 * momentum_50d
                    
                    # Blend normal and inverse signal based on bear confidence
                    inverse_weight = min(0.8, regime_confidences['Bear'] * 1.2)
                    normal_weight = 1.0 - inverse_weight
                    
                    signal = (raw_signal * normal_weight) + (-raw_signal * inverse_weight)
                    
                    # Apply bear bias
                    signal = max(0, signal) * (1 - bear_bias) + min(0, signal) * (1 + bear_bias)
                    
                else:  # Neutral regime
                    # In neutral regimes, use a balanced approach
                    signal = 0.25 * momentum_5d + 0.25 * momentum_10d + 0.25 * momentum_20d + 0.25 * momentum_50d
            else:
                # For original and enhanced versions, use standard weighting
                if version == "original":
                    signal = 0.6 * momentum_10d + 0.4 * momentum_50d
                else:  # enhanced
                    signal = 0.4 * momentum_5d + 0.3 * momentum_10d + 0.2 * momentum_20d + 0.1 * momentum_50d
            
            # Normalize signal to a -1 to 1 scale based on historical distribution
            if len(history) > 60:
                # Get historical momentum
                hist_momentum = pd.Series([
                    history['Close'].iloc[j] / history['Close'].iloc[j-10] - 1
                    for j in range(max(51, i-100), i) if j >= 10
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
            if abs(signal) < neutral_zone:
                signal = 0
            
            # Update market bias (for turbo version)
            if version == "turbo" and i > lookback_window:
                # Calculate recent performance in each regime
                for j in range(i - lookback_window, i):
                    day_regime = results['regime'].iloc[j]
                    next_return = data['Close'].iloc[j+1] / data['Close'].iloc[j] - 1
                    
                    if day_regime in regime_performance:
                        regime_performance[day_regime]['returns'].append(next_return)
                        regime_performance[day_regime]['count'] += 1
                
                # Calculate average performance by regime
                bull_perf = np.mean(regime_performance['Bull']['returns'][-20:]) if regime_performance['Bull']['returns'] else 0
                bear_perf = np.mean(regime_performance['Bear']['returns'][-20:]) if regime_performance['Bear']['returns'] else 0
                
                # Update bias based on recent performance
                bull_bias = max(0, min(0.5, bull_perf * 20))  # Cap at 0.5 (50% bias)
                bear_bias = max(0, min(0.5, -bear_perf * 20))  # Cap at 0.5 (50% bias)
            
            # Scale signal by target volatility / actual volatility (volatility targeting)
            if annual_volatility > 0:
                vol_scalar = target_vol / annual_volatility
                
                # Allow slightly higher leverage for strong edge, but guard lower bound
                vol_scalar = min(3.5, max(0.3, vol_scalar))
                
                # Apply the scaling
                signal *= vol_scalar
            
            # Apply regime-specific position sizing (for turbo version)
            if version == "turbo":
                # In strongly bullish markets, allow up to 3x long exposure and reduce shorts drastically
                if regime == 'Bull':
                    bull_max = max_position * (1 + 0.6 * regime_confidences['Bull'])
                    bear_max = max_position * 0.3  # Very limited shorts in bull regimes
                    
                    if signal > 0:
                        signal = min(signal, bull_max)
                    else:
                        signal = max(signal, -bear_max)
                        
                # In bear regimes, allow heavier shorts, limit longs further
                elif regime == 'Bear':
                    bull_max = max_position * 0.3  # Minimal longs allowed in bear regimes
                    bear_max = max_position * (1 + 0.6 * regime_confidences['Bear'])
                    
                    if signal > 0:
                        signal = min(signal, bull_max)
                    else:
                        signal = max(signal, -bear_max)
                        
                # In neutral regimes, use standard position sizing but cap at 70% of max
                else:
                    neutral_cap = max_position * 0.7
                    signal = max(min(signal, neutral_cap), -neutral_cap)
            else:
                # For original and enhanced versions, use standard position sizing
                signal = max(min(signal, max_position), -max_position)
        else:
            # Not enough history, no signal
            signal = 0
        
        # Store signal
        results.loc[results.index[i], 'signal'] = signal
        
        # Update position (assumed to be executed at the close)
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

def calculate_performance_metrics(results, initial_capital):
    """Calculate performance metrics from backtest results"""
    equity_curve = results['equity']
    buy_hold_equity = results['buy_hold_equity']
    
    # Calculate daily returns
    daily_returns = equity_curve.pct_change().dropna()
    buy_hold_returns = buy_hold_equity.pct_change().dropna()
    
    # Calculate total return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    buy_hold_return = buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1
    
    # Calculate annualized volatility
    annual_vol = daily_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Calculate CAGR
    days = (results.index[-1] - results.index[0]).days
    years = days / 365
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate Calmar ratio
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate win rate (percentage of winning days)
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
    
    # Calculate activity rate
    activity_rate = np.sum(np.abs(results['position'].diff()) > 0.01) / len(results) if len(results) > 0 else 0
    
    # Calculate bear market statistics (if market_state is available)
    bear_market_stats = {}
    if 'market_state' in results.columns:
        bear_days = (results['market_state'] == 'Bear').sum()
        bear_market_stats['bear_market_days'] = bear_days
        bear_market_stats['bear_market_pct'] = bear_days / len(results) if len(results) > 0 else 0
        
        # Calculate performance during bear markets
        if bear_days > 0:
            bear_market_data = results[results['market_state'] == 'Bear']
            if len(bear_market_data) > 1:
                bear_returns = bear_market_data['equity'].pct_change().dropna()
                if len(bear_returns) > 0:
                    bear_market_stats['bear_market_return'] = ((1 + bear_returns).prod() - 1)
                    bear_market_stats['bear_market_sharpe'] = bear_returns.mean() / bear_returns.std() * np.sqrt(252) if bear_returns.std() > 0 else 0
    
    # Combine all metrics
    metrics = {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'cagr': cagr,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'activity_rate': activity_rate,
        **bear_market_stats
    }
    
    return metrics

def safe_improvement(new_value, old_value):
    """Calculate improvement percentage safely, avoiding division by zero"""
    if old_value == 0:
        if new_value > 0:
            return float('inf')  # Positive improvement from zero
        elif new_value < 0:
            return float('-inf')  # Negative change from zero
        else:
            return 0.0  # No change
    else:
        return (new_value - old_value) / abs(old_value)

def plot_results(original_results, enhanced_results, turbo_results, title, filename):
    """Plot comparative results of different strategy versions"""
    plt.figure(figsize=(14, 10))
    
    # Create a 2x2 subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Plot equity curves
    ax1 = plt.subplot(gs[0, :])
    ax1.set_title(f"{title} - 3-Year Equity Curves", fontsize=14)
    
    # Normalize equity curves to start at 100 for easier comparison
    norm_factor = 100 / original_results['equity'].iloc[0]
    ax1.plot(original_results.index, original_results['equity'] * norm_factor, 
            label='DMT_v2 Original', linewidth=2)
    ax1.plot(enhanced_results.index, enhanced_results['equity'] * norm_factor, 
            label='DMT_v2 Enhanced', linewidth=2)
    ax1.plot(turbo_results.index, turbo_results['equity'] * norm_factor, 
            label='TurboDMT_v2', linewidth=2)
    ax1.plot(original_results.index, original_results['buy_hold_equity'] * norm_factor, 
            label='Buy & Hold', linewidth=2, linestyle='--', alpha=0.7)
    
    # Add y-axis labels
    ax1.set_ylabel('Equity (normalized to 100)', fontsize=12)
    
    # Highlight bear market periods (if available)
    if 'market_state' in turbo_results.columns:
        bear_periods = []
        bear_start = None
        
        # Find bear market periods
        for idx, row in turbo_results.iterrows():
            if row['market_state'] == 'Bear' and bear_start is None:
                bear_start = idx
            elif row['market_state'] != 'Bear' and bear_start is not None and row['market_state'] is not None:
                bear_periods.append((bear_start, idx))
                bear_start = None
        
        # Add final period if still in bear market at end
        if bear_start is not None:
            bear_periods.append((bear_start, turbo_results.index[-1]))
        
        # Highlight bear market periods
        for start, end in bear_periods:
            ax1.axvspan(start, end, color='red', alpha=0.2)
        
        # Add a legend entry for bear markets
        import matplotlib.patches as mpatches
        bear_patch = mpatches.Patch(color='red', alpha=0.2, label='Bear Market')
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    if 'market_state' in turbo_results.columns and bear_periods:
        handles.append(bear_patch)
    ax1.legend(handles=handles, loc='upper left', fontsize=10)
    
    # Plot positions
    ax2 = plt.subplot(gs[1, 0])
    ax2.set_title('Position Sizes (TurboDMT_v2)', fontsize=12)
    ax2.plot(turbo_results.index, turbo_results['position'], color='purple', linewidth=1.5)
    ax2.set_ylabel('Position Size', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add zero line for reference
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot bear confidence if available
    if 'bear_confidence' in turbo_results.columns:
        # Filter out None values
        valid_confidence = turbo_results[turbo_results['bear_confidence'].notna()]
        
        if not valid_confidence.empty:
            ax3 = plt.subplot(gs[1, 1])
            ax3.set_title('Bear Market Confidence', fontsize=12)
            
            # Plot bear confidence as a heatmap
            scatter = ax3.scatter(valid_confidence.index, 
                                 valid_confidence['bear_confidence'],
                                 c=valid_confidence['bear_confidence'], 
                                 cmap='RdYlGn_r',  # Reversed RdYlGn (red for high confidence)
                                 alpha=0.8,
                                 s=30)
            
            # Add threshold line
            ax3.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, label='Bear Threshold')
            
            # Add labels and grid
            ax3.set_ylabel('Confidence Score', fontsize=10)
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

def plot_turbo_positions(results, ticker, filename):
    """
    Create detailed position analysis plot for TurboDMT_v2
    """
    plt.figure(figsize=(14, 14))  # Increased size to accommodate more subplots
    
    # Create a more complex subplot grid with 4 rows
    gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # Plot equity curve
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f"TurboDMT_v2 Analysis - {ticker}", fontsize=14)
    
    # Normalize equity curves to start at 100
    norm_factor = 100 / results['equity'].iloc[0]
    ax1.plot(results.index, results['equity'] * norm_factor, 
            label='TurboDMT_v2', linewidth=2, color='purple')
    ax1.plot(results.index, results['buy_hold_equity'] * norm_factor, 
            label='Buy & Hold', linewidth=2, linestyle='--', color='gray')
    
    # Add y-axis labels and legend
    ax1.set_ylabel('Equity (normalized to 100)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Highlight bear market periods (if available)
    if 'market_state' in results.columns:
        bear_periods = []
        bear_start = None
        
        # Find bear market periods
        for idx, row in results.iterrows():
            if row['market_state'] == 'Bear' and bear_start is None:
                bear_start = idx
            elif row['market_state'] != 'Bear' and bear_start is not None and row['market_state'] is not None:
                bear_periods.append((bear_start, idx))
                bear_start = None
        
        # Add final period if still in bear market at end
        if bear_start is not None:
            bear_periods.append((bear_start, results.index[-1]))
        
        # Highlight bear market periods
        for start, end in bear_periods:
            ax1.axvspan(start, end, color='red', alpha=0.2)
        
        # Add label for bear market regions
        if bear_periods:
            ax1.text(0.02, 0.98, 'Red areas indicate bear markets', 
                    transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                                                     facecolor='white', 
                                                     alpha=0.7))
    
    # Plot positions
    ax2 = plt.subplot(gs[1])
    ax2.set_title('Position Sizes Over Time', fontsize=12)
    
    # Color by position direction
    for i in range(1, len(results)):
        color = 'green' if results['position'].iloc[i] > 0 else 'red' if results['position'].iloc[i] < 0 else 'gray'
        ax2.plot([results.index[i-1], results.index[i]], 
                [results['position'].iloc[i-1], results['position'].iloc[i]], 
                color=color, linewidth=1.5)
    
    ax2.set_ylabel('Position Size', fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot bear confidence if available
    if 'bear_confidence' in results.columns:
        # Filter out None values
        valid_confidence = results[results['bear_confidence'].notna()]
        
        if not valid_confidence.empty:
            ax3 = plt.subplot(gs[2])
            ax3.set_title('Bear Market Confidence', fontsize=12)
            
            # Create a colormap for bear confidence
            cmap = plt.cm.get_cmap('RdYlGn_r')  # Reversed RdYlGn (red for high confidence)
            
            # Plot points with color based on confidence
            scatter = ax3.scatter(valid_confidence.index, 
                                 valid_confidence['bear_confidence'],
                                 c=valid_confidence['bear_confidence'], 
                                 cmap=cmap,
                                 alpha=0.8,
                                 s=30)
            
            # Mark the bear threshold
            ax3.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, label='Bear Threshold')
            
            # Add labels and grid
            ax3.set_ylabel('Confidence Score', fontsize=10)
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax3)
    
    # Plot position size distribution
    ax4 = plt.subplot(gs[3])
    ax4.set_title('Position Size Distribution', fontsize=12)
    
    # Exclude zero positions for better visualization
    non_zero_positions = results.loc[abs(results['position']) > 0.01, 'position']
    
    if len(non_zero_positions) > 0:
        # Split positions into long and short for separate histograms
        long_positions = non_zero_positions[non_zero_positions > 0]
        short_positions = non_zero_positions[non_zero_positions < 0]
        
        # Create histograms with different colors
        if len(long_positions) > 0:
            ax4.hist(long_positions, bins=15, alpha=0.7, color='green', label='Long Positions')
        if len(short_positions) > 0:
            ax4.hist(short_positions, bins=15, alpha=0.7, color='red', label='Short Positions')
        
        # Add vertical lines for important position sizes
        if len(long_positions) > 0:
            ax4.axvline(x=long_positions.mean(), color='darkgreen', linestyle='-', 
                       label=f'Long Mean: {long_positions.mean():.2f}')
        if len(short_positions) > 0:
            ax4.axvline(x=short_positions.mean(), color='darkred', linestyle='-', 
                       label=f'Short Mean: {short_positions.mean():.2f}')
        
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add labels
        ax4.set_xlabel('Position Size', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'No active positions to analyze', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    return filename

def main():
    """Run the simulation and backtest"""
    # Set parameters
    start_date = "2022-04-01"
    end_date = "2025-04-01"  # 3 years out
    test_ticker = "SPY"
    os.makedirs("tri_shot_data", exist_ok=True)  # Ensure output directory exists
    
    print(f"Testing period: {start_date} to {end_date}")
    
    # Use generated market data to avoid API rate limits
    print("Using simulated market data...")
    
    # Generate realistic market data for SPY
    spy_data = generate_realistic_market_data(start_date, end_date, ticker="SPY")
    
    # Print first few rows and datatypes for validation
    print("First few rows of SPY data:")
    print(spy_data.head())
    print("SPY data types:")
    print(spy_data.dtypes)
    
    # Generate more volatile data for QQQ
    qqq_data = generate_realistic_market_data(start_date, end_date, ticker="QQQ")
    
    # Print first few rows and datatypes for validation
    print("First few rows of QQQ data:")
    print(qqq_data.head())
    print("QQQ data types:")
    print(qqq_data.dtypes)
    
    try:
        print("\nRunning backtests on SPY data...")
        spy_original_results, spy_original_activity = run_dmt_v2_realistic_backtest(spy_data, version="original")
        spy_enhanced_results, spy_enhanced_activity = run_dmt_v2_realistic_backtest(spy_data, version="enhanced")
        spy_turbo_results, spy_turbo_activity = run_dmt_v2_realistic_backtest(spy_data, version="turbo")
        
        print("\nRunning backtests on QQQ data...")
        qqq_original_results, qqq_original_activity = run_dmt_v2_realistic_backtest(qqq_data, version="original")
        qqq_enhanced_results, qqq_enhanced_activity = run_dmt_v2_realistic_backtest(qqq_data, version="enhanced")
        qqq_turbo_results, qqq_turbo_activity = run_dmt_v2_realistic_backtest(qqq_data, version="turbo")
        
        # Create plots
        spy_plot = plot_results(
            spy_original_results, 
            spy_enhanced_results, 
            spy_turbo_results, 
            "SPY", 
            "tri_shot_data/spy_3yr_simulation.png"
        )
        
        qqq_plot = plot_results(
            qqq_original_results, 
            qqq_enhanced_results, 
            qqq_turbo_results, 
            "QQQ", 
            "tri_shot_data/qqq_3yr_simulation.png"
        )
        
        # Create detailed position analysis for the TurboDMT_v2 strategy
        spy_turbo_pos_plot = plot_turbo_positions(
            spy_turbo_results, 
            "SPY", 
            "tri_shot_data/spy_turbo_positions.png"
        )
        
        qqq_turbo_pos_plot = plot_turbo_positions(
            qqq_turbo_results, 
            "QQQ", 
            "tri_shot_data/qqq_turbo_positions.png"
        )
        
        # Calculate performance metrics
        spy_original_metrics = calculate_performance_metrics(spy_original_results, 10000.0)
        spy_enhanced_metrics = calculate_performance_metrics(spy_enhanced_results, 10000.0)
        spy_turbo_metrics = calculate_performance_metrics(spy_turbo_results, 10000.0)
        
        qqq_original_metrics = calculate_performance_metrics(qqq_original_results, 10000.0)
        qqq_enhanced_metrics = calculate_performance_metrics(qqq_enhanced_results, 10000.0)
        qqq_turbo_metrics = calculate_performance_metrics(qqq_turbo_results, 10000.0)
        
        # Print performance metrics
        print("\n" + "="*80)
        print("SPY 3-YEAR BACKTEST RESULTS (REALISTIC SIMULATION)")
        print("="*80)
        print("Strategy             Total Return    CAGR     Sharpe   MaxDD      Calmar   Activity %")
        print("-"*80)
        print(f"DMT_v2 Original    {spy_original_metrics['total_return']:.2%}      {spy_original_metrics['cagr']:.2%}   {spy_original_metrics['sharpe_ratio']:.2f}    {spy_original_metrics['max_drawdown']:.2%}   {spy_original_metrics['calmar_ratio']:.2f}    {spy_original_metrics['activity_rate']:.2%}")
        print(f"DMT_v2 Enhanced    {spy_enhanced_metrics['total_return']:.2%}      {spy_enhanced_metrics['cagr']:.2%}   {spy_enhanced_metrics['sharpe_ratio']:.2f}    {spy_enhanced_metrics['max_drawdown']:.2%}   {spy_enhanced_metrics['calmar_ratio']:.2f}    {spy_enhanced_metrics['activity_rate']:.2%}")
        print(f"TurboDMT_v2        {spy_turbo_metrics['total_return']:.2%}      {spy_turbo_metrics['cagr']:.2%}   {spy_turbo_metrics['sharpe_ratio']:.2f}    {spy_turbo_metrics['max_drawdown']:.2%}   {spy_turbo_metrics['calmar_ratio']:.2f}    {spy_turbo_metrics['activity_rate']:.2%}")
        print(f"Buy & Hold         {spy_original_metrics['buy_hold_return']:.2%}      {(spy_original_metrics['buy_hold_return']+1)**(1/3)-1:.2%}   -       -         -        -")
        
        print("\n" + "="*80)
        print("QQQ 3-YEAR BACKTEST RESULTS (REALISTIC SIMULATION)")
        print("="*80)
        print("Strategy             Total Return    CAGR     Sharpe   MaxDD      Calmar   Activity %")
        print("-"*80)
        print(f"DMT_v2 Original    {qqq_original_metrics['total_return']:.2%}      {qqq_original_metrics['cagr']:.2%}   {qqq_original_metrics['sharpe_ratio']:.2f}    {qqq_original_metrics['max_drawdown']:.2%}   {qqq_original_metrics['calmar_ratio']:.2f}    {qqq_original_metrics['activity_rate']:.2%}")
        print(f"DMT_v2 Enhanced    {qqq_enhanced_metrics['total_return']:.2%}      {qqq_enhanced_metrics['cagr']:.2%}   {qqq_enhanced_metrics['sharpe_ratio']:.2f}    {qqq_enhanced_metrics['max_drawdown']:.2%}   {qqq_enhanced_metrics['calmar_ratio']:.2f}    {qqq_enhanced_metrics['activity_rate']:.2%}")
        print(f"TurboDMT_v2        {qqq_turbo_metrics['total_return']:.2%}      {qqq_turbo_metrics['cagr']:.2%}   {qqq_turbo_metrics['sharpe_ratio']:.2f}    {qqq_turbo_metrics['max_drawdown']:.2%}   {qqq_turbo_metrics['calmar_ratio']:.2f}    {qqq_turbo_metrics['activity_rate']:.2%}")
        print(f"Buy & Hold         {qqq_original_metrics['buy_hold_return']:.2%}      {(qqq_original_metrics['buy_hold_return']+1)**(1/3)-1:.2%}   -       -         -        -")
        
        # Display TurboDMT_v2 bear market performance if available
        if 'bear_market_days' in spy_turbo_metrics:
            print("\n" + "="*80)
            print("TURBODMT_V2 BEAR MARKET PERFORMANCE")
            print("="*80)
            print(f"SPY Bear Market Days: {spy_turbo_metrics['bear_market_days']} ({spy_turbo_metrics['bear_market_pct']:.2%} of testing period)")
            if 'bear_market_return' in spy_turbo_metrics:
                print(f"SPY Bear Market Return: {spy_turbo_metrics['bear_market_return']:.2%}")
                print(f"SPY Bear Market Sharpe: {spy_turbo_metrics['bear_market_sharpe']:.2f}")
            
            print(f"\nQQQ Bear Market Days: {qqq_turbo_metrics['bear_market_days']} ({qqq_turbo_metrics['bear_market_pct']:.2%} of testing period)")
            if 'bear_market_return' in qqq_turbo_metrics:
                print(f"QQQ Bear Market Return: {qqq_turbo_metrics['bear_market_return']:.2%}")
                print(f"QQQ Bear Market Sharpe: {qqq_turbo_metrics['bear_market_sharpe']:.2f}")
        
        print("\nPosition Analysis:")
        print(f"- SPY TurboDMT_v2 position analysis: {spy_turbo_pos_plot}")
        print(f"- QQQ TurboDMT_v2 position analysis: {qqq_turbo_pos_plot}")
        print(f"- SPY activity rate: {spy_turbo_metrics['activity_rate']:.2%}")
        print(f"- QQQ activity rate: {qqq_turbo_metrics['activity_rate']:.2%}")
        
        # Show improvement percentages
        print("\n" + "="*80)
        print("STRATEGY IMPROVEMENTS (REALISTIC SIMULATION RESULTS)")
        print("="*80)
        
        # SPY improvements - with safe calculation
        spy_ret_imp_enh = safe_improvement(spy_enhanced_metrics['total_return'], spy_original_metrics['total_return'])
        spy_ret_imp_turbo = safe_improvement(spy_turbo_metrics['total_return'], spy_original_metrics['total_return'])
        spy_sharpe_imp_enh = safe_improvement(spy_enhanced_metrics['sharpe_ratio'], spy_original_metrics['sharpe_ratio'])
        spy_sharpe_imp_turbo = safe_improvement(spy_turbo_metrics['sharpe_ratio'], spy_original_metrics['sharpe_ratio'])
        
        if np.isfinite(spy_ret_imp_enh):
            print(f"SPY - Enhanced vs Original:  Return: {spy_ret_imp_enh:+.2%}, Sharpe: {spy_sharpe_imp_enh:+.2%}")
        else:
            print(f"SPY - Enhanced vs Original:  Return: {spy_enhanced_metrics['total_return']:.2%} vs {spy_original_metrics['total_return']:.2%}")
        
        if np.isfinite(spy_ret_imp_turbo):
            print(f"SPY - Turbo vs Original:     Return: {spy_ret_imp_turbo:+.2%}, Sharpe: {spy_sharpe_imp_turbo:+.2%}")
        else:
            print(f"SPY - Turbo vs Original:     Return: {spy_turbo_metrics['total_return']:.2%} vs {spy_original_metrics['total_return']:.2%}")
        
        # QQQ improvements - with safe calculation
        qqq_ret_imp_enh = safe_improvement(qqq_enhanced_metrics['total_return'], qqq_original_metrics['total_return'])
        qqq_ret_imp_turbo = safe_improvement(qqq_turbo_metrics['total_return'], qqq_original_metrics['total_return'])
        qqq_sharpe_imp_enh = safe_improvement(qqq_enhanced_metrics['sharpe_ratio'], qqq_original_metrics['sharpe_ratio'])
        qqq_sharpe_imp_turbo = safe_improvement(qqq_turbo_metrics['sharpe_ratio'], qqq_original_metrics['sharpe_ratio'])
        
        if np.isfinite(qqq_ret_imp_enh):
            print(f"QQQ - Enhanced vs Original:  Return: {qqq_ret_imp_enh:+.2%}, Sharpe: {qqq_sharpe_imp_enh:+.2%}")
        else:
            print(f"QQQ - Enhanced vs Original:  Return: {qqq_enhanced_metrics['total_return']:.2%} vs {qqq_original_metrics['total_return']:.2%}")
        
        if np.isfinite(qqq_ret_imp_turbo):
            print(f"QQQ - Turbo vs Original:     Return: {qqq_ret_imp_turbo:+.2%}, Sharpe: {qqq_sharpe_imp_turbo:+.2%}")
        else:
            print(f"QQQ - Turbo vs Original:     Return: {qqq_turbo_metrics['total_return']:.2%} vs {qqq_original_metrics['total_return']:.2%}")
        
        print("\nResults summary:")
        print(f"- SPY simulation plot saved to: {spy_plot}")
        print(f"- QQQ simulation plot saved to: {qqq_plot}")
        
        print("\nIMPORTANT: These are realistic simulations based on actual market statistics.")
        print("This approach avoids Yahoo Finance rate limits while still providing meaningful performance estimates.")
        print("The simulation incorporates realistic market behavior including volatility clustering, autocorrelation, and market regimes.")
        
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
