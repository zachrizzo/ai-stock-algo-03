#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Simulation Utilities
==========================
Functions for simulating realistic market data based on historical parameters.
These utilities help generate synthetic data for testing and development.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Union, Optional, Tuple, List, Dict, Any

# Set random seed for reproducibility in the module
np.random.seed(42)
random.seed(42)

def generate_realistic_market_data(
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime], 
    ticker: str = 'SPY'
) -> pd.DataFrame:
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
        annual_volatility = 0.20  # 20% annual volatility
        skew = -0.7  # More negative skew
        kurtosis = 6  # Higher excess kurtosis
        autocorrelation = 0.06  # Slightly higher autocorrelation
    elif ticker == 'BTC':
        # Bitcoin characteristics
        annual_return = 0.60  # 60% annual return (historically)
        annual_volatility = 0.80  # 80% annual volatility
        skew = 0.2  # Slight positive skew
        kurtosis = 8  # High excess kurtosis
        autocorrelation = 0.10  # Higher autocorrelation
    elif ticker == 'ETH':
        # Ethereum characteristics
        annual_return = 0.70  # 70% annual return (historically)
        annual_volatility = 0.90  # 90% annual volatility
        skew = 0.3  # Moderate positive skew
        kurtosis = 9  # Very high excess kurtosis
        autocorrelation = 0.12  # Higher autocorrelation
    else:
        # Default to S&P 500 characteristics
        annual_return = 0.10
        annual_volatility = 0.16
        skew = -0.5
        kurtosis = 5
        autocorrelation = 0.05
    
    # Convert annual parameters to daily
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Generate random returns with target characteristics
    n_days = len(date_range)
    
    # First, generate normal random values
    z = np.random.normal(0, 1, n_days)
    
    # Apply skew and kurtosis using Cornish-Fisher expansion
    skew_term = (z**2 - 1) * skew / 6
    kurt_term = (z**3 - 3*z) * (kurtosis - 3) / 24
    cf_z = z + skew_term + kurt_term
    
    # Convert to returns with target mean and volatility
    returns = daily_return + daily_volatility * cf_z
    
    # Apply autocorrelation (AR(1) process)
    for i in range(1, n_days):
        returns[i] = returns[i] + autocorrelation * returns[i-1]
    
    # Create price series
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLC data
    # Typical daily ranges (High-Low) are about 1-2% of price
    data = pd.DataFrame(index=date_range)
    data['Close'] = prices
    
    # Generate realistic H-L range and O values
    for i in range(len(data)):
        if i == 0:
            # First day
            data['Open'].iloc[0] = data['Close'].iloc[0] * 0.995  # First day open
            range_pct = np.random.uniform(0.005, 0.02)  # 0.5% to 2% range
            mid_price = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
            half_range = mid_price * range_pct / 2
            data['High'].iloc[0] = mid_price + half_range
            data['Low'].iloc[0] = mid_price - half_range
        else:
            # Subsequent days - open is related to previous close
            prev_close = data['Close'].iloc[i-1]
            data['Open'].iloc[i] = prev_close * (1 + np.random.normal(0, 0.003))  # Open with small gap
            
            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            price_min = min(data['Open'].iloc[i], data['Close'].iloc[i])
            price_max = max(data['Open'].iloc[i], data['Close'].iloc[i])
            
            # Generate realistic range
            range_pct = np.random.uniform(0.005, 0.025)  # 0.5% to 2.5% range
            extra_range = price_max * range_pct
            
            data['High'].iloc[i] = price_max + extra_range * np.random.uniform(0.2, 0.8)
            data['Low'].iloc[i] = price_min - extra_range * np.random.uniform(0.2, 0.8)
    
    # Generate volume data
    # Volume tends to be higher on down days and has autocorrelation
    base_volume = 1000000  # Base volume level
    vol_volatility = 0.2  # Volume volatility
    
    volumes = []
    for i in range(len(data)):
        if i == 0:
            # First day volume
            volumes.append(base_volume * np.exp(np.random.normal(0, vol_volatility)))
        else:
            # Volume is correlated with previous day and tends to be higher on down days
            ret = returns[i]
            vol_factor = 1.0
            if ret < 0:
                vol_factor = 1.1  # 10% higher volume on down days
            
            # Volume has autocorrelation
            prev_vol = volumes[i-1]
            new_vol = prev_vol * np.exp(np.random.normal(0, vol_volatility)) * vol_factor
            volumes.append(new_vol)
    
    data['Volume'] = volumes
    
    # Verify data types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col])
    
    return data


def add_market_regimes_to_data(
    data: pd.DataFrame, 
    bull_prob: float = 0.6, 
    bear_prob: float = 0.25, 
    min_regime_length: int = 20, 
    max_regime_length: int = 120
) -> pd.DataFrame:
    """
    Add simulated market regime labels to the data
    
    Args:
        data: DataFrame with price data
        bull_prob: Probability of bull market regime
        bear_prob: Probability of bear market regime
        min_regime_length: Minimum length of a regime period (days)
        max_regime_length: Maximum length of a regime period (days)
        
    Returns:
        DataFrame with added regime column
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    df['regime'] = 'Neutral'  # Default regime
    
    # Start with a random regime
    current_day = 0
    total_days = len(df)
    
    while current_day < total_days:
        # Determine next regime
        r = np.random.random()
        if r < bull_prob:
            regime = 'Bull'
        elif r < bull_prob + bear_prob:
            regime = 'Bear'
        else:
            regime = 'Neutral'
        
        # Determine regime length
        regime_length = np.random.randint(min_regime_length, max_regime_length + 1)
        regime_length = min(regime_length, total_days - current_day)
        
        # Set regime for this period
        df.iloc[current_day:current_day + regime_length, df.columns.get_loc('regime')] = regime
        
        # Move to next period
        current_day += regime_length
    
    return df


def generate_correlated_asset(
    base_data: pd.DataFrame, 
    correlation: float = 0.8, 
    volatility_ratio: float = 1.2, 
    return_premium: float = 0.02
) -> pd.DataFrame:
    """
    Generate correlated asset data based on existing market data
    
    Args:
        base_data: DataFrame with base asset price data (e.g., SPY)
        correlation: Target correlation with base asset (0-1)
        volatility_ratio: Ratio of new asset volatility to base asset
        return_premium: Annual return premium over base asset
        
    Returns:
        DataFrame with correlated asset data
    """
    # Create a copy to avoid modifying the original
    df = base_data.copy()
    
    # Calculate base asset returns
    base_returns = df['Close'].pct_change().dropna().values
    
    # Generate correlated return series
    n_days = len(base_returns)
    
    # Generate random noise
    noise = np.random.normal(0, 1, n_days)
    
    # Combine base returns and noise with target correlation
    new_returns = correlation * base_returns + np.sqrt(1 - correlation**2) * noise
    
    # Adjust volatility and add return premium
    base_vol = np.std(base_returns)
    target_vol = base_vol * volatility_ratio
    vol_scalar = target_vol / np.std(new_returns)
    
    daily_premium = return_premium / 252
    new_returns = new_returns * vol_scalar + daily_premium
    
    # Convert returns to prices
    start_idx = df.index[1]  # Skip first day since we don't have a return for it
    new_prices = 100 * np.cumprod(1 + new_returns)
    price_series = pd.Series(new_prices, index=df.index[1:])
    
    # Set first day price
    price_series = pd.concat([pd.Series([100], index=[df.index[0]]), price_series])
    
    # Create new DataFrame with OHLC data
    new_df = pd.DataFrame(index=df.index)
    new_df['Close'] = price_series
    
    # Generate Open, High, Low similar to the base asset patterns
    for i in range(len(new_df)):
        if i == 0:
            new_df['Open'].iloc[i] = new_df['Close'].iloc[i] * 0.995
        else:
            # Use similar pattern for open as the base asset
            base_gap = df['Open'].iloc[i] / df['Close'].iloc[i-1] - 1
            new_df['Open'].iloc[i] = new_df['Close'].iloc[i-1] * (1 + base_gap * correlation + 
                                                              np.random.normal(0, 0.001))
        
        # High and Low similar to base asset's range
        if i > 0:
            base_high_range = df['High'].iloc[i] / df['Close'].iloc[i] - 1
            base_low_range = df['Low'].iloc[i] / df['Close'].iloc[i] - 1
            
            # Apply similar ranges with some noise
            high_range = base_high_range * (correlation + (1-correlation)*np.random.random())
            low_range = base_low_range * (correlation + (1-correlation)*np.random.random())
            
            new_df['High'].iloc[i] = new_df['Close'].iloc[i] * (1 + high_range)
            new_df['Low'].iloc[i] = new_df['Close'].iloc[i] * (1 + low_range)
        else:
            # First day
            range_pct = np.random.uniform(0.005, 0.02)
            mid_price = (new_df['Open'].iloc[i] + new_df['Close'].iloc[i]) / 2
            half_range = mid_price * range_pct / 2
            new_df['High'].iloc[i] = mid_price + half_range
            new_df['Low'].iloc[i] = mid_price - half_range
    
    # Volume has similar pattern to base asset but with some independence
    log_volume = np.log(df['Volume'])
    log_volume_mean = log_volume.mean()
    log_volume_std = log_volume.std()
    
    new_log_vol = log_volume_mean + correlation * (log_volume - log_volume_mean) + \
                 np.sqrt(1 - correlation**2) * np.random.normal(0, log_volume_std, len(df))
    
    new_df['Volume'] = np.exp(new_log_vol)
    
    # Verify data types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        new_df[col] = pd.to_numeric(new_df[col])
    
    return new_df


def fetch_yahoo_data(
    ticker: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    use_cache: bool = True,
    cache_dir: str = 'data_cache'
) -> Optional[pd.DataFrame]:
    """
    Fetch data from Yahoo Finance with caching support
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        use_cache: Whether to use cached data if available
        cache_dir: Directory for cache files
        
    Returns:
        DataFrame with price data or None if fetch fails
    """
    import os
    import yfinance as yf
    
    # Create cache directory if it doesn't exist
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Convert dates to strings for cache filename
    if isinstance(start_date, datetime):
        start_str = start_date.strftime('%Y-%m-%d')
    else:
        start_str = start_date
        
    if isinstance(end_date, datetime):
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        end_str = end_date
    
    # Define cache file path
    cache_file = f"{cache_dir}/{ticker}_{start_str}_{end_str}.csv"
    
    # Try to load from cache first
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Loading {ticker} data from cache...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"Error reading cache file: {e}")
    
    # Fetch from Yahoo Finance
    try:
        print(f"Fetching {ticker} data from Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Save to cache if successful
        if use_cache and not df.empty:
            df.to_csv(cache_file)
            
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
