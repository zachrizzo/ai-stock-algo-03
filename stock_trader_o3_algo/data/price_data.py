"""
Module for fetching and processing price data.
"""
import datetime as dt
import os
from pathlib import Path
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from stock_trader_o3_algo.config.settings import HISTORY_DAYS


def fetch_prices(tickers: List[str], 
                 start_date: Optional[str] = None,
                 days: int = HISTORY_DAYS, 
                 end_date: Optional[dt.datetime] = None,
                 use_cache: bool = True,
                 cache_dir: str = "data_cache") -> pd.DataFrame:
    """
    Fetch historical price data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data fetch (string format 'YYYY-MM-DD')
        days: Number of days of history to fetch
        end_date: End date for the data fetch (defaults to current time)
        use_cache: Whether to use cached data if available
        cache_dir: Directory to store cached data
        
    Returns:
        DataFrame with close prices for each ticker
    """
    # Create cache directory if it doesn't exist and caching is enabled
    if use_cache:
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True, parents=True)
    
    # Set timezone to Eastern Time (market time)
    tz = pytz.timezone("America/New_York")
    
    # Set end date to current time if not provided
    if end_date is None:
        end_date = dt.datetime.now(tz=tz)
    elif not isinstance(end_date, dt.datetime):
        end_date = pd.to_datetime(end_date)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=tz)
    
    # Handle start date
    if start_date is None:
        # If no start_date provided, calculate based on days
        start_dt = end_date - dt.timedelta(days=int(days * 1.5))
    else:
        # Use provided start_date
        start_dt = pd.to_datetime(start_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=tz)
    
    # Check cache for existing data if caching is enabled
    if use_cache:
        # Create a unique cache key based on tickers and date range
        tickers_key = "_".join(sorted(tickers))
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        cache_key = f"{tickers_key}_{start_str}_{end_str}.feather"
        cache_file = cache_path / cache_key
        
        if cache_file.exists():
            try:
                df = pd.read_feather(cache_file)
                df.set_index("Date", inplace=True)
                print(f"Using cached data from {cache_file}")
                return df
            except Exception as e:
                print(f"Error reading cache: {e}, fetching fresh data")
    
    # Fetch data from Yahoo Finance
    print(f"Fetching data for {tickers} from {start_dt} to {end_date}")
    df = yf.download(tickers, start=start_dt, end=end_date, auto_adjust=True, progress=False)["Close"]
    
    if df.empty:
        raise ValueError(f"No data fetched for {tickers} between {start_dt} and {end_date}.")
    
    # Forward-fill then back-fill missing values to handle different inception dates
    df = df.ffill().bfill()
    # Drop rows where all columns are still NaN (shouldn't happen)
    df = df.dropna(how='all')
    
    # Cache the data if caching is enabled
    if use_cache and not df.empty:
        # Save to cache
        df_to_save = df.reset_index()
        df_to_save.to_feather(cache_file)
        print(f"Cached data to {cache_file}")
    
    return df


def calculate_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Calculate percentage returns for the given price data.
    
    Args:
        prices: DataFrame with price data
        periods: Number of periods to use for return calculation
        
    Returns:
        DataFrame with percentage returns
    """
    return prices.pct_change(periods).dropna()


def calculate_realized_vol(returns: Union[pd.DataFrame, pd.Series], 
                           window: int = 20, 
                           annualize: bool = True) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate realized volatility over a rolling window.
    
    Args:
        returns: DataFrame or Series with return data
        window: Window size for volatility calculation
        annualize: Whether to annualize the volatility
        
    Returns:
        DataFrame or Series with realized volatility
    """
    # Calculate rolling standard deviation
    if isinstance(returns, pd.DataFrame):
        vol = returns.rolling(window=window).std()
    else:
        vol = returns.rolling(window=window).std()
    
    # Annualize if requested (√252 is the annualization factor for daily data)
    if annualize:
        vol = vol * np.sqrt(252)
        
    return vol


def calculate_momentum(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    Calculate momentum for each ticker based on price ratio.
    
    Args:
        prices: DataFrame with price data
        lookback_days: Lookback period for momentum calculation
        
    Returns:
        DataFrame with momentum values
    """
    # Calculate momentum as current price / past price - 1
    return prices / prices.shift(lookback_days) - 1
