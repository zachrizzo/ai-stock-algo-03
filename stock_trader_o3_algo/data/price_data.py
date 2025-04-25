"""
Module for fetching and processing price data.
"""
import datetime as dt
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from stock_trader_o3_algo.config.settings import HISTORY_DAYS


def fetch_prices(tickers: List[str], 
                 days: int = HISTORY_DAYS, 
                 end_date: Optional[dt.datetime] = None) -> pd.DataFrame:
    """
    Fetch historical price data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history to fetch
        end_date: End date for the data fetch (defaults to current time)
        
    Returns:
        DataFrame with close prices for each ticker
    """
    # Set timezone to Eastern Time (market time)
    tz = pytz.timezone("America/New_York")
    
    # Set end date to current time if not provided
    if end_date is None:
        end_date = dt.datetime.now(tz=tz)
    elif not isinstance(end_date, dt.datetime):
        end_date = pd.to_datetime(end_date)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=tz)
    
    # Set start date with buffer to ensure we have enough data
    start_date = end_date - dt.timedelta(days=int(days * 1.5))
    
    # Fetch data from Yahoo Finance
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    
    # Clean data by dropping rows with NaN values
    return df.dropna()


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
