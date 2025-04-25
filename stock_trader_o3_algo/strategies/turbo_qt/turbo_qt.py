#!/usr/bin/env python
"""
Turbo-Rotational "QQQ Turbo-4" Strategy

A more aggressive trading strategy using leveraged ETFs:
- 3× exposure when trend is positive (TQQQ)
- -3× exposure when trend is negative (SQQQ)
- 3× Treasuries (TMF) as fallback
- ATR-based stops
- VIX crash overlay for protection

Key Parameters:
- 25% volatility target
- 3× 14-day ATR for stop-loss
- VIX > 30 and 5-day QQQ return < -7% triggers crash protection
- Monday rebalancing with at most 3 trades per week (PDT-compliant)
- 20% drawdown kill-switch
"""

import os
import math
import datetime as dt
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Constants
TICKERS = {
    "UP": "TQQQ",    # 3x QQQ long ETF
    "DN": "SQQQ",    # 3x QQQ short ETF
    "BOND": "TMF",   # 3x long-term Treasury ETF
    "CASH": "BIL",   # Treasury Bills ETF
    "VIX": "^VIX",   # Volatility Index
    "SRC": "QQQ"     # Nasdaq 100 ETF (source for signals)
}

# Strategy parameters
VOL_TARGET = 0.25    # Annual volatility target (balanced for higher returns)
ATR_MULT = 2.0       # ATR multiplier for stops (tightened to catch downturns faster)
MOM_DAYS = [10, 30]  # Much shorter momentum lookback periods for faster signal generation
BOND_DAYS = 30       # Bond momentum lookback
VIX_THRESHOLD = 25   # Lower VIX threshold to catch crashes earlier
CRASH_THRESHOLD = -0.05  # Less severe 5-day return threshold to catch crashes earlier
HEDGE_WEIGHT = 0.25  # Allocation to hedge asset during crashes
KILL_DD = 0.25       # Lower max drawdown to protect capital better
COOLDOWN_WEEKS = 1   # Weeks to remain in cash after stop-loss

# Additional strategy parameters for technical indicators
RSI_PERIOD = 14
RSI_UPPER = 55
RSI_LOWER = 45
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Timezone
TZ = pytz.timezone("America/New_York")

# File paths
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
os.makedirs(DATA_DIR, exist_ok=True)
STOP_FILE = DATA_DIR / "stop.txt"
STATE_FILE = DATA_DIR / "turbo_state.json"


def get_prices(days_back: int = 400, start=None, end=None, tickers=None) -> pd.DataFrame:
    """
    Fetch price data for all required tickers.
    
    Args:
        days_back: Number of calendar days of history to fetch
        start: Optional start date (datetime or string)
        end: Optional end date (datetime or string)
        tickers: Optional list of tickers to fetch (defaults to strategy tickers)
        
    Returns:
        DataFrame with closing prices for all tickers
    """
    if end is None:
        end = dt.datetime.now(tz=TZ)
    
    if start is None:
        start = end - dt.timedelta(days=days_back)
    
    if tickers is None:
        tickers = [
            TICKERS["SRC"], "TLT", TICKERS["UP"], TICKERS["DN"], 
            TICKERS["BOND"], TICKERS["CASH"], TICKERS["VIX"]
        ]
    
    print(f"Fetching price data for {tickers}")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    
    # Get closing prices
    if "Close" in df.columns:
        prices = df["Close"]
    elif df.columns.nlevels > 1 and "Close" in df.columns.levels[0]:
        prices = df["Close"]
    else:
        prices = df
    
    # Forward fill missing data
    prices = prices.ffill().bfill()
    
    return prices


def _compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(series: pd.Series,
                  fast: int = MACD_FAST,
                  slow: int = MACD_SLOW,
                  signal: int = MACD_SIGNAL) -> tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def choose_asset(prices: pd.DataFrame) -> str:
    """
    Choose the best asset based on momentum signals.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        Ticker symbol of the selected asset
    """
    # Calculate QQQ momentum for two timeframes
    mom_short = prices[TICKERS["SRC"]].iloc[-1] / prices[TICKERS["SRC"]].iloc[-MOM_DAYS[0]-1] - 1
    mom_long = prices[TICKERS["SRC"]].iloc[-1] / prices[TICKERS["SRC"]].iloc[-MOM_DAYS[1]-1] - 1

    # Compute RSI & MACD
    rsi = _compute_rsi(prices[TICKERS["SRC"]]).iloc[-1]
    macd_line, macd_signal_line = _compute_macd(prices[TICKERS["SRC"]])
    macd_val = macd_line.iloc[-1]
    macd_sig = macd_signal_line.iloc[-1]

    # LONG condition – leveraged long QQQ
    if (
        mom_short > 0 and mom_long > 0 and  # positive momentum
        rsi > RSI_UPPER and                # RSI bullish
        macd_val > macd_sig                # MACD bullish crossover / above signal
    ):
        return TICKERS["UP"]

    # SHORT condition – leveraged short QQQ
    if (
        mom_short < 0 and mom_long < 0 and  # negative momentum
        rsi < RSI_LOWER and                # RSI bearish
        macd_val < macd_sig                # MACD bearish crossover / below signal
    ):
        return TICKERS["DN"]

    # Bond Shield: If Treasury momentum is positive -> TMF
    bond_mom = prices["TLT"].iloc[-1] / prices["TLT"].shift(BOND_DAYS).iloc[-1] - 1
    if bond_mom > 0:
        return TICKERS["BOND"]
    
    # Default to cash
    return TICKERS["CASH"]


def calculate_atr(series: pd.Series, n: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for a price series.
    
    Args:
        series: Price series
        n: Lookback period for ATR calculation
        
    Returns:
        ATR value
    """
    # Simplified ATR calculation using percentage changes
    return pd.Series(series).pct_change().abs().rolling(n).mean() * series.iloc[-1]


def check_crash_conditions(prices: pd.DataFrame) -> bool:
    """
    Check if crash conditions exist based on VIX and QQQ returns.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        True if crash conditions exist, False otherwise
    """
    # Check VIX level
    vix_level = prices[TICKERS["VIX"]].iloc[-1]
    
    # Check 5-day QQQ return
    five_day_return = prices[TICKERS["SRC"]].pct_change(5).iloc[-1]
    
    # Return True if both conditions are met
    return (vix_level > VIX_THRESHOLD) and (five_day_return < CRASH_THRESHOLD)


def save_stop_price(asset: str, stop_price: float) -> None:
    """
    Save stop price to file for daily check.
    
    Args:
        asset: Ticker symbol
        stop_price: Stop price level
    """
    with open(STOP_FILE, "w") as f:
        f.write(f"{asset},{stop_price}")


def check_stop_hit(api: tradeapi.REST) -> bool:
    """
    Check if stop price has been hit.
    
    Args:
        api: Alpaca API client
        
    Returns:
        True if stop was hit, False otherwise
    """
    if not os.path.exists(STOP_FILE):
        return False
        
    with open(STOP_FILE, "r") as f:
        asset, stop_price = f.read().strip().split(",")
        stop_price = float(stop_price)
    
    # Get current price
    current_price = float(api.get_latest_trade(asset).price)
    
    # Check if stop is hit
    stop_hit = False
    if asset == TICKERS["UP"]:
        # For TQQQ, stop is hit if price falls below stop
        stop_hit = current_price < stop_price
    elif asset == TICKERS["DN"]:
        # For SQQQ, stop is hit if price rises above stop
        stop_hit = current_price > stop_price
    
    # For TMF, we'll use the same logic as TQQQ
    elif asset == TICKERS["BOND"]:
        stop_hit = current_price < stop_price
    
    return stop_hit


def get_alpaca_api() -> tradeapi.REST:
    """
    Initialize Alpaca API client.
    
    Returns:
        Alpaca API client
    """
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    return tradeapi.REST(api_key, api_secret, base_url)
