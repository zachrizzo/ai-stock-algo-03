"""
Ensemble Micro Trend Strategy.

A robust strategy that uses:
1. Ensemble momentum signals (3, 6, 12-month)
2. Volatility targeting
3. Crash protection overlay
4. Minimum holding period to reduce churn
5. Stop-loss with cooldown mechanism
"""
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from stock_trader_o3_algo.config.settings import (
    RISK_ON, RISK_OFF, BOND_ETF, HEDGE_ETF, CASH_ETF,
    TARGET_VOL, MIN_HOLD_DAYS, CRASH_VIX_THRESHOLD, CRASH_DROP_THRESHOLD,
    HEDGE_WEIGHT, KILL_DD, COOLDOWN_WEEKS, LOOKBACK_DAYS
)


def calculate_momentum_votes(prices: pd.DataFrame, asset: str, lookbacks: List[int], date: Optional[pd.Timestamp] = None) -> int:
    """
    Calculate momentum votes based on multiple lookback periods.
    
    Args:
        prices: DataFrame with price data
        asset: Symbol of the asset to check
        lookbacks: List of lookback periods in trading days
        date: Date to check (defaults to latest date in prices)
        
    Returns:
        Number of positive momentum votes
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
    
    # Ensure we're using data available up to the specified date
    prices_subset = prices.loc[:date]
    
    # Count positive momentum across different lookbacks
    votes = 0
    for lookback in lookbacks:
        if len(prices_subset) > lookback:
            momentum = prices_subset[asset].iloc[-1] / prices_subset[asset].iloc[-lookback-1] - 1
            if momentum > 0:
                votes += 1
    
    return votes


def calculate_volatility(prices: pd.DataFrame, asset: str, window: int = 20, date: Optional[pd.Timestamp] = None) -> float:
    """
    Calculate annualized volatility of an asset.
    
    Args:
        prices: DataFrame with price data
        asset: Symbol of the asset to check
        window: Window for volatility calculation in trading days
        date: Date to check (defaults to latest date in prices)
        
    Returns:
        Annualized volatility
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
    
    # Ensure we're using data available up to the specified date
    prices_subset = prices.loc[:date]
    
    # Need enough data for calculation
    if len(prices_subset) <= window:
        return 0.0
    
    # Calculate daily returns and annualized volatility
    returns = prices_subset[asset].pct_change().dropna().iloc[-window:]
    return returns.std() * np.sqrt(252)


def check_crash_conditions(prices: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> bool:
    """
    Check if crash conditions exist based on VIX and QQQ weekly returns.
    
    Args:
        prices: DataFrame with price data
        date: Date to check (defaults to latest date in prices)
        
    Returns:
        True if crash conditions exist, False otherwise
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
    
    # Ensure we're using data available up to the specified date
    prices_subset = prices.loc[:date]
    
    # Need at least 5 days of data for weekly return
    if len(prices_subset) < 6 or '^VIX' not in prices_subset.columns:
        return False
    
    # Get current VIX value
    vix_now = prices_subset['^VIX'].iloc[-1]
    
    # Calculate weekly return for QQQ
    weekly_return = prices_subset[RISK_ON].pct_change(5).iloc[-1]
    
    # Check crash conditions
    return (vix_now > CRASH_VIX_THRESHOLD) and (weekly_return < CRASH_DROP_THRESHOLD)


def check_stop_loss(equity_curve: pd.Series, date: pd.Timestamp, cooldown_end_date: Optional[pd.Timestamp] = None) -> Tuple[bool, Optional[pd.Timestamp]]:
    """
    Check if a stop-loss should be triggered based on drawdown from peak equity.
    
    Args:
        equity_curve: Series with portfolio equity values
        date: Current date to check
        cooldown_end_date: Date when the cooldown period ends (if in cooldown)
        
    Returns:
        Tuple of (trigger_stop_loss, new_cooldown_end_date)
    """
    # If we're already in a cooldown period, check if it's over
    if cooldown_end_date is not None:
        if date <= cooldown_end_date:
            return True, cooldown_end_date  # Still in cooldown
        else:
            cooldown_end_date = None  # Cooldown is over
    
    # Get relevant portion of equity curve (up to current date)
    equity_subset = equity_curve.loc[:date]
    
    # Need at least 2 days of data to calculate drawdown
    if len(equity_subset) < 2:
        return False, None
    
    # Calculate rolling peak (all-time high)
    rolling_peak = equity_subset.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (equity_subset / rolling_peak - 1).iloc[-1]
    
    # Check if drawdown exceeds threshold
    if drawdown < -KILL_DD:
        print(f"STOP LOSS TRIGGERED: Drawdown of {drawdown:.2%} exceeds threshold of -{KILL_DD:.2%}")
        
        # Calculate cooldown end date
        # Find all dates in the index that are after the current date
        future_dates = equity_curve.index[equity_curve.index > date]
        
        if len(future_dates) >= COOLDOWN_WEEKS * 5:  # ~5 trading days per week
            cooldown_end_date = future_dates[COOLDOWN_WEEKS * 5 - 1]
        elif len(future_dates) > 0:
            cooldown_end_date = future_dates[-1]
        else:
            # No future dates available, use a date far in the future
            cooldown_end_date = date + pd.Timedelta(weeks=COOLDOWN_WEEKS)
        
        return True, cooldown_end_date
    
    return False, cooldown_end_date


def choose_regime(prices: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> str:
    """
    Choose investment regime based on ensemble momentum signals.
    
    Args:
        prices: DataFrame with price data
        date: Date to check (defaults to latest date in prices)
        
    Returns:
        Regime identifier ("RISK", "BOND", or "CASH")
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
    
    # Define lookback periods (approximately 3, 6, and 12 months)
    lookbacks = [63, 126, 252]
    
    # Get momentum votes for QQQ
    risk_votes = calculate_momentum_votes(prices, RISK_ON, lookbacks, date)
    
    # If QQQ has at least 2 positive votes, go risk-on
    if risk_votes >= 2:
        return "RISK"
    
    # Get momentum votes for TLT
    bond_votes = calculate_momentum_votes(prices, BOND_ETF, lookbacks, date)
    
    # If TLT has at least 2 positive votes, go to bonds
    if bond_votes >= 2:
        return "BOND"
    
    # Otherwise, go to cash
    return "CASH"


def get_portfolio_allocation(
    prices: pd.DataFrame, 
    date: Optional[pd.Timestamp] = None,
    equity: float = 100.0,
    equity_peak: Optional[float] = None,
    equity_curve: Optional[pd.Series] = None,
    stop_loss_cooldown_end_date: Optional[pd.Timestamp] = None,
    last_trade_date: Optional[pd.Timestamp] = None
) -> Dict[str, float]:
    """
    Calculate portfolio allocation based on ensemble strategy.
    
    Args:
        prices: DataFrame with price data
        date: Date to use for calculation (defaults to latest date in prices)
        equity: Current portfolio equity value
        equity_peak: Peak equity value (for drawdown calculation)
        equity_curve: Full equity curve (for stop-loss calculation)
        stop_loss_cooldown_end_date: End date for stop-loss cooldown period
        last_trade_date: Date of last trade (for minimum hold period)
        
    Returns:
        Dictionary with asset symbols as keys and dollar allocations as values
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
    
    # Use current equity as peak if not specified
    if equity_peak is None:
        equity_peak = equity
    
    # Check for stop-loss condition if equity curve is provided
    in_stop_loss = False
    if equity_curve is not None:
        in_stop_loss, stop_loss_cooldown_end_date = check_stop_loss(
            equity_curve, date, stop_loss_cooldown_end_date
        )
        
        if in_stop_loss:
            print(f"In stop-loss cooldown until {stop_loss_cooldown_end_date}")
            return {CASH_ETF: equity}
    
    # Check if minimum hold period has elapsed
    if last_trade_date is not None and (date - last_trade_date).days < MIN_HOLD_DAYS:
        # Continue with existing allocation by returning None
        print(f"Minimum hold period not met. Last trade was {(date - last_trade_date).days} days ago.")
        return {}
    
    # Choose investment regime
    regime = choose_regime(prices, date)
    
    # Determine target asset based on regime
    if regime == "RISK":
        target_asset = RISK_ON
    elif regime == "BOND":
        target_asset = BOND_ETF
    else:  # Cash regime
        return {CASH_ETF: equity}
    
    # Calculate volatility for target asset
    sigma = calculate_volatility(prices, target_asset, 20, date)
    
    # Calculate position weight using volatility targeting
    weight = min(1.0, TARGET_VOL / sigma) if sigma > 0 else 0.0
    
    # Check crash conditions
    hedge_activated = False
    if target_asset == RISK_ON and check_crash_conditions(prices, date):
        hedge_activated = True
        weight = max(0.0, weight - HEDGE_WEIGHT)
    
    # Create allocation dictionary
    allocations = {
        target_asset: weight * equity,
        CASH_ETF: equity * (1.0 - weight)
    }
    
    # Add hedge if crash conditions exist
    if hedge_activated:
        allocations[HEDGE_ETF] = HEDGE_WEIGHT * equity
        allocations[CASH_ETF] = max(0.0, allocations[CASH_ETF] - (HEDGE_WEIGHT * equity))
    
    # Remove any zero or negative allocations
    allocations = {k: v for k, v in allocations.items() if v > 0.0}
    
    # Always include cash for clarity even if zero
    if CASH_ETF not in allocations:
        allocations[CASH_ETF] = 0.0
    
    return allocations
