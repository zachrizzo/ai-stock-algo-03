#!/usr/bin/env python
"""
Backtesting framework for the Turbo-Rotational QQQ strategy.

This module implements a comprehensive backtest for the Turbo QT strategy,
including:
- Monte Carlo simulation with randomized start dates
- Transaction costs and slippage
- Detailed performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional

from .turbo_qt import (
    get_prices, choose_asset, calculate_atr, check_crash_conditions,
    TICKERS, VOL_TARGET, ATR_MULT, TZ, MOM_DAYS, BOND_DAYS, 
    VIX_THRESHOLD, CRASH_THRESHOLD, HEDGE_WEIGHT, KILL_DD, COOLDOWN_WEEKS
)


class TurboBacktester:
    """Backtester for the Turbo-4 strategy."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100.0,
        trading_days: str = "mon",  # 'mon', 'all'
    ):
        """
        Initialize the backtester.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital
            trading_days: Trading frequency ('mon' for Monday-only, 'all' for daily)
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.trading_days = trading_days
        
        # Trading day mapping
        self.trading_days_map = {
            'mon': {0},       # Monday only
            'all': set(range(5))  # All weekdays
        }
        
        # Results storage
        self.prices = None
        self.equity_curve = None
        self.trades = []
        self.allocations = []
        self.stats = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load historical price data for backtesting."""
        # Calculate start date with buffer for calculations
        buffer_start = self.start_date - pd.Timedelta(days=365)  # 1 year buffer
        
        # Tickers needed for backtesting
        tickers = [
            TICKERS["SRC"], "TLT", TICKERS["UP"], TICKERS["DN"], 
            TICKERS["BOND"], TICKERS["CASH"], TICKERS["VIX"]
        ]
        
        print(f"Loading historical data for {tickers} from {buffer_start} to {self.end_date}")
        
        # Download data
        df = get_prices(tickers, start=buffer_start, end=self.end_date)
        
        # Get closing prices
        if "Close" in df.columns:
            self.prices = df["Close"]
        elif df.columns.nlevels > 1 and "Close" in df.columns.levels[0]:
            self.prices = df["Close"]
        else:
            self.prices = df
        
        # Forward fill missing values
        self.prices = self.prices.ffill().bfill()
        
        # Make sure we have enough data
        if len(self.prices) < 252:
            raise ValueError(f"Not enough historical data: {len(self.prices)} days")
        
        # Trim to actually have data for our required tickers
        for ticker in [TICKERS["SRC"], "TLT", TICKERS["VIX"]]:
            if ticker not in self.prices.columns:
                raise ValueError(f"Missing required ticker: {ticker}")
    
    def _choose_asset(self, date: pd.Timestamp) -> str:
        """
        Choose asset based on strategy rules for a specific date.
        
        Args:
            date: Date to evaluate
            
        Returns:
            Selected asset ticker
        """
        # Get data up to this date
        prices_subset = self.prices.loc[:date]
        
        # Make sure we have enough data
        if len(prices_subset) < max(MOM_DAYS) + 10:
            return TICKERS["CASH"]
        
        # Calculate QQQ momentum for two timeframes
        mom21 = prices_subset[TICKERS["SRC"]].iloc[-1] / prices_subset[TICKERS["SRC"]].iloc[-MOM_DAYS[0]-1] - 1
        mom63 = prices_subset[TICKERS["SRC"]].iloc[-1] / prices_subset[TICKERS["SRC"]].iloc[-MOM_DAYS[1]-1] - 1
        
        # Risk-On: Both momentum signals positive -> TQQQ
        if mom21 > 0 and mom63 > 0:
            return TICKERS["UP"]
        
        # Risk-Off: Both momentum signals negative -> SQQQ
        if mom21 < 0 and mom63 < 0:
            return TICKERS["DN"]
        
        # Bond Shield: If Treasury momentum is positive -> TMF
        if "TLT" in prices_subset.columns:
            bond_mom = prices_subset["TLT"].iloc[-1] / prices_subset["TLT"].iloc[-BOND_DAYS-1] - 1
            if bond_mom > 0:
                return TICKERS["BOND"]
        
        # Default to cash
        return TICKERS["CASH"]
    
    def _check_crash_conditions(self, date: pd.Timestamp) -> bool:
        """
        Check if crash conditions exist for a specific date.
        
        Args:
            date: Date to evaluate
            
        Returns:
            True if crash conditions exist, False otherwise
        """
        # Get data up to this date
        prices_subset = self.prices.loc[:date]
        
        # Make sure we have enough data
        if len(prices_subset) < 10 or TICKERS["VIX"] not in prices_subset.columns:
            return False
        
        # Check VIX level
        vix_level = prices_subset[TICKERS["VIX"]].iloc[-1]
        
        # Check 5-day QQQ return
        if len(prices_subset) < 6:
            return False
            
        five_day_return = prices_subset[TICKERS["SRC"]].iloc[-1] / prices_subset[TICKERS["SRC"]].iloc[-6] - 1
        
        # Return True if both conditions are met
        return (vix_level > VIX_THRESHOLD) and (five_day_return < CRASH_THRESHOLD)
    
    def _calculate_atr(self, ticker: str, date: pd.Timestamp, n: int = 14) -> float:
        """
        Calculate ATR for a ticker on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date to evaluate
            n: ATR period
            
        Returns:
            ATR value
        """
        # Get data up to this date
        prices_subset = self.prices.loc[:date][ticker]
        
        # Make sure we have enough data
        if len(prices_subset) < n + 5:
            return 0.0
        
        # Calculate ATR
        returns = prices_subset.pct_change().abs().rolling(n).mean()
        return returns.iloc[-1] * prices_subset.iloc[-1]
    
    def _check_stop_hit(self, ticker: str, stop_price: float, date: pd.Timestamp) -> bool:
        """
        Check if stop price is hit on a specific date.
        
        Args:
            ticker: Ticker symbol
            stop_price: Stop price
            date: Date to evaluate
            
        Returns:
            True if stop is hit, False otherwise
        """
        # Get current price
        if ticker not in self.prices.columns or date not in self.prices.index:
            return False
            
        current_price = self.prices.loc[date, ticker]
        
        # Check if stop is hit
        if ticker == TICKERS["UP"] or ticker == TICKERS["BOND"]:
            # For long positions, stop is hit if price falls below stop
            return current_price < stop_price
        elif ticker == TICKERS["DN"]:
            # For short positions, stop is hit if price rises above stop
            return current_price > stop_price
        
        return False
    
    def run_backtest(self):
        """Run the backtest."""
        print(f"Running backtest from {self.start_date} to {self.end_date}")
        
        # Initialize variables
        equity = self.initial_capital
        peak_equity = equity
        current_asset = TICKERS["CASH"]
        current_weight = 1.0
        stop_price = None
        in_cooldown = False
        cooldown_until = None
        last_trade_date = None
        min_hold_days = 5  # Set minimum hold period to 5 days
        equity_curve = []
        all_dates = []
        trades = []
        allocations = []
        
        # Trade tracking for PDT rule compliance
        trade_dates = set()
        weekly_trade_count = {}  # Week number -> trade count
        
        # Get all trading dates in range
        trading_dates = self.prices.loc[self.start_date:self.end_date].index
        
        # Filter for specified trading days (e.g., only Mondays)
        if self.trading_days in self.trading_days_map:
            allowed_days = self.trading_days_map[self.trading_days]
            trading_dates = [d for d in trading_dates if d.weekday() in allowed_days]
        
        print(f"Total trading days: {len(trading_dates)}")
        
        # Position leverage multiplier (simulate leveraged ETF performance)
        leveraged_return_multiplier = {
            TICKERS["UP"]: 2.8,    # Slightly less than 3x due to decay
            TICKERS["DN"]: 2.8,    # Slightly less than 3x due to decay
            TICKERS["BOND"]: 2.8,  # Slightly less than 3x due to decay
            TICKERS["CASH"]: 1.0   # No leverage for cash
        }
        
        # Daily cost due to expense ratios (annualized expense ratio / 252 trading days)
        daily_cost = {
            TICKERS["UP"]: 0.0009,    # 0.95% annual expense ratio
            TICKERS["DN"]: 0.0009,    # 0.95% annual expense ratio
            TICKERS["BOND"]: 0.0004,  # 0.46% annual expense ratio
            TICKERS["CASH"]: 0.0001   # 0.14% annual expense ratio
        }
        
        # Run through each trading date
        for i, date in enumerate(trading_dates):
            # Track weekly trade count (for PDT rule)
            week_num = date.isocalendar()[1]
            year = date.year
            week_key = f"{year}-{week_num}"
            if week_key not in weekly_trade_count:
                weekly_trade_count[week_key] = 0
            
            # Update peak equity for trailing high watermark
            peak_equity = max(peak_equity, equity)
            
            # Check if we're in cooldown period
            if in_cooldown and date <= cooldown_until:
                current_asset = TICKERS["CASH"]
                current_weight = 1.0
            elif in_cooldown and date > cooldown_until:
                in_cooldown = False
                cooldown_until = None
            
            # Check stop-loss if we have one and minimum hold period passed
            # Only check if we have trades available this week (PDT rule)
            if (stop_price is not None and 
                current_asset in [TICKERS["UP"], TICKERS["DN"], TICKERS["BOND"]] and
                (last_trade_date is None or (date - last_trade_date).days >= min_hold_days) and
                weekly_trade_count[week_key] < 3):
                
                if self._check_stop_hit(current_asset, stop_price, date):
                    # Stop hit - move to cash
                    print(f"{date}: STOP HIT for {current_asset} at {stop_price:.2f}")
                    trades.append({
                        'date': date,
                        'type': 'stop_loss',
                        'asset': current_asset,
                        'direction': 'sell',
                        'price': self.prices.loc[date, current_asset],
                        'equity': equity
                    })
                    
                    current_asset = TICKERS["CASH"]
                    current_weight = 1.0
                    stop_price = None
                    last_trade_date = date
                    weekly_trade_count[week_key] += 1
                    trade_dates.add(date)
            
            # Monday rebalancing or first day
            # Only rebalance if we have trades available this week (PDT rule)
            is_monday = date.weekday() == 0
            if (i == 0 or is_monday) and weekly_trade_count[week_key] < 3:
                # Check minimum hold period
                min_hold_met = True
                if last_trade_date is not None:
                    days_since_last_trade = (date - last_trade_date).days
                    min_hold_met = days_since_last_trade >= min_hold_days
                
                if min_hold_met and not in_cooldown:
                    # Choose asset
                    new_asset = self._choose_asset(date)
                    
                    # Check for crash conditions
                    crash_conditions = self._check_crash_conditions(date)
                    
                    # Calculate volatility for position sizing
                    if new_asset != TICKERS["CASH"] and new_asset in self.prices.columns:
                        # Get data up to this date
                        prices_subset = self.prices.loc[:date][new_asset]
                        
                        # Calculate volatility
                        if len(prices_subset) > 20:
                            # Calculate realized volatility using returns
                            returns = prices_subset.pct_change().dropna()
                            if len(returns) >= 20:
                                sigma = returns.tail(20).std() * np.sqrt(252)
                                weight = min(1.0, VOL_TARGET / sigma) if sigma > 0 else 0.5
                            else:
                                weight = 0.5  # Default if not enough return data
                        else:
                            weight = 0.5  # Conservative default
                    else:
                        weight = 1.0
                    
                    # Apply hedge if needed
                    hedge_needed = (new_asset != TICKERS["DN"] and crash_conditions)
                    
                    # Adjust weight for hedge
                    if hedge_needed:
                        weight = weight * (1.0 - HEDGE_WEIGHT)
                    
                    # Calculate allocations
                    allocations_dict = {}
                    
                    if new_asset != TICKERS["CASH"]:
                        allocations_dict[new_asset] = weight
                        
                        if hedge_needed:
                            allocations_dict[TICKERS["DN"]] = HEDGE_WEIGHT
                        
                        allocations_dict[TICKERS["CASH"]] = max(0.0, 1.0 - sum(allocations_dict.values()))
                    else:
                        allocations_dict[TICKERS["CASH"]] = 1.0
                    
                    # Record allocation
                    allocations.append({
                        'date': date,
                        'allocations': allocations_dict,
                        'equity': equity
                    })
                    
                    # Record trades if asset changed
                    trades_executed = 0
                    if new_asset != current_asset:
                        if current_asset != TICKERS["CASH"]:
                            trades.append({
                                'date': date,
                                'type': 'rebalance',
                                'asset': current_asset,
                                'direction': 'sell',
                                'price': self.prices.loc[date, current_asset],
                                'equity': equity
                            })
                            trades_executed += 1
                        
                        if new_asset != TICKERS["CASH"]:
                            trades.append({
                                'date': date,
                                'type': 'rebalance',
                                'asset': new_asset,
                                'direction': 'buy',
                                'price': self.prices.loc[date, new_asset],
                                'equity': equity
                            })
                            trades_executed += 1
                    
                    # Add hedge if needed (counts as a trade)
                    if hedge_needed and not (current_asset == TICKERS["DN"] or current_asset == TICKERS["CASH"]):
                        trades.append({
                            'date': date,
                            'type': 'hedge',
                            'asset': TICKERS["DN"],
                            'direction': 'buy',
                            'price': self.prices.loc[date, TICKERS["DN"]],
                            'equity': equity
                        })
                        trades_executed += 1
                    
                    # Update trade count for PDT rule
                    if trades_executed > 0:
                        weekly_trade_count[week_key] += min(trades_executed, 3)  # Cap at 3 trades per week
                        trade_dates.add(date)
                    
                    # Update current position
                    current_asset = new_asset
                    current_weight = weight
                    
                    # Calculate and save stop price
                    if current_asset in [TICKERS["UP"], TICKERS["DN"], TICKERS["BOND"]] and current_asset in self.prices.columns:
                        # Calculate ATR
                        atr_value = self._calculate_atr(current_asset, date)
                        
                        # Calculate stop price
                        if current_asset == TICKERS["UP"] or current_asset == TICKERS["BOND"]:
                            # For long positions, stop is below current price
                            stop_price = self.prices.loc[date, current_asset] - (ATR_MULT * atr_value)
                        else:
                            # For short positions, stop is above current price
                            stop_price = self.prices.loc[date, current_asset] + (ATR_MULT * atr_value)
                    else:
                        stop_price = None
                    
                    # Update last trade date
                    last_trade_date = date
            
            # Calculate returns and update equity
            if i < len(trading_dates) - 1:
                next_date = trading_dates[i+1]
                daily_return = 0.0
                
                # Calculate weighted return based on allocations
                if current_asset == TICKERS["CASH"]:
                    if TICKERS["CASH"] in self.prices.columns:
                        daily_return = (self.prices.loc[next_date, TICKERS["CASH"]] / self.prices.loc[date, TICKERS["CASH"]] - 1)
                        # Apply cash expense ratio
                        daily_return -= daily_cost[TICKERS["CASH"]]
                else:
                    # Calculate return for main asset (apply leverage multiplier)
                    if current_asset in self.prices.columns:
                        asset_weight = current_weight
                        base_asset_return = (self.prices.loc[next_date, current_asset] / self.prices.loc[date, current_asset] - 1)
                        
                        # Apply leverage multiplier if it's a leveraged ETF
                        if current_asset in [TICKERS["UP"], TICKERS["DN"], TICKERS["BOND"]]:
                            # For SQQQ (inverse ETF), returns work opposite
                            if current_asset == TICKERS["DN"]:
                                # For inverse ETF, we need to simulate the inverse return
                                # Use the base QQQ return to simulate
                                if TICKERS["SRC"] in self.prices.columns:
                                    qqq_return = (self.prices.loc[next_date, TICKERS["SRC"]] / self.prices.loc[date, TICKERS["SRC"]] - 1)
                                    # Negative 2.8x of QQQ return for SQQQ
                                    asset_return = -1 * leveraged_return_multiplier[current_asset] * qqq_return
                                else:
                                    # If QQQ data not available, approximate
                                    asset_return = leveraged_return_multiplier[current_asset] * base_asset_return
                            else:
                                # For regular leveraged ETFs
                                asset_return = leveraged_return_multiplier[current_asset] * base_asset_return
                        else:
                            asset_return = base_asset_return
                            
                        # Apply expense ratio cost
                        if current_asset in daily_cost:
                            asset_return -= daily_cost[current_asset]
                            
                        daily_return += asset_weight * asset_return
                    
                    # Calculate return for hedge if active
                    if hedge_needed and TICKERS["DN"] in self.prices.columns:
                        # For inverse ETF hedge, use QQQ return if available
                        if TICKERS["SRC"] in self.prices.columns:
                            qqq_return = (self.prices.loc[next_date, TICKERS["SRC"]] / self.prices.loc[date, TICKERS["SRC"]] - 1)
                            # Negative 2.8x of QQQ return for SQQQ hedge
                            hedge_return = -1 * leveraged_return_multiplier[TICKERS["DN"]] * qqq_return
                        else:
                            base_hedge_return = (self.prices.loc[next_date, TICKERS["DN"]] / self.prices.loc[date, TICKERS["DN"]] - 1)
                            hedge_return = leveraged_return_multiplier[TICKERS["DN"]] * base_hedge_return
                            
                        # Apply expense ratio cost
                        hedge_return -= daily_cost[TICKERS["DN"]]
                        
                        daily_return += HEDGE_WEIGHT * hedge_return
                    
                    # Calculate return for cash portion
                    cash_weight = max(0.0, 1.0 - current_weight - (HEDGE_WEIGHT if hedge_needed else 0.0))
                    if cash_weight > 0 and TICKERS["CASH"] in self.prices.columns:
                        cash_return = (self.prices.loc[next_date, TICKERS["CASH"]] / self.prices.loc[date, TICKERS["CASH"]] - 1)
                        # Apply cash expense ratio
                        cash_return -= daily_cost[TICKERS["CASH"]]
                        daily_return += cash_weight * cash_return
                
                # Update equity
                equity *= (1 + daily_return)
                equity_curve.append(equity)
                all_dates.append(next_date)
                
                # Check for kill-switch (max drawdown)
                drawdown = (equity / peak_equity) - 1
                if drawdown < -KILL_DD and not in_cooldown:
                    print(f"{next_date}: KILL-SWITCH TRIGGERED: Drawdown {drawdown:.2%}")
                    
                    # Record trade if we have trades available this week
                    if weekly_trade_count[week_key] < 3:
                        trades.append({
                            'date': next_date,
                            'type': 'kill_switch',
                            'asset': current_asset,
                            'direction': 'sell',
                            'price': self.prices.loc[next_date, current_asset],
                            'equity': equity
                        })
                        weekly_trade_count[week_key] += 1
                        trade_dates.add(next_date)
                    
                    # Set cooldown period
                    in_cooldown = True
                    cooldown_until = next_date + pd.Timedelta(weeks=COOLDOWN_WEEKS)
                    
                    # Move to cash
                    current_asset = TICKERS["CASH"]
                    current_weight = 1.0
                    stop_price = None
        
        # Store results
        self.equity_curve = pd.Series(equity_curve, index=all_dates)
        self.trades = trades
        self.allocations = allocations
        
        # Calculate statistics
        self._calculate_stats()
        
        # Add trade count statistics
        self.stats['unique_trade_dates'] = len(trade_dates)
        self.stats['weekly_trades_avg'] = sum(weekly_trade_count.values()) / len(weekly_trade_count) if weekly_trade_count else 0
    
    def _calculate_stats(self):
        """Calculate performance statistics."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data available")
            return
        
        # Basic metrics
        total_return = self.equity_curve.iloc[-1] / self.initial_capital - 1
        
        # Calculate CAGR
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate maximum drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / peak - 1)
        max_drawdown = drawdown.min()
        
        # Calculate volatility
        daily_returns = self.equity_curve.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (cagr / volatility) if volatility > 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (cagr / downside_deviation) if downside_deviation > 0 else 0
        
        # Calculate win rate
        winning_days = len(daily_returns[daily_returns > 0])
        losing_days = len(daily_returns[daily_returns <= 0])
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # Calculate trade metrics
        trade_count = len(self.trades)
        avg_trade_size = 0
        if trade_count > 0:
            avg_trade_size = sum(trade['equity'] for trade in self.trades) / trade_count
        
        # Store statistics
        self.stats = {
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve.iloc[-1],
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'trades_per_year': trade_count / years if years > 0 else 0,
            'avg_trade_size': avg_trade_size
        }
    
    def print_results(self):
        """Print backtest results."""
        if not self.stats:
            print("No statistics available")
            return
        
        print("\nBacktest Results:")
        print(f"Start Date: {self.stats['start_date']}")
        print(f"End Date: {self.stats['end_date']}")
        print(f"Initial Capital: ${self.stats['initial_capital']:.2f}")
        print(f"Final Capital: ${self.stats['final_capital']:.2f}")
        print(f"Total Return: {self.stats['total_return']:.2%}")
        print(f"CAGR: {self.stats['cagr']:.2%}")
        print(f"Volatility: {self.stats['volatility']:.2%}")
        print(f"Max Drawdown: {self.stats['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.stats['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.stats['sortino_ratio']:.2f}")
        print(f"Win Rate: {self.stats['win_rate']:.2%}")
        print(f"Number of Trades: {self.stats['trade_count']}")
        print(f"Trades per Year: {self.stats['trades_per_year']:.1f}")
    
    def plot_results(self):
        """Plot equity curve and drawdowns."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data available")
            return
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        axs[0].plot(self.equity_curve.index, self.equity_curve)
        axs[0].set_title('Equity Curve')
        axs[0].set_ylabel('Equity ($)')
        axs[0].grid(True)
        
        # Calculate and plot drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / peak - 1) * 100  # Convert to percentage
        
        axs[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axs[1].set_title('Drawdown (%)')
        axs[1].set_ylabel('Drawdown (%)')
        axs[1].grid(True)
        axs[1].set_ylim(bottom=drawdown.min()*1.1, top=2)  # Set y-axis limits
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Add legend and title
        fig.suptitle(f'Turbo-4 Strategy ({self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")})', 
                     fontsize=16)
        
        # Annotate CAGR and max drawdown
        stats_text = (f'CAGR: {self.stats["cagr"]:.2%}\n'
                     f'Max DD: {self.stats["max_drawdown"]:.2%}\n'
                     f'Sharpe: {self.stats["sharpe_ratio"]:.2f}\n'
                     f'Trades/Year: {self.stats["trades_per_year"]:.1f}')
        
        axs[0].annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                       fontsize=10, backgroundcolor='white',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function for running as a command-line script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Turbo-4 Strategy Backtest")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-03-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100.0, help="Initial capital")
    parser.add_argument("--trading-days", type=str, default="mon", choices=["mon", "all"],
                        help="Trading frequency (mon=Monday only, all=daily)")
    parser.add_argument("--no-plot", action="store_true", help="Don't show plots")
    
    args = parser.parse_args()
    
    # Create and run backtester
    backtester = TurboBacktester(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        trading_days=args.trading_days
    )
    
    backtester.run_backtest()
    backtester.print_results()
    
    if not args.no_plot:
        backtester.plot_results()


if __name__ == "__main__":
    main()
