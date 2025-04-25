#!/usr/bin/env python
"""
Backtest the Ensemble Micro Trend Strategy.

Usage:
  python ensemble_backtest.py --start-date 2010-01-01 --end-date 2024-03-31 --trading-days mon

Options:
  --start-date YYYY-MM-DD  Start date for the backtest
  --end-date YYYY-MM-DD    End date for the backtest
  --trading-days DAYS      Days to trade on: mon, tue, wed, thu, fri, mwf (Monday/Wednesday/Friday), or all
  --output FILE            Save results to file
  --plot                   Show performance charts
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from stock_trader_o3_algo.core.ensemble_strategy import (
    get_portfolio_allocation, choose_regime, calculate_volatility
)
from stock_trader_o3_algo.config.settings import (
    RISK_ON, RISK_OFF, BOND_ETF, HEDGE_ETF, CASH_ETF, HISTORY_DAYS,
    BACKTEST_RESULTS_DIR
)


class EnsembleBacktester:
    """Backtester for Ensemble Micro Trend Strategy."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        trading_days: Optional[Set[int]] = None,
        initial_capital: float = 100.0,
    ):
        """
        Initialize the backtester.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            trading_days: Set of weekdays to trade on (0=Monday, 4=Friday)
            initial_capital: Initial capital for the backtest
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.trading_days = trading_days  # Days of the week to trade on
        self.initial_capital = initial_capital
        self.prices = None
        self.allocations = []
        self.equity_curve = None
        self.daily_returns = None
        self.portfolio_stats = None
        
        # Fetch necessary price data
        self.tickers = [RISK_ON, RISK_OFF, BOND_ETF, HEDGE_ETF, CASH_ETF, "^VIX"]
        
        # Calculate how many days of extra history we need for calculations
        # We need at least 252 days (1 year) for the longest momentum lookback
        history_days = max(HISTORY_DAYS, 252)
        
        # Adjust start date to include history
        self.data_start_date = self.start_date - pd.Timedelta(days=history_days)
        
        # Fetch data
        self._fetch_data()
    
    def _fetch_data(self):
        """Fetch price data for the backtest."""
        print(f"Fetching data for {self.tickers} from {self.data_start_date} to {self.end_date}")
        
        # Download data with yfinance
        df = yf.download(self.tickers, start=self.data_start_date, end=self.end_date, 
                         progress=False, auto_adjust=True)
        
        # yfinance now returns adjusted prices directly when auto_adjust=True
        # Handle both multi-column and single-column cases
        if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
            self.prices = df['Close']
        elif isinstance(df, pd.DataFrame) and df.columns.nlevels > 1 and 'Close' in df.columns.levels[0]:
            self.prices = df['Close']
        else:
            # Single ticker case or other structure
            self.prices = df
        
        # Forward fill missing data
        self.prices = self.prices.ffill().bfill()
        
        # Check if we have enough data
        if self.prices.empty:
            raise ValueError("No data returned from Yahoo Finance")
        
        actual_start = self.prices.index[0]
        if actual_start > self.data_start_date:
            print(f"Warning: Limited historical data available before backtest start date. "
                  f"Have data from {actual_start}, but ideally need data from {self.data_start_date} "
                  f"for calculations.")
    
    def run_backtest(self):
        """Run the backtest."""
        # Initialize results
        equity = self.initial_capital
        equity_curve = []
        dates = []
        allocations_history = []
        last_trade_date = None
        stop_loss_cooldown_end_date = None
        
        # Get trading dates
        trading_dates = self.prices.loc[self.start_date:self.end_date].index
        
        # Filter for specified trading days (e.g., only Mondays)
        if self.trading_days is not None:
            trading_dates = [d for d in trading_dates if d.weekday() in self.trading_days]
        
        # Run through each trading date
        for date in trading_dates:
            # Calculate portfolio allocation
            allocation = get_portfolio_allocation(
                prices=self.prices,
                date=date,
                equity=equity,
                equity_peak=max([equity] + equity_curve) if equity_curve else equity,
                equity_curve=pd.Series(equity_curve, index=dates) if equity_curve else None,
                stop_loss_cooldown_end_date=stop_loss_cooldown_end_date,
                last_trade_date=last_trade_date
            )
            
            # Skip if no allocation is returned (minimum hold period)
            if not allocation:
                # Use previous allocation
                if allocations_history:
                    allocation = allocations_history[-1]['allocation']
                else:
                    # Default to cash if no previous allocation
                    allocation = {CASH_ETF: equity}
            else:
                # Update last trade date since we have a new allocation
                last_trade_date = date
            
            # Record regime and allocations
            regime = choose_regime(self.prices, date)
            regime_asset = RISK_ON if regime == "RISK" else BOND_ETF if regime == "BOND" else CASH_ETF
            allocations_history.append({
                'date': date,
                'regime': regime,
                'allocation': allocation,
                'equity': equity
            })
            
            # Calculate returns for the next day or until the next trading date
            next_idx = trading_dates.index(date) + 1
            next_date = trading_dates[next_idx] if next_idx < len(trading_dates) else None
            
            if next_date:
                # Get all dates from current date (exclusive) to next trading date (inclusive)
                subperiod = self.prices.loc[date:next_date].index[1:]
                
                for sub_date in subperiod:
                    daily_return = 0
                    
                    # Calculate weighted return based on allocations
                    for ticker, amount in allocation.items():
                        if ticker in self.prices.columns:
                            weight = amount / equity
                            daily_return += weight * (
                                self.prices[ticker].loc[sub_date] / self.prices[ticker].loc[date] - 1
                            )
                    
                    # Update equity
                    equity *= (1 + daily_return)
                    equity_curve.append(equity)
                    dates.append(sub_date)
            
        # Create equity curve DataFrame
        self.equity_curve = pd.Series(equity_curve, index=dates)
        self.allocations = allocations_history
        
        # Calculate daily returns
        self.daily_returns = self.equity_curve.pct_change().dropna()
        
        # Calculate portfolio statistics
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate portfolio statistics."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return
        
        # Calculate total return
        total_return = self.equity_curve.iloc[-1] / self.initial_capital - 1
        
        # Calculate CAGR (Compound Annual Growth Rate)
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate maximum drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / peak - 1)
        max_drawdown = drawdown.min()
        
        # Calculate volatility (annualized)
        volatility = self.daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (annualized, assuming risk-free rate of 0)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = cagr / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate win/loss metrics
        winning_days = len(self.daily_returns[self.daily_returns > 0])
        losing_days = len(self.daily_returns[self.daily_returns <= 0])
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # Calculate number of trades
        trades = 0
        for i in range(1, len(self.allocations)):
            prev_alloc = set(self.allocations[i-1]['allocation'].keys())
            curr_alloc = set(self.allocations[i]['allocation'].keys())
            
            # Check if allocations have changed
            if prev_alloc != curr_alloc:
                trades += 1
            else:
                # Check if weights have changed significantly
                for ticker in prev_alloc.intersection(curr_alloc):
                    prev_weight = self.allocations[i-1]['allocation'].get(ticker, 0) / self.allocations[i-1]['equity']
                    curr_weight = self.allocations[i]['allocation'].get(ticker, 0) / self.allocations[i]['equity']
                    
                    if abs(prev_weight - curr_weight) > 0.05:  # 5% threshold
                        trades += 1
                        break
        
        # Calculate monthly metrics
        monthly_returns = None
        win_months = 0
        loss_months = 0
        
        if len(self.equity_curve) > 20:  # Only if we have at least a month of data
            # Resample to month-end and calculate returns
            monthly_returns = self.equity_curve.resample('ME').last().pct_change().dropna()
            win_months = len(monthly_returns[monthly_returns > 0])
            loss_months = len(monthly_returns[monthly_returns <= 0])
        
        # Store results
        self.portfolio_stats = {
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve.iloc[-1],
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'win_months': win_months,
            'loss_months': loss_months,
            'number_of_trades': trades
        }
    
    def print_results(self):
        """Print backtest results."""
        if self.portfolio_stats is None:
            print("No backtest results available.")
            return
        
        print("\nBacktest Results:")
        print(f"Start Date: {self.portfolio_stats['start_date']}")
        print(f"End Date: {self.portfolio_stats['end_date']}")
        print(f"Initial Capital: ${self.portfolio_stats['initial_capital']:.2f}")
        print(f"Final Capital: ${self.portfolio_stats['final_capital']:.2f}")
        print(f"Total Return: {self.portfolio_stats['total_return']:.2%}")
        print(f"CAGR: {self.portfolio_stats['cagr']:.2%}")
        print(f"Max Drawdown: {self.portfolio_stats['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {self.portfolio_stats['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.portfolio_stats['sortino_ratio']:.2f}")
        print(f"Number of Trades: {self.portfolio_stats['number_of_trades']}")
    
    def plot_results(self):
        """Plot backtest results."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve available to plot.")
            return
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(self.equity_curve.index, self.equity_curve, linewidth=2)
        axs[0].set_title('Equity Curve')
        axs[0].set_ylabel('Equity ($)')
        axs[0].grid(True)
        
        # Plot drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve / peak - 1) * 100  # Convert to percentage
        axs[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        axs[1].set_title('Drawdown (%)')
        axs[1].set_ylabel('Drawdown (%)')
        axs[1].grid(True)
        
        # Plot regime transitions
        regimes = [item['regime'] for item in self.allocations]
        regime_dates = [item['date'] for item in self.allocations]
        
        # Create a numeric representation of regimes for plotting
        regime_map = {'RISK': 1, 'BOND': 0, 'CASH': -1}
        regime_values = [regime_map.get(r, -1) for r in regimes]
        
        # Plot regime transitions
        axs[2].plot(regime_dates, regime_values, drawstyle='steps-post', linewidth=2)
        axs[2].set_title('Investment Regime')
        axs[2].set_ylabel('Regime')
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels(['CASH', 'BOND', 'RISK'])
        axs[2].grid(True)
        
        # Set common x-axis label and format
        plt.xlabel('Date')
        fig.autofmt_xdate()
        
        # Tight layout and show plot
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: str = None):
        """Save backtest results to file."""
        if self.portfolio_stats is None:
            print("No backtest results available to save.")
            return
        
        if filename is None:
            # Generate a filename based on strategy and dates
            filename = f"ensemble_backtest_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.json"
        
        # Ensure directory exists
        os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
        
        # Prepare data for saving
        results = {
            'stats': self.portfolio_stats,
            'equity_curve': list(zip(self.equity_curve.index.strftime('%Y-%m-%d').tolist(), 
                                     self.equity_curve.tolist())),
            'allocations': self.allocations
        }
        
        # Save to file
        filepath = os.path.join(BACKTEST_RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            # Convert dates to strings for JSON serialization
            serializable_allocations = []
            for alloc in self.allocations:
                serializable_alloc = {
                    'date': alloc['date'].strftime('%Y-%m-%d'),
                    'regime': alloc['regime'],
                    'allocation': alloc['allocation'],
                    'equity': alloc['equity']
                }
                serializable_allocations.append(serializable_alloc)
            
            results['allocations'] = serializable_allocations
            
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest the Ensemble Micro Trend Strategy')
    
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date for the backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date for the backtest (YYYY-MM-DD)')
    
    parser.add_argument('--trading-days', type=str, default='mon',
                        choices=['mon', 'tue', 'wed', 'thu', 'fri', 'mwf', 'all'],
                        help='Days to trade on')
    
    parser.add_argument('--output', type=str,
                        help='Save results to file')
    
    parser.add_argument('--plot', action='store_true',
                        help='Show performance charts')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Map trading days to weekday numbers
    trading_days_map = {
        'mon': {0},         # Monday
        'tue': {1},         # Tuesday
        'wed': {2},         # Wednesday
        'thu': {3},         # Thursday
        'fri': {4},         # Friday
        'mwf': {0, 2, 4},   # Monday, Wednesday, Friday
        'all': None         # All days
    }
    
    trading_days = trading_days_map.get(args.trading_days)
    
    # Create and run backtester
    backtester = EnsembleBacktester(
        start_date=args.start_date,
        end_date=args.end_date,
        trading_days=trading_days
    )
    
    try:
        backtester.run_backtest()
        backtester.print_results()
        
        if args.output:
            backtester.save_results(args.output)
        
        if args.plot:
            backtester.plot_results()
            
    except Exception as e:
        print(f"Error during backtest: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
