#!/usr/bin/env python3
"""
Backtest engine for the hybrid strategy that combines
Tri-Shot, DMT, and TurboQT strategies.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from stock_trader_o3_algo.strategies.hybrid.hybrid_strategy import get_hybrid_allocation
from stock_trader_o3_algo.core.performance import calculate_performance_metrics
from stock_trader_o3_algo.strategies.tri_shot.tri_shot_features import fetch_data
from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.core.strategy import CASH_ETF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridBacktester(BacktestEngine):
    """Backtest engine for the hybrid strategy."""
    
    def __init__(self, 
                 start_date=None, 
                 end_date=None, 
                 initial_capital=10000.0,
                 transaction_cost_pct=0.0003,
                 rebalance_freq="W-MON"):
        """
        Initialize the hybrid strategy backtester.
        
        Args:
            start_date: Start date for backtest (str or datetime)
            end_date: End date for backtest (str or datetime)
            initial_capital: Starting capital
            transaction_cost_pct: Transaction cost as a percentage of trade value
            rebalance_freq: Frequency for rebalancing
        """
        # Convert None dates to strings to avoid type errors
        if start_date is None:
            start_date = "2024-01-01"
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct,
            rebalance_freq=rebalance_freq
        )
        
        self.strategy_name = "Hybrid Strategy"
        self.capital = initial_capital
        self.equity_curve = pd.Series()
        self.equity_peak = initial_capital
        
        # Create directory for results if it doesn't exist
        self.results_dir = os.path.join(os.path.dirname(__file__), '../../../tri_shot_data')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def fetch_backtest_data(self, days=365):
        """Fetch data for backtest."""
        tickers = ["QQQ", "TQQQ", "SQQQ", "TMF", "TLT", "SPY", "^VIX", CASH_ETF]
        logger.info(f"Fetching data for tickers: {tickers}")
        self.prices = fetch_data(tickers, days=days)
        
        # Set start and end dates based on available data if not specified
        if self.start_date is None:
            self.start_date = self.prices.index[0]
        if self.end_date is None:
            self.end_date = self.prices.index[-1]
            
        # Trim data to backtest period
        self.prices = self.prices.loc[self.start_date:self.end_date]
        
        # Ensure we have enough data
        if len(self.prices) < 20:
            raise ValueError(f"Not enough price data for backtest. Found {len(self.prices)} days.")
            
        logger.info(f"Backtest period: {self.start_date} to {self.end_date} ({len(self.prices)} trading days)")
        
    def run_individual_strategy_backtests(self):
        """Run backtests for each individual strategy for comparison."""
        # This will be implemented to run tri-shot, DMT, and TurboQT separately
        # for performance comparison
        pass
        
    def run_backtest(self):
        """Run the backtest for the hybrid strategy."""
        logger.info(f"Running backtest for {self.strategy_name}")
        
        # Initialize backtest variables
        self.portfolio = {CASH_ETF: self.initial_capital}
        self.capital = self.initial_capital
        self.equity_curve = pd.Series(index=self.prices.index)
        self.equity_peak = self.initial_capital
        self.trades = []
        self.allocations = []
        
        # Run through each day in the backtest
        for i, date in enumerate(self.prices.index):
            # Skip first day as we need at least one day of history
            if i == 0:
                self.equity_curve.loc[date] = self.capital
                continue
                
            # Calculate current portfolio value
            portfolio_value = 0
            for symbol, shares in self.portfolio.items():
                if symbol == CASH_ETF:
                    portfolio_value += shares  # Cash is always $1
                else:
                    portfolio_value += shares * self.prices.loc[date, symbol]
                    
            # Update capital and equity peak
            self.capital = portfolio_value
            self.equity_peak = max(self.equity_peak, self.capital)
            
            # Record equity
            self.equity_curve.loc[date] = self.capital
            
            # Get target allocation from hybrid strategy
            target_allocation = get_hybrid_allocation(
                self.prices.iloc[:i+1], 
                date=date, 
                equity=self.capital,
                equity_peak=self.equity_peak,
                equity_curve=self.equity_curve
            )
            
            # Record allocation
            self.allocations.append({
                'date': date,
                'allocation': target_allocation
            })
            
            # Calculate rebalance trades
            trades_for_day = self.calculate_rebalance_trades(date, target_allocation)
            
            # Execute trades
            for trade in trades_for_day:
                self.execute_trade(date, trade)
                
        # Calculate final portfolio value
        final_value = 0
        for symbol, shares in self.portfolio.items():
            if symbol == CASH_ETF:
                final_value += shares
            else:
                final_value += shares * self.prices.loc[self.prices.index[-1], symbol]
                
        self.final_value = final_value
        
        # Calculate performance metrics
        self.metrics = calculate_performance_metrics(self.equity_curve)
        self.metrics['final_value'] = final_value
        self.metrics['total_return'] = final_value / self.initial_capital - 1
        
        logger.info(f"Backtest complete. Final value: ${final_value:.2f}")
        
    def calculate_rebalance_trades(self, date, target_allocation):
        """Calculate trades needed to rebalance portfolio to target allocation."""
        trades = []
        
        # Calculate current portfolio value
        current_value = 0
        current_allocation = {}
        
        for symbol, shares in self.portfolio.items():
            if symbol == CASH_ETF:
                value = shares
            else:
                value = shares * self.prices.loc[date, symbol]
                
            current_value += value
            current_allocation[symbol] = value
            
        # Calculate target shares for each symbol
        for symbol, target_value in target_allocation.items():
            current_value_in_symbol = current_allocation.get(symbol, 0)
            
            if symbol == CASH_ETF:
                price = 1.0
            else:
                price = self.prices.loc[date, symbol]
                
            target_shares = target_value / price
            current_shares = self.portfolio.get(symbol, 0)
            
            # Calculate shares to trade
            shares_to_trade = target_shares - current_shares
            
            # Only trade if the difference is significant
            if abs(shares_to_trade * price) > 10:  # $10 minimum trade
                trade = {
                    'symbol': symbol,
                    'shares': shares_to_trade,
                    'price': price,
                    'value': shares_to_trade * price
                }
                trades.append(trade)
                
        return trades
        
    def execute_trade(self, date, trade):
        """Execute a trade in the backtest."""
        symbol = trade['symbol']
        shares = trade['shares']
        price = trade['price']
        value = trade['value']
        
        # Apply slippage and commission
        if shares > 0:  # Buy
            adjusted_price = price * (1 + self.slippage_bps / 10000)
            commission = value * self.commission_bps / 10000
            total_cost = value + commission
        else:  # Sell
            adjusted_price = price * (1 - self.slippage_bps / 10000)
            commission = abs(value) * self.commission_bps / 10000
            total_cost = value - commission
            
        # Update cash
        self.portfolio[CASH_ETF] = self.portfolio.get(CASH_ETF, 0) - total_cost
        
        # Update position
        self.portfolio[symbol] = self.portfolio.get(symbol, 0) + shares
        
        # Remove positions with zero shares
        if abs(self.portfolio.get(symbol, 0)) < 0.001:
            del self.portfolio[symbol]
            
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'adjusted_price': adjusted_price,
            'value': value,
            'commission': commission,
            'total_cost': total_cost
        })
        
    def compare_to_benchmarks(self):
        """Compare strategy performance to benchmarks."""
        # Calculate buy and hold performance for QQQ
        qqq_returns = self.prices['QQQ'] / self.prices['QQQ'].iloc[0]
        qqq_equity = self.initial_capital * qqq_returns
        
        # Calculate buy and hold for 60/40 portfolio (60% QQQ, 40% TLT)
        if 'TLT' in self.prices.columns:
            balanced_returns = 0.6 * (self.prices['QQQ'] / self.prices['QQQ'].iloc[0]) + \
                               0.4 * (self.prices['TLT'] / self.prices['TLT'].iloc[0])
            balanced_equity = self.initial_capital * balanced_returns
        else:
            balanced_equity = None
            
        # Calculate metrics
        qqq_metrics = calculate_performance_metrics(qqq_equity)
        if balanced_equity is not None:
            balanced_metrics = calculate_performance_metrics(balanced_equity)
        else:
            balanced_metrics = None
            
        # Return benchmark comparison
        return {
            'strategy': self.metrics,
            'qqq': qqq_metrics,
            'balanced': balanced_metrics,
            'equity_curves': {
                'strategy': self.equity_curve,
                'qqq': qqq_equity,
                'balanced': balanced_equity
            }
        }
        
    def plot_results(self, show_benchmarks=True):
        """Plot backtest results with benchmarks."""
        # Get benchmark comparison
        comparison = self.compare_to_benchmarks()
        equity_curves = comparison['equity_curves']
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot strategy
        plt.plot(equity_curves['strategy'], 'b-', label=f"{self.strategy_name} (CAGR: {self.metrics['cagr']*100:.1f}%)")
        
        # Plot benchmarks
        if show_benchmarks:
            plt.plot(equity_curves['qqq'], 'r-', label=f"QQQ (CAGR: {comparison['qqq']['cagr']*100:.1f}%)")
            if equity_curves['balanced'] is not None:
                plt.plot(
                    equity_curves['balanced'], 'g-', 
                    label=f"60/40 Portfolio (CAGR: {comparison['balanced']['cagr']*100:.1f}%)"
                )
                
        # Add drawdown plot at bottom
        drawdowns = {}
        for name, curve in equity_curves.items():
            if curve is not None:
                # Calculate running maximum
                running_max = curve.cummax()
                # Calculate drawdown percentage
                drawdown = (curve / running_max - 1) * 100
                drawdowns[name] = drawdown
                
        # Plot drawdowns
        plt.subplot(2, 1, 1)
        plt.plot(equity_curves['strategy'], 'b-', label=f"{self.strategy_name}")
        if show_benchmarks:
            plt.plot(equity_curves['qqq'], 'r-', label="QQQ")
            if equity_curves['balanced'] is not None:
                plt.plot(equity_curves['balanced'], 'g-', label="60/40 Portfolio")
        plt.title(f"{self.strategy_name} Backtest Results")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdowns['strategy'], 'b-', label=f"{self.strategy_name} DD")
        if show_benchmarks:
            plt.plot(drawdowns['qqq'], 'r-', label="QQQ DD")
            if drawdowns.get('balanced') is not None:
                plt.plot(drawdowns['balanced'], 'g-', label="60/40 DD")
        plt.title("Drawdowns")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'hybrid_backtest_results.png'))
        plt.close()
        
        logger.info(f"Plot saved to {os.path.join(self.results_dir, 'hybrid_backtest_results.png')}")
        
    def save_results(self):
        """Save backtest results to CSV files."""
        # Save equity curve
        self.equity_curve.to_csv(os.path.join(self.results_dir, 'hybrid_equity_curve.csv'))
        
        # Save trades
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(self.results_dir, 'hybrid_trades.csv'), index=False)
            
        # Save allocations
        allocations_df = pd.DataFrame([
            {'date': a['date'], **{k: v for k, v in a['allocation'].items()}}
            for a in self.allocations
        ])
        if not allocations_df.empty:
            allocations_df.to_csv(os.path.join(self.results_dir, 'hybrid_allocations.csv'), index=False)
            
        # Save performance metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(os.path.join(self.results_dir, 'hybrid_metrics.csv'), index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
        
    def print_results(self):
        """Print backtest results."""
        comparison = self.compare_to_benchmarks()
        
        # Print header
        print("\n" + "=" * 50)
        print(f"{self.strategy_name} BACKTEST RESULTS".center(50))
        print("=" * 50)
        
        # Print basic information
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')} "
              f"({len(self.prices)} trading days)")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print()
        
        # Print performance comparison
        print("--- Performance Comparison ---")
        strategy_final = self.metrics['final_value']
        qqq_final = comparison['qqq']['final_value']
        print(f"Strategy Final Value:   ${strategy_final:.2f}")
        print(f"QQQ Final Value:        ${qqq_final:.2f}")
        print(f"Strategy Return:        {self.metrics['total_return']*100:.2f}%")
        print(f"QQQ Return:             {comparison['qqq']['total_return']*100:.2f}%")
        print(f"Strategy CAGR:          {self.metrics['cagr']*100:.2f}%")
        print(f"QQQ CAGR:               {comparison['qqq']['cagr']*100:.2f}%")
        print(f"Outperformance:         {(self.metrics['cagr'] - comparison['qqq']['cagr'])*100:.2f}%")
        print()
        
        # Print risk metrics
        print("--- Risk Metrics ---")
        print(f"Strategy Volatility:    {self.metrics['volatility']*100:.2f}%")
        print(f"QQQ Volatility:         {comparison['qqq']['volatility']*100:.2f}%")
        print(f"Strategy Max Drawdown:  {self.metrics['max_drawdown']*100:.2f}%")
        print(f"QQQ Max DD:             {comparison['qqq']['max_drawdown']*100:.2f}%")
        print()
        
        # Print risk-adjusted metrics
        print("--- Risk-Adjusted Returns ---")
        print(f"Strategy Sharpe:        {self.metrics['sharpe']:.2f}")
        print(f"QQQ Sharpe:             {comparison['qqq']['sharpe']:.2f}")
        print(f"Strategy Calmar:        {self.metrics['cagr']/self.metrics['max_drawdown']:.2f}")
        print(f"QQQ Calmar:             {comparison['qqq']['cagr']/comparison['qqq']['max_drawdown']:.2f}")
        print()
        
        # Print trading statistics
        print("--- Trading Statistics ---")
        print(f"Total Trades:           {len(self.trades)}")
        print(f"Trades per Year:        {len(self.trades)/len(self.prices)*252:.1f}")
        total_commission = sum(t['commission'] for t in self.trades)
        print(f"Total Commission:       ${total_commission:.2f}")
        print(f"Commission as % of Return: {total_commission/(strategy_final-self.initial_capital)*100:.2f}%")
        print()
        
        # Print save locations
        print(f"Plot saved to {os.path.join(self.results_dir, 'hybrid_backtest_results.png')}")
        print(f"Detailed results saved to {self.results_dir}")
        
def run_hybrid_backtest(days=365, plot=True, initial_capital=10000.0, start_date=None, end_date=None):
    """
    Run a backtest of the hybrid strategy.
    
    Args:
        days: Number of days to backtest
        plot: Whether to plot results
        initial_capital: Initial capital
        start_date: Start date (format: YYYY-MM-DD)
        end_date: End date (format: YYYY-MM-DD)
        
    Returns:
        HybridBacktester instance
    """
    # Set up dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create backtester
    backtester = HybridBacktester(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_pct=0.0003  # 3 basis points
    )
    
    # Fetch data
    backtester.fetch_backtest_data(days=days)
    
    # Run backtest
    backtester.run_backtest()
    
    # Print results
    backtester.print_results()
    
    # Plot results
    if plot:
        backtester.plot_results()
        
    # Save results
    backtester.save_results()
    
    return backtester
    
if __name__ == "__main__":
    # Run backtest with default parameters
    run_hybrid_backtest(days=365, plot=True, initial_capital=10000.0)
