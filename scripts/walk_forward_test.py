"""
Walk-forward testing script for the trading strategy.
This script runs the strategy on multiple time windows to test robustness.
"""

import os
import sys
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.config.settings import RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF
from stock_trader_o3_algo.data.price_data import fetch_prices


def run_walk_forward_test(
    start_date: str = "2005-01-01",
    end_date: str = "2024-03-31",
    window_years: int = 3,
    step_months: int = 6,
    tickers: List[str] = None,
    trade_weekdays: Tuple[int, ...] = (0, 2, 4)  # Mon, Wed, Fri
) -> pd.DataFrame:
    """
    Run walk-forward testing with rolling windows.
    
    Args:
        start_date: Overall start date for testing
        end_date: Overall end date for testing
        window_years: Size of each test window in years
        step_months: Months to advance for each new window
        tickers: List of tickers to include (default: strategy default tickers)
        trade_weekdays: Days of the week to trade (0=Monday)
    
    Returns:
        DataFrame with performance metrics for each window
    """
    if tickers is None:
        tickers = [RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF]
    
    # Fetch all data at once and cache it
    print(f"Fetching all data from {start_date} to {end_date}...")
    all_data = fetch_prices(tickers, start_date=start_date, end_date=end_date, use_cache=True)
    print(f"Data range: {all_data.index[0]} to {all_data.index[-1]}")
    
    # Create date ranges for rolling windows
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Calculate window size in days
    window_days = window_years * 365
    step_days = step_months * 30
    
    # Create list of start dates for each window
    window_starts = []
    current = start_dt
    while current + pd.Timedelta(days=window_days) <= end_dt:
        window_starts.append(current)
        current += pd.Timedelta(days=step_days)
    
    # Run backtest for each window
    results = []
    
    for window_start in window_starts:
        window_end = window_start + pd.Timedelta(days=window_days)
        
        # Convert dates to string format
        start_str = window_start.strftime("%Y-%m-%d")
        end_str = window_end.strftime("%Y-%m-%d")
        
        print(f"\nTesting window: {start_str} to {end_str}")
        
        # Create and run backtest
        backtest = BacktestEngine(
            start_date=start_str,
            end_date=end_str,
            initial_capital=100.0,
            transaction_cost_pct=0.0003,
            trade_weekdays=trade_weekdays
        )
        
        # Run backtest (it will use the cached price data)
        equity_curve = backtest.run_backtest()
        
        # Store results
        window_results = {
            'window_start': start_str,
            'window_end': end_str,
            'final_capital': backtest.portfolio_stats['final_capital'],
            'total_return': backtest.portfolio_stats['total_return'],
            'cagr': backtest.portfolio_stats['cagr'],
            'max_drawdown': backtest.portfolio_stats['max_drawdown'],
            'sharpe_ratio': backtest.portfolio_stats['sharpe_ratio'],
            'sortino_ratio': backtest.portfolio_stats['sortino_ratio'],
            'num_trades': backtest.portfolio_stats['num_trades']
        }
        
        results.append(window_results)
        
        print(f"Window results: Return: {window_results['total_return']:.2%}, "
              f"Sharpe: {window_results['sharpe_ratio']:.2f}, "
              f"Max DD: {window_results['max_drawdown']:.2%}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall performance metrics
    overall_metrics = {
        'avg_return': results_df['total_return'].mean(),
        'avg_cagr': results_df['cagr'].mean(),
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_sortino': results_df['sortino_ratio'].mean(),
        'worst_drawdown': results_df['max_drawdown'].min(),
        'pct_positive_windows': (results_df['total_return'] > 0).mean() * 100
    }
    
    print("\nOverall walk-forward results:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create plots directory
    plots_dir = Path("backtest_results/walk_forward")
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    results_df['total_return'].plot(kind='bar')
    plt.title('Returns by Window')
    plt.xticks([])
    
    plt.subplot(2, 2, 2)
    results_df['sharpe_ratio'].plot(kind='bar')
    plt.title('Sharpe Ratio by Window')
    plt.xticks([])
    
    plt.subplot(2, 2, 3)
    results_df['max_drawdown'].plot(kind='bar')
    plt.title('Max Drawdown by Window')
    plt.xticks([])
    
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['max_drawdown'], results_df['total_return'])
    plt.title('Return vs. Drawdown')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Total Return')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "walk_forward_summary.png")
    print(f"Saved summary plot to {plots_dir / 'walk_forward_summary.png'}")
    
    return results_df


if __name__ == "__main__":
    # Run the walk-forward test with default settings
    results = run_walk_forward_test(
        start_date="2007-01-01",  # Start from 2007 to have enough data
        end_date="2024-03-31",
        window_years=2,  # 2-year windows
        step_months=12,  # Move 1 year at a time
        trade_weekdays=(0, 2, 4)  # Mon, Wed, Fri
    )
    
    # Save results to CSV
    output_dir = Path("backtest_results/walk_forward")
    output_dir.mkdir(exist_ok=True, parents=True)
    results.to_csv(output_dir / "walk_forward_results.csv", index=False)
    print(f"Saved results to {output_dir / 'walk_forward_results.csv'}")
