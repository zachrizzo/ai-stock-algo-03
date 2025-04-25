"""
Simple example of how to run a backtest of the micro-CTA strategy.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_trader_o3_algo.data.price_data import fetch_prices
from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.config.settings import RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF, LOOKBACK_DAYS


def run_simple_backtest():
    """Run a simple backtest of the micro-CTA strategy."""
    print("Running simple backtest example...")
    
    # Set backtest parameters - use a more recent period where we have better data
    start_date = "2019-01-01"  # Use longer history for better performance
    end_date = "2024-03-31"
    initial_capital = 100.0
    transaction_cost_pct = 0.0003  # 3 basis points
    
    print(f"Fetching price data from before {start_date} to {end_date}...")
    
    # First check what data is available
    tickers = [RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF]
    print("Checking data availability...")
    prices = fetch_prices(tickers, days=500, end_date=end_date)
    print(f"Data available from {prices.index[0]} to {prices.index[-1]}")
    
    # Create and run backtest
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        trade_weekdays=(0, 2, 4)  # Trade on Monday, Wednesday, Friday
    )
    
    # Run the backtest
    print("Running backtest...")
    equity_curve = backtest.run_backtest()
    
    # Print summary statistics
    print("\nBacktest Results:")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${equity_curve.iloc[-1]:.2f}")
    print(f"Total Return: {(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1):.2%}")
    print(f"CAGR: {backtest.portfolio_stats['cagr']:.2%}")
    print(f"Max Drawdown: {backtest.portfolio_stats['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {backtest.portfolio_stats['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {backtest.portfolio_stats['sortino_ratio']:.2f}")
    print(f"Number of Trades: {backtest.portfolio_stats['num_trades']}")
    
    # Plot equity curve
    backtest.plot_equity_curve()
    
    # Plot asset allocation
    backtest.plot_asset_allocation()
    
    return backtest


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Run the backtest
    backtest = run_simple_backtest()
    
    # Show the plots
    plt.show()
