#!/usr/bin/env python
"""
Command-line interface for the stock trading strategy.
This script provides easy access to backtest, optimization, and live trading functions.
"""

import argparse
import datetime as dt
from pathlib import Path
import pandas as pd
import sys

from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.config.settings import RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF
from stock_trader_o3_algo.data.price_data import fetch_prices
from scripts.walk_forward_test import run_walk_forward_test
from scripts.grid_search import run_grid_search


def run_simple_backtest(args):
    """Run a simple backtest with the specified parameters."""
    print(f"Running backtest from {args.start_date} to {args.end_date}")
    
    # Parse trading days
    if args.trading_days == "mon":
        trade_weekdays = (0,)
    elif args.trading_days == "wed":
        trade_weekdays = (2,)
    elif args.trading_days == "fri":
        trade_weekdays = (4,)
    elif args.trading_days == "mwf":
        trade_weekdays = (0, 2, 4)
    else:
        trade_weekdays = None  # Use default in BacktestEngine
    
    # Create and run backtest
    backtest = BacktestEngine(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        transaction_cost_pct=args.cost,
        trade_weekdays=trade_weekdays
    )
    
    # Run backtest
    equity_curve = backtest.run_backtest()
    
    # Print results
    print("\nBacktest Results:")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Initial Capital: ${args.capital:.2f}")
    print(f"Final Capital: ${backtest.portfolio_stats['final_capital']:.2f}")
    print(f"Total Return: {backtest.portfolio_stats['total_return']:.2%}")
    print(f"CAGR: {backtest.portfolio_stats['cagr']:.2%}")
    print(f"Max Drawdown: {backtest.portfolio_stats['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {backtest.portfolio_stats['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {backtest.portfolio_stats['sortino_ratio']:.2f}")
    print(f"Number of Trades: {backtest.portfolio_stats['num_trades']}")
    
    # Save results
    if args.save:
        results_dir = Path("backtest_results/cli")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamp for the filename
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save equity curve
        equity_curve.to_csv(results_dir / f"equity_curve_{timestamp}.csv")
        
        # Save summary stats
        pd.Series(backtest.portfolio_stats).to_csv(results_dir / f"stats_{timestamp}.csv")
        
        print(f"\nResults saved to {results_dir}")

    return backtest


def run_walk_forward(args):
    """Run walk-forward testing with the specified parameters."""
    results = run_walk_forward_test(
        start_date=args.start_date,
        end_date=args.end_date,
        window_years=args.window,
        step_months=args.step,
        trade_weekdays=(0, 2, 4) if args.mwf else (0,)
    )
    
    # Save results
    if args.save:
        output_dir = Path("backtest_results/walk_forward")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamp for the filename
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(output_dir / f"walk_forward_{timestamp}.csv", index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    return results


def run_optimization(args):
    """Run grid search optimization with the specified parameters."""
    results = run_grid_search(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days_values=[40, 60, 90] if args.full else [60],
        short_lookback_values=[5, 10, 15] if args.full else [10],
        vol_target_values=[0.02, 0.03, 0.04] if args.full else [0.03],
        trade_weekdays_options=[(0,), (0, 2, 4)] if args.full else [(0, 2, 4)],
        parallel=args.parallel
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Stock Trading Strategy CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('--start-date', type=str, default="2010-01-01", help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, default="2024-03-31", help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    backtest_parser.add_argument('--cost', type=float, default=0.0003, help='Transaction cost percentage')
    backtest_parser.add_argument('--trading-days', type=str, default="mwf", choices=['mon', 'wed', 'fri', 'mwf'], 
                               help='Trading days (mon, wed, fri, or mwf for all)')
    backtest_parser.add_argument('--save', action='store_true', help='Save results to CSV')
    
    # Parser for walk-forward command
    walk_parser = subparsers.add_parser('walk', help='Run walk-forward testing')
    walk_parser.add_argument('--start-date', type=str, default="2010-01-01", help='Start date (YYYY-MM-DD)')
    walk_parser.add_argument('--end-date', type=str, default="2024-03-31", help='End date (YYYY-MM-DD)')
    walk_parser.add_argument('--window', type=int, default=2, help='Window size in years')
    walk_parser.add_argument('--step', type=int, default=12, help='Step size in months')
    walk_parser.add_argument('--mwf', action='store_true', help='Trade Mon/Wed/Fri (default: Monday only)')
    walk_parser.add_argument('--save', action='store_true', help='Save results to CSV')
    
    # Parser for optimization command
    opt_parser = subparsers.add_parser('optimize', help='Run grid search optimization')
    opt_parser.add_argument('--start-date', type=str, default="2010-01-01", help='Start date (YYYY-MM-DD)')
    opt_parser.add_argument('--end-date', type=str, default="2024-03-31", help='End date (YYYY-MM-DD)')
    opt_parser.add_argument('--full', action='store_true', help='Run full grid search (slow)')
    opt_parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        run_simple_backtest(args)
    elif args.command == 'walk':
        run_walk_forward(args)
    elif args.command == 'optimize':
        run_optimization(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
