#!/usr/bin/env python3
"""
Unified command-line interface for the stock trading algorithms.

This is the main entry point for all trading strategies:
- Tri-Shot (tri_shot): A strategy running on Monday/Wednesday/Friday
- DMT (dmt): Differentiable Market Twin strategy
- TurboQT (turbo_qt): Turbo Rotational QQQ strategy

Usage:
  trader_cli.py [strategy] [command] [options]

Examples:
  trader_cli.py tri_shot run
  trader_cli.py dmt backtest --days 365
  trader_cli.py turbo_qt rebalance
"""

import os
import sys
import argparse
import datetime as dt
import importlib.util

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_tri_shot_parser(subparsers):
    """Set up parser for tri_shot strategy commands."""
    parser = subparsers.add_parser('tri_shot', help='Run Tri-Shot trading strategy')
    tri_cmds = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run strategy
    run_cmd = tri_cmds.add_parser('run', help='Run strategy based on current day')
    run_cmd.add_argument('--force', action='store_true', help='Force run regardless of day')
    
    # Train model
    train_cmd = tri_cmds.add_parser('train', help='Train or retrain the model')
    train_cmd.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    
    # Backtest
    backtest_cmd = tri_cmds.add_parser('backtest', help='Run a comprehensive backtest')
    backtest_cmd.add_argument('--days', type=int, default=365, help='Number of days to backtest')
    backtest_cmd.add_argument('--plot', action='store_true', help='Plot results')
    backtest_cmd.add_argument('--capital', type=float, default=500.0, help='Initial capital')
    backtest_cmd.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    backtest_cmd.add_argument('--slippage', type=int, default=1, help='Slippage in basis points')
    backtest_cmd.add_argument('--commission', type=int, default=1, help='Commission in basis points')
    backtest_cmd.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    backtest_cmd.add_argument('--mc-runs', type=int, default=10, help='Number of Monte Carlo runs')
    
    # Paper trade
    paper_cmd = tri_cmds.add_parser('paper', help='Set up paper trading')
    paper_cmd.add_argument('--capital', type=float, default=500.0, help='Initial capital')
    paper_cmd.add_argument('--days', type=int, default=30, help='Days to run paper trading')

def setup_dmt_parser(subparsers):
    """Set up parser for DMT strategy commands."""
    parser = subparsers.add_parser('dmt', help='Run DMT strategy')
    dmt_cmds = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model
    train_cmd = dmt_cmds.add_parser('train', help='Train the DMT model')
    train_cmd.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_cmd.add_argument('--lookback', type=int, default=30, help='Lookback window in days')
    train_cmd.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for LSTM')
    train_cmd.add_argument('--save-path', type=str, help='Path to save model')
    
    # Backtest
    backtest_cmd = dmt_cmds.add_parser('backtest', help='Run DMT backtest')
    backtest_cmd.add_argument('--days', type=int, default=365, help='Days to backtest')
    backtest_cmd.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    backtest_cmd.add_argument('--plot', action='store_true', help='Plot results')
    backtest_cmd.add_argument('--model-path', type=str, help='Path to trained model')

def setup_turbo_qt_parser(subparsers):
    """Set up parser for Turbo QT strategy commands."""
    parser = subparsers.add_parser('turbo_qt', help='Run Turbo QT strategy')
    turbo_cmds = parser.add_subparsers(dest='command', help='Command to run')
    
    # Rebalance
    rebalance_cmd = turbo_cmds.add_parser('rebalance', help='Run rebalancing')
    rebalance_cmd.add_argument('--dry-run', action='store_true', help='Do not execute trades')
    
    # Check stops
    check_cmd = turbo_cmds.add_parser('check_stops', help='Check if stops are hit')
    check_cmd.add_argument('--dry-run', action='store_true', help='Do not execute trades')
    
    # Backtest
    backtest_cmd = turbo_cmds.add_parser('backtest', help='Run backtest')
    backtest_cmd.add_argument('--days', type=int, default=365, help='Days to backtest')
    backtest_cmd.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    backtest_cmd.add_argument('--plot', action='store_true', help='Plot results')
    backtest_cmd.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo')
    backtest_cmd.add_argument('--mc-runs', type=int, default=10, help='Number of Monte Carlo runs')

def main():
    """Main entry point for the trader CLI."""
    parser = argparse.ArgumentParser(
        description='Unified CLI for stock trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Create strategy subparsers
    subparsers = parser.add_subparsers(dest='strategy', help='Trading strategy to use')
    
    # Set up parsers for each strategy
    setup_tri_shot_parser(subparsers)
    setup_dmt_parser(subparsers)
    setup_turbo_qt_parser(subparsers)
    
    args = parser.parse_args()
    
    if not args.strategy:
        parser.print_help()
        return
    
    # Import appropriate module based on selected strategy
    if args.strategy == 'tri_shot':
        # Import here to avoid unnecessary imports if not using this strategy
        from stock_trader_o3_algo.strategies.tri_shot import (
            run_monday_strategy, run_wednesday_strategy, run_friday_strategy,
            ensure_state_dir, get_alpaca_api
        )
        
        # For commands like train and backtest, import from tri_shot_cli
        # but don't let it affect our path
        spec = importlib.util.spec_from_file_location("tri_shot_cli", "bin/tri_shot_cli.py")
        tri_shot_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tri_shot_cli)
        
        ensure_state_dir()
        
        if args.command == 'run':
            if args.force:
                tri_shot_cli.run_strategy(force=True)
            else:
                tri_shot_cli.run_strategy()
        elif args.command == 'train':
            tri_shot_cli.train_model(force=args.force)
        elif args.command == 'backtest':
            tri_shot_cli.backtest(
                days=args.days, 
                plot=args.plot, 
                initial_capital=args.capital,
                start_date=args.start_date,
                slippage_bps=args.slippage,
                commission_bps=args.commission,
                monte_carlo=args.monte_carlo,
                mc_runs=args.mc_runs
            )
        elif args.command == 'paper':
            tri_shot_cli.setup_paper_trade(
                initial_capital=args.capital,
                days=args.days
            )
        else:
            print("Please specify a command for tri_shot strategy")
    
    elif args.strategy == 'dmt':
        if not args.command:
            print("Please specify a command for dmt strategy")
            return
            
        try:
            import torch
            from stock_trader_o3_algo.strategies.dmt import (
                MarketTwinLSTM, train_market_twin, load_market_twin,
                run_dmt_backtest
            )
        except ImportError:
            print("DMT strategy requires PyTorch. Please install with: pip install torch")
            return
            
        # Import tri_shot_cli for the DMT functions
        spec = importlib.util.spec_from_file_location("tri_shot_cli", "bin/tri_shot_cli.py")
        tri_shot_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tri_shot_cli)
        
        if args.command == 'train':
            # We need the tri_shot features for this
            tri_shot_cli.train_model(force=True)
            
            # Now train DMT model
            import pandas as pd
            from stock_trader_o3_algo.strategies.tri_shot import tri_shot_features
            
            # Get data for training
            tickers = ["QQQ", "TQQQ", "SQQQ", "TMF", "TLT", "^VIX"]
            prices = tri_shot_features.fetch_data(tickers, days=1000)
            
            # Train model
            train_market_twin(
                prices=prices,
                target_ticker='QQQ',
                lookback_window=args.lookback,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                save_path=args.save_path
            )
        elif args.command == 'backtest':
            # Create simulated args object for run_dmt_command
            class Args:
                pass
            
            dmt_args = Args()
            dmt_args.days = args.days
            dmt_args.initial_capital = args.capital
            dmt_args.model_path = args.model_path
            dmt_args.plot = args.plot
            dmt_args.cpu = True
            dmt_args.start_date = None
            dmt_args.end_date = None
            dmt_args.n_epochs = 10
            dmt_args.epochs = 10
            dmt_args.learning_rate = 0.001
            
            tri_shot_cli.run_dmt_command(dmt_args)
    
    elif args.strategy == 'turbo_qt':
        from stock_trader_o3_algo.strategies.turbo_qt import (
            rebalance, check_stops, TurboBacktester
        )
        
        if args.command == 'rebalance':
            rebalance(dry_run=args.dry_run)
        elif args.command == 'check_stops':
            check_stops(dry_run=args.dry_run)
        elif args.command == 'backtest':
            # Create and run backtester
            backtester = TurboBacktester(
                start_date="2024-01-01",  # Use a recent date
                end_date=dt.datetime.now().strftime('%Y-%m-%d'),
                initial_capital=args.capital,
                trading_days="mon"
            )
            
            backtester.run_backtest()
            backtester.print_results()
            
            if args.plot:
                backtester.plot_results()
        else:
            print("Please specify a command for turbo_qt strategy")
    
if __name__ == '__main__':
    main()
