#!/usr/bin/env python3
"""
Unified Trading Strategy CLI
============================
Command-line interface for all trading strategies:

- Tri-Shot: A strategy running on Monday/Wednesday/Friday
- DMT (v1 & v2): Differentiable Market Twin strategies
- TurboQT: Turbo Rotational QQQ strategy
- TurboDMT: Enhanced DMT with dynamic positioning
- Hybrid: Combined strategy approach

This script provides a modern, consistent interface while maintaining
backward compatibility with the previous CLI format.

Usage:
  trade.py [strategy] [command] [options]

Examples:
  # Run the DMT_v2 strategy
  trade.py dmt_v2 run
  
  # Backtest the Tri-Shot strategy
  trade.py tri_shot backtest --start-date 2022-01-01 --end-date 2022-12-31
  
  # Run the TurboDMT_v2 strategy with crypto data
  trade.py turbo_dmt_v2 run --asset-type crypto --symbol BTC
  
  # Compare multiple strategies
  trade.py compare --strategies dmt_v2,turbo_dmt_v2 --start-date 2022-01-01
"""

import os
import sys
import argparse
import datetime as dt
import importlib
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trade-cli')

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our refactored modules
try:
    from stock_trader_o3_algo.backtester.core import Backtester, BatchBacktester
    from stock_trader_o3_algo.backtester.performance import calculate_performance_metrics
    from stock_trader_o3_algo.data_utils.market_simulator import (
        generate_realistic_market_data, 
        fetch_yahoo_data
    )
    from stock_trader_o3_algo.strategies.dmt_v2_strategy import DMT_v2_Strategy
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    logger.error("Please ensure you've run the refactoring process to set up the package structure.")
    sys.exit(1)

# Helper functions
def parse_date(date_str: str) -> dt.datetime:
    """Parse date string in YYYY-MM-DD format"""
    try:
        return dt.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")
        sys.exit(1)

def get_date_range(args: argparse.Namespace) -> Tuple[dt.datetime, dt.datetime]:
    """Get start and end dates from args or use defaults"""
    end_date = dt.datetime.now()
    
    if hasattr(args, 'end_date') and args.end_date:
        end_date = parse_date(args.end_date)
    
    # Default to 1 year ago if not specified
    start_date = end_date - dt.timedelta(days=365)
    
    if hasattr(args, 'start_date') and args.start_date:
        start_date = parse_date(args.start_date)
    
    return start_date, end_date

def load_data(
    symbol: str, 
    start_date: dt.datetime, 
    end_date: dt.datetime,
    use_cache: bool = True,
    use_simulation: bool = False
) -> pd.DataFrame:
    """Load historical price data for the given symbol"""
    if use_simulation:
        logger.info(f"Generating simulated data for {symbol} from {start_date.date()} to {end_date.date()}...")
        return generate_realistic_market_data(start_date, end_date, ticker=symbol)
    
    # Try to load from Yahoo Finance
    data = fetch_yahoo_data(
        symbol,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    if data is None or data.empty:
        logger.error(f"Failed to load data for {symbol}. Falling back to simulation.")
        return generate_realistic_market_data(start_date, end_date, ticker=symbol)
    
    return data

# Strategy factory functions
def create_dmt_v2_strategy(
    version: str = "original",
    asset_type: str = "equity",
    lookback_period: int = 252,
    initial_capital: float = 10000.0
) -> DMT_v2_Strategy:
    """Create DMT_v2 strategy instance"""
    return DMT_v2_Strategy(
        version=version,
        asset_type=asset_type,
        lookback_period=lookback_period,
        initial_capital=initial_capital
    )

# Command functions
def run_backtest(
    args: argparse.Namespace, 
    strategy_name: str
) -> None:
    """Run backtest for a specific strategy"""
    start_date, end_date = get_date_range(args)
    symbol = args.symbol if hasattr(args, 'symbol') and args.symbol else 'SPY'
    initial_capital = args.capital if hasattr(args, 'capital') and args.capital else 10000.0
    asset_type = args.asset_type if hasattr(args, 'asset_type') and args.asset_type else 'equity'
    
    logger.info(f"Running {strategy_name} backtest on {symbol} from {start_date.date()} to {end_date.date()}")
    
    # Load data
    use_sim = hasattr(args, 'use_simulation') and args.use_simulation
    data = load_data(symbol, start_date, end_date, use_simulation=use_sim)
    
    if data.empty:
        logger.error("No data available for backtest")
        sys.exit(1)
    
    # Create appropriate strategy based on name
    if strategy_name in ['dmt_v2', 'enhanced_dmt_v2', 'turbo_dmt_v2']:
        # Map strategy name to version
        version_map = {
            'dmt_v2': 'original',
            'enhanced_dmt_v2': 'enhanced',
            'turbo_dmt_v2': 'turbo'
        }
        
        version = version_map.get(strategy_name, 'original')
        
        # Create backtester with DMT_v2 strategy
        backtester = Backtester(
            data=data,
            strategy='dmt_v2',
            strategy_params={
                'version': version,
                'asset_type': asset_type,
                'lookback_period': args.lookback if hasattr(args, 'lookback') else 252
            },
            initial_capital=initial_capital,
            benchmark_key='Buy & Hold'
        )
        
    else:
        logger.error(f"Strategy {strategy_name} not implemented yet")
        sys.exit(1)
    
    # Run backtest
    results = backtester.run()
    
    # Print metrics
    backtester.print_metrics()
    
    # Generate plots
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"{output_dir}/{symbol.lower()}_{strategy_name}_backtest.png"
    backtester.plot_results(
        title=f"{symbol} - {strategy_name.upper()} Backtest",
        filename=plot_filename
    )
    
    # Generate regime plot if applicable
    if strategy_name == 'turbo_dmt_v2':
        regime_filename = f"{output_dir}/{symbol.lower()}_{strategy_name}_regimes.png"
        backtester.plot_regime_analysis(
            ticker=symbol,
            filename=regime_filename
        )
    
    # Save results if requested
    if hasattr(args, 'save_results') and args.save_results:
        results_filename = f"{output_dir}/{symbol.lower()}_{strategy_name}_results.csv"
        backtester.save_results(results_filename)

def run_compare_strategies(args: argparse.Namespace) -> None:
    """Run comparison of multiple strategies"""
    start_date, end_date = get_date_range(args)
    symbol = args.symbol if hasattr(args, 'symbol') and args.symbol else 'SPY'
    initial_capital = args.capital if hasattr(args, 'capital') and args.capital else 10000.0
    asset_type = args.asset_type if hasattr(args, 'asset_type') and args.asset_type else 'equity'
    
    # Get strategies to compare
    if not hasattr(args, 'strategies') or not args.strategies:
        logger.error("No strategies specified for comparison")
        sys.exit(1)
    
    strategy_names = args.strategies.split(',')
    logger.info(f"Comparing strategies: {', '.join(strategy_names)} on {symbol}")
    
    # Load data
    use_sim = hasattr(args, 'use_simulation') and args.use_simulation
    data = load_data(symbol, start_date, end_date, use_simulation=use_sim)
    
    if data.empty:
        logger.error("No data available for backtest")
        sys.exit(1)
    
    # Create strategy configs for batch backtest
    strategy_configs = []
    
    for name in strategy_names:
        if name in ['dmt_v2', 'enhanced_dmt_v2', 'turbo_dmt_v2']:
            # Map strategy name to version
            version_map = {
                'dmt_v2': 'original',
                'enhanced_dmt_v2': 'enhanced', 
                'turbo_dmt_v2': 'turbo'
            }
            
            version = version_map.get(name, 'original')
            
            strategy_configs.append({
                'name': name.upper(),
                'strategy': 'dmt_v2',
                'params': {
                    'version': version,
                    'asset_type': asset_type,
                    'lookback_period': args.lookback if hasattr(args, 'lookback') else 252
                }
            })
        else:
            logger.warning(f"Strategy {name} not implemented yet, skipping")
    
    if not strategy_configs:
        logger.error("No valid strategies to compare")
        sys.exit(1)
    
    # Run batch backtest
    batch_tester = BatchBacktester(
        data=data,
        strategy_configs=strategy_configs,
        initial_capital=initial_capital,
        benchmark_key='Buy & Hold'
    )
    
    batch_tester.run()
    batch_tester.print_summary()
    
    # Generate plot
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else 'backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"{output_dir}/{symbol.lower()}_strategy_comparison.png"
    batch_tester.plot_comparison(
        title=f"{symbol} - Strategy Comparison",
        filename=plot_filename
    )

def run_live(args: argparse.Namespace, strategy_name: str) -> None:
    """Run live trading for a specific strategy"""
    logger.warning("Live trading not implemented yet")
    # This is a placeholder for future implementation
    print(f"Would run {strategy_name} in live mode with args: {args}")
    sys.exit(0)

def run_paper(args: argparse.Namespace, strategy_name: str) -> None:
    """Run paper trading for a specific strategy"""
    logger.warning("Paper trading not implemented yet")
    # This is a placeholder for future implementation
    print(f"Would run {strategy_name} in paper trading mode with args: {args}")
    sys.exit(0)

# Parser setup functions
def setup_common_parser_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser"""
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbol', default='SPY', help='Symbol to trade')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--output-dir', default='backtest_results', help='Output directory for results')
    parser.add_argument('--save-results', action='store_true', help='Save results to CSV')
    parser.add_argument('--use-simulation', action='store_true', help='Use simulated market data')

def setup_dmt_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for DMT strategy commands"""
    parser = subparsers.add_parser('dmt_v2', help='DMT v2 strategy commands')
    subcommands = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest = subcommands.add_parser('backtest', help='Run backtest')
    setup_common_parser_args(backtest)
    backtest.add_argument('--lookback', type=int, default=252, help='Lookback period for calculations')
    backtest.add_argument('--asset-type', default='equity', choices=['equity', 'crypto'], 
                         help='Asset type (impacts strategy parameters)')
    
    # Live command
    live = subcommands.add_parser('run', help='Run live trading')
    setup_common_parser_args(live)
    
    # Paper command
    paper = subcommands.add_parser('paper', help='Run paper trading')
    setup_common_parser_args(paper)

def setup_enhanced_dmt_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for Enhanced DMT strategy commands"""
    parser = subparsers.add_parser('enhanced_dmt_v2', help='Enhanced DMT v2 strategy commands')
    subcommands = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest = subcommands.add_parser('backtest', help='Run backtest')
    setup_common_parser_args(backtest)
    backtest.add_argument('--lookback', type=int, default=252, help='Lookback period for calculations')
    backtest.add_argument('--asset-type', default='equity', choices=['equity', 'crypto'], 
                         help='Asset type (impacts strategy parameters)')
    
    # Live command
    live = subcommands.add_parser('run', help='Run live trading')
    setup_common_parser_args(live)
    
    # Paper command
    paper = subcommands.add_parser('paper', help='Run paper trading')
    setup_common_parser_args(paper)

def setup_turbo_dmt_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for TurboDMT strategy commands"""
    parser = subparsers.add_parser('turbo_dmt_v2', help='TurboDMT v2 strategy commands')
    subcommands = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest = subcommands.add_parser('backtest', help='Run backtest')
    setup_common_parser_args(backtest)
    backtest.add_argument('--lookback', type=int, default=252, help='Lookback period for calculations')
    backtest.add_argument('--asset-type', default='equity', choices=['equity', 'crypto'], 
                         help='Asset type (impacts strategy parameters)')
    
    # Live command
    live = subcommands.add_parser('run', help='Run live trading')
    setup_common_parser_args(live)
    
    # Paper command
    paper = subcommands.add_parser('paper', help='Run paper trading')
    setup_common_parser_args(paper)

def setup_tri_shot_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for Tri-Shot strategy commands"""
    parser = subparsers.add_parser('tri_shot', help='Tri-Shot strategy commands')
    subcommands = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest = subcommands.add_parser('backtest', help='Run backtest')
    setup_common_parser_args(backtest)
    
    # Live command
    live = subcommands.add_parser('run', help='Run live trading')
    setup_common_parser_args(live)
    
    # Paper command
    paper = subcommands.add_parser('paper', help='Run paper trading')
    setup_common_parser_args(paper)

def setup_turbo_qt_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for TurboQT strategy commands"""
    parser = subparsers.add_parser('turbo_qt', help='TurboQT strategy commands')
    subcommands = parser.add_subparsers(dest='command')
    
    # Backtest command
    backtest = subcommands.add_parser('backtest', help='Run backtest')
    setup_common_parser_args(backtest)
    
    # Live command
    live = subcommands.add_parser('run', help='Run live trading')
    setup_common_parser_args(live)
    
    # Paper command
    paper = subcommands.add_parser('paper', help='Run paper trading')
    setup_common_parser_args(paper)

def setup_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up parser for strategy comparison"""
    parser = subparsers.add_parser('compare', help='Compare multiple strategies')
    setup_common_parser_args(parser)
    parser.add_argument('--strategies', required=True, 
                       help='Comma-separated list of strategies to compare')
    parser.add_argument('--lookback', type=int, default=252, help='Lookback period for calculations')
    parser.add_argument('--asset-type', default='equity', choices=['equity', 'crypto'], 
                      help='Asset type (impacts strategy parameters)')

def main() -> None:
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Unified Trading Strategy CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Create subparsers for each strategy
    subparsers = parser.add_subparsers(dest='strategy', help='Strategy to use')
    
    # Set up parsers for each strategy
    setup_dmt_parser(subparsers)
    setup_enhanced_dmt_parser(subparsers)
    setup_turbo_dmt_parser(subparsers)
    setup_tri_shot_parser(subparsers)
    setup_turbo_qt_parser(subparsers)
    setup_compare_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle the case when no arguments are provided
    if not args.strategy:
        parser.print_help()
        sys.exit(0)
    
    # Forward to appropriate handler based on strategy and command
    if args.strategy == 'compare':
        run_compare_strategies(args)
        sys.exit(0)
    
    # Handle strategy-specific commands
    if not hasattr(args, 'command') or not args.command:
        logger.error(f"No command specified for {args.strategy} strategy")
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'backtest':
        run_backtest(args, args.strategy)
    elif args.command == 'run':
        run_live(args, args.strategy)
    elif args.command == 'paper':
        run_paper(args, args.strategy)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
