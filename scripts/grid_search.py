"""
Grid search optimization for the trading strategy.
This script tests multiple parameter combinations to find optimal settings.
"""

import os
import sys
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.config.settings import RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF
from stock_trader_o3_algo.data.price_data import fetch_prices


def run_backtest_with_params(params: Dict) -> Dict:
    """
    Run a single backtest with the given parameters.
    
    Args:
        params: Dictionary of parameters to use
        
    Returns:
        Dictionary with backtest results
    """
    # Extract parameters
    start_date = params.get('start_date', '2007-01-01')
    end_date = params.get('end_date', '2024-03-31')
    lookback_days = params.get('lookback_days', 60)
    short_lookback = params.get('short_lookback', 10)
    vol_target = params.get('vol_target', 0.02)
    trade_weekdays = params.get('trade_weekdays', (0, 2, 4))
    
    # Temporarily modify settings module
    from stock_trader_o3_algo.config import settings
    original_lookback = settings.LOOKBACK_DAYS
    original_short_lookback = settings.SHORT_LOOKBACK
    original_vol_target = settings.WEEKLY_VOL_TARGET
    
    # Apply new settings
    settings.LOOKBACK_DAYS = lookback_days
    settings.SHORT_LOOKBACK = short_lookback
    settings.WEEKLY_VOL_TARGET = vol_target
    
    try:
        # Create and run backtest
        backtest = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100.0,
            transaction_cost_pct=0.0003,
            trade_weekdays=trade_weekdays
        )
        
        # Run backtest
        equity_curve = backtest.run_backtest()
        
        # Get results
        results = {
            'lookback_days': lookback_days,
            'short_lookback': short_lookback,
            'vol_target': vol_target,
            'trade_weekdays': trade_weekdays,
            'final_capital': backtest.portfolio_stats['final_capital'],
            'total_return': backtest.portfolio_stats['total_return'],
            'cagr': backtest.portfolio_stats['cagr'],
            'max_drawdown': backtest.portfolio_stats['max_drawdown'],
            'sharpe_ratio': backtest.portfolio_stats['sharpe_ratio'],
            'sortino_ratio': backtest.portfolio_stats['sortino_ratio'],
            'num_trades': backtest.portfolio_stats['num_trades']
        }
        
        return results
    
    finally:
        # Restore original settings
        settings.LOOKBACK_DAYS = original_lookback
        settings.SHORT_LOOKBACK = original_short_lookback
        settings.WEEKLY_VOL_TARGET = original_vol_target


def run_grid_search(
    start_date: str = "2007-01-01",
    end_date: str = "2024-03-31",
    lookback_days_values: List[int] = [40, 60, 90],
    short_lookback_values: List[int] = [5, 10, 15],
    vol_target_values: List[float] = [0.01, 0.02, 0.03, 0.04],
    trade_weekdays_options: List[Tuple[int, ...]] = [(0,), (2,), (4,), (0, 2, 4)],
    parallel: bool = True
) -> pd.DataFrame:
    """
    Run a grid search over multiple parameter combinations.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        lookback_days_values: List of LOOKBACK_DAYS values to test
        short_lookback_values: List of SHORT_LOOKBACK values to test
        vol_target_values: List of WEEKLY_VOL_TARGET values to test
        trade_weekdays_options: List of trading day combinations to test
        parallel: Whether to use parallel processing
        
    Returns:
        DataFrame with results for all parameter combinations
    """
    # Fetch all price data once to have it cached
    tickers = [RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF]
    fetch_prices(tickers, start_date=start_date, end_date=end_date, use_cache=True)
    
    # Generate all parameter combinations
    param_combos = []
    for lookback, short_lookback, vol_target, weekdays in product(
        lookback_days_values, short_lookback_values, vol_target_values, trade_weekdays_options
    ):
        # Skip invalid combinations (short lookback must be less than long lookback)
        if short_lookback >= lookback:
            continue
            
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'lookback_days': lookback,
            'short_lookback': short_lookback,
            'vol_target': vol_target,
            'trade_weekdays': weekdays
        }
        param_combos.append(params)
    
    print(f"Running grid search with {len(param_combos)} parameter combinations...")
    
    # Run backtests
    if parallel and mp.cpu_count() > 1:
        # Use parallel processing
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            results = list(pool.map(run_backtest_with_params, param_combos))
    else:
        # Use sequential processing
        results = []
        for i, params in enumerate(param_combos):
            print(f"Testing combination {i+1}/{len(param_combos)}: {params}")
            result = run_backtest_with_params(params)
            results.append(result)
            print(f"Result: Return: {result['total_return']:.2%}, Sharpe: {result['sharpe_ratio']:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Create output directory
    output_dir = Path("backtest_results/grid_search")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save results to CSV
    results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    print(f"Saved grid search results to {output_dir / 'grid_search_results.csv'}")
    
    # Plot top 10 parameter combinations
    top_10 = results_df.head(10)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(top_10.index, top_10['sharpe_ratio'])
    plt.title('Top 10 Sharpe Ratios')
    plt.xticks([])
    
    plt.subplot(2, 2, 2)
    plt.bar(top_10.index, top_10['total_return'])
    plt.title('Total Returns for Top 10')
    plt.xticks([])
    
    plt.subplot(2, 2, 3)
    plt.bar(top_10.index, top_10['max_drawdown'])
    plt.title('Max Drawdowns for Top 10')
    plt.xticks([])
    
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['max_drawdown'], results_df['total_return'], 
               c=results_df['sharpe_ratio'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Return vs. Drawdown (All Combinations)')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Total Return')
    
    plt.tight_layout()
    plt.savefig(output_dir / "grid_search_summary.png")
    print(f"Saved grid search plots to {output_dir / 'grid_search_summary.png'}")
    
    # Print best parameters
    best_params = results_df.iloc[0]
    print("\nBest parameter combination:")
    print(f"LOOKBACK_DAYS: {best_params['lookback_days']}")
    print(f"SHORT_LOOKBACK: {best_params['short_lookback']}")
    print(f"WEEKLY_VOL_TARGET: {best_params['vol_target']}")
    print(f"TRADE_WEEKDAYS: {best_params['trade_weekdays']}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_params['total_return']:.2%}")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2%}")
    
    return results_df


if __name__ == "__main__":
    # Run grid search with default parameters
    results = run_grid_search(
        start_date="2010-01-01",  
        end_date="2024-03-31",
        lookback_days_values=[40, 60, 90],
        short_lookback_values=[5, 10, 15],
        vol_target_values=[0.02, 0.03, 0.04],
        trade_weekdays_options=[(0,), (0, 2, 4)],  # Test Monday only vs. M/W/F
        parallel=True
    )
