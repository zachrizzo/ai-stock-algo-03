"""
Main script for running backtest of the micro-CTA strategy.
"""
import argparse
import datetime as dt
import os

import pandas as pd
import matplotlib.pyplot as plt

from stock_trader_o3_algo.backtest.backtest_engine import BacktestEngine
from stock_trader_o3_algo.config.settings import RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtest for micro-CTA strategy')
    
    # Date range arguments
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=dt.datetime.now().strftime('%Y-%m-%d'),
                        help='End date for backtest (YYYY-MM-DD)')
    
    # Capital arguments
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Initial capital for backtest')
    
    # Trading cost arguments
    parser.add_argument('--cost', type=float, default=0.0003,
                        help='Transaction cost as percentage (default: 0.0003 = 3 basis points)')
    
    # Output arguments
    parser.add_argument('--save', action='store_true',
                        help='Save backtest results to disk')
    
    # Benchmark arguments
    parser.add_argument('--benchmark', type=str, default=RISK_ON,
                        help=f'Benchmark symbol (default: {RISK_ON})')
    
    # Window arguments for rolling analysis
    parser.add_argument('--window', type=int, default=0,
                        help='Window size in years for rolling analysis (0 to disable)')
    
    # Rolling window step
    parser.add_argument('--step', type=int, default=12,
                        help='Step size in months for rolling windows')
    
    return parser.parse_args()


def run_single_backtest(start_date, end_date, initial_capital, transaction_cost_pct, 
                        save_results=False, benchmark_symbol=RISK_ON):
    """Run a single backtest and display results."""
    print(f"Running backtest from {start_date} to {end_date} with {initial_capital:.2f} initial capital")
    
    # Create and run backtest
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct
    )
    
    # Run the backtest
    equity_curve = backtest.run_backtest()
    
    # Print statistics
    print("\nBacktest Results:")
    for key, value in backtest.portfolio_stats.items():
        if isinstance(value, float):
            if key.endswith('rate') or key.endswith('ratio') or key == 'cagr':
                print(f"{key}: {value:.2f}")
            elif key.endswith('return') or key.endswith('drawdown'):
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot equity curve
    backtest.plot_equity_curve(benchmark_symbol=benchmark_symbol)
    
    # Plot asset allocation
    backtest.plot_asset_allocation()
    
    # Save results if requested
    if save_results:
        output_dir = backtest.save_results()
        print(f"\nResults saved to: {output_dir}")
    
    return backtest


def run_rolling_window_analysis(start_date, end_date, window_years, step_months, 
                                initial_capital, transaction_cost_pct, benchmark_symbol=RISK_ON):
    """Run rolling window analysis."""
    print(f"Running rolling window analysis with {window_years} year windows, "
          f"stepping {step_months} months at a time")
    
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Calculate window size in days
    window_days = int(window_years * 365.25)
    step_days = int(step_months * 30.44)  # Average month length
    
    # Generate windows
    windows = []
    window_start = start
    
    while window_start + pd.Timedelta(days=window_days) <= end:
        window_end = window_start + pd.Timedelta(days=window_days)
        windows.append((window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')))
        window_start += pd.Timedelta(days=step_days)
    
    if not windows:
        print("No valid windows found with the specified parameters.")
        return
    
    # Run backtest for each window
    results = []
    
    for i, (window_start, window_end) in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}: {window_start} to {window_end}")
        
        backtest = BacktestEngine(
            start_date=window_start,
            end_date=window_end,
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost_pct
        )
        
        backtest.run_backtest()
        
        # Store results
        result = {
            'start_date': window_start,
            'end_date': window_end,
            'cagr': backtest.portfolio_stats['cagr'],
            'sharpe': backtest.portfolio_stats['sharpe_ratio'],
            'sortino': backtest.portfolio_stats['sortino_ratio'],
            'max_dd': backtest.portfolio_stats['max_drawdown'],
            'ulcer_index': backtest.portfolio_stats['ulcer_index'],
            'win_rate': backtest.portfolio_stats['win_rate_monthly'],
            'num_trades': backtest.portfolio_stats['num_trades'],
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate benchmark metrics for comparison
    benchmark_results = []
    
    for start, end in windows:
        # Fetch benchmark data
        from stock_trader_o3_algo.data.price_data import fetch_prices
        prices = fetch_prices([benchmark_symbol], end_date=end)
        
        # Filter to window
        mask = (prices.index >= start) & (prices.index <= end)
        benchmark_prices = prices.loc[mask, benchmark_symbol]
        
        if len(benchmark_prices) < 2:
            benchmark_results.append({
                'start_date': start,
                'end_date': end,
                'benchmark_cagr': 0,
                'benchmark_max_dd': 0,
                'benchmark_ulcer': 0
            })
            continue
        
        # Calculate metrics
        from stock_trader_o3_algo.utils.performance import (
            calculate_cagr, calculate_max_drawdown, calculate_ulcer_index
        )
        
        benchmark_results.append({
            'start_date': start,
            'end_date': end,
            'benchmark_cagr': calculate_cagr(benchmark_prices),
            'benchmark_max_dd': calculate_max_drawdown(benchmark_prices),
            'benchmark_ulcer': calculate_ulcer_index(benchmark_prices)
        })
    
    benchmark_df = pd.DataFrame(benchmark_results)
    
    # Merge results
    combined_results = pd.merge(results_df, benchmark_df, on=['start_date', 'end_date'])
    
    # Calculate outperformance metrics
    combined_results['cagr_diff'] = combined_results['cagr'] - combined_results['benchmark_cagr']
    combined_results['dd_ratio'] = combined_results['max_dd'] / combined_results['benchmark_max_dd']
    combined_results['ulcer_ratio'] = combined_results['ulcer_index'] / combined_results['benchmark_ulcer']
    
    # Print summary statistics
    print("\nRolling Window Analysis Summary:")
    print("\nStrategy Metrics:")
    print(results_df.describe().round(4))
    
    print("\nComparison to Benchmark:")
    print(f"Outperformed benchmark in {(combined_results['cagr_diff'] > 0).mean():.1%} of windows")
    print(f"Lower drawdown than benchmark in {(combined_results['dd_ratio'] < 1).mean():.1%} of windows")
    print(f"Lower Ulcer Index than benchmark in {(combined_results['ulcer_ratio'] < 1).mean():.1%} of windows")
    print(f"CAGR > 15% in {(results_df['cagr'] > 0.15).mean():.1%} of windows")
    print(f"Ulcer Index < half of benchmark in {(combined_results['ulcer_ratio'] < 0.5).mean():.1%} of windows")
    
    # Plot rolling metrics
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: CAGR comparison
    combined_results.plot(x='start_date', y=['cagr', 'benchmark_cagr'], 
                         title='CAGR Comparison', ax=axes[0])
    axes[0].set_ylabel('CAGR')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Max Drawdown comparison
    combined_results.plot(x='start_date', y=['max_dd', 'benchmark_max_dd'], 
                         title='Max Drawdown Comparison', ax=axes[1])
    axes[1].set_ylabel('Max Drawdown')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Ulcer Index comparison
    combined_results.plot(x='start_date', y=['ulcer_index', 'benchmark_ulcer'], 
                         title='Ulcer Index Comparison', ax=axes[2])
    axes[2].set_ylabel('Ulcer Index')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return combined_results


def main():
    """Main entry point for the backtest script."""
    args = parse_args()
    
    if args.window > 0:
        # Run rolling window analysis
        combined_results = run_rolling_window_analysis(
            start_date=args.start,
            end_date=args.end,
            window_years=args.window,
            step_months=args.step,
            initial_capital=args.capital,
            transaction_cost_pct=args.cost,
            benchmark_symbol=args.benchmark
        )
        
        # Save results if requested
        if args.save and combined_results is not None:
            output_dir = os.path.join(
                os.path.dirname(__file__), 
                "backtest_results", 
                f"rolling_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(output_dir, exist_ok=True)
            combined_results.to_csv(os.path.join(output_dir, "rolling_results.csv"), index=False)
            
            plt.figure(figsize=(10, 6))
            combined_results.plot(x='start_date', y=['cagr', 'benchmark_cagr'])
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "cagr_comparison.png"), dpi=300, bbox_inches='tight')
            
            print(f"\nResults saved to: {output_dir}")
    else:
        # Run single backtest
        run_single_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            transaction_cost_pct=args.cost,
            save_results=args.save,
            benchmark_symbol=args.benchmark
        )


if __name__ == "__main__":
    main()
