#!/usr/bin/env python3
"""
Compare the performance of all three trading strategies.
This script runs the backtests and displays them in a standardized format.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import traceback
from datetime import datetime

# Add project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import necessary backtest functions
from stock_trader_o3_algo.strategies.dmt.dmt_backtest import run_dmt_backtest
from stock_trader_o3_algo.strategies.turbo_qt.turbo_qt_backtest import TurboBacktester
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest
from stock_trader_o3_algo.strategies.tri_shot.tri_shot_features import fetch_data_from_date
from bin import tri_shot_cli # Import from bin directory

def format_percent(value):
    """Format a value as a percentage."""
    return f"{value*100:.2f}%"

def format_dollar(value):
    """Format a value as dollars."""
    return f"${value:.2f}"

def run_and_collect_results(strategy_name, backtest_func, start_date, end_date, capital, **kwargs):
    """Runs a backtest function and formats its results."""
    print(f"--- Running {strategy_name} Backtest --- ")
    try:
        results_df = None
        performance_metrics = {}
        
        # Different handling based on strategy
        if strategy_name == "Tri-Shot":
            # Tri-Shot returns a dictionary of metrics directly
            performance_metrics = backtest_func(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=capital,
                plot=False,
                monte_carlo=False,
                slippage_bps=kwargs.get('slippage_bps', 1),
                commission_bps=kwargs.get('commission_bps', 1)
            )
            
            # Results saved to CSV by the backtest function, load it
            try:
                results_file = os.path.join('tri_shot_data', 'backtest_results.csv')
                results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
                performance_metrics['Strategy'] = strategy_name
                
                # Actual start/end dates based on data in CSV
                actual_start = results_df.index.min()
                actual_end = results_df.index.max()
                
                # Add date range to metrics
                performance_metrics['Period'] = f"{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')} (partial)"
                performance_metrics['Trading Days'] = (actual_end - actual_start).days
                
                # Add equity curve to metrics
                if 'equity' in results_df.columns:
                    performance_metrics['Equity Curve'] = results_df['equity']
                
            except Exception as e:
                print(f"Warning: Error loading backtest results: {e}")
            
            return results_df, performance_metrics
        
        elif strategy_name == "DMT":
            try:
                print("--- Running DMT Backtest ---")
                
                # Import yfinance for data fetching
                import yfinance as yf
                
                # Fetch price data for DMT
                ticker = 'SPY'
                start_fetch = (start_date - pd.Timedelta(days=252)).strftime('%Y-%m-%d')  # Extra year for lookback
                end_fetch = end_date.strftime('%Y-%m-%d')
                
                print(f"Fetching {ticker} data from {start_fetch} to {end_fetch}")
                raw_prices = yf.download(ticker, start=start_fetch, end=end_fetch)
                
                if isinstance(raw_prices.columns, pd.MultiIndex):
                    # Flatten MultiIndex if present
                    prices = pd.DataFrame()
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if (col, ticker) in raw_prices.columns:
                            prices[col] = raw_prices[(col, ticker)]
                else:
                    prices = raw_prices
                
                print(f"Downloaded {len(prices)} days of {ticker} price data")
                print(f"Price index type: {type(prices.index)}")
                print(f"First date: {prices.index[0]}, type: {type(prices.index[0])}")
                print(f"start_date: {start_date}, type: {type(start_date)}")
                print(f"end_date: {end_date}, type: {type(end_date)}")
                
                # Option 1: Manual filtering using dates (avoiding index comparison issues)
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                dates_in_range = [(date.strftime('%Y-%m-%d') >= start_date_str) and 
                                (date.strftime('%Y-%m-%d') <= end_date_str) 
                                for date in prices.index]
                filtered_prices = prices.iloc[dates_in_range]
                print(f"Filtered to {len(filtered_prices)} days using string comparison")
                
                # Run actual DMT backtest with proper date parameters
                from stock_trader_o3_algo.strategies.dmt.dmt_backtest import run_dmt_backtest
                
                results_df, metrics = run_dmt_backtest(
                    prices=filtered_prices,
                    initial_capital=capital,
                    n_epochs=100,
                    learning_rate=0.01,
                    target_annual_vol=0.20,
                    vol_window=20,
                    max_position_size=1.0,
                    neutral_zone=0.05
                )
                
                # Create comprehensive performance metrics
                performance_metrics = {
                    'Strategy': strategy_name,
                    'Period': f"{results_df.index[0].strftime('%Y-%m-%d')} to {results_df.index[-1].strftime('%Y-%m-%d')}",
                    'Days': len(results_df),
                    'Initial Value': metrics['initial_value'],
                    'Final Value': metrics['final_value'],
                    'Total Return': (metrics['final_value'] / metrics['initial_value'] - 1) * 100,
                    'CAGR': metrics['cagr'] * 100,
                    'Volatility': metrics['volatility'] * 100,
                    'Max Drawdown': metrics['max_drawdown'] * 100,
                    'Sharpe Ratio': metrics['sharpe_ratio']
                }
                
                print(f"--- DMT Backtest Complete ---")
            except Exception as e:
                print(f"Warning: Error running DMT backtest: {e}")
                print(f"Warning: DMT backtest did not return valid results.")
                return None, None
        
        elif strategy_name == "TurboQT":
            try:
                backtester = backtest_func(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    initial_capital=capital
                )
                
                # Run the backtest
                backtester.run_backtest()
                
                # Load the results from CSV
                turbo_file = os.path.join('tri_shot_data', 'turbo_qt_backtest_results.csv')
                if os.path.exists(turbo_file):
                    results_df = pd.read_csv(turbo_file, index_col=0, parse_dates=True)
                    
                    # Find equity column (should be strategy_equity)
                    equity_col = 'strategy_equity' if 'strategy_equity' in results_df.columns else None
                    if equity_col is None:
                        for col in results_df.columns:
                            if 'equity' in col.lower() and 'bench' not in col.lower():
                                equity_col = col
                                break
                    
                    if equity_col:
                        equity_curve = results_df[equity_col]
                        performance_metrics['Strategy'] = strategy_name
                        performance_metrics['Initial Value'] = equity_curve.iloc[0]
                        performance_metrics['Final Value'] = equity_curve.iloc[-1]
                        performance_metrics['Total Return'] = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
                        performance_metrics['Period'] = f"{results_df.index.min().strftime('%Y-%m-%d')} to {results_df.index.max().strftime('%Y-%m-%d')}"
                        performance_metrics['Trading Days'] = len(results_df)
                        performance_metrics['Equity Curve'] = equity_curve
                        
                        # Calculate years for CAGR
                        days = (results_df.index.max() - results_df.index.min()).days
                        years = days / 365.25
                        performance_metrics['CAGR'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
                        
                        # Calculate volatility
                        returns = equity_curve.pct_change().dropna()
                        performance_metrics['Volatility'] = returns.std() * np.sqrt(252)
                        
                        # Calculate drawdown
                        peak = equity_curve.cummax()
                        drawdown = (equity_curve / peak - 1)
                        performance_metrics['Max Drawdown'] = drawdown.min()
                        
                        # Calculate Sharpe ratio
                        performance_metrics['Sharpe Ratio'] = (performance_metrics['CAGR'] - 0.02) / performance_metrics['Volatility']
            except Exception as e:
                print(f"Warning: Error processing TurboQT results: {e}")
                
            return results_df, performance_metrics
        
        elif strategy_name == "DMT_v2":
            try:
                print("--- Running DMT_v2 Backtest ---")
                
                # Import yfinance for data fetching
                import yfinance as yf
                
                # Fetch price data for DMT_v2
                ticker = 'SPY'
                start_fetch = (start_date - pd.Timedelta(days=252)).strftime('%Y-%m-%d')  # Extra year for lookback
                end_fetch = end_date.strftime('%Y-%m-%d')
                
                print(f"Fetching {ticker} data from {start_fetch} to {end_fetch}")
                raw_prices = yf.download(ticker, start=start_fetch, end=end_fetch)
                
                if isinstance(raw_prices.columns, pd.MultiIndex):
                    # Flatten MultiIndex if present
                    prices = pd.DataFrame()
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if (col, ticker) in raw_prices.columns:
                            prices[col] = raw_prices[(col, ticker)]
                else:
                    prices = raw_prices
                
                print(f"Downloaded {len(prices)} days of {ticker} price data")
                print(f"Price index type: {type(prices.index)}")
                print(f"First date: {prices.index[0]}, type: {type(prices.index[0])}")
                print(f"start_date: {start_date}, type: {type(start_date)}")
                print(f"end_date: {end_date}, type: {type(end_date)}")
                
                # Option 1: Manual filtering using dates (avoiding index comparison issues)
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                dates_in_range = [(date.strftime('%Y-%m-%d') >= start_date_str) and 
                                (date.strftime('%Y-%m-%d') <= end_date_str) 
                                for date in prices.index]
                filtered_prices = prices.iloc[dates_in_range]
                print(f"Filtered to {len(filtered_prices)} days using string comparison")
                
                # Run actual DMT_v2 backtest with enhanced parameters but ensure same date range
                from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest
                
                results_df, metrics = run_dmt_v2_backtest(
                    prices=filtered_prices,
                    initial_capital=capital,
                    n_epochs=100,  # Use more epochs for better results
                    learning_rate=0.015,  # Higher learning rate
                    neutral_zone=0.03,  # Smaller neutral zone for more trades
                    target_annual_vol=0.35,  # Higher target volatility
                    max_position_size=2.0,  # More aggressive position sizing
                    plot=False
                )
                
                # Create comprehensive performance metrics
                performance_metrics = {
                    'Strategy': strategy_name,
                    'Period': f"{results_df.index[0].strftime('%Y-%m-%d')} to {results_df.index[-1].strftime('%Y-%m-%d')}",
                    'Days': len(results_df),
                    'Initial Value': metrics['initial_value'],
                    'Final Value': metrics['final_value'],
                    'Total Return': (metrics['final_value'] / metrics['initial_value'] - 1),  # Store as ratio, not percentage
                    'CAGR': metrics['cagr'],  # Store as ratio, not percentage
                    'Volatility': metrics['volatility'],  # Store as ratio, not percentage
                    'Max Drawdown': metrics['max_drawdown'],  # Store as ratio, not percentage
                    'Sharpe Ratio': metrics['sharpe_ratio']
                }
                
                print(f"--- DMT_v2 Backtest Complete ---")
            except Exception as e:
                print(f"Warning: Error running DMT_v2 backtest: {e}")
                print(f"Warning: DMT_v2 backtest did not return valid results.")
                return None, None
        
        # Common processing for any strategy
        if results_df is None:
            print(f"Warning: {strategy_name} backtest did not return valid results.")
            return None
            
        # Set strategy name in metrics
        if 'Strategy' not in performance_metrics:
            performance_metrics['Strategy'] = strategy_name
            
        # Add period if not already set
        if 'Period' not in performance_metrics:
            performance_metrics['Period'] = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            
        # Add trading days count
        if 'Trading Days' not in performance_metrics and results_df is not None:
            performance_metrics['Trading Days'] = len(results_df)
            
        # Common checks for standard metrics and equity curve
        if 'strategy_equity' in results_df.columns:
            equity_curve = results_df['strategy_equity']
            
            # Add equity curve to metrics for plotting
            performance_metrics['Equity Curve'] = equity_curve
            
            # Ensure basic metrics exist
            if 'Initial Value' not in performance_metrics:
                performance_metrics['Initial Value'] = equity_curve.iloc[0]
                
            if 'Final Value' not in performance_metrics:
                performance_metrics['Final Value'] = equity_curve.iloc[-1]
                
            if 'Total Return' not in performance_metrics:
                performance_metrics['Total Return'] = performance_metrics['Final Value'] / performance_metrics['Initial Value'] - 1
                
            # Calculate CAGR if missing
            if 'CAGR' not in performance_metrics:
                days = (end_date - start_date).days
                years = days / 365.25 if days > 0 else 1
                performance_metrics['CAGR'] = (performance_metrics['Final Value'] / performance_metrics['Initial Value']) ** (1 / years) - 1
                
            # Calculate volatility if missing
            if 'Volatility' not in performance_metrics:
                returns = equity_curve.pct_change().dropna()
                performance_metrics['Volatility'] = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                
            # Calculate max drawdown if missing
            if 'Max Drawdown' not in performance_metrics:
                peak = equity_curve.cummax()
                drawdown = (equity_curve / peak - 1)
                performance_metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
                
            # Calculate Sharpe ratio if missing
            if 'Sharpe Ratio' not in performance_metrics:
                vol = performance_metrics['Volatility']
                cagr = performance_metrics['CAGR']
                performance_metrics['Sharpe Ratio'] = (cagr - 0.02) / vol if vol > 0 else 0
        
        print(f"--- {strategy_name} Backtest Complete --- ")
        return results_df, performance_metrics

    except Exception as e:
        print(f"Error running {strategy_name} backtest: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None

def compare_strategies(start_date, end_date, initial_capital=500.0):
    """Run all strategy backtests and compare performance."""
    print("\n" + "=" * 80)
    print("                          STRATEGY COMPARISON SUMMARY                           ".center(80))
    print("=" * 80)
    
    # Create list to store all strategy results
    all_results = []
    
    # Run Tri-Shot backtest
    tri_shot_df, tri_shot_metrics = run_and_collect_results(
        "Tri-Shot", 
        tri_shot_cli.backtest, 
        start_date, 
        end_date, 
        initial_capital
    )
    if tri_shot_metrics is not None:
        all_results.append(tri_shot_metrics)
    
    # Run DMT backtest
    dmt_df, dmt_metrics = run_and_collect_results(
        "DMT", 
        None,  # Pass None since we do custom data loading in run_and_collect_results
        start_date, 
        end_date, 
        initial_capital
    )
    if dmt_metrics is not None:
        all_results.append(dmt_metrics)
    
    # Run TurboQT backtest
    turbo_qt_df, turbo_qt_metrics = run_and_collect_results(
        "TurboQT", 
        TurboBacktester, 
        start_date, 
        end_date, 
        initial_capital
    )
    if turbo_qt_metrics is not None:
        all_results.append(turbo_qt_metrics)
    
    # Run DMT_v2 backtest
    dmt_v2_df, dmt_v2_metrics = run_and_collect_results(
        "DMT_v2", 
        None,  # Pass None since we do custom data loading in run_and_collect_results
        start_date, 
        end_date, 
        initial_capital
    )
    if dmt_v2_metrics is not None:
        all_results.append(dmt_v2_metrics)
    
    # Create summary dataframe
    if all_results:
        # --- Display results table ---
        summary_df = pd.DataFrame(all_results)
        
        # Remove any _Note column and add it to Period if present
        if '_Note' in summary_df.columns:
            for idx, row in summary_df.iterrows():
                if pd.notna(row.get('_Note')):
                    summary_df.loc[idx, 'Period'] = f"{row['Period']} {row['_Note']}"
            summary_df = summary_df.drop(columns=['_Note'])
            
        summary_df = summary_df.set_index('Strategy')

        # Ensure each strategy has the proper percentage values
        for idx, row in summary_df.iterrows():
            # Make sure percentages are actual percentages and not ratios
            for col in ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown']:
                if col in summary_df.columns and pd.notna(row.get(col)):
                    # Fix values that are exaggerated by 100x
                    if abs(row[col]) > 100 and idx == 'DMT_v2':
                        summary_df.loc[idx, col] = row[col] / 100

        # Formatting
        formatters = {
            'Initial Value': format_dollar,
            'Final Value': format_dollar,
            'Total Return': format_percent,
            'CAGR': format_percent,
            'Volatility': format_percent,
            'Max Drawdown': format_percent,
            'Sharpe Ratio': '{:.2f}'.format
        }
        formatted_summary = summary_df.copy()
        for col, formatter in formatters.items():
            if col in formatted_summary.columns:
                 try:
                     formatted_summary[col] = formatted_summary[col].apply(formatter)
                 except:
                     print(f"Warning: Could not format column '{col}'")
            else:
                 print(f"Warning: Column '{col}' not found for formatting.")

        # Rename columns for display
        if 'Trading Days' in formatted_summary.columns:
            formatted_summary = formatted_summary.rename(columns={'Trading Days': 'Days'})

        # Select and order columns for display
        display_cols = ['Period', 'Days', 'Initial Value', 'Final Value', 'Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Sharpe Ratio']
        # Ensure columns exist before selecting
        display_cols = [col for col in display_cols if col in formatted_summary.columns]
        formatted_summary = formatted_summary[display_cols]

        print("\n" + "="*80)
        print(f"{'STRATEGY COMPARISON SUMMARY':^80}")
        print("="*80)
        print(formatted_summary.to_string(justify='right'))
        print("-"*80)

        # --- Generate comparison plot ---
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 8))

        for strategy_name in ["Tri-Shot", "DMT", "TurboQT", "DMT_v2"]:
            for res in all_results:
                if res.get('Strategy') == strategy_name and 'Equity Curve' in res:
                    equity_curve = res['Equity Curve']
                    # Find Sharpe ratio if available
                    sharpe = "N/A"
                    for r in all_results:
                        if r.get('Strategy') == strategy_name and 'Sharpe Ratio' in r:
                            sharpe = f"{r['Sharpe Ratio']:.2f}"
                            break
                            
                    # Normalize equity curves to start at 1.0 for comparison
                    normalized_equity = equity_curve / equity_curve.iloc[0]
                    plt.plot(normalized_equity.index, normalized_equity, label=f"{strategy_name} (Sharpe: {sharpe})")

        plt.title(f'Strategy Performance Comparison\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
        plt.ylabel('Normalized Equity')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.yscale('log') # Use log scale for better visibility of differences
        plt.tight_layout()

        # Save plot
        output_dir = 'tri_shot_data'
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'strategy_comparison.png')
        plt.savefig(plot_file)
        print(f"Comparison chart saved to {plot_file}")
        # plt.show() # Optionally display the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare trading strategies by running backtests.')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=500.0, help='Initial capital for each strategy')

    args = parser.parse_args()

    # Parse dates with timezone awareness (assuming America/New_York like trader_cli)
    try:
        start_dt = pd.to_datetime(args.start_date).tz_localize('America/New_York')
        # Ensure end_date includes the full day
        end_dt = pd.to_datetime(args.end_date).tz_localize('America/New_York') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if start_dt >= end_dt:
             raise ValueError("Start date must be before end date.")
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    compare_strategies(start_dt, end_dt, args.capital)
