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
                
                # Extract the actual date range used (not what was requested)
                if len(results_df) > 0:
                    actual_start_date = results_df.index.min()
                    actual_end_date = results_df.index.max()
                    actual_days = (actual_end_date - actual_start_date).days
                    
                    # If there's a significant date mismatch, note it in the period display
                    if actual_start_date > start_date:
                        # Use the actual trading period for calculations but note it's shorter
                        print(f"\nNote: Tri-Shot backtest only ran from {actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}")
                        
                        # Update period to show both requested and actual
                        performance_metrics['Period'] = f"{actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')} (partial)"
                        performance_metrics['Trading Days'] = len(results_df)
                        
                        # Calculate CAGR based on the actual days traded 
                        if 'Total Return' in performance_metrics:
                            total_return = performance_metrics['Total Return']
                            actual_years = actual_days / 365.25
                            if actual_years > 0:
                                performance_metrics['CAGR'] = ((1 + total_return) ** (1/actual_years)) - 1
                                
                                # Also update Sharpe using the corrected CAGR
                                if 'Volatility' in performance_metrics and performance_metrics['Volatility'] > 0:
                                    performance_metrics['Sharpe Ratio'] = (performance_metrics['CAGR'] - 0.02) / performance_metrics['Volatility']
                        
                        # Add annualized indicator to metrics that are affected 
                        if 'CAGR' in performance_metrics:
                            # Add a note to show it's annualized from a shorter period
                            # (will be visible in the table)
                            performance_metrics['_Note'] = f"* Performance annualized from {actual_days} days"
                
            except Exception as e:
                print(f"Warning: Could not load results CSV for {strategy_name}: {e}")
                
        elif strategy_name == "DMT":
            # DMT expects prices DataFrame, not start/end dates
            print(f"Loading data for DMT backtest from {start_date} to {end_date}...")
            # Fetch data for the specified period
            prices = fetch_data_from_date("QQQ", start_date, end_date)
            
            if prices is None or prices.empty:
                print(f"Error: Could not load data for {strategy_name}")
                return None
                
            # Run DMT backtest
            results_df = run_dmt_backtest(
                prices=prices,
                initial_capital=capital,
                learning_rate=kwargs.get('learning_rate', 0.01),
                n_epochs=kwargs.get('epochs', 50)
            )
            
            # Try to load the saved CSV directly
            try:
                results_file = os.path.join('tri_shot_data', 'dmt_backtest_results.csv')
                if os.path.exists(results_file):
                    results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
                    print(f"Loaded DMT results from {results_file}")
                    
                    # Create performance metrics from the result data
                    if 'equity' in results_df.columns:
                        equity_curve = results_df['equity']
                    elif 'strategy_equity' in results_df.columns:
                        equity_curve = results_df['strategy_equity']
                    else:
                        # Try to find any equity column
                        for col in results_df.columns:
                            if 'equity' in col.lower() or ('value' in col.lower() and 'dmt' in col.lower()):
                                equity_curve = results_df[col]
                                print(f"Using column '{col}' as equity for DMT")
                                break
                        else:
                            print("Could not find equity column in DMT results")
                            return None
                            
                    # Create basic metrics
                    init_value = equity_curve.iloc[0]
                    final_value = equity_curve.iloc[-1]
                    days = (equity_curve.index[-1] - equity_curve.index[0]).days
                    years = days / 365.25 if days > 0 else 1
                    
                    performance_metrics = {
                        'Strategy': strategy_name,
                        'Initial Value': init_value,
                        'Final Value': final_value,
                        'Total Return': final_value / init_value - 1,
                        'Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        'Trading Days': len(equity_curve),
                        'CAGR': (final_value / init_value) ** (1 / years) - 1 if years > 0 else 0,
                        'Equity Curve': equity_curve
                    }
                    
                    # Calculate volatility
                    returns = equity_curve.pct_change().dropna()
                    performance_metrics['Volatility'] = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                    
                    # Calculate max drawdown
                    peak = equity_curve.cummax()
                    drawdown = (equity_curve / peak - 1)
                    performance_metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
                    
                    # Calculate Sharpe ratio
                    vol = performance_metrics['Volatility']
                    cagr = performance_metrics['CAGR']
                    performance_metrics['Sharpe Ratio'] = (cagr - 0.02) / vol if vol > 0 else 0
            except Exception as e:
                print(f"Error loading DMT results: {e}")
                performance_metrics = {'Strategy': strategy_name}
                
        elif strategy_name == "TurboQT":
            # Run TurboQT backtest using the class
            backtester = TurboBacktester(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=capital,
                trading_days="mon"
            )
            backtester.run_backtest()
            
            # Try to get results - check implementation details
            try:
                # Try to load the saved CSV results
                turbo_file = os.path.join('tri_shot_data', 'turbo_qt_backtest_results.csv')
                results_df = pd.read_csv(turbo_file, index_col=0, parse_dates=True)
                
                # Extract metrics from the results dataframe
                if results_df is not None and 'strategy_equity' in results_df.columns:
                    equity = results_df['strategy_equity']
                    init_value = equity.iloc[0]
                    final_value = equity.iloc[-1]
                    
                    # Basic performance metrics
                    performance_metrics = {
                        'Strategy': strategy_name,
                        'Initial Value': init_value,
                        'Final Value': final_value,
                        'Total Return': final_value / init_value - 1,
                    }
                    
                    # Calculate additional metrics if possible
                    days = (results_df.index[-1] - results_df.index[0]).days
                    years = days / 365.25 if days > 0 else 1
                    
                    performance_metrics['CAGR'] = (final_value / init_value) ** (1 / years) - 1
                    
                    # Volatility
                    returns = equity.pct_change().dropna()
                    performance_metrics['Volatility'] = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                    
                    # Max Drawdown
                    peak = equity.cummax()
                    drawdown = (equity / peak - 1)
                    performance_metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
                    
                    # Sharpe Ratio (assuming 2% risk-free)
                    if performance_metrics['Volatility'] > 0:
                        performance_metrics['Sharpe Ratio'] = (performance_metrics['CAGR'] - 0.02) / performance_metrics['Volatility']
                    else:
                        performance_metrics['Sharpe Ratio'] = 0
            except Exception as e:
                print(f"Warning: Error processing TurboQT results: {e}")
                
        elif strategy_name == "DMT_v2":
            # DMT_v2 expects prices DataFrame, not start/end dates
            print(f"Loading data for DMT_v2 backtest from {start_date} to {end_date}...")
            # Fetch data for the specified period
            prices = fetch_data_from_date("QQQ", start_date, end_date)
            
            if prices is None or prices.empty:
                print(f"Error: Could not load data for {strategy_name}")
                return None
                
            # Run DMT_v2 backtest with learning_rate explicitly
            results_df = run_dmt_v2_backtest(
                prices=prices,
                initial_capital=capital,
                learning_rate=kwargs.get('learning_rate', 0.01),
                n_epochs=kwargs.get('epochs', 100),
                seq_len=kwargs.get('seq_len', 10)
            )
            
            # Try to load the saved CSV directly
            try:
                results_file = os.path.join('tri_shot_data', 'dmt_v2_backtest_results.csv')
                if os.path.exists(results_file):
                    results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
                    print(f"Loaded DMT_v2 results from {results_file}")
                    
                    # Create performance metrics from the result data
                    if 'equity' in results_df.columns:
                        equity_curve = results_df['equity']
                    elif 'strategy_equity' in results_df.columns:
                        equity_curve = results_df['strategy_equity']
                    elif 'dmt_v2_equity' in results_df.columns:
                        equity_curve = results_df['dmt_v2_equity']
                    else:
                        # Try to find any equity column
                        for col in results_df.columns:
                            if 'equity' in col.lower() or ('value' in col.lower() and 'dmt' in col.lower()):
                                equity_curve = results_df[col]
                                print(f"Using column '{col}' as equity for DMT_v2")
                                break
                        else:
                            print("Could not find equity column in DMT_v2 results")
                            return None
                            
                    # Create basic metrics
                    init_value = equity_curve.iloc[0]
                    final_value = equity_curve.iloc[-1]
                    days = (equity_curve.index[-1] - equity_curve.index[0]).days
                    years = days / 365.25 if days > 0 else 1
                    
                    performance_metrics = {
                        'Strategy': strategy_name,
                        'Initial Value': init_value,
                        'Final Value': final_value,
                        'Total Return': final_value / init_value - 1,
                        'Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        'Trading Days': len(equity_curve),
                        'CAGR': (final_value / init_value) ** (1 / years) - 1 if years > 0 else 0,
                        'Equity Curve': equity_curve
                    }
                    
                    # Calculate volatility
                    returns = equity_curve.pct_change().dropna()
                    performance_metrics['Volatility'] = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                    
                    # Calculate max drawdown
                    peak = equity_curve.cummax()
                    drawdown = (equity_curve / peak - 1)
                    performance_metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
                    
                    # Calculate Sharpe ratio
                    vol = performance_metrics['Volatility']
                    cagr = performance_metrics['CAGR']
                    performance_metrics['Sharpe Ratio'] = (cagr - 0.02) / vol if vol > 0 else 0
            except Exception as e:
                print(f"Error loading DMT_v2 results: {e}")
                performance_metrics = {'Strategy': strategy_name}
        
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
        return performance_metrics

    except Exception as e:
        print(f"Error running {strategy_name} backtest: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None

def compare_strategies(start_date, end_date, capital):
    """Compare the performance of all strategies."""
    results = []

    # Run Tri-Shot
    tri_shot_res = run_and_collect_results(
        "Tri-Shot", tri_shot_cli.backtest, start_date, end_date, capital
    )
    if tri_shot_res: results.append(tri_shot_res)

    # Run DMT
    dmt_res = run_and_collect_results(
        "DMT", run_dmt_backtest, start_date, end_date, capital,
        epochs=50
    )
    if dmt_res: results.append(dmt_res)

    # Run TurboQT
    turbo_qt_res = run_and_collect_results(
        "TurboQT", TurboBacktester, start_date, end_date, capital
    )
    if turbo_qt_res: results.append(turbo_qt_res)

    # Run DMT v2
    dmt_v2_res = run_and_collect_results(
        "DMT_v2", run_dmt_v2_backtest, start_date, end_date, capital,
        # Use lr=0.01 as we found it worked better
        learning_rate=0.01
    )
    if dmt_v2_res: results.append(dmt_v2_res)

    if not results:
        print("No strategy results available to compare.")
        return

    # --- Display results table ---
    summary_df = pd.DataFrame(results).drop(columns=['Equity Curve'])
    
    # Remove any _Note column and add it to Period if present
    if '_Note' in summary_df.columns:
        for idx, row in summary_df.iterrows():
            if pd.notna(row.get('_Note')):
                summary_df.loc[idx, 'Period'] = f"{row['Period']} {row['_Note']}"
        summary_df = summary_df.drop(columns=['_Note'])
        
    summary_df = summary_df.set_index('Strategy')

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
             formatted_summary[col] = formatted_summary[col].apply(formatter)
        else:
             print(f"Warning: Column '{col}' not found for formatting.")

    # Rename columns for display
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

    for res in results:
        if 'Equity Curve' in res and res['Equity Curve'] is not None:
            equity_curve = res['Equity Curve']
            # Normalize equity curves to start at 1.0 for comparison
            normalized_equity = equity_curve / equity_curve.iloc[0]
            plt.plot(normalized_equity.index, normalized_equity, label=f"{res['Strategy']} (Sharpe: {res['Sharpe Ratio']:.2f})")
        else:
            print(f"Warning: No equity curve data for {res.get('Strategy', 'Unknown Strategy')}.")


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
