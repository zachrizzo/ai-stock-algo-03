#!/usr/bin/env python3
"""
Test and compare original vs enhanced DMT_v2 implementations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time

# Import original DMT_v2 backtest function (preserve as reference)
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

# Set parameters
initial_capital = 500.0
test_period = 30  # Use a shorter period to avoid rate limits
end_date = datetime.now() - timedelta(days=100)  # Use past data to ensure availability
start_date = end_date - timedelta(days=test_period)

# Format dates
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')
lookback_start = start_date - timedelta(days=365)  # Need a year of data for features
lookback_start_str = lookback_start.strftime('%Y-%m-%d')

print(f"Testing period: {start_date_str} to {end_date_str}")
print(f"Fetching data from {lookback_start_str} to include lookback period")

# Add delay to avoid rate limits
time.sleep(2)

# Fetch SPY data for testing
try:
    data = yf.download('SPY', start=lookback_start_str, end=end_date_str, progress=False)
    if len(data) < 30:
        raise ValueError(f"Insufficient data: Got only {len(data)} days")
    print(f"Retrieved {len(data)} days of SPY data")
except Exception as e:
    print(f"Error fetching data: {e}")
    import sys
    sys.exit(1)

# Configure test parameters
original_params = {
    'initial_capital': initial_capital,
    'n_epochs': 30,  # Reduced for quicker testing
    'target_annual_vol': 0.35,
    'max_position_size': 2.0,
    'neutral_zone': 0.03,
    'plot': False,
    'use_ensemble': False,
    'use_dynamic_stops': False,
    'max_drawdown_threshold': 0.0  # Effectively disabled
}

enhanced_params = {
    'initial_capital': initial_capital,
    'n_epochs': 30,  # Reduced for quicker testing
    'target_annual_vol': 0.35,
    'max_position_size': 2.0,
    'neutral_zone': 0.03,
    'plot': False,
    'use_ensemble': True,  # Use ensemble modeling
    'use_dynamic_stops': True,  # Use dynamic stop-losses
    'max_drawdown_threshold': 0.15  # Enable drawdown protection
}

# Add slight delay between tests
time.sleep(3)

# Run original DMT_v2 backtest
print("\nRunning original DMT_v2 backtest...")
try:
    orig_results_df, orig_metrics = run_dmt_v2_backtest(data, **original_params)
    print(f"\nOriginal DMT_v2 Results:")
    print(f"Total Return: {orig_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {orig_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {orig_metrics['max_drawdown']:.2%}")
    print(f"CAGR: {orig_metrics['cagr']:.2%}")
except Exception as e:
    print(f"Error running original backtest: {e}")
    import traceback
    traceback.print_exc()
    orig_results_df = None
    orig_metrics = None

# Add delay to avoid rate limits
time.sleep(3)

# Run enhanced DMT_v2 backtest
print("\nRunning enhanced DMT_v2 backtest...")
try:
    enh_results_df, enh_metrics = run_dmt_v2_backtest(data, **enhanced_params)
    print(f"\nEnhanced DMT_v2 Results:")
    print(f"Total Return: {enh_metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {enh_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {enh_metrics['max_drawdown']:.2%}")
    print(f"CAGR: {enh_metrics['cagr']:.2%}")
except Exception as e:
    print(f"Error running enhanced backtest: {e}")
    import traceback
    traceback.print_exc()
    enh_results_df = None
    enh_metrics = None

# Compare results if both tests ran successfully
if orig_results_df is not None and enh_results_df is not None:
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    plt.plot(orig_results_df.index, orig_results_df['dmt_v2_equity'], label='Original DMT_v2')
    plt.plot(enh_results_df.index, enh_results_df['dmt_v2_equity'], label='Enhanced DMT_v2')
    plt.plot(orig_results_df.index, orig_results_df['buy_hold_equity'], label='Buy & Hold', linestyle=':')
    
    # Add metrics to the plot
    orig_ret = orig_metrics['total_return']
    enh_ret = enh_metrics['total_return']
    orig_sharpe = orig_metrics['sharpe_ratio']
    enh_sharpe = enh_metrics['sharpe_ratio']
    orig_dd = orig_metrics['max_drawdown']
    enh_dd = enh_metrics['max_drawdown']
    
    plt.title(f'DMT_v2 Original vs Enhanced Comparison ({start_date_str} to {end_date_str})')
    plt.annotate(f"Original Return: {orig_ret:.2%}, Sharpe: {orig_sharpe:.2f}, MaxDD: {orig_dd:.2%}", 
                 xy=(0.05, 0.95), xycoords='axes fraction')
    plt.annotate(f"Enhanced Return: {enh_ret:.2%}, Sharpe: {enh_sharpe:.2f}, MaxDD: {enh_dd:.2%}", 
                 xy=(0.05, 0.90), xycoords='axes fraction')
    
    # Calculate and display improvement metrics
    ret_improvement = (enh_ret - orig_ret) / abs(orig_ret) if orig_ret != 0 else float('inf')
    sharpe_improvement = (enh_sharpe - orig_sharpe) / abs(orig_sharpe) if orig_sharpe != 0 else float('inf')
    dd_improvement = (orig_dd - enh_dd) / abs(orig_dd) if orig_dd != 0 else float('inf')
    
    improvement_msg = f"Improvements: Return: {ret_improvement:.2%}, Sharpe: {sharpe_improvement:.2%}, DrawDown: {dd_improvement:.2%}"
    plt.annotate(improvement_msg, xy=(0.05, 0.85), xycoords='axes fraction')
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    
    # Save comparison plot
    output_dir = os.path.join('tri_shot_data')
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'dmt_v2_comparison.png')
    plt.savefig(comparison_file)
    
    print(f"\nComparison Results:")
    print(f"Return: {orig_ret:.2%} → {enh_ret:.2%} ({ret_improvement:+.2%})")
    print(f"Sharpe: {orig_sharpe:.2f} → {enh_sharpe:.2f} ({sharpe_improvement:+.2%})")
    print(f"Max Drawdown: {orig_dd:.2%} → {enh_dd:.2%} ({dd_improvement:+.2%})")
    print(f"Comparison chart saved to {comparison_file}")
else:
    print("\nUnable to complete comparison due to errors in one or both backtests")
