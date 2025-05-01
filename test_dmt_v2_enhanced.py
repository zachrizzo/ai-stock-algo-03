#!/usr/bin/env python3
"""
Test script for enhanced DMT_v2 model implementation
"""

from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest
import pandas as pd
import yfinance as yf

print('Fetching SPY data for testing enhanced DMT_v2 model...')
data = yf.download('SPY', start='2024-01-01', end='2024-03-01', progress=False)
print(f'Retrieved {len(data)} days of data')

if len(data) > 0:
    print('Running backtest with enhanced DMT_v2 model...')
    results_df, metrics = run_dmt_v2_backtest(
        data, 
        initial_capital=500.0, 
        n_epochs=40, 
        plot=True, 
        use_ensemble=True, 
        max_drawdown_threshold=0.15, 
        use_dynamic_stops=True
    )
    
    print("\n=== Enhanced DMT_v2 Performance ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Results saved to tri_shot_data/dmt_v2_backtest_results.csv")
else:
    print("Error: Failed to download data")
