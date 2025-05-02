#!/usr/bin/env python3
"""
Test the robustness of the DMT_v2 strategy across different time periods.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

def test_time_periods():
    """Test DMT_v2 over different time periods to assess robustness."""
    periods = [
        # Bull market
        {"name": "Bull Market", "start": "2021-01-01", "end": "2021-12-31"},
        # Bear market 
        {"name": "Bear Market", "start": "2022-01-01", "end": "2022-12-31"},
        # Recovery
        {"name": "Recovery", "start": "2023-01-01", "end": "2023-12-31"},
        # Recent market
        {"name": "Recent Market", "start": "2024-01-01", "end": "2025-04-26"},
        # Full period
        {"name": "Full Period", "start": "2020-01-01", "end": "2025-04-26"}
    ]
    
    results = []
    
    # Create output directory
    os.makedirs("validation_tests", exist_ok=True)
    
    # Test across periods
    for period in periods:
        print(f"\nTesting {period['name']} ({period['start']} to {period['end']})...")
        
        try:
            # Run backtest
            df, metrics = run_dmt_v2_backtest(
                ticker_symbol='SPY',
                initial_capital=10000,
                start_date=period['start'],
                end_date=period['end'],
                plot=False
            )
            
            # Store result summary
            results.append({
                "Period": period['name'], 
                "Date Range": f"{period['start']} to {period['end']}",
                "Total Return": metrics['Total Return'],
                "CAGR": metrics['CAGR'],
                "Volatility": metrics['Volatility'],
                "Max Drawdown": metrics['Max Drawdown'],
                "Sharpe Ratio": metrics['Sharpe Ratio']
            })
            
            # Save individual period result
            df.to_csv(f"validation_tests/dmt_v2_{period['name'].replace(' ', '_').lower()}.csv")
            
        except Exception as e:
            print(f"Error testing period {period['name']}: {e}")
    
    # Create summary table
    if results:
        summary_df = pd.DataFrame(results)
        
        # Format for display
        summary_df['Total Return'] = summary_df['Total Return'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['CAGR'] = summary_df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Volatility'] = summary_df['Volatility'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Max Drawdown'] = summary_df['Max Drawdown'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Sharpe Ratio'] = summary_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        
        print("\n" + "="*80)
        print(f"{'DMT_v2 ROBUSTNESS TEST RESULTS':^80}")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("-"*80)
        
        # Save summary
        summary_df.to_csv("validation_tests/dmt_v2_robustness_summary.csv", index=False)
        
        return summary_df
    
    return None

def test_parameter_sensitivity():
    """Test the sensitivity of DMT_v2 to different parameter values."""
    base_params = {
        'ticker_symbol': 'SPY',
        'initial_capital': 10000,
        'start_date': '2024-01-01',
        'end_date': '2025-04-26',
        'plot': False
    }
    
    # Parameters to test
    parameter_variations = {
        'target_annual_vol': [0.25, 0.30, 0.35, 0.40, 0.45],
        'max_position_size': [1.0, 1.5, 2.0, 2.5, 3.0],
        'neutral_zone': [0.02, 0.03, 0.04, 0.05, 0.06],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05]
    }
    
    # Test each parameter variation
    for param_name, param_values in parameter_variations.items():
        results = []
        print(f"\nTesting sensitivity to {param_name}...")
        
        for value in param_values:
            print(f"  Testing {param_name}={value}...")
            test_params = base_params.copy()
            test_params[param_name] = value
            
            try:
                # Run backtest with the specific parameter variation
                df, metrics = run_dmt_v2_backtest(**test_params)
                
                # Store result summary
                results.append({
                    "Parameter": param_name,
                    "Value": value,
                    "Total Return": metrics['Total Return'],
                    "CAGR": metrics['CAGR'],
                    "Volatility": metrics['Volatility'],
                    "Max Drawdown": metrics['Max Drawdown'],
                    "Sharpe Ratio": metrics['Sharpe Ratio']
                })
                
            except Exception as e:
                print(f"Error testing {param_name}={value}: {e}")
        
        # Create summary table for this parameter
        if results:
            summary_df = pd.DataFrame(results)
            
            # Save results
            summary_df.to_csv(f"validation_tests/dmt_v2_param_{param_name}.csv", index=False)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(summary_df['Value'], summary_df['Total Return'], marker='o', label='Total Return')
            plt.title(f'DMT_v2 Sensitivity to {param_name} - Returns')
            plt.xlabel(param_name)
            plt.ylabel('Total Return')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(summary_df['Value'], summary_df['Sharpe Ratio'], marker='o', label='Sharpe Ratio')
            plt.title(f'DMT_v2 Sensitivity to {param_name} - Sharpe Ratio')
            plt.xlabel(param_name)
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"validation_tests/dmt_v2_param_{param_name}.png")
    
    return "Parameter sensitivity tests completed"

def test_different_assets():
    """Test DMT_v2 on different assets to assess generalization."""
    assets = ['SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'GLD', 'TLT']
    
    results = []
    print("\nTesting DMT_v2 across different assets...")
    
    for ticker in assets:
        print(f"  Testing {ticker}...")
        try:
            # Run backtest
            df, metrics = run_dmt_v2_backtest(
                ticker_symbol=ticker,
                initial_capital=10000,
                start_date='2024-01-01',
                end_date='2025-04-26',
                plot=False
            )
            
            # Store result summary
            results.append({
                "Asset": ticker,
                "Total Return": metrics['Total Return'],
                "CAGR": metrics['CAGR'],
                "Volatility": metrics['Volatility'],
                "Max Drawdown": metrics['Max Drawdown'],
                "Sharpe Ratio": metrics['Sharpe Ratio']
            })
            
            # Save individual asset result
            df.to_csv(f"validation_tests/dmt_v2_asset_{ticker}.csv")
            
        except Exception as e:
            print(f"Error testing asset {ticker}: {e}")
    
    # Create summary table
    if results:
        summary_df = pd.DataFrame(results)
        
        # Format for display
        summary_df['Total Return'] = summary_df['Total Return'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['CAGR'] = summary_df['CAGR'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Volatility'] = summary_df['Volatility'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Max Drawdown'] = summary_df['Max Drawdown'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['Sharpe Ratio'] = summary_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        
        print("\n" + "="*80)
        print(f"{'DMT_v2 ASSET TEST RESULTS':^80}")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("-"*80)
        
        # Save summary
        summary_df.to_csv("validation_tests/dmt_v2_assets_summary.csv", index=False)
        
        # Create bar chart of returns by asset
        plt.figure(figsize=(12, 8))
        assets = summary_df['Asset']
        returns = summary_df['Total Return'].str.rstrip('%').astype(float)
        sharpes = summary_df['Sharpe Ratio'].astype(float)
        
        x = np.arange(len(assets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, returns, width, label='Return (%)')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, sharpes, width, color='orange', label='Sharpe Ratio')
        
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.set_xlabel('Asset')
        ax.set_ylabel('Return (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax.set_title('DMT_v2 Performance Across Different Assets')
        
        # Add two legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("validation_tests/dmt_v2_assets_comparison.png")
        
        return summary_df
    
    return None

if __name__ == "__main__":
    print("=== DMT_v2 Strategy Validation Testing ===\n")
    
    print("1. Testing robustness across different time periods...")
    time_results = test_time_periods()
    
    print("\n2. Testing parameter sensitivity...")
    param_results = test_parameter_sensitivity()
    
    print("\n3. Testing performance across different assets...")
    asset_results = test_different_assets()
    
    print("\nAll validation tests completed!")
    print(f"Results saved to the 'validation_tests' directory.")
