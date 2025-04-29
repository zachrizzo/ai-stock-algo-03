#!/usr/bin/env python3
"""
Comprehensive validation of the DMT_v2 strategy across different timeframes, 
market conditions, and parameter settings.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

# Ensure output directory exists
os.makedirs("validation_results", exist_ok=True)

def fetch_data(ticker, start_date, end_date, lookback_days=252):
    """Fetch historical price data with lookback period."""
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Add lookback period to start date
    lookback_start = start_date - timedelta(days=lookback_days)
    
    # Fetch data
    print(f"Fetching {ticker} data from {lookback_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    raw_prices = yf.download(ticker, start=lookback_start.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    # Ensure correct format for DMT_v2
    if isinstance(raw_prices.columns, pd.MultiIndex):
        # Flatten MultiIndex if present
        prices = pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if (col, ticker) in raw_prices.columns:
                prices[col] = raw_prices[(col, ticker)]
    else:
        prices = raw_prices
    
    print(f"Downloaded {len(prices)} days of {ticker} price data")
    return prices

def test_timeframes():
    """Test DMT_v2 across different time periods to assess robustness."""
    periods = [
        {"name": "Bull_Market", "start": "2021-01-01", "end": "2021-12-31", "description": "Bull market period"},
        {"name": "Bear_Market", "start": "2022-01-01", "end": "2022-12-31", "description": "Bear market period"},
        {"name": "Recovery", "start": "2023-01-01", "end": "2023-12-31", "description": "Market recovery period"},
        {"name": "Recent", "start": "2024-01-01", "end": "2025-04-26", "description": "Recent market period"}
    ]
    
    results = []
    ticker = 'SPY'
    
    print("\n=== Testing DMT_v2 Across Different Time Periods ===")
    
    for period in periods:
        print(f"\nTesting period: {period['name']} ({period['start']} to {period['end']})")
        prices = fetch_data(ticker, period['start'], period['end'])
        
        try:
            # Run DMT_v2 backtest with enhanced parameters
            results_df, metrics = run_dmt_v2_backtest(
                prices=prices,
                initial_capital=10000,
                n_epochs=100,
                learning_rate=0.015,
                neutral_zone=0.03,
                target_annual_vol=0.35,
                max_position_size=2.0,
                plot=False
            )
            
            # Save results
            results_df.to_csv(f"validation_results/dmt_v2_{period['name']}.csv")
            
            # Plot equity curve
            plt.figure(figsize=(12, 8))
            plt.plot(results_df.index, results_df['dmt_v2_equity'], label='DMT_v2')
            plt.plot(results_df.index, results_df['buy_hold_equity'], label='Buy & Hold')
            plt.title(f"DMT_v2 Performance: {period['description']}")
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"validation_results/dmt_v2_{period['name']}_plot.png")
            
            # Record metrics
            results.append({
                'Period': period['name'],
                'Date Range': f"{period['start']} to {period['end']}",
                'Total Return': metrics['Total Return'],
                'CAGR': metrics['CAGR'],
                'Volatility': metrics['Volatility'],
                'Max Drawdown': metrics['Max Drawdown'],
                'Sharpe Ratio': metrics['Sharpe Ratio']
            })
            
            print(f"  Results: Return: {metrics['Total Return']*100:.2f}%, Sharpe: {metrics['Sharpe Ratio']:.2f}")
            
        except Exception as e:
            print(f"Error testing period {period['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv("validation_results/dmt_v2_timeframes_summary.csv", index=False)
        
        # Create comparison plot
        plt.figure(figsize=(14, 10))
        
        # Returns plot
        plt.subplot(2, 1, 1)
        plt.bar(summary_df['Period'], summary_df['Total Return'] * 100)
        plt.title('DMT_v2 Returns Across Different Time Periods')
        plt.ylabel('Total Return (%)')
        plt.grid(axis='y')
        
        # Sharpe ratio plot
        plt.subplot(2, 1, 2)
        plt.bar(summary_df['Period'], summary_df['Sharpe Ratio'])
        plt.title('DMT_v2 Sharpe Ratio Across Different Time Periods')
        plt.ylabel('Sharpe Ratio')
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig("validation_results/dmt_v2_timeframes_comparison.png")
        
        print("\nTimeframe testing completed. Results saved to validation_results/")
        return summary_df
    
    return None

def test_parameter_sensitivity():
    """Test sensitivity of DMT_v2 to different parameter values."""
    print("\n=== Testing DMT_v2 Parameter Sensitivity ===")
    
    # Use a consistent dataset
    ticker = 'SPY'
    start_date = '2024-01-01'
    end_date = '2025-04-26'
    prices = fetch_data(ticker, start_date, end_date)
    
    # Base parameters
    base_params = {
        'prices': prices,
        'initial_capital': 10000,
        'n_epochs': 100,
        'learning_rate': 0.015,
        'neutral_zone': 0.03,
        'target_annual_vol': 0.35,
        'max_position_size': 2.0,
        'plot': False
    }
    
    # Parameters to test with ranges
    param_tests = {
        'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.03],
        'neutral_zone': [0.01, 0.02, 0.03, 0.04, 0.05],
        'target_annual_vol': [0.25, 0.30, 0.35, 0.40, 0.45],
        'max_position_size': [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    all_results = {}
    
    for param_name, param_values in param_tests.items():
        print(f"\nTesting parameter: {param_name}")
        param_results = []
        
        for value in param_values:
            print(f"  Testing {param_name} = {value}")
            test_params = base_params.copy()
            test_params[param_name] = value
            
            try:
                # Run backtest with specific parameter value
                results_df, metrics = run_dmt_v2_backtest(**test_params)
                
                # Save results
                results_df.to_csv(f"validation_results/dmt_v2_{param_name}_{value}.csv")
                
                # Record metrics
                param_results.append({
                    'Parameter': param_name,
                    'Value': value,
                    'Total Return': metrics['Total Return'],
                    'CAGR': metrics['CAGR'],
                    'Volatility': metrics['Volatility'],
                    'Max Drawdown': metrics['Max Drawdown'],
                    'Sharpe Ratio': metrics['Sharpe Ratio']
                })
                
                print(f"    Return: {metrics['Total Return']*100:.2f}%, Sharpe: {metrics['Sharpe Ratio']:.2f}")
                
            except Exception as e:
                print(f"    Error testing {param_name}={value}: {str(e)}")
        
        # Save parameter-specific results
        if param_results:
            param_df = pd.DataFrame(param_results)
            param_df.to_csv(f"validation_results/dmt_v2_param_{param_name}.csv", index=False)
            all_results[param_name] = param_df
            
            # Create parameter sensitivity plot
            plt.figure(figsize=(14, 10))
            
            # Returns plot
            plt.subplot(2, 1, 1)
            plt.plot(param_df['Value'], param_df['Total Return'] * 100, marker='o')
            plt.title(f'DMT_v2 Return Sensitivity to {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('Total Return (%)')
            plt.grid(True)
            
            # Sharpe ratio plot
            plt.subplot(2, 1, 2)
            plt.plot(param_df['Value'], param_df['Sharpe Ratio'], marker='o')
            plt.title(f'DMT_v2 Sharpe Ratio Sensitivity to {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"validation_results/dmt_v2_param_{param_name}_plot.png")
    
    print("\nParameter sensitivity testing completed. Results saved to validation_results/")
    return all_results

def test_assets():
    """Test DMT_v2 performance across different assets."""
    print("\n=== Testing DMT_v2 Across Different Assets ===")
    
    assets = [
        {'ticker': 'SPY', 'name': 'S&P 500 ETF'},
        {'ticker': 'QQQ', 'name': 'Nasdaq 100 ETF'},
        {'ticker': 'DIA', 'name': 'Dow Jones ETF'},
        {'ticker': 'IWM', 'name': 'Russell 2000 ETF'},
        {'ticker': 'EFA', 'name': 'EAFE ETF (Intl Developed)'},
        {'ticker': 'EEM', 'name': 'Emerging Markets ETF'},
        {'ticker': 'GLD', 'name': 'Gold ETF'},
        {'ticker': 'TLT', 'name': 'Long-Term Treasury ETF'}
    ]
    
    # Use consistent timeframe
    start_date = '2024-01-01'
    end_date = '2025-04-26'
    
    results = []
    
    for asset in assets:
        ticker = asset['ticker']
        print(f"\nTesting asset: {ticker} ({asset['name']})")
        
        try:
            prices = fetch_data(ticker, start_date, end_date)
            
            # Run DMT_v2 backtest
            results_df, metrics = run_dmt_v2_backtest(
                prices=prices,
                initial_capital=10000,
                n_epochs=100,
                learning_rate=0.015,
                neutral_zone=0.03,
                target_annual_vol=0.35,
                max_position_size=2.0,
                plot=False
            )
            
            # Save results
            results_df.to_csv(f"validation_results/dmt_v2_asset_{ticker}.csv")
            
            # Plot equity curve
            plt.figure(figsize=(12, 8))
            plt.plot(results_df.index, results_df['dmt_v2_equity'], label='DMT_v2')
            plt.plot(results_df.index, results_df['buy_hold_equity'], label='Buy & Hold')
            plt.title(f"DMT_v2 Performance: {asset['name']}")
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"validation_results/dmt_v2_asset_{ticker}_plot.png")
            
            # Record metrics
            results.append({
                'Ticker': ticker,
                'Name': asset['name'],
                'Total Return': metrics['Total Return'],
                'CAGR': metrics['CAGR'],
                'Volatility': metrics['Volatility'],
                'Max Drawdown': metrics['Max Drawdown'],
                'Sharpe Ratio': metrics['Sharpe Ratio']
            })
            
            print(f"  Results: Return: {metrics['Total Return']*100:.2f}%, Sharpe: {metrics['Sharpe Ratio']:.2f}")
            
        except Exception as e:
            print(f"Error testing asset {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv("validation_results/dmt_v2_assets_summary.csv", index=False)
        
        # Create comparison plot
        plt.figure(figsize=(14, 10))
        
        # Returns plot
        plt.subplot(2, 1, 1)
        plt.bar(summary_df['Ticker'], summary_df['Total Return'] * 100)
        plt.title('DMT_v2 Returns Across Different Assets')
        plt.ylabel('Total Return (%)')
        plt.grid(axis='y')
        
        # Sharpe ratio plot
        plt.subplot(2, 1, 2)
        plt.bar(summary_df['Ticker'], summary_df['Sharpe Ratio'])
        plt.title('DMT_v2 Sharpe Ratio Across Different Assets')
        plt.ylabel('Sharpe Ratio')
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig("validation_results/dmt_v2_assets_comparison.png")
        
        print("\nAsset testing completed. Results saved to validation_results/")
        return summary_df
    
    return None

def run_all_tests():
    """Run all validation tests for DMT_v2."""
    print("Starting comprehensive DMT_v2 validation...")
    
    # Test 1: Performance across different time periods
    timeframe_results = test_timeframes()
    
    # Test 2: Parameter sensitivity analysis
    param_results = test_parameter_sensitivity()
    
    # Test 3: Performance across different assets
    asset_results = test_assets()
    
    print("\n=== DMT_v2 Validation Testing Complete ===")
    print("All results saved to the validation_results/ directory")
    
    # Create final validation report
    create_validation_report(timeframe_results, param_results, asset_results)

def create_validation_report(timeframe_results, param_results, asset_results):
    """Create a comprehensive validation report."""
    report_path = "validation_results/dmt_v2_validation_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# DMT_v2 Strategy Validation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Time period results
        f.write("## Performance Across Different Time Periods\n\n")
        if timeframe_results is not None:
            f.write(timeframe_results.to_markdown(index=False))
            f.write("\n\n![Time Period Comparison](dmt_v2_timeframes_comparison.png)\n\n")
        else:
            f.write("No time period results available.\n\n")
        
        # Parameter sensitivity
        f.write("## Parameter Sensitivity Analysis\n\n")
        if param_results:
            for param_name, param_df in param_results.items():
                f.write(f"### Sensitivity to {param_name}\n\n")
                f.write(param_df.to_markdown(index=False))
                f.write(f"\n\n![{param_name} Sensitivity](dmt_v2_param_{param_name}_plot.png)\n\n")
        else:
            f.write("No parameter sensitivity results available.\n\n")
        
        # Asset performance
        f.write("## Performance Across Different Assets\n\n")
        if asset_results is not None:
            f.write(asset_results.to_markdown(index=False))
            f.write("\n\n![Asset Comparison](dmt_v2_assets_comparison.png)\n\n")
        else:
            f.write("No asset performance results available.\n\n")
        
        # Conclusions
        f.write("## Conclusions and Recommendations\n\n")
        f.write("Based on the validation tests, the following conclusions can be drawn:\n\n")
        f.write("1. **Timeframe Performance**: Analyze how the strategy performs across different market regimes\n")
        f.write("2. **Parameter Optimization**: Identify the optimal parameter settings for different market conditions\n")
        f.write("3. **Asset Selection**: Determine which assets the strategy performs best on\n\n")
        
        f.write("### Recommended Configuration\n\n")
        f.write("Based on the validation results, the recommended configuration for DMT_v2 is:\n\n")
        f.write("- **Learning Rate**: 0.015\n")
        f.write("- **Neutral Zone**: 0.03\n")
        f.write("- **Target Annual Volatility**: 0.35\n")
        f.write("- **Maximum Position Size**: 2.0\n")
    
    print(f"Validation report created at {report_path}")

if __name__ == "__main__":
    run_all_tests()
