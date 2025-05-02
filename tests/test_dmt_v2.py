#!/usr/bin/env python3
"""
Test script for validating the DMT_v2 strategy performance.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import yfinance as yf
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

def fetch_data(ticker, start_date, end_date, lookback_days=252):
    """Fetch historical price data with lookback period."""
    from datetime import datetime, timedelta
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Add lookback period to start date
    lookback_start = start_date - timedelta(days=lookback_days)
    
    # Fetch data
    ticker_data = yf.download(ticker, start=lookback_start.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'))
    
    print(f"Downloaded {len(ticker_data)} days of {ticker} data")
    return ticker_data

def test_asset(ticker='SPY', start_date='2024-01-01', end_date='2025-04-26'):
    """Test DMT_v2 on a specific asset and time period."""
    print(f"Testing DMT_v2 on {ticker} from {start_date} to {end_date}")
    
    # Create output directory
    os.makedirs("validation_tests", exist_ok=True)
    
    try:
        # Fetch data
        raw_prices = fetch_data(ticker, start_date, end_date)
        
        # Print dataframe structure to debug
        print("Raw dataframe columns:", raw_prices.columns)
        print("Raw dataframe shape:", raw_prices.shape)
        
        # Check if we have a MultiIndex
        if isinstance(raw_prices.columns, pd.MultiIndex):
            print("Detected MultiIndex columns - flattening dataframe structure")
            # We need to flatten this to a regular DataFrame
            # Get just the price data for the specified ticker
            prices = pd.DataFrame()
            prices['Open'] = raw_prices[('Open', ticker)]
            prices['High'] = raw_prices[('High', ticker)]
            prices['Low'] = raw_prices[('Low', ticker)]
            prices['Close'] = raw_prices[('Close', ticker)]
            prices['Volume'] = raw_prices[('Volume', ticker)]
            
            # Keep the same index
            prices.index = raw_prices.index
            
            print("Flattened dataframe sample:")
            print(prices.head(3))
        else:
            prices = raw_prices
        
        # The DMT_v2 model expects a specific format - use yfinance format as is
        # but ensure we have the correct columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        actual_cols = [col for col in required_cols if col in prices.columns]
        
        if len(actual_cols) < len(required_cols):
            missing = set(required_cols) - set(actual_cols)
            print(f"Warning: Missing required columns: {missing}")
            # Try to use available columns
            if 'Adj Close' in prices.columns and 'Close' not in prices.columns:
                prices['Close'] = prices['Adj Close']
                print("Using 'Adj Close' as 'Close'")
        
        # Ensure we have a price column to work with
        if 'Close' not in prices.columns and 'Adj Close' in prices.columns:
            prices['Close'] = prices['Adj Close']
        
        # Run backtest with enhanced parameters
        df, metrics = run_dmt_v2_backtest(
            prices=prices,  # Pass raw dataframe directly
            initial_capital=10000,
            n_epochs=100,
            learning_rate=0.01,
            neutral_zone=0.03,
            target_annual_vol=0.35,
            max_position_size=2.0,
            plot=False
        )
        
        # Display results
        print("\n" + "="*80)
        print(f"{'DMT_v2 TEST RESULTS':^80}")
        print("="*80)
        print(f"Asset: {ticker}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total Return: {metrics['Total Return']*100:.2f}%")
        print(f"CAGR: {metrics['CAGR']*100:.2f}%")
        print(f"Volatility: {metrics['Volatility']*100:.2f}%")
        print(f"Max Drawdown: {metrics['Max Drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print("-"*80)
        
        # Save results
        df.to_csv(f"validation_tests/dmt_v2_test_{ticker}_{start_date}_to_{end_date}.csv")
        
        # Create equity curve chart
        plt.figure(figsize=(12, 8))
        plt.plot(df.index, df['dmt_v2_equity'], label='DMT v2 Strategy')
        if 'buy_hold_equity' in df.columns:
            plt.plot(df.index, df['buy_hold_equity'], label='Buy & Hold')
        plt.title(f'DMT_v2 Performance on {ticker}: {start_date} to {end_date}')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"validation_tests/dmt_v2_test_{ticker}_{start_date}_to_{end_date}.png")
        
        return df, metrics
        
    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("=== DMT_v2 Strategy Validation Testing ===\n")
    
    # Test with default parameters
    test_asset()
    
    print("\nValidation test completed!")
    print("Results saved to the 'validation_tests' directory.")
