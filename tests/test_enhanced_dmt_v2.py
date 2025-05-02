#!/usr/bin/env python3
"""
Test script for enhanced DMT_v2 strategy using cached data to avoid Yahoo Finance rate limits
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import DMT_v2 strategy components
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_model import Config, PredictionModel, EnsembleModel
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

def main():
    """Run test of enhanced DMT_v2 strategy"""
    print("Testing Enhanced DMT_v2 Strategy with Optimized Parameters")
    print("=" * 50)
    
    # Try to use already cached data from previous successful run
    try:
        # Look for existing backtest results to extract price data
        print("Searching for cached price data...")
        
        # Option 1: Use data from previous backtest results
        tri_shot_path = os.path.join('tri_shot_data')
        if not os.path.exists(tri_shot_path):
            os.makedirs(tri_shot_path, exist_ok=True)
            
        # Try to find any existing OHLCV data we can use
        data = None
        
        # Check if we can create synthetic test data from previous backtest
        backtest_file = os.path.join(tri_shot_path, 'dmt_v2_backtest_results.csv')
        if os.path.exists(backtest_file):
            print(f"Found previous backtest results: {backtest_file}")
            results = pd.read_csv(backtest_file, index_col=0, parse_dates=True)
            
            if 'buy_hold_equity' in results.columns:
                # Recreate price data from buy and hold equity
                initial_capital = 500.0  # Default initial capital
                
                # Create synthetic price data
                start_date = results.index[0]
                end_date = results.index[-1]
                
                print(f"Creating synthetic OHLCV data from {start_date} to {end_date}")
                
                # Calculate returns
                bh_returns = results['buy_hold_equity'].pct_change().fillna(0)
                
                # Create synthetic OHLCV data
                dates = results.index
                closes = [100.0]  # Start price at 100
                
                for ret in bh_returns:
                    closes.append(closes[-1] * (1 + ret))
                    
                # Create OHLCV DataFrame
                synthetic_data = pd.DataFrame(index=dates)
                synthetic_data['Close'] = closes[1:]  # Skip the first dummy value
                
                # Generate synthetic Open, High, Low values
                noise = np.random.normal(0, 0.005, len(synthetic_data))
                synthetic_data['Open'] = synthetic_data['Close'] * (1 + noise)
                
                noise_high = np.abs(np.random.normal(0, 0.01, len(synthetic_data)))
                synthetic_data['High'] = synthetic_data['Close'] * (1 + noise_high)
                
                noise_low = np.abs(np.random.normal(0, 0.01, len(synthetic_data)))
                synthetic_data['Low'] = synthetic_data['Close'] * (1 - noise_low)
                
                # Generate synthetic volume
                synthetic_data['Volume'] = np.random.randint(1000000, 10000000, len(synthetic_data))
                
                data = synthetic_data
                
                print(f"Created synthetic OHLCV data with {len(data)} days")
        
        if data is None:
            print("No cached data found. Creating simple test data...")
            
            # Create a simple test dataset with synthetic data
            days = 400
            dates = pd.date_range(start='2023-01-01', periods=days)
            
            # Create a pattern that's somewhat predictable but with noise
            t = np.linspace(0, 8*np.pi, days)
            trend = 100 + 20 * np.sin(t/10) + t/5  # Underlying trend
            noise = np.random.normal(0, 1, days)
            close = trend + noise
            
            # Add some realistic volatility clusters
            volatility = 1 + 0.5 * np.sin(t/5)**2
            close = np.cumsum(np.random.normal(0.0005, 0.01 * volatility, days))
            close = 100 * np.exp(close)  # Start at 100
            
            # Create OHLCV data
            data = pd.DataFrame(index=dates)
            data['Close'] = close
            
            # Generate synthetic Open, High, Low values
            noise = np.random.normal(0, 0.005, days)
            data['Open'] = data['Close'] * (1 + noise)
            
            noise_high = np.abs(np.random.normal(0, 0.01, days))
            data['High'] = data['Close'] * (1 + noise_high)
            
            noise_low = np.abs(np.random.normal(0, 0.01, days))
            data['Low'] = data['Close'] * (1 - noise_low)
            
            # Generate synthetic volume
            data['Volume'] = np.random.randint(1000000, 10000000, days)
            
            print(f"Created synthetic test data with {len(data)} days")
        
        # Run backtest with our enhanced models
        print("\nRunning backtest with enhanced DMT_v2 model...")
        start_time = datetime.now()
        
        # Default parameters that worked well
        params = {
            'initial_capital': 500.0,
            'n_epochs': 50,  # Reduced for quicker testing
            'target_annual_vol': 0.35,
            'max_position_size': 2.0,
            'neutral_zone': 0.025,  # Reduced from 0.03 to 0.025
            'learning_rate': 0.015,
            'plot': True,
            'use_ensemble': True,
            'use_dynamic_stops': True,
            'max_drawdown_threshold': 0.2,  # Increased from 0.15 to 0.2
        }
        
        # Run backtest with enhanced configuration
        results_df, metrics = run_dmt_v2_backtest(data, **params)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        # Print results
        print("\n" + "=" * 50)
        print("Enhanced DMT_v2 Backtest Results")
        print("=" * 50)
        print(f"Time elapsed: {elapsed.total_seconds():.1f} seconds")
        print(f"Initial Capital: ${params['initial_capital']:.2f}")
        print(f"Final Value: ${metrics['final_equity']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"CAGR: {metrics['cagr']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit/Loss Ratio: {metrics['avg_profit_loss_ratio']:.2f}")
        
        if 'regime_probs' in metrics:
            print("\nAverage Regime Probabilities:")
            for i, prob in enumerate(metrics['regime_probs']):
                regime_name = ['Bull', 'Neutral', 'Bear'][i]
                print(f"{regime_name} Market: {prob:.2%}")
        
        print("\nResults saved to:")
        print(f"- tri_shot_data/dmt_v2_backtest_results.csv")
        print(f"- tri_shot_data/dmt_v2_backtest.png")
        
        return results_df, metrics
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
