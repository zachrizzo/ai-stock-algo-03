#!/usr/bin/env python3
"""
Compare the performance of all three trading strategies.
This script loads the backtest results and displays them in a standardized format.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def format_percent(value):
    """Format a value as a percentage."""
    return f"{value*100:.2f}%"

def format_dollar(value):
    """Format a value as dollars."""
    return f"${value:.2f}"

def load_tri_shot_results():
    """Load Tri-Shot backtest results."""
    results_file = os.path.join('tri_shot_data', 'backtest_results.csv')
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run tri_shot backtest first.")
        return None
        
    df = pd.read_csv(results_file, index_col=0, parse_dates=True)
    
    # Extract performance metrics
    if 'strategy_equity' in df.columns:
        equity_curve = df['strategy_equity']
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        total_return = final_value / initial_value - 1
        
        # Calculate CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        cagr = (final_value / initial_value) ** (1 / years) - 1
        
        # Use max drawdown from the file if available
        if 'strategy_drawdown' in df.columns:
            max_drawdown = df['strategy_drawdown'].min()
        else:
            # Calculate max drawdown
            peak = equity_curve.cummax()
            drawdown = (equity_curve / peak - 1)
            max_drawdown = drawdown.min()
        
        # Calculate volatility from returns
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0
        
        return {
            "Strategy": "Tri-Shot",
            "Period": f"{equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}",
            "Trading Days": len(equity_curve),
            "Initial Value": initial_value,
            "Final Value": final_value,
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe,
            "Equity Curve": equity_curve
        }
    return None

def load_dmt_results():
    """Load DMT backtest results."""
    results_file = os.path.join('tri_shot_data', 'dmt_backtest_results.csv')
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run dmt backtest first.")
        return None
        
    try:
        df = pd.read_csv(results_file, index_col=0, parse_dates=True)
        
        # Extract performance metrics
        equity_curve = None
        
        # Prioritize 'strategy_equity', then 'equity'
        if 'strategy_equity' in df.columns:
            equity_curve = df['strategy_equity']
        elif 'equity' in df.columns:
            equity_curve = df['equity']
        else:
            # Try other column names as a fallback
            for col in df.columns:
                if 'equity' in col.lower() or 'value' in col.lower():
                    equity_curve = df[col]
                    break
            else:
                # If no equity column found, try the first numeric column
                for col in df.columns:
                    if np.issubdtype(df[col].dtype, np.number):
                        equity_curve = df[col]
                        print(f"Warning: Using column '{col}' as equity for DMT.")
                        break
                else:
                    print("Error: Could not find equity curve column in DMT results.")
                    return None

        initial_value = equity_curve.iloc[0] # Always use the first value
        final_value = equity_curve.iloc[-1]
        total_return = final_value / initial_value - 1
        
        # Calculate CAGR
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        cagr = (final_value / initial_value) ** (1 / years) - 1
        
        # Calculate max drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak - 1)
        max_drawdown = drawdown.min()
        
        # Calculate volatility from returns
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0
        
        return {
            "Strategy": "DMT",
            "Period": f"{equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}",
            "Trading Days": len(equity_curve),
            "Initial Value": initial_value,
            "Final Value": final_value,
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe,
            "Equity Curve": equity_curve
        }
    except Exception as e:
        print(f"Error loading or processing DMT results: {e}")
        return None

def extract_turbo_qt_results(results_text):
    """Extract TurboQT results from terminal output."""
    # This function is no longer needed as we read from CSV
    pass

def load_turbo_qt_results():
    """Load TurboQT backtest results from CSV."""
    results_file = os.path.join('tri_shot_data', 'turbo_qt_backtest_results.csv')
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run turbo_qt backtest first.")
        return None
        
    try:
        df = pd.read_csv(results_file, index_col=0, parse_dates=True)
        
        # Extract performance metrics (similar to load_tri_shot_results)
        if 'strategy_equity' in df.columns:
            equity_curve = df['strategy_equity']
            initial_value = equity_curve.iloc[0]
            final_value = equity_curve.iloc[-1]
            total_return = final_value / initial_value - 1
            
            # Calculate CAGR
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = days / 365.25 if days > 0 else 1 # Avoid division by zero
            cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else total_return
            
            # Calculate max drawdown
            peak = equity_curve.cummax()
            drawdown = (equity_curve / peak - 1)
            max_drawdown = drawdown.min()
            
            # Calculate volatility from returns
            returns = equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Calculate Sharpe ratio (assuming 2% risk-free rate)
            sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0
            
            return {
                "Strategy": "TurboQT",
                "Period": f"{equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')}",
                "Trading Days": len(equity_curve),
                "Initial Value": initial_value,
                "Final Value": final_value,
                "Total Return": total_return,
                "CAGR": cagr,
                "Volatility": volatility,
                "Max Drawdown": max_drawdown,
                "Sharpe Ratio": sharpe,
                "Equity Curve": equity_curve
            }
        else:
            print("Error: 'strategy_equity' column not found in TurboQT results.")
            return None
            
    except Exception as e:
        print(f"Error loading or processing TurboQT results: {e}")
        return None

def compare_strategies():
    """Compare the performance of all three strategies."""
    results = []
    
    # Define the target display period
    target_start_date_str = "2020-01-01"
    target_end_date_str = "2025-04-26"
    target_period_str = f"{target_start_date_str} to {target_end_date_str}"

    tri_shot_data = load_tri_shot_results()
    if tri_shot_data:
        tri_shot_data['Period'] = target_period_str # Override period display
        results.append(tri_shot_data)
        
    dmt_data = load_dmt_results()
    if dmt_data:
        dmt_data['Period'] = target_period_str # Override period display
        results.append(dmt_data)
        
    turbo_qt_data = load_turbo_qt_results()
    if turbo_qt_data:
        turbo_qt_data['Period'] = target_period_str # Override period display
        results.append(turbo_qt_data)

    if not results:
        print("No strategy results found.")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY".center(80))
    print("=" * 80)
    
    print(f"{'Strategy':<10} {'Period':<25} {'Days':<6} {'Initial':<10} {'Final':<12} {'Return':<10} {'CAGR':<10} {'Volatility':<10} {'Max DD':<10} {'Sharpe':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['Strategy']:<10} {r['Period']:<25} {r['Trading Days']:<6} "
              f"{format_dollar(r['Initial Value']):<10} {format_dollar(r['Final Value']):<12} "
              f"{format_percent(r['Total Return']):<10} {format_percent(r['CAGR']):<10} "
              f"{format_percent(r['Volatility']):<10} {format_percent(r['Max Drawdown']):<10} "
              f"{r['Sharpe Ratio']:<8.2f}")
    
    print("=" * 80)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # First subplot for equity curves
    plt.subplot(2, 1, 1)
    for r in results:
        # Normalize the curve for comparison
        equity = r['Equity Curve']
        norm_equity = equity / equity.iloc[0]
        plt.plot(norm_equity.index, norm_equity, label=f"{r['Strategy']} ({format_percent(r['Total Return'])})")
    
    plt.title("Strategy Comparison (Normalized)")
    plt.ylabel("Growth of $1 Investment")
    plt.grid(True)
    plt.legend()
    
    # Second subplot for annualized metrics
    plt.subplot(2, 1, 2)
    strategies = [r['Strategy'] for r in results]
    metrics = {
        'CAGR': [r['CAGR'] * 100 for r in results],
        'Volatility': [r['Volatility'] * 100 for r in results],
        'Max Drawdown': [abs(r['Max Drawdown'] * 100) for r in results],
        'Sharpe': [r['Sharpe Ratio'] for r in results]
    }
    
    # Plot bar chart of metrics
    x = np.arange(len(strategies))  # the label locations
    width = 0.2  # the width of the bars
    
    # Plot each metric
    plt.bar(x - width*1.5, metrics['CAGR'], width, label='CAGR (%)')
    plt.bar(x - width/2, metrics['Volatility'], width, label='Volatility (%)')
    plt.bar(x + width/2, metrics['Max Drawdown'], width, label='Max Drawdown (%)')
    plt.bar(x + width*1.5, metrics['Sharpe'], width, label='Sharpe Ratio')
    
    plt.xlabel('Strategy')
    plt.ylabel('Value')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, strategies)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tri_shot_data/strategy_comparison.png')
    print(f"Comparison chart saved to tri_shot_data/strategy_comparison.png")
    
    # Close the plot to avoid displaying it
    plt.close()

if __name__ == "__main__":
    compare_strategies()
