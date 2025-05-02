#!/usr/bin/env python3
"""
Test the TurboDMT_v2 strategy against the DMT_v2 strategy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
import torch

# Import DMT_v2 backtest function as reference
from stock_trader_o3_algo.strategies.dmt_v2.dmt_v2_backtest import run_dmt_v2_backtest

# Import TurboDMT_v2 components
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, HybridTransformerLSTM, TurboDMTEnsemble
from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import create_feature_matrix, prepare_data
from stock_trader_o3_algo.strategies.turbo_dmt_v2.risk_management import DynamicRiskManager

def run_turbo_dmt_v2_backtest(data, initial_capital=10000.0, n_epochs=100, target_annual_vol=0.35,
                           max_position_size=3.0, neutral_zone=0.01, plot=True, 
                           ensemble_size=7, max_drawdown_threshold=0.15):
    """
    Run a backtest of the TurboDMT_v2 strategy with enhanced features.
    
    Args:
        data (pandas.DataFrame): OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        initial_capital (float): Initial capital for the backtest
        n_epochs (int): Number of training epochs
        target_annual_vol (float): Target annualized volatility
        max_position_size (float): Maximum position size as a multiple of capital
        neutral_zone (float): Neutral zone size for position calculation
        plot (bool): Whether to plot the backtest results
        ensemble_size (int): Number of models in the ensemble
        max_drawdown_threshold (float): Maximum drawdown allowed before reducing exposure
        
    Returns:
        tuple: (results_df, metrics_dict)
    """
    # Prepare and process the data
    X_tensor, y_tensor, ret_tensor, dates, processed_df, _ = prepare_data(data)
    
    # Set device (use CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TurboDMT_v2 config
    config = TurboDMTConfig(
        max_position_size=max_position_size,
        target_vol=target_annual_vol,
        neutral_zone=neutral_zone,
        max_drawdown_threshold=max_drawdown_threshold,
        ensemble_size=ensemble_size
    )
    
    # Create ensemble model
    ensemble = TurboDMTEnsemble(config, device=device)
    
    # Create DataLoaders
    train_size = int(0.7 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    # Train the ensemble
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("Training TurboDMT_v2 ensemble...")
    ensemble.train(train_loader, epochs=n_epochs)
    
    # Create risk manager
    risk_manager = DynamicRiskManager(
        max_position_size=max_position_size,
        target_vol=target_annual_vol,
        neutral_zone=neutral_zone,
        max_drawdown_threshold=max_drawdown_threshold
    )
    
    # Run backtest
    print("Running backtest...")
    results = []
    portfolio_value = initial_capital
    position = 0
    buy_hold_equity = initial_capital
    
    # Use all data for the backtest (both train and test)
    for i in range(len(X_tensor) - 1):
        if i < config.seq_len:
            continue
            
        # Get prediction and uncertainty from ensemble
        x = X_tensor[i-config.seq_len:i].unsqueeze(0).to(device)
        prediction, uncertainty, regime = ensemble.predict(x)
        
        # Get current price and next day's price
        current_price = data.iloc[i]['Close']
        next_price = data.iloc[i + 1]['Close']
        
        # Calculate returns
        daily_return = (next_price / current_price) - 1
        
        # Get position from risk manager
        new_position = risk_manager.calculate_position(
            prediction.item(), 
            uncertainty.item(),
            regime.argmax().item(),
            portfolio_value,
            current_price,
            data.iloc[max(0, i-20):i]
        )
        
        # Calculate P&L
        pnl = position * portfolio_value * daily_return
        portfolio_value += pnl
        
        # Update buy & hold equity
        buy_hold_equity *= (1 + daily_return)
        
        # Update position for next day
        position = new_position
        
        # Store results
        results.append({
            'date': data.index[i],
            'price': current_price,
            'prediction': prediction.item(),
            'uncertainty': uncertainty.item(),
            'regime': regime.argmax().item(),
            'position': position,
            'portfolio_value': portfolio_value,
            'buy_hold_equity': buy_hold_equity,
            'daily_return': daily_return,
            'strategy_return': pnl / (portfolio_value - pnl) if portfolio_value > pnl else 0
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results_df, initial_capital)
    
    # Plot results if requested
    if plot:
        plot_backtest_results(results_df, metrics)
    
    return results_df, metrics

def calculate_performance_metrics(results_df, initial_capital):
    """
    Calculate performance metrics from backtest results
    
    Args:
        results_df (pandas.DataFrame): Backtest results
        initial_capital (float): Initial capital
        
    Returns:
        dict: Performance metrics
    """
    # Extract relevant data
    equity_curve = results_df['portfolio_value']
    buy_hold_equity = results_df['buy_hold_equity']
    daily_returns = results_df['strategy_return']
    
    # Calculate metrics
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    buy_hold_return = (buy_hold_equity.iloc[-1] / initial_capital) - 1
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
    
    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate CAGR
    days = (results_df.index[-1] - results_df.index[0]).days
    years = days / 365
    cagr = (equity_curve.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
    
    # Win rate
    win_rate = (daily_returns > 0).sum() / len(daily_returns)
    
    # Average win/loss ratio
    avg_win = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
    avg_loss = daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Calmar ratio
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cagr': cagr,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'calmar_ratio': calmar_ratio
    }

def plot_backtest_results(results_df, metrics):
    """
    Plot backtest results with enhanced visualization
    
    Args:
        results_df (pandas.DataFrame): Backtest results
        metrics (dict): Performance metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index, results_df['portfolio_value'], label='TurboDMT_v2')
    plt.plot(results_df.index, results_df['buy_hold_equity'], label='Buy & Hold', linestyle=':')
    
    plt.title('TurboDMT_v2 Strategy Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Add metrics to the plot
    txt = (
        f"Total Return: {metrics['total_return']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"CAGR: {metrics['cagr']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}\n"
        f"Calmar Ratio: {metrics['calmar_ratio']:.2f}"
    )
    plt.annotate(txt, xy=(0.01, 0.6), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.grid(True)
    plt.legend()
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    rolling_max = results_df['portfolio_value'].cummax()
    drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max
    plt.fill_between(results_df.index, 0, drawdown, color='red', alpha=0.3)
    plt.plot(results_df.index, drawdown, color='red', linestyle=':')
    
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown %')
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join('tri_shot_data')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'turbo_dmt_v2_backtest.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Set parameters
    initial_capital = 500.0
    test_period = 30
    end_date = datetime.now() - timedelta(days=100)
    start_date = end_date - timedelta(days=test_period)
    
    # Format dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    lookback_start = start_date - timedelta(days=365)
    lookback_start_str = lookback_start.strftime('%Y-%m-%d')
    
    print(f"Testing period: {start_date_str} to {end_date_str}")
    print(f"Fetching data from {lookback_start_str} to include lookback period")
    
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
    dmt_params = {
        'initial_capital': initial_capital,
        'n_epochs': 30,  # Reduced for quicker testing
        'target_annual_vol': 0.35,
        'max_position_size': 2.0,
        'neutral_zone': 0.03,
        'plot': False,
        'use_ensemble': True,
        'use_dynamic_stops': True,
        'max_drawdown_threshold': 0.15
    }
    
    turbo_params = {
        'initial_capital': initial_capital,
        'n_epochs': 30,  # Reduced for quicker testing
        'target_annual_vol': 0.35,
        'max_position_size': 3.0,  # Increased for TurboDMT_v2
        'neutral_zone': 0.01,  # Reduced for more active trading
        'plot': False,
        'ensemble_size': 7,
        'max_drawdown_threshold': 0.15
    }
    
    # Add delay to avoid rate limits
    time.sleep(2)
    
    # Run DMT_v2 backtest
    print("\nRunning DMT_v2 backtest...")
    try:
        dmt_results_df, dmt_metrics = run_dmt_v2_backtest(data, **dmt_params)
        print(f"\nDMT_v2 Results:")
        print(f"Total Return: {dmt_metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {dmt_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {dmt_metrics['max_drawdown']:.2%}")
        print(f"CAGR: {dmt_metrics['cagr']:.2%}")
    except Exception as e:
        print(f"Error running DMT_v2 backtest: {e}")
        import traceback
        traceback.print_exc()
        dmt_results_df = None
        dmt_metrics = None
    
    # Add delay to avoid rate limits
    time.sleep(3)
    
    # Run TurboDMT_v2 backtest
    print("\nRunning TurboDMT_v2 backtest...")
    try:
        turbo_results_df, turbo_metrics = run_turbo_dmt_v2_backtest(data, **turbo_params)
        print(f"\nTurboDMT_v2 Results:")
        print(f"Total Return: {turbo_metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {turbo_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {turbo_metrics['max_drawdown']:.2%}")
        print(f"CAGR: {turbo_metrics['cagr']:.2%}")
        print(f"Win Rate: {turbo_metrics['win_rate']:.2%}")
        print(f"Win/Loss Ratio: {turbo_metrics['win_loss_ratio']:.2f}")
        print(f"Calmar Ratio: {turbo_metrics['calmar_ratio']:.2f}")
    except Exception as e:
        print(f"Error running TurboDMT_v2 backtest: {e}")
        import traceback
        traceback.print_exc()
        turbo_results_df = None
        turbo_metrics = None
    
    # Compare results if both tests ran successfully
    if dmt_results_df is not None and turbo_results_df is not None:
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        plt.plot(dmt_results_df.index, dmt_results_df['dmt_v2_equity'], label='DMT_v2')
        plt.plot(turbo_results_df.index, turbo_results_df['portfolio_value'], label='TurboDMT_v2')
        plt.plot(dmt_results_df.index, dmt_results_df['buy_hold_equity'], label='Buy & Hold', linestyle=':')
        
        # Add metrics to the plot
        dmt_ret = dmt_metrics['total_return']
        turbo_ret = turbo_metrics['total_return']
        dmt_sharpe = dmt_metrics['sharpe_ratio']
        turbo_sharpe = turbo_metrics['sharpe_ratio']
        dmt_dd = dmt_metrics['max_drawdown']
        turbo_dd = turbo_metrics['max_drawdown']
        
        plt.title(f'DMT_v2 vs TurboDMT_v2 Comparison ({start_date_str} to {end_date_str})')
        plt.annotate(f"DMT_v2 Return: {dmt_ret:.2%}, Sharpe: {dmt_sharpe:.2f}, MaxDD: {dmt_dd:.2%}", 
                     xy=(0.05, 0.95), xycoords='axes fraction')
        plt.annotate(f"TurboDMT_v2 Return: {turbo_ret:.2%}, Sharpe: {turbo_sharpe:.2f}, MaxDD: {turbo_dd:.2%}", 
                     xy=(0.05, 0.90), xycoords='axes fraction')
        
        # Calculate and display improvement metrics
        ret_improvement = (turbo_ret - dmt_ret) / abs(dmt_ret) if dmt_ret != 0 else float('inf')
        sharpe_improvement = (turbo_sharpe - dmt_sharpe) / abs(dmt_sharpe) if dmt_sharpe != 0 else float('inf')
        dd_improvement = (dmt_dd - turbo_dd) / abs(dmt_dd) if dmt_dd != 0 else float('inf')
        
        improvement_msg = f"Improvements: Return: {ret_improvement:.2%}, Sharpe: {sharpe_improvement:.2%}, DrawDown: {dd_improvement:.2%}"
        plt.annotate(improvement_msg, xy=(0.05, 0.85), xycoords='axes fraction')
        
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Save comparison plot
        output_dir = os.path.join('tri_shot_data')
        os.makedirs(output_dir, exist_ok=True)
        comparison_file = os.path.join(output_dir, 'dmt_v2_vs_turbo_dmt_v2_comparison.png')
        plt.savefig(comparison_file)
        
        print(f"\nComparison Results:")
        print(f"Return: {dmt_ret:.2%} → {turbo_ret:.2%} ({ret_improvement:+.2%})")
        print(f"Sharpe: {dmt_sharpe:.2f} → {turbo_sharpe:.2f} ({sharpe_improvement:+.2%})")
        print(f"Max Drawdown: {dmt_dd:.2%} → {turbo_dd:.2%} ({dd_improvement:+.2%})")
        print(f"Comparison chart saved to {comparison_file}")
    else:
        print("\nUnable to complete comparison due to errors in one or both backtests")
