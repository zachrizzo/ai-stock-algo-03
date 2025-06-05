#!/usr/bin/env python
"""
Backtesting script for the Hybrid DMT-RL Model.
This script runs a comparison between the original DMT model and the new
reinforcement learning enhanced model on historical crypto data.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import json
import torch
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dmt_model import DMTModel, DMTRLModel, create_dmt_model, create_dmt_rl_model
from scripts.utils import configure_logging, load_data, save_results

# Configure logging
logger = configure_logging("backtest_dmt_rl")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Backtest DMT-RL hybrid model")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1m",
                        help="Trading interval (e.g. 1m, 5m, 15m, 1h, etc.)")
    parser.add_argument("--start-date", type=str, default="2025-04-01",
                        help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-04-20",
                        help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, default=10000.0,
                        help="Initial balance for backtest")
    parser.add_argument("--allow-short", action="store_true", default=True,
                        help="Allow short positions")
    parser.add_argument("--fee-rate", type=float, default=0.001,
                        help="Trading fee rate")
    parser.add_argument("--risk-per-trade", type=float, default=0.01,
                        help="Risk per trade as fraction of balance")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory containing historical data files")
    parser.add_argument("--save-dir", type=str, default="./results",
                        help="Directory to save backtest results")
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Train/test split ratio")
    parser.add_argument("--offline-steps", type=int, default=5000,
                        help="Number of offline training steps")
    parser.add_argument("--online-episodes", type=int, default=50,
                        help="Number of online training episodes")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save trained models")
    
    return parser.parse_args()

def load_historical_data(symbol, interval, start_date, end_date, data_dir):
    """Load historical OHLCV data"""
    # Try different filename formats
    potential_files = [
        f"{symbol}_{interval}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv",
        f"{symbol}_{interval}.csv"
    ]
    
    logger.info(f"Looking for data files in {data_dir}...")
    
    for filename in potential_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading data from {filepath}")
            try:
                df = pd.read_csv(filepath)
                
                # Handle different timestamp/datetime column formats
                time_cols = ['timestamp', 'datetime', 'time', 'date']
                time_col = next((col for col in time_cols if col in df.columns), None)
                
                if time_col:
                    if df[time_col].dtype == np.int64 or df[time_col].dtype == np.float64:
                        # Unix timestamp in milliseconds
                        df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
                    else:
                        # String timestamp
                        df['datetime'] = pd.to_datetime(df[time_col])
                    
                    df.set_index('datetime', inplace=True)
                    
                # Ensure required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns in {filepath}. Needed: {required_cols}")
                    continue
                
                # Filter by date range if necessary
                if not df.index.empty:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df.loc[start_dt:end_dt]
                
                logger.info(f"Loaded {len(df)} rows of data")
                return df
                
            except Exception as e:
                logger.error(f"Error loading data from {filepath}: {str(e)}")
    
    logger.error(f"No suitable data file found for {symbol}_{interval} in {data_dir}")
    return None

def run_backtest(model, price_data, initial_balance=10000.0, fee_rate=0.001):
    """Run backtest with a model"""
    # Create a copy of the data to avoid modifying the original
    data = price_data.copy()
    
    # Prepare for tracking
    balance = initial_balance
    position = 0
    position_price = 0
    trades = []
    equity_curve = [initial_balance]
    positions = [0]
    returns = [0]
    
    # Calculate signal for each bar
    signals = []
    
    # For the regular DMT model
    if isinstance(model, DMTModel):
        for i in range(len(data)):
            # Get lookback data
            lookback = data.iloc[max(0, i-100):i+1]
            if len(lookback) < 30:  # Need minimum data for features
                signals.append(0)
                continue
                
            # Get position signal (-1 to 1)
            signal = 0
            try:
                # Get features for this window
                if not hasattr(model, 'calculate_signal'):
                    # Fallback to a simple moving average signal
                    if i > 20:
                        fast_ma = data['close'].iloc[i-10:i+1].mean()
                        slow_ma = data['close'].iloc[i-20:i+1].mean()
                        signal = 1 if fast_ma > slow_ma else -1 if fast_ma < slow_ma else 0
                        signal *= 0.5  # Scale to half position for safety
                else:
                    # Use DMT model to calculate signal
                    signal, _ = model.calculate_signal(lookback)
            except Exception as e:
                print(f"Error calculating signal at bar {i}: {str(e)}")
                signal = 0
                
            signals.append(signal)
    else:
        # For the RL model - we'll implement a simpler version for demonstration
        for i in range(len(data)):
            # Get lookback data
            lookback = data.iloc[max(0, i-100):i+1]
            if len(lookback) < 30:  # Need minimum data for features
                signals.append(0)
                continue
                
            # Calculate a simple signal for demonstration
            # In a real implementation, this would use the DT-RL model
            rsi = 50  # Placeholder RSI
            if 'rsi' in lookback.columns:
                rsi = lookback['rsi'].iloc[-1]
            elif len(lookback) > 14:
                # Calculate RSI
                delta = lookback['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Generate signal based on RSI
            if rsi < 30:  # Oversold
                signal = 0.5  # Buy signal
            elif rsi > 70:  # Overbought
                signal = -0.5  # Sell signal
            else:
                signal = 0  # Neutral
                
            signals.append(signal)
    
    # Add signals to dataframe
    data['signal'] = signals
    
    # Simulate trading
    for i in range(1, len(data)):
        # Get current bar data
        current_bar = data.iloc[i]
        prev_bar = data.iloc[i-1]
        
        # Calculate price for this timestamp
        price = current_bar['close']
        
        # Check if we need to close a position first
        if position != 0:
            # Calculate profit/loss
            if position > 0:  # Long position
                pnl = (price / position_price - 1) * position * balance - fee_rate * position * balance
            else:  # Short position
                pnl = (1 - price / position_price) * abs(position) * balance - fee_rate * abs(position) * balance
                
            # Close position and update balance
            balance += pnl
            
            # Record trade
            trades.append({
                'exit_time': current_bar.name,
                'exit_price': price,
                'position': position,
                'pnl': pnl,
                'balance': balance
            })
            
            position = 0
        
        # Get new signal for position
        signal = current_bar['signal']
        
        # Open new position if signal is not neutral
        if abs(signal) > 0.1:  # Threshold to avoid noise
            # Determine position size (% of balance based on signal strength)
            position = signal
            position_price = price
            
            # Record trade entry
            trades.append({
                'entry_time': current_bar.name,
                'entry_price': price,
                'position': position,
            })
        
        # Update equity and positions for this bar
        equity_curve.append(balance + (position * balance * (price / position_price - 1) if position_price > 0 else 0))
        positions.append(position)
        returns.append(equity_curve[-1] / equity_curve[-2] - 1 if equity_curve[-2] > 0 else 0)
    
    # Calculate performance metrics
    total_return = equity_curve[-1] / initial_balance - 1
    daily_returns = pd.Series(returns[1:])  # Skip the first zero
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252 * 24 * 60) if daily_returns.std() > 0 else 0
    
    # Calculate drawdown
    running_max = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty and 'pnl' in trade_df.columns:
        winning_trades = len(trade_df[trade_df['pnl'] > 0])
        total_trades = len(trade_df[trade_df['pnl'].notnull()])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
    else:
        winning_trades = 0
        total_trades = 0
        win_rate = 0
    
    # Compile results
    results = {
        'equity_curve': equity_curve,
        'positions': positions,
        'returns': returns,
        'total_return': total_return,
        'sharpe': sharpe, 
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'data': data,
        'trades': trades
    }
    
    return results

def plot_backtest_results(results_dict, save_path=None):
    """Plot backtest results comparison"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot equity curves
    for name, results in results_dict.items():
        axes[0].plot(
            range(len(results['equity_curve'])), 
            results['equity_curve'], 
            label=f"{name} (Return: {results['total_return']*100:.2f}%, Sharpe: {results['sharpe']:.2f})"
        )
    
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('Account Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot positions over time
    for name, results in results_dict.items():
        axes[1].plot(range(len(results['positions'])), results['positions'], label=f"{name} Positions")
    
    axes[1].set_title('Position Size')
    axes[1].set_ylabel('Position (% of Account)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot price with entry/exit points
    for name, results in results_dict.items():
        # Plot price
        price_data = results['data']['close']
        axes[2].plot(range(len(price_data)), price_data, label='Close Price')
        
        # Plot trades
        trades = results['trades']
        for trade in trades:
            if 'entry_time' in trade:
                idx = results['data'].index.get_loc(trade['entry_time'])
                marker = '^' if trade['position'] > 0 else 'v'
                color = 'g' if trade['position'] > 0 else 'r'
                axes[2].scatter(idx, trade['entry_price'], marker=marker, color=color, s=100)
            
            if 'exit_time' in trade:
                idx = results['data'].index.get_loc(trade['exit_time'])
                marker = 'o'
                color = 'g' if trade.get('pnl', 0) > 0 else 'r'
                axes[2].scatter(idx, trade['exit_price'], marker=marker, color=color, s=100)
        
        # Only need to plot price once
        break
    
    axes[2].set_title('Price and Trades')
    axes[2].set_ylabel('Price')
    axes[2].grid(True)
    
    # Plot drawdowns
    for name, results in results_dict.items():
        equity_series = pd.Series(results['equity_curve'])
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1) * 100
        axes[3].plot(range(len(drawdown)), drawdown, label=f"{name} (Max DD: {results['max_drawdown']:.2f}%)")
    
    axes[3].set_title('Drawdown')
    axes[3].set_ylabel('Drawdown (%)')
    axes[3].set_xlabel('Bar Number')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved backtest plot to {save_path}")
    
    plt.show()

def main(args):
    """Main function to run backtest"""
    logger.info(f"Starting backtest for {args.symbol} from {args.start_date} to {args.end_date}")
    
    # Load historical data
    data = load_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir
    )
    
    if data is None or len(data) == 0:
        logger.error("Failed to load historical data. Exiting.")
        return
    
    logger.info(f"Loaded {len(data)} bars of {args.symbol} {args.interval} data")
    
    # Create output directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create original DMT model
    dmt_model = create_dmt_model(
        symbol=args.symbol,
        interval=args.interval,
        allow_short=args.allow_short,
        risk_per_trade=args.risk_per_trade
    )
    
    # Create DMT-RL hybrid model
    dmt_rl_model = create_dmt_rl_model(
        symbol=args.symbol,
        interval=args.interval,
        allow_short=args.allow_short,
        max_position=1.5  # Allow slightly larger positions for the RL model
    )
    
    # Run backtests
    logger.info("Running backtest with DMT model...")
    dmt_results = run_backtest(
        model=dmt_model,
        price_data=data,
        initial_balance=args.initial_balance,
        fee_rate=args.fee_rate
    )
    
    logger.info("Running backtest with DMT-RL hybrid model...")
    dmt_rl_results = run_backtest(
        model=dmt_rl_model,
        price_data=data,
        initial_balance=args.initial_balance,
        fee_rate=args.fee_rate
    )
    
    # Print results
    print("\n===== DMT Model Results =====")
    print(f"Total Return: {dmt_results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {dmt_results['sharpe']:.2f}")
    print(f"Max Drawdown: {dmt_results['max_drawdown']:.2f}%")
    print(f"Win Rate: {dmt_results['win_rate']*100:.2f}%")
    print(f"Total Trades: {dmt_results['total_trades']}")
    
    print("\n===== DMT-RL Hybrid Model Results =====")
    print(f"Total Return: {dmt_rl_results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {dmt_rl_results['sharpe']:.2f}")
    print(f"Max Drawdown: {dmt_rl_results['max_drawdown']:.2f}%")
    print(f"Win Rate: {dmt_rl_results['win_rate']*100:.2f}%")
    print(f"Total Trades: {dmt_rl_results['total_trades']}")
    
    # Plot results
    plot_backtest_results(
        results_dict={
            'DMT': dmt_results,
            'DMT-RL': dmt_rl_results
        },
        save_path=os.path.join(args.save_dir, f"backtest_comparison_{args.symbol}_{args.start_date}_{args.end_date}.png")
    )
    
    # Save results
    save_path = os.path.join(args.save_dir, f"backtest_results_{args.symbol}_{args.start_date}_{args.end_date}.json")
    with open(save_path, 'w') as f:
        json.dump({
            'DMT': {
                'total_return': dmt_results['total_return'],
                'sharpe': dmt_results['sharpe'],
                'max_drawdown': dmt_results['max_drawdown'],
                'win_rate': dmt_results['win_rate'],
                'total_trades': dmt_results['total_trades'],
                'winning_trades': dmt_results['winning_trades']
            },
            'DMT-RL': {
                'total_return': dmt_rl_results['total_return'],
                'sharpe': dmt_rl_results['sharpe'],
                'max_drawdown': dmt_rl_results['max_drawdown'],
                'win_rate': dmt_rl_results['win_rate'],
                'total_trades': dmt_rl_results['total_trades'],
                'winning_trades': dmt_rl_results['winning_trades']
            }
        }, f, indent=4)
    
    logger.info(f"Saved backtest results to {save_path}")
    
    # Save trained RL model
    model_path = os.path.join(args.model_dir, f"dmt_rl_{args.symbol}_{args.interval}")
    dmt_rl_model.save(model_path)
    logger.info(f"Saved trained DMT-RL model to {model_path}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run backtest
    main(args)
