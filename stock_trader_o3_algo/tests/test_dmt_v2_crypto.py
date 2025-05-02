#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMT_v2 Cryptocurrency Backtesting
=================================
This script tests the DMT_v2 strategy on cryptocurrency data.
It fetches historical price data for major cryptocurrencies (BTC, ETH)
and runs the various DMT_v2 strategy versions on this data.

Features:
- Fetches historical cryptocurrency data from public APIs
- Runs all three versions of DMT_v2 (original, enhanced, turbo)
- Compares performance against buy & hold
- Generates performance metrics and visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple, Union, Optional

# Add parent directory to path to allow importing from packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import DMT_v2 strategy
from stock_trader_o3_algo.strategies.dmt_v2_strategy import DMT_v2_Strategy


def fetch_crypto_data(symbol: str, start_date: str, end_date: str, 
                     interval: str = 'day') -> pd.DataFrame:
    """
    Fetch historical cryptocurrency data from CryptoCompare API.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Time interval ('day', 'hour')
        
    Returns:
        DataFrame with historical OHLCV data
    """
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    
    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate number of days needed
    days_diff = (end_dt - start_dt).days + 1
    
    # Define API endpoint and parameters
    base_url = 'https://min-api.cryptocompare.com/data/v2'
    
    if interval == 'day':
        endpoint = f"{base_url}/histoday"
        limit = min(2000, days_diff)  # API limit is 2000 data points per request
    elif interval == 'hour':
        endpoint = f"{base_url}/histohour"
        limit = min(2000, days_diff * 24)
    else:
        raise ValueError(f"Invalid interval: {interval}. Must be 'day' or 'hour'.")
    
    # Fetch data in batches if needed
    all_data = []
    current_date = end_dt
    
    while current_date >= start_dt:
        # Build API URL
        url = f"{endpoint}?fsym={symbol}&tsym=USD&limit={limit}&toTs={int(current_date.timestamp())}"
        
        # Make API request with user agent header to avoid rate limits
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            
            # Check for successful response
            if data.get('Response') == 'Success':
                batch_data = data['Data']['Data']
                all_data = batch_data + all_data
                
                # Update current date for next batch
                if batch_data:
                    oldest_timestamp = batch_data[0]['time']
                    current_date = datetime.fromtimestamp(oldest_timestamp) - timedelta(days=1)
                else:
                    break
            else:
                print(f"Error fetching data: {data.get('Message', 'Unknown error')}")
                break
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
        
        # Respect API rate limits
        time.sleep(0.5)
    
    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volumefrom': 'Volume'
        })
        
        # Set Date as index
        df = df.set_index('Date')
        
        # Sort by date
        df = df.sort_index()
        
        # Keep only required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    return pd.DataFrame()


def load_or_fetch_crypto_data(symbol: str, start_date: str, end_date: str, 
                             cache_dir: str = 'data/crypto') -> pd.DataFrame:
    """
    Load crypto data from cache if available, otherwise fetch from API.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Directory to store cached data
        
    Returns:
        DataFrame with historical OHLCV data
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f"{symbol.lower()}_{start_date}_{end_date}.csv")
    
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading {symbol} data from cache...")
        df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
        return df
    
    # Fetch data from API
    df = fetch_crypto_data(symbol, start_date, end_date)
    
    # Cache data if fetch was successful
    if not df.empty:
        df.to_csv(cache_file)
    
    return df


def calculate_performance_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        results: DataFrame with equity curve and buy & hold equity
        
    Returns:
        Dictionary of performance metrics
    """
    equity_curve = results['equity']
    buy_hold_equity = results['buy_hold_equity']
    
    # Calculate daily returns
    daily_returns = equity_curve.pct_change().dropna()
    buy_hold_returns = buy_hold_equity.pct_change().dropna()
    
    # Calculate total return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    buy_hold_return = buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1
    
    # Calculate annualized volatility
    annual_vol = daily_returns.std() * np.sqrt(252)
    buy_hold_vol = buy_hold_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    buy_hold_sharpe = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252) if buy_hold_returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    buy_hold_cum_returns = (1 + buy_hold_returns).cumprod()
    buy_hold_peak = buy_hold_cum_returns.expanding(min_periods=1).max()
    buy_hold_drawdown = (buy_hold_cum_returns / buy_hold_peak) - 1
    buy_hold_max_drawdown = buy_hold_drawdown.min()
    
    # Calculate CAGR
    days = (results.index[-1] - results.index[0]).days
    years = days / 365
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    buy_hold_cagr = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate Calmar ratio
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    buy_hold_calmar = abs(buy_hold_cagr / buy_hold_max_drawdown) if buy_hold_max_drawdown != 0 else 0
    
    # Calculate activity rate
    activity_rate = np.sum(np.abs(results['position'].diff()) > 0.01) / len(results) if len(results) > 0 else 0
    
    # Calculate alpha
    alpha = cagr - buy_hold_cagr
    
    # Calculate beta
    if buy_hold_returns.var() > 0:
        beta = daily_returns.cov(buy_hold_returns) / buy_hold_returns.var()
    else:
        beta = 0
    
    # Calculate Sortino ratio (downside risk)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = daily_returns.mean() * 252 / downside_deviation if downside_deviation > 0 else 0
    
    # Return metrics as dictionary
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'alpha': alpha,
        'beta': beta,
        'cagr': cagr,
        'buy_hold_cagr': buy_hold_cagr,
        'annual_volatility': annual_vol,
        'buy_hold_volatility': buy_hold_vol,
        'sharpe_ratio': sharpe_ratio,
        'buy_hold_sharpe': buy_hold_sharpe,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'buy_hold_max_drawdown': buy_hold_max_drawdown,
        'calmar_ratio': calmar_ratio,
        'buy_hold_calmar': buy_hold_calmar,
        'activity_rate': activity_rate
    }


def plot_results(results_dict: Dict[str, pd.DataFrame], symbol: str, 
               output_file: Optional[str] = None) -> str:
    """
    Plot equity curves for different strategy versions.
    
    Args:
        results_dict: Dictionary of results DataFrames for different strategy versions
        symbol: Cryptocurrency symbol
        output_file: File path to save the plot
        
    Returns:
        Path to saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    for version, results in results_dict.items():
        plt.plot(results.index, results['equity'], label=f"{version} DMT_v2")
    
    # Plot buy & hold equity
    first_results = next(iter(results_dict.values()))
    plt.plot(first_results.index, first_results['buy_hold_equity'], label='Buy & Hold', linestyle='--')
    
    # Format plot
    plt.title(f"{symbol} - DMT_v2 Strategy Performance", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format dates on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        return output_file
    
    return "Plot displayed but not saved"


def plot_regime_analysis(results: pd.DataFrame, symbol: str, 
                       output_file: Optional[str] = None) -> str:
    """
    Plot regime analysis for the TurboDMT_v2 strategy.
    
    Args:
        results: Results DataFrame for TurboDMT_v2
        symbol: Cryptocurrency symbol
        output_file: File path to save the plot
        
    Returns:
        Path to saved plot file
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # Plot price and equity curves
    ax0 = axes[0]
    ax0.plot(results.index, results['Close'], label=f"{symbol} Price", color='gray', alpha=0.5)
    ax0.set_ylabel(f"{symbol} Price ($)", fontsize=12)
    ax0.set_title(f"{symbol} - TurboDMT_v2 Regime Analysis", fontsize=16)
    ax0.grid(True, alpha=0.3)
    
    ax0_twin = ax0.twinx()
    ax0_twin.plot(results.index, results['equity'], label='Strategy Equity', color='green')
    ax0_twin.plot(results.index, results['buy_hold_equity'], label='Buy & Hold Equity', color='blue', linestyle='--')
    ax0_twin.set_ylabel('Portfolio Value ($)', fontsize=12)
    
    # Combine legends from both y-axes
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines0_twin, labels0_twin = ax0_twin.get_legend_handles_labels()
    ax0.legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper left')
    
    # Plot positions
    ax1 = axes[1]
    ax1.plot(results.index, results['position'], label='Position', color='purple')
    ax1.set_ylabel('Position Size', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot regime confidence
    ax2 = axes[2]
    ax2.fill_between(results.index, results['bull_confidence'], color='green', alpha=0.5, label='Bull Confidence')
    ax2.fill_between(results.index, results['bear_confidence'], color='red', alpha=0.5, label='Bear Confidence')
    ax2.fill_between(results.index, results['neutral_confidence'], color='gray', alpha=0.5, label='Neutral Confidence')
    ax2.set_ylabel('Regime Confidence', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Plot drawdowns
    ax3 = axes[3]
    
    # Calculate drawdowns
    strategy_returns = results['equity'].pct_change().dropna()
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    
    buy_hold_returns = results['buy_hold_equity'].pct_change().dropna()
    buy_hold_cum_returns = (1 + buy_hold_returns).cumprod()
    buy_hold_peak = buy_hold_cum_returns.expanding(min_periods=1).max()
    buy_hold_drawdown = (buy_hold_cum_returns / buy_hold_peak) - 1
    
    ax3.fill_between(drawdown.index, drawdown, color='green', alpha=0.5, label='Strategy Drawdown')
    ax3.fill_between(buy_hold_drawdown.index, buy_hold_drawdown, color='blue', alpha=0.3, label='Buy & Hold Drawdown')
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_ylim(-1, 0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left')
    
    # Format dates on x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        return output_file
    
    return "Plot displayed but not saved"


def main():
    """Run cryptocurrency backtests with DMT_v2 strategy."""
    # Set parameters
    start_date = "2018-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    symbols = ["BTC", "ETH"]
    output_dir = "crypto_results"
    initial_capital = 10000.0
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Print header
    print("="*80)
    print(f"DMT_v2 CRYPTOCURRENCY BACKTEST ({start_date} to {end_date})")
    print("="*80)
    
    # Loop through each cryptocurrency
    for symbol in symbols:
        print(f"\nRunning backtest for {symbol}...")
        
        # Load or fetch data
        data = load_or_fetch_crypto_data(symbol, start_date, end_date)
        
        if data.empty:
            print(f"Error: No data available for {symbol}")
            continue
        
        print(f"Data loaded: {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")
        
        # Initialize dictionaries to store results
        results_dict = {}
        metrics_dict = {}
        activity_rates = {}
        
        # Run each strategy version
        for version in ["original", "enhanced", "turbo"]:
            print(f"  Running {version} DMT_v2 strategy...")
            
            # Initialize strategy
            strategy = DMT_v2_Strategy(
                version=version,
                asset_type="crypto",
                lookback_period=252,
                initial_capital=initial_capital
            )
            
            # Run backtest
            results, activity_rate = strategy.run_backtest(data)
            
            # Store results
            results_dict[version] = results
            activity_rates[version] = activity_rate
            
            # Calculate performance metrics
            metrics_dict[version] = calculate_performance_metrics(results)
        
        # Create plots
        performance_plot = plot_results(
            results_dict,
            symbol,
            os.path.join(output_dir, f"{symbol.lower()}_performance.png")
        )
        
        regime_plot = plot_regime_analysis(
            results_dict["turbo"],
            symbol,
            os.path.join(output_dir, f"{symbol.lower()}_regime_analysis.png")
        )
        
        # Print performance metrics
        print("\n" + "-"*80)
        print(f"{symbol} PERFORMANCE METRICS")
        print("-"*80)
        print("Strategy          Total Return    CAGR       Sharpe     MaxDD       Calmar    Alpha")
        print("-"*80)
        
        for version in ["original", "enhanced", "turbo"]:
            metrics = metrics_dict[version]
            print(f"{version.ljust(18)} {metrics['total_return']:.2%}      {metrics['cagr']:.2%}      {metrics['sharpe_ratio']:.2f}      {metrics['max_drawdown']:.2%}      {metrics['calmar_ratio']:.2f}      {metrics['alpha']:.2%}")
        
        print(f"Buy & Hold         {metrics_dict['original']['buy_hold_return']:.2%}      {metrics_dict['original']['buy_hold_cagr']:.2%}      {metrics_dict['original']['buy_hold_sharpe']:.2f}      {metrics_dict['original']['buy_hold_max_drawdown']:.2%}      {metrics_dict['original']['buy_hold_calmar']:.2f}      0.00%")
        
        # Print strategy activity rates
        print("\n" + "-"*80)
        print(f"{symbol} ACTIVITY RATES")
        print("-"*80)
        for version, rate in activity_rates.items():
            print(f"{version.ljust(18)} {rate:.2%}")
        
        # Print TurboDMT_v2 regime statistics
        turbo_results = results_dict["turbo"]
        bull_days = (turbo_results['regime'] == 'Bull').sum()
        bear_days = (turbo_results['regime'] == 'Bear').sum()
        neutral_days = (turbo_results['regime'] == 'Neutral').sum()
        
        print("\n" + "-"*80)
        print(f"{symbol} REGIME STATISTICS (TurboDMT_v2)")
        print("-"*80)
        print(f"Bull Market:    {bull_days} days ({bull_days/len(turbo_results):.2%})")
        print(f"Bear Market:    {bear_days} days ({bear_days/len(turbo_results):.2%})")
        print(f"Neutral Market: {neutral_days} days ({neutral_days/len(turbo_results):.2%})")
        
        # Print plot locations
        print("\n" + "-"*80)
        print(f"{symbol} PLOTS")
        print("-"*80)
        print(f"Performance Plot:  {performance_plot}")
        print(f"Regime Analysis:   {regime_plot}")
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
