#!/usr/bin/env python3
import argparse
import os
import sys
import datetime as dt
import pytz
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import matplotlib.pyplot as plt

# Import our modules
import tri_shot_features as tsf
from tri_shot_model import WalkForwardModel, load_walk_forward_model
from tri_shot import (
    STATE_DIR, MODEL_FILE, TZ,
    ensure_state_dir, get_alpaca_api,
    run_monday_strategy, run_wednesday_strategy, run_friday_strategy
)

# Import DMT modules
try:
    import torch
    from dmt_model import MarketTwinLSTM, train_market_twin, load_market_twin
    from dmt_strategy import DifferentiableTriShot
    from dmt_backtest import run_dmt_backtest
    HAS_DMT_DEPS = True
except ImportError:
    HAS_DMT_DEPS = False

def setup_environment():
    """Ensure the environment is properly set up."""
    ensure_state_dir()

    # Check API keys only if we're not running a backtest
    if sys.argv[1] not in ["backtest", "dmt"]:
        if not os.getenv("ALPACA_KEY") or not os.getenv("ALPACA_SECRET"):
            print("ERROR: Alpaca API credentials not found in environment variables")
            print("Please set ALPACA_KEY and ALPACA_SECRET environment variables")
            return False

    return True

def train_model(force=False):
    """Train or retrain the XGBoost model."""
    if MODEL_FILE.exists() and not force:
        print(f"Model already exists at {MODEL_FILE}. Use --force to retrain.")
        return

    print("Fetching data for model training...")
    tickers = list(TICKERS.values())
    prices = tsf.fetch_data(tickers, days=1500)  # Use more data for training

    print("Training model...")
    model = tsf.train_model(prices)

    print(f"Saving model to {MODEL_FILE}")
    tsf.save_model(model, MODEL_FILE)
    print("Model training complete!")

def backtest(days=365, plot=False, initial_capital=500.0, start_date=None, slippage_bps=1, commission_bps=1, monte_carlo=False, mc_runs=10):
    """Run a comprehensive backtest on historical data with realistic execution.
    
    Args:
        days: Number of days to backtest (if start_date is None)
        plot: Whether to plot results
        initial_capital: Starting capital amount
        start_date: Optional specific start date (format: 'YYYY-MM-DD')
        slippage_bps: Slippage in basis points per side (1 bps = 0.01%)
        commission_bps: Commission in basis points per side
        monte_carlo: Whether to run Monte Carlo with randomized start dates
        mc_runs: Number of Monte Carlo runs if monte_carlo=True
    """
    if monte_carlo:
        return run_monte_carlo_backtest(days, initial_capital, slippage_bps, commission_bps, mc_runs)
    
    # Convert transaction costs to decimal
    slippage = slippage_bps / 10000  # Convert bps to decimal
    commission = commission_bps / 10000  # Convert bps to decimal
    total_cost_per_side = slippage + commission
    
    if start_date:
        print(f"Running enhanced backtest from {start_date} with ${initial_capital:.2f} initial capital...")
        print(f"Including slippage ({slippage_bps} bps) and commission ({commission_bps} bps) per side...")
    else:
        print(f"Running enhanced backtest over the last {days} days with ${initial_capital:.2f} initial capital...")
        print(f"Including slippage ({slippage_bps} bps) and commission ({commission_bps} bps) per side...")

    # Define tickers
    TICKERS = {
        "UP": "TQQQ",    # 3x long QQQ
        "DN": "SQQQ",    # 3x short QQQ
        "BOND": "TMF",   # 3x long treasury
        "CASH": "BIL",   # Short-term treasury ETF (cash equivalent)
        "SRC": "QQQ",    # Base asset to track
        "VIX": "^VIX"    # Volatility index
    }

    # Fetch data - use longer history for feature calculation and training
    lookback_window = days + 400  # Add buffer for feature calculation
    tickers = list(TICKERS.values())
    # Add TLT and other data sources for enhanced features
    additional_tickers = ['TLT', 'UUP']
    for ticker in additional_tickers:
        if ticker not in tickers:
            tickers.append(ticker)

    print("Fetching historical data...")
    if start_date:
        # Convert start_date string to datetime
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        # Get data from start_date to present
        prices = tsf.fetch_data_from_date(tickers, start_date=start_dt)
    else:
        prices = tsf.fetch_data(tickers, days=lookback_window)

    # Train walk-forward model or load existing one
    model_file = STATE_DIR / "tri_shot_ensemble.pkl"
    if model_file.exists() and days <= 400 and not start_date:  # Use existing model for short backtests
        print("Loading existing model...")
        model = load_walk_forward_model(STATE_DIR)
        if model is None:  # Fall back to training if loading fails
            print("Model loading failed. Training new model...")
            model, _ = train_walk_forward_model(prices, save_model=True)
    else:
        print("Training walk-forward ensemble model...")
        model, metrics = train_walk_forward_model(prices, save_model=True)
        print(f"Model trained with directional accuracy: {metrics['directional_accuracy']:.4f}")

    # Create features for the full period
    print("Generating features for backtest...")
    X, y = tsf.make_feature_matrix(prices)

    # Generate predictions
    print("Generating predictions...")
    probabilities = model.predict(X)

    # Create backtest dataframe
    results = pd.DataFrame(index=X.index)
    results['date'] = results.index
    results['actual'] = y
    results['probability'] = probabilities
    results['predicted'] = (probabilities > 0.5).astype(int)

    # Copy price data for matching dates
    for ticker in ['QQQ', 'TQQQ', 'SQQQ', 'TMF', 'TLT']:
        if ticker in prices.columns:
            results[ticker] = prices[ticker].reindex(results.index)

    # Add VIX if available
    if '^VIX' in prices.columns:
        results['VIX'] = prices['^VIX'].reindex(results.index)

    # Signal quality metrics and market regime detection
    results['signal_strength'] = abs(results['probability'] - 0.5) / 0.5
    results['price_momentum'] = results['QQQ'].pct_change(5)
    results['high_conviction'] = results['signal_strength'] > 0.15
    
    # Market regime detection
    if len(results) > 200:  # Only if we have enough data
        results['sma_200'] = results['QQQ'].rolling(200).mean()
        results['bull_market'] = results['QQQ'] > results['sma_200']
    else:
        results['bull_market'] = True  # Default to bull if not enough data
    
    # VIX-based opportunity detection
    if '^VIX' in prices.columns:
        results['vix_spike'] = (results['VIX'].pct_change(5) > 0.15) & (results['VIX'] > 20)
        results['vix_collapse'] = (results['VIX'].pct_change(5) < -0.15) & (results['VIX'] < 30)
    else:
        results['vix_spike'] = False
        results['vix_collapse'] = False

    # Determine positions
    results['position'] = 'CASH'
    
    # Long condition
    long_condition = (
        (results['probability'] >= 0.52) & 
        (results['price_momentum'] > 0)
    )
    
    # Short condition
    short_condition = (
        (results['probability'] <= 0.48) &
        (results['price_momentum'] < 0)
    )
    
    # Opportunistic conditions
    opportunistic_long = results['vix_collapse'] & results['bull_market']
    opportunistic_short = results['vix_spike'] & (~results['bull_market'])
    
    # Assign positions
    results.loc[long_condition | opportunistic_long, 'position'] = 'TQQQ'
    results.loc[short_condition | opportunistic_short, 'position'] = 'SQQQ'
    
    # Bond condition
    bond_momentum = prices['TLT'].pct_change(20).reindex(results.index) if 'TLT' in prices.columns else pd.Series(0, index=results.index)
    results.loc[(results['probability'] > 0.48) & 
                (results['probability'] < 0.52) & 
                (bond_momentum > 0.01), 'position'] = 'TMF'

    # Apply PDT constraint - limit to 3 trades per 5-day rolling window
    results['position_change'] = results['position'] != results['position'].shift(1)
    results['trade_counter'] = results['position_change'].rolling(5).sum()
    results.loc[results['trade_counter'] > 3, 'position'] = results['position'].shift(1)
    
    # Calculate position sizes with more aggressive sizing for small accounts
    for ticker in ['TQQQ', 'SQQQ', 'TMF']:
        if ticker in prices.columns:
            # Calculate rolling volatility
            vol = prices[ticker].pct_change().rolling(20).std() * np.sqrt(252)
            vol = vol.reindex(results.index).fillna(0.50)  # Use conservative default if NA
            
            # More aggressive for small accounts
            results[f'{ticker}_weight'] = 0.0
            mask = results['position'] == ticker
            
            if initial_capital <= 1000:  # Small account
                # High conviction positions
                high_conviction = results.loc[mask, 'signal_strength'] > 0.25
                medium_conviction = (results.loc[mask, 'signal_strength'] > 0.15) & (results.loc[mask, 'signal_strength'] <= 0.25)
                
                # Assign weights based on conviction
                results.loc[mask & high_conviction, f'{ticker}_weight'] = 1.0  # Full account
                results.loc[mask & medium_conviction, f'{ticker}_weight'] = 0.8  # 80% of account
                results.loc[mask & ~(high_conviction | medium_conviction), f'{ticker}_weight'] = 0.6  # 60% of account
            else:  # Larger account
                # Scale down position sizes for larger accounts
                if initial_capital <= 10000:
                    results.loc[mask, f'{ticker}_weight'] = 0.5  # 50% for medium accounts
                else:
                    results.loc[mask, f'{ticker}_weight'] = 0.3  # 30% for large accounts
    
    # Apply weights to calculate returns
    results['position_weight'] = 0.0
    for ticker in ['TQQQ', 'SQQQ', 'TMF']:
        results.loc[results['position'] == ticker, 'position_weight'] = results.loc[results['position'] == ticker, f'{ticker}_weight']
    
    # Calculate daily returns for each asset
    for ticker in ['QQQ', 'TQQQ', 'SQQQ', 'TMF']:
        if ticker in results.columns:
            results[f'{ticker}_return'] = results[ticker].pct_change()
    
    # Calculate strategy returns with explicit slippage and commission
    results['strategy_return'] = 0.0
    
    # Apply transaction costs when position changes
    results['transaction_cost'] = 0.0
    results.loc[results['position_change'], 'transaction_cost'] = total_cost_per_side * 2  # Entry and exit costs
    
    # Calculate weighted returns
    for ticker in ['TQQQ', 'SQQQ', 'TMF']:
        ticker_mask = results['position'] == ticker
        if ticker in results.columns:
            results.loc[ticker_mask, 'strategy_return'] = (
                results.loc[ticker_mask, f'{ticker}_return'] * 
                results.loc[ticker_mask, 'position_weight'] - 
                results.loc[ticker_mask, 'transaction_cost']
            )
    
    # Calculate equity curves with initial capital
    results['strategy_equity'] = (1 + results['strategy_return']).cumprod() * initial_capital
    results['buy_hold_equity'] = (1 + results['QQQ_return']).cumprod() * initial_capital
    
    # Calculate trailing high watermark and drawdowns
    results['strategy_peak'] = results['strategy_equity'].cummax()
    results['strategy_drawdown'] = (results['strategy_equity'] / results['strategy_peak'] - 1)
    
    # Calculate buy & hold drawdown for comparison
    results['buy_hold_peak'] = results['buy_hold_equity'].cummax()
    results['buy_hold_drawdown'] = (results['buy_hold_equity'] / results['buy_hold_peak'] - 1)
    
    # Calculate key metrics
    total_days = len(results.dropna())
    if total_days == 0:
        print("Warning: No valid trading days in results. Check your data and filters.")
        total_days = 1  # Safe default
    
    trading_days_per_year = 252
    
    # Performance metrics
    correct_predictions = ((results['predicted'] == 1) & (results['actual'] == 1) | 
                           (results['predicted'] == 0) & (results['actual'] == 0)).sum()
    accuracy = correct_predictions / len(results) if len(results) > 0 else 0
    
    # Returns
    strategy_returns = results['strategy_return'].dropna()
    buy_hold_returns = results['QQQ_return'].dropna()
    
    strategy_cagr = (results['strategy_equity'].iloc[-1] / initial_capital) ** (trading_days_per_year / total_days) - 1
    buy_hold_cagr = (results['buy_hold_equity'].iloc[-1] / initial_capital) ** (trading_days_per_year / total_days) - 1
    
    # Risk metrics
    strategy_vol = strategy_returns.std() * np.sqrt(trading_days_per_year)
    buy_hold_vol = buy_hold_returns.std() * np.sqrt(trading_days_per_year)
    
    strategy_sharpe = strategy_cagr / strategy_vol if strategy_vol > 0 else 0
    buy_hold_sharpe = buy_hold_cagr / buy_hold_vol if buy_hold_vol > 0 else 0
    
    max_dd = results['strategy_drawdown'].min()
    buy_hold_max_dd = results['buy_hold_drawdown'].min()
    
    # Calmar ratio (CAGR / Max DD)
    strategy_calmar = abs(strategy_cagr / max_dd) if max_dd != 0 else 0
    buy_hold_calmar = abs(buy_hold_cagr / buy_hold_max_dd) if buy_hold_max_dd != 0 else 0
    
    # Count trades
    trade_count = results['position_change'].sum()
    trades_per_year = trade_count * (trading_days_per_year / total_days)
    
    # Win rate (days with positive returns)
    win_rate = (results['strategy_return'] > 0).mean()
    
    # Final equity values
    final_strategy_equity = results['strategy_equity'].iloc[-1]
    final_buy_hold_equity = results['buy_hold_equity'].iloc[-1]
    
    # Print results in a clear, tabular format
    print("\n" + "="*50)
    print(" "*15 + "BACKTEST RESULTS")
    print("="*50)
    print(f"Period: {results.index[0].date()} to {results.index[-1].date()} ({total_days} trading days)")
    print(f"Initial Capital: ${initial_capital:.2f}")
    
    print("\n--- Performance Metrics ---")
    print(f"Strategy Final Value:   ${final_strategy_equity:.2f}")
    print(f"Buy & Hold Final Value: ${final_buy_hold_equity:.2f}")
    print(f"Absolute Return:        ${final_strategy_equity - initial_capital:.2f} ({(final_strategy_equity/initial_capital - 1)*100:.1f}%)")
    print(f"Strategy CAGR:          {strategy_cagr:.2%}")
    print(f"Buy & Hold CAGR:        {buy_hold_cagr:.2%}")
    print(f"Outperformance:         {strategy_cagr - buy_hold_cagr:.2%}")
    
    print("\n--- Risk Metrics ---")
    print(f"Strategy Volatility:    {strategy_vol:.2%}")
    print(f"Buy & Hold Volatility:  {buy_hold_vol:.2%}")
    print(f"Strategy Max Drawdown:  {max_dd:.2%}")
    print(f"Buy & Hold Max DD:      {buy_hold_max_dd:.2%}")
    
    print("\n--- Risk-Adjusted Returns ---")
    print(f"Strategy Sharpe:        {strategy_sharpe:.2f}")
    print(f"Buy & Hold Sharpe:      {buy_hold_sharpe:.2f}")
    print(f"Strategy Calmar:        {strategy_calmar:.2f}")
    print(f"Buy & Hold Calmar:      {buy_hold_calmar:.2f}")
    
    print("\n--- Trading Statistics ---")
    print(f"Model Accuracy:         {accuracy:.2%}")
    print(f"Win Rate:               {win_rate:.2%}")
    print(f"Total Trades:           {trade_count}")
    print(f"Trades per Year:        {trades_per_year:.1f}")
    print(f"Avg. Position Size:     {results[results['position'] != 'CASH']['position_weight'].mean():.2%}")
    print(f"Transaction Costs:      {total_cost_per_side*2*100:.1f} bps per round trip")
    
    # Plot results if requested
    if plot:
        try:
            # Plot 1: Equity curves
            plt.figure(figsize=(14, 18))
            
            plt.subplot(3, 1, 1)
            plt.plot(results.index, results['strategy_equity'], label='Tri-Shot Strategy')
            plt.plot(results.index, results['buy_hold_equity'], label='QQQ Buy & Hold')
            plt.title('Equity Curves ($)', fontsize=14)
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot 2: Drawdowns
            plt.subplot(3, 1, 2)
            plt.plot(results.index, results['strategy_drawdown'] * 100, 'r', label='Strategy')
            plt.plot(results.index, results['buy_hold_drawdown'] * 100, 'b--', label='Buy & Hold')
            plt.title('Drawdowns', fontsize=14)
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.legend()
            
            # Plot 3: Position weights over time
            plt.subplot(3, 1, 3)
            
            # Create a colormap for positions
            positions = results['position'].unique()
            cmap = plt.cm.get_cmap('viridis', len(positions))
            colors = {pos: cmap(i) for i, pos in enumerate(positions)}
            
            # Plot position weights
            for ticker in ['TQQQ', 'SQQQ', 'TMF']:
                mask = results['position'] == ticker
                if any(mask):
                    plt.scatter(
                        results.index[mask], 
                        results.loc[mask, 'position_weight'] * 100,
                        label=ticker, 
                        color=colors.get(ticker, 'gray'),
                        alpha=0.7,
                        s=30
                    )
            
            plt.title('Position Weights', fontsize=14)
            plt.ylabel('Weight (%)')
            plt.ylim(0, 105)
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = STATE_DIR / 'backtest_results.png'
            plt.savefig(plot_file)
            print(f"Plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    # Save detailed results
    results_file = STATE_DIR / 'backtest_results.csv'
    results.to_csv(results_file)
    print(f"Detailed results saved to {results_file}")
    
    # Return performance summary dictionary
    return {
        'initial_capital': initial_capital,
        'final_value': final_strategy_equity,
        'absolute_return': final_strategy_equity - initial_capital,
        'return_pct': (final_strategy_equity/initial_capital - 1) * 100,
        'cagr': strategy_cagr,
        'volatility': strategy_vol,
        'sharpe': strategy_sharpe,
        'max_drawdown': max_dd,
        'calmar': strategy_calmar,
        'win_rate': win_rate,
        'trade_count': trade_count,
        'trades_per_year': trades_per_year
    }

def run_monte_carlo_backtest(days, initial_capital, slippage_bps, commission_bps, num_runs=10):
    """Run Monte Carlo backtest with randomized start dates."""
    print(f"Running Monte Carlo backtest with {num_runs} random start dates...")
    
    # Get a list of dates from which to start backtests
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days*2)  # Double the days to have enough range
    
    # Generate random dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    if len(date_range) < num_runs:
        print(f"Warning: Not enough trading days ({len(date_range)}) for {num_runs} runs. Using {len(date_range)} runs instead.")
        num_runs = len(date_range)
    
    # Randomly select start dates
    random_indices = np.random.choice(len(date_range), size=num_runs, replace=False)
    start_dates = [date_range[i].strftime('%Y-%m-%d') for i in random_indices]
    
    # Run backtests with different start dates
    results = []
    for i, start_date in enumerate(start_dates):
        print(f"\nMonte Carlo Run {i+1}/{num_runs} - Starting from {start_date}")
        try:
            result = backtest(
                days=days, 
                plot=(i == 0),  # Only plot the first run
                initial_capital=initial_capital,
                start_date=start_date,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps
            )
            results.append(result)
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            traceback.print_exc()
    
    # Analyze Monte Carlo results
    if results:
        # Convert results to DataFrame
        mc_df = pd.DataFrame(results)
        
        print("\n" + "="*50)
        print(" "*10 + "MONTE CARLO BACKTEST SUMMARY")
        print("="*50)
        
        # Calculate statistics
        print(f"Number of successful runs: {len(results)}/{num_runs}")
        print(f"\nCAGR:")
        print(f"  Mean:   {mc_df['cagr'].mean():.2%}")
        print(f"  Median: {mc_df['cagr'].median():.2%}")
        print(f"  Min:    {mc_df['cagr'].min():.2%}")
        print(f"  Max:    {mc_df['cagr'].max():.2%}")
        print(f"  Std:    {mc_df['cagr'].std():.2%}")
        
        print(f"\nMax Drawdown:")
        print(f"  Mean:   {mc_df['max_drawdown'].mean():.2%}")
        print(f"  Median: {mc_df['max_drawdown'].median():.2%}")
        print(f"  Worst:  {mc_df['max_drawdown'].min():.2%}")
        
        print(f"\nSharpe Ratio:")
        print(f"  Mean:   {mc_df['sharpe'].mean():.2f}")
        print(f"  Median: {mc_df['sharpe'].median():.2f}")
        print(f"  Min:    {mc_df['sharpe'].min():.2f}")
        print(f"  Max:    {mc_df['sharpe'].max():.2f}")
        
        print(f"\nWin Rate:")
        print(f"  Mean:   {mc_df['win_rate'].mean():.2%}")
        print(f"  Median: {mc_df['win_rate'].median():.2%}")
        print(f"  Min:    {mc_df['win_rate'].min():.2%}")
        print(f"  Max:    {mc_df['win_rate'].max():.2%}")
        
        # Plot distribution of key metrics
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.hist(mc_df['cagr'] * 100, bins=10, alpha=0.7)
            plt.axvline(mc_df['cagr'].mean() * 100, color='r', linestyle='--', label=f"Mean: {mc_df['cagr'].mean():.2%}")
            plt.title('CAGR Distribution (%)')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.hist(mc_df['max_drawdown'] * 100, bins=10, alpha=0.7)
            plt.axvline(mc_df['max_drawdown'].mean() * 100, color='r', linestyle='--', label=f"Mean: {mc_df['max_drawdown'].mean():.2%}")
            plt.title('Max Drawdown Distribution (%)')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.hist(mc_df['sharpe'], bins=10, alpha=0.7)
            plt.axvline(mc_df['sharpe'].mean(), color='r', linestyle='--', label=f"Mean: {mc_df['sharpe'].mean():.2f}")
            plt.title('Sharpe Ratio Distribution')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.hist(mc_df['win_rate'] * 100, bins=10, alpha=0.7)
            plt.axvline(mc_df['win_rate'].mean() * 100, color='r', linestyle='--', label=f"Mean: {mc_df['win_rate'].mean():.2%}")
            plt.title('Win Rate Distribution (%)')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = STATE_DIR / 'monte_carlo_results.png'
            plt.savefig(plot_file)
            print(f"Monte Carlo distribution plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error generating Monte Carlo plots: {e}")
        
        return mc_df
    else:
        print("No successful Monte Carlo runs to analyze.")
        return None

def train_walk_forward_model(prices: pd.DataFrame, target_ticker: str = "QQQ", save_model: bool = True):
    """Train a walk-forward model with the given price data."""
    from tri_shot_model import WalkForwardModel

    # Initialize model
    model = WalkForwardModel(STATE_DIR)

    # Train model
    metrics = model.train(prices, target_ticker)

    # Save model if requested
    if save_model:
        model.save()

    return model, metrics

def run_strategy(force=False):
    """Run the tri-shot strategy based on the current day and time."""
    now = dt.datetime.now(TZ)
    day_of_week = now.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
    current_time = now.time()

    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    api = get_alpaca_api()

    if day_of_week == 0 and (current_time >= dt.time(16, 0) or force):  # Monday after 4:00 PM ET
        print("Running Monday strategy...")
        run_monday_strategy(api)
    elif day_of_week == 2 and (current_time >= dt.time(11, 0) or force):  # Wednesday after 11:00 AM ET
        print("Running Wednesday strategy...")
        run_wednesday_strategy(api)
    elif day_of_week == 4 and (current_time >= dt.time(15, 30) or force):  # Friday after 3:30 PM ET
        print("Running Friday strategy...")
        run_friday_strategy(api)
    else:
        if force:
            print("Forcing strategy run despite day/time...")
            if day_of_week == 0:
                run_monday_strategy(api)
            elif day_of_week == 2:
                run_wednesday_strategy(api)
            elif day_of_week == 4:
                run_friday_strategy(api)
            else:
                print("No specific strategy for today. Running Monday strategy by default...")
                run_monday_strategy(api)
        else:
            print(f"No scheduled strategy for day {day_of_week} at time {current_time}")
            print("Use --force to run the strategy anyway")

def setup_paper_trade(initial_capital=500.0, days=30):
    """
    Set up paper trading for the Tri-Shot strategy on Alpaca.
    
    Args:
        initial_capital: Initial capital for paper trading
        days: Number of days to run the paper trading simulation
    """
    import json
    
    # Check if Alpaca API keys are set
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        print("ERROR: Alpaca API keys not found in environment variables.")
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        return
    
    # Initialize Alpaca API client (paper trading)
    api = get_alpaca_api(paper=True)
    
    # Check if account exists and is set up for paper trading
    try:
        account = api.get_account()
        print(f"Connected to Alpaca paper trading account: {account.id}")
        print(f"Current account status: {account.status}")
        
        # Check if account needs to be reset to initial capital
        current_equity = float(account.equity)
        print(f"Current paper account equity: ${current_equity:.2f}")
        
        if abs(current_equity - initial_capital) > 1.0:
            print(f"WARNING: Paper account equity (${current_equity:.2f}) differs from desired initial capital (${initial_capital:.2f}).")
            print("Consider resetting your paper account in the Alpaca dashboard to match your desired initial capital.")
    
    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return
    
    # Set up the paper trading configuration
    end_date = dt.datetime.now() + dt.timedelta(days=days)
    
    # Create paper trading configuration file
    config = {
        "strategy": "tri_shot",
        "initial_capital": initial_capital,
        "start_date": dt.datetime.now().strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "max_position_size": 1.0,  # 100% of account for small accounts
        "slippage_model": "realistic",  # Use realistic slippage model
        "pdt_rule_enforced": True,  # Enforce Pattern Day Trading rule
        "paper_trading": True
    }
    
    # Save configuration
    config_file = STATE_DIR / "paper_trading_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"\nPaper trading configuration saved to {config_file}")
    print(f"Paper trading will run until {end_date.strftime('%Y-%m-%d')}")
    print("\nTo start paper trading, run:")
    print("python tri_shot_cli.py run --paper")
    
    # Create a sample paper trading script
    paper_script = """#!/usr/bin/env python3
import os
import json
import time
import datetime as dt
from pathlib import Path

# Import Tri-Shot modules
from tri_shot import STATE_DIR, get_alpaca_api
from tri_shot_model import load_walk_forward_model
import tri_shot_features as tsf

def run_paper_trading():
    \"\"\"Run the Tri-Shot strategy in paper trading mode.\"\"\"
    # Load configuration
    config_file = STATE_DIR / "paper_trading_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Initialize Alpaca API
    api = get_alpaca_api(paper=True)
    
    # Load the model
    model = load_walk_forward_model(STATE_DIR)
    if model is None:
        print("ERROR: Failed to load model. Please train the model first.")
        return
    
    # Check if market is open
    clock = api.get_clock()
    if not clock.is_open:
        next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Market is closed. Next market open: {next_open}")
        return
    
    # Get current positions
    positions = api.list_positions()
    current_position = None
    if positions:
        current_position = positions[0].symbol
        print(f"Current position: {current_position} ({positions[0].qty} shares)")
    else:
        print("No current positions")
    
    # Get account info
    account = api.get_account()
    buying_power = float(account.buying_power)
    print(f"Account buying power: ${buying_power:.2f}")
    
    # Check PDT rule
    day_trades = int(account.daytrade_count)
    print(f"Day trades in last 5 days: {day_trades}/3")
    
    # Get market data for prediction
    tickers = ['QQQ', 'TQQQ', 'SQQQ', 'TMF', 'TLT', '^VIX']
    prices = tsf.fetch_data(tickers, days=30)
    
    # Generate features and prediction
    X, _ = tsf.make_feature_matrix(prices)
    probability = model.predict(X.iloc[-1:]).item()
    signal_strength = abs(probability - 0.5) / 0.5
    
    # Get market regime
    bull_market = prices['QQQ'].iloc[-1] > prices['QQQ'].rolling(200).mean().iloc[-1]
    price_momentum = prices['QQQ'].pct_change(5).iloc[-1]
    
    # Determine position
    new_position = 'CASH'
    
    # Long condition
    if probability >= 0.52 and price_momentum > 0:
        new_position = 'TQQQ'
    # Short condition
    elif probability <= 0.48 and price_momentum < 0:
        new_position = 'SQQQ'
    # Bond condition
    elif 0.48 < probability < 0.52 and prices['TLT'].pct_change(20).iloc[-1] > 0.01:
        new_position = 'TMF'
    
    print(f"Model probability: {probability:.4f} (signal strength: {signal_strength:.2f})")
    print(f"Market regime: {'Bullish' if bull_market else 'Bearish'}")
    print(f"Price momentum: {price_momentum:.2%}")
    print(f"Recommended position: {new_position}")
    
    # Check if we need to change position
    if new_position != current_position:
        # Check PDT rule
        if day_trades >= 3 and current_position is not None:
            print("WARNING: Day trade limit reached. Cannot change position.")
            return
        
        # Close current position if any
        if current_position:
            print(f"Closing position: {current_position}")
            api.close_position(current_position)
        
        # Open new position if not CASH
        if new_position != 'CASH' and buying_power > 0:
            # Calculate position size
            position_size = 1.0  # 100% for small accounts
            if signal_strength > 0.25:
                position_size = 1.0  # Full account for high conviction
            elif signal_strength > 0.15:
                position_size = 0.8  # 80% for medium conviction
            else:
                position_size = 0.6  # 60% for lower conviction
            
            # Calculate dollar amount
            amount = buying_power * position_size
            
            print(f"Opening position: {new_position} (${amount:.2f}, {position_size*100:.0f}% of account)")
            api.submit_order(
                symbol=new_position,
                qty=None,
                notional=amount,
                side='buy',
                type='market',
                time_in_force='day'
            )
    else:
        print("No position change needed")
    
    print("Paper trading execution complete")

if __name__ == "__main__":
    run_paper_trading()
"""
    
    # Save the paper trading script
    script_file = STATE_DIR / "run_paper_trading.py"
    with open(script_file, "w") as f:
        f.write(paper_script)
    
    print(f"\nPaper trading script created at {script_file}")
    print("You can also run this script directly with:")
    print(f"python {script_file}")
    
    return config

def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description='Tri-Shot Trading Strategy CLI')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the strategy now')
    run_parser.add_argument('--force', action='store_true', help='Force run even if not the right day/time')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--days', type=int, default=365, help='Number of days to backtest')
    backtest_parser.add_argument('--plot', action='store_true', help='Plot backtest results')
    backtest_parser.add_argument('--initial_capital', type=float, default=500.0, help='Initial capital')
    backtest_parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--slippage_bps', type=float, default=2.0, help='Slippage in basis points (bps)')
    backtest_parser.add_argument('--commission_bps', type=float, default=1.0, help='Commission in basis points (bps)')
    backtest_parser.add_argument('--monte_carlo', action='store_true', help='Run Monte Carlo simulation')
    backtest_parser.add_argument('--mc_runs', type=int, default=10, help='Number of Monte Carlo runs')
    
    # Paper trade command
    paper_parser = subparsers.add_parser('paper', help='Set up paper trading')
    paper_parser.add_argument('--initial_capital', type=float, default=500.0, help='Initial capital for paper trading')
    paper_parser.add_argument('--days', type=int, default=30, help='Number of days to run paper trading')
    
    # DMT backtest command (new)
    if HAS_DMT_DEPS:
        dmt_parser = subparsers.add_parser('dmt', help='Run differentiable market twin backtest')
        dmt_parser.add_argument('--initial_capital', type=float, default=500.0, help='Initial capital')
        dmt_parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)')
        dmt_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for optimization')
        dmt_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for optimization')
        dmt_parser.add_argument('--cpu', action='store_true', help='Force CPU computation (default uses GPU if available)')
    
    args = parser.parse_args()
    
    # Ensure environment is set up
    if not setup_environment():
        return
    
    # Run the appropriate command
    if args.command == 'train':
        train_model(force=args.force)
    elif args.command == 'run':
        run_strategy(force=args.force)
    elif args.command == 'backtest':
        backtest(days=args.days, 
                plot=args.plot, 
                initial_capital=args.initial_capital,
                start_date=args.start_date,
                slippage_bps=args.slippage_bps,
                commission_bps=args.commission_bps,
                monte_carlo=args.monte_carlo,
                mc_runs=args.mc_runs)
    elif args.command == 'paper':
        setup_paper_trade(initial_capital=args.initial_capital, days=args.days)
    elif args.command == 'dmt' and HAS_DMT_DEPS:
        run_dmt_command(args)
    else:
        parser.print_help()

def run_dmt_command(args):
    """Run the DMT backtest command."""
    if not HAS_DMT_DEPS:
        print("ERROR: PyTorch and other DMT dependencies not found. Install with:")
        print("pip install torch>=2.0.0")
        return
        
    print("Running Differentiable Market Twin (DMT) backtest...")
    
    # Set device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define tickers
    TICKERS = {
        "UP": "TQQQ",    # 3x long QQQ
        "DN": "SQQQ",    # 3x short QQQ
        "BOND": "TMF",   # 3x long treasury
        "CASH": "BIL",   # Short-term treasury ETF (cash equivalent)
        "SRC": "QQQ",    # Base asset to track
        "VIX": "^VIX"    # Volatility index
    }
    
    # Fetch data
    tickers = list(TICKERS.values())
    additional_tickers = ['TLT', 'UUP']
    for ticker in additional_tickers:
        if ticker not in tickers:
            tickers.append(ticker)
    
    if args.start_date:
        # Convert start_date string to datetime
        start_dt = dt.datetime.strptime(args.start_date, '%Y-%m-%d')
        print(f"Fetching data from {args.start_date} to present...")
        prices = tsf.fetch_data_from_date(tickers, start_date=start_dt)
    else:
        # Default to 2 years of data for DMT
        days = 730
        print(f"Fetching data for the last {days} days...")
        prices = tsf.fetch_data(tickers, days=days)
    
    # Run the simplified DMT backtest
    results = run_dmt_backtest(
        prices,
        initial_capital=args.initial_capital,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    print("DMT backtest complete!")
    
    # Optional traditional backtest for comparison 
    if args.start_date:
        print("\nRunning traditional backtest for comparison...")
        backtest(
            start_date=args.start_date,
            initial_capital=args.initial_capital,
            slippage_bps=2.0,
            commission_bps=1.0,
            plot=True
        )
    
    print("\nBacktest comparison complete! Check the plots in the tri_shot_data directory.")

if __name__ == '__main__':
    main()
