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

def setup_environment():
    """Ensure the environment is properly set up."""
    ensure_state_dir()

    # Check API keys only if we're not running a backtest
    if sys.argv[1] != "backtest":
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

def backtest(days=365, plot=False, initial_capital=500.0):
    """Run a comprehensive backtest on historical data with realistic execution."""
    print(f"Running enhanced backtest over the last {days} days with ${initial_capital:.2f} initial capital...")

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
    prices = tsf.fetch_data(tickers, days=lookback_window)

    # Train walk-forward model or load existing one
    model_file = STATE_DIR / "tri_shot_ensemble.pkl"
    if model_file.exists() and days <= 400:  # Use existing model for short backtests
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
                results.loc[mask, f'{ticker}_weight'] = 0.5  # More conservative
    
    # Apply weights to calculate returns
    results['position_weight'] = 0.0
    for ticker in ['TQQQ', 'SQQQ', 'TMF']:
        results.loc[results['position'] == ticker, 'position_weight'] = results.loc[results['position'] == ticker, f'{ticker}_weight']
    
    # Calculate returns with transaction costs
    TRANSACTION_COST = 0.0005  # 5bps per trade, one-way
    
    # Calculate daily returns for each asset
    for ticker in ['QQQ', 'TQQQ', 'SQQQ', 'TMF']:
        if ticker in results.columns:
            results[f'{ticker}_return'] = results[ticker].pct_change()
    
    # Calculate strategy returns
    results['strategy_return'] = 0.0
    
    # Apply transaction costs when position changes
    results['transaction_cost'] = 0.0
    results.loc[results['position_change'], 'transaction_cost'] = TRANSACTION_COST
    
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

def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description='Tri-Shot Strategy CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the XGBoost model')
    train_parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')

    # Run strategy command
    run_parser = subparsers.add_parser('run', help='Run the tri-shot strategy')
    run_parser.add_argument('--force', action='store_true', help='Force strategy execution regardless of day/time')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest on historical data')
    backtest_parser.add_argument('--days', type=int, default=365, help='Number of days to backtest')
    backtest_parser.add_argument('--plot', action='store_true', help='Generate plots of backtest results')
    backtest_parser.add_argument('--initial_capital', type=float, default=500.0, help='Initial capital for backtest')

    args = parser.parse_args()

    try:
        if not setup_environment():
            return

        if args.command == 'train':
            train_model(args.force)
        elif args.command == 'run':
            run_strategy(args.force)
        elif args.command == 'backtest':
            backtest(args.days, args.plot, args.initial_capital)
        else:
            parser.print_help()

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
