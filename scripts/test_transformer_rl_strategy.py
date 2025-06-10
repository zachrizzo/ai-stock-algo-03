#!/usr/bin/env python3
"""
Test Script for Transformer+RL Strategy with Sliding Window Validation
=====================================================================
Comprehensive testing of the new Transformer+RL architecture using
accelerated simulation and walk-forward validation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional, will use matplotlib style instead
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our strategy components
from stock_trader_o3_algo.strategies.turbo_dmt_v2.transformer_rl_strategy import (
    TransformerRLStrategy, TransformerRLConfig
)
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig
from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingConfig

# Data fetching
import yfinance as yf


def download_market_data(symbols: list = ["SPY", "QQQ", "MSFT"], 
                        start_date: str = "2018-01-01", 
                        end_date: str = "2024-01-01") -> dict:
    """Download market data for testing"""
    print(f"Downloading data for {symbols} from {start_date} to {end_date}...")
    
    data_dict = {}
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            data = data.reset_index()
            data_dict[symbol] = data
            print(f"Downloaded {len(data)} periods for {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    
    return data_dict


def create_test_configurations() -> list:
    """Create different configurations for testing"""
    configs = []
    
    # Conservative configuration
    conservative_config = TransformerRLConfig()
    conservative_config.transformer_config = TurboDMTConfig(
        max_position_size=1.0,
        target_vol=0.12,
        neutral_zone=0.03
    )
    conservative_config.trading_config = TradingConfig(
        max_position_size=0.8,
        transaction_cost=0.001
    )
    conservative_config.total_timesteps = 20000
    configs.append(("Conservative", conservative_config))
    
    # Aggressive configuration
    aggressive_config = TransformerRLConfig()
    aggressive_config.transformer_config = TurboDMTConfig(
        max_position_size=2.0,
        target_vol=0.25,
        neutral_zone=0.015
    )
    aggressive_config.trading_config = TradingConfig(
        max_position_size=1.0,
        transaction_cost=0.001
    )
    aggressive_config.total_timesteps = 25000
    configs.append(("Aggressive", aggressive_config))
    
    # Balanced configuration
    balanced_config = TransformerRLConfig()
    balanced_config.transformer_config = TurboDMTConfig(
        max_position_size=1.5,
        target_vol=0.18,
        neutral_zone=0.02
    )
    balanced_config.trading_config = TradingConfig(
        max_position_size=0.95,
        transaction_cost=0.001
    )
    balanced_config.total_timesteps = 30000
    configs.append(("Balanced", balanced_config))
    
    return configs


def test_single_strategy(name: str, config: TransformerRLConfig, data: pd.DataFrame) -> dict:
    """Test a single strategy configuration"""
    print(f"\n{'='*50}")
    print(f"Testing {name} Configuration")
    print(f"{'='*50}")
    
    # Split data (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"Training period: {train_data.iloc[0]['Date']} to {train_data.iloc[-1]['Date']}")
    print(f"Testing period: {test_data.iloc[0]['Date']} to {test_data.iloc[-1]['Date']}")
    
    # Create and train strategy
    strategy = TransformerRLStrategy(config)
    
    print(f"Training strategy with {config.total_timesteps} timesteps...")
    start_time = datetime.now()
    
    try:
        training_history = strategy.train(
            training_data=train_data,
            validation_data=None,  # Use built-in validation split
            pretrain_transformer=True,
            verbose=True
        )
        
        training_time = datetime.now() - start_time
        print(f"Training completed in {training_time}")
        
        # Backtest on out-of-sample data
        print("Running out-of-sample backtest...")
        backtest_results = strategy.backtest(test_data, initial_balance=10000.0)
        
        # Calculate additional metrics
        portfolio_values = np.array(backtest_results['portfolio_history'])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Information ratio vs buy-and-hold
        bh_return = (test_data.iloc[-1]['Close'] / test_data.iloc[0]['Close']) - 1
        strategy_return = backtest_results['total_return']
        excess_return = strategy_return - bh_return
        tracking_error = np.std(returns) * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Compile results
        results = {
            'name': name,
            'training_time': training_time.total_seconds(),
            'backtest_results': backtest_results,
            'buy_hold_return': bh_return,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'training_history': training_history
        }
        
        # Print summary
        print(f"\n{name} Strategy Results:")
        print(f"Total Return: {strategy_return*100:.2f}% vs Buy&Hold: {bh_return*100:.2f}%")
        print(f"Excess Return: {excess_return*100:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"Information Ratio: {information_ratio:.3f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"Total Trades: {backtest_results['total_trades']}")
        
        return results
        
    except Exception as e:
        print(f"Error testing {name} configuration: {e}")
        return {
            'name': name,
            'error': str(e),
            'training_time': 0,
            'backtest_results': {}
        }


def run_walk_forward_test(symbol: str, data: pd.DataFrame) -> dict:
    """Run comprehensive walk-forward validation"""
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation for {symbol}")
    print(f"{'='*60}")
    
    # Use balanced configuration for walk-forward test
    config = TransformerRLConfig()
    config.total_timesteps = 15000  # Reduced for faster testing
    config.train_window = 252  # 1 year
    config.test_window = 63   # 3 months
    config.step_size = 21     # 1 month
    
    strategy = TransformerRLStrategy(config)
    
    print(f"Running walk-forward test with:")
    print(f"- Train window: {config.train_window} periods")
    print(f"- Test window: {config.test_window} periods") 
    print(f"- Step size: {config.step_size} periods")
    
    start_time = datetime.now()
    
    try:
        wf_results = strategy.walk_forward_test(
            data=data,
            training_epochs=20,  # Reduced for speed
            verbose=True
        )
        
        walk_forward_time = datetime.now() - start_time
        print(f"Walk-forward test completed in {walk_forward_time}")
        
        # Print aggregate results
        agg_stats = wf_results['aggregate_stats']
        print(f"\nWalk-Forward Aggregate Results:")
        print(f"Number of windows: {agg_stats['num_windows']}")
        print(f"Mean return: {agg_stats['mean_return']*100:.2f}% ± {agg_stats['std_return']*100:.2f}%")
        print(f"Mean Sharpe: {agg_stats['mean_sharpe']:.3f} ± {agg_stats['std_sharpe']:.3f}")
        print(f"Win rate: {agg_stats['win_rate']*100:.1f}%")
        print(f"Worst drawdown: {agg_stats['worst_drawdown']*100:.2f}%")
        
        wf_results['walk_forward_time'] = walk_forward_time.total_seconds()
        return wf_results
        
    except Exception as e:
        print(f"Error in walk-forward test: {e}")
        return {'error': str(e)}


def create_performance_plots(results: list, symbol: str):
    """Create performance visualization plots"""
    print(f"\nGenerating performance plots for {symbol}...")
    
    # Set up the plotting style
    plt.style.use('default')  # Use default matplotlib style
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Transformer+RL Strategy Performance Analysis - {symbol}', fontsize=16)
    
    # Plot 1: Returns comparison
    ax1 = axes[0, 0]
    strategy_returns = [r['backtest_results']['total_return'] * 100 for r in results if 'backtest_results' in r]
    bh_returns = [r['buy_hold_return'] * 100 for r in results if 'buy_hold_return' in r]
    names = [r['name'] for r in results if 'backtest_results' in r]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, strategy_returns, width, label='Strategy', alpha=0.8)
    ax1.bar(x + width/2, bh_returns, width, label='Buy & Hold', alpha=0.8)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Total Returns Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-adjusted metrics
    ax2 = axes[0, 1]
    sharpe_ratios = [r['backtest_results']['sharpe_ratio'] for r in results if 'backtest_results' in r]
    info_ratios = [r['information_ratio'] for r in results if 'information_ratio' in r]
    
    x = np.arange(len(names))
    ax2.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8)
    ax2.bar(x + width/2, info_ratios, width, label='Information Ratio', alpha=0.8)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Risk-Adjusted Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown and volatility
    ax3 = axes[1, 0]
    max_drawdowns = [r['backtest_results']['max_drawdown'] * 100 for r in results if 'backtest_results' in r]
    volatilities = [r['backtest_results']['volatility'] * 100 for r in results if 'backtest_results' in r]
    
    ax3.bar(x - width/2, max_drawdowns, width, label='Max Drawdown (%)', alpha=0.8, color='red')
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, volatilities, width, label='Volatility (%)', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Max Drawdown (%)', color='red')
    ax3_twin.set_ylabel('Volatility (%)', color='orange')
    ax3.set_title('Risk Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trading activity
    ax4 = axes[1, 1]
    total_trades = [r['backtest_results']['total_trades'] for r in results if 'backtest_results' in r]
    win_rates = [r['backtest_results']['win_rate'] * 100 for r in results if 'backtest_results' in r]
    
    ax4.bar(x - width/2, total_trades, width, label='Total Trades', alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + width/2, win_rates, width, label='Win Rate (%)', alpha=0.8, color='green')
    
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Total Trades')
    ax4_twin.set_ylabel('Win Rate (%)', color='green')
    ax4.set_title('Trading Activity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transformer_rl_performance_{symbol}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved as {filename}")
    
    plt.show()


def main():
    """Main testing function"""
    print("Transformer+RL Strategy Comprehensive Testing")
    print("=" * 50)
    
    # Download test data
    symbols = ["SPY", "QQQ"]  # Reduced for faster testing
    data_dict = download_market_data(symbols, "2019-01-01", "2024-01-01")
    
    if not data_dict:
        print("No data downloaded. Exiting.")
        return
    
    # Get test configurations
    configs = create_test_configurations()
    
    all_results = {}
    
    # Test each symbol
    for symbol, data in data_dict.items():
        print(f"\n\nTesting on {symbol} ({len(data)} periods)")
        
        symbol_results = []
        
        # Test each configuration
        for config_name, config in configs:
            result = test_single_strategy(config_name, config, data)
            symbol_results.append(result)
        
        all_results[symbol] = symbol_results
        
        # Create performance plots
        if symbol_results:
            create_performance_plots(symbol_results, symbol)
        
        # Run walk-forward test on first symbol only (time intensive)
        if symbol == list(data_dict.keys())[0]:
            wf_results = run_walk_forward_test(symbol, data)
            all_results[f"{symbol}_walk_forward"] = wf_results
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for symbol, results in all_results.items():
        if symbol.endswith('_walk_forward'):
            continue
            
        print(f"\n{symbol} Results:")
        best_result = None
        best_sharpe = -999
        
        for result in results:
            if 'backtest_results' in result:
                sharpe = result['backtest_results']['sharpe_ratio']
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
        
        if best_result:
            print(f"Best configuration: {best_result['name']}")
            print(f"Sharpe ratio: {best_result['backtest_results']['sharpe_ratio']:.3f}")
            print(f"Total return: {best_result['backtest_results']['total_return']*100:.2f}%")
            print(f"Max drawdown: {best_result['backtest_results']['max_drawdown']*100:.2f}%")
    
    print(f"\nTesting completed at {datetime.now()}")


if __name__ == "__main__":
    main()