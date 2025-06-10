#!/usr/bin/env python3
"""
Quick Bitcoin Walk-Forward Validation
===================================
Streamlined version for faster results on real BTC 3-minute data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingEnvironment, TradingConfig
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, TurboDMTEnsemble
from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import AdvancedFeatureGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def load_btc_data():
    """Load and prepare Bitcoin data"""
    print("📊 Loading Bitcoin 3-minute data...")
    
    df = pd.read_csv("/Users/zachrizzo/Desktop/programming/ai-stock-algo-03/data/BTCUSDT_3m_20250101_20250430.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Rename columns
    df = df.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low', 
        'close': 'Close',
        'volume': 'Volume'
    })
    df['close'] = df['Close']
    
    # Clean data
    df = df.dropna()
    df = df[df['Volume'] > 0]
    
    # Remove obvious errors
    for col in ['Open', 'High', 'Low', 'Close']:
        df = df[(df[col] > 1000) & (df[col] < 500000)]
    
    print(f"✅ Loaded {len(df)} candles from {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
    
    return df


def create_optimized_strategy():
    """Create optimized strategy configuration for speed"""
    transformer_config = TurboDMTConfig(
        feature_dim=25,  # Will be adjusted
        seq_len=15,      # Shorter sequence
        ensemble_size=1, # Single model for speed
        hidden_dim=128,  # Smaller model
        transformer_dim=96,
        transformer_layers=2,
        attention_heads=8,
        max_position_size=1.0,
        target_vol=0.30,
        neutral_zone=0.02
    )
    
    trading_config = TradingConfig(
        initial_balance=10000,
        max_position_size=0.95,
        transaction_cost=0.001,  # Binance fee
        slippage=0.0005
    )
    
    return transformer_config, trading_config


def run_single_test(train_data, test_data, window_idx):
    """Run a single training/testing cycle"""
    print(f"\n🔄 Window {window_idx}")
    print(f"📈 Train: {len(train_data)} candles ({train_data['Date'].iloc[0]} to {train_data['Date'].iloc[-1]})")
    print(f"🧪 Test: {len(test_data)} candles ({test_data['Date'].iloc[0]} to {test_data['Date'].iloc[-1]})")
    
    try:
        # Create simplified feature extractor
        feature_gen = AdvancedFeatureGenerator(
            use_spectral=False,
            use_multi_timeframe=False,
            use_market_regime=False,
            use_volatility_surface=False,
            use_orderflow=False,
            standardize=True
        )
        
        # Get feature dimensions
        sample_features = feature_gen.generate_features(train_data.iloc[-50:])
        feature_dim = sample_features.shape[1] - 1
        
        # Create strategy
        transformer_config, trading_config = create_optimized_strategy()
        transformer_config.feature_dim = feature_dim
        
        transformer = TurboDMTEnsemble(transformer_config)
        
        # Create and train agent
        def make_env():
            return TradingEnvironment(
                price_data=train_data,
                transformer_model=transformer,
                config=trading_config,
                training_mode=True
            )
        
        train_env = DummyVecEnv([make_env])
        agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            n_steps=128,         # Smaller buffer
            batch_size=32,
            n_epochs=5,          # Fewer epochs
            verbose=0
        )
        
        # Quick training
        start_time = datetime.now()
        agent.learn(total_timesteps=1500)  # Reduced timesteps
        training_time = datetime.now() - start_time
        
        # Test on out-of-sample data
        test_env = TradingEnvironment(
            price_data=test_data,
            transformer_model=transformer,
            config=trading_config,
            training_mode=False
        )
        
        obs, info = test_env.reset()
        done = False
        actions = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            actions.append(int(action))
        
        # Get results
        stats = test_env.get_episode_stats()
        bh_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1
        
        result = {
            'window': window_idx,
            'training_time': training_time.total_seconds(),
            'strategy_return': stats.get('total_return', 0),
            'benchmark_return': bh_return,
            'excess_return': stats.get('total_return', 0) - bh_return,
            'sharpe_ratio': stats.get('sharpe_ratio', 0),
            'max_drawdown': stats.get('max_drawdown', 0),
            'total_trades': stats.get('total_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'final_value': stats.get('final_portfolio_value', 0),
            'actions': actions
        }
        
        print(f"✅ Return: {result['strategy_return']*100:+.2f}% vs BH: {bh_return*100:+.2f}% | "
              f"Sharpe: {result['sharpe_ratio']:.2f} | DD: {result['max_drawdown']*100:.1f}% | "
              f"Trades: {result['total_trades']} | Time: {training_time}")
        
        return result
        
    except Exception as e:
        print(f"❌ Window {window_idx} failed: {e}")
        return {'window': window_idx, 'error': str(e)}


def main():
    """Main execution function"""
    print("🚀 Quick Bitcoin Walk-Forward Validation")
    print("=" * 50)
    
    # Load data
    data = load_btc_data()
    
    # Configuration for quick testing
    train_hours = 48      # 2 days (48 * 20 = 960 candles)
    test_hours = 12       # 12 hours (12 * 20 = 240 candles)
    step_hours = 6        # 6 hour steps (6 * 20 = 120 candles)
    
    train_candles = train_hours * 20
    test_candles = test_hours * 20
    step_candles = step_hours * 20
    
    print(f"\n📊 Configuration:")
    print(f"  Training: {train_hours}h ({train_candles} candles)")
    print(f"  Testing: {test_hours}h ({test_candles} candles)")
    print(f"  Step: {step_hours}h ({step_candles} candles)")
    
    # Calculate windows
    total_candles = len(data)
    max_windows = (total_candles - train_candles - test_candles) // step_candles
    num_windows = min(max_windows, 5)  # Limit to 5 windows for quick demo
    
    print(f"  Will test {num_windows} windows")
    
    # Run walk-forward validation
    results = []
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_candles
        train_end_idx = start_idx + train_candles
        test_end_idx = train_end_idx + test_candles
        
        if test_end_idx >= len(data):
            break
        
        train_data = data.iloc[start_idx:train_end_idx].copy()
        test_data = data.iloc[train_end_idx:test_end_idx].copy()
        
        result = run_single_test(train_data, test_data, window_idx)
        results.append(result)
    
    # Analyze results
    print("\n📈 RESULTS SUMMARY")
    print("=" * 30)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        strategy_returns = [r['strategy_return'] for r in valid_results]
        benchmark_returns = [r['benchmark_return'] for r in valid_results]
        excess_returns = [r['excess_return'] for r in valid_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in valid_results]
        max_drawdowns = [r['max_drawdown'] for r in valid_results]
        
        print(f"Windows tested: {len(valid_results)}")
        print(f"Avg strategy return: {np.mean(strategy_returns)*100:+.2f}% ± {np.std(strategy_returns)*100:.2f}%")
        print(f"Avg benchmark return: {np.mean(benchmark_returns)*100:+.2f}%")
        print(f"Avg excess return: {np.mean(excess_returns)*100:+.2f}%")
        print(f"Avg Sharpe ratio: {np.mean(sharpe_ratios):.3f}")
        print(f"Avg max drawdown: {np.mean(max_drawdowns)*100:.2f}%")
        print(f"Win rate: {np.mean([1 if r > 0 else 0 for r in strategy_returns])*100:.1f}%")
        print(f"Best return: {np.max(strategy_returns)*100:+.2f}%")
        print(f"Worst return: {np.min(strategy_returns)*100:+.2f}%")
        
        # Create simple visualization
        if len(valid_results) > 1:
            print("\n📊 Creating performance chart...")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Bitcoin 3-Min Trading Results', fontsize=14)
            
            # Returns comparison
            windows = [r['window'] for r in valid_results]
            ax1.plot(windows, [r*100 for r in strategy_returns], 'bo-', label='Strategy')
            ax1.plot(windows, [r*100 for r in benchmark_returns], 'ro-', label='Buy & Hold')
            ax1.set_title('Returns by Window')
            ax1.set_ylabel('Return (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Excess returns
            ax2.bar(windows, [r*100 for r in excess_returns], alpha=0.7, 
                   color=['green' if r > 0 else 'red' for r in excess_returns])
            ax2.set_title('Excess Returns')
            ax2.set_ylabel('Excess Return (%)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # Sharpe ratios
            ax3.bar(windows, sharpe_ratios, alpha=0.7)
            ax3.set_title('Sharpe Ratios')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
            
            # Drawdowns
            ax4.bar(windows, [r*100 for r in max_drawdowns], alpha=0.7, color='red')
            ax4.set_title('Max Drawdowns')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_quick_validation_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Chart saved as '{filename}'")
            plt.show()
        
        # Action analysis
        all_actions = []
        for r in valid_results:
            all_actions.extend(r.get('actions', []))
        
        if all_actions:
            action_counts = np.bincount(all_actions, minlength=9)
            action_names = ["Hold", "Buy 25%", "Buy 50%", "Buy 75%", "Buy 100%",
                           "Sell 25%", "Sell 50%", "Sell 75%", "Sell 100%"]
            
            print(f"\n🎯 Action Distribution:")
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                pct = count / len(all_actions) * 100 if all_actions else 0
                print(f"  {name}: {count} ({pct:.1f}%)")
        
        print(f"\n🎉 Bitcoin walk-forward validation completed!")
        print(f"✅ The Transformer+RL strategy {'outperformed' if np.mean(excess_returns) > 0 else 'underperformed'} buy-and-hold")
        
    else:
        print("❌ No valid results generated")
    
    failed_count = len([r for r in results if 'error' in r])
    if failed_count > 0:
        print(f"⚠️  {failed_count} windows failed")


if __name__ == "__main__":
    main()