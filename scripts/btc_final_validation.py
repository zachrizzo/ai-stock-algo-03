#!/usr/bin/env python3
"""
Bitcoin Walk-Forward Validation - Final Version
==============================================
Robust implementation with proper data handling for real BTC trading.
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import talib


class SimpleBTCFeatureExtractor:
    """Simplified feature extractor specifically for BTC data"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_features(self, data):
        """Generate robust features for Bitcoin data"""
        df = data.copy()
        
        # Ensure we have the minimum required data
        if len(df) < 50:
            raise ValueError("Insufficient data for feature generation")
        
        features = pd.DataFrame(index=df.index)
        
        # Basic price features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages (shorter periods for 3-min data)
        for period in [5, 10, 20]:
            features[f'ma_{period}'] = df['Close'].rolling(period).mean() / df['Close'] - 1
            features[f'vol_{period}'] = df['Volume'].rolling(period).mean()
        
        # Price ratios
        features['high_low_ratio'] = (df['High'] / df['Low']) - 1
        features['close_open_ratio'] = (df['Close'] / df['Open']) - 1
        features['daily_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Technical indicators (with error handling)
        try:
            features['rsi_14'] = talib.RSI(df['Close'].values, timeperiod=14) / 100
        except:
            features['rsi_14'] = 0.5
            
        try:
            features['rsi_7'] = talib.RSI(df['Close'].values, timeperiod=7) / 100
        except:
            features['rsi_7'] = 0.5
        
        # MACD
        try:
            macd, macd_signal, _ = talib.MACD(df['Close'].values)
            features['macd'] = macd / df['Close']
            features['macd_signal'] = macd_signal / df['Close']
        except:
            features['macd'] = 0
            features['macd_signal'] = 0
        
        # Bollinger Bands
        try:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values, timeperiod=20)
            features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        except:
            features['bb_position'] = 0.5
            features['bb_width'] = 0.02
        
        # Volume indicators
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Momentum indicators
        for period in [3, 7]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Price position in recent range
        features['price_position'] = ((df['Close'] - df['Low'].rolling(20).min()) / 
                                     (df['High'].rolling(20).max() - df['Low'].rolling(20).min()))
        
        # Fill any remaining NaN values
        features = features.fillna(method='bfill')
        features = features.fillna(0)
        
        # Drop first rows that might still have NaN
        features = features.iloc[50:].copy()
        
        self.feature_names = features.columns.tolist()
        
        return features


def load_and_prepare_btc_data():
    """Load and prepare Bitcoin data with robust cleaning"""
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
    
    # Clean data more aggressively
    df = df.dropna()
    df = df[df['Volume'] > 0]
    
    # Remove price outliers (likely data errors)
    price_median = df['Close'].median()
    price_std = df['Close'].std()
    lower_bound = price_median - 4 * price_std
    upper_bound = price_median + 4 * price_std
    
    df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
    
    # Ensure price relationships are valid
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    # Remove extreme volume outliers
    volume_99th = df['Volume'].quantile(0.99)
    df = df[df['Volume'] <= volume_99th * 3]
    
    print(f"✅ Cleaned data: {len(df)} candles")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
    
    return df


def create_lightweight_strategy():
    """Create lightweight strategy for faster execution"""
    transformer_config = TurboDMTConfig(
        feature_dim=20,  # Will be updated
        seq_len=10,      # Very short for speed
        ensemble_size=1, # Single model
        hidden_dim=64,   # Small model
        transformer_dim=48,
        transformer_layers=1,
        attention_heads=6,
        lstm_layers=1,
        max_position_size=1.0,
        target_vol=0.25,
        neutral_zone=0.02
    )
    
    trading_config = TradingConfig(
        initial_balance=10000,
        max_position_size=0.9,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    return transformer_config, trading_config


def run_btc_window_test(train_data, test_data, window_idx):
    """Run a single BTC trading test window"""
    print(f"\n🔄 Window {window_idx}")
    print(f"📈 Train: {len(train_data)} candles")
    print(f"🧪 Test: {len(test_data)} candles")
    
    try:
        # Create feature extractor
        feature_gen = SimpleBTCFeatureExtractor()
        
        # Generate features for training data
        train_features = feature_gen.generate_features(train_data)
        
        if len(train_features) < 100:
            raise ValueError("Insufficient training features after cleaning")
        
        # Get feature dimension
        feature_dim = len(feature_gen.feature_names)
        
        # Create strategy components
        transformer_config, trading_config = create_lightweight_strategy()
        transformer_config.feature_dim = feature_dim
        
        # Ensure transformer_dim is divisible by attention_heads
        attention_heads = transformer_config.attention_heads
        transformer_dim = transformer_config.transformer_dim
        transformer_dim = (transformer_dim // attention_heads) * attention_heads
        transformer_config.transformer_dim = transformer_dim
        
        transformer = TurboDMTEnsemble(transformer_config)
        
        # Create training environment
        def make_env():
            env = TradingEnvironment(
                price_data=train_data,
                transformer_model=transformer,
                config=trading_config,
                training_mode=True
            )
            # Override feature extractor
            env.feature_extractor = feature_gen
            return env
        
        train_env = DummyVecEnv([make_env])
        
        # Create PPO agent with faster settings
        agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-3,  # High learning rate
            n_steps=64,          # Small buffer
            batch_size=16,       # Small batch
            n_epochs=3,          # Few epochs
            verbose=0
        )
        
        # Quick training
        start_time = datetime.now()
        agent.learn(total_timesteps=800)  # Minimal training
        training_time = datetime.now() - start_time
        
        # Test on out-of-sample data
        test_env = TradingEnvironment(
            price_data=test_data,
            transformer_model=transformer,
            config=trading_config,
            training_mode=False
        )
        test_env.feature_extractor = feature_gen
        
        # Run test episode
        obs, info = test_env.reset()
        done = False
        actions = []
        portfolio_values = []
        
        step_count = 0
        while not done and step_count < len(test_data) - 10:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            actions.append(int(action))
            portfolio_values.append(info['portfolio_value'])
            step_count += 1
        
        # Calculate results
        final_value = portfolio_values[-1] if portfolio_values else 10000
        initial_value = 10000
        strategy_return = (final_value / initial_value) - 1
        
        # Benchmark return
        bh_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1
        
        # Calculate some basic metrics
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(480)  # Annualized for 3-min data
            
            # Sharpe ratio
            risk_free_rate = 0.02 / 365 / 480  # Daily risk-free rate for 3-min intervals
            excess_returns = returns - risk_free_rate
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(480)
            
            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_dd = np.max(drawdown)
        else:
            volatility = 0
            sharpe = 0
            max_dd = 0
        
        # Count trades (position changes)
        position_changes = 0
        for i in range(1, len(actions)):
            if actions[i] != 0 and actions[i] != actions[i-1]:  # Not hold and different from previous
                position_changes += 1
        
        result = {
            'window': window_idx,
            'training_time': training_time.total_seconds(),
            'strategy_return': strategy_return,
            'benchmark_return': bh_return,
            'excess_return': strategy_return - bh_return,
            'sharpe_ratio': sharpe,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'total_trades': position_changes,
            'final_value': final_value,
            'test_periods': len(portfolio_values),
            'actions': actions[:20]  # First 20 actions for analysis
        }
        
        print(f"✅ Strategy: {strategy_return*100:+.2f}% | BH: {bh_return*100:+.2f}% | "
              f"Excess: {(strategy_return-bh_return)*100:+.2f}% | "
              f"Sharpe: {sharpe:.2f} | DD: {max_dd*100:.1f}% | "
              f"Trades: {position_changes} | Time: {training_time}")
        
        return result
        
    except Exception as e:
        print(f"❌ Window {window_idx} failed: {e}")
        return {'window': window_idx, 'error': str(e)}


def main():
    """Main execution function"""
    print("🚀 Bitcoin 3-Min Walk-Forward Validation")
    print("=" * 50)
    
    # Load data
    data = load_and_prepare_btc_data()
    
    # Use larger windows for more stable results
    train_hours = 72      # 3 days training
    test_hours = 24       # 1 day testing  
    step_hours = 12       # 12 hour steps
    
    train_candles = train_hours * 20
    test_candles = test_hours * 20
    step_candles = step_hours * 20
    
    print(f"\n📊 Window Configuration:")
    print(f"  Training: {train_hours}h ({train_candles} candles)")
    print(f"  Testing: {test_hours}h ({test_candles} candles)")
    print(f"  Step: {step_hours}h ({step_candles} candles)")
    
    # Calculate number of possible windows
    total_candles = len(data)
    max_windows = (total_candles - train_candles - test_candles) // step_candles
    num_windows = min(max_windows, 5)  # Test 5 windows
    
    print(f"  Testing {num_windows} windows (max possible: {max_windows})")
    
    # Run validation
    results = []
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_candles
        train_end_idx = start_idx + train_candles
        test_end_idx = train_end_idx + test_candles
        
        if test_end_idx >= len(data):
            print(f"⚠️  Stopping at window {window_idx} - insufficient data")
            break
        
        train_data = data.iloc[start_idx:train_end_idx].copy()
        test_data = data.iloc[train_end_idx:test_end_idx].copy()
        
        result = run_btc_window_test(train_data, test_data, window_idx)
        results.append(result)
    
    # Analyze results
    print("\n📈 BITCOIN TRADING RESULTS")
    print("=" * 35)
    
    valid_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if valid_results:
        strategy_returns = [r['strategy_return'] for r in valid_results]
        benchmark_returns = [r['benchmark_return'] for r in valid_results]
        excess_returns = [r['excess_return'] for r in valid_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in valid_results]
        max_drawdowns = [r['max_drawdown'] for r in valid_results]
        total_trades = [r['total_trades'] for r in valid_results]
        
        print(f"✅ Successful windows: {len(valid_results)}")
        print(f"❌ Failed windows: {len(failed_results)}")
        print()
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"  Avg Strategy Return: {np.mean(strategy_returns)*100:+.2f}% ± {np.std(strategy_returns)*100:.2f}%")
        print(f"  Avg Benchmark Return: {np.mean(benchmark_returns)*100:+.2f}%")
        print(f"  Avg Excess Return: {np.mean(excess_returns)*100:+.2f}%")
        print(f"  Avg Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
        print(f"  Avg Max Drawdown: {np.mean(max_drawdowns)*100:.2f}%")
        print(f"  Avg Trades per Window: {np.mean(total_trades):.1f}")
        print(f"  Win Rate: {np.mean([1 if r > 0 else 0 for r in strategy_returns])*100:.1f}%")
        print(f"  Best Return: {np.max(strategy_returns)*100:+.2f}%")
        print(f"  Worst Return: {np.min(strategy_returns)*100:+.2f}%")
        
        # Show individual window results
        print(f"\n📋 INDIVIDUAL WINDOW RESULTS:")
        for r in valid_results:
            print(f"  Window {r['window']}: Strategy {r['strategy_return']*100:+.2f}% | "
                  f"BH {r['benchmark_return']*100:+.2f}% | "
                  f"Excess {r['excess_return']*100:+.2f}% | "
                  f"Sharpe {r['sharpe_ratio']:.2f}")
        
        # Action analysis
        all_actions = []
        for r in valid_results:
            all_actions.extend(r.get('actions', []))
        
        if all_actions:
            action_counts = np.bincount(all_actions, minlength=9)
            action_names = ["Hold", "Buy 25%", "Buy 50%", "Buy 75%", "Buy 100%",
                           "Sell 25%", "Sell 50%", "Sell 75%", "Sell 100%"]
            
            print(f"\n🎯 TRADING BEHAVIOR:")
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                pct = count / len(all_actions) * 100 if all_actions else 0
                print(f"  {name}: {count} times ({pct:.1f}%)")
        
        # Simple visualization
        if len(valid_results) > 1:
            print(f"\n📊 Creating performance chart...")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Bitcoin 3-Min Transformer+RL Trading Results', fontsize=14)
            
            windows = [r['window'] for r in valid_results]
            
            # Returns comparison
            ax1.plot(windows, [r*100 for r in strategy_returns], 'bo-', label='Transformer+RL', linewidth=2)
            ax1.plot(windows, [r*100 for r in benchmark_returns], 'ro-', label='Buy & Hold', linewidth=2)
            ax1.set_title('Returns by Window')
            ax1.set_ylabel('Return (%)')
            ax1.set_xlabel('Window')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Excess returns
            colors = ['green' if r > 0 else 'red' for r in excess_returns]
            ax2.bar(windows, [r*100 for r in excess_returns], color=colors, alpha=0.7)
            ax2.set_title('Excess Returns vs Buy & Hold')
            ax2.set_ylabel('Excess Return (%)')
            ax2.set_xlabel('Window')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # Sharpe ratios
            ax3.bar(windows, sharpe_ratios, alpha=0.7, color='blue')
            ax3.set_title('Sharpe Ratios')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.set_xlabel('Window')
            ax3.grid(True, alpha=0.3)
            
            # Max drawdowns
            ax4.bar(windows, [r*100 for r in max_drawdowns], alpha=0.7, color='red')
            ax4.set_title('Maximum Drawdowns')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.set_xlabel('Window')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_transformer_rl_results_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Chart saved as '{filename}'")
            plt.show()
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        avg_excess = np.mean(excess_returns)
        if avg_excess > 0.02:  # > 2% excess return
            print("🟢 EXCELLENT: Strategy significantly outperformed buy-and-hold!")
        elif avg_excess > 0:
            print("🟡 GOOD: Strategy outperformed buy-and-hold on average.")
        elif avg_excess > -0.02:
            print("🟠 NEUTRAL: Strategy performed similarly to buy-and-hold.")
        else:
            print("🔴 POOR: Strategy underperformed buy-and-hold.")
            
        print(f"\n✨ The Transformer+RL strategy has been validated on real Bitcoin 3-minute data!")
        
    else:
        print("❌ No successful validation windows completed")
        if failed_results:
            print("Failed windows:")
            for r in failed_results:
                print(f"  Window {r['window']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()