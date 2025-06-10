#!/usr/bin/env python3
"""
Bitcoin Simple Validation - Working Version
==========================================
Simplified approach that bypasses complex ensemble for reliable validation.
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
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, HybridTransformerLSTM
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import talib


class MinimalFeatureExtractor:
    """Ultra-simple feature extractor for reliable operation"""
    
    def generate_features(self, data):
        """Generate minimal but robust features"""
        df = data.copy()
        
        if len(df) < 30:
            raise ValueError("Need at least 30 candles")
        
        features = pd.DataFrame(index=df.index)
        
        # Basic price features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Simple moving averages
        features['ma5_ratio'] = df['Close'] / df['Close'].rolling(5).mean() - 1
        features['ma10_ratio'] = df['Close'] / df['Close'].rolling(10).mean() - 1
        features['ma20_ratio'] = df['Close'] / df['Close'].rolling(20).mean() - 1
        
        # Price ratios
        features['high_low_ratio'] = (df['High'] / df['Low']) - 1
        features['close_open_ratio'] = (df['Close'] / df['Open']) - 1
        
        # Simple momentum
        features['momentum_3'] = df['Close'].pct_change(3)
        features['momentum_5'] = df['Close'].pct_change(5)
        
        # Volume
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
        
        # Simple volatility
        features['volatility'] = features['returns'].rolling(10).std()
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        # Remove first 25 rows to ensure all features are clean
        features = features.iloc[25:].copy()
        
        return features


class SimpleTradingSimulator:
    """Simple trading simulator using single transformer model"""
    
    def __init__(self):
        self.feature_extractor = MinimalFeatureExtractor()
        
    def create_single_transformer(self, feature_dim):
        """Create a single transformer model (no ensemble)"""
        config = TurboDMTConfig(
            feature_dim=feature_dim,
            seq_len=10,
            hidden_dim=64,
            transformer_dim=48,  # 48 is divisible by 6 heads
            transformer_layers=1,
            attention_heads=6,   # 48/6 = 8 (valid)
            lstm_layers=1,
            ensemble_size=1,
            max_position_size=1.0,
            target_vol=0.25,
            neutral_zone=0.02
        )
        
        return HybridTransformerLSTM(config)
    
    def run_window_test(self, train_data, test_data, window_idx):
        """Run a single window test"""
        print(f"\n🔄 Window {window_idx}")
        print(f"📈 Train: {len(train_data)} candles")
        print(f"🧪 Test: {len(test_data)} candles")
        
        try:
            # Generate features
            train_features = self.feature_extractor.generate_features(train_data)
            
            if len(train_features) < 50:
                raise ValueError("Insufficient features")
            
            # Create single transformer model
            feature_dim = train_features.shape[1]
            transformer = self.create_single_transformer(feature_dim)
            
            # Wrap in a simple ensemble-like wrapper
            class SimpleTransformerWrapper:
                def __init__(self, model):
                    self.model = model
                    
                def predict(self, x):
                    with torch.no_grad():
                        pred, vol, regime = self.model(x)
                        # Return in ensemble format
                        uncertainty = torch.zeros_like(pred)
                        return pred, vol, regime, uncertainty
            
            wrapped_transformer = SimpleTransformerWrapper(transformer)
            
            # Create trading environment with simple config
            trading_config = TradingConfig(
                initial_balance=10000,
                max_position_size=0.8,
                transaction_cost=0.001,
                slippage=0.0005
            )
            
            # Create environment that bypasses complex feature extraction
            class SimpleTradingEnvironment(TradingEnvironment):
                def _get_observation(self):
                    """Override to use simple features"""
                    if self.current_step >= len(self.price_data):
                        return np.zeros(32, dtype=np.float32)
                    
                    # Get recent price data
                    start_idx = max(0, self.current_step - 30)
                    recent_data = self.price_data.iloc[start_idx:self.current_step+1].copy()
                    
                    if len(recent_data) < 10:
                        return np.zeros(32, dtype=np.float32)
                    
                    # Simple features
                    current_price = recent_data['close'].iloc[-1]
                    prices = recent_data['close'].values
                    
                    # Basic features (32 total)
                    features = np.array([
                        # Price ratios
                        prices[-1] / prices[-2] - 1 if len(prices) > 1 else 0,
                        prices[-1] / prices[-5] - 1 if len(prices) > 5 else 0,
                        prices[-1] / prices[-10] - 1 if len(prices) > 10 else 0,
                        
                        # Moving average ratios
                        prices[-1] / np.mean(prices[-5:]) - 1 if len(prices) >= 5 else 0,
                        prices[-1] / np.mean(prices[-10:]) - 1 if len(prices) >= 10 else 0,
                        
                        # Volatility
                        np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0,
                        
                        # Portfolio features
                        self.position,
                        self.total_portfolio_value / self.config.initial_balance,
                        self.current_drawdown,
                        self.balance / self.total_portfolio_value if self.total_portfolio_value > 0 else 0,
                        
                        # Fill remaining with zeros or simple features
                        *[0.0] * 22  # Pad to 32 features
                    ], dtype=np.float32)
                    
                    return features
            
            # Create environments
            def make_train_env():
                return SimpleTradingEnvironment(
                    price_data=train_data,
                    transformer_model=wrapped_transformer,
                    config=trading_config,
                    training_mode=True
                )
            
            train_env = DummyVecEnv([make_train_env])
            
            # Create PPO agent
            agent = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=1e-3,
                n_steps=64,
                batch_size=16,
                n_epochs=3,
                verbose=0
            )
            
            # Train
            start_time = datetime.now()
            agent.learn(total_timesteps=500)  # Quick training
            training_time = datetime.now() - start_time
            
            # Test
            test_env = SimpleTradingEnvironment(
                price_data=test_data,
                transformer_model=wrapped_transformer,
                config=trading_config,
                training_mode=False
            )
            
            # Run test episode
            obs, info = test_env.reset()
            done = False
            actions = []
            portfolio_values = []
            
            steps = 0
            while not done and steps < len(test_data) - 5:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                actions.append(int(action))
                portfolio_values.append(info['portfolio_value'])
                steps += 1
            
            # Calculate results
            if portfolio_values:
                final_value = portfolio_values[-1]
                strategy_return = (final_value / 10000) - 1
            else:
                strategy_return = 0
                
            bh_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1
            
            # Simple metrics
            excess_return = strategy_return - bh_return
            trade_count = len([a for a in actions if a != 0])
            
            result = {
                'window': window_idx,
                'strategy_return': strategy_return,
                'benchmark_return': bh_return,
                'excess_return': excess_return,
                'total_trades': trade_count,
                'training_time': training_time.total_seconds(),
                'test_steps': len(portfolio_values)
            }
            
            print(f"✅ Strategy: {strategy_return*100:+.2f}% | BH: {bh_return*100:+.2f}% | "
                  f"Excess: {excess_return*100:+.2f}% | Trades: {trade_count} | "
                  f"Time: {training_time}")
            
            return result
            
        except Exception as e:
            print(f"❌ Window {window_idx} failed: {e}")
            return {'window': window_idx, 'error': str(e)}


def load_btc_data():
    """Load Bitcoin data"""
    print("📊 Loading Bitcoin data...")
    
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
    
    # Basic cleaning
    df = df.dropna()
    df = df[df['Volume'] > 0]
    
    # Remove obvious outliers
    price_median = df['Close'].median()
    price_std = df['Close'].std()
    df = df[np.abs(df['Close'] - price_median) < 3 * price_std]
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"📅 Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"💰 Price: ${df['Close'].min():,.0f} to ${df['Close'].max():,.0f}")
    
    return df


def main():
    """Main function"""
    print("🚀 Bitcoin Simple Walk-Forward Validation")
    print("=" * 50)
    
    # Import torch here to avoid issues
    import torch
    
    # Load data
    data = load_btc_data()
    
    # Simple window configuration
    train_hours = 48   # 2 days training
    test_hours = 12    # 12 hours testing
    step_hours = 8     # 8 hour steps
    
    train_candles = train_hours * 20
    test_candles = test_hours * 20
    step_candles = step_hours * 20
    
    print(f"\n📊 Configuration:")
    print(f"  Training: {train_hours}h ({train_candles} candles)")
    print(f"  Testing: {test_hours}h ({test_candles} candles)")
    print(f"  Step: {step_hours}h ({step_candles} candles)")
    
    # Calculate windows
    max_windows = (len(data) - train_candles - test_candles) // step_candles
    num_windows = min(max_windows, 3)  # Test just 3 windows
    
    print(f"  Testing {num_windows} windows")
    
    # Create simulator
    simulator = SimpleTradingSimulator()
    
    # Run tests
    results = []
    
    for window_idx in range(num_windows):
        start_idx = window_idx * step_candles
        train_end_idx = start_idx + train_candles
        test_end_idx = train_end_idx + test_candles
        
        train_data = data.iloc[start_idx:train_end_idx].copy()
        test_data = data.iloc[train_end_idx:test_end_idx].copy()
        
        result = simulator.run_window_test(train_data, test_data, window_idx)
        results.append(result)
    
    # Analyze results
    print("\n📈 SIMPLE VALIDATION RESULTS")
    print("=" * 35)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        strategy_returns = [r['strategy_return'] for r in valid_results]
        benchmark_returns = [r['benchmark_return'] for r in valid_results]
        excess_returns = [r['excess_return'] for r in valid_results]
        
        print(f"✅ Successful windows: {len(valid_results)}")
        print(f"📊 Average strategy return: {np.mean(strategy_returns)*100:+.2f}%")
        print(f"📊 Average benchmark return: {np.mean(benchmark_returns)*100:+.2f}%")
        print(f"📊 Average excess return: {np.mean(excess_returns)*100:+.2f}%")
        print(f"📊 Win rate: {np.mean([1 if r > 0 else 0 for r in strategy_returns])*100:.0f}%")
        
        print(f"\n📋 Individual Results:")
        for r in valid_results:
            print(f"  Window {r['window']}: {r['strategy_return']*100:+.2f}% vs {r['benchmark_return']*100:+.2f}% "
                  f"(excess: {r['excess_return']*100:+.2f}%)")
        
        # Simple chart
        if len(valid_results) > 1:
            print(f"\n📊 Creating chart...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            windows = [r['window'] for r in valid_results]
            
            # Returns comparison
            ax1.plot(windows, [r*100 for r in strategy_returns], 'bo-', label='Strategy', linewidth=2)
            ax1.plot(windows, [r*100 for r in benchmark_returns], 'ro-', label='Buy & Hold', linewidth=2)
            ax1.set_title('Bitcoin 3-Min Trading Returns')
            ax1.set_ylabel('Return (%)')
            ax1.set_xlabel('Window')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Excess returns
            colors = ['green' if r > 0 else 'red' for r in excess_returns]
            ax2.bar(windows, [r*100 for r in excess_returns], color=colors, alpha=0.7)
            ax2.set_title('Excess Returns')
            ax2.set_ylabel('Excess Return (%)')
            ax2.set_xlabel('Window')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_simple_validation_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Chart saved as '{filename}'")
            plt.show()
        
        print(f"\n🎉 Bitcoin validation completed!")
        if np.mean(excess_returns) > 0:
            print("🟢 Strategy outperformed buy-and-hold on average!")
        else:
            print("🟡 Strategy underperformed buy-and-hold on average.")
            
    else:
        print("❌ No successful windows")
        for r in results:
            if 'error' in r:
                print(f"  Window {r['window']}: {r['error']}")


if __name__ == "__main__":
    main()