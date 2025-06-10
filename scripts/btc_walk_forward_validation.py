#!/usr/bin/env python3
"""
Bitcoin Walk-Forward Validation with Transformer+RL Strategy
==========================================================
Real-world trading simulation using sliding window methodology
on 3-minute BTC/USDT data from January to April 2025.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json

from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingEnvironment, TradingConfig, SlidingWindowTester
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, TurboDMTEnsemble
from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import AdvancedFeatureGenerator
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class BTCTradingSimulator:
    """Bitcoin trading simulator with walk-forward validation"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = []
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare Bitcoin data"""
        print("📊 Loading Bitcoin 3-minute data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Add lowercase close for compatibility
        df['close'] = df['Close']
        
        # Remove any invalid data
        df = df.dropna()
        df = df[df['Volume'] > 0]  # Remove zero volume candles
        
        # Filter out obvious data errors
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df = df[(df[col] > 1000) & (df[col] < 500000)]  # Reasonable BTC price range
        
        self.data = df
        print(f"✅ Loaded {len(self.data)} 3-minute candles")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"💰 Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
        
    def create_strategy_configs(self):
        """Create different strategy configurations for testing"""
        configs = []
        
        # Conservative configuration for crypto
        conservative = {
            'name': 'Conservative',
            'transformer_config': TurboDMTConfig(
                feature_dim=30,  # Will adjust based on actual features
                seq_len=20,
                ensemble_size=2,  # Smaller for speed
                max_position_size=0.8,
                target_vol=0.25,  # Higher for crypto
                neutral_zone=0.015
            ),
            'trading_config': TradingConfig(
                initial_balance=10000,
                max_position_size=0.8,
                transaction_cost=0.001,  # 0.1% fee
                slippage=0.0005
            ),
            'training_steps': 3000
        }
        
        # Aggressive configuration  
        aggressive = {
            'name': 'Aggressive',
            'transformer_config': TurboDMTConfig(
                feature_dim=30,
                seq_len=30,
                ensemble_size=3,
                max_position_size=1.5,
                target_vol=0.35,
                neutral_zone=0.01
            ),
            'trading_config': TradingConfig(
                initial_balance=10000,
                max_position_size=1.0,
                transaction_cost=0.001,
                slippage=0.0005
            ),
            'training_steps': 4000
        }
        
        configs.extend([conservative, aggressive])
        return configs
    
    def run_single_window_test(self, train_data, test_data, config, window_idx):
        """Run training and testing on a single window"""
        print(f"\n🔄 Window {window_idx}: {config['name']} Strategy")
        print(f"📈 Train: {train_data['Date'].iloc[0]} to {train_data['Date'].iloc[-1]}")
        print(f"🧪 Test: {test_data['Date'].iloc[0]} to {test_data['Date'].iloc[-1]}")
        
        try:
            # Create feature extractor (simplified for crypto)
            feature_gen = AdvancedFeatureGenerator(
                use_spectral=False,
                use_multi_timeframe=True,  # Good for crypto
                use_market_regime=True,   # Important for crypto
                use_volatility_surface=False,
                use_orderflow=False,
                standardize=True
            )
            
            # Extract features to get actual feature count
            sample_features = feature_gen.generate_features(train_data.iloc[-100:])
            actual_feature_dim = sample_features.shape[1] - 1  # Exclude target
            
            # Update config with actual feature dimension
            config['transformer_config'].feature_dim = actual_feature_dim
            
            # Create transformer model
            transformer = TurboDMTEnsemble(config['transformer_config'])
            
            # Create and train PPO agent
            def make_train_env():
                return TradingEnvironment(
                    price_data=train_data,
                    transformer_model=transformer,
                    config=config['trading_config'],
                    training_mode=True
                )
            
            train_env = DummyVecEnv([make_train_env])
            agent = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=256,  # Smaller for 3-min data
                batch_size=32,
                verbose=0
            )
            
            # Train the agent
            start_time = datetime.now()
            agent.learn(total_timesteps=config['training_steps'])
            training_time = datetime.now() - start_time
            
            print(f"⏱️  Training completed in {training_time}")
            
            # Test on out-of-sample data
            test_env = TradingEnvironment(
                price_data=test_data,
                transformer_model=transformer,
                config=config['trading_config'],
                training_mode=False
            )
            
            # Run test episode
            obs, info = test_env.reset()
            done = False
            actions_taken = []
            portfolio_values = []
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                actions_taken.append(int(action))
                portfolio_values.append(info['portfolio_value'])
            
            # Get final statistics
            test_stats = test_env.get_episode_stats()
            
            # Calculate buy-and-hold benchmark
            bh_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0]) - 1
            
            # Compile results
            result = {
                'window_idx': window_idx,
                'config_name': config['name'],
                'train_start': train_data['Date'].iloc[0],
                'train_end': train_data['Date'].iloc[-1],
                'test_start': test_data['Date'].iloc[0],
                'test_end': test_data['Date'].iloc[-1],
                'training_time': training_time.total_seconds(),
                'strategy_return': test_stats.get('total_return', 0),
                'benchmark_return': bh_return,
                'excess_return': test_stats.get('total_return', 0) - bh_return,
                'sharpe_ratio': test_stats.get('sharpe_ratio', 0),
                'max_drawdown': test_stats.get('max_drawdown', 0),
                'volatility': test_stats.get('volatility', 0),
                'total_trades': test_stats.get('total_trades', 0),
                'win_rate': test_stats.get('win_rate', 0),
                'final_portfolio': test_stats.get('final_portfolio_value', 0),
                'actions_taken': actions_taken,
                'portfolio_history': portfolio_values
            }
            
            print(f"📊 Results: Return={result['strategy_return']*100:+.2f}%, "
                  f"Sharpe={result['sharpe_ratio']:.3f}, "
                  f"DD={result['max_drawdown']*100:.2f}%, "
                  f"Trades={result['total_trades']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Window {window_idx} failed: {e}")
            return {
                'window_idx': window_idx,
                'config_name': config['name'],
                'error': str(e),
                'strategy_return': 0,
                'benchmark_return': 0
            }
    
    def run_walk_forward_validation(self):
        """Run comprehensive walk-forward validation"""
        print("\n🚀 Starting Bitcoin Walk-Forward Validation")
        print("=" * 60)
        
        # Configuration
        train_window_hours = 72  # 3 days of 3-min data (72 * 20 = 1440 candles)
        test_window_hours = 24   # 1 day of testing (24 * 20 = 480 candles) 
        step_hours = 12          # Move forward by 12 hours (240 candles)
        
        train_candles = train_window_hours * 20  # 20 candles per hour for 3-min data
        test_candles = test_window_hours * 20
        step_candles = step_hours * 20
        
        print(f"📊 Window Configuration:")
        print(f"  Training: {train_window_hours} hours ({train_candles} candles)")
        print(f"  Testing: {test_window_hours} hours ({test_candles} candles)")
        print(f"  Step size: {step_hours} hours ({step_candles} candles)")
        
        # Calculate number of windows
        total_candles = len(self.data)
        num_windows = (total_candles - train_candles - test_candles) // step_candles
        print(f"  Total windows: {num_windows}")
        
        # Get strategy configurations
        configs = self.create_strategy_configs()
        
        # Run walk-forward validation
        all_results = []
        
        for window_idx in range(min(num_windows, 10)):  # Limit to 10 windows for demo
            start_idx = window_idx * step_candles
            train_end_idx = start_idx + train_candles
            test_end_idx = train_end_idx + test_candles
            
            if test_end_idx >= len(self.data):
                break
                
            train_data = self.data.iloc[start_idx:train_end_idx].copy()
            test_data = self.data.iloc[train_end_idx:test_end_idx].copy()
            
            # Test each configuration
            for config in configs:
                result = self.run_single_window_test(train_data, test_data, config, window_idx)
                all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """Analyze and summarize walk-forward results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print("\n📈 WALK-FORWARD VALIDATION RESULTS")
        print("=" * 50)
        
        # Filter out failed results
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            print("❌ No valid results found")
            return
        
        # Group by strategy
        by_strategy = {}
        for result in valid_results:
            strategy = result['config_name']
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(result)
        
        # Calculate aggregate statistics
        summary = {}
        for strategy, results in by_strategy.items():
            returns = [r['strategy_return'] for r in results]
            benchmark_returns = [r['benchmark_return'] for r in results]
            excess_returns = [r['excess_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            
            summary[strategy] = {
                'num_windows': len(results),
                'avg_return': np.mean(returns),
                'avg_benchmark': np.mean(benchmark_returns),
                'avg_excess_return': np.mean(excess_returns),
                'return_std': np.std(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_max_dd': np.mean(max_drawdowns),
                'worst_dd': np.max(max_drawdowns),
                'win_rate': np.mean([1 if r > 0 else 0 for r in returns]),
                'best_return': np.max(returns),
                'worst_return': np.min(returns)
            }
        
        # Print summary
        for strategy, stats in summary.items():
            print(f"\n🎯 {strategy} Strategy:")
            print(f"  Windows tested: {stats['num_windows']}")
            print(f"  Avg return: {stats['avg_return']*100:+.2f}% ± {stats['return_std']*100:.2f}%")
            print(f"  Avg benchmark: {stats['avg_benchmark']*100:+.2f}%")
            print(f"  Avg excess return: {stats['avg_excess_return']*100:+.2f}%")
            print(f"  Avg Sharpe ratio: {stats['avg_sharpe']:.3f}")
            print(f"  Avg max drawdown: {stats['avg_max_dd']*100:.2f}%")
            print(f"  Worst drawdown: {stats['worst_dd']*100:.2f}%")
            print(f"  Win rate: {stats['win_rate']*100:.1f}%")
            print(f"  Best/Worst return: {stats['best_return']*100:+.2f}% / {stats['worst_return']*100:+.2f}%")
        
        return summary
    
    def create_visualizations(self):
        """Create performance visualizations"""
        if not self.results:
            return
        
        print("\n📊 Creating performance visualizations...")
        
        valid_results = [r for r in self.results if 'error' not in r]
        if not valid_results:
            return
        
        # Set up plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bitcoin 3-Min Walk-Forward Validation Results', fontsize=16)
        
        # Group by strategy
        strategies = {}
        for result in valid_results:
            strategy = result['config_name']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(result)
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot 1: Returns over time
        ax1 = axes[0, 0]
        for i, (strategy, results) in enumerate(strategies.items()):
            windows = [r['window_idx'] for r in results]
            returns = [r['strategy_return'] * 100 for r in results]
            ax1.plot(windows, returns, marker='o', label=strategy, color=colors[i % len(colors)])
        ax1.set_title('Strategy Returns by Window')
        ax1.set_xlabel('Window Index')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Strategy vs Benchmark
        ax2 = axes[0, 1]
        for i, (strategy, results) in enumerate(strategies.items()):
            strategy_returns = [r['strategy_return'] * 100 for r in results]
            benchmark_returns = [r['benchmark_return'] * 100 for r in results]
            ax2.scatter(benchmark_returns, strategy_returns, label=strategy, alpha=0.7, color=colors[i % len(colors)])
        
        # Add diagonal line (equal performance)
        min_ret = min([r['benchmark_return'] * 100 for r in valid_results])
        max_ret = max([r['benchmark_return'] * 100 for r in valid_results])
        ax2.plot([min_ret, max_ret], [min_ret, max_ret], 'k--', alpha=0.5, label='Equal Performance')
        ax2.set_title('Strategy vs Benchmark Returns')
        ax2.set_xlabel('Benchmark Return (%)')
        ax2.set_ylabel('Strategy Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sharpe ratios
        ax3 = axes[0, 2]
        strategy_names = list(strategies.keys())
        sharpe_values = []
        for strategy in strategy_names:
            sharpes = [r['sharpe_ratio'] for r in strategies[strategy]]
            sharpe_values.append(sharpes)
        
        ax3.boxplot(sharpe_values, labels=strategy_names)
        ax3.set_title('Sharpe Ratio Distribution')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown analysis
        ax4 = axes[1, 0]
        for i, (strategy, results) in enumerate(strategies.items()):
            drawdowns = [r['max_drawdown'] * 100 for r in results]
            ax4.hist(drawdowns, alpha=0.6, label=strategy, bins=10, color=colors[i % len(colors)])
        ax4.set_title('Max Drawdown Distribution')
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Win rate analysis
        ax5 = axes[1, 1]
        win_rates = []
        for strategy in strategy_names:
            results = strategies[strategy]
            wins = [1 if r['strategy_return'] > 0 else 0 for r in results]
            win_rates.append(np.mean(wins) * 100)
        
        bars = ax5.bar(strategy_names, win_rates, color=colors[:len(strategy_names)])
        ax5.set_title('Win Rate by Strategy')
        ax5.set_ylabel('Win Rate (%)')
        ax5.set_ylim(0, 100)
        for bar, rate in zip(bars, win_rates):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Cumulative performance
        ax6 = axes[1, 2]
        for i, (strategy, results) in enumerate(strategies.items()):
            # Sort by window index
            sorted_results = sorted(results, key=lambda x: x['window_idx'])
            cumulative_returns = np.cumprod([1 + r['strategy_return'] for r in sorted_results]) - 1
            windows = [r['window_idx'] for r in sorted_results]
            ax6.plot(windows, cumulative_returns * 100, marker='o', label=strategy, color=colors[i % len(colors)])
        ax6.set_title('Cumulative Performance')
        ax6.set_xlabel('Window Index')
        ax6.set_ylabel('Cumulative Return (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btc_walk_forward_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as '{filename}'")
        
        plt.show()
    
    def save_results(self):
        """Save detailed results to JSON"""
        if not self.results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btc_walk_forward_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        results_for_json = []
        for result in self.results:
            json_result = result.copy()
            for key in ['train_start', 'train_end', 'test_start', 'test_end']:
                if key in json_result and hasattr(json_result[key], 'isoformat'):
                    json_result[key] = json_result[key].isoformat()
            results_for_json.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved as '{filename}'")


def main():
    """Main execution function"""
    print("🚀 Bitcoin Walk-Forward Validation with Transformer+RL")
    print("=" * 60)
    
    # Initialize simulator
    data_path = "/Users/zachrizzo/Desktop/programming/ai-stock-algo-03/data/BTCUSDT_3m_20250101_20250430.csv"
    simulator = BTCTradingSimulator(data_path)
    
    # Run walk-forward validation
    print(f"\n⏰ Starting validation at {datetime.now()}")
    results = simulator.run_walk_forward_validation()
    
    if results:
        # Analyze results
        summary = simulator.analyze_results()
        
        # Create visualizations
        simulator.create_visualizations()
        
        # Save results
        simulator.save_results()
        
        print(f"\n🎉 Walk-forward validation completed!")
        print(f"✅ Tested {len([r for r in results if 'error' not in r])} successful windows")
        print(f"❌ Failed {len([r for r in results if 'error' in r])} windows")
        
        # Print final summary
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            all_returns = [r['strategy_return'] for r in valid_results]
            all_benchmark = [r['benchmark_return'] for r in valid_results]
            
            print(f"\n📈 OVERALL SUMMARY:")
            print(f"  Strategy average return: {np.mean(all_returns)*100:+.2f}%")
            print(f"  Benchmark average return: {np.mean(all_benchmark)*100:+.2f}%")
            print(f"  Average excess return: {np.mean([r['excess_return'] for r in valid_results])*100:+.2f}%")
            print(f"  Win rate: {np.mean([1 if r['strategy_return'] > 0 else 0 for r in valid_results])*100:.1f}%")
            
    else:
        print("❌ No results generated")
    
    print(f"\n⏰ Completed at {datetime.now()}")


if __name__ == "__main__":
    main()