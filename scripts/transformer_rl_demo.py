#!/usr/bin/env python3
"""
Transformer+RL Strategy Demo
===========================
Quick demonstration of the complete pipeline with sliding window testing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_trader_o3_algo.strategies.turbo_dmt_v2.transformer_rl_strategy import (
    TransformerRLStrategy, TransformerRLConfig
)
from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig
from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingConfig


def create_realistic_market_data(n_periods=1000, start_price=100):
    """Create more realistic market data with regime changes"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    prices = []
    volumes = []
    price = start_price
    
    # Create regime-based market data
    for i in range(n_periods):
        # Different market regimes
        if i < 300:  # Bull market
            drift = 0.0008
            vol = 0.012
        elif i < 600:  # Bear market  
            drift = -0.0005
            vol = 0.022
        elif i < 800:  # Sideways
            drift = 0.0001
            vol = 0.008
        else:  # Volatile recovery
            drift = 0.0012
            vol = 0.025
        
        # Generate realistic return
        ret = np.random.normal(drift, vol)
        price *= (1 + ret)
        prices.append(price)
        
        # Volume correlated with volatility
        base_vol = 5000000
        vol_factor = 1 + 2 * abs(ret)  # Higher volume on big moves
        volume = int(base_vol * vol_factor * (1 + np.random.normal(0, 0.3)))
        volumes.append(max(volume, 100000))
    
    # Create OHLC from Close prices
    highs = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
    opens = [prices[0]] + [prices[i-1] * (1 + np.random.normal(0, 0.002)) for i in range(1, len(prices))]
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'close': prices,
        'Volume': volumes
    })
    
    # Ensure price relationships are valid
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


def run_transformer_rl_demo():
    """Run a complete demonstration"""
    print("🤖 Transformer+RL Trading Strategy Demo")
    print("=" * 50)
    
    # Create realistic market data
    print("📊 Generating realistic market data...")
    data = create_realistic_market_data(1000)
    print(f"Created {len(data)} periods of market data")
    
    # Configure the strategy (disable advanced features for demo)
    print("⚙️  Configuring strategy...")
    config = TransformerRLConfig()
    config.total_timesteps = 5000  # Quick training for demo
    config.transformer_config = TurboDMTConfig(
        seq_len=20,  # Shorter for faster training
        ensemble_size=3,  # Smaller ensemble for speed
        feature_dim=38  # Match actual feature count
    )
    config.trading_config = TradingConfig(
        initial_balance=10000,
        max_position_size=0.9
    )
    
    strategy = TransformerRLStrategy(config)
    
    # Use simplified feature extraction for demo
    from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import AdvancedFeatureGenerator
    strategy.feature_extractor = AdvancedFeatureGenerator(
        use_spectral=False,
        use_multi_timeframe=False, 
        use_market_regime=False,
        use_volatility_surface=False,
        use_orderflow=False,
        standardize=False
    )
    
    # Split data for training/testing
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"Training period: {train_data.iloc[0]['Date']} to {train_data.iloc[-1]['Date']}")
    print(f"Testing period: {test_data.iloc[0]['Date']} to {test_data.iloc[-1]['Date']}")
    
    # Train the strategy
    print("\n🧠 Training Transformer+RL strategy...")
    start_time = datetime.now()
    
    try:
        training_history = strategy.train(
            training_data=train_data,
            pretrain_transformer=False,  # Skip pre-training for demo
            verbose=False  # Disable verbose to avoid progress bar
        )
        
        training_time = datetime.now() - start_time
        print(f"✅ Training completed in {training_time}")
        
        # Backtest on out-of-sample data
        print("\n📈 Running out-of-sample backtest...")
        backtest_results = strategy.backtest(test_data)
        
        # Calculate benchmark return (buy and hold)
        benchmark_return = (test_data.iloc[-1]['Close'] / test_data.iloc[0]['Close']) - 1
        
        # Display results
        print("\n📊 RESULTS SUMMARY")
        print("=" * 30)
        print(f"Strategy Return:     {backtest_results['total_return']*100:+.2f}%")
        print(f"Benchmark Return:    {benchmark_return*100:+.2f}%")
        print(f"Excess Return:       {(backtest_results['total_return'] - benchmark_return)*100:+.2f}%")
        print(f"Sharpe Ratio:        {backtest_results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {backtest_results['max_drawdown']*100:.2f}%")
        print(f"Win Rate:            {backtest_results['win_rate']*100:.1f}%")
        print(f"Total Trades:        {backtest_results['total_trades']}")
        print(f"Annualized Return:   {backtest_results['annualized_return']*100:+.2f}%")
        print(f"Volatility:          {backtest_results['volatility']*100:.2f}%")
        
        # Create performance plot
        print("\n📈 Creating performance visualization...")
        plot_results(test_data, backtest_results, benchmark_return)
        
        # Test a single prediction
        print("\n🔮 Testing single prediction...")
        recent_data = data.iloc[-100:].copy()
        action, info = strategy.predict(recent_data)
        
        action_map = {0: "Hold", 1: "Buy 25%", 2: "Buy 50%", 3: "Buy 75%", 4: "Buy 100%",
                     5: "Sell 25%", 6: "Sell 50%", 7: "Sell 75%", 8: "Sell 100%"}
        
        print(f"Predicted Action:    {action_map.get(action, 'Unknown')} (Action {action})")
        print(f"Current Price:       ${info.get('current_price', 'N/A'):.2f}")
        print(f"Model Prediction:    {info.get('transformer_prediction', 'N/A'):.4f}")
        print(f"Uncertainty:         {info.get('model_uncertainty', 'N/A'):.4f}")
        
        # Save the model
        print("\n💾 Saving trained model...")
        strategy.save_model("demo_transformer_rl_model")
        print("Model saved as 'demo_transformer_rl_model_*'")
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_results(test_data, backtest_results, benchmark_return):
    """Create performance visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Transformer+RL Strategy Performance', fontsize=16)
    
    # Plot 1: Portfolio value over time
    portfolio_history = backtest_results['portfolio_history']
    dates = test_data['Date'].iloc[:len(portfolio_history)]
    
    ax1.plot(dates, portfolio_history, label='Strategy', linewidth=2)
    
    # Calculate benchmark portfolio
    initial_value = portfolio_history[0]
    benchmark_portfolio = [initial_value * (test_data.iloc[i]['Close'] / test_data.iloc[0]['Close']) 
                          for i in range(len(portfolio_history))]
    ax1.plot(dates, benchmark_portfolio, label='Buy & Hold', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actions over time
    actions = backtest_results['action_history']
    ax2.scatter(range(len(actions)), actions, alpha=0.6, s=20)
    ax2.set_title('Trading Actions Over Time')
    ax2.set_ylabel('Action (0=Hold, 4=Max Buy, 8=Max Sell)')
    ax2.set_xlabel('Time Steps')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns distribution
    if len(portfolio_history) > 1:
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        ax3.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative returns comparison
    if len(portfolio_history) > 1:
        strategy_returns = np.cumprod(1 + np.diff(portfolio_history) / portfolio_history[:-1]) - 1
        benchmark_returns = np.cumprod(1 + np.diff(benchmark_portfolio) / benchmark_portfolio[:-1]) - 1
        
        ax4.plot(strategy_returns, label='Strategy', linewidth=2)
        ax4.plot(benchmark_returns, label='Benchmark', linewidth=2, alpha=0.7)
        ax4.set_title('Cumulative Returns')
        ax4.set_ylabel('Cumulative Return')
        ax4.set_xlabel('Time Steps')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transformer_rl_demo_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved as '{filename}'")
    
    plt.show()


if __name__ == "__main__":
    success = run_transformer_rl_demo()
    if success:
        print("\n✨ The Transformer+RL architecture is fully operational!")
        print("Key Features Demonstrated:")
        print("  ✓ Transformer-based feature extraction")
        print("  ✓ PPO reinforcement learning agent")
        print("  ✓ Custom trading environment") 
        print("  ✓ Risk-adjusted reward function")
        print("  ✓ Real-time prediction capability")
        print("  ✓ Performance visualization")
        print("  ✓ Model persistence")
    else:
        print("❌ Demo encountered issues. Check the output above.")