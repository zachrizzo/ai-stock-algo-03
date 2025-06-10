#!/usr/bin/env python3
"""
Simplified Test for Transformer+RL Strategy
==========================================
A minimal test to validate the architecture works.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Create simple test data instead of downloading
def create_test_data(n_periods=500):
    """Create synthetic stock price data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.015, n_periods)  # Daily returns
    price = 100.0
    prices = []
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    # Create OHLCV data
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'close': prices,  # Also add lowercase for compatibility
        'Volume': np.random.randint(1000000, 10000000, n_periods)
    })
    
    # Ensure High >= Low
    data['High'] = np.maximum(data['High'], data['Low'])
    data['High'] = np.maximum(data['High'], data['Close'])
    data['Low'] = np.minimum(data['Low'], data['Close'])
    
    return data

def test_basic_components():
    """Test individual components work"""
    print("Testing basic components...")
    
    try:
        # Test feature generation
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import AdvancedFeatureGenerator
        feature_gen = AdvancedFeatureGenerator()
        print("✓ Feature generator imported")
        
        # Test model
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, TurboDMTEnsemble
        config = TurboDMTConfig()
        model = TurboDMTEnsemble(config)
        print("✓ Transformer model created")
        
        # Test trading environment
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingEnvironment, TradingConfig
        trading_config = TradingConfig()
        print("✓ Trading environment imported")
        
        return True
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction on synthetic data"""
    print("\nTesting feature extraction...")
    
    try:
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.features import AdvancedFeatureGenerator
        
        # Create test data with more periods for rolling calculations
        data = create_test_data(300)
        
        # Extract features with simpler configuration
        feature_gen = AdvancedFeatureGenerator(
            use_spectral=False,  # Disable spectral for testing
            use_multi_timeframe=False,  # Disable multi-timeframe for testing
            use_market_regime=False,  # Disable regime detection for testing
            use_volatility_surface=False,  # Disable volatility surface for testing
            use_orderflow=False,  # Disable orderflow for testing
            standardize=False  # Disable standardization for testing
        )
        features = feature_gen.generate_features(data)
        
        print(f"✓ Generated {features.shape[1]} features for {features.shape[0]} periods")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transformer_forward_pass():
    """Test transformer forward pass"""
    print("\nTesting transformer forward pass...")
    
    try:
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, HybridTransformerLSTM
        import torch
        
        config = TurboDMTConfig()
        model = HybridTransformerLSTM(config)
        
        # Create dummy input
        batch_size = 4
        seq_len = 30
        feature_dim = 32
        x = torch.randn(batch_size, seq_len, feature_dim)
        
        # Forward pass
        pred, vol, regime = model(x)
        
        print(f"✓ Transformer output shapes: pred={pred.shape}, vol={vol.shape}, regime={regime.shape}")
        return True
    except Exception as e:
        print(f"✗ Transformer test failed: {e}")
        return False

def test_trading_environment():
    """Test trading environment"""
    print("\nTesting trading environment...")
    
    try:
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingEnvironment, TradingConfig
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, TurboDMTEnsemble
        
        # Create test data and model
        data = create_test_data(200)
        config = TurboDMTConfig()
        transformer = TurboDMTEnsemble(config)
        
        # Create environment
        trading_config = TradingConfig()
        env = TradingEnvironment(
            price_data=data,
            transformer_model=transformer,
            config=trading_config,
            training_mode=True
        )
        
        # Test basic operations
        obs, info = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i}: action={action}, reward={reward:.4f}, done={done}")
            if done:
                break
        
        return True
    except Exception as e:
        print(f"✗ Trading environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_integration():
    """Test PPO integration (minimal)"""
    print("\nTesting PPO integration...")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.rl_trading_env import TradingEnvironment, TradingConfig
        from stock_trader_o3_algo.strategies.turbo_dmt_v2.model import TurboDMTConfig, TurboDMTEnsemble
        
        # Create test data and model
        data = create_test_data(200)
        config = TurboDMTConfig()
        transformer = TurboDMTEnsemble(config)
        
        # Create environment
        def make_env():
            return TradingEnvironment(
                price_data=data,
                transformer_model=transformer,
                config=TradingConfig(),
                training_mode=True
            )
        
        env = DummyVecEnv([make_env])
        
        # Create PPO agent
        agent = PPO("MlpPolicy", env, verbose=0)
        
        print("✓ PPO agent created successfully")
        
        # Quick training test (just a few steps)
        agent.learn(total_timesteps=100)
        print("✓ PPO training test passed")
        
        return True
    except Exception as e:
        print(f"✗ PPO integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running Simplified Transformer+RL Tests")
    print("=" * 50)
    
    tests = [
        test_basic_components,
        test_feature_extraction,
        test_transformer_forward_pass,
        test_trading_environment,
        test_ppo_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! The Transformer+RL architecture is working.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)