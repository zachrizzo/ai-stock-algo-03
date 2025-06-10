#!/usr/bin/env python3
"""
Unified Transformer + RL Trading Strategy
========================================
Integration of TurboDMT v2 transformer with PPO reinforcement learning
for autonomous trading decisions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .model import TurboDMTEnsemble, TurboDMTConfig
from .features import AdvancedFeatureGenerator
from .risk_management import DynamicRiskManager
from .rl_trading_env import TradingEnvironment, TradingConfig, SlidingWindowTester


@dataclass
class TransformerRLConfig:
    """Configuration for the unified Transformer+RL strategy"""
    # Model configuration
    transformer_config: TurboDMTConfig = None
    trading_config: TradingConfig = None
    
    # RL training parameters
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training validation
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    min_improvement: float = 0.001
    
    # Walk-forward testing
    train_window: int = 252  # 1 year
    test_window: int = 63   # 3 months
    step_size: int = 21     # 1 month
    
    # Model persistence
    model_save_path: str = "models/transformer_rl_strategy"
    save_frequency: int = 10000  # Save every N timesteps


class TransformerRLCallback(BaseCallback):
    """Custom callback for monitoring RL training progress"""
    
    def __init__(
        self,
        config: TransformerRLConfig,
        validation_env: TradingEnvironment,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.config = config
        self.validation_env = validation_env
        self.best_reward = -np.inf
        self.patience_counter = 0
        self.episode_rewards = []
        self.validation_rewards = []
        
    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Save model periodically
        if self.num_timesteps % self.config.save_frequency == 0:
            save_path = f"{self.config.model_save_path}_step_{self.num_timesteps}.zip"
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.num_timesteps}")
        
        return True
    
    def _on_rollout_end(self) -> bool:
        """Called after each rollout"""
        # Run validation episode
        if hasattr(self, 'validation_env'):
            val_reward = self._validate_model()
            self.validation_rewards.append(val_reward)
            
            # Check for improvement
            if val_reward > self.best_reward + self.config.min_improvement:
                self.best_reward = val_reward
                self.patience_counter = 0
                
                # Save best model
                best_path = f"{self.config.model_save_path}_best.zip"
                self.model.save(best_path)
                if self.verbose > 0:
                    print(f"New best validation reward: {val_reward:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered after {self.patience_counter} episodes without improvement")
                return False
        
        return True
    
    def _validate_model(self) -> float:
        """Run validation episode and return average reward"""
        obs = self.validation_env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.validation_env.step(action)
            total_reward += reward
        
        return total_reward


class TransformerRLStrategy:
    """
    Unified strategy combining TurboDMT v2 transformer with PPO reinforcement learning
    """
    
    def __init__(self, config: TransformerRLConfig = None):
        self.config = config or TransformerRLConfig()
        
        # Initialize default configs if not provided
        if self.config.transformer_config is None:
            self.config.transformer_config = TurboDMTConfig()
        if self.config.trading_config is None:
            self.config.trading_config = TradingConfig()
        
        # Initialize components
        self.transformer_model = None
        self.rl_agent = None
        self.feature_extractor = AdvancedFeatureGenerator()
        self.risk_manager = DynamicRiskManager()
        
        # Training history
        self.training_history = {
            'rewards': [],
            'validation_rewards': [],
            'transformer_losses': [],
            'portfolio_values': []
        }
        
        # Performance tracking
        self.live_trading_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_position': 0.0
        }
    
    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        pretrain_transformer: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the unified Transformer+RL strategy
        
        Args:
            training_data: Historical price data for training
            validation_data: Optional validation data
            pretrain_transformer: Whether to pre-train transformer first
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print("Starting Transformer+RL training pipeline...")
        
        # Step 1: Initialize and pre-train transformer
        if pretrain_transformer:
            if verbose:
                print("Pre-training transformer model...")
            self._pretrain_transformer(training_data, verbose)
        else:
            # Initialize transformer without pre-training
            self.transformer_model = TurboDMTEnsemble(
                self.config.transformer_config,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        
        # Step 2: Prepare validation data
        if validation_data is None and self.config.validation_split > 0:
            split_idx = int(len(training_data) * (1 - self.config.validation_split))
            validation_data = training_data.iloc[split_idx:].copy()
            training_data = training_data.iloc[:split_idx].copy()
        
        # Step 3: Create training environment
        if verbose:
            print("Creating training environment...")
        
        def make_train_env():
            return TradingEnvironment(
                price_data=training_data,
                transformer_model=self.transformer_model,
                config=self.config.trading_config,
                training_mode=True
            )
        
        train_env = DummyVecEnv([make_train_env])
        
        # Step 4: Create validation environment
        validation_env = None
        if validation_data is not None:
            validation_env = TradingEnvironment(
                price_data=validation_data,
                transformer_model=self.transformer_model,
                config=self.config.trading_config,
                training_mode=False
            )
        
        # Step 5: Initialize PPO agent
        if verbose:
            print("Initializing PPO agent...")
        
        self.rl_agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=1 if verbose else 0
        )
        
        # Step 6: Set up callback for monitoring
        callback = None
        if validation_env is not None:
            callback = TransformerRLCallback(
                config=self.config,
                validation_env=validation_env,
                verbose=1 if verbose else 0
            )
        
        # Step 7: Train RL agent
        if verbose:
            print(f"Training RL agent for {self.config.total_timesteps} timesteps...")
        
        # Create model save directory
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        
        self.rl_agent.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            progress_bar=verbose
        )
        
        # Step 8: Final model save
        final_path = f"{self.config.model_save_path}_final.zip"
        self.rl_agent.save(final_path)
        
        if verbose:
            print(f"Training completed. Final model saved to {final_path}")
        
        # Extract training history from callback
        if callback is not None:
            self.training_history['validation_rewards'] = callback.validation_rewards
        
        return self.training_history
    
    def _pretrain_transformer(self, data: pd.DataFrame, verbose: bool = True):
        """Pre-train transformer on historical data patterns"""
        if verbose:
            print("Extracting features for transformer pre-training...")
        
        # Extract features
        features_df = self.feature_extractor.generate_features(data)
        
        # Prepare sequences for training
        seq_len = self.config.transformer_config.seq_len
        sequences = []
        targets = []
        
        for i in range(seq_len, len(features_df)):
            # Input sequence
            seq = features_df.iloc[i-seq_len:i].values
            sequences.append(seq)
            
            # Target: next period return
            next_return = (data.iloc[i]['close'] / data.iloc[i-1]['close']) - 1
            targets.append(next_return)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        if verbose:
            print(f"Created {len(sequences)} training sequences")
        
        # Create transformer model
        self.transformer_model = TurboDMTEnsemble(
            self.config.transformer_config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(targets).unsqueeze(1)
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Pre-train transformer
        if verbose:
            print("Pre-training transformer...")
        
        losses = self.transformer_model.train(dataloader, epochs=20, lr=0.001)
        self.training_history['transformer_losses'] = losses
        
        if verbose:
            print(f"Transformer pre-training completed. Final loss: {losses[-1]:.6f}")
    
    def predict(self, current_data: pd.DataFrame) -> Tuple[int, Dict[str, Any]]:
        """
        Generate trading action based on current market data
        
        Args:
            current_data: Recent price data (must include enough history for features)
            
        Returns:
            Tuple of (action, info_dict)
        """
        if self.rl_agent is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create temporary environment for observation
        temp_env = TradingEnvironment(
            price_data=current_data,
            transformer_model=self.transformer_model,
            config=self.config.trading_config,
            training_mode=False
        )
        
        # Get current observation
        obs = temp_env._get_observation()
        
        # Get action from RL agent
        action, _ = self.rl_agent.predict(obs, deterministic=True)
        
        # Get additional info from transformer
        try:
            # Get transformer predictions
            features_df = self.feature_extractor.generate_features(current_data)
            feature_sequence = features_df.iloc[-30:].values
            
            if len(feature_sequence) < 30:
                padding = np.zeros((30 - len(feature_sequence), feature_sequence.shape[1]))
                feature_sequence = np.vstack([padding, feature_sequence])
            
            feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
            
            with torch.no_grad():
                pred, vol, regime_logits, uncertainty = self.transformer_model.predict(feature_tensor)
                
            info = {
                'action': int(action),
                'transformer_prediction': float(pred.item()),
                'volatility_estimate': float(vol.item()),
                'regime_probabilities': torch.softmax(regime_logits, dim=1).numpy().flatten().tolist(),
                'model_uncertainty': float(uncertainty.item()),
                'observation_shape': obs.shape,
                'current_price': current_data.iloc[-1]['close']
            }
            
        except Exception as e:
            info = {
                'action': int(action),
                'error': str(e),
                'observation_shape': obs.shape,
                'current_price': current_data.iloc[-1]['close']
            }
        
        return int(action), info
    
    def backtest(
        self,
        test_data: pd.DataFrame,
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data
        
        Args:
            test_data: Historical price data for backtesting
            initial_balance: Starting portfolio value
            
        Returns:
            Backtest results dictionary
        """
        if self.rl_agent is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create test environment
        test_config = TradingConfig()
        test_config.initial_balance = initial_balance
        
        test_env = TradingEnvironment(
            price_data=test_data,
            transformer_model=self.transformer_model,
            config=test_config,
            training_mode=False
        )
        
        # Run backtest
        obs = test_env.reset()
        done = False
        
        portfolio_history = []
        action_history = []
        reward_history = []
        
        while not done:
            action, _ = self.rl_agent.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            portfolio_history.append(info['portfolio_value'])
            action_history.append(info['action'])
            reward_history.append(reward)
        
        # Get final statistics
        stats = test_env.get_episode_stats()
        
        # Add additional analysis
        stats.update({
            'portfolio_history': portfolio_history,
            'action_history': action_history,
            'reward_history': reward_history,
            'num_steps': len(portfolio_history)
        })
        
        return stats
    
    def walk_forward_test(
        self,
        data: pd.DataFrame,
        training_epochs: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation using sliding window methodology
        
        Args:
            data: Historical price data
            training_epochs: Number of epochs for each window training
            verbose: Print progress
            
        Returns:
            Walk-forward test results
        """
        def model_factory():
            """Factory function to create new model instances"""
            return {
                'transformer': TurboDMTEnsemble(
                    self.config.transformer_config,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            }
        
        # Create sliding window tester
        tester = SlidingWindowTester(
            data=data,
            train_window=self.config.train_window,
            test_window=self.config.test_window,
            step_size=self.config.step_size
        )
        
        # Run sliding window test
        results = tester.run_sliding_test(
            model_factory=model_factory,
            training_epochs=training_epochs,
            verbose=verbose
        )
        
        # Get aggregate statistics
        aggregate_stats = tester.get_aggregate_results()
        
        return {
            'individual_results': results,
            'aggregate_stats': aggregate_stats,
            'num_windows': len(results)
        }
    
    def save_model(self, filepath: str):
        """Save the complete strategy (transformer + RL agent)"""
        if self.rl_agent is None or self.transformer_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Save RL agent
        rl_path = f"{filepath}_rl_agent.zip"
        self.rl_agent.save(rl_path)
        
        # Save transformer model
        transformer_path = f"{filepath}_transformer.pkl"
        with open(transformer_path, 'wb') as f:
            pickle.dump(self.transformer_model, f)
        
        # Save configuration
        config_path = f"{filepath}_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Model saved to {filepath}_*")
    
    def load_model(self, filepath: str):
        """Load a previously saved strategy"""
        # Load configuration
        config_path = f"{filepath}_config.pkl"
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        
        # Load transformer model
        transformer_path = f"{filepath}_transformer.pkl"
        with open(transformer_path, 'rb') as f:
            self.transformer_model = pickle.load(f)
        
        # Load RL agent
        rl_path = f"{filepath}_rl_agent.zip"
        self.rl_agent = PPO.load(rl_path)
        
        print(f"Model loaded from {filepath}_*")


if __name__ == "__main__":
    # Example usage and testing
    import yfinance as yf
    
    print("Testing Transformer+RL Strategy...")
    
    # Download test data
    data = yf.download("SPY", start="2020-01-01", end="2023-01-01")
    data = data.reset_index()
    
    # Create strategy
    config = TransformerRLConfig()
    config.total_timesteps = 5000  # Reduced for testing
    strategy = TransformerRLStrategy(config)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"Training on {len(train_data)} periods...")
    print(f"Testing on {len(test_data)} periods...")
    
    # Train strategy
    history = strategy.train(train_data, verbose=True)
    
    # Backtest
    print("\nRunning backtest...")
    backtest_results = strategy.backtest(test_data)
    
    print(f"Backtest Results:")
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {backtest_results['win_rate']*100:.1f}%")
    
    # Test single prediction
    print("\nTesting single prediction...")
    recent_data = data.iloc[-100:].copy()  # Last 100 periods
    action, info = strategy.predict(recent_data)
    print(f"Predicted action: {action}")
    print(f"Transformer prediction: {info.get('transformer_prediction', 'N/A')}")
    print(f"Model uncertainty: {info.get('model_uncertainty', 'N/A')}")