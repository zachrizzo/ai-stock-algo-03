#!/usr/bin/env python3
"""
RL Trading Environment for TurboDMT v2
======================================
Custom OpenAI Gym environment for training reinforcement learning agents
on financial markets using the TurboDMT v2 transformer as feature extractor.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from .model import TurboDMTEnsemble, TurboDMTConfig
from .features import AdvancedFeatureGenerator
from .risk_management import DynamicRiskManager


@dataclass
class TradingConfig:
    """Configuration for the trading environment"""
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # Maximum position as fraction of portfolio
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.0005  # 0.05% slippage
    lookback_window: int = 60  # Number of periods to look back
    reward_lookback: int = 20  # Periods to calculate reward over
    risk_free_rate: float = 0.02  # Annual risk-free rate
    target_vol: float = 0.15  # Target annualized volatility
    max_drawdown_threshold: float = 0.20  # Maximum allowed drawdown


class TradingEnvironment(gym.Env):
    """
    Custom trading environment that integrates TurboDMT v2 transformer
    with reinforcement learning for portfolio management.
    
    Action Space: Discrete(9)
        0: Hold (0% change)
        1: Buy 25% (increase position by 25% of portfolio)
        2: Buy 50% (increase position by 50% of portfolio) 
        3: Buy 75% (increase position by 75% of portfolio)
        4: Buy 100% (go to maximum long position)
        5: Sell 25% (decrease position by 25% of portfolio)
        6: Sell 50% (decrease position by 50% of portfolio)
        7: Sell 75% (decrease position by 75% of portfolio)
        8: Sell 100% (close all positions/go to maximum short)
    
    Observation Space: Box(64,)
        - 32 features from TurboDMT transformer output
        - 16 features from current portfolio state
        - 16 features from market regime and risk metrics
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        transformer_model: TurboDMTEnsemble,
        config: TradingConfig = None,
        training_mode: bool = True
    ):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.price_data = price_data.copy()
        self.transformer_model = transformer_model
        self.training_mode = training_mode
        
        # Initialize feature extractor and risk manager
        self.feature_extractor = AdvancedFeatureGenerator()
        self.risk_manager = DynamicRiskManager()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(64,),
            dtype=np.float32
        )
        
        # Trading state
        self.reset()
        
        # Performance tracking
        self.episode_returns = []
        self.episode_positions = []
        self.episode_drawdowns = []
        
    def reset(self, seed=None, options=None) -> np.ndarray:
        """Reset the environment to initial state"""
        # Initialize portfolio state
        self.balance = self.config.initial_balance
        self.position = 0.0  # Current position (-1 to 1, where 1 is max long)
        self.shares_held = 0.0
        self.total_portfolio_value = self.config.initial_balance
        self.peak_portfolio_value = self.config.initial_balance
        self.current_drawdown = 0.0
        
        # Initialize time index
        self.current_step = self.config.lookback_window
        self.max_steps = len(self.price_data) - self.config.lookback_window - 1
        
        # Performance tracking
        self.portfolio_history = deque(maxlen=252)  # 1 year of daily data
        self.return_history = deque(maxlen=self.config.reward_lookback)
        self.action_history = deque(maxlen=10)
        
        # Risk metrics
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading step"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, {}
        
        # Get current price
        current_price = self.price_data.iloc[self.current_step]['close']
        
        # Execute action
        old_position = self.position
        reward, trade_executed = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Update portfolio value
        if self.current_step < len(self.price_data):
            new_price = self.price_data.iloc[self.current_step]['close']
            self._update_portfolio_value(new_price)
        
        # Check if episode should end
        done = self._check_done()
        
        # Prepare info dictionary
        info = {
            'portfolio_value': self.total_portfolio_value,
            'position': self.position,
            'current_price': current_price,
            'action': action,
            'trade_executed': trade_executed,
            'drawdown': self.current_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.profitable_trades / max(1, self.total_trades)
        }
        
        truncated = False  # Add truncated flag for gymnasium compatibility
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> Tuple[float, bool]:
        """Execute the trading action and return reward"""
        old_position = self.position
        old_portfolio_value = self.total_portfolio_value
        
        # Convert action to position change
        target_position = self._action_to_position(action)
        
        # Apply risk management constraints  
        # Simple position validation for now
        if self.current_drawdown > 0.15:  # Reduce position if drawdown > 15%
            target_position *= 0.5
        if self.consecutive_losses > 3:  # Reduce position after consecutive losses
            target_position *= 0.8
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if significant change
        trade_executed = False
        if abs(position_change) > 0.01:  # Only trade if change > 1%
            # Calculate transaction costs
            trade_size = abs(position_change) * self.total_portfolio_value
            transaction_cost = trade_size * self.config.transaction_cost
            slippage_cost = trade_size * self.config.slippage
            
            # Update position and balance
            self.position = target_position
            self.balance -= (transaction_cost + slippage_cost)
            
            # Update shares held
            self.shares_held = self.position * self.total_portfolio_value / current_price
            
            # Track trade
            self.total_trades += 1
            trade_executed = True
            self.action_history.append(action)
        
        # Calculate immediate reward
        reward = self._calculate_reward(old_portfolio_value, trade_executed)
        
        return reward, trade_executed
    
    def _action_to_position(self, action: int) -> float:
        """Convert discrete action to target position"""
        # Ensure action is an integer
        action = int(action) if hasattr(action, '__iter__') else action
        action_map = {
            0: self.position,  # Hold
            1: min(1.0, self.position + 0.25),  # Buy 25%
            2: min(1.0, self.position + 0.50),  # Buy 50%
            3: min(1.0, self.position + 0.75),  # Buy 75%
            4: 1.0,  # Buy 100% (max long)
            5: max(-1.0, self.position - 0.25),  # Sell 25%
            6: max(-1.0, self.position - 0.50),  # Sell 50%
            7: max(-1.0, self.position - 0.75),  # Sell 75%
            8: -1.0  # Sell 100% (max short)
        }
        return action_map.get(action, self.position)
    
    def _update_portfolio_value(self, new_price: float):
        """Update portfolio value based on new price"""
        if self.shares_held != 0:
            position_value = self.shares_held * new_price
            self.total_portfolio_value = self.balance + position_value
        else:
            self.total_portfolio_value = self.balance
        
        # Update peak and drawdown
        if self.total_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.total_portfolio_value
            self.current_drawdown = 0.0
            self.consecutive_losses = 0
        else:
            self.current_drawdown = (self.peak_portfolio_value - self.total_portfolio_value) / self.peak_portfolio_value
            if self.total_portfolio_value < self.portfolio_history[-1] if self.portfolio_history else self.config.initial_balance:
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Track portfolio history
        self.portfolio_history.append(self.total_portfolio_value)
        
        # Calculate return
        if len(self.portfolio_history) > 1:
            period_return = (self.total_portfolio_value / self.portfolio_history[-2]) - 1
            self.return_history.append(period_return)
    
    def _calculate_reward(self, old_portfolio_value: float, trade_executed: bool) -> float:
        """Calculate reward based on risk-adjusted returns and trading behavior"""
        if len(self.return_history) == 0:
            return 0.0
        
        # Base reward: Sharpe ratio over recent period
        returns = np.array(list(self.return_history))
        if len(returns) > 1:
            excess_returns = returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
            sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
            base_reward = sharpe
        else:
            base_reward = returns[-1] if len(returns) > 0 else 0.0
        
        # Risk penalties
        drawdown_penalty = -2.0 * max(0, self.current_drawdown - 0.05)  # Penalty if DD > 5%
        volatility_penalty = 0.0
        
        if len(returns) > 5:
            vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            vol_target = self.config.target_vol
            if vol > vol_target * 1.5:  # Penalty if vol > 1.5x target
                volatility_penalty = -0.5 * (vol - vol_target * 1.5)
        
        # Trading frequency penalty (discourage overtrading)
        trade_penalty = -0.01 if trade_executed else 0.0
        
        # Consecutive loss penalty
        consecutive_loss_penalty = -0.1 * max(0, self.consecutive_losses - 3)
        
        # Position sizing reward (encourage dynamic position sizing)
        position_reward = 0.0
        if abs(self.position) < 0.95:  # Reward for not being at maximum position always
            position_reward = 0.02
        
        # Combine all components
        total_reward = (
            base_reward + 
            drawdown_penalty + 
            volatility_penalty + 
            trade_penalty + 
            consecutive_loss_penalty + 
            position_reward
        )
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= len(self.price_data):
            return np.zeros(64, dtype=np.float32)
        
        # Get historical price data
        start_idx = max(0, self.current_step - self.config.lookback_window)
        historical_data = self.price_data.iloc[start_idx:self.current_step+1].copy()
        
        # Extract features using TurboDMT feature extractor
        try:
            features_df = self.feature_extractor.generate_features(historical_data)
            
            # Get transformer encoding (last 30 periods)
            feature_sequence = features_df.iloc[-30:].values  # Last 30 periods
            if len(feature_sequence) < 30:
                # Pad with zeros if insufficient data
                padding = np.zeros((30 - len(feature_sequence), feature_sequence.shape[1]))
                feature_sequence = np.vstack([padding, feature_sequence])
            
            # Convert to tensor and get transformer output
            feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)  # Add batch dim
            
            with torch.no_grad():
                pred, vol, regime_logits, uncertainty = self.transformer_model.predict(feature_tensor)
                transformer_output = torch.cat([
                    pred.flatten(),
                    vol.flatten(), 
                    torch.softmax(regime_logits, dim=1).flatten(),
                    uncertainty.flatten()
                ])[:32]  # Take first 32 features
        
        except Exception as e:
            # Fallback to zero features if extraction fails
            transformer_output = torch.zeros(32)
        
        # Portfolio state features (16 features)
        portfolio_features = np.array([
            self.position,  # Current position
            self.total_portfolio_value / self.config.initial_balance,  # Portfolio value ratio
            self.current_drawdown,  # Current drawdown
            self.balance / self.total_portfolio_value,  # Cash ratio
            len(self.return_history) / self.config.reward_lookback,  # Data completeness
            self.consecutive_losses / 10.0,  # Normalized consecutive losses
            self.total_trades / 100.0,  # Normalized trade count
            self.profitable_trades / max(1, self.total_trades),  # Win rate
            np.mean(list(self.return_history)) if self.return_history else 0.0,  # Average return
            np.std(list(self.return_history)) if len(self.return_history) > 1 else 0.0,  # Return volatility
            min(1.0, max(-1.0, np.sum(list(self.action_history)) / max(1, len(self.action_history)))),  # Average action
            self.peak_portfolio_value / self.config.initial_balance,  # Peak portfolio ratio
            (self.current_step / self.max_steps) if self.max_steps > 0 else 0.0,  # Time progress
            0.0,  # Reserved
            0.0,  # Reserved  
            0.0   # Reserved
        ], dtype=np.float32)
        
        # Market regime and risk features (16 features)
        current_price = self.price_data.iloc[self.current_step]['close']
        
        # Calculate some basic market features
        recent_prices = self.price_data.iloc[max(0, self.current_step-20):self.current_step+1]['close']
        
        market_features = np.array([
            (current_price / recent_prices.iloc[0] - 1) if len(recent_prices) > 1 else 0.0,  # 20-day return
            recent_prices.std() / recent_prices.mean() if len(recent_prices) > 1 else 0.0,  # CV
            (recent_prices.iloc[-1] / recent_prices.mean() - 1) if len(recent_prices) > 1 else 0.0,  # Price vs MA
            self.config.max_drawdown_threshold - self.current_drawdown,  # Drawdown buffer
            min(recent_prices) / max(recent_prices) if len(recent_prices) > 1 else 1.0,  # Price range ratio
            0.0,  # Reserved for regime classification
            0.0,  # Reserved for volatility regime
            0.0,  # Reserved for trend strength
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            0.0,  # Reserved
            0.0   # Reserved
        ], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([
            transformer_output.numpy() if isinstance(transformer_output, torch.Tensor) else transformer_output,
            portfolio_features,
            market_features
        ])
        
        # Ensure observation is the right size
        if len(observation) != 64:
            observation = np.resize(observation, 64)
        
        return observation.astype(np.float32)
    
    def _check_done(self) -> bool:
        """Check if episode should terminate"""
        # End if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # End if excessive drawdown
        if self.current_drawdown > self.config.max_drawdown_threshold:
            return True
        
        # End if portfolio value too low
        if self.total_portfolio_value < self.config.initial_balance * 0.5:
            return True
        
        return False
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for the completed episode"""
        if len(self.portfolio_history) == 0:
            return {}
        
        portfolio_values = np.array(list(self.portfolio_history))
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0.0
        
        max_dd = 0.0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': self.total_trades,
            'win_rate': self.profitable_trades / max(1, self.total_trades),
            'final_portfolio_value': portfolio_values[-1],
            'max_consecutive_losses': self.max_consecutive_losses
        }


class SlidingWindowTester:
    """
    Implements sliding window (screen door) testing methodology
    for walk-forward validation of RL trading strategies
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        train_window: int = 252,  # 1 year training
        test_window: int = 63,   # 3 months testing  
        step_size: int = 21,     # 1 month step
        min_train_size: int = 126  # Minimum 6 months training
    ):
        self.data = data
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        
        self.results = []
        
    def run_sliding_test(
        self,
        model_factory,  # Function that returns a new model instance
        training_epochs: int = 50,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run sliding window testing with accelerated simulation
        """
        total_windows = (len(self.data) - self.train_window - self.test_window) // self.step_size
        
        if verbose:
            print(f"Running sliding window test with {total_windows} windows")
            print(f"Train window: {self.train_window}, Test window: {self.test_window}")
            print(f"Step size: {self.step_size}")
        
        for window_idx in range(total_windows):
            start_idx = window_idx * self.step_size
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window
            
            if test_end_idx >= len(self.data):
                break
                
            # Extract training and testing data
            train_data = self.data.iloc[start_idx:train_end_idx].copy()
            test_data = self.data.iloc[train_end_idx:test_end_idx].copy()
            
            if verbose:
                print(f"\nWindow {window_idx + 1}/{total_windows}")
                print(f"Training: {train_data.index[0]} to {train_data.index[-1]}")
                print(f"Testing: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Create and train model
            try:
                model, train_stats = self._train_model_window(
                    train_data, model_factory, training_epochs, verbose
                )
                
                # Test model
                test_stats = self._test_model_window(test_data, model, verbose)
                
                # Store results
                window_result = {
                    'window_idx': window_idx,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'train_stats': train_stats,
                    'test_stats': test_stats
                }
                
                self.results.append(window_result)
                
                if verbose:
                    print(f"Test Sharpe: {test_stats.get('sharpe_ratio', 0):.3f}, "
                          f"Return: {test_stats.get('total_return', 0)*100:.2f}%, "
                          f"Max DD: {test_stats.get('max_drawdown', 0)*100:.2f}%")
                
            except Exception as e:
                if verbose:
                    print(f"Error in window {window_idx}: {e}")
                continue
        
        return self.results
    
    def _train_model_window(
        self, 
        train_data: pd.DataFrame, 
        model_factory, 
        epochs: int,
        verbose: bool
    ) -> Tuple[Any, Dict[str, float]]:
        """Train model on a single window"""
        # Create model and environment
        model = model_factory()
        
        # Import here to avoid circular imports
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create training environment
        def make_env():
            return TradingEnvironment(
                price_data=train_data,
                transformer_model=model['transformer'],
                config=TradingConfig(),
                training_mode=True
            )
        
        env = DummyVecEnv([make_env])
        
        # Create PPO agent
        ppo_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0
        )
        
        # Train agent
        total_timesteps = min(10000, len(train_data) * 2)  # Accelerated training
        ppo_agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        train_env = make_env()
        train_env.reset()
        
        # Quick evaluation on training data
        done = False
        while not done:
            obs = train_env._get_observation()
            action, _ = ppo_agent.predict(obs, deterministic=True)
            obs, reward, done, info = train_env.step(action)
        
        train_stats = train_env.get_episode_stats()
        
        return {'ppo_agent': ppo_agent, 'transformer': model['transformer']}, train_stats
    
    def _test_model_window(
        self,
        test_data: pd.DataFrame,
        model: Dict[str, Any],
        verbose: bool
    ) -> Dict[str, float]:
        """Test model on a single window"""
        # Create test environment
        test_env = TradingEnvironment(
            price_data=test_data,
            transformer_model=model['transformer'],
            config=TradingConfig(),
            training_mode=False
        )
        
        # Run episode
        obs = test_env.reset()
        done = False
        
        while not done:
            action, _ = model['ppo_agent'].predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
        
        return test_env.get_episode_stats()
    
    def get_aggregate_results(self) -> Dict[str, float]:
        """Get aggregate statistics across all windows"""
        if not self.results:
            return {}
        
        # Extract test statistics
        test_returns = [r['test_stats'].get('total_return', 0) for r in self.results]
        test_sharpes = [r['test_stats'].get('sharpe_ratio', 0) for r in self.results]
        test_drawdowns = [r['test_stats'].get('max_drawdown', 0) for r in self.results]
        test_vol = [r['test_stats'].get('volatility', 0) for r in self.results]
        
        return {
            'mean_return': np.mean(test_returns),
            'std_return': np.std(test_returns),
            'mean_sharpe': np.mean(test_sharpes),
            'std_sharpe': np.std(test_sharpes),
            'mean_max_dd': np.mean(test_drawdowns),
            'worst_drawdown': np.max(test_drawdowns),
            'mean_volatility': np.mean(test_vol),
            'win_rate': np.mean([1 if r > 0 else 0 for r in test_returns]),
            'num_windows': len(self.results)
        }


if __name__ == "__main__":
    # Test the environment
    import yfinance as yf
    
    # Download test data
    data = yf.download("SPY", start="2020-01-01", end="2023-01-01")
    data = data.reset_index()
    
    # Create transformer model
    config = TurboDMTConfig()
    transformer = TurboDMTEnsemble(config)
    
    # Create environment
    env = TradingEnvironment(
        price_data=data,
        transformer_model=transformer,
        config=TradingConfig()
    )
    
    # Test random actions
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.4f}, Done={done}")
        if done:
            break
    
    stats = env.get_episode_stats()
    print(f"Episode stats: {stats}")