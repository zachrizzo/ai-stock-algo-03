#!/usr/bin/env python3
"""
DMT_v4 Model with Transformer-RL Hybrid
---------------------------------------
Enhanced trading model that combines transformer pattern recognition with 
reinforcement learning for adaptive decision making and risk management.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import traceback
import time
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("DMT_v4")

# ————————————————————————————————————————————————————————————
# Configuration constants
# ————————————————————————————————————————————————————————————
DAILY_RISK_LIMIT = 0.02  # 2% of equity max daily loss
TRADE_RISK_LIMIT = 0.005  # 0.5% per position max risk
MIN_BARS_FOR_TRAINING = 200  # Absolute minimum to train

# ————————————————————————————————————————————————————————————
# Data handling
# ————————————————————————————————————————————————————————————
class BarDataset(Dataset):
    """Minute‑bar dataset that forms sequences of length `context`"""
    def __init__(self, df: pd.DataFrame, context: int = 512):
        self.ctx = context
        self.df = df.dropna().reset_index(drop=True)
        self.x_cols = [c for c in df.columns if c not in ("future_return",)]

    def __len__(self):
        return len(self.df) - self.ctx

    def __getitem__(self, idx):
        x = self.df.loc[idx:idx+self.ctx-1, self.x_cols].values.astype(np.float32)
        y = self.df.loc[idx+self.ctx, "future_return"].astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y)

# ————————————————————————————————————————————————————————————
# Model
# ————————————————————————————————————————————————————————————
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, :x.size(1)]
        return x

class DMTTransformer(nn.Module):
    def __init__(self, num_feats: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(num_feats, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()  # output in -1..1
        )

    def forward(self, x: torch.Tensor):  # x: [B, T, F]
        x = self.in_proj(x)
        x = self.pos_enc(x)
        enc = self.encoder(x)
        cls = enc[:, -1]  # last token pooling
        out = self.head(cls).squeeze(-1)
        return out

class DecisionTransformer(nn.Module):
    """
    The Decision Transformer architecture for reinforcement learning
    
    Based on the paper "Decision Transformer: Reinforcement Learning via Sequence Modeling"
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        max_length: int = 20,
        n_layers: int = 3,
        n_heads: int = 2,
        n_positions: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Embeddings for states, actions, returns, and timesteps
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        self.timestep_encoder = nn.Embedding(n_positions, hidden_size)
        
        # Position embeddings
        self.position_encoder = nn.Embedding(n_positions, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=4*hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Forward pass through the model
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len, 1)
            timesteps: (batch_size, seq_len)
            
        Returns:
            action_preds: (batch_size, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Ensure conversion to PyTorch tensors
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        if not isinstance(returns_to_go, torch.Tensor):
            returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.long)
        
        # Encode states, actions, returns
        state_embeddings = self.state_encoder(states)  # (batch, seq_len, hidden_size)
        action_embeddings = self.action_encoder(actions)  # (batch, seq_len, hidden_size)
        returns_embeddings = self.return_encoder(returns_to_go)  # (batch, seq_len, hidden_size)
        
        # Encode positions and timesteps
        position_embeddings = self.position_encoder(torch.arange(seq_len, device=states.device).unsqueeze(0).repeat(batch_size, 1))
        time_embeddings = self.timestep_encoder(timesteps)
        
        # Sequence representation: interleave state, action, return
        sequence = torch.zeros(
            (batch_size, 3 * seq_len, self.hidden_size),
            device=states.device,
            dtype=torch.float32
        )
        
        # Alternating pattern: 
        # (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_t, s_t, a_t)
        sequence[:, 0::3, :] = returns_embeddings + time_embeddings
        sequence[:, 1::3, :] = state_embeddings + position_embeddings
        sequence[:, 2::3, :] = action_embeddings + position_embeddings
        
        # Apply transformer to the sequence
        transformer_outputs = self.transformer(sequence)
        
        # Extract action predictions from every third position
        action_preds = self.action_head(transformer_outputs[:, 2::3, :])
        
        return action_preds
    
    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Get predicted action for the last timestep
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len, 1)
            timesteps: (batch_size, seq_len)
            
        Returns:
            action: (batch_size, action_dim)
        """
        # Forward pass
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        # Return action for the last timestep
        return action_preds[:, -1].detach().numpy()
    
    def train_on_batch(self, batch):
        """
        Train on a batch of data
        
        Args:
            batch: Dictionary containing 'states', 'actions', 'returns_to_go', 'timesteps'
            
        Returns:
            loss: Training loss
        """
        states = batch['states']
        actions = batch['actions'] 
        returns_to_go = batch['returns_to_go']
        timesteps = batch['timesteps']
        
        # Forward pass
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        
        # Compute loss
        loss = F.mse_loss(action_preds, actions)
        
        return loss

# ————————————————————————————————————————————————————————————
# Trade State & Risk Management
# ————————————————————————————————————————————————————————————
class TradeState:
    """Encapsulates trade state information and risk management"""
    def __init__(self):
        # Risk/capital tracking
        self.equity = 1.0  # Normalized to 1.0 for relative calculations
        self.daily_pnl = 0.0
        self.current_position = 0.0  # -1.0 to 1.0 (% of max position)
        
        # Trade tracking
        self.entry_price = 0.0
        self.last_price = 0.0
        self.stop_loss = 0.0
        self.trailing_stop = 0.0
        self.win_streak = 0
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Signal tracking
        self.last_signal = 0.0
        self.last_trade_time = 0
        self.pending_signal = None
        self.pending_count = 0
        self.regime = "neutral"  # bull, bear, neutral
    
    def reset_daily(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
    
    def update_trade_result(self, is_win: bool, pnl: float):
        """Update trade statistics"""
        self.total_trades += 1
        self.daily_pnl += pnl
        self.equity += pnl  # Update equity
        
        if is_win:
            self.win_count += 1
            self.win_streak = max(0, self.win_streak + 1)
        else:
            self.loss_count += 1
            self.win_streak = 0
    
    def calculate_position_size(self, atr: float, price: float, signal_strength: float) -> float:
        """Calculate appropriate position size based on risk parameters"""
        # Risk-based position sizing
        dollar_risk_per_unit = atr
        if dollar_risk_per_unit <= 0:
            return 0.0
            
        # Base max position on risk cap
        max_position = (TRADE_RISK_LIMIT * self.equity) / dollar_risk_per_unit
        
        # Scale by win streak (momentum) up to 50% more
        streak_factor = min(1.5, 1.0 + (self.win_streak * 0.1))
        
        # Scale by signal strength
        signal_factor = abs(signal_strength)
        
        # Check daily risk cap
        if self.daily_pnl <= -DAILY_RISK_LIMIT * self.equity:
            return 0.0  # Hit daily loss limit
            
        # Final position size, capped at 2.0 (200% of equity, means up to 2x leverage)
        final_size = min(2.0, max_position * streak_factor * signal_factor)
        return final_size

    def set_stops(self, entry_price: float, atr: float, is_long: bool, 
                  atr_stop_multiplier: float = 3.0):
        """Set stop-loss and trailing stop levels"""
        self.entry_price = entry_price
        
        if is_long:
            self.stop_loss = entry_price - (atr * atr_stop_multiplier)
            self.trailing_stop = self.stop_loss
        else:
            self.stop_loss = entry_price + (atr * atr_stop_multiplier)
            self.trailing_stop = self.stop_loss
    
    def update_stops(self, current_price: float, atr: float, is_long: bool):
        """Update trailing stop based on current price"""
        self.last_price = current_price
        
        # Only move the trailing stop in profitable direction
        if is_long and current_price > self.entry_price:
            new_stop = current_price - (atr * 2.0)  # Tighter trailing stop (2 ATR)
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        elif not is_long and current_price < self.entry_price:
            new_stop = current_price + (atr * 2.0)
            if new_stop < self.trailing_stop or self.trailing_stop == 0:
                self.trailing_stop = new_stop
    
    def check_stop_hit(self, current_price: float, is_long: bool) -> bool:
        """Check if stop-loss or trailing stop is hit"""
        if is_long and current_price < self.trailing_stop:
            return True
        elif not is_long and current_price > self.trailing_stop:
            return True
        return False

# ————————————————————————————————————————————————————————————
# RL Environment for training
# ————————————————————————————————————————————————————————————
class MarketEnv:
    """Trading environment for reinforcement learning with realistic market dynamics"""
    
    def __init__(self, price_data, features, initial_balance=10000.0, max_position=1.0, 
                 fee_rate=0.001, slippage=0.0001, window_size=30, reward_scaling=1.0):
        """
        Initialize the market environment
        
        Args:
            price_data: DataFrame with OHLCV data
            features: Engineered features for the agent
            initial_balance: Starting capital
            max_position: Maximum position size (1.0 = 100% of capital)
            fee_rate: Trading fee as decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as decimal of price (e.g., 0.0001 = 0.01%)
            window_size: Look-back window for observations
            reward_scaling: Scale factor for rewards
        """
        self.price_data = price_data
        self.features = features
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position = max_position
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Current state
        self.position = 0.0  # -1.0 (full short) to 1.0 (full long)
        self.entry_price = 0.0
        self.current_step = window_size
        self.done = False
        
        # Performance tracking
        self.equity_curve = [initial_balance]
        self.returns = []
        self.positions = [0.0]
        self.trades = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        
        # Risk management
        self.daily_losses = 0.0
        self.last_trade_pnl = 0.0
        self.consecutive_losses = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Store ATR values for stop placement
        if 'atr_14' in self.price_data.columns:
            self.atr_history = self.price_data['atr_14'].values
        else:
            # Calculate it if not available
            self.atr_history = self._calculate_atr(price_data)
    
    def reset(self):
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.done = False
        self.equity_curve = [self.initial_balance]
        self.returns = []
        self.positions = [0.0]
        self.trades = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        self.daily_losses = 0.0
        self.consecutive_losses = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment with the given action
        
        Args:
            action: Value between -1.0 (full short) and 1.0 (full long)
                   representing the target position as % of capital
        
        Returns:
            next_observation, reward, done, info
        """
        # Ensure action is within bounds
        action = max(-self.max_position, min(self.max_position, action))
        
        # Get current price data
        current_price = self.price_data['close'].iloc[self.current_step]
        current_atr = self.atr_history[self.current_step] if len(self.atr_history) > self.current_step else current_price * 0.01
        
        # Calculate equity before trade
        pre_trade_equity = self.balance + self.position * current_price
        
        # Execute trade (calculate position delta and apply fees/slippage)
        position_delta = action - self.position
        
        # Risk checks before executing trade
        daily_loss_exceeded = self._check_daily_loss_limit(pre_trade_equity)
        stop_loss_hit = self._check_stop_loss(current_price, current_atr)
        
        # If risk limits are exceeded, force close position
        if daily_loss_exceeded or stop_loss_hit:
            action = 0.0
            position_delta = -self.position
        
        # Execute the trade
        if abs(position_delta) > 0.001:  # Threshold to avoid tiny trades
            # Apply slippage (worse price when entering position)
            effective_price = current_price * (1 + self.slippage * np.sign(position_delta))
            
            # Apply trading fees
            fee_amount = abs(position_delta * effective_price * self.fee_rate)
            self.balance -= fee_amount
            
            # Update position and equity
            if self.position == 0.0 and position_delta != 0:
                # New position - record entry price
                self.entry_price = effective_price
            elif self.position != 0.0 and position_delta != 0:
                # Calculate P&L from position change
                trade_pnl = self.position * (current_price - self.entry_price)
                self.balance += trade_pnl
                
                # Record trade result
                trade_result = {
                    'entry_price': self.entry_price,
                    'exit_price': effective_price,
                    'position': self.position,
                    'pnl': trade_pnl,
                    'step': self.current_step
                }
                self.trades.append(trade_result)
                
                # Update win/loss streaks
                self._update_trade_statistics(trade_pnl)
                
                # If fully closing or reversing, reset entry price
                if self.position * position_delta <= 0:
                    self.entry_price = effective_price if position_delta != 0 else 0.0
            
            # Update position
            self.position = action
            self.positions.append(self.position)
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= len(self.price_data) - 1:
            self.done = True
        
        # Calculate current equity and returns
        next_price = self.price_data['close'].iloc[self.current_step]
        current_equity = self.balance + self.position * next_price
        self.equity_curve.append(current_equity)
        
        # Calculate return for this step
        step_return = (current_equity / self.equity_curve[-2]) - 1 if len(self.equity_curve) > 1 else 0
        self.returns.append(step_return)
        
        # Update drawdown calculations
        self._update_drawdown(current_equity)
        
        # Calculate reward with risk-aware penalty function
        reward = self._calculate_reward(step_return, current_equity, pre_trade_equity)
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'price': next_price,
            'position': self.position,
            'equity': current_equity,
            'return': step_return,
            'drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'balance': self.balance,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
        
        return next_observation, reward, self.done, info
    
    def _get_observation(self):
        """Construct the state observation for the agent"""
        # Get price features for window
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.window_size)
        
        # Price and volume features
        price_window = self.price_data.iloc[start_idx:end_idx+1]
        
        # Technical indicators
        feature_window = self.features.iloc[start_idx:end_idx+1]
        
        # Current position and equity
        position_info = np.array([
            self.position,  # Current position (-1 to 1)
            self.balance / self.initial_balance,  # Normalized balance
            (self.entry_price / price_window['close'].iloc[-1]) - 1 if self.entry_price > 0 else 0,  # Entry price relative to current
            self.current_drawdown,  # Current drawdown
            self.consecutive_losses / 5.0  # Normalized consecutive losses
        ])
        
        # Combine the observations
        observation = {
            'price': price_window.values,
            'features': feature_window.values,
            'position': position_info
        }
        
        return observation
    
    def _calculate_reward(self, step_return, current_equity, prev_equity):
        """
        Calculate reward with risk-aware shaping
        Uses a combination of returns and penalties for drawdowns
        """
        # Base reward is the step return scaled up for RL learning
        base_reward = step_return * self.reward_scaling * 100  # Scale up for learning
        
        # Add asymmetric penalty for drawdowns
        drawdown_penalty = max(0, self.current_drawdown * 200)  # Heavier penalty for drawdowns
        
        # Adjust rewards for position changes
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step - 1:
            last_trade = self.trades[-1]
            # Encourage taking profits, penalize taking losses
            if last_trade['pnl'] > 0:
                base_reward += 0.1  # Small bonus for winning trades
            else:
                base_reward -= 0.2  # Larger penalty for losing trades
        
        # Penalize excessive trading
        if len(self.positions) > 2 and self.positions[-1] != self.positions[-2]:
            # Changed position - small penalty to discourage overtrading
            base_reward -= 0.05
        
        # Final reward is base return minus penalties
        final_reward = base_reward - drawdown_penalty
        
        return final_reward
    
    def _update_drawdown(self, current_equity):
        """Update drawdown calculations"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Update max drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
    
    def _update_trade_statistics(self, trade_pnl):
        """Update trade statistics for win/loss streaks"""
        self.last_trade_pnl = trade_pnl
        
        if trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            # Accumulate daily losses
            self.daily_losses += abs(trade_pnl)
    
    def _check_daily_loss_limit(self, current_equity):
        """Check if daily loss limit is exceeded"""
        daily_loss_limit = self.initial_balance * 0.02  # 2% daily loss limit
        return self.daily_losses > daily_loss_limit
    
    def _check_stop_loss(self, current_price, current_atr):
        """Check if stop loss is hit"""
        if self.position == 0.0 or self.entry_price == 0.0:
            return False
        
        # Calculate stop distance based on ATR
        stop_distance = current_atr * 1.5  # 1.5 ATR stop distance
        
        if self.position > 0:  # Long position
            stop_level = self.entry_price - stop_distance
            return current_price < stop_level
        else:  # Short position
            stop_level = self.entry_price + stop_distance
            return current_price > stop_level
    
    def _calculate_atr(self, price_data, period=14):
        """Calculate ATR if not provided in price_data"""
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        
        # True Range calculation
        tr1 = high[1:] - low[1:]
        tr2 = abs(high[1:] - close[:-1])
        tr3 = abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Average True Range
        atr = np.zeros_like(close)
        atr[:period] = np.mean(tr[:period])  # Initialize first values
        for i in range(period, len(atr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period
        
        return atr

# ————————————————————————————————————————————————————————————
# Wrapper class (train / inference)
# ————————————————————————————————————————————————————————————
class DMTModel:
    """DMT Trading Model v4 - Optimized for Short-Term Trading With Short-Selling"""
    
    def __init__(self, symbol, interval, version="short_term", 
                 lookback_period=30, context_window=30, 
                 target_annual_vol=0.40, max_position_size=1.5, 
                 neutral_zone=0.02, min_holding_minutes=5, 
                 consecutive_signals=2, transformer_dims=64, 
                 attention_heads=4, num_layers=3, multi_timeframe=True, 
                 signal_smoothing=0.4, max_win_streak_boost=1.3, 
                 allow_short=True, short_bias_adjustment=0.9, 
                 max_short_position=1.5, stop_loss_atr_multiplier=1.5, 
                 trailing_stop_atr_multiplier=2.0, take_profit_atr_multiplier=3.0, 
                 max_daily_loss_pct=2.0, max_trade_risk_pct=0.5):
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        self.context_window = context_window
        self.last_signal = 0.0     # For signal smoothing
        
        # Transaction history for minimum holding period
        self.last_trade_time = 0
        self.last_trade_type = None
        
        # Performance parameters
        self.target_annual_vol = target_annual_vol
        self.max_position_size = max_position_size
        self.neutral_zone = neutral_zone
        self.min_holding_minutes = min_holding_minutes
        self.consecutive_signals = consecutive_signals
        
        # Architecture parameters
        self.transformer_dims = transformer_dims
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.multi_timeframe = multi_timeframe
        
        # Signal processing
        self.signal_smoothing = signal_smoothing
        
        # Win streak boosting
        self.max_win_streak_boost = max_win_streak_boost
        self.current_win_streak = 0
        self.max_win_streak = 0
        
        # Short selling parameters
        self.allow_short = allow_short
        self.short_bias_adjustment = short_bias_adjustment
        self.max_short_position = max_short_position
        
        # Stop-loss and take-profit parameters
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        
        # Risk management
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trade_risk_pct = max_trade_risk_pct
        
        # Signal history for consecutive signal requirement
        self.signal_history = [0] * self.consecutive_signals
        
        # Initial feature weights - enhanced for intraday
        self.weights = {
            'momentum': 0.25,
            'rsi': 0.20,
            'volume': 0.15,
            'regime': 0.10,
            'vwap': 0.15,    # Added VWAP for intraday
            'bb_signal': 0.15  # Added Bollinger Band signals
        }
        
        # For dynamic position sizing
        self.short_vol_window = 5    # Reduced for short-term responsiveness
        self.long_vol_window = 15    # Reduced for short-term responsiveness
        self.current_regime = "neutral"  # Current market regime
        self.current_position_scale = 1.0  # Dynamic position sizing scale
        self.volatility_adjustment = 1.0  # Volatility-based position sizing
        self.regime_change_threshold = 1.5  # Reduced % threshold for faster regime detection
        
        # Trend detection windows - shorter for intraday
        self.trend_short_window = 10
        self.trend_long_window = 25
        
        # Transformer-like attention mechanism
        self.attention_weights = None  # Will be initialized during training
        self.historical_context = []  # Historical context for transformer memory
        
        # Feature scaler
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Stop-loss and take-profit levels
        self.stop_loss_level = None
        self.take_profit_level = None
        self.trailing_stop_level = None
        
        # Daily tracking
        self.daily_pl = 0.0
        self.daily_trades = 0
        self.daily_start_equity = None
        
        # Create logger
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        
        # Initialize trade state
        self.trade_state = TradeState()
        
        # Initialize model attribute to null
        self.model = None

    # —————————————————— feature engineering ——————————————————
    def _add_indicators(self, df):
        """Calculate technical indicators optimized for short-term trading with shorting signals"""
        # Calculate basic returns
        df['returns'] = df['close'].pct_change()
        
        # Short-term momentum indicators (fast-response)
        df['mom_1'] = df['close'].pct_change(1)
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        
        # Momentum for short-term signals
        df['momentum'] = df['close'] / df['close'].shift(10) - 1

        # RSI - standard (faster version with EWM)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Faster RSI for short-term trading (7-period)
        avg_gain_fast = gain.ewm(com=6, adjust=False).mean()
        avg_loss_fast = loss.ewm(com=6, adjust=False).mean()
        rs_fast = np.where(avg_loss_fast != 0, avg_gain_fast / avg_loss_fast, 100)
        df['rsi_fast'] = 100 - (100 / (1 + rs_fast))
        
        # VWAP calculation (key for intraday trading)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # VWAP bands for mean reversion signals
        vwap_std = df['close'].rolling(20).std()
        df['vwap_upper'] = df['vwap'] + (vwap_std * 1.5)
        df['vwap_lower'] = df['vwap'] - (vwap_std * 1.5)
        
        # VWAP distance (normalized)
        df['vwap_dist'] = (df['close'] - df['vwap']) / vwap_std
        
        # Bollinger Bands (key for volatility-based signals)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (bb_std * 2)
        df['bb_lower'] = df['sma_20'] - (bb_std * 2)
        
        # Bollinger Band %B (position within bands - great for mean reversion)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band Width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Bollinger Band signal (1 = overbought, -1 = oversold, 0 = neutral)
        df['bb_signal'] = np.where(df['close'] > df['bb_upper'], 1, 
                                 np.where(df['close'] < df['bb_lower'], -1, 0))
        
        # Short-term volatility (for position sizing)
        df['volatility_5'] = df['returns'].rolling(window=5).std() * np.sqrt(252 * 24 * 60)
        df['volatility_15'] = df['returns'].rolling(window=15).std() * np.sqrt(252 * 24 * 60)
        
        # ATR for stop-loss calculations (key for risk management)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        # Normalized ATR (as % of price)
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Volume indicators for liquidity analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volume spikes (for breakout detection)
        df['volume_spike'] = df['volume'] / df['volume'].rolling(10).mean()
        
        # Short-term regime indicators based on price trends
        df['regime'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # Short squeeze potential (for shorting strategies)
        # High RSI + Decreasing volume + Price near upper BB = potential short opportunity
        df['short_squeeze_potential'] = ((df['rsi'] > 70) & 
                                       (df['volume_ratio'] < 0.8) & 
                                       (df['close'] > df['bb_upper'] * 0.98)).astype(int)
        
        # Long squeeze potential (for covering shorts or going long)
        # Low RSI + Increasing volume + Price near lower BB = potential long opportunity
        df['long_squeeze_potential'] = ((df['rsi'] < 30) & 
                                      (df['volume_ratio'] > 1.2) & 
                                      (df['close'] < df['bb_lower'] * 1.02)).astype(int)
        
        # Fill NaN values with appropriate defaults
        for col in df.columns:
            if df[col].isna().any():
                if col in ['momentum', 'vwap_dist', 'regime', 'short_squeeze_potential', 'long_squeeze_potential']:
                    df[col] = df[col].fillna(0)
                elif col in ['rsi', 'rsi_fast']:
                    df[col] = df[col].fillna(50)
                elif col.startswith('volatility') or col.startswith('atr'):
                    df[col] = df[col].fillna(df[col].mean())
                elif col.startswith('bb_') or col.startswith('volume'):
                    df[col] = df[col].fillna(1.0)
                else:
                    df[col] = df[col].bfill().ffill()
        
        return df
    
    def _analyze_market_regimes(self, df):
        """
        Enhanced market regime analysis for short-term trading
        Faster detection of regime changes with separate short-selling bias
        """
        try:
            # Must have enough data
            if len(df) < max(self.short_vol_window, self.long_vol_window):
                self.current_regime = "neutral"
                self.current_position_scale = 1.0
                self.volatility_adjustment = 1.0
                return
                
            # Calculate short and long-term volatility (faster windows for intraday)
            returns = df['close'].pct_change().dropna()
            short_vol = returns.iloc[-self.short_vol_window:].std() * np.sqrt(252 * 24 * 60)
            long_vol = returns.iloc[-self.long_vol_window:].std() * np.sqrt(252 * 24 * 60)
            
            # Calculate intraday trend indicators
            sma_short = df['close'].rolling(window=self.trend_short_window).mean().iloc[-1]
            sma_long = df['close'].rolling(window=self.trend_long_window).mean().iloc[-1]
            
            # VWAP relative position (important for intraday)
            vwap = df['vwap'].iloc[-1] if 'vwap' in df.columns else df['close'].iloc[-1]
            vwap_position = (df['close'].iloc[-1] - vwap) / df['close'].iloc[-1]
            
            # Calculate very short-term momentum (1-5 bars)
            mom_1 = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1 if len(df) > 2 else 0
            mom_5 = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1 if len(df) > 6 else 0
            
            # Calculate price position within Bollinger Bands (mean reversion indicator)
            bb_position = df['bb_pct_b'].iloc[-1] if 'bb_pct_b' in df.columns else 0.5
            
            # Calculate volume trend
            vol_trend = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean() if len(df) > 10 else 1.0
            
            # RAPID REGIME CLASSIFICATION - optimized for intraday
            
            # Strong bullish signals:
            # 1. Price > SMAs and accelerating
            # 2. Above VWAP with increasing volume
            # 3. Recent momentum positive across timeframes
            if (sma_short > sma_long * 1.005 and 
                vwap_position > 0.001 and 
                mom_1 > 0 and mom_5 > 0 and 
                vol_trend > 1.1):
                
                if bb_position > 0.8:  # Near upper BB - potential reversal zone
                    self.current_regime = "bullish_extended"  # Bullish but extended
                    self.current_position_scale = 1.2  # Slightly reduced from peak
                else:
                    self.current_regime = "strong_bullish"  # Strong sustained uptrend
                    self.current_position_scale = 1.5  # Full bullish exposure
            
            # Strong bearish signals:
            # 1. Price < SMAs and accelerating downward
            # 2. Below VWAP with increasing volume
            # 3. Recent momentum negative across timeframes
            elif (sma_short < sma_long * 0.995 and 
                 vwap_position < -0.001 and 
                 mom_1 < 0 and mom_5 < 0 and
                 vol_trend > 1.1):
                
                if bb_position < 0.2:  # Near lower BB - potential reversal zone
                    self.current_regime = "bearish_extended"  # Bearish but extended
                    self.current_position_scale = 1.2 * self.short_bias_adjustment  # Slightly reduced short
                else:
                    self.current_regime = "strong_bearish"  # Strong sustained downtrend
                    self.current_position_scale = 1.5 * self.short_bias_adjustment  # Full bearish exposure
            
            # Choppy/ranging signals:
            # 1. SMAs close together 
            # 2. Price oscillating around VWAP
            # 3. Mixed momentum signals
            elif (abs(sma_short/sma_long - 1) < 0.003 and 
                 abs(vwap_position) < 0.003 and
                 abs(mom_5) < 0.005):
                
                self.current_regime = "ranging"
                self.current_position_scale = 0.6  # Reduced size in ranges
            
            # Trend transition signals:
            # 1. Early trend change indicators
            # 2. Momentum shifting but SMAs haven't crossed yet
            elif ((sma_short > sma_long * 0.999 and sma_short < sma_long * 1.001) and
                 ((mom_1 > 0 and mom_5 < 0) or (mom_1 < 0 and mom_5 > 0))):
                
                self.current_regime = "transition"
                self.current_position_scale = 0.5  # Very conservative during transitions
            
            # Moderate bullish/bearish conditions
            elif sma_short > sma_long:
                self.current_regime = "bullish"
                self.current_position_scale = 1.0
            elif sma_short < sma_long:
                self.current_regime = "bearish"
                self.current_position_scale = 1.0 * self.short_bias_adjustment
            else:
                self.current_regime = "neutral"
                self.current_position_scale = 0.7
                
            # Volatility-based position sizing adjustment
            # Asymmetric volatility response for long vs short
            if short_vol > 0 and long_vol > 0:
                vol_ratio = long_vol / short_vol
                
                # In high volatility environments
                if vol_ratio < 0.7:
                    # Especially careful with shorts in volatile markets
                    if self.current_regime in ["bearish", "strong_bearish", "bearish_extended"]:
                        self.volatility_adjustment = max(0.3, vol_ratio * 0.6)
                    else:
                        self.volatility_adjustment = max(0.4, vol_ratio * 0.7)
                
                # In low volatility (stable) environments
                elif vol_ratio > 1.3:
                    # More aggressive with shorts in stable downtrends
                    if self.current_regime in ["bearish", "strong_bearish"]:
                        self.volatility_adjustment = min(2.0, vol_ratio * 1.1)
                    else:
                        self.volatility_adjustment = min(1.8, vol_ratio)
                else:
                    self.volatility_adjustment = 1.0
            else:
                self.volatility_adjustment = 1.0
                
            # Log significant regime changes
            if hasattr(self, 'prev_regime') and self.prev_regime != self.current_regime:
                self.logger.info(f"Regime change: {self.prev_regime} -> {self.current_regime} | "
                               f"Scale: {self.current_position_scale:.2f}, Vol Adj: {self.volatility_adjustment:.2f}")
                
            self.prev_regime = self.current_regime
                
        except Exception as e:
            self.logger.error(f"Error analyzing market regimes: {str(e)}")
            self.current_regime = "neutral"
            self.current_position_scale = 1.0
            self.volatility_adjustment = 1.0

    def position_size(self, price, atr, signal):
        """
        Calculate position size with enhanced risk management for short-term trading
        Returns a value between 0.0 and max_position_size (positive for long, negative for short)
        """
        # Skip if signal is in neutral zone
        if abs(signal) < self.neutral_zone:
            return 0.0
            
        # Base size calculation
        base_size = abs(signal)
        
        # Apply dynamic position sizing based on market regime
        position_scale = self.current_position_scale * self.volatility_adjustment
        
        # Add win streak boost (more conservative for short positions)
        win_streak_factor = min(1.0 + (self.current_win_streak * 0.05), self.max_win_streak_boost)
        
        # Adjust size based on signal direction
        if signal > 0:  # Long position
            size = base_size * position_scale * win_streak_factor
            max_size = self.max_position_size
        else:  # Short position
            size = base_size * position_scale * win_streak_factor * self.short_bias_adjustment
            max_size = self.max_short_position
        
        # Normalize to max position size
        position_size = min(max_size, size)
        
        # Risk-based size adjustment using ATR
        risk_per_unit = atr
        if risk_per_unit <= 0:
            return 0.0
            
        # Base max position on risk cap
        max_position = (self.max_trade_risk_pct * self.trade_state.equity) / risk_per_unit
        
        # Cap position based on risk limit
        position_size = min(position_size, max_position)
        
        # Apply sign based on signal direction
        position_size = position_size if signal > 0 else -position_size
        
        return position_size

    def check_stops(self, current_price, is_long):
        """
        Check if stop-loss or take-profit levels are hit
        Returns True if position should be closed
        """
        if is_long:  # Long position
            # Check stop-loss
            if self.stop_loss_level and current_price <= self.stop_loss_level:
                self.logger.info(f"Long stop-loss hit: {current_price:.2f} <= {self.stop_loss_level:.2f}")
                return True
                
            # Check trailing stop
            if self.trailing_stop_level and current_price <= self.trailing_stop_level:
                self.logger.info(f"Long trailing-stop hit: {current_price:.2f} <= {self.trailing_stop_level:.2f}")
                return True
                
            # Check take-profit
            if self.take_profit_level and current_price >= self.take_profit_level:
                self.logger.info(f"Long take-profit hit: {current_price:.2f} >= {self.take_profit_level:.2f}")
                return True
        else:  # Short position
            # Check stop-loss
            if self.stop_loss_level and current_price >= self.stop_loss_level:
                self.logger.info(f"Short stop-loss hit: {current_price:.2f} >= {self.stop_loss_level:.2f}")
                return True
                
            # Check trailing stop
            if self.trailing_stop_level and current_price >= self.trailing_stop_level:
                self.logger.info(f"Short trailing-stop hit: {current_price:.2f} >= {self.trailing_stop_level:.2f}")
                return True
                
            # Check take-profit
            if self.take_profit_level and current_price <= self.take_profit_level:
                self.logger.info(f"Short take-profit hit: {current_price:.2f} <= {self.take_profit_level:.2f}")
                return True
                
        return False

    def update_stops(self, current_price, atr, is_long):
        """
        Update trailing stop levels based on price movement
        """
        # Skip if no stops set
        if self.stop_loss_level is None:
            return
            
        if is_long:  # Long position
            # Update trailing stop only if price moves in favorable direction
            new_trailing_stop = current_price - (atr * self.trailing_stop_atr_multiplier)
            
            # Only move trailing stop up, never down
            if self.trailing_stop_level is None or new_trailing_stop > self.trailing_stop_level:
                self.trailing_stop_level = new_trailing_stop
                self.logger.debug(f"Updated long trailing-stop: {self.trailing_stop_level:.2f}")
        else:  # Short position
            # Update trailing stop only if price moves in favorable direction
            new_trailing_stop = current_price + (atr * self.trailing_stop_atr_multiplier)
            
            # Only move trailing stop down, never up
            if self.trailing_stop_level is None or new_trailing_stop < self.trailing_stop_level:
                self.trailing_stop_level = new_trailing_stop
                self.logger.debug(f"Updated short trailing-stop: {self.trailing_stop_level:.2f}")

    def set_stops(self, entry_price, atr, is_long):
        """
        Set initial stop-loss, take-profit and trailing-stop levels for a new position
        """
        if is_long:  # Long position
            # Stop-loss: entry price minus X ATRs
            self.stop_loss_level = entry_price - (atr * self.stop_loss_atr_multiplier)
            
            # Take-profit: entry price plus Y ATRs
            self.take_profit_level = entry_price + (atr * self.take_profit_atr_multiplier)
            
            # Initial trailing stop same as stop-loss
            self.trailing_stop_level = self.stop_loss_level
            
            self.logger.info(f"Long position stops set: Entry {entry_price:.2f}, "
                           f"Stop-loss {self.stop_loss_level:.2f}, "
                           f"Take-profit {self.take_profit_level:.2f}")
        else:  # Short position
            # Stop-loss: entry price plus X ATRs
            self.stop_loss_level = entry_price + (atr * self.stop_loss_atr_multiplier)
            
            # Take-profit: entry price minus Y ATRs
            self.take_profit_level = entry_price - (atr * self.take_profit_atr_multiplier)
            
            # Initial trailing stop same as stop-loss
            self.trailing_stop_level = self.stop_loss_level
            
            self.logger.info(f"Short position stops set: Entry {entry_price:.2f}, "
                           f"Stop-loss {self.stop_loss_level:.2f}, "
                           f"Take-profit {self.take_profit_level:.2f}")

    def reset_stops(self):
        """Reset all stop levels when position is closed"""
        self.stop_loss_level = None
        self.take_profit_level = None
        self.trailing_stop_level = None

    # —————————————————— training ——————————————————
    def train(self, raw_bars: List[List]):
        log.info(f"Training model for {self.symbol} with {len(raw_bars)} bars...")
        if len(raw_bars) < MIN_BARS_FOR_TRAINING:
            raise ValueError(f"Need at least {MIN_BARS_FOR_TRAINING} bars for training")
            
        df = pd.DataFrame(raw_bars, columns=["ts","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = self._add_indicators(df).dropna()
        
        # Extended feature set
        feat_cols = ['returns', 'momentum', 'rsi', 'bb_z', 'vol_ratio', 'vwap_z', 'regime']
        
        # Scale features
        self.scaler.fit(df[feat_cols])
        df[feat_cols] = self.scaler.transform(df[feat_cols])
        
        # Create dataset and loader
        dataset = BarDataset(df[feat_cols+['future_return']], context=self.context_window)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
        
        # Initialize model
        self.model = DMTTransformer(num_feats=len(feat_cols), 
                                    d_model=self.transformer_dims, 
                                    nhead=self.attention_heads, 
                                    num_layers=self.num_layers).to(self.device)
        
        # Train model
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        self.model.train()
        
        for epoch in range(20):
            total = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            log.info(f"Epoch {epoch} loss {total/len(loader.dataset):.6f}")
        
        log.info("Training finished successfully")
        return True

    # —————————————————— inference ——————————————————
    def _calculate_fallback_signal(self, ohlcv_data):
        """Fallback signal calculation when model is not trained or encounters errors"""
        try:
            # Use a simple momentum-based approach
            price_df, feature_df = self.preprocess_data(ohlcv_data)
            
            momentum = price_df['close'].pct_change(5).iloc[-1]
            signal = max(min(momentum * 10, 1.0), -1.0)  # Scale and clamp
            
            # Add market conditions info
            atr = price_df['atr_14'].iloc[-1] if 'atr_14' in price_df.columns else price_df['close'].iloc[-1] * 0.01
            close = price_df['close'].iloc[-1]
            rsi = feature_df['rsi'].iloc[-1] if 'rsi' in feature_df.columns else 50.0
            
            self.logger.info("Using fallback momentum-based signal calculation")
            
            return signal, {
                "atr": atr,
                "close": close,
                "market_regime": "bullish" if momentum > 0 else "bearish",
                "volatility": price_df['volatility_15'].iloc[-1] if 'volatility_15' in price_df.columns else 0.01,
                "rsi": rsi,
                "fallback": True
            }
        except Exception as e:
            self.logger.error(f"Error in fallback signal calculation: {str(e)}")
            # Last resort - neutral signal
            return 0.0, {"error": str(e), "fallback": True}
            
    def calculate_signal(self, latest_ctx: List[List]) -> Tuple[float, dict]:
        """
        Calculate trading signal and return additional info for risk management
        
        Args:
            latest_ctx: Latest OHLCV data
        
        Returns:
            tuple: (signal, info) - Signal between -1 and 1, and additional info
        """
        # If model is not trained, use a simple momentum-based fallback
        if not self.is_trained:
            return self._calculate_fallback_signal(latest_ctx)
            
        try:
            # Process data
            df = pd.DataFrame(latest_ctx, columns=["ts","open","high","low","close","volume"])
            df = self._add_indicators(df).dropna().tail(self.context_window)
            
            if len(df) < self.context_window:
                return 0.0, {"reason": "insufficient_data"}
                
            # Get features
            feat_cols = ['returns', 'momentum', 'rsi', 'bb_z', 'vol_ratio', 'vwap_z', 'regime']
            x = self.scaler.transform(df[feat_cols]).astype(np.float32)[None]  # [1,T,F]
            x_t = torch.from_numpy(x).to(self.device)
            
            # Get model output
            self.model.eval()
            with torch.no_grad():
                raw_signal = self.model(x_t).item()
                
            # Apply signal processing
            # 1. Smoothing
            smoothed = self.signal_smoothing * self.trade_state.last_signal + (1-self.signal_smoothing) * raw_signal
            
            # 2. Handle short ban if configured
            if not self.allow_short and smoothed < 0:
                smoothed = 0.0
                
            # 3. Apply signal delay filter if configured
            if self.consecutive_signals > 0:
                if (self.trade_state.pending_signal is None or 
                    np.sign(smoothed) != np.sign(self.trade_state.pending_signal)):
                    self.trade_state.pending_signal = smoothed 
                    self.trade_state.pending_count = 1
                    return 0.0, {"reason": "signal_delay", "pending": smoothed}
                    
                self.trade_state.pending_count += 1
                if self.trade_state.pending_count < self.consecutive_signals:
                    return 0.0, {"reason": "signal_delay", "pending": smoothed, 
                                "count": self.trade_state.pending_count}
                                
                smoothed = self.trade_state.pending_signal
                self.trade_state.pending_signal = None
                
            # 4. Final clip
            final_signal = float(np.clip(smoothed, -1, 1))
            
            # 5. Store for next iteration
            self.trade_state.last_signal = final_signal
            
            # 6. Extract additional state info
            latest_row = df.iloc[-1]
            info = {
                "price": latest_row["close"],
                "atr": latest_row["atr"],
                "regime": "bull" if latest_row["regime"] > 0 else "bear",
                "rsi": latest_row["rsi"],
                "bb_z": latest_row["bb_z"],
                "vwap_z": latest_row["vwap_z"]
            }
            
            # Update trade state regime
            self.trade_state.regime = info["regime"]
            
            return final_signal, info
            
        except Exception as e:
            log.error(f"Signal calculation error: {str(e)}\n{traceback.format_exc()}")
            return 0.0, {"error": str(e)}

    def position_size(self, price: float, atr: float, signal: float) -> float:
        """Determine position size based on ATR and risk parameters"""
        return self.trade_state.calculate_position_size(atr, price, signal)
        
    def set_stops(self, entry_price: float, atr: float, is_long: bool):
        """Set stop levels for current position"""
        self.trade_state.set_stops(entry_price, atr, is_long, 
                                   atr_stop_multiplier=self.stop_loss_atr_multiplier)
        
    def update_stops(self, current_price: float, atr: float, is_long: bool):
        """Update trailing stops for current position"""
        self.trade_state.update_stops(current_price, atr, is_long)
        
    def check_stops(self, current_price: float, is_long: bool) -> bool:
        """Check if stop levels are hit"""
        return self.trade_state.check_stops(current_price, is_long)
        
    def record_trade_result(self, is_win: bool, pnl: float):
        """Record trade result for tracking performance"""
        self.trade_state.update_trade_result(is_win, pnl)
        
    def reset_daily(self):
        """Reset daily risk metrics (call at start of each trading day)"""
        self.trade_state.reset_daily()

    # —————————————————— save / load ——————————————————
    def save(self, path: Union[str, Path]):
        """Save model to file"""
        if self.model is None:
            raise RuntimeError("No trained model to save")
            
        torch.save({
            "model_state": self.model.state_dict(),
            "cfg": self.cfg,
            "scaler": self.scaler,
            "allow_short": self.allow_short,
            "signal_delay": self.consecutive_signals,
            "signal_smoothing": self.signal_smoothing,
            "context": self.context_window
        }, path)
        log.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load model from file"""
        chk = torch.load(path, map_location=self.device)
        
        # Load configuration
        self.cfg = chk["cfg"]
        self.scaler = chk["scaler"]
        
        # Load other parameters if available
        if "allow_short" in chk:
            self.allow_short = chk["allow_short"]
        if "signal_delay" in chk:
            self.consecutive_signals = chk["signal_delay"]
        if "signal_smoothing" in chk:
            self.signal_smoothing = chk["signal_smoothing"]
        if "context" in chk:
            self.context_window = chk["context"]
        
        # Rebuild model
        feat_dim = 7  # keep in sync with calculate_signal()
        self.model = DMTTransformer(num_feats=feat_dim, 
                                    d_model=self.transformer_dims, 
                                    nhead=self.attention_heads, 
                                    num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(chk["model_state_dict"])
        self.model.eval()
        log.info(f"Model loaded from {path}")

class DecisionTransformerAgent:
    """Agent that uses Decision Transformer for trading decisions"""
    
    def __init__(self, state_dim, action_dim=1, hidden_size=128, learning_rate=1e-4, 
                 max_length=30, n_layer=4, n_head=4, weight_decay=1e-4, 
                 device=None, discount=0.99, target_return=None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.target_return = target_return
        
        # Initialize Decision Transformer model
        self.model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            n_layer=n_layer,
            n_head=n_head
        ).to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Replay buffer for experience collection
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=int(1e6),
            device=self.device
        )
        
        # Initialize target return per step if not provided
        if self.target_return is None:
            self.target_return = 0.005  # Target 0.5% return per step by default
    
    def get_action(self, states, actions, returns_to_go, timesteps):
        """Get action from Decision Transformer"""
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32).to(self.device)
        timesteps = torch.tensor(timesteps, dtype=torch.long).to(self.device)
        
        # Create attention mask
        mask = torch.ones((1, timesteps.shape[1])).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            predicted_actions, _ = self.model(states, actions, returns_to_go, mask)
            
            # Get the predicted action for the last timestep
            predicted_action = predicted_actions[0, -1].cpu().numpy()
        
        return predicted_action
    
    def train_offline(self, buffer, batch_size=64, update_steps=1000, print_freq=10):
        """Train the model on offline data"""
        self.model.train()
        
        total_loss = 0
        for step in range(update_steps):
            # Sample batch from buffer
            batch = buffer.sample(batch_size)
            
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Print progress
            if (step + 1) % print_freq == 0:
                print(f"Step {step+1}/{update_steps}, Avg Loss: {total_loss / print_freq:.4f}")
                total_loss = 0
    
    def train_online(self, env, episodes=10, max_steps=1000, update_freq=100, 
                     target_return=None, exploration_noise=0.1):
        """Online fine-tuning on environment interactions"""
        self.model.train()
        
        if target_return is not None:
            self.target_return = target_return
        
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            done = False
            episode_return = 0
            episode_steps = 0
            
            # Initialize sequence
            states = [np.zeros(self.state_dim)]
            actions = [np.zeros(self.action_dim)]
            returns_to_go = [np.array([self.target_return * (max_steps - episode_steps)])]
            timesteps = [0]
            
            while not done and episode_steps < max_steps:
                # Get states and returns for decision transformer
                dt_states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)
                dt_actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device)
                dt_returns = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(0).to(self.device)
                dt_timesteps = torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get action from policy
                action = self.get_action(dt_states, dt_actions, dt_returns, dt_timesteps)
                
                # Add exploration noise for online learning
                noise = np.random.normal(0, exploration_noise, size=self.action_dim)
                noisy_action = np.clip(action + noise, -1.0, 1.0)
                
                # Take step in environment
                next_state, reward, done, info = env.step(noisy_action)
                
                # Track returns
                episode_return += reward
                episode_steps += 1
                
                # Update returns-to-go
                returns_to_go.append(np.array([self.target_return * (max_steps - episode_steps)]))
                
                # Store transition in replay buffer
                self.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done)
                )
                
                # Update sequences
                states.append(next_state)
                actions.append(action)
                timesteps.append(episode_steps)
                
                # Limit sequence length
                if len(states) > self.max_length:
                    states = states[-self.max_length:]
                    actions = actions[-self.max_length:]
                    returns_to_go = returns_to_go[-self.max_length:]
                    timesteps = timesteps[-self.max_length:]
                
                # Move to next state
                state = next_state
                
                # Update model periodically
                if episode_steps % update_freq == 0 and self.replay_buffer.size >= 1000:
                    for _ in range(10):  # 10 updates per interaction period
                        batch = self.replay_buffer.sample(64)
                        loss = self._compute_loss(batch)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.optimizer.step()
            
            print(f"Episode {episode+1}/{episodes}, Return: {episode_return:.2f}, Steps: {episode_steps}")
    
    def save(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'target_return': self.target_return,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with saved dimensions
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.hidden_size = checkpoint['hidden_size']
        self.max_length = checkpoint['max_length']
        self.target_return = checkpoint['target_return']
        
        # Reinitialize model
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
            max_length=self.max_length
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Reinitialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay
        )
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {path}")
    
    def _compute_loss(self, batch):
        """Calculate loss for training"""
        states = batch['states']
        actions = batch['actions']
        returns_to_go = batch['returns_to_go']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        predicted_actions, predicted_values = self.model(
            states, actions, returns_to_go, attention_mask
        )
        
        # Action prediction loss
        action_mask = attention_mask[:, 1:]  # Shift to match predicted actions
        action_loss = F.mse_loss(
            predicted_actions[:, :-1][action_mask], 
            actions[:, 1:][action_mask]
        )
        
        # Value prediction loss
        value_mask = attention_mask[:, :]
        value_loss = F.mse_loss(
            predicted_values[value_mask],
            returns_to_go[value_mask]
        )
        
        # Combined loss
        loss = action_loss + 0.1 * value_loss
        
        return loss


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences
    """
    
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
        self.returns_to_go = np.zeros((max_size, 1))
        self.timesteps = np.zeros(max_size, dtype=np.int64)
        
        self.ptr = 0
        self.size = 0
        self.episode_starts = []
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Ensure states are flattened if needed
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = state.flatten()
        if isinstance(next_state, np.ndarray) and len(next_state.shape) > 1:
            next_state = next_state.flatten()
            
        # Handle dictionary state format
        if isinstance(state, dict) and 'features' in state:
            state = state['features']
        if isinstance(next_state, dict) and 'features' in next_state:
            next_state = next_state['features']
        
        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.timesteps[self.ptr] = self.size
        
        # Mark episode start 
        if self.ptr == 0 or self.dones[self.ptr-1]:
            self.episode_starts.append(self.ptr)
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def compute_returns(self, discount=0.99):
        """Compute returns-to-go for all transitions"""
        episode_ends = np.where(self.dones == 1)[0]
        start_idx = 0
        
        for end_idx in episode_ends:
            # Compute returns for each episode
            returns = np.zeros(end_idx - start_idx + 1)
            returns[-1] = self.rewards[end_idx]
            
            for i in range(end_idx - start_idx - 1, -1, -1):
                returns[i] = self.rewards[i + start_idx] + discount * returns[i + 1]
            
            # Store returns
            self.returns_to_go[start_idx:end_idx+1] = returns.reshape(-1, 1)
            
            # Update start index
            start_idx = end_idx + 1
            
    def sample(self, batch_size, seq_length=None):
        """Sample a batch of experiences"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # If sequence length is specified, sample sequences
        if seq_length is not None:
            states = []
            actions = []
            returns = []
            timesteps = []
            
            for idx in indices:
                # Find episode start for this index
                episode_start = max(filter(lambda x: x <= idx, self.episode_starts), default=0)
                episode_end = idx
                
                # Determine valid sequence length
                avail_length = min(seq_length, episode_end - episode_start + 1)
                
                # If available length is less than sequence length, pad with zeros
                if avail_length < seq_length:
                    # Get actual sequence
                    s = self.states[episode_end - avail_length + 1:episode_end + 1]
                    a = self.actions[episode_end - avail_length + 1:episode_end + 1]
                    r = self.returns_to_go[episode_end - avail_length + 1:episode_end + 1]
                    t = self.timesteps[episode_end - avail_length + 1:episode_end + 1]
                    
                    # Create padded sequence
                    padded_s = np.zeros((seq_length, self.state_dim))
                    padded_a = np.zeros((seq_length, self.action_dim))
                    padded_r = np.zeros((seq_length, 1))
                    padded_t = np.zeros(seq_length, dtype=np.int64)
                    
                    # Fill in padded sequence
                    padded_s[-avail_length:] = s
                    padded_a[-avail_length:] = a
                    padded_r[-avail_length:] = r
                    padded_t[-avail_length:] = t
                    
                    states.append(padded_s)
                    actions.append(padded_a)
                    returns.append(padded_r)
                    timesteps.append(padded_t)
                else:
                    # Take exact sequence
                    states.append(self.states[episode_end - seq_length + 1:episode_end + 1])
                    actions.append(self.actions[episode_end - seq_length + 1:episode_end + 1])
                    returns.append(self.returns_to_go[episode_end - seq_length + 1:episode_end + 1])
                    timesteps.append(self.timesteps[episode_end - seq_length + 1:episode_end + 1])
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            returns = torch.tensor(np.array(returns), dtype=torch.float32)
            timesteps = torch.tensor(np.array(timesteps), dtype=torch.long)
            
            return {
                'states': states,
                'actions': actions, 
                'returns_to_go': returns,
                'timesteps': timesteps
            }
        else:
            # Basic random sampling
            return {
                'states': self.states[indices],
                'actions': self.actions[indices],
                'rewards': self.rewards[indices],
                'next_states': self.next_states[indices],
                'dones': self.dones[indices],
                'returns_to_go': self.returns_to_go[indices]
            }
{{ ... }}
