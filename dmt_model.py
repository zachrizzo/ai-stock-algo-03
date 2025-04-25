#!/usr/bin/env python3
"""
Differentiable Market Twin (DMT) Model

This module implements a differentiable market simulator using PyTorch.
The model can generate synthetic market paths that are statistically 
similar to real market behavior, and allows for gradient-based 
optimization of trading strategies through backpropagation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import datetime as dt

# Local imports
import tri_shot_features as tsf


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for sequence modeling of market data."""
    
    def __init__(self, sequences, targets, device='cpu'):
        """
        Initialize dataset with sequences and targets.
        
        Args:
            sequences: Tensor of input sequences (lookback_window, features)
            targets: Tensor of target values
            device: Torch device to use
        """
        self.sequences = sequences.to(device)
        self.targets = targets.to(device)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class MarketTwinLSTM(nn.Module):
    """
    Differentiable Market Twin using LSTM architecture.
    
    This model generates synthetic market data conditioned on initial state
    and action sequences. It's fully differentiable to enable gradient-based
    optimization of trading strategies.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 1,
                 dropout: float = 0.2,
                 device: str = 'cpu'):
        """
        Initialize the Market Twin model.
        
        Args:
            input_dim: Input dimension (features + action dimensions)
            hidden_dim: Hidden dimensions in LSTM
            num_layers: Number of LSTM layers
            output_dim: Output dimension (typically 1 for returns)
            dropout: Dropout probability
            device: Torch device to use
        """
        super(MarketTwinLSTM, self).__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers - multiple heads for different aspects of market behavior
        self.fc_return = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # For modeling return volatility 
        self.fc_vol = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensures positive volatility
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            Dictionary with predicted returns and volatility
        """
        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get predictions from each output head
        returns = self.fc_return(lstm_out)
        volatility = self.fc_vol(lstm_out)
        
        return {
            'returns': returns,
            'volatility': volatility,
            'hidden': hidden
        }
    
    def sample_path(self, initial_state, actions, seq_len=252, noise_scale=1.0):
        """
        Generate a sample price path conditioned on actions.
        
        Args:
            initial_state: Initial market state (features) - shape (batch_size, feature_dim)
            actions: Actions to take for each step - shape (batch_size, seq_len, action_dim)
            seq_len: Length of sequence to generate
            noise_scale: Scale of the random noise to add
            
        Returns:
            Tensor of synthetic price paths - shape (batch_size, seq_len)
        """
        batch_size = actions.shape[0]
        
        # Initialize price at 1.0 (will track returns)
        prices = torch.ones(batch_size, seq_len + 1).to(self.device)
        
        # Initial hidden state
        hidden = None
        
        # Current state starts as initial state
        current_state = initial_state.unsqueeze(1)  # Add sequence dimension
        
        # Generate path step by step
        for t in range(seq_len):
            # Combine state with action
            action_t = actions[:, t:t+1, :]  # Get current action
            x_t = torch.cat([current_state, action_t], dim=2)
            
            # Forward pass through model
            with torch.no_grad():  # Don't need gradients for sampling
                outputs = self(x_t, hidden)
                
            # Extract predictions
            mu = outputs['returns'][:, -1, 0]  # Mean return
            sigma = outputs['volatility'][:, -1, 0]  # Return volatility
            
            # Sample return with randomness
            if noise_scale > 0:
                epsilon = torch.randn(batch_size).to(self.device)
                ret = mu + sigma * epsilon * noise_scale
            else:
                ret = mu
                
            # Update price
            prices[:, t+1] = prices[:, t] * (1 + ret)
            
            # Update hidden state for next iteration
            hidden = outputs['hidden']
            
            # Update market state (this will depend on the feature engineering)
            # For simplicity, we'll use the realized return as the new state
            current_state = torch.cat([current_state[:, -1:, 1:], ret.unsqueeze(-1).unsqueeze(1)], dim=2)
        
        return prices
    
    def generate_paths_with_gradient(self, initial_state, actions):
        """
        Generate price paths while maintaining the computational graph for backpropagation.
        Unlike sample_path, this doesn't add random noise, making it suitable for
        gradient-based optimization.
        
        Args:
            initial_state: Initial market state
            actions: Actions to take (e.g., position sizes)
            
        Returns:
            Tensor of price paths with computational graph intact
        """
        batch_size = actions.shape[0]
        seq_len = actions.shape[1]
        
        # Initialize price at 1.0
        prices = torch.ones(batch_size, seq_len + 1).to(self.device)
        
        # Initial hidden state
        hidden = None
        
        # Current state starts as initial state
        current_state = initial_state.unsqueeze(1)
        
        # Generate path step by step
        for t in range(seq_len):
            # Combine state with action
            action_t = actions[:, t:t+1, :]
            x_t = torch.cat([current_state, action_t], dim=2)
            
            # Forward pass through model
            outputs = self(x_t, hidden)
            
            # Extract mean return
            mu = outputs['returns'][:, -1, 0]
            
            # Update price (retains gradients)
            prices[:, t+1] = prices[:, t] * (1 + mu)
            
            # Update hidden state
            hidden = outputs['hidden']
            
            # Update state for next step using the predicted return
            if t < seq_len - 1:
                current_state = torch.cat([current_state[:, -1:, 1:], mu.unsqueeze(-1).unsqueeze(1)], dim=2)
        
        return prices


class GumbelSoftmax(nn.Module):
    """
    Gumbel Softmax layer for differentiable discrete decisions.
    
    This is used to make discrete trading decisions (like buy/sell/hold)
    while maintaining differentiability for backpropagation.
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize Gumbel Softmax layer.
        
        Args:
            temperature: Temperature parameter controlling discreteness
        """
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        
    def forward(self, logits, hard=False):
        """
        Forward pass with Gumbel Softmax trick.
        
        Args:
            logits: Input logits
            hard: Whether to use straight-through estimator for hard sampling
            
        Returns:
            Differentiable samples from categorical distribution
        """
        if self.training:
            # Sample from Gumbel distribution
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.temperature
            y_soft = gumbels.softmax(dim=-1)
            
            if hard:
                # Straight through estimator
                index = y_soft.max(dim=-1, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                return y_hard - y_soft.detach() + y_soft
            else:
                return y_soft
        else:
            # During evaluation, just use argmax
            return torch.softmax(logits, dim=-1)


class MarketTwinTrainer:
    """Class to handle training of the Market Twin model."""
    
    def __init__(self,
                 model: MarketTwinLSTM,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: MarketTwinLSTM model instance
            learning_rate: Learning rate for Adam optimizer
            device: Torch device to use
        """
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def prepare_data(self, 
                     prices: pd.DataFrame,
                     lookback_window: int = 30,
                     target_ticker: str = 'QQQ',
                     feature_cols: Optional[List[str]] = None,
                     batch_size: int = 32):
        """
        Prepare data for training.
        
        Args:
            prices: DataFrame with price data
            lookback_window: Number of days to look back for features
            target_ticker: Ticker to predict
            feature_cols: Feature columns to use
            batch_size: Batch size for DataLoader
            
        Returns:
            DataLoader for training
        """
        # Create features
        X, y = tsf.make_feature_matrix(prices, target_ticker)
        
        # If feature_cols is provided, filter columns
        if feature_cols is not None:
            X = X[feature_cols]
            
        # Handle missing values
        X = X.fillna(0)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(lookback_window, len(X)):
            seq = X.iloc[i-lookback_window:i].values
            target = y.iloc[i]
            sequences.append(seq)
            targets.append(target)
            
        # Convert to torch tensors
        X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
        
        # Create dataset
        dataset = SequenceDataset(X_tensor, y_tensor, self.device)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train_epoch(self, dataloader, alpha=0.5):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with training data
            alpha: Weight for the volatility loss component
            
        Returns:
            Mean loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Extract predictions
            pred_returns = outputs['returns'][:, -1, 0]  # Get last time step
            pred_vols = outputs['volatility'][:, -1, 0]
            
            # For demonstration, we'll calculate actual returns and volatility
            actual_returns = target
            
            # Negative log likelihood loss for Gaussian distribution
            return_loss = nn.MSELoss()(pred_returns, actual_returns)
            
            # Variance matching loss 
            # Here we assume variance is close to squared return, which isn't
            # quite right but it's a simplification
            vol_loss = nn.MSELoss()(pred_vols**2, torch.abs(actual_returns))
            
            # Combined loss
            loss = return_loss + alpha * vol_loss
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)
    
    def train(self, 
              dataloader, 
              epochs: int = 10, 
              alpha: float = 0.5,
              verbose: bool = True):
        """
        Train the model.
        
        Args:
            dataloader: DataLoader with training data
            epochs: Number of epochs to train
            alpha: Weight for the volatility loss
            verbose: Whether to print training progress
            
        Returns:
            List of training losses
        """
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, alpha)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
                
        return losses
    
    def evaluate(self, dataloader):
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                pred_returns = outputs['returns'][:, -1, 0]
                
                # Calculate loss
                loss = nn.MSELoss()(pred_returns, target)
                total_loss += loss.item()
                
                # Store predictions and targets
                all_preds.extend(pred_returns.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Directional accuracy
        directional_accuracy = np.mean((all_preds > 0) == (all_targets > 0))
        
        # RMSE
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
        
        return {
            'loss': total_loss / len(dataloader),
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model.load_state_dict(torch.load(path))
        return self.model


def train_market_twin(prices: pd.DataFrame, 
                     target_ticker: str = 'QQQ',
                     lookback_window: int = 30,
                     hidden_dim: int = 128,
                     batch_size: int = 32,
                     epochs: int = 50,
                     lr: float = 0.001,
                     save_path: Optional[str] = None,
                     device: str = 'cpu'):
    """
    Train a Market Twin model and save it.
    
    Args:
        prices: DataFrame with price data
        target_ticker: Ticker to predict
        lookback_window: Number of days for lookback window
        hidden_dim: Hidden dimension for LSTM
        batch_size: Batch size for training
        epochs: Number of epochs to train
        lr: Learning rate
        save_path: Path to save the model (if None, will save to default location)
        device: Device to use for training
        
    Returns:
        Trained model and trainer
    """
    # Create features
    X, y = tsf.make_feature_matrix(prices, target_ticker)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Determine input dimension
    input_dim = X.shape[1]
    
    # Create model
    model = MarketTwinLSTM(
        input_dim=input_dim + 1,  # +1 for action dimension
        hidden_dim=hidden_dim,
        device=device
    ).to(device)
    
    # Create trainer
    trainer = MarketTwinTrainer(model, learning_rate=lr, device=device)
    
    # Prepare data
    dataloader = trainer.prepare_data(
        prices, 
        lookback_window=lookback_window,
        target_ticker=target_ticker,
        batch_size=batch_size
    )
    
    # Train model
    print(f"Training Market Twin with {epochs} epochs...")
    losses = trainer.train(dataloader, epochs=epochs, verbose=True)
    
    # Save model if path provided
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "tri_shot_data", "market_twin.pt")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    trainer.save_model(save_path)
    print(f"Market Twin model saved to {save_path}")
    
    return model, trainer


def load_market_twin(path: Optional[str] = None,
                    input_dim: int = 0,
                    hidden_dim: int = 128,
                    device: str = 'cpu'):
    """
    Load a trained Market Twin model.
    
    Args:
        path: Path to the saved model
        input_dim: Input dimension (required if model not previously initialized)
        hidden_dim: Hidden dimension
        device: Device to load the model to
        
    Returns:
        Loaded model
    """
    if path is None:
        path = os.path.join(os.getcwd(), "tri_shot_data", "market_twin.pt")
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
        
    model = MarketTwinLSTM(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device))
    
    return model


if __name__ == "__main__":
    # Code for testing the market twin model
    pass
