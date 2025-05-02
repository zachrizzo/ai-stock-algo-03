#!/usr/bin/env python3
"""
Direct test for enhanced DMT_v2 strategy with synthetic data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

# Create necessary model components directly in this script to avoid import issues
class Config:
    def __init__(self, 
                 eps=1e-8, 
                 tau_max=0.35, 
                 max_pos=2.0, 
                 neutral_zone=0.025,
                 lr=0.015,
                 seq_len=15,
                 max_drawdown_threshold=0.2,
                 risk_scaling_factor=0.6,
                 uncertainty_threshold=0.25,
                 use_ensemble=True,
                 use_dynamic_stops=True,
                 stop_loss_atr_multiple=2.5,
                 use_regime_detection=True,
                 regime_smoothing_window=3):
        self.eps = eps
        self.tau_max = tau_max
        self.max_pos = max_pos
        self.neutral_zone = neutral_zone
        self.lr = lr
        self.seq_len = seq_len
        self.max_drawdown_threshold = max_drawdown_threshold
        self.risk_scaling_factor = risk_scaling_factor
        self.uncertainty_threshold = uncertainty_threshold
        self.use_ensemble = use_ensemble
        self.use_dynamic_stops = use_dynamic_stops
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.use_regime_detection = use_regime_detection
        self.regime_smoothing_window = regime_smoothing_window

class RegimeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, n_regimes=3, smoothing_window=3):
        super().__init__()
        self.smoothing_window = smoothing_window
        self.n_regimes = n_regimes
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.regime_head = nn.Linear(hidden_dim // 2, n_regimes)
        self.regime_history = []
    
    def forward(self, x):
        features = self.feature_extractor(x)
        regime_logits = self.regime_head(features)
        return regime_logits
    
    def get_regime_probs(self):
        if len(self.regime_history) > 0:
            return self.regime_history[-1]
        else:
            return torch.ones(1, self.n_regimes) / self.n_regimes

class PredictionModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, transformer_dim=64, out_dim=1, 
                 n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Volatility head
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Regime classifier
        self.regime_classifier = RegimeClassifier(hidden_dim, hidden_dim // 2)
    
    def forward(self, x):
        """Forward pass with a single vector or a sequence"""
        # Extract features
        if len(x.shape) == 3:
            # Handle sequence data (batch, seq_len, features)
            batch_size, seq_len, _ = x.shape
            # Process the last timestep only for simplicity
            x = x[:, -1, :]
            
        features = self.feature_extractor(x)
        
        # Get prediction, volatility, and regime outputs
        p_t = self.pred_head(features)
        sigma_t = self.sigma_head(features) + 0.01  # Add small constant to avoid 0
        regime_logits = self.regime_classifier(features)
        
        return p_t, sigma_t, regime_logits
    
    def loss_fn(self, p_t, sigma_t, y):
        # Simple MSE loss
        return torch.mean((p_t - y) ** 2)

class StrategyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nz_lin = nn.Linear(3, 1)
    
    def forward(self, p_t, sigma_t, regime_logits, equity_curve=None, atr=None, uncertainty=None):
        # Calculate position size with neutral zone and volatility scaling
        regime_probs = F.softmax(regime_logits, dim=1)
        neutral_zone = torch.sigmoid(self.nz_lin(regime_probs)) * self.config.neutral_zone * 2
        
        # Simple position sizing - scale based on prediction vs neutral zone
        above_neutral = p_t > (0.5 + neutral_zone/2)
        below_neutral = p_t < (0.5 - neutral_zone/2)
        
        position = torch.zeros_like(p_t)
        position[above_neutral] = (p_t[above_neutral] - (0.5 + neutral_zone[above_neutral]/2)) / 0.5 * self.config.max_pos
        position[below_neutral] = (p_t[below_neutral] - (0.5 - neutral_zone[below_neutral]/2)) / 0.5 * -self.config.max_pos
        
        # Apply drawdown protection if provided
        if equity_curve is not None and equity_curve.shape[0] > 1:
            peak = torch.max(equity_curve)
            current = equity_curve[-1]
            drawdown = (current / peak) - 1.0
            
            if drawdown < -self.config.max_drawdown_threshold:
                scale = self.config.risk_scaling_factor
                position = position * scale
        
        # Clamp to ensure max position limits
        position = torch.clamp(position, -self.config.max_pos, self.config.max_pos)
        
        return position

def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR)"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def prepare_sequences(data, feature_cols, seq_len=15, target_col='target'):
    """Create sequence data for the model"""
    features = data[feature_cols].values
    target = data[target_col].values
    
    X = []
    y = []
    
    for i in range(len(data) - seq_len):
        seq = features[i:i+seq_len]
        target_val = target[i+seq_len-1]
        
        X.append(seq)
        y.append(target_val)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    return X, y, data.iloc[seq_len-1:-1]

def train_model(model, train_loader, optimizer, epochs, device):
    """Train the model"""
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            p_t, sigma_t, regime_logits = model(x_batch)
            loss = model.loss_fn(p_t, sigma_t, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def backtest_strategy(data, model, feature_cols, seq_len, initial_capital, config, device):
    """Run backtest on the strategy"""
    # Initialize results
    dates = data.index
    equity_curve = [initial_capital]
    positions = [0.0]
    daily_returns = [0.0]
    current_position = 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare strategy layer
    strategy = StrategyLayer(config).to(device)
    
    # Loop through data day by day
    for i in range(seq_len, len(data)):
        # Get feature sequence up to current day
        feature_sequence = data[feature_cols].iloc[i-seq_len:i].values
        
        # Convert to PyTorch tensor
        x = torch.from_numpy(feature_sequence).float().unsqueeze(0).to(device)
        
        # Get model predictions
        with torch.no_grad():
            p_t, sigma_t, regime_logits = model(x)
        
        # Extract ATR for stop-loss calculation
        current_atr = data['atr'].iloc[i] if 'atr' in data.columns else 0.01
        
        # Create equity tensor for drawdown calculation
        equity_tensor = torch.tensor([equity_curve], dtype=torch.float32).to(device)
        
        # Calculate new position
        new_position = strategy(
            p_t, sigma_t, regime_logits,
            equity_curve=equity_tensor,
            atr=torch.tensor([[current_atr]], dtype=torch.float32).to(device)
        ).item()
        
        # Calculate daily return based on position
        daily_return = current_position * data['returns'].iloc[i]
        
        # Update equity
        new_equity = equity_curve[-1] * (1 + daily_return)
        
        # Record results
        equity_curve.append(new_equity)
        positions.append(new_position)
        daily_returns.append(daily_return)
        
        # Update position for next day
        current_position = new_position
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'dmt_v2_equity': equity_curve[1:],  # Skip initial equity
        'position': positions[1:],          # Skip initial position
        'daily_return': daily_returns[1:]   # Skip initial return
    }, index=dates[seq_len:])
    
    # Calculate metrics
    final_equity = equity_curve[-1]
    total_return = final_equity / initial_capital - 1
    days = len(results_df)
    years = days / 252  # Trading days per year
    cagr = (final_equity / initial_capital) ** (1 / years) - 1
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = np.array(equity_curve) / peak - 1
    max_drawdown = np.min(drawdown)
    
    # Calculate Sharpe ratio
    daily_returns_np = np.array(daily_returns[1:])
    annual_vol = np.std(daily_returns_np) * np.sqrt(252)
    sharpe_ratio = cagr / annual_vol if annual_vol > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_equity': final_equity
    }
    
    return results_df, metrics

def plot_results(results_df, metrics, title="DMT_v2 Enhanced Backtest"):
    """Plot backtest results"""
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['dmt_v2_equity'], label='DMT_v2 Strategy')
    
    # Create buy & hold line for comparison
    returns = results_df['daily_return'].values
    prices = np.cumprod(1 + returns)
    plt.plot(results_df.index, prices * results_df['dmt_v2_equity'].iloc[0], 
             label='Buy & Hold', linestyle='--')
    
    plt.title(f"{title} - Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    os.makedirs('tri_shot_data', exist_ok=True)
    plt.savefig(os.path.join('tri_shot_data', 'dmt_v2_enhanced_backtest.png'))
    plt.close()
    
    # Also plot positions
    plt.figure(figsize=(12, 4))
    plt.plot(results_df.index, results_df['position'], color='purple')
    plt.fill_between(results_df.index, results_df['position'], 0, 
                    where=results_df['position']>0, color='green', alpha=0.3)
    plt.fill_between(results_df.index, results_df['position'], 0, 
                    where=results_df['position']<0, color='red', alpha=0.3)
    plt.title('DMT_v2 Enhanced Position Sizing')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    plt.grid(True)
    plt.savefig(os.path.join('tri_shot_data', 'dmt_v2_enhanced_positions.png'))
    plt.close()

def main():
    # Create synthetic data
    print("Creating synthetic test data...")
    days = 400
    dates = pd.date_range(start='2023-01-01', periods=days)
    
    # Create a pattern that's somewhat predictable but with noise
    t = np.linspace(0, 8*np.pi, days)
    trend = np.cumsum(np.sin(t/5) * 0.01 + 0.0005)
    close = 100 * np.exp(trend)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = close
    
    # Generate Open, High, Low values
    noise = np.random.normal(0, 0.005, days)
    data['Open'] = data['Close'] * (1 + noise)
    
    noise_high = np.abs(np.random.normal(0, 0.01, days))
    data['High'] = data['Close'] * (1 + noise_high)
    
    noise_low = np.abs(np.random.normal(0, 0.01, days))
    data['Low'] = data['Close'] * (1 - noise_low)
    
    # Generate Volume
    data['Volume'] = np.random.randint(1000000, 10000000, days)
    
    # Calculate features
    print("Preparing features...")
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['atr'] = calculate_atr(data)
    data['realized_vol'] = data['returns'].rolling(21).std() * np.sqrt(252)
    
    # Momentum indicators
    for period in [5, 10, 21]:
        data[f'mom_{period}'] = data['Close'].pct_change(period)
    
    # Mean reversion indicators
    for period in [5, 10, 21]:
        data[f'mean_rev_{period}'] = (data['Close'] - data['Close'].rolling(period).mean()) / data['Close'].rolling(period).std()
    
    # Target for prediction
    data['target'] = data['returns'].shift(-1)
    
    # Drop NaNs
    data = data.dropna()
    
    # Feature columns
    feature_cols = [
        'returns', 'log_returns', 'atr', 'realized_vol',
        'mom_5', 'mom_10', 'mom_21',
        'mean_rev_5', 'mean_rev_10', 'mean_rev_21'
    ]
    
    # Prepare sequences
    seq_len = 15
    X, y, df = prepare_sequences(data, feature_cols, seq_len)
    
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create PyTorch datasets and dataloaders
    batch_size = 32
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    in_dim = X_train.shape[2]
    model = PredictionModel(
        in_dim=in_dim,
        hidden_dim=128,
        transformer_dim=96,
        n_heads=8,
        n_layers=6,
        dropout=0.1
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train model
    print("Training model...")
    losses = train_model(model, train_loader, optimizer, epochs=50, device=device)
    
    # Create config for strategy
    config = Config(
        eps=1e-8,
        tau_max=0.35,
        max_pos=2.0,
        neutral_zone=0.025,
        lr=0.015,
        seq_len=seq_len,
        max_drawdown_threshold=0.2,
        risk_scaling_factor=0.6,
        uncertainty_threshold=0.25,
        use_ensemble=False,  # Using single model for simplicity
        use_dynamic_stops=True,
        stop_loss_atr_multiple=2.5
    )
    
    # Run backtest
    print("Running backtest...")
    results_df, metrics = backtest_strategy(
        data.iloc[seq_len:],  # Skip first seq_len rows used for sequences
        model,
        feature_cols,
        seq_len,
        initial_capital=500.0,
        config=config,
        device=device
    )
    
    # Print results
    print("\n=== Enhanced DMT_v2 Backtest Results ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return (CAGR): {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Plot results
    plot_results(results_df, metrics)
    print("Results saved to tri_shot_data/dmt_v2_enhanced_backtest.png")
    
    # Compare with original parameters
    print("\nRunning comparison with original (less optimized) parameters...")
    
    # Create original config
    original_config = Config(
        eps=1e-8,
        tau_max=0.35,
        max_pos=2.0,
        neutral_zone=0.03,  # Original was 0.03 vs 0.025
        lr=0.015,
        seq_len=seq_len,
        max_drawdown_threshold=0.15,  # Original was 0.15 vs 0.2
        risk_scaling_factor=0.5,  # Original was 0.5 vs 0.6
        uncertainty_threshold=0.2,  # Original was 0.2 vs 0.25
        use_ensemble=False,
        use_dynamic_stops=True,
        stop_loss_atr_multiple=2.0  # Original was 2.0 vs 2.5
    )
    
    # Run backtest with original config
    orig_results_df, orig_metrics = backtest_strategy(
        data.iloc[seq_len:],
        model,
        feature_cols,
        seq_len,
        initial_capital=500.0,
        config=original_config,
        device=device
    )
    
    # Print comparison
    print("\n=== DMT_v2 Strategy Comparison ===")
    print("Original vs Enhanced Parameters")
    print(f"Total Return: {orig_metrics['total_return']:.2%} -> {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {orig_metrics['sharpe_ratio']:.2f} -> {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {orig_metrics['max_drawdown']:.2%} -> {metrics['max_drawdown']:.2%}")
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['dmt_v2_equity'], label='Enhanced DMT_v2')
    plt.plot(orig_results_df.index, orig_results_df['dmt_v2_equity'], label='Original DMT_v2')
    
    plt.title('DMT_v2 Strategy Comparison: Original vs Enhanced Parameters')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join('tri_shot_data', 'dmt_v2_parameter_comparison.png'))
    print("Comparison chart saved to tri_shot_data/dmt_v2_parameter_comparison.png")
    
    # Return results for further analysis
    return {
        'enhanced': (results_df, metrics),
        'original': (orig_results_df, orig_metrics)
    }

if __name__ == "__main__":
    main()
