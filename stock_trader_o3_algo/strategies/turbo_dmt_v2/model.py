#!/usr/bin/env python3
"""
TurboDMT_v2 Model Architecture
==============================
Advanced trading model architecture with hybrid Transformer-LSTM design,
multi-headed attention mechanisms, and skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Union


class TurboDMTConfig:
    """Configuration settings for the TurboDMT_v2 model"""
    
    def __init__(
        self,
        # Model architecture settings
        feature_dim: int = 32,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        transformer_dim: int = 192,
        transformer_layers: int = 3,
        attention_heads: int = 12,
        dropout: float = 0.15,
        
        # Trading strategy parameters
        max_position_size: float = 3.0,        # Increased from 2.0 to 3.0 for more aggressive positions
        target_vol: float = 0.35,               # Annual volatility target
        neutral_zone: float = 0.02,            # Reduced from 0.025 for faster entries
        
        # Risk management parameters
        max_drawdown_threshold: float = 0.15,
        risk_scaling_factor: float = 0.5,
        uncertainty_threshold: float = 0.2,
        stop_loss_atr_multiple: float = 2.5,
        
        # Training parameters
        learning_rate: float = 0.001,
        batch_size: int = 64,
        seq_len: int = 30,                     # Increased sequence length for better pattern detection
        
        # Ensemble parameters
        use_ensemble: bool = True,
        ensemble_size: int = 7,
        
        # Flags and options
        use_skip_connections: bool = True,
        use_spectral_features: bool = True,
        use_multi_timeframe: bool = True,
        use_regime_detection: bool = True,
        use_dynamic_stops: bool = True,
    ):
        # Model architecture settings
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.transformer_dim = transformer_dim
        self.transformer_layers = transformer_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        
        # Trading strategy parameters
        self.max_position_size = max_position_size
        self.target_vol = target_vol
        self.neutral_zone = neutral_zone
        
        # Risk management parameters
        self.max_drawdown_threshold = max_drawdown_threshold
        self.risk_scaling_factor = risk_scaling_factor
        self.uncertainty_threshold = uncertainty_threshold
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        
        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Ensemble parameters
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        
        # Flags and options
        self.use_skip_connections = use_skip_connections
        self.use_spectral_features = use_spectral_features
        self.use_multi_timeframe = use_multi_timeframe
        self.use_regime_detection = use_regime_detection
        self.use_dynamic_stops = use_dynamic_stops


class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with improved scaling and regularization"""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings for improved positional encoding
        self.use_rotary = use_rotary
        
    def _rotary_emb(self, x, seq_len):
        """Apply rotary positional embeddings to the input tensor"""
        device = x.device
        
        # Create position indices
        position = torch.arange(0, seq_len, device=device).unsqueeze(1)
        
        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, self.head_dim, 2, device=device) * 
            -(math.log(10000.0) / self.head_dim)
        )
        
        # Create sinusoidal pattern
        pos_enc = torch.zeros(seq_len, self.head_dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # Apply rotary embeddings
        x_rope = torch.zeros_like(x)
        x_rope[:, :, :, 0::2] = x[:, :, :, 0::2] * torch.cos(pos_enc[:, 0::2]) - \
                                x[:, :, :, 1::2] * torch.sin(pos_enc[:, 0::2])
        x_rope[:, :, :, 1::2] = x[:, :, :, 1::2] * torch.cos(pos_enc[:, 0::2]) + \
                                x[:, :, :, 0::2] * torch.sin(pos_enc[:, 0::2])
        return x_rope
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to bring heads dimension before sequence length
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            q = self._rotary_emb(q, seq_len)
            k = self._rotary_emb(k, seq_len)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_probs, v)
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attention_probs


class TransformerEncoderLayerWithSkipConnection(nn.Module):
    """Enhanced transformer encoder layer with skip connections and layer normalization"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.gelu if activation == "gelu" else F.relu
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Multi-head attention with skip connection
        src2, _ = self.self_attn(
            query=self.norm1(src),
            key=self.norm1(src),
            value=self.norm1(src),
            attn_mask=src_mask,
        )
        src = src + self.dropout1(src2) if self.use_skip else self.dropout1(src2)
        
        # Feedforward network with skip connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2) if self.use_skip else self.dropout2(src2)
        
        return src


class HybridTransformerLSTM(nn.Module):
    """Hybrid Transformer-LSTM architecture that combines pattern recognition with sequential memory"""
    
    def __init__(self, config: TurboDMTConfig):
        super().__init__()
        self.config = config
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.transformer_dim),
            nn.LayerNorm(config.transformer_dim),
        )
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithSkipConnection(
                d_model=config.transformer_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                activation="gelu",
                use_skip=config.use_skip_connections,
            )
            for _ in range(config.transformer_layers)
        ])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.transformer_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # Output is a probability
        )
        
        # Volatility estimation head
        self.vol_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Market regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # 3 regimes (bullish, neutral, bearish)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved techniques for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Use orthogonal initialization for LSTM
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'linear' in name:
                    # Use Kaiming initialization for linear layers
                    nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            tuple: (prediction, volatility, regime_logits)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed features
        embedded = self.feature_embedding(x)  # (batch_size, seq_len, transformer_dim)
        
        # Apply transformer layers
        transformer_out = embedded
        for transformer_layer in self.transformer_layers:
            transformer_out = transformer_layer(transformer_out)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(transformer_out)
        
        # Use the final hidden state for predictions
        final_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Generate predictions
        pred = self.pred_head(final_hidden)  # (batch_size, 1)
        vol = self.vol_head(final_hidden)  # (batch_size, 1)
        regime_logits = self.regime_head(final_hidden)  # (batch_size, 3)
        
        return pred, vol, regime_logits
    
    def loss_fn(self, pred: torch.Tensor, vol: torch.Tensor, regime_logits: torch.Tensor, 
                targets: torch.Tensor, regime_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate combined loss for training
        
        Args:
            pred: Prediction probabilities
            vol: Volatility estimates
            regime_logits: Market regime logits
            targets: Target values (next period returns)
            regime_targets: Optional regime classification targets
            
        Returns:
            Combined loss value
        """
        # Convert prediction from probability to expected return
        expected_return = (pred - 0.5) * 0.04  # Scale to ±2% range
        
        # Mean squared error for return prediction
        mse_loss = F.mse_loss(expected_return, targets)
        
        # Volatility calibration (negative log-likelihood of Gaussian distribution)
        vol_loss = torch.mean(0.5 * torch.log(vol) + 0.5 * (targets - expected_return)**2 / vol)
        
        # Regime classification loss (if targets provided)
        regime_loss = 0.0
        if regime_targets is not None:
            regime_loss = F.cross_entropy(regime_logits, regime_targets)
        
        # Combine losses
        combined_loss = mse_loss + 0.2 * vol_loss + 0.1 * regime_loss
        
        return combined_loss


class TurboDMTEnsemble:
    """Ensemble of TurboDMT models with diverse architectures and dynamic weighting"""
    
    def __init__(self, config: TurboDMTConfig, device: torch.device = torch.device('cpu')):
        self.config = config
        self.device = device
        self.models = []
        self.model_weights = []
        
        # Create diverse models for the ensemble
        self._create_ensemble()
    
    def _create_ensemble(self):
        """Create diverse models for the ensemble"""
        # Base model with default configuration
        self.models.append(HybridTransformerLSTM(self.config).to(self.device))
        
        # Model with larger transformer but smaller LSTM
        config2 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim * 1.5,
            transformer_layers=self.config.transformer_layers + 1,
            lstm_layers=1,
            attention_heads=self.config.attention_heads + 4,
            dropout=self.config.dropout * 0.7,  # Lower dropout
        )
        self.models.append(HybridTransformerLSTM(config2).to(self.device))
        
        # Model with larger LSTM but smaller transformer
        config3 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim * 0.8,
            transformer_layers=self.config.transformer_layers - 1,
            lstm_layers=self.config.lstm_layers + 1,
            hidden_dim=self.config.hidden_dim * 1.25,
            dropout=self.config.dropout * 1.2,  # Higher dropout
        )
        self.models.append(HybridTransformerLSTM(config3).to(self.device))
        
        # Model focused on short-term patterns
        config4 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim,
            transformer_layers=self.config.transformer_layers,
            attention_heads=self.config.attention_heads * 2,  # More attention heads
            dropout=self.config.dropout,
        )
        self.models.append(HybridTransformerLSTM(config4).to(self.device))
        
        # Model focused on regime detection
        config5 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim,
            hidden_dim=self.config.hidden_dim * 1.5,  # Larger hidden dim
            dropout=self.config.dropout,
        )
        self.models.append(HybridTransformerLSTM(config5).to(self.device))
        
        # Model focused on high volatility periods
        config6 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim * 0.75,
            transformer_layers=self.config.transformer_layers,
            lstm_layers=self.config.lstm_layers * 2,  # More LSTM layers
            dropout=self.config.dropout * 1.5,  # Much higher dropout
        )
        self.models.append(HybridTransformerLSTM(config6).to(self.device))
        
        # Model with GELU throughout
        config7 = TurboDMTConfig(
            transformer_dim=self.config.transformer_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
        self.models.append(HybridTransformerLSTM(config7).to(self.device))
        
        # Initialize weights equally
        self.model_weights = [1.0 / len(self.models)] * len(self.models)
    
    def update_weights(self, performances: List[float]):
        """Update model weights based on recent performance"""
        if not performances:
            return
        
        # Convert to numpy array
        perfs = np.array(performances)
        
        # Softmax to get new weights (higher performance = higher weight)
        weights = np.exp(perfs - np.max(perfs))
        weights = weights / np.sum(weights)
        
        # Update with exponential moving average
        alpha = 0.3  # Learning rate for weight updates
        for i in range(len(self.models)):
            self.model_weights[i] = (1 - alpha) * self.model_weights[i] + alpha * weights[i]
        
        # Normalize weights
        total = sum(self.model_weights)
        self.model_weights = [w / total for w in self.model_weights]
    
    def train(self, train_loader, epochs, lr=0.001, weight_decay=1e-4):
        """Train all models in the ensemble"""
        # Create optimizers for each model
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            for model in self.models
        ]
        
        # Create schedulers
        schedulers = [
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs
            )
            for optimizer in optimizers
        ]
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            for i, model in enumerate(self.models):
                model.train()
                total_loss = 0
                
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizers[i].zero_grad()
                    
                    # Forward pass
                    pred, vol, regime_logits = model(x_batch)
                    
                    # Calculate loss
                    loss = model.loss_fn(pred, vol, regime_logits, y_batch)
                    
                    # Backward pass and optimize
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
                    optimizers[i].step()
                    schedulers[i].step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                epoch_losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Model {i+1}: Loss = {avg_loss:.6f}")
            
            # Update weights based on inverse loss (lower loss = better performance)
            if epoch > 0 and epoch % 5 == 0:
                inverse_losses = [1.0 / (loss + 1e-6) for loss in epoch_losses]
                self.update_weights(inverse_losses)
                
                # Print current weights
                if epoch % 10 == 0:
                    weight_str = ", ".join([f"{w:.3f}" for w in self.model_weights])
                    print(f"Model weights: [{weight_str}]")
        
        return epoch_losses
    
    def predict(self, x):
        """Make ensemble prediction with uncertainty estimation"""
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
        
        all_preds = []
        all_vols = []
        all_regime_logits = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                pred, vol, regime_logits = model(x)
                
                # Apply model weight
                weight = self.model_weights[i]
                all_preds.append(pred * weight)
                all_vols.append(vol * weight)
                all_regime_logits.append(regime_logits * weight)
        
        # Weighted ensemble predictions
        pred_ensemble = sum(all_preds)
        vol_ensemble = sum(all_vols)
        regime_ensemble = sum(all_regime_logits)
        
        # Calculate prediction uncertainty (variance between models)
        pred_variance = torch.zeros_like(pred_ensemble)
        for pred in all_preds:
            pred_variance += (pred - pred_ensemble) ** 2
        pred_variance /= len(self.models)
        
        # Adjust volatility estimate based on model disagreement
        vol_ensemble = torch.sqrt(vol_ensemble ** 2 + pred_variance)
        
        return pred_ensemble, vol_ensemble, regime_ensemble, pred_variance


if __name__ == "__main__":
    # Simple test code
    config = TurboDMTConfig()
    model = HybridTransformerLSTM(config)
    
    # Test forward pass
    batch_size = 8
    seq_len = 30
    feature_dim = 32
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    pred, vol, regime_logits = model(x)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Volatility shape: {vol.shape}")
    print(f"Regime logits shape: {regime_logits.shape}")
    
    # Test ensemble
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = TurboDMTEnsemble(config, device)
    pred, vol, regime, uncertainty = ensemble.predict(x)
    
    print(f"Ensemble prediction shape: {pred.shape}")
    print(f"Ensemble volatility shape: {vol.shape}")
    print(f"Ensemble regime shape: {regime.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
