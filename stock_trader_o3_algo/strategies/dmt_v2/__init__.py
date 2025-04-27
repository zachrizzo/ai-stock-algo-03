#!/usr/bin/env python3
"""
DMT v2 Strategy - Transformer-Enhanced, Regime-Adaptive Strategy.

This module contains a more advanced version of the Differentiable Market Twin (DMT)
strategy, leveraging transformer architecture for predictions and incorporating
regime-awareness and adaptive controls.

Key components:
- Transformer-based prediction model
- EGARCH-like volatility forecasting
- Regime classification
- Adaptive neutral zone and target volatility
- End-to-end differentiable backtesting

This implementation uses PyTorch and can be used with the trader CLI.
"""

from .dmt_v2_model import Config, VolatilityEstimator, RegimeClassifier, PredictionModel, StrategyLayer
from .dmt_v2_backtest import run_dmt_v2_backtest

__all__ = [
    'Config',
    'VolatilityEstimator',
    'RegimeClassifier',
    'PredictionModel', 
    'StrategyLayer',
    'run_dmt_v2_backtest'
]
