"""
Differentiable Market Twin (DMT) Strategy

This strategy uses a deep learning approach with differentiable models to
optimize trading on leveraged ETFs:

Components:
- dmt_model.py: PyTorch-based market twin models
- dmt_strategy.py: Strategy implementation using DMT models
- dmt_backtest.py: Backtesting framework for DMT strategies
"""

from .dmt_model import (
    MarketTwinLSTM,
    GumbelSoftmax,
    train_market_twin,
    load_market_twin
)

from .dmt_strategy import (
    DifferentiableTriShot
)

from .dmt_backtest import (
    run_dmt_backtest
)
