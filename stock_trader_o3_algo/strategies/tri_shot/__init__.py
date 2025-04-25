"""
Tri-Shot Trading Strategy

A strategy that runs three times per week:
- Monday: Determine directional view and enter position
- Wednesday: Volatility rebalancing
- Friday: Risk gate check

Components:
- tri_shot.py: Core strategy implementation
- tri_shot_model.py: ML models for market prediction
- tri_shot_features.py: Feature engineering for the models
"""

from .tri_shot import (
    run_monday_strategy,
    run_wednesday_strategy,
    run_friday_strategy,
    get_alpaca_api,
    ensure_state_dir,
    STATE_DIR,
    MODEL_FILE,
    TZ
)

from .tri_shot_model import (
    WalkForwardModel,
    load_walk_forward_model,
    train_walk_forward_model
)

from .tri_shot_features import (
    fetch_data,
    fetch_data_from_date,
    make_feature_matrix,
    train_model,
    save_model
)
