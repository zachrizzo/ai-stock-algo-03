"""
Turbo-Rotational QQQ Turbo-4 Strategy

A more aggressive trading strategy using leveraged ETFs with:
- Technical indicators (RSI, MACD)
- ATR-based stops
- VIX crash overlay for protection

Components:
- turbo_qt.py: Core strategy logic and indicators
- turbo_qt_impl.py: Implementation functions for execution
- turbo_qt_backtest.py: Backtesting framework for the strategy
"""

from .turbo_qt import (
    get_prices,
    choose_asset,
    calculate_atr,
    check_crash_conditions,
    save_stop_price,
    check_stop_hit,
    get_alpaca_api,
    TICKERS,
    VOL_TARGET,
    ATR_MULT
)

from .turbo_qt_impl import (
    rebalance,
    check_stops
)

from .turbo_qt_backtest import (
    run_backtest,
    run_monte_carlo
)
