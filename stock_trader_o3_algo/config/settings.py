"""
Configuration settings for the micro-CTA strategy.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

# ETF tickers
RISK_ON = "QQQ"       # Nasdaq 100 ETF
RISK_OFF = "GLD"      # Gold ETF (best performing 2022-2024)
HEDGE_ETF = "SH"      # Short S&P 500 ETF
CASH_ETF = "BIL"      # Treasury Bills ETF

# Additional ETFs for multi-asset strategy
QQQ_ETF = "QQQ"
XLK_ETF = "XLK"
GOLD_ETF = "GLD"

# Algorithm parameters
LOOKBACK_DAYS = 30     # Longer lookback for more reliable momentum signals
SHORT_LOOKBACK = 10    # Longer short-term window to filter out noise
VOL_LOOK = 20          # Lookback period for volatility calculation
WEEKLY_VOL_TARGET = 0.03  # Higher volatility target (3%) for more aggressive positioning
CRASH_THRESHOLD = -0.05  # Threshold for crash detection
HEDGE_WEIGHT = 0.15    # Weight of the hedge asset during crash
KILL_DD = 0.2          # Drawdown to trigger kill switch
HISTORY_DAYS = 260     # Days of history to fetch for calculations

# Signal Thresholds
RSI_OVERSOLD = 30      # RSI oversold threshold (more aggressive entry)
RSI_OVERBOUGHT = 70    # RSI overbought threshold (allow more upside)
FAST_SMA = 10          # Fast simple moving average days
SLOW_SMA = 30          # Slow simple moving average days

# File storage for backtesting
BACKTEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "backtest_results")
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
