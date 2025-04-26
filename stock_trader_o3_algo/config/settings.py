"""
Configuration settings for the ensemble micro-trend strategy.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# API credentials
API_KEY = os.getenv("ALPACA_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

# ETF tickers
RISK_ON = "QQQ"       # Nasdaq 100 ETF - primary investment
RISK_OFF = "SPY"      # S&P 500 ETF (more conservative)
BOND_ETF = "TLT"      # 20-year Treasury Bond ETF
HEDGE_ETF = "SQQQ"    # 3x inverse QQQ ETF (for crash protection)
CASH_ETF = "BIL"      # Treasury Bills ETF

# Voting system parameters
LOOKBACK_DAYS = [63, 126, 252]  # 3, 6, 12 months lookback periods for voting

# Volatility targeting
TARGET_VOL = 0.18     # Annual volatility target (18%)
VOL_WINDOW = 20       # Window for volatility calculation
WEEKLY_VOL_TARGET = 0.05  # Weekly volatility target (5%)
VOL_LOOK = 63         # Lookback period for volatility calculation

# Crash protection
CRASH_VIX_THRESHOLD = 25.0     # VIX threshold for crash protection
CRASH_DROP_THRESHOLD = -0.05   # Weekly return threshold for crash protection
CRASH_THRESHOLD = -0.05        # Synonym for CRASH_DROP_THRESHOLD
HEDGE_WEIGHT = 0.10            # Allocation to hedge asset during crashes

# Trading restrictions
MIN_HOLD_DAYS = 10    # Minimum holding period for any position (trading days)

# Technical indicators
SHORT_LOOKBACK = 20   # Short lookback period for momentum
RSI_OVERSOLD = 30     # RSI oversold threshold
RSI_OVERBOUGHT = 70   # RSI overbought threshold
FAST_SMA = 50         # Fast simple moving average period
SLOW_SMA = 200        # Slow simple moving average period
SMA_DAYS = [50, 200]  # SMA periods to check for trend

# Position sizing
MAX_GROSS_EXPOSURE = 1.0  # Maximum gross exposure

# Stop loss
STOP_LOSS_THRESHOLD = 0.10     # Stop loss threshold (10%)
STOP_LOSS_COOLDOWN_DAYS = 20   # Cooldown days after stop loss

# Risk management
KILL_DD = 0.20        # Maximum drawdown before stop-loss (20%)
COOLDOWN_WEEKS = 4    # Weeks to remain in cash after stop-loss

# History requirements
HISTORY_DAYS = 260     # Days of history to fetch for calculations

# Timezone
TZ = pytz.timezone("America/New_York")

# Base directory for data/state files
BASE_DIR = Path(__file__).resolve().parent.parent # stock_trader_o3_algo

# File storage for backtesting
BACKTEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "backtest_results")
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
