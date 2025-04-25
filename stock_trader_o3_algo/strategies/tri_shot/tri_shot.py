import os
import json
import datetime as dt
import pytz
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# Import our feature engineering and model modules
from . import tri_shot_features as tsf
from .tri_shot_model import WalkForwardModel, load_walk_forward_model

# Constants
TICKERS = {
    "UP": "TQQQ",    # 3x long QQQ
    "DN": "SQQQ",    # 3x short QQQ
    "BOND": "TMF",   # 3x long treasury
    "CASH": "BIL",   # Short-term treasury ETF (cash equivalent)
    "SRC": "QQQ",    # Base asset to track
    "VIX": "^VIX"    # Volatility index
}

# Strategy parameters
VOL_TARGET = 0.25     # Reduced volatility target (25% instead of 30%)
ATR_MULT = 2.0        # Tighter ATR multiplier for trailing stops (2.0 instead of 2.5)
PROB_THRESHOLD_UP = 0.60   # Stronger conviction threshold for going long (0.60 vs 0.55)
PROB_THRESHOLD_DN = 0.40   # Stronger conviction threshold for going short (0.40 vs 0.45) 
CRASH_VIX_THRESHOLD = 25   # Lower VIX level to trigger crash protection (25 vs 28)
CRASH_RETURN_THRESHOLD = -0.05  # Less severe drawdown to trigger crash protection (-5% vs -6%)
CRASH_HEDGE_WEIGHT = 0.40  # Increased allocation to SQQQ during crash protection (40% vs 35%)
KILL_SWITCH_THRESHOLD = 0.80  # Less drawdown allowed before kill-switch (80% vs 75%)
COOLDOWN_DAYS = 15    # Longer cooldown after kill-switch (15 vs 10 days)

# File paths for state persistence
STATE_DIR = Path(os.path.join(os.getcwd(), "tri_shot_data"))
STOP_FILE = STATE_DIR / "stop.json"
COOLDOWN_FILE = STATE_DIR / "cooldown.txt"
MODEL_FILE = STATE_DIR / "model.pkl"
EQUITY_HIGH_FILE = STATE_DIR / "equity_high.txt"

# Timezone
TZ = pytz.timezone("America/New_York")

def ensure_state_dir():
    """Create the state directory if it doesn't exist."""
    if not STATE_DIR.exists():
        STATE_DIR.mkdir(parents=True)
        print(f"Created state directory: {STATE_DIR}")

def get_alpaca_api() -> tradeapi.REST:
    """Initialize Alpaca API client."""
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    return tradeapi.REST(api_key, api_secret, base_url)

def get_current_positions(api: tradeapi.REST) -> Dict[str, float]:
    """Get current positions and their quantities."""
    positions = {}
    try:
        for p in api.list_positions():
            positions[p.symbol] = float(p.qty)
    except Exception as e:
        print(f"Error getting positions: {e}")
    return positions

def calculate_position_direction(positions: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    """Determine the current position ticker and direction."""
    if not positions:
        return None, None
    
    for symbol, qty in positions.items():
        if symbol in [TICKERS["UP"], TICKERS["DN"], TICKERS["BOND"], TICKERS["CASH"]]:
            direction = "long" if qty > 0 else "short"
            return symbol, direction
    
    return None, None

def is_stop_hit(api: tradeapi.REST, ticker: str, direction: str) -> bool:
    """Check if the stop loss has been hit."""
    if not STOP_FILE.exists():
        return False
    
    try:
        stop_data = json.loads(STOP_FILE.read_text())
        if stop_data["asset"] != ticker:
            return False
        
        current_price = float(api.get_latest_trade(ticker).price)
        stop_level = float(stop_data["level"])
        
        if direction == "long" and current_price < stop_level:
            print(f"Stop hit for LONG {ticker}: Price {current_price} < Stop {stop_level}")
            return True
        elif direction == "short" and current_price > stop_level:
            print(f"Stop hit for SHORT {ticker}: Price {current_price} > Stop {stop_level}")
            return True
            
        return False
    except Exception as e:
        print(f"Error checking stop: {e}")
        return False

def is_in_cooldown() -> bool:
    """Check if we're in the cooldown period."""
    if not COOLDOWN_FILE.exists():
        return False
    
    try:
        cooldown_date = dt.date.fromisoformat(COOLDOWN_FILE.read_text().strip())
        return dt.date.today() <= cooldown_date
    except Exception as e:
        print(f"Error checking cooldown: {e}")
        return False

def start_cooldown():
    """Start the cooldown period."""
    cooldown_end = dt.date.today() + dt.timedelta(days=COOLDOWN_DAYS)
    COOLDOWN_FILE.write_text(str(cooldown_end))
    print(f"Started cooldown until {cooldown_end}")

def update_equity_high(api: tradeapi.REST):
    """Update the all-time equity high watermark."""
    try:
        current_equity = float(api.get_account().equity)
        
        if EQUITY_HIGH_FILE.exists():
            previous_high = float(EQUITY_HIGH_FILE.read_text().strip())
            if current_equity > previous_high:
                EQUITY_HIGH_FILE.write_text(str(current_equity))
                print(f"New equity high: ${current_equity:.2f}")
        else:
            EQUITY_HIGH_FILE.write_text(str(current_equity))
            print(f"Initial equity high: ${current_equity:.2f}")
    except Exception as e:
        print(f"Error updating equity high: {e}")

def is_kill_switch_triggered(api: tradeapi.REST) -> bool:
    """Check if the kill-switch should be triggered based on drawdown."""
    if not EQUITY_HIGH_FILE.exists():
        return False
    
    try:
        equity_high = float(EQUITY_HIGH_FILE.read_text().strip())
        current_equity = float(api.get_account().equity)
        drawdown = 1 - (current_equity / equity_high)
        
        if drawdown > (1 - KILL_SWITCH_THRESHOLD):
            print(f"Kill-switch triggered: ${current_equity:.2f} is {drawdown:.2%} below high of ${equity_high:.2f}")
            return True
        return False
    except Exception as e:
        print(f"Error checking kill-switch: {e}")
        return False

def calculate_vol_weight(prices: pd.DataFrame, asset: str, proba_up: float) -> float:
    """
    Calculate position weight based on volatility targeting and signal strength.
    
    Args:
        prices: DataFrame with price history
        asset: Asset symbol to size
        proba_up: Probability of upward move from model
    
    Returns:
        Position weight (0.0 to 1.0)
    """
    try:
        # Get volatility component
        asset_prices = prices[asset].iloc[-20:]
        sigma = asset_prices.pct_change().std() * np.sqrt(252)
        vol_weight = min(1.0, VOL_TARGET / sigma) if sigma > 0 else 0.0
        
        # Get signal strength component (0.5 = neutral, 1.0 = max confidence)
        signal_strength = abs(proba_up - 0.5) / 0.5
        
        # Combined weight with floor
        weight = max(0.10, vol_weight * signal_strength)
        
        print(f"Asset: {asset}, Volatility: {sigma:.2%}, Signal Strength: {signal_strength:.2f}, Weight: {weight:.2f}")
        return weight
    except Exception as e:
        print(f"Error calculating vol weight: {e}")
        return 0.25  # Conservative default

def calculate_atr(prices: pd.DataFrame, asset: str) -> float:
    """Calculate the Average True Range for the asset."""
    try:
        return prices[asset].pct_change().abs().rolling(14).mean().iloc[-1] * prices[asset].iloc[-1]
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return 0.0

def update_stop_loss(prices: pd.DataFrame, asset: str, direction: str):
    """Update the stop loss level based on dynamic ATR."""
    try:
        current_price = prices[asset].iloc[-1]
        atr = calculate_atr(prices, asset)
        
        # Dynamic ATR multiplier based on VIX level
        dynamic_atr_mult = ATR_MULT
        if '^VIX' in prices.columns:
            vix_level = prices['^VIX'].iloc[-1]
            # Reduce ATR multiplier in high volatility environments
            if vix_level > 30:
                dynamic_atr_mult = max(1.0, ATR_MULT - 1.0)  # Tighter stop in high vol
            elif vix_level > 25:
                dynamic_atr_mult = max(1.2, ATR_MULT - 0.8)  # Somewhat tighter stop
            elif vix_level < 15:
                dynamic_atr_mult = min(3.0, ATR_MULT + 0.5)  # Wider stop in low vol
        
        if direction == "long":
            stop_level = current_price - (dynamic_atr_mult * atr)
        else:  # short
            stop_level = current_price + (dynamic_atr_mult * atr)
        
        stop_data = {"asset": asset, "level": float(stop_level), "atr_mult": float(dynamic_atr_mult)}
        STOP_FILE.write_text(json.dumps(stop_data))
        print(f"Updated {direction.upper()} stop for {asset}: {stop_level:.2f} (ATR mult: {dynamic_atr_mult:.1f})")
    except Exception as e:
        print(f"Error updating stop loss: {e}")

def check_for_crash(prices: pd.DataFrame) -> bool:
    """Check if we're in a market crash scenario."""
    try:
        vix_level = prices[TICKERS["VIX"]].iloc[-1]
        five_day_return = prices[TICKERS["SRC"]].iloc[-1] / prices[TICKERS["SRC"]].iloc[-5] - 1
        
        is_crash = vix_level > CRASH_VIX_THRESHOLD and five_day_return < CRASH_RETURN_THRESHOLD
        
        if is_crash:
            print(f"CRASH DETECTED: VIX at {vix_level:.2f}, 5-day return {five_day_return:.2%}")
        
        return is_crash
    except Exception as e:
        print(f"Error checking for crash: {e}")
        return False

def place_order(api: tradeapi.REST, symbol: str, notional: float):
    """Place a market order for a specific notional amount."""
    if notional <= 1:
        print(f"Skipping order for {symbol}: notional amount too small (${notional:.2f})")
        return
    
    try:
        order = api.submit_order(
            symbol=symbol,
            notional=round(notional, 2),
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"Placed order for ${notional:.2f} of {symbol}")
        return order
    except Exception as e:
        print(f"Error placing order for {symbol}: {e}")

def close_all_positions(api: tradeapi.REST):
    """Close all open positions."""
    try:
        api.close_all_positions()
        print("Closed all positions")
    except Exception as e:
        print(f"Error closing positions: {e}")

def run_monday_strategy(api: tradeapi.REST):
    """
    Monday strategy: Determine directional view and enter position.
    """
    print("Running Monday strategy...")
    
    # Check if we're in cooldown
    if is_in_cooldown():
        print("In cooldown period. No trading today.")
        return
    
    # Fetch data
    tickers = list(TICKERS.values())
    prices = tsf.fetch_data(tickers, days=300)
    
    # Load model
    if not MODEL_FILE.exists():
        print(f"Model file not found at {MODEL_FILE}. Training new model...")
        model = WalkForwardModel(prices)
        model.save(MODEL_FILE)
    else:
        try:
            model = load_walk_forward_model(MODEL_FILE)
        except Exception as e:
            print(f"Error loading model: {e}. Training new model...")
            model = WalkForwardModel(prices)
            model.save(MODEL_FILE)
    
    # Get latest features
    latest_features = tsf.latest_features(prices)
    
    # Predict market direction
    proba_up = model.predict(latest_features)[0, 1]
    print(f"Probability of up move: {proba_up:.2%}")
    
    # Decide on asset
    if proba_up >= PROB_THRESHOLD_UP:
        asset = TICKERS["UP"]
        direction = "long"
        print(f"High probability of up move ({proba_up:.2%} > {PROB_THRESHOLD_UP}). Going LONG with {asset}")
    elif proba_up <= PROB_THRESHOLD_DN:
        asset = TICKERS["DN"]
        direction = "short"
        print(f"High probability of down move ({proba_up:.2%} < {PROB_THRESHOLD_DN}). Going SHORT with {asset}")
    else:
        # No clear edge, check bonds
        if prices["TLT"].pct_change(20).iloc[-1] > 0:
            asset = TICKERS["BOND"]
            direction = "long"
            print(f"No edge ({PROB_THRESHOLD_DN} < {proba_up:.2%} < {PROB_THRESHOLD_UP}), but bonds are rising. Using {asset}")
        else:
            asset = TICKERS["CASH"]
            direction = "long"
            print(f"No edge ({PROB_THRESHOLD_DN} < {proba_up:.2%} < {PROB_THRESHOLD_UP}) and bonds are falling. Going to CASH with {asset}")
    
    # Calculate weight based on volatility and signal strength
    weight = calculate_vol_weight(prices, asset, proba_up)
    account = api.get_account()
    equity = float(account.equity)
    notional = equity * weight
    
    # Update equity high water mark
    update_equity_high(api)
    
    # Check for kill switch
    if is_kill_switch_triggered(api):
        close_all_positions(api)
        place_order(api, TICKERS["CASH"], equity)
        start_cooldown()
        return
    
    # Execute the trade
    close_all_positions(api)
    if asset != TICKERS["CASH"]:
        place_order(api, asset, notional)
        
        # Set initial stop loss
        update_stop_loss(prices, asset, direction)
    
    # Check for crash overlay
    if check_for_crash(prices) and asset != TICKERS["DN"]:
        hedge_notional = equity * CRASH_HEDGE_WEIGHT
        print(f"Adding crash hedge: ${hedge_notional:.2f} of {TICKERS['DN']}")
        place_order(api, TICKERS["DN"], hedge_notional)

def run_wednesday_strategy(api: tradeapi.REST):
    """
    Wednesday strategy: Volatility rebalancing.
    """
    print("Running Wednesday strategy...")
    
    # Check if we're in cooldown
    if is_in_cooldown():
        print("In cooldown period. No trading today.")
        return
    
    # Get current positions
    positions = get_current_positions(api)
    ticker, direction = calculate_position_direction(positions)
    
    if ticker is None or ticker == TICKERS["CASH"]:
        print("No active position to rebalance.")
        return
    
    # Check for stop loss hit
    if is_stop_hit(api, ticker, direction):
        print(f"Stop loss hit for {ticker}. Exiting position.")
        close_all_positions(api)
        account = api.get_account()
        place_order(api, TICKERS["CASH"], float(account.equity))
        start_cooldown()
        return
    
    # Fetch data
    tickers = list(TICKERS.values())
    prices = tsf.fetch_data(tickers, days=300)
    
    # Update equity high water mark
    update_equity_high(api)
    
    # Check for kill switch
    if is_kill_switch_triggered(api):
        close_all_positions(api)
        account = api.get_account()
        place_order(api, TICKERS["CASH"], float(account.equity))
        start_cooldown()
        return
    
    # Rebalance based on volatility
    weight = calculate_vol_weight(prices, ticker, 0.5)  # Use neutral signal strength for rebalancing
    account = api.get_account()
    equity = float(account.equity)
    current_value = sum([float(p.market_value) for p in api.list_positions() if p.symbol == ticker])
    target_value = equity * weight
    
    # Only rebalance if the difference is significant (>5%)
    if abs(current_value - target_value) / equity > 0.05:
        print(f"Rebalancing {ticker}: Current ${current_value:.2f}, Target ${target_value:.2f}")
        
        # Close positions and re-enter with correct size
        close_all_positions(api)
        place_order(api, ticker, target_value)
        
        # Update stop loss
        update_stop_loss(prices, ticker, direction)
    else:
        print(f"No significant rebalance needed for {ticker}")
    
    # Check for crash overlay
    if check_for_crash(prices) and ticker != TICKERS["DN"]:
        # If we already have a crash hedge, don't add another
        has_hedge = any(p.symbol == TICKERS["DN"] for p in api.list_positions())
        if not has_hedge:
            hedge_notional = equity * CRASH_HEDGE_WEIGHT
            print(f"Adding crash hedge: ${hedge_notional:.2f} of {TICKERS['DN']}")
            place_order(api, TICKERS["DN"], hedge_notional)

def run_friday_strategy(api: tradeapi.REST):
    """
    Friday strategy: Risk gate check.
    """
    print("Running Friday strategy...")
    
    # Check if we're in cooldown
    if is_in_cooldown():
        print("In cooldown period. No trading today.")
        return
    
    # Get current positions
    positions = get_current_positions(api)
    ticker, direction = calculate_position_direction(positions)
    
    if ticker is None or ticker == TICKERS["CASH"]:
        print("No active position to check.")
        return
    
    # Check for stop loss hit
    if is_stop_hit(api, ticker, direction):
        print(f"Stop loss hit for {ticker}. Exiting position.")
        close_all_positions(api)
        account = api.get_account()
        place_order(api, TICKERS["CASH"], float(account.equity))
        start_cooldown()
        return
    
    # Fetch data
    tickers = list(TICKERS.values())
    prices = tsf.fetch_data(tickers, days=300)
    
    # Update equity high water mark
    update_equity_high(api)
    
    # Check for kill switch
    if is_kill_switch_triggered(api):
        close_all_positions(api)
        account = api.get_account()
        place_order(api, TICKERS["CASH"], float(account.equity))
        start_cooldown()
        return
    
    # Check for crash
    if check_for_crash(prices):
        if ticker != TICKERS["DN"]:  # If not already short
            print("Crash detected. Moving to cash then adding crash hedge.")
            close_all_positions(api)
            account = api.get_account()
            equity = float(account.equity)
            
            # Allocate portion to crash hedge
            hedge_notional = equity * CRASH_HEDGE_WEIGHT
            place_order(api, TICKERS["DN"], hedge_notional)
            
            # Rest to cash
            place_order(api, TICKERS["CASH"], equity - hedge_notional)
        return
    
    # If we're still holding, update the stop loss for the weekend
    update_stop_loss(prices, ticker, direction)
    print(f"Updated stop loss for {ticker} for the weekend.")

def main():
    """Main function to run the tri-shot strategy based on the day of the week."""
    # Ensure state directory exists
    ensure_state_dir()
    
    # Initialize API
    api = get_alpaca_api()
    
    # Get current time in NY
    now = dt.datetime.now(TZ)
    day_of_week = now.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday
    current_time = now.time()
    
    # Run appropriate strategy based on day and time
    if day_of_week == 0:  # Monday
        # After market close (4:05 PM ET)
        target_time = dt.time(16, 5)
        if current_time >= target_time:
            run_monday_strategy(api)
        else:
            print(f"Waiting for Monday 16:05 ET. Current time: {current_time}")
    
    elif day_of_week == 2:  # Wednesday
        # Mid-day (11:30 AM ET)
        target_time = dt.time(11, 30)
        if current_time >= target_time:
            run_wednesday_strategy(api)
        else:
            print(f"Waiting for Wednesday 11:30 ET. Current time: {current_time}")
    
    elif day_of_week == 4:  # Friday
        # Before market close (3:45 PM ET)
        target_time = dt.time(15, 45)
        if current_time >= target_time:
            run_friday_strategy(api)
        else:
            print(f"Waiting for Friday 15:45 ET. Current time: {current_time}")
    
    else:
        print(f"No scheduled strategy for day {day_of_week} (0=Monday, 6=Sunday)")

if __name__ == "__main__":
    main()
