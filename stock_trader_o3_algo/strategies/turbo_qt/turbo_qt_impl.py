#!/usr/bin/env python
"""
Implementation functions for Turbo-Rotational "QQQ Turbo-4" Strategy

This module contains the main execution functions for the strategy:
- rebalance: Monday rebalancing logic
- check_stops: Daily stop-loss check
- main: Entry point for running as a script
"""

import os
import json
import datetime as dt
import pytz
import pandas as pd
import numpy as np

from turbo_qt import (
    get_prices, choose_asset, calculate_atr, check_crash_conditions,
    save_stop_price, check_stop_hit, get_alpaca_api,
    TICKERS, VOL_TARGET, ATR_MULT, HEDGE_WEIGHT, KILL_DD, COOLDOWN_WEEKS,
    TZ, DATA_DIR, STOP_FILE, STATE_FILE
)


def rebalance(dry_run: bool = False) -> dict:
    """
    Perform Monday rebalancing for the Turbo-4 strategy.
    
    Args:
        dry_run: If True, don't execute trades, just return allocations
        
    Returns:
        Dictionary with allocation details
    """
    print(f"Running rebalance at {dt.datetime.now(tz=TZ)}")
    
    # Get current prices
    prices = get_prices()
    
    # Check if in cooldown
    in_cooldown = False
    cooldown_until = None
    
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            if "cooldown_until" in state:
                cooldown_date = dt.datetime.fromisoformat(state["cooldown_until"])
                in_cooldown = dt.datetime.now(tz=TZ) < cooldown_date
                if in_cooldown:
                    cooldown_until = cooldown_date
    
    # If in cooldown, allocate to cash
    if in_cooldown:
        print(f"In cooldown period until {cooldown_until}")
        allocation = {TICKERS["CASH"]: 1.0}
        
        if not dry_run:
            # Initialize API
            api = get_alpaca_api()
            
            # Close all positions
            api.close_all_positions()
            
            # Get account equity
            account = api.get_account()
            equity = float(account.cash)
            
            # Allocate to cash
            api.submit_order(
                symbol=TICKERS["CASH"],
                notional=equity,
                side="buy",
                type="market",
                time_in_force="day"
            )
        
        return {"allocations": allocation, "in_cooldown": True}
    
    # Choose asset based on momentum signals
    asset = choose_asset(prices)
    print(f"Selected asset: {asset}")
    
    # Calculate volatility for position sizing
    if asset != TICKERS["CASH"]:
        sigma = prices[asset].pct_change().dropna().tail(20).std() * np.sqrt(252)
        weight = min(1.0, VOL_TARGET / sigma) if sigma > 0 else 0.0
    else:
        weight = 1.0
        sigma = 0.0
    
    print(f"Asset volatility: {sigma:.2%}, Target weight: {weight:.2%}")
    
    # Check for crash conditions
    crash_conditions = check_crash_conditions(prices)
    hedge_needed = (asset != TICKERS["DN"] and crash_conditions)
    
    # Calculate allocations
    allocations = {}
    
    if asset != TICKERS["CASH"]:
        # Allocate to selected asset
        allocations[asset] = weight * (1.0 - (HEDGE_WEIGHT if hedge_needed else 0.0))
        
        # Add hedge if needed
        if hedge_needed:
            allocations[TICKERS["DN"]] = HEDGE_WEIGHT
        
        # Allocate remainder to cash
        allocations[TICKERS["CASH"]] = max(0.0, 1.0 - sum(allocations.values()))
    else:
        # Allocate everything to cash
        allocations[TICKERS["CASH"]] = 1.0
    
    # Execute trades if not dry run
    if not dry_run:
        # Initialize API
        api = get_alpaca_api()
        
        # Close all positions
        api.close_all_positions()
        
        # Get account equity
        account = api.get_account()
        equity = float(account.cash)
        
        # Check for kill-switch (max drawdown)
        peak_equity = equity
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                if "peak_equity" in state:
                    peak_equity = max(equity, state["peak_equity"])
        
        # Update peak equity in state
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({"peak_equity": peak_equity}, f)
        
        # Check for kill-switch
        drawdown = (equity / peak_equity) - 1
        if drawdown < -KILL_DD:
            print(f"KILL-SWITCH TRIGGERED: Drawdown {drawdown:.2%} exceeds threshold {-KILL_DD:.2%}")
            # Set cooldown period
            cooldown_until = dt.datetime.now(tz=TZ) + dt.timedelta(weeks=COOLDOWN_WEEKS)
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "peak_equity": peak_equity,
                    "cooldown_until": cooldown_until.isoformat()
                }, f)
            
            # Allocate to cash
            api.submit_order(
                symbol=TICKERS["CASH"],
                notional=equity,
                side="buy",
                type="market",
                time_in_force="day"
            )
            
            return {"allocations": {TICKERS["CASH"]: 1.0}, "kill_switch": True}
        
        # Execute allocation trades
        for symbol, alloc in allocations.items():
            if alloc > 0.001:  # Only trade if allocation is significant
                api.submit_order(
                    symbol=symbol,
                    notional=equity * alloc,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
        
        # Calculate and save stop prices
        if asset in [TICKERS["UP"], TICKERS["DN"], TICKERS["BOND"]]:
            # Calculate ATR
            atr_value = calculate_atr(prices[asset])
            
            # Calculate stop price
            if asset == TICKERS["UP"] or asset == TICKERS["BOND"]:
                # For long positions, stop is below current price
                stop_price = prices[asset].iloc[-1] - (ATR_MULT * atr_value)
            else:
                # For short positions, stop is above current price
                stop_price = prices[asset].iloc[-1] + (ATR_MULT * atr_value)
            
            # Save stop price
            save_stop_price(asset, stop_price)
            
            print(f"Stop price for {asset}: {stop_price:.2f}")
    
    return {"allocations": allocations, "hedge": hedge_needed}


def check_stops(dry_run: bool = False) -> dict:
    """
    Check if any stop has been hit and execute stop-out if needed.
    
    Args:
        dry_run: If True, don't execute trades, just return status
        
    Returns:
        Dictionary with stop check results
    """
    print(f"Running stop check at {dt.datetime.now(tz=TZ)}")
    
    # Check if stop file exists
    if not os.path.exists(STOP_FILE):
        print("No active stop found")
        return {"stop_hit": False}
    
    # Initialize API if not dry run
    api = None
    if not dry_run:
        api = get_alpaca_api()
    
    # Check if stop is hit
    stop_hit = check_stop_hit(api if not dry_run else None)
    
    if stop_hit:
        print("STOP HIT - Executing stop-out")
        
        if not dry_run:
            # Close all positions
            api.close_all_positions()
            
            # Get account equity
            account = api.get_account()
            equity = float(account.cash)
            
            # Allocate to cash
            api.submit_order(
                symbol=TICKERS["CASH"],
                notional=equity,
                side="buy",
                type="market",
                time_in_force="day"
            )
            
            # Remove stop file
            os.remove(STOP_FILE)
    
    return {"stop_hit": stop_hit}


def main():
    """Main function for running as a command-line script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Turbo-4 Strategy Execution")
    parser.add_argument("--rebalance", action="store_true", help="Run Monday rebalancing")
    parser.add_argument("--check-stops", action="store_true", help="Check stop-loss levels")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades")
    parser.add_argument("--backtest", action="store_true", help="Run backtest simulation")
    
    args = parser.parse_args()
    
    if args.rebalance:
        result = rebalance(dry_run=args.dry_run)
        print(f"Rebalance result: {result}")
    
    if args.check_stops:
        result = check_stops(dry_run=args.dry_run)
        print(f"Stop check result: {result}")
    
    if not args.rebalance and not args.check_stops:
        # Default: determine what to do based on day of week and time
        now = dt.datetime.now(tz=TZ)
        
        # Check if it's Monday after market close (16:00 ET)
        is_monday = now.weekday() == 0
        is_after_close = now.hour >= 16
        
        if is_monday and is_after_close:
            result = rebalance(dry_run=args.dry_run)
            print(f"Monday rebalance result: {result}")
        else:
            # Any day, run stop check
            result = check_stops(dry_run=args.dry_run)
            print(f"Daily stop check result: {result}")


if __name__ == "__main__":
    main()
