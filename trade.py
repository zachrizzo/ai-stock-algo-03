"""
Main script for running the micro-CTA strategy in live trading mode.
"""
import argparse
import datetime as dt
import logging
import os
import sys
import time
from typing import Dict, Optional

import pandas as pd
import pytz

from stock_trader_o3_algo.config.settings import (
    RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF,
    HISTORY_DAYS, LOOKBACK_DAYS, VOL_LOOK,
    WEEKLY_VOL_TARGET, CRASH_THRESHOLD, HEDGE_WEIGHT, KILL_DD
)
from stock_trader_o3_algo.data.price_data import fetch_prices
from stock_trader_o3_algo.execution.alpaca_trader import AlpacaTrader


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("micro_cta.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("micro_cta")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run micro-CTA strategy in live mode')
    
    # Mode arguments
    parser.add_argument('--paper', action='store_true',
                        help='Use paper trading mode')
    
    # Dry run arguments
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode (no actual trading)')
    
    return parser.parse_args()


def is_market_open() -> bool:
    """
    Check if the market is currently open.
    
    Returns:
        True if the market is open, False otherwise
    """
    # Create a trader instance to check market status
    trader = AlpacaTrader()
    
    try:
        clock = trader.api.get_clock()
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False


def is_trading_day() -> bool:
    """
    Check if today is a trading day.
    
    Returns:
        True if today is a trading day, False otherwise
    """
    # Create a trader instance to check market calendar
    trader = AlpacaTrader()
    
    try:
        # Get today's date in Eastern Time
        now = dt.datetime.now(pytz.timezone('America/New_York'))
        today = now.strftime('%Y-%m-%d')
        
        # Check if today is a trading day
        calendar = trader.api.get_calendar(start=today, end=today)
        return len(calendar) > 0
    except Exception as e:
        logger.error(f"Error checking trading calendar: {e}")
        return False


def is_execution_time() -> bool:
    """
    Check if it's the right time to execute the strategy (Monday at 9:35 ET).
    
    Returns:
        True if it's the right time, False otherwise
    """
    # Get current time in Eastern Time
    now = dt.datetime.now(pytz.timezone('America/New_York'))
    
    # Check if it's Monday (0 = Monday in datetime.weekday())
    is_monday = now.weekday() == 0
    
    # Check if it's between 9:35 and 9:40 (to give a window for execution)
    is_execution_window = (
        (now.hour == 9 and now.minute >= 35 and now.minute <= 40) or
        # Also allow for manual testing at other times with the environment variable
        os.environ.get("FORCE_EXECUTION") == "1"
    )
    
    return is_monday and is_execution_window


def wait_for_market_open():
    """Wait until the market opens if it's not already open."""
    while not is_market_open():
        # Check every 5 minutes
        logger.info("Market is closed. Waiting for market to open...")
        time.sleep(5 * 60)  # 5 minutes


def execute_strategy(dry_run: bool = False) -> Dict:
    """
    Execute the micro-CTA strategy.
    
    Args:
        dry_run: If True, don't actually place trades
        
    Returns:
        Dictionary with execution information
    """
    logger.info("Executing micro-CTA strategy")
    
    # Fetch price data
    logger.info("Fetching price data")
    tickers = [RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF]
    prices = fetch_prices(tickers, days=HISTORY_DAYS)
    
    # Create trader instance
    trader = AlpacaTrader()
    
    if dry_run:
        logger.info("DRY RUN MODE - no trades will be executed")
        
        # Calculate allocation without placing trades
        account_info = trader.get_account_info()
        equity = account_info['equity']
        peak_equity = trader.update_peak_equity(equity)
        
        # Get target allocation
        from stock_trader_o3_algo.core.strategy import get_portfolio_allocation
        allocation = get_portfolio_allocation(
            prices,
            equity=equity,
            equity_peak=peak_equity
        )
        
        logger.info(f"Current equity: ${equity:.2f}")
        logger.info(f"Peak equity: ${peak_equity:.2f}")
        logger.info(f"Target allocation: {allocation}")
        
        # Calculate trades that would be made
        current_positions = trader.get_positions()
        trades = []
        
        for symbol, target_amount in allocation.items():
            current_amount = current_positions.get(symbol, 0)
            trade_amount = target_amount - current_amount
            
            if abs(trade_amount) > 0.01:
                trades.append({
                    'symbol': symbol,
                    'amount': trade_amount,
                    'action': 'BUY' if trade_amount > 0 else 'SELL'
                })
        
        logger.info(f"Trades that would be executed: {trades}")
        
        return {
            'date': prices.index[-1],
            'status': 'DRY_RUN',
            'equity': equity,
            'peak_equity': peak_equity,
            'allocation': allocation,
            'trades': trades
        }
    else:
        # Execute the strategy
        result = trader.execute_strategy(prices)
        
        # Log the result
        logger.info(f"Strategy executed with status: {result['status']}")
        logger.info(f"Current equity: ${result['equity']:.2f}")
        logger.info(f"Peak equity: ${result['peak_equity']:.2f}")
        logger.info(f"Orders placed: {len(result['orders'])}")
        
        return result


def main():
    """Main entry point for the trading script."""
    args = parse_args()
    
    logger.info("Starting micro-CTA trading system")
    
    # Check if it's a trading day
    if not is_trading_day():
        logger.info("Today is not a trading day. Exiting.")
        return
    
    # Check if it's the right time to execute
    if not is_execution_time():
        logger.info("Not execution time yet. Exiting.")
        return
    
    # Wait for market open if necessary
    if not args.dry_run:
        wait_for_market_open()
    
    # Execute the strategy
    execute_strategy(dry_run=args.dry_run)
    
    logger.info("Execution completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        
        # Attempt to send notification of failure
        try:
            # Simple webhook notification example - replace with your preferred notification method
            import requests
            
            # Check if webhook URL is defined
            webhook_url = os.environ.get("NOTIFICATION_WEBHOOK")
            if webhook_url:
                requests.post(
                    webhook_url,
                    json={
                        'text': f"ERROR in micro-CTA: {str(e)}",
                        'timestamp': dt.datetime.now().isoformat()
                    }
                )
        except Exception as notify_err:
            logger.error(f"Failed to send error notification: {notify_err}")
