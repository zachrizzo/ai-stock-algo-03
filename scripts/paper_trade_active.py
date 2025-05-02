#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMT Strategy Active Paper Trading with Binance
=============================================
Run a more active version of the DMT_v2 strategy in paper trading mode.

This script is a modified version of paper_trade.py that uses more aggressive
parameters for the DMT_v2 strategy for better performance in paper trading.

Usage:
    paper_trade_active.py [--symbol SYMBOL] [--interval INTERVAL] [--capital AMOUNT] [--version VERSION]
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client as BinanceClient
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

# Load variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our modules
from stock_trader_o3_algo.strategies.dmt_v2_strategy import DMT_v2_Strategy
from scripts.paper_trade import BinancePaperTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trade_active.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trade_active")


class ActiveDMTStrategy(DMT_v2_Strategy):
    """A more active version of the DMT_v2 strategy with reduced neutral zone"""
    
    def __init__(self, 
                 version: str = "turbo", 
                 asset_type: str = "equity",
                 lookback_period: int = 252,
                 initial_capital: float = 10000.0):
        """Initialize with parent class then override parameters"""
        super().__init__(version, asset_type, lookback_period, initial_capital)
        
        # Make strategy more active by reducing neutral zone
        self.neutral_zone = 0.01  # Reduced from 0.03/0.05
        
        # Increase position sizing slightly
        self.max_position *= 1.2
        
        # Add a small long bias for crypto
        if self.asset_type == "crypto":
            self.core_long_bias = 0.1
            
        logger.info(f"Active DMT parameters: neutral_zone={self.neutral_zone}, max_position={self.max_position}")


class ActivePaperTrader(BinancePaperTrader):
    """Paper trading implementation using the more active DMT strategy"""
    
    def __init__(self, **kwargs):
        """Initialize with parent class constructor"""
        super().__init__(**kwargs)
        
        # Override the strategy with our active version
        self.strategy = ActiveDMTStrategy(
            version=self.strategy_version,
            asset_type=self.asset_type
        )
        
        print(f"✅ Using ActiveDMTStrategy with {self.strategy_version} version")
        
    def _load_initial_data(self):
        """Load more historical data for better context"""
        try:
            # Determine how many bars to load based on interval
            lookback_periods = {
                '1m': 5000,   # More 1-minute data for better context
                '5m': 1000,   # More 5-minute data
                '15m': 500,   # More 15-minute data
                '30m': 300,   # More 30-minute data
                '1h': 200,    # More hourly data
                '4h': 150,    # More 4-hour data
                '1d': 200     # More daily data
            }
            
            # Get the number of bars to load
            bars_to_load = lookback_periods.get(self.interval, 200)
            
            print(f"\n📊 Preloading {bars_to_load} historical {self.interval} bars to build context...")
            
            # Get historical klines
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=bars_to_load
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string columns to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Store the data
            self.klines = df
            
            print(f"✅ Loaded {len(df)} bars of {self.interval} historical data")
            print(f"📋 Data columns: {', '.join(df.columns[:5])}")
            
            # Set the last timestamp
            if not df.empty:
                self.last_timestamp = df.index[-1].timestamp() * 1000
                self.current_price = df['Close'].iloc[-1]
            
            print("\n🧠 Building model context from historical data...")
            
            # Run backtest on historical data to build context
            backtest_results, _ = self.strategy.run_backtest(df)
            
            # Get the latest regime
            if len(backtest_results) > 0:
                latest_regime = backtest_results['regime'].iloc[-1]
                logger.info(f"Current market regime: {latest_regime}")
                
        except Exception as e:
            print(f"⚠️ Error loading initial data: {e}")
            self.klines = pd.DataFrame()
            self.last_timestamp = 0


def main():
    """Main entry point for active paper trading script"""
    parser = argparse.ArgumentParser(
        description="Active DMT Paper Trading on Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to trade')
    parser.add_argument('--interval', type=str, choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                      default='1h', help='Trading interval')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--version', type=str, choices=['original', 'enhanced', 'turbo'],
                      default='enhanced', help='Strategy version')
    parser.add_argument('--trading-tier', type=str, 
                      choices=['regular', 'vip1', 'vip2', 'vip3', 'vip4', 'vip5', 'vip6', 'vip7', 'vip8', 'vip9'],
                      default='regular', help='Trading tier for fee calculation')
    parser.add_argument('--no-bnb-discount', action='store_true', 
                      help='Disable BNB fee discount')
    parser.add_argument('--simulation', action='store_true',
                      help='Run in full simulation mode without attempting API trades')
    parser.add_argument('--use-binance-us', action='store_true',
                      help='Use Binance US API instead of Binance Global Testnet (not recommended for paper trading)')

    # Constants and configuration
    API_KEY_ENV_NAME = 'BINANCE_API_KEY_TEST'  # Environment variable for testnet API key
    API_SECRET_ENV_NAME = 'BINANCE_API_SECRET_TEST'  # Environment variable for testnet API secret

    # Warn if using non-testnet keys
    if 'BINANCE_API_KEY' in os.environ and not 'BINANCE_API_KEY_TEST' in os.environ:
        print("\n⚠️ WARNING: You're using the production API key. Please use testnet keys instead.")
        print("Set BINANCE_API_KEY_TEST and BINANCE_API_SECRET_TEST environment variables.\n")
        print("Exiting for safety...")
        sys.exit(1)

    # Load API keys from environment variables
    api_key = os.environ.get(API_KEY_ENV_NAME)
    api_secret = os.environ.get(API_SECRET_ENV_NAME)

    if not api_key or not api_secret:
        print("\n❌ Error: Binance testnet API keys not found in environment variables")
        print(f"Please set {API_KEY_ENV_NAME} and {API_SECRET_ENV_NAME}")
        print("\nTo get testnet API keys:")
        print("1. Go to https://testnet.binancefuture.com/en/futures")
        print("2. Create a testnet account")
        print("3. Generate API keys from your testnet dashboard")
        print("4. Add to your environment with:")
        print(f"   export {API_KEY_ENV_NAME}=\"your_testnet_api_key\"")
        print(f"   export {API_SECRET_ENV_NAME}=\"your_testnet_api_secret\"\n")
        sys.exit(1)

    print("\n✅ Found Binance API keys in environment variables")

    args = parser.parse_args()
    
    # Create and run paper trader
    trader = ActivePaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbol=args.symbol,
        interval=args.interval,
        initial_capital=args.capital,
        strategy_version=args.version,
        use_bnb_for_fees=not args.no_bnb_discount,
        trading_tier=args.trading_tier,
        simulation_mode=args.simulation,
        use_binance_us=args.use_binance_us
    )
    
    # Determine check interval based on trading interval
    interval_to_seconds = {
        '1m': 10,
        '5m': 30,
        '15m': 60,
        '30m': 60,
        '1h': 300,
        '4h': 600,
        '1d': 3600
    }
    check_interval = interval_to_seconds.get(args.interval, 60)
    
    trader.run(check_interval_seconds=check_interval)


if __name__ == "__main__":
    main()
