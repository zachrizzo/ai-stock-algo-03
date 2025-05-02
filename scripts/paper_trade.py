#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMT Strategy Paper Trading with Binance
======================================
Run the DMT_v2 strategy in paper trading mode against Binance's testnet.

This script connects to the Binance testnet and executes trades based on
the TurboDMT_v2 strategy signals. It provides a realistic simulation of
live trading without risking real money.

Usage:
    paper_trade.py [--symbol SYMBOL] [--interval INTERVAL] [--capital AMOUNT] [--version VERSION]

Options:
    --symbol SYMBOL       Symbol to trade [default: BTCUSDT]
    --interval INTERVAL   Timeframe for trading [default: 1d]
    --capital AMOUNT      Initial capital [default: 10000]
    --version VERSION     Strategy version (original, enhanced, turbo) [default: turbo]
"""

import os
import sys
import time
import logging
import datetime as dt
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our modules
from stock_trader_o3_algo.strategies.dmt_v2_strategy import DMT_v2_Strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trade")


class BinancePaperTrader:
    """Paper trading implementation for Binance testnet"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        initial_capital: float = 10000.0,
        strategy_version: str = "turbo",
        asset_type: str = "crypto"
    ):
        """
        Initialize the paper trader
        
        Args:
            api_key: Binance testnet API key
            api_secret: Binance testnet API secret
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Trading interval ('1m', '5m', '1h', '4h', '1d')
            initial_capital: Starting capital in USDT
            strategy_version: DMT_v2 strategy version
            asset_type: Asset type for strategy parameters
        """
        self.symbol = symbol
        self.interval = interval
        self.initial_capital = initial_capital
        self.strategy_version = strategy_version
        self.asset_type = asset_type
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=True)
        logger.info(f"Connected to Binance Testnet as {self.client.get_account()['email']}")
        
        # Initialize strategy
        self.strategy = DMT_v2_Strategy(
            version=strategy_version,
            asset_type=asset_type,
            lookback_period=252,
            initial_capital=initial_capital
        )
        
        # Trading state
        self.current_position = 0.0
        self.cash = initial_capital
        self.holdings = 0.0
        self.equity = initial_capital
        self.trades = []
        
        # Market data
        self.historical_data = pd.DataFrame()
        self.last_update_time = None
        
        # Load initial data
        self._load_initial_data()
    
    def _load_initial_data(self) -> None:
        """Load historical data to initialize the strategy"""
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(self.interval, Client.KLINE_INTERVAL_1DAY)
        
        try:
            # Calculate how many bars we need based on strategy lookback
            lookback_days = 252  # Default lookback period
            
            # Adjust for interval
            if self.interval == "1d":
                lookback_bars = lookback_days
            elif self.interval == "4h":
                lookback_bars = lookback_days * 6  # 6 4-hour bars per day
            elif self.interval == "1h":
                lookback_bars = lookback_days * 24  # 24 hours per day
            else:
                # For minute-based intervals, cap at 1000 to avoid API limits
                lookback_bars = 1000
            
            # Get historical klines
            klines = self.client.get_historical_klines(
                self.symbol, 
                binance_interval,
                f"{lookback_bars} {self.interval} ago UTC"
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 
                'buy_base_volume', 'buy_quote_volume', 'ignored'
            ])
            
            # Format data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype({
                'Open': float, 'High': float, 'Low': float, 
                'Close': float, 'Volume': float
            })
            
            self.historical_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            logger.info(f"Loaded {len(self.historical_data)} bars of {self.interval} historical data")
            
            # Set last update time
            self.last_update_time = self.historical_data.index[-1]
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            sys.exit(1)
    
    def _update_market_data(self) -> bool:
        """
        Update market data with latest price information
        
        Returns:
            bool: True if new data was added, False otherwise
        """
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(self.interval, Client.KLINE_INTERVAL_1DAY)
        
        try:
            # Get latest klines since last update
            start_time = int(self.last_update_time.timestamp() * 1000) + 1
            klines = self.client.get_historical_klines(
                self.symbol,
                binance_interval,
                start_str=start_time,
                limit=10  # Just get the latest few bars
            )
            
            if not klines:
                logger.debug("No new data available")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_volume', 'trades', 
                'buy_base_volume', 'buy_quote_volume', 'ignored'
            ])
            
            # Format data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype({
                'Open': float, 'High': float, 'Low': float, 
                'Close': float, 'Volume': float
            })
            
            new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Filter out any rows we already have
            new_data = new_data[~new_data.index.isin(self.historical_data.index)]
            
            if len(new_data) == 0:
                return False
            
            # Append new data
            self.historical_data = pd.concat([self.historical_data, new_data])
            
            # Update last update time
            self.last_update_time = self.historical_data.index[-1]
            
            logger.info(f"Added {len(new_data)} new data points. Latest: {self.last_update_time}")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return False
    
    def _calculate_signal(self) -> float:
        """
        Calculate trading signal based on current market data
        
        Returns:
            float: Position size (-1 to +1)
        """
        try:
            # Run strategy on historical data
            results, _ = self.strategy.run_backtest(self.historical_data)
            
            # Get latest position signal
            latest_position = results['position'].iloc[-1]
            
            # Get regime information for logging
            if 'regime' in results.columns:
                latest_regime = results['regime'].iloc[-1]
                logger.info(f"Current market regime: {latest_regime}")
            
            return latest_position
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return 0.0
    
    def _execute_trade(self, target_position: float) -> bool:
        """
        Execute trade to reach the target position
        
        Args:
            target_position: Target position size (-1 to +1)
            
        Returns:
            bool: True if trade was executed successfully
        """
        # Calculate position difference
        position_diff = target_position - self.current_position
        
        # Skip if position change is minimal
        if abs(position_diff) < 0.01:
            logger.debug(f"Position change too small ({position_diff:.4f}), skipping trade")
            return True
        
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calculate amount to trade
            base_asset, quote_asset = self.symbol[:-4], self.symbol[-4:]
            
            if position_diff > 0:
                # Buy
                amount = position_diff * self.cash / current_price
                order_type = "BUY"
                logger.info(f"Buying {amount:.6f} {base_asset} at {current_price} {quote_asset}")
                
                # Place test order
                order = self.client.create_test_order(
                    symbol=self.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=amount
                )
                
                # Update state
                trade_value = amount * current_price
                self.cash -= trade_value
                self.holdings += amount
                
            else:
                # Sell
                amount = abs(position_diff) * self.holdings
                order_type = "SELL"
                logger.info(f"Selling {amount:.6f} {base_asset} at {current_price} {quote_asset}")
                
                # Place test order
                order = self.client.create_test_order(
                    symbol=self.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=amount
                )
                
                # Update state
                trade_value = amount * current_price
                self.cash += trade_value
                self.holdings -= amount
            
            # Record the trade
            self.trades.append({
                'timestamp': dt.datetime.now(),
                'type': order_type,
                'amount': amount,
                'price': current_price,
                'value': amount * current_price
            })
            
            # Update current position and equity
            self.current_position = target_position
            self.equity = self.cash + (self.holdings * current_price)
            
            logger.info(f"Trade executed. New position: {self.current_position:.2f}, Equity: {self.equity:.2f} {quote_asset}")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def run(self, check_interval_seconds: int = 60) -> None:
        """
        Run paper trading loop
        
        Args:
            check_interval_seconds: How often to check for new data/signals
        """
        logger.info(f"Starting paper trading for {self.symbol} with {self.initial_capital} USDT")
        logger.info(f"Strategy: DMT_v2 {self.strategy_version}, Timeframe: {self.interval}")
        
        try:
            while True:
                # Update market data
                data_updated = self._update_market_data()
                
                if data_updated:
                    # Calculate new signal
                    signal = self._calculate_signal()
                    logger.info(f"New signal: {signal:.4f}")
                    
                    # Execute trade if needed
                    self._execute_trade(signal)
                    
                # Sleep until next check
                logger.debug(f"Sleeping for {check_interval_seconds} seconds")
                time.sleep(check_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Paper trading stopped by user")
            self._print_summary()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of trading performance"""
        if not self.trades:
            logger.info("No trades executed")
            return
        
        # Get current price for final equity calculation
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Calculate final equity
            final_equity = self.cash + (self.holdings * current_price)
            
            # Calculate performance metrics
            total_return = final_equity / self.initial_capital - 1
            num_trades = len(self.trades)
            
            # Print summary
            logger.info("\n" + "=" * 50)
            logger.info("PAPER TRADING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Symbol:             {self.symbol}")
            logger.info(f"Strategy:           DMT_v2 {self.strategy_version}")
            logger.info(f"Timeframe:          {self.interval}")
            logger.info(f"Initial capital:    {self.initial_capital:.2f} USDT")
            logger.info(f"Final equity:       {final_equity:.2f} USDT")
            logger.info(f"Total return:       {total_return:.2%}")
            logger.info(f"Number of trades:   {num_trades}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


def main():
    """Main entry point for paper trading script"""
    parser = argparse.ArgumentParser(
        description="DMT_v2 Paper Trading on Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to trade')
    parser.add_argument('--interval', default='1d', 
                      choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                      help='Trading interval')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--version', default='turbo', 
                      choices=['original', 'enhanced', 'turbo'],
                      help='Strategy version')
    
    args = parser.parse_args()
    
    # Get API keys
    api_key = os.environ.get('BINANCE_API_KEY_TEST')
    api_secret = os.environ.get('BINANCE_API_SECRET_TEST')
    
    if not api_key or not api_secret:
        print("Error: Binance API keys not found in environment variables")
        print("Please set BINANCE_API_KEY_TEST and BINANCE_API_SECRET_TEST")
        print("\nYou can get these keys from:")
        print("https://testnet.binance.vision/")
        sys.exit(1)
    
    # Create and run paper trader
    trader = BinancePaperTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbol=args.symbol,
        interval=args.interval,
        initial_capital=args.capital,
        strategy_version=args.version
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

if __name__ == '__main__':
    main()
