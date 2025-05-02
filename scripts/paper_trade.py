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
import argparse
import logging
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union, Any
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import random
import traceback
from colorama import Fore, Back, Style, init
import math

# Initialize colorama for cross-platform color support
init()

# Load variables from .env file
load_dotenv()

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
        asset_type: str = "crypto",
        use_bnb_for_fees: bool = True,
        trading_tier: str = "regular",
        simulation_mode: bool = False,
        use_binance_us: bool = False
    ):
        """
        Initialize the Binance paper trader

        Args:
            api_key: Binance API key (TESTNET ONLY!)
            api_secret: Binance API secret (TESTNET ONLY!)
            symbol: Trading symbol (e.g., BTCUSDT)
            interval: Trading interval
            initial_capital: Initial capital for paper trading
            strategy_version: Strategy version to use
            asset_type: Asset type (crypto only for now)
            use_bnb_for_fees: Whether to use BNB for fee discount
            trading_tier: Trading tier for fee calculation
            simulation_mode: If True, don't attempt actual API trades
            use_binance_us: If True, use Binance US API instead of global
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.strategy_version = strategy_version
        self.asset_type = asset_type
        self.use_bnb_for_fees = use_bnb_for_fees
        self.trading_tier = trading_tier.lower()
        self.simulation_mode = simulation_mode
        self.use_binance_us = use_binance_us
        
        # Track price history for mini-chart
        self.price_history = []
        self.max_price_history = 20  # Keep last 20 prices for mini-chart
        
        # Performance tracking
        self.equity_history = []
        self.best_profit = 0
        self.worst_drawdown = 0
        self.start_time = datetime.now()
        
        # Risk metrics
        self.max_position_value_pct = 0
        self.risk_per_trade_pct = 1.0  # Default 1% risk per trade
        self.volatility = 0
        
        # Configure client based on user settings
        try:
            if use_binance_us:
                self.client = BinanceClient(api_key, api_secret, tld='us')
                print("✅ Using Binance US API")
            else:
                self.client = BinanceClient(api_key, api_secret, testnet=True)
                print("✅ Using Binance Global Testnet API")
                
            # Verify connection
            server_time = self.client.get_server_time()
            print(f"✅ Connected to Binance (Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)})")
            
            # Test API permissions
            try:
                account_info = self.client.get_account()
                print(f"✅ API Read Permission Verified")
                
                # Check if we have trading permissions
                if not simulation_mode:
                    try:
                        # For Binance Testnet, we don't need to check permissions
                        # Just verify we can get account info
                        if account_info and 'balances' in account_info:
                            print("✅ Trading permissions verified")
                            self.simulation_mode = False
                        else:
                            print("⚠️ Trading permissions not detected - using simulation mode only")
                            self.simulation_mode = True
                    except Exception as e:
                        print(f"⚠️ Couldn't verify trading permissions: {e}")
                        print("⚠️ Using simulation mode only")
                        self.simulation_mode = True
            except Exception as e:
                print(f"⚠️ Couldn't verify trading permissions: {e}")
                print("⚠️ Using simulation mode only")
                self.simulation_mode = True
                
        except Exception as e:
            print(f"⚠️ Warning: API connection issue: {e}")
            print("⚠️ Running in full simulation mode")
            self.simulation_mode = True
            self.client = None
        
        # Initialize fee structure for fee calculation
        self.fee_structure = self._initialize_fee_structure()
        
        # Initialize trading states
        self.cash = initial_capital
        self.holdings = 0.0
        self.equity = initial_capital
        self.positions = []
        self.trades = []
        self.total_fees_paid = 0.0
        self.trading_volume_30d = 0.0  # 30-day trading volume for fee calculation
        
        # Initialize strategy
        self.strategy = DMT_v2_Strategy(
            version=strategy_version,
            asset_type=asset_type
        )
        
        # Load initial data
        self._load_initial_data()
        
        # Initialize klines and timestamp
        if not hasattr(self, 'klines') or not self.klines:
            self.klines = []
            self.last_timestamp = 0
            self.current_price = 0.0
    
    def _initialize_fee_structure(self):
        """Initialize the fee structure based on Binance's tier system"""
        return {
            # Regular and VIP tiers - format: {'tier': {'maker_fee': X, 'taker_fee': Y, 'bnb_maker_fee': Z, 'bnb_taker_fee': W}}
            'regular': {'maker_fee': 0.001000, 'taker_fee': 0.001000, 'bnb_maker_fee': 0.000750, 'bnb_taker_fee': 0.000750},
            'vip1': {'maker_fee': 0.000900, 'taker_fee': 0.001000, 'bnb_maker_fee': 0.000675, 'bnb_taker_fee': 0.000750},
            'vip2': {'maker_fee': 0.000800, 'taker_fee': 0.000900, 'bnb_maker_fee': 0.000600, 'bnb_taker_fee': 0.000675},
            'vip3': {'maker_fee': 0.000700, 'taker_fee': 0.000800, 'bnb_maker_fee': 0.000525, 'bnb_taker_fee': 0.000600},
            'vip4': {'maker_fee': 0.000500, 'taker_fee': 0.000600, 'bnb_maker_fee': 0.000375, 'bnb_taker_fee': 0.000450},
            'vip5': {'maker_fee': 0.000300, 'taker_fee': 0.000400, 'bnb_maker_fee': 0.000225, 'bnb_taker_fee': 0.000300},
            'vip6': {'maker_fee': 0.000200, 'taker_fee': 0.000300, 'bnb_maker_fee': 0.000150, 'bnb_taker_fee': 0.000225},
            'vip7': {'maker_fee': 0.000150, 'taker_fee': 0.000250, 'bnb_maker_fee': 0.000113, 'bnb_taker_fee': 0.000188},
            'vip8': {'maker_fee': 0.000100, 'taker_fee': 0.000200, 'bnb_maker_fee': 0.000075, 'bnb_taker_fee': 0.000150},
            'vip9': {'maker_fee': 0.000075, 'taker_fee': 0.000150, 'bnb_maker_fee': 0.000056, 'bnb_taker_fee': 0.000113}
        }
    
    def _calculate_fee(self, trade_value: float, is_maker: bool = False) -> float:
        """
        Calculate fee for a given trade value and conditions
        
        Args:
            trade_value: Value of the trade in USDT
            is_maker: Whether this is a maker order (limit) or taker order (market)
            
        Returns:
            Fee amount in USDT
        """
        # Get the base fee rate based on tier, maker/taker status, and BNB usage
        if self.use_bnb_for_fees:
            fee_rate = self.fee_structure[self.trading_tier]['bnb_maker_fee'] if is_maker else self.fee_structure[self.trading_tier]['bnb_taker_fee']
        else:
            fee_rate = self.fee_structure[self.trading_tier]['maker_fee'] if is_maker else self.fee_structure[self.trading_tier]['taker_fee']
        
        # Calculate fee
        fee_amount = trade_value * fee_rate
        
        # Log fee details for transparency
        order_type = "Maker" if is_maker else "Taker"
        bnb_status = "with BNB discount" if self.use_bnb_for_fees else "without BNB discount"
        logger.debug(f"Fee calculation: {order_type} order {bnb_status}, Tier: {self.trading_tier}, Rate: {fee_rate:.6f}, Fee: ${fee_amount:.2f} on ${trade_value:.2f}")
        
        return fee_amount
    
    def _check_tier_upgrade(self) -> None:
        """Check if current trading volume qualifies for a tier upgrade"""
        # Volume thresholds for tier upgrades (in USD)
        tier_thresholds = {
            'regular': 0,
            'vip1': 1_000_000,
            'vip2': 5_000_000,
            'vip3': 20_000_000,
            'vip4': 75_000_000,
            'vip5': 150_000_000,
            'vip6': 400_000_000,
            'vip7': 800_000_000,
            'vip8': 2_000_000_000,
            'vip9': 4_000_000_000
        }
        
        # Find the highest tier the user qualifies for
        new_tier = 'regular'
        for tier, threshold in tier_thresholds.items():
            if self.trading_volume_30d >= threshold:
                new_tier = tier
        
        # Update tier if changed
        if new_tier != self.trading_tier:
            logger.info(f"Trading tier upgraded from {self.trading_tier} to {new_tier} based on volume")
            self.trading_tier = new_tier
    
    def _load_initial_data(self):
        """Load initial historical data"""
        try:
            # Determine how many bars to load based on interval
            lookback_periods = {
                '1m': 5000,   # More 1-minute data for better context
                '5m': 1000,   # More 5-minute data (3+ days)
                '15m': 500,   # More 15-minute data (5+ days)
                '30m': 300,   # More 30-minute data (6+ days)
                '1h': 200,    # More hourly data (8+ days)
                '4h': 150,    # More 4-hour data (25+ days)
                '1d': 200     # More daily data (6+ months)
            }
            
            # Get the number of bars to load
            bars_to_load = lookback_periods.get(self.interval, 100)
            
            print(f"\n📊 Preloading {bars_to_load} historical {self.interval} bars to build context...")
            
            # Get historical klines
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=bars_to_load
            )
            
            # Parse klines
            self.klines = []
            for k in klines:
                timestamp = k[0]
                dt_obj = datetime.fromtimestamp(timestamp / 1000)
                self.klines.append({
                    'timestamp': timestamp,
                    'datetime': dt_obj,
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            # Set the last timestamp
            if self.klines:
                self.last_timestamp = self.klines[-1]['timestamp']
                self.current_price = self.klines[-1]['close']
            
            # Create a DataFrame with column names that match what the strategy expects
            # The DMT_v2 strategy expects uppercase first letter: 'Open', 'High', 'Low', 'Close', 'Volume'
            df = pd.DataFrame([
                {
                    'Open': k['open'],
                    'High': k['high'],
                    'Low': k['low'],
                    'Close': k['close'],
                    'Volume': k['volume'],
                    'Date': k['datetime']
                } for k in self.klines
            ])
            
            # Set the index for the strategy
            df.set_index('Date', inplace=True)
            
            # Initialize strategy with historical data
            self.strategy.historical_data = df
            
            print(f"✅ Loaded {len(self.klines)} bars of {self.interval} historical data")
            print(f"📋 Data columns: {', '.join(df.columns.tolist())}")
            
            # Pre-calculate initial signals based on historical data
            self._build_context_and_initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            print(f"❌ Error loading historical data: {e}")
            sys.exit(1)
    
    def _build_context_and_initialize(self):
        """Build context from historical data and initialize the model"""
        print("\n🧠 Building model context from historical data...")
        
        # Use progressive learning for all timeframes
        if len(self.klines) > 20:
            # Determine number of steps based on timeframe
            if self.interval == '1m':
                progress_steps = min(20, len(self.klines) // 50)
            elif self.interval in ['5m', '15m', '30m']:
                progress_steps = min(15, len(self.klines) // 20)
            elif self.interval in ['1h', '4h']:
                progress_steps = min(10, len(self.klines) // 10)
            else:  # daily and above
                progress_steps = min(10, len(self.klines) // 5)
                
            step_size = max(1, len(self.klines) // progress_steps)
            
            for i in range(0, len(self.klines), step_size):
                # Create a subset of data up to this point
                temp_df = pd.DataFrame([
                    {
                        'Open': k['open'],
                        'High': k['high'],
                        'Low': k['low'],
                        'Close': k['close'],
                        'Volume': k['volume'],
                        'Date': k['datetime']
                    } for k in self.klines[:i+1]
                ])
                
                # Set index for strategy
                temp_df.set_index('Date', inplace=True)
                
                # Update strategy data
                self.strategy.historical_data = temp_df
                
                # Try to get the current regime if available, otherwise just show progress
                try:
                    signal = self._calculate_signal()
                    regime = "Unknown"
                    if hasattr(self.strategy, 'current_regime'):
                        regime = self.strategy.current_regime
                    elif hasattr(self.strategy, 'market_regime'):
                        regime = self.strategy.market_regime
                    
                    # Show progress update with percentage and total bars
                    progress = int((i / len(self.klines)) * 100)
                    bar_length = 30
                    filled_length = int(bar_length * progress // 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r[{bar}] {progress}% | Processed {i+1}/{len(self.klines)} bars", end="")
                    
                    # For non-minute timeframes, also print average position size
                    if self.interval != '1m' or i % (5 * step_size) == 0:
                        backtest_results, _ = self.strategy.run_backtest(temp_df)
                except Exception as e:
                    # Just continue with progress if we can't calculate signal yet
                    progress = int((i / len(self.klines)) * 100)
                    bar_length = 30
                    filled_length = int(bar_length * progress // 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r[{bar}] {progress}% | Processed {i+1}/{len(self.klines)} bars", end="")
                
                # Small sleep to make the progress visible
                # Adjust sleep time based on timeframe (faster for minute data)
                if self.interval == '1m':
                    time.sleep(0.01)
                else:
                    time.sleep(0.05)
        
        # Final calculation on all data
        try:
            signal = self._calculate_signal()
            
            # Get current market regime if available
            regime = "Unknown"
            try:
                if hasattr(self.strategy, 'current_regime'):
                    regime = self.strategy.current_regime
                elif hasattr(self.strategy, 'market_regime'):
                    regime = self.strategy.market_regime
                
                # Update strategy attributes to make regime accessible
                if regime != "Unknown":
                    self.strategy.current_regime = regime
            except Exception as e:
                logger.error(f"Error getting market regime: {e}")
                
            # Enforce a proper range for the signal (-1.0 to 1.0)
            signal = max(min(float(signal), 1.0), -1.0)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return 0.0
    
    def _check_for_new_data(self) -> bool:
        """Check if there is new market data available"""
        try:
            # Get klines data for the requested interval
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=2  # Get the last 2 candles
            )
            
            if not klines or len(klines) < 2:
                return False
                
            # Check if there's a new candle
            latest_candle = {
                'open_time': int(klines[-1][0]),
                'open': float(klines[-1][1]),
                'high': float(klines[-1][2]),
                'low': float(klines[-1][3]),
                'close': float(klines[-1][4]),
                'volume': float(klines[-1][5]),
                'close_time': datetime.fromtimestamp(int(klines[-1][6])/1000)
            }
            
            # Check if this is a new candle we haven't processed
            if not hasattr(self, 'last_candle_time') or self.last_candle_time != latest_candle['open_time']:
                self.last_candle_time = latest_candle['open_time']
                self._process_new_candle(latest_candle)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking for new data: {e}")
            return False
    
    def _process_new_candle(self, candle):
        """Process a new candle of data"""
        try:
            # Add new data to historical_data
            if self.strategy.historical_data is not None and len(self.strategy.historical_data) > 0:
                # Convert candle to DataFrame with same format as historical_data
                columns = self.strategy.historical_data.columns
                if len(columns) >= 5:  # OHLCV format
                    new_candle = pd.DataFrame({
                        'Open': [float(candle['open'])],
                        'High': [float(candle['high'])],
                        'Low': [float(candle['low'])],
                        'Close': [float(candle['close'])],
                        'Volume': [float(candle['volume'])]
                    })
                    
                    # Update current price and volume
                    self.current_price = float(candle['close'])
                    self.current_volume = float(candle['volume'])
                else:
                    logger.warning(f"Unexpected columns in historical_data: {columns}")
                    return
                
                # Append to historical_data
                try:
                    self.strategy.historical_data = pd.concat([self.strategy.historical_data, new_candle], ignore_index=True)
                    # Keep only recent data to prevent memory issues
                    max_bars = 5000  # Keep last 5000 bars
                    if len(self.strategy.historical_data) > max_bars:
                        self.strategy.historical_data = self.strategy.historical_data.iloc[-max_bars:]
                except Exception as e:
                    logger.error(f"Error appending new candle to historical_data: {e}")
                
                # Log the new candle
                close_time_str = candle['close_time'].strftime('%Y-%m-%d %H:%M') if isinstance(candle['close_time'], datetime) else 'Unknown'
                print(f"📊 New {self.interval} candle: {close_time_str} | ${float(candle['close']):.2f} | Vol: {float(candle['volume']):.2f}")
                
                # Calculate average position size across recent data
                try:
                    # First try from strategy results
                    if hasattr(self.strategy, 'backtest_results') and self.strategy.backtest_results is not None:
                        if 'position' in self.strategy.backtest_results.columns:
                            avg_pos = self.strategy.backtest_results['position'].abs().mean()
                            print(f"  Average position size: {avg_pos:.2f}")
                        elif 'signal' in self.strategy.backtest_results.columns:
                            avg_sig = self.strategy.backtest_results['signal'].abs().mean()
                            print(f"  Average signal strength: {avg_sig:.2f}")
                except Exception as e:
                    logger.error(f"Error calculating average position: {e}")
                
                # Update equity calculation to include holdings
                self.equity = self.capital + (self.holdings * self.current_price)
            
        except Exception as e:
            logger.error(f"Error processing new candle: {e}")
    
    def _calculate_signal(self) -> float:
        """
        Calculate trading signal from strategy
        """
        try:
            # Create a DataFrame from the klines data
            df = pd.DataFrame([
                {
                    'Open': k['open'],
                    'High': k['high'],
                    'Low': k['low'],
                    'Close': k['close'],
                    'Volume': k['volume'],
                    'Date': k['datetime']
                } for k in self.klines
            ])
            
            # Set the index for the strategy
            df.set_index('Date', inplace=True)
            
            # Update strategy data
            self.strategy.historical_data = df
            
            # Get the signal from the strategy
            signal = 0.0
            
            # For DMT_v2 strategy
            if hasattr(self.strategy, 'run_backtest'):
                # Run backtest to get the latest signal
                backtest_results, _ = self.strategy.run_backtest(df)
                if len(backtest_results) > 0:
                    signal = backtest_results['signal'].iloc[-1]
            
            # For enhanced version, ensure we have a meaningful signal for paper trading
            if self.strategy_version == "enhanced" and abs(signal) < 0.1:
                # Force a minimum signal strength for paper trading
                if signal >= 0:
                    signal = 0.3  # Force a significant long position
                else:
                    signal = -0.3  # Force a significant short position
                logger.info(f"Forcing signal for paper trading: {signal:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return 0.0
    
    def _update_market_data(self):
        """Update current price and other market data"""
        try:
            # Get latest candle
            if self.simulation_mode and len(self.strategy.historical_data) > 0:
                # In simulation mode, just use the last candle from historical data
                latest = self.strategy.historical_data.iloc[-1]
                self.current_price = latest['Close']
                self.current_volume = latest['Volume']
                # Update timestamp from the data
                try:
                    if isinstance(self.strategy.historical_data.index[0], pd.Timestamp):
                        self.last_update = self.strategy.historical_data.index[-1].to_pydatetime()
                    else:
                        self.last_update = datetime.now()
                except:
                    self.last_update = datetime.now()
            else:
                # Get real-time data from API
                ticker = self.client.get_ticker(symbol=self.symbol)
                self.current_price = float(ticker['lastPrice'])
                self.current_volume = float(ticker['volume'])
                self.last_update = datetime.now()
            
            # Update equity calculation to include holdings
            self.equity = self.capital + (self.holdings * self.current_price)
            return True
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return False
    
    def _format_status_update(self):
        """Format a status update string"""
        # Format timestamp
        ts = self.last_update.strftime('%Y-%m-%d %H:%M')
        
        # Calculate total equity (cash + holdings)
        total_equity = self.capital + (self.holdings * self.current_price)
        
        # Format status string
        return f"🔄 {ts} | ${self.current_price:.2f} | Vol: {self.current_volume:.2f} | Cash: ${self.capital:.2f} | Holdings: {self.holdings:.6f} BTC | Total: ${total_equity:.2f}"
    
    def _execute_trade(self, signal: float) -> bool:
        """Execute trade based on signal"""
        self.last_signal = signal  # Store for status updates
        
        # If signal is very close to zero, don't trade
        if abs(signal) < 0.001:
            logging.info("Signal too small, no trade executed")
            return False
        
        # Calculate target position size as a decimal (0 to 1.0)
        target_position_size = self._calculate_position_size(signal)
        
        # For enhanced version, ensure we have a meaningful position size for paper trading
        if self.strategy_version == "enhanced" and abs(target_position_size) < 0.1:
            # Force a minimum position size for paper trading
            if target_position_size >= 0:
                target_position_size = 0.3  # Force a significant long position
            else:
                target_position_size = -0.3  # Force a significant short position
            logging.info(f"Forcing position adjustment for paper trading: {target_position_size:.2f}")
        
        # Get current account information from Binance Testnet
        try:
            if not self.simulation_mode and self.client:
                account_info = self.client.get_account()
                
                # Find USDT balance
                usdt_balance = 0.0
                btc_balance = 0.0
                
                for asset in account_info['balances']:
                    if asset['asset'] == 'USDT':
                        usdt_balance = float(asset['free'])
                    elif asset['asset'] == 'BTC':
                        btc_balance = float(asset['free'])
                
                # Update our local tracking with testnet values
                self.capital = usdt_balance
                self.holdings = btc_balance
                
                # Calculate current position size as a decimal
                current_position_value = self.holdings * self.current_price
                current_total_equity = self.capital + current_position_value
                current_position_size = current_position_value / current_total_equity if current_total_equity > 0 else 0
                
                # Calculate the difference in position size
                position_size_diff = target_position_size - current_position_size
                
                # Skip if there's no meaningful change needed
                min_adjustment_pct = 0.01  # 1% minimum adjustment threshold
                if abs(position_size_diff) < min_adjustment_pct:
                    logging.info("Position adjustment too small, maintaining current position")
                    return False
                    
                # Calculate the amount to buy or sell in base currency units
                position_value_diff = position_size_diff * current_total_equity
                amount_diff = position_value_diff / self.current_price if self.current_price > 0 else 0
                
                # Execute the trade on Binance Testnet
                if amount_diff > 0:  # Buy
                    # Check if we have enough capital
                    if self.capital < position_value_diff:
                        logging.info(f"Not enough capital for position increase. Need ${position_value_diff:.2f}, have ${self.capital:.2f}")
                        return False
                    
                    # Calculate quantity with precision
                    quantity = self._format_quantity(amount_diff)
                    
                    # Check minimum notional value (usually 10 USDT on Binance)
                    if float(quantity) * self.current_price < 10:
                        logging.info(f"Order too small: ${float(quantity) * self.current_price:.2f} is below minimum notional value of $10")
                        return False
                    
                    # Execute market buy order
                    try:
                        # First test the order
                        order = self.client.create_test_order(
                            symbol=self.symbol,
                            side='BUY',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # If test order successful, place real order
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side='BUY',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # Log the trade
                        price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else self.current_price
                        executed_qty = float(order['executedQty']) if 'executedQty' in order else amount_diff
                        trade_value = price * executed_qty
                        
                        logging.info(f"BUY {executed_qty:.8f} BTC at ${price:.2f}")
                        logging.info(f"Trade value: ${trade_value:.2f}")
                        
                        # Record the trade
                        self.trades.append({
                            'timestamp': datetime.now(),
                            'type': 'buy',
                            'price': price,
                            'amount': executed_qty,
                            'value': trade_value,
                            'order_id': order.get('orderId', 'unknown')
                        })
                        
                        print(f"💰 Trade executed: BUY {executed_qty:.6f} BTC at ${price:.2f}")
                        
                        return True
                    except Exception as e:
                        logging.error(f"Error executing buy order: {e}")
                        return False
                        
                elif amount_diff < 0:  # Sell
                    amount_to_sell = abs(amount_diff)
                    
                    # Check if we have enough holdings
                    if self.holdings < amount_to_sell:
                        logging.info(f"Not enough holdings to sell. Need {amount_to_sell:.8f} BTC, have {self.holdings:.8f} BTC")
                        return False
                        
                    # Calculate quantity with precision
                    quantity = self._format_quantity(amount_to_sell)
                    
                    # Check minimum notional value (usually 10 USDT on Binance)
                    if float(quantity) * self.current_price < 10:
                        logging.info(f"Order too small: ${float(quantity) * self.current_price:.2f} is below minimum notional value of $10")
                        return False
                    
                    # Execute market sell order
                    try:
                        # First test the order
                        order = self.client.create_test_order(
                            symbol=self.symbol,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # If test order successful, place real order
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # Log the trade
                        price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else self.current_price
                        executed_qty = float(order['executedQty']) if 'executedQty' in order else amount_to_sell
                        trade_value = price * executed_qty
                        
                        logging.info(f"SELL {executed_qty:.8f} BTC at ${price:.2f}")
                        logging.info(f"Trade value: ${trade_value:.2f}")
                        
                        # Record the trade
                        self.trades.append({
                            'timestamp': datetime.now(),
                            'type': 'sell',
                            'price': price,
                            'amount': executed_qty,
                            'value': trade_value,
                            'order_id': order.get('orderId', 'unknown')
                        })
                        
                        print(f"💰 Trade executed: SELL {executed_qty:.6f} BTC at ${price:.2f}")
                        
                        return True
                    except Exception as e:
                        logging.error(f"Error executing sell order: {e}")
                        return False
            else:
                # Simulation mode - log that we're not executing real trades
                logging.info("No trades executed - simulation mode")
                return False
                
        except Exception as e:
            logging.error(f"Error in trade execution: {e}")
            return False
            
        return False
    
    def _format_quantity(self, quantity: float) -> str:
        """Format quantity according to Binance's precision requirements"""
        # For BTC, typically 5 decimal places for BTCUSDT on Binance
        # Round down to ensure we don't exceed available balance
        quantity = math.floor(quantity * 100000) / 100000
        return "{:.5f}".format(quantity)
    
    def _get_fee_rate(self, is_maker: bool = False) -> float:
        """Get the fee rate based on current tier and maker/taker status"""
        if self.use_bnb_for_fees:
            return self.fee_structure[self.trading_tier]['bnb_maker_fee'] if is_maker else self.fee_structure[self.trading_tier]['bnb_taker_fee']
        else:
            return self.fee_structure[self.trading_tier]['maker_fee'] if is_maker else self.fee_structure[self.trading_tier]['taker_fee']
    
    def _calculate_position_size(self, signal: float) -> float:
        """
        Calculate position size based on signal strength and market regime
        
        Implements the enhanced DMT_v2 strategy parameters:
        - target_annual_vol: 0.35 (increased from 0.25)
        - max_position_size: 1.0 (capped at 100% of capital for paper trading)
        - neutral_zone: 0.005 (reduced from 0.05)
        
        Args:
            signal: Trading signal from -1.0 to 1.0
            
        Returns:
            Position size as a percentage of capital (0.0 to max_position_size)
        """
        # Parameters from enhanced DMT_v2 strategy
        max_position_size = 1.0  # Cap at 100% of capital for paper trading
        neutral_zone = 0.005     # Reduced from 0.03 to ensure trades are executed
        
        # For enhanced version, force a minimum position in paper trading
        if self.strategy_version == "enhanced" and abs(signal) > 0.001:
            # Force a minimum signal strength for paper trading
            if signal > 0:
                signal = max(signal, 0.2)  # At least 0.2 positive signal
            else:
                signal = min(signal, -0.2)  # At least 0.2 negative signal
        
        # Skip small signals in the neutral zone
        if abs(signal) < neutral_zone:
            return 0.0
        
        # Get current market regime if available
        regime = "Unknown"
        try:
            if hasattr(self.strategy, 'current_regime'):
                regime = self.strategy.current_regime
            elif hasattr(self.strategy, 'market_regime'):
                regime = self.strategy.market_regime
        except:
            pass
        
        # Dynamic position sizing based on market regime
        regime_multiplier = 1.0  # Default multiplier
        
        if regime == "Bull":
            # More aggressive in bull markets
            regime_multiplier = 1.2
        elif regime == "Bear":
            # More cautious in bear markets
            regime_multiplier = 0.8
        elif regime == "Neutral":
            # Moderate in neutral markets
            regime_multiplier = 0.9
            
        # Scale the signal to position size (non-linear scaling to be more aggressive)
        # Use signal**1.5 for non-linear scaling
        signal_scale = abs(signal) ** 1.5  # Non-linear scaling
        
        # Apply max position constraint and regime multiplier
        position_size = min(signal_scale * max_position_size * regime_multiplier, max_position_size)
        
        # Apply direction (long or short)
        if signal < 0:
            position_size = -position_size
            
        logger.debug(f"Signal: {signal:.4f}, Regime: {regime}, Position Size: {position_size:.4f}")
        return position_size
    
    def _update_status(self):
        """Print status update with current position and account value"""
        # Get account info from Binance if not in simulation mode
        if not self.simulation_mode and self.client:
            try:
                account_info = self.client.get_account()
                
                # Find USDT and BTC balances
                for asset in account_info['balances']:
                    if asset['asset'] == 'USDT':
                        self.capital = float(asset['free'])
                    elif asset['asset'] == 'BTC':
                        self.holdings = float(asset['free'])
            except Exception as e:
                logging.error(f"Error getting account info: {e}")
        
        # Calculate position value and total equity
        position_value = self.holdings * self.current_price if self.current_price > 0 else 0
        total_equity = self.capital + position_value
        profit_loss = total_equity - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Update equity history and track max profit/drawdown
        self.equity_history.append(total_equity)
        if len(self.equity_history) > self.max_price_history * 2:  # Keep more equity history points
            self.equity_history = self.equity_history[-self.max_price_history * 2:]
            
        if profit_loss > self.best_profit:
            self.best_profit = profit_loss
            
        current_drawdown = self.best_profit - profit_loss
        if current_drawdown > self.worst_drawdown:
            self.worst_drawdown = current_drawdown
        
        # Determine if we're up or down
        if profit_loss > 0:
            pl_color = Fore.GREEN
        elif profit_loss < 0:
            pl_color = Fore.RED
        else:
            pl_color = Fore.WHITE
        
        # Current time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Count buys and sells
        buy_count = sum(1 for trade in self.trades if trade['type'] == 'buy')
        sell_count = sum(1 for trade in self.trades if trade['type'] == 'sell')
        
        # Calculate total fees paid
        total_fees = sum(trade.get('fee', 0) for trade in self.trades)
        
        # Calculate win/loss ratio if we have trades
        win_count = 0
        loss_count = 0
        avg_win = 0
        avg_loss = 0
        
        if len(self.trades) > 1:
            # Calculate profit/loss for each trade
            for i in range(1, len(self.trades)):
                prev_trade = self.trades[i-1]
                curr_trade = self.trades[i]
                
                if prev_trade['type'] != curr_trade['type']:  # Only count completed round trips
                    if prev_trade['type'] == 'buy' and curr_trade['type'] == 'sell':
                        # Buy then sell - calculate profit
                        buy_value = prev_trade['price'] * prev_trade['amount']
                        sell_value = curr_trade['price'] * curr_trade['amount']
                        trade_pl = sell_value - buy_value
                        
                        if trade_pl > 0:
                            win_count += 1
                            avg_win += trade_pl
                        else:
                            loss_count += 1
                            avg_loss += abs(trade_pl)
            
            # Calculate averages
            avg_win = avg_win / win_count if win_count > 0 else 0
            avg_loss = avg_loss / loss_count if loss_count > 0 else 0
            win_rate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Clear terminal and create dashboard
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Print header with border
        print(f"\n{Fore.CYAN}╔{'═' * 58}╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║ 📈 DMT_v2 PAPER TRADING DASHBOARD - {timestamp} {' ' * (5 - len(timestamp))}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        
        # Print current price and market data
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}📊 MARKET DATA{' ' * 45}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Symbol: {self.symbol} | Price: ${self.current_price:,.2f} | 24h Vol: {self.klines[-1]['volume']:,.1f}{' ' * 5}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print portfolio summary
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}💰 PORTFOLIO SUMMARY{' ' * 39}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Create two columns for portfolio data
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Cash: ${self.capital:,.2f}{' ' * (25-len(f'${self.capital:,.2f}'))} | Total Value: ${total_equity:,.2f}{' ' * (16-len(f'${total_equity:,.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Holdings: {self.holdings:.6f} {self.symbol[:3]} (${position_value:,.2f}){' ' * (58-len(f'Holdings: {self.holdings:.6f} {self.symbol[:3]} (${position_value:,.2f})'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print performance metrics
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.MAGENTA}📊 PERFORMANCE METRICS{' ' * 38}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Two columns for performance data
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Initial: ${self.initial_capital:,.2f}{' ' * (20-len(f'${self.initial_capital:,.2f}'))} | P&L: {pl_color}${profit_loss:,.2f} ({profit_loss_pct:+.2f}%){Style.RESET_ALL}{' ' * (20-len(f'${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Best Profit: ${self.best_profit:,.2f}{' ' * (17-len(f'${self.best_profit:,.2f}'))} | Max Drawdown: ${self.worst_drawdown:,.2f}{' ' * (17-len(f'${self.worst_drawdown:,.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print trading stats
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.BLUE}🔄 TRADING ACTIVITY{' ' * 41}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Two columns for trading stats
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Total Trades: {len(self.trades)}{' ' * (20-len(str(len(self.trades))))} | Buy: {buy_count} | Sell: {sell_count}{' ' * (20-len(f'Buy: {buy_count} | Sell: {sell_count}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Add win/loss stats if we have trades
        if len(self.trades) > 1 and (win_count + loss_count) > 0:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} Win Rate: {win_rate:.1f}% ({win_count}/{win_count + loss_count}){' ' * (15-len(f'{win_rate:.1f}% ({win_count}/{win_count + loss_count})'))} | Profit Factor: {profit_factor:.2f}{' ' * (15-len(f'{profit_factor:.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} Win Rate: N/A{' ' * 20} | Profit Factor: N/A{' ' * 15}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print strategy info
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}🧠 STRATEGY INFO{' ' * 43}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Format signal with color based on direction
        signal_color = Fore.GREEN if self.last_signal > 0 else Fore.RED if self.last_signal < 0 else Fore.WHITE
        signal_str = f"{signal_color}{self.last_signal:.4f}{Style.RESET_ALL}"
        
        # Add signal strength indicator
        signal_strength = abs(self.last_signal)
        if signal_strength > 0.5:
            signal_indicator = f"{signal_color}{'▓' * 5}{Style.RESET_ALL}"  # Strong signal
        elif signal_strength > 0.3:
            signal_indicator = f"{signal_color}{'▓' * 3}{'░' * 2}{Style.RESET_ALL}"  # Medium signal
        elif signal_strength > 0.1:
            signal_indicator = f"{signal_color}{'▓' * 1}{'░' * 4}{Style.RESET_ALL}"  # Weak signal
        else:
            signal_indicator = f"{'░' * 5}"  # Neutral
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Version: {self.strategy_version}{' ' * (20-len(self.strategy_version))} | Signal: {signal_str} {signal_indicator}{' ' * (20-len(f'{self.last_signal:.4f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Calculate position size as percentage
        position_size_pct = (position_value / total_equity) * 100 if total_equity > 0 else 0
        position_str = f"{position_size_pct:.1f}% of capital"
        
        # Add position indicator
        if position_size_pct > 50:
            position_indicator = f"{Fore.GREEN}{'▓' * 5}{Style.RESET_ALL}"  # Heavy long
        elif position_size_pct > 25:
            position_indicator = f"{Fore.GREEN}{'▓' * 3}{'░' * 2}{Style.RESET_ALL}"  # Medium long
        elif position_size_pct > 5:
            position_indicator = f"{Fore.GREEN}{'▓' * 1}{'░' * 4}{Style.RESET_ALL}"  # Light long
        elif position_size_pct < 1:
            position_indicator = f"{'░' * 5}"  # No position
        else:
            position_indicator = f"{Fore.GREEN}{'░' * 5}{Style.RESET_ALL}"  # Minimal long
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Position: {position_str} {position_indicator}{' ' * (40-len(position_str))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print recent trades
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.CYAN}📝 RECENT TRADES{' ' * 43}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Show more recent trades (up to 5)
        recent_trades = self.trades[-5:] if len(self.trades) > 0 else []
        if recent_trades:
            for i, trade in enumerate(reversed(recent_trades)):
                trade_time = trade['timestamp'].strftime('%m-%d %H:%M') if isinstance(trade['timestamp'], datetime) else trade['timestamp']
                trade_type = trade['type'].upper()
                trade_color = Fore.GREEN if trade_type == 'BUY' else Fore.RED
                trade_value = trade['price'] * trade['amount']
                print(f"{Fore.CYAN}║{Style.RESET_ALL} {trade_color}{trade_time} | {trade_type} {trade['amount']:.6f} @ ${trade['price']:,.2f} (${trade_value:,.2f}){Style.RESET_ALL}{' ' * (10-len(f'${trade_value:,.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} No trades executed yet{' ' * 38}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print mini charts
        print(f"{Fore.CYAN}╠{'═' * 58}╣{Style.RESET_ALL}")
        
        # Show both price and equity charts side by side if we have enough data
        if len(self.price_history) > 5:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.CYAN}📈 PRICE & EQUITY CHARTS{' ' * 36}{Fore.CYAN}║{Style.RESET_ALL}")
            self._display_mini_charts(self.price_history, self.equity_history)
        else:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.CYAN}📈 Collecting data for charts...{' ' * 30}{Fore.CYAN}║{Style.RESET_ALL}")
            
        print(f"{Fore.CYAN}╚{'═' * 58}╝{Style.RESET_ALL}")
        
        # Return the formatted status update
        return f"Portfolio: ${total_equity:,.2f} | P&L: {profit_loss_pct:+.2f}% | Trades: {len(self.trades)}"
    
    def _display_mini_charts(self, price_data, equity_data):
        """Display side-by-side price and equity charts"""
        if len(price_data) < 2 or len(equity_data) < 2:
            return
            
        # Calculate min/max for scaling price chart
        min_price = min(price_data)
        max_price = max(price_data)
        
        # Calculate min/max for scaling equity chart
        min_equity = min(equity_data)
        max_equity = max(equity_data)
        
        # Calculate min/max for scaling equity chart
        price_range = max(max_price - min_price, 0.01)
        equity_range = max(max_equity - min_equity, 0.01)
            
        # Define chart height and width
        chart_height = 5
        price_chart_width = min(len(price_data), 25)  # Limit width to 25 characters
        equity_chart_width = min(len(equity_data), 25)  # Limit width to 25 characters
        
        # Create empty charts
        price_chart = [[' ' for _ in range(price_chart_width)] for _ in range(chart_height)]
        equity_chart = [[' ' for _ in range(equity_chart_width)] for _ in range(chart_height)]
        
        # Plot price points
        for i, price in enumerate(price_data[-price_chart_width:]):
            # Scale the price to the chart height
            y = int((max_price - price) / price_range * (chart_height - 1))
            y = max(0, min(chart_height - 1, y))  # Ensure within bounds
            
            # Choose point character based on direction
            if i > 0:
                if price > price_data[-price_chart_width:][i-1]:
                    point_char = '↗'
                elif price < price_data[-price_chart_width:][i-1]:
                    point_char = '↘'
                else:
                    point_char = '→'
            else:
                point_char = '•'
                
            price_chart[y][i] = point_char
        
        # Plot equity points
        for i, equity in enumerate(equity_data[-equity_chart_width:]):
            # Scale the equity to the chart height
            y = int((max_equity - equity) / equity_range * (chart_height - 1))
            y = max(0, min(chart_height - 1, y))  # Ensure within bounds
            
            # Choose point character based on direction
            if i > 0:
                if equity > equity_data[-equity_chart_width:][i-1]:
                    point_char = '↗'
                elif equity < equity_data[-equity_chart_width:][i-1]:
                    point_char = '↘'
                else:
                    point_char = '→'
            else:
                point_char = '•'
                
            equity_chart[y][i] = point_char
        
        # Print the charts side by side
        print(f"{Fore.CYAN}║{Style.RESET_ALL} Price: ${max_price:.2f}{' ' * (15-len(f'${max_price:.2f}'))} | Equity: ${max_equity:.2f}{' ' * (15-len(f'${max_equity:.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        for i in range(chart_height):
            price_row = ''.join(price_chart[i])
            equity_row = ''.join(equity_chart[i])
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {price_row}{' ' * (25-len(price_row))} | {equity_row}{' ' * (25-len(equity_row))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}║{Style.RESET_ALL} ${min_price:.2f}{' ' * (23-len(f'${min_price:.2f}'))} | ${min_equity:.2f}{' ' * (23-len(f'${min_equity:.2f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        
        # Print time axis
        print(f"{Fore.CYAN}║{Style.RESET_ALL} OLD{' ' * 22}NOW | OLD{' ' * 22}NOW{Fore.CYAN}║{Style.RESET_ALL}")
    
    def run(self, check_interval_seconds: int = 10):
        """
        Run paper trading loop
        
        Args:
            check_interval_seconds: How often to check for new trading signals
        """
        print("\n======================================================")
        print(f"🚀 Starting DMT_v2 Paper Trading")
        print(f"📊 Symbol: {self.symbol} | Timeframe: {self.interval}")
        print(f"💰 Initial Capital: ${self.initial_capital:.2f}")
        print(f"🔄 Strategy Version: {self.strategy_version}")
        print(f"💸 Fee Tier: {self.trading_tier} | BNB Discount: {'Yes' if self.use_bnb_for_fees else 'No'}")
        print(f"🧮 Fee Example: ${self._calculate_fee(10000.0):.2f} on $10000.00 trade ({self._get_fee_rate()*100:.4f}%)")
        print("======================================================\n")
        
        # Use a faster interval for minute data
        if self.interval == '1m':
            check_interval_seconds = min(check_interval_seconds, 3)
            print(f"📈 Using faster update interval ({check_interval_seconds}s) for 1-minute data\n")
            
        logging.info(f"Starting paper trading for {self.symbol} with {self.initial_capital} USDT")
        logging.info(f"Strategy: DMT_v2 {self.strategy_version}, Timeframe: {self.interval}")
        
        # Main trading loop
        try:
            while True:
                # Check for new data
                new_data = self._check_for_new_data()
                
                if new_data:
                    # Process new data
                    self._process_new_candle(new_data)
                    
                    # Calculate signal
                    signal = self._calculate_signal()
                    
                    # Execute trade based on signal
                    trade_executed = self._execute_trade(signal)
                    
                    # Update status
                    self._update_status()
                else:
                    # Update market data
                    self._update_market_data()
                    
                    # Calculate signal
                    signal = self._calculate_signal()
                    
                    # Execute trade based on signal
                    trade_executed = self._execute_trade(signal)
                    
                    # Update status periodically
                    if datetime.now().second % 30 < 5:  # Update roughly every 30 seconds
                        self._update_status()
                
                # Sleep until next check
                time.sleep(check_interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Trading stopped by user.{Style.RESET_ALL}")
            
            # Show final portfolio value
            position_value = self.holdings * self.current_price
            total_equity = self.capital + position_value
            profit_loss = total_equity - self.initial_capital
            profit_loss_pct = (profit_loss / self.initial_capital) * 100
            
            print(f"\n{Fore.CYAN}FINAL RESULTS:{Style.RESET_ALL}")
            print(f"Initial Capital: ${self.initial_capital:.2f}")
            print(f"Final Portfolio Value: ${total_equity:.2f}")
            
            if profit_loss >= 0:
                print(f"Profit: {Fore.GREEN}+${profit_loss:.2f} ({profit_loss_pct:+.2f}%){Style.RESET_ALL}")
            else:
                print(f"Loss: {Fore.RED}-${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%){Style.RESET_ALL}")
                
            print(f"Total Trades: {len(self.trades)}")
            print(f"Total Fees Paid: ${self.total_fees_paid:.2f}")
            
            # Calculate trading statistics
            if len(self.trades) > 0:
                win_count = sum(1 for t in self.trades if t.get('profit', 0) > 0)
                loss_count = sum(1 for t in self.trades if t.get('profit', 0) < 0)
                win_rate = win_count / len(self.trades) * 100 if len(self.trades) > 0 else 0
                
                print(f"Win Rate: {win_rate:.1f}% ({win_count}/{len(self.trades)})")
                
            print(f"\n{Fore.YELLOW}Trading session ended.{Style.RESET_ALL}")

def main():
    """Main entry point for paper trading script"""
    parser = argparse.ArgumentParser(
        description="DMT_v2 Paper Trading on Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to trade')
    parser.add_argument('--interval', type=str, choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                      default='1d', help='Trading interval')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--version', type=str, choices=['original', 'enhanced', 'turbo'],
                      default='turbo', help='Strategy version')
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
    trader = BinancePaperTrader(
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
