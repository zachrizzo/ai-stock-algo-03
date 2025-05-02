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
                    # Just check if we might be able to trade (don't actually trade)
                    self.simulation_mode = not any(
                        permission['permissionType'] == 'SPOT' and 
                        permission.get('enabled', False) 
                        for permission in account_info.get('permissions', [])
                    )
                    
                    if self.simulation_mode:
                        print("⚠️ Trading permissions not detected - using simulation mode only")
                    else:
                        print("✅ Trading permissions detected")
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
                '1m': 2000,   # 500 minutes (8+ hours) of historical data
                '5m': 288,   # 24 hours of 5-minute data
                '15m': 192,  # 2 days of 15-minute data
                '30m': 96,   # 2 days of 30-minute data
                '1h': 72,    # 3 days of hourly data
                '4h': 90,    # ~2 weeks of 4-hour data
                '1d': 60     # 2 months of daily data
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
        
        # When using 1-minute data, show progress
        if self.interval == '1m' and len(self.klines) > 100:
            # Simulate stepping through data to build context
            progress_steps = min(20, len(self.klines) // 5)
            step_size = len(self.klines) // progress_steps
            
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
                except Exception as e:
                    # Just continue with progress if we can't calculate signal yet
                    progress = int((i / len(self.klines)) * 100)
                    bar_length = 30
                    filled_length = int(bar_length * progress // 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r[{bar}] {progress}% | Processed {i+1}/{len(self.klines)} bars", end="")
                
                # Small sleep to make the progress visible
                time.sleep(0.02)
        
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
                logger.info(f"Current market regime: {regime}")
                
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
            # Process the most recent historical data
            if self.strategy.historical_data is None or len(self.strategy.historical_data) == 0:
                logger.error("No historical data available to calculate signal")
                return 0.0
                
            # Get the current signal from the strategy
            results, _ = self.strategy.run_backtest(self.strategy.historical_data)
            
            if results is None or len(results) == 0:
                logger.error("Strategy returned empty results")
                return 0.0
                
            # Get the most recent signal
            signal = results['signal'].iloc[-1]
            
            # Get current market regime if available
            regime = "Unknown"
            try:
                if hasattr(self.strategy, 'current_regime'):
                    regime = self.strategy.current_regime
                elif hasattr(self.strategy, 'market_regime'):
                    regime = self.strategy.market_regime
                
                # Try to extract regime from results if not available as attribute
                if regime == "Unknown" and 'regime' in results.columns:
                    regime = results['regime'].iloc[-1]
                    
                logger.info(f"Current market regime: {regime}")
                
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
        
        # Calculate target position size as a decimal (0 to 1.0)
        target_position_size = self._calculate_position_size(signal)
        
        # Calculate current position size as a decimal
        current_position_value = self.holdings * self.current_price
        current_total_equity = self.capital + current_position_value
        current_position_size = current_position_value / current_total_equity if current_total_equity > 0 else 0
        
        # Calculate the difference in position size
        position_size_diff = target_position_size - current_position_size
        
        # Minimum adjustment threshold (don't trade for tiny adjustments)
        min_adjustment_pct = 0.05  # 5% minimum change to trigger a trade
        
        if abs(position_size_diff) < min_adjustment_pct:
            logging.info("Position adjustment too small, maintaining current position")
            logging.info("No trades executed")
            return
        
        # Calculate the amount to buy or sell in base currency units
        position_value_diff = position_size_diff * current_total_equity
        amount_diff = position_value_diff / self.current_price if self.current_price > 0 else 0
        
        # Execute the trade
        if amount_diff > 0:  # Buy
            # Check if we have enough capital
            if self.capital < position_value_diff:
                logging.info(f"Not enough capital for position increase. Need ${position_value_diff:.2f}, have ${self.capital:.2f}")
                return
            
            # Calculate fee
            fee = self._calculate_fee(position_value_diff, is_maker=False)
            
            # Check if we can still afford it with the fee
            if self.capital < position_value_diff + fee:
                position_value_diff = self.capital * 0.9995  # Adjust to 99.95% of available capital
                amount_diff = position_value_diff / self.current_price
                fee = self._calculate_fee(position_value_diff, is_maker=False)
                
            # Execute buy
            if not self.simulation_mode and self.client:
                try:
                    # In a real implementation, we'd place an order here
                    pass
                except Exception as e:
                    print(f"{Fore.RED}❌ API Error: {e}{Style.RESET_ALL}")
                    return
                    
            # Update balances in our simulation
            self.capital -= position_value_diff + fee
            self.holdings += amount_diff
            self.total_fees_paid += fee
            
            # Add to positions list
            self.positions.append({
                'type': 'long',
                'amount': amount_diff,
                'price': self.current_price,
                'value': position_value_diff,
                'fee': fee,
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Add to trades list
            self.trades.append({
                'type': 'buy',
                'amount': amount_diff,
                'price': self.current_price,
                'value': position_value_diff,
                'fee': fee,
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'signal': signal
            })
            
            # Log the trade with colorful output
            print(f"{Fore.GREEN}💰 Trade executed: BUY {amount_diff:.6f} {self.symbol[:3]} at ${self.current_price:,.2f}{Style.RESET_ALL}")
            print(f"💼 New position: {len(self.positions)} positions | Cash: ${self.capital:.2f} | Holdings: {self.holdings:.6f} {self.symbol[:3]}")
            print(f"   Fee paid: ${fee:.2f}")
            
            # Update trading volume for tier calculation
            self.trading_volume_30d += position_value_diff
            
        elif amount_diff < 0:  # Sell
            amount_to_sell = abs(amount_diff)
            
            # Check if we have enough holdings
            if self.holdings < amount_to_sell:
                amount_to_sell = self.holdings  # Sell what we have
                
            # Calculate the value
            sell_value = amount_to_sell * self.current_price
            
            # Calculate fee
            fee = self._calculate_fee(sell_value, is_maker=False)
            
            # Execute sell
            if not self.simulation_mode and self.client:
                try:
                    # In a real implementation, we'd place an order here
                    pass
                except Exception as e:
                    print(f"{Fore.RED}❌ API Error: {e}{Style.RESET_ALL}")
                    return
            
            # Calculate profit/loss for this trade
            avg_buy_price = 0
            if self.holdings > 0:
                # Calculate average price of accumulated position
                buy_positions = [p for p in self.positions if p['type'] == 'long']
                total_buy_value = sum(p['value'] for p in buy_positions)
                total_buy_amount = sum(p['amount'] for p in buy_positions)
                avg_buy_price = total_buy_value / total_buy_amount if total_buy_amount > 0 else 0
                
            # Calculate profit/loss
            trade_profit = (self.current_price - avg_buy_price) * amount_to_sell if avg_buy_price > 0 else 0
            
            # Update balances in our simulation
            self.capital += sell_value - fee
            self.holdings -= amount_to_sell
            self.total_fees_paid += fee
            
            # Remove from positions list if fully sold
            if self.holdings <= 0.000001:
                self.positions = []
                self.holdings = 0
            else:
                # Adjust positions proportionally
                self.positions = [
                    {**position, 'amount': position['amount'] * (1 - amount_to_sell/self.holdings + amount_to_sell)}
                    for position in self.positions
                ]
            
            # Add to trades list with profit/loss
            self.trades.append({
                'type': 'sell',
                'amount': amount_to_sell,
                'price': self.current_price,
                'value': sell_value,
                'fee': fee,
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'signal': signal,
                'profit': trade_profit,
                'profit_pct': (trade_profit / (avg_buy_price * amount_to_sell)) * 100 if avg_buy_price > 0 else 0
            })
            
            # Log the trade with profit/loss information
            profit_indicator = ""
            if avg_buy_price > 0:
                profit_pct = (self.current_price - avg_buy_price) / avg_buy_price * 100
                if profit_pct > 0:
                    profit_indicator = f"{Fore.GREEN}(+{profit_pct:.2f}%){Style.RESET_ALL}"
                else:
                    profit_indicator = f"{Fore.RED}({profit_pct:.2f}%){Style.RESET_ALL}"
                    
            print(f"{Fore.RED}💰 Trade executed: SELL {amount_to_sell:.6f} {self.symbol[:3]} at ${self.current_price:,.2f} {profit_indicator}{Style.RESET_ALL}")
            print(f"💼 New position: {len(self.positions)} positions | Cash: ${self.capital:.2f} | Holdings: {self.holdings:.6f} {self.symbol[:3]}")
            print(f"   Fee paid: ${fee:.2f}")
            
            if trade_profit != 0:
                if trade_profit > 0:
                    profit_color = Fore.GREEN
                    print(f"   {profit_color}Profit: +${trade_profit:.2f}{Style.RESET_ALL}")
                else:
                    profit_color = Fore.RED
                    print(f"   {profit_color}Loss: -${abs(trade_profit):.2f}{Style.RESET_ALL}")
            
            # Update trading volume for tier calculation
            self.trading_volume_30d += sell_value
            
        # Check for tier upgrade
        self._check_tier_upgrade()
    
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
        - neutral_zone: 0.03 (reduced from 0.05)
        
        Args:
            signal: Trading signal from -1.0 to 1.0
            
        Returns:
            Position size as a percentage of capital (0.0 to max_position_size)
        """
        # Parameters from enhanced DMT_v2 strategy
        max_position_size = 1.0  # Cap at 100% of capital for paper trading
        neutral_zone = 0.03      # Signal threshold for taking positions
        
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
        position_value = self.holdings * self.current_price if self.current_price > 0 else 0
        total_equity = self.capital + position_value
        profit_loss = total_equity - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Update equity history and track max profit/drawdown
        self.equity_history.append(total_equity)
        if len(self.equity_history) > self.max_price_history:
            self.equity_history = self.equity_history[-self.max_price_history:]
            
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
            
        # Get current market regime
        regime = self.strategy.detect_market_regime(self.strategy.historical_data)
        if regime == "bullish":
            regime_color = Fore.GREEN
        elif regime == "bearish":
            regime_color = Fore.RED
        else:
            regime_color = Fore.YELLOW
            
        # Current time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Format signal indicator based on last signal
        if hasattr(self, 'last_signal'):
            if self.last_signal > 0.5:
                signal_indicator = f"{Fore.GREEN}↑ {self.last_signal:.2f}{Fore.RESET}"
                confidence = "HIGH" if self.last_signal > 0.75 else "MEDIUM"
            elif self.last_signal < -0.5:
                signal_indicator = f"{Fore.RED}↓ {self.last_signal:.2f}{Fore.RESET}"
                confidence = "HIGH" if self.last_signal < -0.75 else "MEDIUM"
            elif self.last_signal > 0:
                signal_indicator = f"{Fore.GREEN}→ {self.last_signal:.2f}{Fore.RESET}"
                confidence = "LOW"
            elif self.last_signal < 0:
                signal_indicator = f"{Fore.RED}← {self.last_signal:.2f}{Fore.RESET}"
                confidence = "LOW"
            else:
                signal_indicator = f"{Fore.WHITE}• {self.last_signal:.2f}{Fore.RESET}"
                confidence = "NEUTRAL"
        else:
            signal_indicator = f"{Fore.WHITE}• N/A{Fore.RESET}"
            confidence = "UNKNOWN"
        
        # Calculate position metrics
        position_pct = (position_value / total_equity) * 100 if total_equity > 0 else 0
        
        # Calculate trading session duration
        session_duration = datetime.now() - self.start_time
        hours = session_duration.seconds // 3600
        minutes = (session_duration.seconds % 3600) // 60
        seconds = session_duration.seconds % 60
        
        # Update price history for chart
        if self.current_price > 0:
            self.price_history.append(self.current_price)
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]
        
        print(f"\n{Back.BLUE}{Fore.WHITE} STATUS UPDATE @ {timestamp} {Style.RESET_ALL}")
        print(f"{'='*60}")
        
        # Print price information with change indicators
        if hasattr(self, 'last_price') and self.last_price > 0:
            price_change = self.current_price - self.last_price
            price_change_pct = (price_change / self.last_price) * 100 if self.last_price > 0 else 0
            
            if price_change > 0:
                price_indicator = f"{Fore.GREEN}▲ ${self.current_price:.2f} (+${price_change:.2f}, +{price_change_pct:.2f}%){Fore.RESET}"
            elif price_change < 0:
                price_indicator = f"{Fore.RED}▼ ${self.current_price:.2f} (-${abs(price_change):.2f}, {price_change_pct:.2f}%){Fore.RESET}"
            else:
                price_indicator = f"{Fore.WHITE}► ${self.current_price:.2f} (0.00%){Fore.RESET}"
                
            print(f"📊 Price: {price_indicator}")
        else:
            print(f"📊 Price: ${self.current_price:.2f}")
        
        # Generate and print mini price chart using ASCII/Unicode
        if len(self.price_history) >= 2:
            print(f"{'='*60}")
            print(f"📈 PRICE CHART (Last {len(self.price_history)} prices):")
            self._display_mini_chart(self.price_history)
            
        # Print market regime
        print(f"{'='*60}")
        print(f"🌍 Market Regime: {regime_color}{regime.capitalize()}{Style.RESET_ALL}")
        
        # Print signal with confidence level
        print(f"💡 Signal: {signal_indicator} | Confidence: {confidence}")
        
        # Print portfolio information
        print(f"{'='*60}")
        print(f"💰 PORTFOLIO SUMMARY:")
        print(f"   Cash: ${self.capital:.2f}")
        
        if self.holdings != 0:
            print(f"   Holdings: {self.holdings:.6f} {self.symbol[:3]} (${position_value:.2f}, {position_pct:.2f}% of portfolio)")
            
        print(f"   Total Equity: ${total_equity:.2f}")
        print(f"   P&L: {pl_color}${profit_loss:.2f} ({profit_loss_pct:.2f}%){Style.RESET_ALL}")
        print(f"   Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Print trade metrics
        print(f"{'='*60}")
        print(f"📈 TRADING METRICS:")
        print(f"   Trades: {len(self.trades)}")
        print(f"   Fees Paid: ${self.total_fees_paid:.2f}")
        
        # Calculate win/loss if we have trades
        if len(self.trades) > 0:
            winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
            win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
            print(f"   Win Rate: {win_rate:.2f}%")
            print(f"   Best Profit: ${self.best_profit:.2f}")
            print(f"   Max Drawdown: ${self.worst_drawdown:.2f}")
            
            # Show the last 3 trades if available
            print(f"{'='*60}")
            print(f"📝 RECENT TRADES:")
            recent_trades = self.trades[-3:] if len(self.trades) >= 3 else self.trades
            for i, trade in enumerate(reversed(recent_trades)):
                trade_type = trade.get('type', 'unknown')
                amount = trade.get('amount', 0)
                price = trade.get('price', 0)
                trade_time = trade.get('datetime', 'unknown')
                profit = trade.get('profit', 0)
                
                if trade_type == 'buy':
                    trade_color = Fore.GREEN
                    trade_symbol = "▲"
                else:
                    trade_color = Fore.RED
                    trade_symbol = "▼"
                    
                profit_str = ""
                if profit != 0:
                    profit_color = Fore.GREEN if profit > 0 else Fore.RED
                    profit_str = f" | P/L: {profit_color}${profit:.2f}{Style.RESET_ALL}"
                    
                print(f"   {i+1}. {trade_color}{trade_symbol} {trade_type.upper()}{Style.RESET_ALL} {amount:.6f} @ ${price:.2f} ({trade_time}){profit_str}")
            
        # Risk assessment
        if self.holdings > 0:
            print(f"{'='*60}")
            print(f"⚠️ RISK ASSESSMENT:")
            
            # Calculate position exposure
            exposure_pct = position_pct
            if exposure_pct > 25:
                exposure_color = Fore.RED
                exposure_risk = "HIGH"
            elif exposure_pct > 15:
                exposure_color = Fore.YELLOW
                exposure_risk = "MEDIUM"
            else:
                exposure_color = Fore.GREEN
                exposure_risk = "LOW"
                
            # Calculate volatility if we have price history
            if len(self.price_history) >= 5:
                price_changes = [abs((self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]) 
                               for i in range(1, len(self.price_history))]
                self.volatility = sum(price_changes) / len(price_changes) * 100
                
                if self.volatility > 1.0:  # More than 1% average price change
                    vol_color = Fore.RED
                    vol_risk = "HIGH"
                elif self.volatility > 0.5:
                    vol_color = Fore.YELLOW
                    vol_risk = "MEDIUM"
                else:
                    vol_color = Fore.GREEN
                    vol_risk = "LOW"
                    
                print(f"   Market Volatility: {vol_color}{self.volatility:.2f}%{Style.RESET_ALL} ({vol_risk})")
                
            print(f"   Position Exposure: {exposure_color}{exposure_pct:.2f}%{Style.RESET_ALL} ({exposure_risk})")
            
            # Potential loss calculation
            potential_loss_5pct = position_value * 0.05  # 5% price drop
            potential_loss_10pct = position_value * 0.10  # 10% price drop
            
            print(f"   Potential Loss (5% Drop): ${potential_loss_5pct:.2f}")
            print(f"   Potential Loss (10% Drop): ${potential_loss_10pct:.2f}")
            
            if confidence == "LOW" and regime != "bullish":
                print(f"   {Fore.YELLOW}⚠️ Consider reducing position size (weak signal in non-bullish market){Style.RESET_ALL}")
            
        # Save last price for next comparison
        self.last_price = self.current_price
        
        print(f"{'='*60}")

    def _display_mini_chart(self, price_data):
        """Display a simple ASCII chart of price data"""
        if len(price_data) < 2:
            return
            
        # Calculate min/max for scaling
        min_price = min(price_data)
        max_price = max(price_data)
        
        # Avoid division by zero
        if max_price == min_price:
            price_range = 1
        else:
            price_range = max_price - min_price
            
        # Define chart height and width
        chart_height = 5
        chart_width = len(price_data)
        
        # Create empty chart
        chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Plot points
        for i, price in enumerate(price_data):
            # Scale the price to the chart height
            if price_range == 0:
                y = chart_height // 2
            else:
                y = int((max_price - price) / price_range * (chart_height - 1))
                y = max(0, min(chart_height - 1, y))  # Ensure within bounds
            
            # Choose point character based on direction
            if i > 0:
                if price > price_data[i-1]:
                    point_char = '↗'
                elif price < price_data[i-1]:
                    point_char = '↘'
                else:
                    point_char = '→'
            else:
                point_char = '•'
                
            chart[y][i] = point_char
        
        # Print the chart
        print(f"   ${max_price:.2f}")
        for row in chart:
            print("   " + ''.join(row))
        print(f"   ${min_price:.2f}")
        
        # Print time axis
        time_indicators = ["OLD", "   ", "   ", "   ", "NOW"]
        time_position = [0, chart_width//4, chart_width//2, 3*chart_width//4, chart_width-3]
        time_line = ['   '] + [' ' for _ in range(chart_width)]
        
        for i, pos in enumerate(time_position):
            if pos < len(time_line)-3:  # Ensure we're within bounds
                time_line[pos+1:pos+1+len(time_indicators[i])] = time_indicators[i]
                
        print(''.join(time_line))
    
    def run(self, check_interval_seconds: int = 10):
        """
        Run paper trading loop
        
        Args:
            check_interval_seconds: How often to check for new trading signals
        """
        # First print a nice header
        symbol_base = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol
        symbol_quote = 'USDT' if self.symbol.endswith('USDT') else self.symbol[-3:]
        
        # Determine interval description
        interval_desc = {
            '1m': '1-minute',
            '5m': '5-minute',
            '15m': '15-minute',
            '30m': '30-minute',
            '1h': '1-hour',
            '4h': '4-hour',
            '1d': '1-day'
        }.get(self.interval, self.interval)
        
        # Calculate fee example
        example_trade_value = 10000.0  # $10,000 trade
        example_fee = self._calculate_fee(example_trade_value, is_maker=False)
        example_fee_pct = (example_fee / example_trade_value) * 100
        
        # Set faster interval for short timeframe data
        if self.interval == '1m':
            check_interval_seconds = min(check_interval_seconds, 3)
            print(f"\n{Fore.CYAN}📈 Using faster update interval ({check_interval_seconds}s) for 1-minute data{Style.RESET_ALL}")
        
        # Use bright colors for the banner
        print(f"\n{Back.BLUE}{Fore.WHITE}======================================================{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}🚀 Starting TurboDMT_v2 Paper Trading{' Simulation' if self.simulation_mode else ''}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}📊 Symbol: {self.symbol} | Timeframe: {self.interval}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}💰 Initial Capital: ${self.initial_capital:.2f}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}🔄 Strategy Version: {self.strategy_version}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}💸 Fee Tier: {self.trading_tier} | BNB Discount: {'Yes' if self.use_bnb_for_fees else 'No'}{Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}🧮 Fee Example: ${example_fee:.2f} on ${example_trade_value:.2f} trade ({example_fee_pct:.4f}%){Style.RESET_ALL}")
        print(f"{Back.BLUE}{Fore.WHITE}======================================================{Style.RESET_ALL}")
        print("")
        
        # Log start of trading
        logging.info(f"Starting paper trading for {self.symbol} with {self.initial_capital} {symbol_quote}")
        logging.info(f"Strategy: DMT_v2 {self.strategy_version}, Timeframe: {self.interval}")
        
        last_status_time = datetime.now()
        last_signal_check = datetime.now()
        
        # Main trading loop
        try:
            while True:
                # Update current price and market data
                self._update_market_data()
                
                # Check for new candle data
                new_candle = self._check_for_new_data()
                
                # Calculate new signal
                signal = self._calculate_signal()
                
                # Try to execute trade based on signal
                trade_executed = self._execute_trade(signal)
                if not trade_executed:
                    logger.info("No trades executed")
                
                # Calculate total portfolio value including positions
                position_value = self.holdings * self.current_price
                total_equity = self.capital + position_value
                
                # Print status update periodically
                current_time = datetime.now()
                if (current_time - last_status_time).total_seconds() >= 60:
                    print(f"🔄 {self.last_update.strftime('%Y-%m-%d %H:%M')} | ${self.current_price:.2f} | Vol: {self.current_volume:.2f}")
                    print(f"💰 Portfolio: Cash ${self.capital:.2f} + BTC ${position_value:.2f} = Total ${total_equity:.2f}")
                    last_status_time = current_time
                
                # Sleep before next check
                time.sleep(check_interval_seconds)
                
        except KeyboardInterrupt:
            print("\n✅ Paper trading stopped by user")
            position_value = self.holdings * self.current_price
            total_equity = self.capital + position_value
            print(f"💰 Final equity: ${total_equity:.2f} (Cash: ${self.capital:.2f} + BTC: ${position_value:.2f})")
            print(f"📈 P&L: {((total_equity / self.initial_capital) - 1) * 100:.2f}%")
            print(f"💸 Total fees paid: ${self.total_fees_paid:.2f}")
            return
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            traceback.print_exc()
            return
    
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
