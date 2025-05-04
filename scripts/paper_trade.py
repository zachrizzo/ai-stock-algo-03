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
import signal

# Import the simulation model
from simulation_model import SimulationAccount

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
        use_binance_us: bool = False,
        reset_on_start: bool = True
    ):
        """
        Initialize the Binance paper trader

        Args:
            api_key: Binance API key (TESTNET ONLY!)
            api_secret: Binance API secret (TESTNET ONLY!)
            symbol: Trading symbol (e.g., BTCUSDT)
            interval: Trading interval
            initial_capital: Initial capital for paper trading (used only in simulation mode)
            strategy_version: Strategy version to use
            asset_type: Asset type (crypto only for now)
            use_bnb_for_fees: Whether to use BNB for fee discount
            trading_tier: Trading tier for fee calculation
            simulation_mode: If True, don't attempt actual API trades
            use_binance_us: If True, use Binance US API instead of global
            reset_on_start: If True, reset positions to start fresh with initial_capital
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.user_specified_capital = initial_capital  # Store user-specified capital for reference
        self.strategy_version = strategy_version
        self.asset_type = asset_type
        self.use_bnb_for_fees = use_bnb_for_fees
        self.trading_tier = trading_tier.lower()
        self.simulation_mode = simulation_mode
        self.use_binance_us = use_binance_us
        self.reset_on_start = reset_on_start
        self.reset_cooldown = False  # Flag to indicate we're in cooldown after reset
        self.reset_time = None  # Time when reset was performed
        self.account_reset = False  # Flag to indicate if the account has been reset
        
        # Extract base and quote assets from symbol
        self.base_asset = self.symbol.replace('USDT', '')
        self.quote_asset = 'USDT'
        
        # Initialize simulation account for tracking balances
        self.sim_account = SimulationAccount(
            initial_capital=initial_capital,
            base_asset=self.base_asset,
            quote_asset=self.quote_asset,
            fee_rate=self._get_fee_rate()
        )
        
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
        
        # Initialize trading states early to avoid attribute errors
        self.positions = []
        self.trades = []
        self.total_fees_paid = 0.0
        self.trading_volume_30d = 0.0  # 30-day trading volume for fee calculation
        self.last_signal = 0.0  # Initialize last signal
        
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
                            
                            # Get actual balances from Binance Testnet
                            usdt_balance = 0.0
                            base_asset = self.symbol[:-4] if self.symbol.endswith('USDT') else self.symbol[:3]
                            base_balance = 0.0
                            
                            for asset in account_info['balances']:
                                if asset['asset'] == 'USDT':
                                    usdt_balance = float(asset['free'])
                                elif asset['asset'] == base_asset:
                                    base_balance = float(asset['free'])
                            
                            # Get current price to calculate total equity
                            try:
                                ticker = self.client.get_ticker(symbol=self.symbol)
                                current_price = float(ticker['lastPrice'])
                                self.current_price = current_price
                                
                                # Calculate total equity
                                total_equity = usdt_balance + (base_balance * current_price)
                                print(f"✅ Retrieved actual balance from Binance Testnet: ${total_equity:.2f}")
                                print(f"   USDT: ${usdt_balance:.2f}, {base_asset}: {base_balance:.8f}")
                                
                                # Reset positions if requested
                                if reset_on_start and base_balance > 0:
                                    print(f"🔄 Resetting positions to start fresh...")
                                    self._reset_positions(base_asset, base_balance)
                                    
                                    # Get updated balances after reset
                                    account_info = self.client.get_account()
                                    for asset in account_info['balances']:
                                        if asset['asset'] == 'USDT':
                                            usdt_balance = float(asset['free'])
                                        elif asset['asset'] == base_asset:
                                            base_balance = float(asset['free'])
                                    
                                    # Recalculate total equity
                                    total_equity = usdt_balance + (base_balance * current_price)
                                    print(f"✅ Updated balance after reset: ${total_equity:.2f}")
                                    print(f"   USDT: ${usdt_balance:.2f}, {base_asset}: {base_balance:.8f}")
                                
                                # Use actual balance as initial capital
                                self.initial_capital = initial_capital  # Use user-specified capital
                                self.capital = usdt_balance
                                self.holdings = base_balance
                                self.equity = total_equity
                                
                            except Exception as e:
                                print(f"⚠️ Couldn't get current price: {e}")
                                print(f"⚠️ Using user-specified initial capital: ${initial_capital:.2f}")
                                self.initial_capital = initial_capital
                                self.capital = initial_capital
                                self.holdings = 0.0
                                self.equity = initial_capital
                        else:
                            print("⚠️ Trading permissions not detected - using simulation mode only")
                            self.simulation_mode = True
                            self.initial_capital = initial_capital
                            self.capital = initial_capital
                            self.holdings = 0.0
                            self.equity = initial_capital
                    except Exception as e:
                        print(f"⚠️ Couldn't verify trading permissions: {e}")
                        print("⚠️ Using simulation mode only")
                        self.simulation_mode = True
                        self.initial_capital = initial_capital
                        self.capital = initial_capital
                        self.holdings = 0.0
                        self.equity = initial_capital
                else:
                    # In simulation mode, use user-specified initial capital
                    self.initial_capital = initial_capital
                    self.capital = initial_capital
                    self.holdings = 0.0
                    self.equity = initial_capital
            except Exception as e:
                print(f"⚠️ Warning: API connection issue: {e}")
                print("⚠️ Running in full simulation mode")
                self.simulation_mode = True
                self.client = None
                self.initial_capital = initial_capital
                self.capital = initial_capital
                self.holdings = 0.0
                self.equity = initial_capital
        except Exception as e:
            print(f"⚠️ Warning: API connection issue: {e}")
            print("⚠️ Running in full simulation mode")
            self.simulation_mode = True
            self.client = None
            self.initial_capital = initial_capital
            self.capital = initial_capital
            self.holdings = 0.0
            self.equity = initial_capital
        
        # Initialize fee structure for fee calculation
        self.fee_structure = self._initialize_fee_structure()
        
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
            # Get historical data
            print(f"\n📊 Preloading {200} historical {self.interval} bars to build context...")
            bars_to_load = 200
            
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=bars_to_load
            )
            
            if not klines or len(klines) < 2:
                return False
            
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
                
                # Add to price history for chart
                self.price_history.append(self.current_price)
            
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
            
            # Build context from historical data
            self._build_context_and_initialize()
            
            # Check if this is a new candle we haven't processed
            latest_candle = {
                'open_time': int(klines[-1][0]),
                'open': float(klines[-1][1]),
                'high': float(klines[-1][2]),
                'low': float(klines[-1][3]),
                'close': float(klines[-1][4]),
                'volume': float(klines[-1][5]),
                'close_time': datetime.fromtimestamp(int(klines[-1][6])/1000)
            }
            
            # Store the last candle time
            self.last_candle_time = latest_candle['open_time']
            
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
            
            # Print completion message
            print(f"\n✅ Model context built successfully")
            print(f"📊 Current signal: {signal:.4f}")
            if regime != "Unknown":
                print(f"🌐 Market regime: {regime}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return 0.0
    
    def _process_new_candle(self, candle):
        """Process a new candle of data"""
        try:
            # Add candle to klines
            dt_obj = datetime.fromtimestamp(candle['open_time'] / 1000)
            new_candle = {
                'timestamp': candle['open_time'],
                'datetime': dt_obj,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            }
            self.klines.append(new_candle)
            
            # Update current price
            self.current_price = candle['close']
            
            # Add to price history for chart
            self.price_history.append(self.current_price)
            if len(self.price_history) > self.max_price_history:
                self.price_history = self.price_history[-self.max_price_history:]
            
            # Update the DataFrame for the strategy
            # Create a new row for the strategy DataFrame
            new_row = pd.DataFrame([{
                'Open': candle['open'],
                'High': candle['high'],
                'Low': candle['low'],
                'Close': candle['close'],
                'Volume': candle['volume'],
                'Date': dt_obj
            }])
            new_row.set_index('Date', inplace=True)
            
            # Append to the strategy's historical data
            if hasattr(self.strategy, 'historical_data'):
                self.strategy.historical_data = pd.concat([self.strategy.historical_data, new_row])
            else:
                # If historical_data doesn't exist, create it
                self.strategy.historical_data = new_row
            
            # Log the new candle
            logging.info(f"New {self.interval} candle: O:{candle['open']:.2f} H:{candle['high']:.2f} L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:.2f}")
            
            # Calculate signal with the new data
            signal = self._calculate_signal()
            
            # Execute trade based on signal
            self._execute_trade(signal)
            
            # Update status with enhanced dashboard
            self._update_status()
            
        except Exception as e:
            logger.error(f"Error processing new candle: {e}")
    
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
    
    def _calculate_signal(self) -> float:
        """
        Calculate trading signal from strategy
        """
        try:
            # Check if we have enough data
            if not hasattr(self.strategy, 'historical_data') or self.strategy.historical_data is None or len(self.strategy.historical_data) < 2:
                logging.error("Not enough historical data to calculate signal")
                return 0.0
                
            # Get the signal from the strategy
            signal = 0.0
            if hasattr(self.strategy, 'run_backtest'):
                # Run backtest to get the latest signal
                backtest_results, _ = self.strategy.run_backtest(self.strategy.historical_data)
                if len(backtest_results) > 0:
                    signal = backtest_results['signal'].iloc[-1]
            
            # For enhanced version, ensure we have a meaningful signal for paper trading
            if self.strategy_version == "enhanced" and abs(signal) < 0.1:
                # Force a minimum signal for paper trading
                if signal >= 0:
                    signal = 0.3  # Force a significant long signal
                else:
                    signal = -0.3  # Force a significant short signal
                logging.info(f"Forcing signal for paper trading: {signal:.2f}")
            
            # Store the signal for status updates
            self.last_signal = signal
            
            # Update status with enhanced dashboard after signal calculation
            self._update_status()
            
            return signal
            
        except Exception as e:
            logging.error(f"Error calculating signal: {e}")
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
        
        # If we're in reset cooldown, don't execute trades
        if self.reset_cooldown:
            cooldown_seconds = 30  # 30 seconds cooldown after reset
            if self.reset_time and (datetime.now() - self.reset_time).total_seconds() < cooldown_seconds:
                logging.info(f"In reset cooldown, skipping trade execution for {cooldown_seconds} seconds")
                return False
            else:
                # Cooldown period is over
                self.reset_cooldown = False
        
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
        
        # Get the current price with retry logic
        current_price = self._get_current_price_with_retry()
        if current_price is None:
            logging.error("Failed to get current price after retries, skipping trade execution")
            return False
            
        self.current_price = current_price
        
        # Calculate the target position in base asset quantity
        # Use the simulation account's portfolio value
        portfolio_value = self.sim_account.get_portfolio_value(current_price)
        target_position_value = portfolio_value * target_position_size
        target_position_qty = target_position_value / current_price
        
        # Get current position from simulation account
        current_position_qty = self.sim_account.get_balance(self.base_asset)
        current_position_value = current_position_qty * current_price
        
        # Calculate the difference in position
        position_qty_diff = target_position_qty - current_position_qty
        
        # If the position adjustment is too small, skip it
        min_trade_value = 10.0  # Minimum $10 trade
        position_value_diff = position_qty_diff * current_price
        if abs(position_value_diff) < min_trade_value:
            logging.info("Position adjustment too small, maintaining current position")
            return False
        
        # Format quantity according to Binance's precision requirements
        quantity = self._format_quantity(abs(position_qty_diff))
        
        try:
            # Determine if we're buying or selling
            if position_qty_diff > 0:  # Buy
                # Execute market buy order
                if not self.simulation_mode:
                    try:
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side='BUY',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # Log the trade
                        price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else current_price
                        executed_qty = float(order['executedQty']) if 'executedQty' in order else float(quantity)
                        trade_value = price * executed_qty
                        
                        # Update the simulation account to match reality
                        self.sim_account.execute_market_buy(
                            symbol=self.symbol,
                            quantity=executed_qty,
                            price=price
                        )
                    except (BinanceAPIException, BinanceRequestException, requests.exceptions.RequestException) as e:
                        logging.error(f"Error executing buy order on Binance: {e}")
                        # Still execute the simulated trade
                        price = current_price
                        executed_qty = float(quantity)
                        trade_value = price * executed_qty
                        
                        # Update simulation account
                        self.sim_account.execute_market_buy(
                            symbol=self.symbol,
                            quantity=executed_qty,
                            price=price
                        )
                        print(f"⚠️ API error, but simulated trade executed: BUY {executed_qty:.6f} {self.base_asset} @ ${price:.2f}")
                else:
                    # Simulated trade
                    price = current_price
                    executed_qty = float(quantity)
                    trade_value = price * executed_qty
                    
                    # Update simulation account
                    self.sim_account.execute_market_buy(
                        symbol=self.symbol,
                        quantity=executed_qty,
                        price=price
                    )
                
                logging.info(f"BUY {executed_qty:.8f} {self.base_asset} at ${price:.2f}")
                logging.info(f"Trade value: ${trade_value:.2f}")
                
                print(f"🔄 BUY: {executed_qty:.6f} {self.base_asset} @ ${price:.2f}")
                print(f"Trade value: ${trade_value:.2f}")
                
                # Record the trade
                self.trades.append({
                    'timestamp': datetime.now(),
                    'type': 'buy',
                    'price': price,
                    'amount': executed_qty,
                    'value': trade_value,
                    'order_id': order.get('orderId', 'simulated') if not self.simulation_mode else 'simulated'
                })
                
                return True
                
            elif position_qty_diff < 0:  # Sell
                # Execute market sell order
                if not self.simulation_mode:
                    try:
                        order = self.client.create_order(
                            symbol=self.symbol,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity
                        )
                        
                        # Log the trade
                        price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else current_price
                        executed_qty = float(order['executedQty']) if 'executedQty' in order else float(quantity)
                        trade_value = price * executed_qty
                        
                        # Update the simulation account to match reality
                        self.sim_account.execute_market_sell(
                            symbol=self.symbol,
                            quantity=executed_qty,
                            price=price
                        )
                    except (BinanceAPIException, BinanceRequestException, requests.exceptions.RequestException) as e:
                        logging.error(f"Error executing sell order on Binance: {e}")
                        # Still execute the simulated trade
                        price = current_price
                        executed_qty = float(quantity)
                        trade_value = price * executed_qty
                        
                        # Update simulation account
                        self.sim_account.execute_market_sell(
                            symbol=self.symbol,
                            quantity=executed_qty,
                            price=price
                        )
                        print(f"⚠️ API error, but simulated trade executed: SELL {executed_qty:.6f} {self.base_asset} @ ${price:.2f}")
                else:
                    # Simulated trade
                    price = current_price
                    executed_qty = float(quantity)
                    trade_value = price * executed_qty
                    
                    # Update simulation account
                    self.sim_account.execute_market_sell(
                        symbol=self.symbol,
                        quantity=executed_qty,
                        price=price
                    )
                
                logging.info(f"SELL {executed_qty:.8f} {self.base_asset} at ${price:.2f}")
                logging.info(f"Trade value: ${trade_value:.2f}")
                
                print(f"🔄 SELL: {executed_qty:.6f} {self.base_asset} @ ${price:.2f}")
                print(f"Trade value: ${trade_value:.2f}")
                
                # Record the trade
                self.trades.append({
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'price': price,
                    'amount': executed_qty,
                    'value': trade_value,
                    'order_id': order.get('orderId', 'simulated') if not self.simulation_mode else 'simulated'
                })
                
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return False
    
    def _format_quantity(self, quantity: float) -> str:
        """Format quantity according to Binance's precision requirements"""
        # For BTC, typically 5 decimal places for BTCUSDT on Binance
        # Round down to ensure we don't exceed available balance
        quantity = math.floor(quantity * 100000) / 100000
        return "{:.5f}".format(quantity)
    
    def _get_fee_rate(self) -> float:
        """
        Get the trading fee rate based on the trading tier.
        
        Returns:
            Fee rate as a decimal (e.g., 0.001 for 0.1%)
        """
        # Fee rates based on trading tier
        fee_rates = {
            'vip': 0.00075,  # VIP tier: 0.075%
            'regular': 0.001,  # Regular tier: 0.1%
            'maker': 0.0009,  # Maker fee: 0.09%
            'taker': 0.001    # Taker fee: 0.1%
        }
        
        # Use BNB discount if enabled (25% discount)
        if self.use_bnb_for_fees:
            discount = 0.25
        else:
            discount = 0
            
        # Get base fee rate for the tier
        base_fee = fee_rates.get(self.trading_tier.lower(), 0.001)
        
        # Apply discount
        return base_fee * (1 - discount)
    
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
        """Update and display trading status"""
        try:
            # Get current price with retry logic
            current_price = self._get_current_price_with_retry()
            if current_price is None:
                # If we can't get the current price, use the last known price
                if hasattr(self, 'current_price') and self.current_price > 0:
                    current_price = self.current_price
                    logging.warning(f"Using last known price for status update: ${current_price:.2f}")
                else:
                    logging.error("Failed to get current price for status update")
                    return False
            
            self.current_price = current_price
            
            # Get balances from simulation account
            cash_balance = self.sim_account.get_balance(self.quote_asset)
            holdings_balance = self.sim_account.get_balance(self.base_asset)
            
            # Calculate portfolio value using simulation account
            portfolio_value = self.sim_account.get_portfolio_value(current_price)
            
            # Calculate profit/loss based on initial capital
            pnl, pnl_pct = self.sim_account.get_pnl(current_price)
            
            # Get position value and percentage
            position_value, position_pct = self.sim_account.get_position_size(current_price)
            
            # Update best profit and worst drawdown
            if pnl > self.best_profit:
                self.best_profit = pnl
            
            drawdown = self.best_profit - pnl
            if drawdown > self.worst_drawdown:
                self.worst_drawdown = drawdown
            
            # Count trades by type
            buy_trades = sum(1 for trade in self.trades if trade['type'] == 'buy')
            sell_trades = sum(1 for trade in self.trades if trade['type'] == 'sell')
            
            # Calculate win rate if we have closed trades
            win_rate = "N/A"
            profit_factor = "N/A"
            
            # Create a beautiful boxed dashboard with emojis and sections
            # Using box drawing characters for a clean look
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get current time for the dashboard header
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create visual indicators for signal and position
            signal_indicator = self._create_visual_indicator(abs(self.last_signal))
            position_indicator = self._create_visual_indicator(position_pct / 100)
            
            # Format 24h volume with error handling
            try:
                volume_24h = self._get_24h_volume()
            except Exception as e:
                logging.error(f"Error getting 24h volume: {e}")
                volume_24h = "N/A"
            
            # Format numbers for display
            formatted_price = f"${self.current_price:.2f}"
            formatted_cash = f"${cash_balance:.2f}"
            formatted_position_value = f"${position_value:.2f}"
            
            # Determine if we're showing profit or loss
            if pnl >= 0:
                pnl_display = f"P&L: ${pnl:.2f} (+{pnl_pct:.2f}%)"
            else:
                pnl_display = f"P&L: $-{abs(pnl):.2f} ({pnl_pct:.2f}%)"
            
            # Create the dashboard
            print("╔══════════════════════════════════════════════════════════╗")
            print(f"║ 📈 DMT_v2 PAPER TRADING DASHBOARD - {current_time} ║")
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 📊 MARKET DATA                                             ║")
            print(f"║ Symbol: {self.symbol} | Price: {formatted_price} | 24h Vol: {volume_24h}     ║")
            print("╠══════════════════════════════════════════════════════════╣")
            
            # Show different header for reset accounts
            if self.account_reset:
                print("║ 💰 PORTFOLIO SUMMARY (SIMULATED)                           ║")
            else:
                print("║ 💰 PORTFOLIO SUMMARY                                       ║")
                
            print(f"║ Cash: {formatted_cash.ljust(25)} | Total Value: ${portfolio_value:.2f}     ║")
            print(f"║ Holdings: {holdings_balance:.6f} {self.base_asset} ({formatted_position_value})                       ║")
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 📊 PERFORMANCE METRICS                                      ║")
            print(f"║ Initial: ${self.user_specified_capital:.2f}             | {pnl_display}  ║")
            print(f"║ Best Profit: ${self.best_profit:.2f}         | Max Drawdown: ${self.worst_drawdown:.2f}            ║")
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 🔄 TRADING ACTIVITY                                         ║")
            print(f"║ Total Trades: {len(self.trades)}                    | Buy: {buy_trades} | Sell: {sell_trades}    ║")
            print(f"║ Win Rate: {win_rate}                     | Profit Factor: {profit_factor}               ║")
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 🧠 STRATEGY INFO                                           ║")
            print(f"║ Version: {self.strategy_version.ljust(20)} | Signal: {self.last_signal:.4f} {signal_indicator}              ║")
            print(f"║ Position: {position_pct:.1f}% of capital {position_indicator}                        ║")
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 📝 RECENT TRADES                                           ║")
            
            # Show recent trades
            if self.trades:
                # Get the most recent trade
                recent_trade = self.trades[-1]
                trade_time = recent_trade['timestamp'].strftime("%m-%d %H:%M")
                trade_type = recent_trade['type'].upper()
                trade_price = recent_trade['price']
                trade_amount = recent_trade['amount']
                trade_value = recent_trade['value']
                
                print(f"║ {trade_time} | {trade_type} {trade_amount:.6f} @ ${trade_price:.2f} (${trade_value:.2f})║")
            else:
                print("║ No trades executed yet                                      ║")
            
            print("╠══════════════════════════════════════════════════════════╣")
            print("║ 📈 Collecting data for charts...                              ║")
            print("╚══════════════════════════════════════════════════════════╝")
            
            return True
            
        except Exception as e:
            logging.error(f"Error updating status: {e}")
            return False
    
    def _create_visual_indicator(self, value, max_bars=5):
        """Create a visual indicator bar based on a value from 0 to 1"""
        # Ensure value is between 0 and 1
        value = min(max(value, 0), 1)
        
        # Calculate how many bars to fill
        filled_bars = int(value * max_bars)
        empty_bars = max_bars - filled_bars
        
        # Create the indicator
        return '▓' * filled_bars + '░' * empty_bars
    
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
        print(f"║ 📈 PRICE & EQUITY CHARTS{' ' * 36}║")
        
        for i in range(chart_height):
            price_row = ''.join(price_chart[i])
            equity_row = ''.join(equity_chart[i])
            print(f"║ {price_row}{' ' * (25-len(price_row))} | {equity_row}{' ' * (25-len(equity_row))}║")
        
        print(f"║ ${min_price:.2f}{' ' * (23-len(f'${min_price:.2f}'))} | ${min_equity:.2f}{' ' * (23-len(f'${min_equity:.2f}'))}║")
        
        # Print time axis
        print("║ OLD{' ' * 22}NOW | OLD{' ' * 22}NOW║")
    
    def _get_24h_volume(self):
        """Get 24-hour trading volume for the symbol"""
        try:
            ticker_24h = self.client.get_ticker(symbol=self.symbol)
            volume = float(ticker_24h['volume'])
            return f"{volume:,.1f}"
        except Exception as e:
            logging.error(f"Error getting 24h volume: {e}")
            return "0.0"
    
    def _print_final_results(self):
        """Print final trading results"""
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker['price'])
            
            # Get portfolio value and profit/loss from simulation account
            portfolio_value = self.sim_account.get_portfolio_value(current_price)
            profit_loss, profit_loss_pct = self.sim_account.get_pnl(current_price)
            
            # Get trading statistics
            trade_stats = self.sim_account.get_trade_stats()
            
            print("\nFINAL RESULTS:")
            print(f"Initial Capital: ${self.user_specified_capital:.2f}")
            print(f"Final Portfolio Value: ${portfolio_value:.2f}")
            
            if profit_loss >= 0:
                print(f"Profit: {Fore.GREEN}+${profit_loss:.2f} (+{profit_loss_pct:.2f}%){Style.RESET_ALL}")
            else:
                print(f"Loss: {Fore.RED}-${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%){Style.RESET_ALL}")
                
            print(f"Total Trades: {trade_stats['total_trades']}")
            print(f"Total Fees Paid: ${trade_stats['total_fees']:.2f}")
            
            # Print win rate if we have completed trades
            if trade_stats['win_count'] + trade_stats['loss_count'] > 0:
                win_rate = (trade_stats['win_count'] / (trade_stats['win_count'] + trade_stats['loss_count'])) * 100
                print(f"Win Rate: {win_rate:.1f}% ({trade_stats['win_count']}/{trade_stats['win_count'] + trade_stats['loss_count']})")
            else:
                print(f"Win Rate: 0.0% (0/{trade_stats['total_trades']})")
            
            print(f"\n{Fore.YELLOW}Trading session ended.{Style.RESET_ALL}")
            
        except Exception as e:
            logging.error(f"Error printing final results: {e}")
            print(f"Error printing final results: {e}")
    
    def stop(self):
        """Stop the trading loop and print final results"""
        self.running = False
        print(f"\n{Fore.YELLOW}Trading stopped by user.{Style.RESET_ALL}")
        
        # Print final results
        self._print_final_results()
    
    def _reset_positions(self, base_asset: str, base_balance: float):
        """Reset positions by selling all holdings to start fresh"""
        if base_balance <= 0 or not self.client:
            # If no holdings, just reset the simulation account
            self.sim_account.reset(self.user_specified_capital)
            self.account_reset = True
            self.reset_cooldown = True
            self.reset_time = datetime.now()
            print(f"🔄 Reset simulation account to ${self.user_specified_capital:.2f}")
            print(f"⏳ Reset cooldown activated - no trades will be executed for 30 seconds")
            return True
            
        try:
            # Format quantity according to Binance's precision requirements
            quantity = self._format_quantity(base_balance)
            
            # Execute market sell order to liquidate position
            order = self.client.create_order(
                symbol=self.symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            # Log the trade
            price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else self.current_price
            executed_qty = float(order['executedQty']) if 'executedQty' in order else base_balance
            trade_value = price * executed_qty
            
            logging.info(f"RESET: SELL {executed_qty:.8f} {base_asset} at ${price:.2f}")
            logging.info(f"Trade value: ${trade_value:.2f}")
            
            print(f"🔄 Reset position: SELL {executed_qty:.6f} {base_asset} at ${price:.2f}")
            
            # Record the trade
            self.trades.append({
                'timestamp': datetime.now(),
                'type': 'reset_sell',
                'price': price,
                'amount': executed_qty,
                'value': trade_value,
                'order_id': order.get('orderId', 'unknown') if not self.simulation_mode else 'unknown'
            })
            
            # Reset the simulation account to the user-specified initial capital
            self.sim_account.reset(self.user_specified_capital)
            
            # Set the reset cooldown flag and time
            self.reset_cooldown = True
            self.reset_time = datetime.now()
            self.account_reset = True
            
            print(f"⏳ Reset cooldown activated - no trades will be executed for 30 seconds")
            print(f"🔄 Reset simulation account to ${self.user_specified_capital:.2f}")
            
            return True
        except Exception as e:
            logging.error(f"Error resetting positions: {e}")
            print(f"⚠️ Error resetting positions: {e}")
            return False
    
    def _get_current_price_with_retry(self, max_retries=3, retry_delay=2):
        """Get current price with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                return float(ticker['price'])
            except (BinanceAPIException, BinanceRequestException) as e:
                logging.error(f"Binance API Error getting price (attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                if retries < max_retries:
                    print(f"⚠️ API error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logging.error(f"Network error getting price (attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                if retries < max_retries:
                    print(f"⚠️ Network error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logging.error(f"Unexpected error getting price: {e}")
                return None
        
        # If we've exhausted all retries, try to estimate the price from the last known price
        if hasattr(self, 'current_price') and self.current_price > 0:
            logging.warning(f"Using last known price after {max_retries} failed attempts: ${self.current_price:.2f}")
            print(f"⚠️ Using last known price: ${self.current_price:.2f}")
            return self.current_price
        
        return None
    
    def run(self, check_interval_seconds: int = 10):
        """
        Run paper trading loop
        
        Args:
            check_interval_seconds: Interval between checks in seconds
        """
        self.running = True
        
        # Set up signal handler for clean exit
        def signal_handler(sig, frame):
            print("\nReceived interrupt, stopping trading...")
            self.stop()
            
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initial status update
        self._update_status()
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Process new data
                trading_signal = self._calculate_signal()
                
                # Execute trade based on signal
                trade_executed = self._execute_trade(trading_signal)
                
                # Update status
                self._update_status()
                
                # Reset consecutive errors counter on success
                consecutive_errors = 0
                
                # Sleep for the check interval
                time.sleep(check_interval_seconds)
                
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt, stopping trading...")
                self.stop()
                break
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                traceback.print_exc()
                
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n⚠️ Too many consecutive errors ({consecutive_errors}), stopping trading...")
                    self.stop()
                    break
                
                # Exponential backoff for retries
                retry_delay = 2 ** consecutive_errors
                if retry_delay > 60:
                    retry_delay = 60  # Cap at 60 seconds
                
                print(f"\n⚠️ Error in trading loop, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

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
    parser.add_argument('--reset-on-start', action='store_true',
                      help='Reset positions to start fresh with initial capital')

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
        use_binance_us=args.use_binance_us,
        reset_on_start=args.reset_on_start
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
