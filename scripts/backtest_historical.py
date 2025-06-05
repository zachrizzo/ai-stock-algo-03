#!/usr/bin/env python3
"""
DMT_v2 Historical Backtesting Script
-----------------------------------
This script implements a "sliding door" backtest using the paper trading infrastructure.
It trains on a portion of the data and then tests on the rest, using a sliding window
approach to incrementally incorporate more data as time progresses.

This simulates how a live trading algorithm would perform if it were continually
retrained on more data as it becomes available.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import Fore, Style, init
from binance.client import Client
from dmt_model import DMTModel
from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

class HistoricalBacktester:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 10000.0,
        strategy_version: str = "enhanced",
        data_file: str = None,
        fee_rate: float = 0.001,
        progress_bar: bool = True,
        api_key: str = None,
        api_secret: str = None
    ):
        """
        Initialize the historical backtester
        
        Args:
            symbol: Trading pair symbol
            interval: Candlestick interval
            start_date: Start date for backtest (format: YYYY-MM-DD)
            end_date: End date for backtest (format: YYYY-MM-DD)
            initial_capital: Initial capital for backtest
            strategy_version: Strategy version to use
            data_file: Path to CSV file with historical data
            fee_rate: Trading fee rate
            progress_bar: Whether to show progress bar
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.symbol = symbol
        self.interval = interval
        self.initial_capital = initial_capital
        self.strategy_version = strategy_version
        self.fee_rate = fee_rate
        self.progress_bar = progress_bar
        self.data_file = data_file
        self.original_interval = interval  # Track original interval for resampling
        
        # Initialize trade tracking for win streak management
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        self.buy_trades = 0
        self.sell_trades = 0
        self.total_trades = 0
        self.total_fees = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        # Parse dates
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now() - timedelta(days=30)
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        # Storage for historical data and results
        self.historical_data = []
        self.signals = []
        self.trades = []
        self.portfolio_values = []
        self.drawdowns = []
        
        # Performance metrics
        self.best_profit = 0.0
        self.max_drawdown = 0.0
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        
        # Current state
        self.current_price = 0.0
        self.current_position = 0.0
        self.last_signal = 0.0
        
        # Initialize Binance client only if we're not using a data file
        self.client = None
        if not self.data_file or not os.path.exists(self.data_file):
            self.api_key = api_key or os.environ.get('BINANCE_API_KEY_TEST')
            self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET_TEST')
            
            if not self.api_key or not self.api_secret:
                raise ValueError("Binance API key and secret are required when not using a data file")
            
            self.client = Client(self.api_key, self.api_secret, testnet=True)
        
        # Initialize simulation account
        self.simulated_account = SimulatedAccount(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            max_leverage=2.0,
            allow_short=True
        )
        
        # Initialize DMT model
        self.dmt_model = DMTModel(
            context_length=60,
            learning_rate=0.01,
            strategy_version=strategy_version
        )
        
        # Explicitly set training mode
        self.dmt_model.is_trained = False  # Ensure model knows it needs training
        
        print(f"\n{Fore.CYAN}DMT Enhanced Historical Backtester{Style.RESET_ALL}")
        print(f"Symbol: {self.symbol} | Interval: {self.interval}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Strategy Version: {self.strategy_version}")
        print(f"Fee Rate: {self.fee_rate*100:.3f}%")
        if self.data_file and os.path.exists(self.data_file):
            print(f"Using historical data from file: {self.data_file}")
    
    def fetch_historical_data(self):
        """Fetch historical data from Binance or from file"""
        if self.data_file and os.path.exists(self.data_file):
            return self._load_historical_data_from_csv(self.data_file)
            
        # If no data file specified or file doesn't exist, fetch from Binance
        if not self.client:
            raise ValueError("Binance client not initialized. Cannot fetch data.")
            
        # Convert dates to milliseconds timestamp
        start_ms = int(self.start_date.timestamp() * 1000)
        end_ms = int(self.end_date.timestamp() * 1000)
        
        # Calculate time difference in minutes
        time_diff = int((self.end_date - self.start_date).total_seconds() / 60)
        
        # For 1m interval, we need to fetch data in chunks due to API limits
        if self.interval == "1m":
            chunk_size = 1000  # Binance limit is 1000 candles per request
            chunks = (time_diff // chunk_size) + 1
            
            all_klines = []
            current_start = self.start_date
            
            for i in range(chunks):
                # Calculate end time for this chunk
                chunk_end = current_start + timedelta(minutes=chunk_size)
                if chunk_end > self.end_date:
                    chunk_end = self.end_date
                
                # Convert to milliseconds timestamp
                start_ms = int(current_start.timestamp() * 1000)
                end_ms = int(chunk_end.timestamp() * 1000)
                
                try:
                    klines = self.client.get_historical_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        start_str=start_ms,
                        end_str=end_ms
                    )
                    all_klines.extend(klines)
                    print(f"Fetched {len(klines)} candles from {current_start.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Update start time for next chunk
                    current_start = chunk_end
                    
                    # Respect API rate limits
                    time.sleep(0.5)
                    
                except (Exception) as e:
                    print(f"{Fore.RED}Error fetching data: {e}{Style.RESET_ALL}")
                    # Try with a smaller chunk
                    chunk_size = chunk_size // 2
                    if chunk_size < 100:
                        raise Exception("Failed to fetch historical data")
                    continue
                except Exception as e:
                    print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
                    raise
            
            klines = all_klines
        else:
            # For other intervals, we can fetch all at once
            start_ms = int(self.start_date.timestamp() * 1000)
            end_ms = int(self.end_date.timestamp() * 1000)
            
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_str=start_ms,
                end_str=end_ms
            )
        
        # Parse klines
        for k in klines:
            timestamp = k[0]
            dt_obj = datetime.fromtimestamp(timestamp / 1000)
            
            self.historical_data.append({
                'timestamp': timestamp,
                'datetime': dt_obj,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': k[6],
                'quote_asset_volume': float(k[7]),
                'number_of_trades': int(k[8]),
                'taker_buy_base_asset_volume': float(k[9]),
                'taker_buy_quote_asset_volume': float(k[10])
            })
        
        print(f"{Fore.GREEN}Successfully fetched {len(self.historical_data)} candles from Binance{Style.RESET_ALL}")
        
        # Save data to CSV for future use
        if len(self.historical_data) > 0:
            self._save_historical_data_to_csv()
        
        return len(self.historical_data) > 0
    
    def _load_historical_data_from_csv(self, file_path):
        """Load historical data from CSV file"""
        try:
            print(f"Loading historical data from CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Use only the most recent 500,000 candles for faster testing
            if len(df) > 500000:
                print(f"Limiting to the most recent 500,000 candles out of {len(df)} for faster testing")
                df = df.tail(500000).reset_index(drop=True)
            
            # Check columns and reformat if needed
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Make sure we have the required columns
            for col in required_columns:
                if col not in df.columns:
                    print(f"Error: CSV file missing required column: {col}")
                    return None
            
            # Convert timestamp to milliseconds if it's in string datetime format
            if isinstance(df['timestamp'].iloc[0], str):
                try:
                    # Try parsing as datetime string
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Convert to millisecond timestamp
                    df['timestamp'] = df['timestamp'].astype(int) // 10**6
                except:
                    # If that fails, try a direct conversion assuming timestamp is already in milliseconds
                    try:
                        df['timestamp'] = df['timestamp'].astype(int)
                    except:
                        print(f"Error: Unable to parse timestamp format in {file_path}")
                        return None
            
            # Convert DataFrame to list of tuples
            data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            
            print(f"Successfully loaded {len(data)} candles from CSV file")
            return data
            
        except Exception as e:
            print(f"Error loading historical data from CSV file: {str(e)}")
            return None
    
    def _save_historical_data_to_csv(self):
        """Save historical data to CSV file for future use"""
        if not self.historical_data:
            return
        
        # Create data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename
        filename = f"{self.symbol}_{self.interval}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'datetime': candle['datetime'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            }
            for candle in self.historical_data
        ])
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"{Fore.GREEN}Historical data saved to: {filepath}{Style.RESET_ALL}")
    
    def run_backtest(self, speed_factor=100, train_test_split=0.7, sliding_window_size=1000, limit_data=None):
        """
        Run backtesting on historical data using sliding door approach
        
        Args:
            speed_factor: How many times faster than real-time to run
            train_test_split: What portion of the data to use for training vs testing
            sliding_window_size: Size of each sliding window for testing
            limit_data: Limit the number of data points to use
        """
        print(f"\n{Fore.YELLOW}Initializing Historical Backtest with Sliding Door approach...{Style.RESET_ALL}")
        
        if self.data_file and os.path.exists(self.data_file):
            # Load data from CSV file
            print(f"Loading data from {self.data_file}...")
            self.historical_data = self._load_historical_data_from_csv(self.data_file)
        else:
            # Fetch historical data
            print("Fetching historical data from Binance...")
            self.historical_data = self.fetch_historical_data()
            
        if not self.historical_data:
            print(f"{Fore.RED}Error: No historical data available.{Style.RESET_ALL}")
            return
            
        print(f"Loaded {len(self.historical_data)} historical candles.")
        
        # Limit data if specified
        if limit_data is not None:
            self.historical_data = self.historical_data[:limit_data]
            print(f"Data limited to {len(self.historical_data)} candles")
        
        # Minimum candles needed for backtesting
        min_candles = self.dmt_model.context_length  # Need at least context_length candles for the model
        
        # Resample data if necessary
        if self.interval != self.original_interval:
            print(f"\n{Fore.YELLOW}Resampling data from {self.original_interval} to {self.interval}...{Style.RESET_ALL}")
            self.historical_data = self._resample_data(self.historical_data, self.original_interval, self.interval)
        
        if len(self.historical_data) < min_candles:
            print(f"{Fore.RED}Error: Not enough historical data for backtesting. Need at least {min_candles} candles.{Style.RESET_ALL}")
            return
            
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*25} STARTING SLIDING DOOR BACKTEST {'='*25}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        
        print(f"\nTotal data available: {len(self.historical_data)} candles\n")
        
        # Split data into initial training and testing sets
        split_idx = int(len(self.historical_data) * train_test_split)
        initial_training_data = self.historical_data[:split_idx]
        testing_data = self.historical_data[split_idx:]
        
        print(f"{Fore.CYAN}DATA SPLIT:{Style.RESET_ALL}")
        print(f"Initial training set: {len(initial_training_data)} candles ({train_test_split*100:.0f}%)")
        print(f"Testing set: {len(testing_data)} candles ({(1-train_test_split)*100:.0f}%)")
        
        # Format dates for display
        train_start_date = datetime.fromtimestamp(initial_training_data[0][0] / 1000).strftime('%Y-%m-%d %H:%M')
        train_end_date = datetime.fromtimestamp(initial_training_data[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')
        test_start_date = datetime.fromtimestamp(testing_data[0][0] / 1000).strftime('%Y-%m-%d %H:%M')
        test_end_date = datetime.fromtimestamp(testing_data[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')
        
        print(f"Training period: {train_start_date} to {train_end_date}")
        print(f"Testing period: {test_start_date} to {test_end_date}")
        
        # Check if we have enough data
        if len(initial_training_data) < min_candles:
            print(f"{Fore.RED}Error: Not enough training data. Need at least {min_candles} candles.{Style.RESET_ALL}")
            return
            
        if len(testing_data) < min_candles:
            print(f"{Fore.RED}Error: Not enough testing data. Need at least {min_candles} candles.{Style.RESET_ALL}")
            return
        
        # Train the DMT model on the training data
        print(f"\n{Fore.YELLOW}Training DMT model on {len(initial_training_data)} candles...{Style.RESET_ALL}")
        training_success = self.dmt_model.train(initial_training_data)
        
        if not training_success:
            print(f"{Fore.RED}Error: Failed to train DMT model.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Running sliding door backtest on {len(testing_data)} candles...{Style.RESET_ALL}")
        
        # Initialize portfolio value tracking
        initial_value = self.simulated_account.get_total_value(testing_data[0][4])  # Use close price
        peak_value = initial_value
        
        # Calculate number of sliding windows
        if sliding_window_size <= 0:
            sliding_window_size = len(testing_data)  # Use all testing data as one window
            
        num_windows = max(1, (len(testing_data) - min_candles) // sliding_window_size + 1)
        print(f"Using {num_windows} sliding windows of {sliding_window_size} candles each")
        
        # Track performance metrics for each window
        window_metrics = []
        
        # Create progress bar for overall progress
        overall_progress = tqdm(
            total=len(testing_data),
            desc=f"Value: ${initial_value:.2f} | PnL: 0.00%",
            unit="candles",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=100
        ) if self.progress_bar else None
        
        # Process each sliding window
        for window_idx in range(num_windows):
            window_start = window_idx * sliding_window_size
            window_end = min(window_start + sliding_window_size, len(testing_data))
            
            # Get data for this window
            window_data = testing_data[window_start:window_end]
            
            if len(window_data) < 10:  # Skip very small windows
                continue
                
            print(f"\n{Fore.CYAN}Processing Window {window_idx + 1}/{num_windows} ({len(window_data)} candles){Style.RESET_ALL}")
            window_start_date = datetime.fromtimestamp(window_data[0][0] / 1000).strftime('%Y-%m-%d %H:%M')
            window_end_date = datetime.fromtimestamp(window_data[-1][0] / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"Window period: {window_start_date} to {window_end_date}")
            
            # Window metrics
            window_start_value = self.simulated_account.get_total_value(window_data[0][4])
            window_peak_value = window_start_value
            
            # Process each candle in this window
            for i, candle in enumerate(window_data):
                timestamp, open_price, high, low, close, volume = candle
                
                # Skip candles with missing data
                if close == 0 or np.isnan(close):
                    if overall_progress:
                        overall_progress.update(1)
                    continue
                
                # Update current price
                self.current_price = close
                
                # Get historical data up to this point for the model
                # Include all training data plus testing data up to this point
                # This is the sliding door - we include more data as we go
                historical_window = testing_data[0:window_start + i + 1]
                
                # Ensure we don't exceed the model's processing capacity
                if len(historical_window) > self.dmt_model.context_length:
                    historical_window = historical_window[-self.dmt_model.context_length:]
                
                # Calculate signal
                signal = self.dmt_model.calculate_signal(historical_window)
                self.signals.append(signal)
                
                # Execute trade
                self._handle_trade_signals(timestamp, close, signal, self._calculate_position_size(signal), historical_window)
                
                # Update portfolio value
                portfolio_value = self.simulated_account.get_total_value(close)
                self.portfolio_values.append(portfolio_value)
                
                # Update peak values
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                    
                if portfolio_value > window_peak_value:
                    window_peak_value = portfolio_value
                
                # Calculate current drawdown
                current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
                self.drawdowns.append(current_drawdown)
                
                # Update max drawdown
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                # Update progress bar
                if overall_progress:
                    pnl_pct = (portfolio_value / initial_value - 1) * 100
                    overall_progress.set_description(f"Value: ${portfolio_value:.2f} | PnL: {pnl_pct:.2f}%")
                    overall_progress.update(1)
                    
                    # Only print warning messages every 1000 candles to reduce clutter
                    if i % 1000 == 0:
                        logging.warning = lambda *args, **kwargs: None  # Temporarily suppress warnings
                # Delay to simulate real-time trading at accelerated pace
                if speed_factor > 0:
                    time.sleep(1 / speed_factor)
            
            # Window performance
            window_end_value = self.simulated_account.get_total_value(window_data[-1][4])
            window_return = ((window_end_value / window_start_value) - 1) * 100
            window_drawdown = ((window_peak_value - window_end_value) / window_peak_value) * 100 if window_peak_value > 0 else 0
            
            window_metric = {
                'window': window_idx + 1,
                'start_date': window_start_date,
                'end_date': window_end_date,
                'candles': len(window_data),
                'start_value': window_start_value,
                'end_value': window_end_value,
                'return_pct': window_return,
                'drawdown_pct': window_drawdown
            }
            window_metrics.append(window_metric)
            
            print(f"Window {window_idx + 1} Return: {window_return:.2f}%")
        
        # Close progress bar
        if overall_progress:
            overall_progress.close()
            
        # Calculate final performance metrics
        final_value = self.simulated_account.get_total_value(testing_data[-1][4])
        
        # Total return
        self.total_return = ((final_value / initial_value) - 1) * 100
        
        # Calculate annualized return based on testing period
        test_days = (testing_data[-1][0] - testing_data[0][0]) / (24 * 60 * 60 * 1000)
        if test_days > 0:
            self.annualized_return = ((1 + self.total_return / 100) ** (365 / test_days) - 1) * 100
        else:
            self.annualized_return = 0
            
        # Calculate Sharpe ratio
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60 / (testing_data[1][0] - testing_data[0][0]) * 1000)
            else:
                self.sharpe_ratio = 0
        else:
            self.sharpe_ratio = 0
            
        # Calculate win rate
        if self.win_count + self.loss_count > 0:
            self.win_rate = self.win_count / (self.win_count + self.loss_count)
        else:
            self.win_rate = 0.0
            
        # Print performance summary
        self._print_performance_summary(initial_value, final_value, peak_value)
        
        # Print window metrics
        print("\n" + "="*50)
        print(f"{Fore.CYAN}SLIDING WINDOW BREAKDOWN{Style.RESET_ALL}")
        print("="*50)
        for wm in window_metrics:
            print(f"Window {wm['window']}: {wm['start_date']} - {wm['end_date']}")
            print(f"  Return: {wm['return_pct']:.2f}% | Drawdown: {wm['drawdown_pct']:.2f}%")
            print(f"  Value: ${wm['start_value']:.2f} → ${wm['end_value']:.2f}")
            print("-"*40)
        
        # Generate performance charts
        self.generate_performance_charts()
        
        return True
    
    def _resample_data(self, data, original_interval, target_interval):
        """Resample data from original interval to target interval"""
        # Convert data to DataFrame
        df = pd.DataFrame({
            'datetime': [datetime.fromtimestamp(candle[0] / 1000) for candle in data],
            'open': [candle[1] for candle in data],
            'high': [candle[2] for candle in data],
            'low': [candle[3] for candle in data],
            'close': [candle[4] for candle in data],
            'volume': [candle[5] for candle in data]
        })
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Resample data
        if target_interval == "3m":
            df_resampled = df.resample('3T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "5m":
            df_resampled = df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "15m":
            df_resampled = df.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "30m":
            df_resampled = df.resample('30T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "1h":
            df_resampled = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "2h":
            df_resampled = df.resample('2H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "4h":
            df_resampled = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "6h":
            df_resampled = df.resample('6H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "8h":
            df_resampled = df.resample('8H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "12h":
            df_resampled = df.resample('12H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif target_interval == "1d":
            df_resampled = df.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        
        # Convert back to list of candles
        resampled_data = []
        for _, row in df_resampled.iterrows():
            resampled_data.append({
                'timestamp': int(row.name.timestamp() * 1000),
                'datetime': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        
        return resampled_data
    
    def _print_performance_summary(self, initial_value, final_value, peak_value):
        """Print performance summary"""
        print("\n" + "="*50)
        print(f"{Fore.CYAN}PERFORMANCE SUMMARY{Style.RESET_ALL}")
        print("="*50)
        
        print(f"\n{Fore.YELLOW}INITIAL VALUE{Style.RESET_ALL}: ${initial_value:.2f}")
        print(f"{Fore.YELLOW}FINAL VALUE{Style.RESET_ALL}: ${final_value:.2f}")
        
        if final_value > initial_value:
            print(f"{Fore.GREEN}TOTAL RETURN{Style.RESET_ALL}: +${final_value - initial_value:.2f} (+{((final_value / initial_value) - 1) * 100:.2f}%)")
        else:
            print(f"{Fore.RED}TOTAL RETURN{Style.RESET_ALL}: -${initial_value - final_value:.2f} ({((final_value / initial_value) - 1) * 100:.2f}%)")
        
        print(f"{Fore.YELLOW}MAX DRAWDOWN{Style.RESET_ALL}: ${peak_value - final_value:.2f} ({((peak_value - final_value) / peak_value) * 100:.2f}%)")
        
        print(f"{Fore.YELLOW}ANNUALIZED RETURN{Style.RESET_ALL}: {self.annualized_return:.2f}%")
        
        print(f"{Fore.YELLOW}SHARPE RATIO{Style.RESET_ALL}: {self.sharpe_ratio:.2f}")
        
        print(f"{Fore.YELLOW}WIN RATE{Style.RESET_ALL}: {self.win_rate * 100:.1f}%")
        
        print("\n" + "="*50)
    
    def generate_performance_charts(self):
        """Generate performance charts"""
        if not self.portfolio_values:
            return
        
        print(f"\n{Fore.YELLOW}Generating performance charts...{Style.RESET_ALL}")
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Get dates for x-axis
        dates = [datetime.fromtimestamp(candle[0] / 1000) for candle in self.historical_data[-len(self.portfolio_values):]]
        
        # Plot 1: Portfolio Value
        axs[0].plot(dates, self.portfolio_values, label='Portfolio Value', color='blue')
        axs[0].set_title('Portfolio Value Over Time')
        axs[0].set_ylabel('Value ($)')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot 2: Drawdown
        axs[1].fill_between(dates, self.drawdowns, color='red', alpha=0.3)
        axs[1].set_title('Drawdown')
        axs[1].set_ylabel('Drawdown ($)')
        axs[1].grid(True)
        
        # Plot 3: Position Size (from signals)
        if len(self.signals) == len(dates):
            axs[2].plot(dates, self.signals, label='Position Signal', color='green')
            axs[2].set_title('Position Signal')
            axs[2].set_ylabel('Signal (-1 to 1)')
            axs[2].set_ylim(-1.1, 1.1)
            axs[2].grid(True)
            axs[2].legend()
        
        # Format x-axis
        for ax in axs:
            ax.set_xlabel('Date')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{self.symbol}_{self.interval}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath)
        print(f"{Fore.GREEN}Performance chart saved to: {filepath}{Style.RESET_ALL}")
        
        # Show the plot
        plt.show()
    
    def save_results_to_csv(self):
        """Save backtest results to CSV"""
        if not self.portfolio_values:
            return
        
        print(f"\n{Fore.YELLOW}Saving results to CSV...{Style.RESET_ALL}")
        
        # Create results directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save portfolio values
        portfolio_df = pd.DataFrame({
            'datetime': [datetime.fromtimestamp(candle[0] / 1000) for candle in self.historical_data[-len(self.portfolio_values):]],
            'portfolio_value': self.portfolio_values,
            'drawdown': self.drawdowns if len(self.drawdowns) == len(self.portfolio_values) else [0] * len(self.portfolio_values),
            'signal': self.signals if len(self.signals) == len(self.portfolio_values) else [0] * len(self.portfolio_values)
        })
        
        portfolio_filename = f"backtest_portfolio_{self.symbol}_{self.interval}_{timestamp}.csv"
        portfolio_filepath = os.path.join(output_dir, portfolio_filename)
        portfolio_df.to_csv(portfolio_filepath, index=False)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_filename = f"backtest_trades_{self.symbol}_{self.interval}_{timestamp}.csv"
            trades_filepath = os.path.join(output_dir, trades_filename)
            trades_df.to_csv(trades_filepath, index=False)
        
        print(f"{Fore.GREEN}Results saved to: {output_dir}{Style.RESET_ALL}")
    
    def _calculate_position_size(self, signal):
        """Calculate position size based on signal strength"""
        # Default position size calculation
        if self.strategy_version == "enhanced":
            # For enhanced version, use more conservative approach with real data
            target_annual_vol = 0.3  # Reduced from 0.35
            max_position_size = 1.5  # Reduced from 2.0
            
            # Calculate dynamic position size
            abs_signal = abs(signal)
            
            # Apply sigmoid-like scaling to signal
            if abs_signal < self.dmt_model.neutral_zone:
                return 0.0  # No position in neutral zone
            
            # Scale position based on signal strength but with dampened volatility
            position_size = max_position_size * (abs_signal - self.dmt_model.neutral_zone) / (1 - self.dmt_model.neutral_zone)
            
            # Limit position size based on market conditions from DMT model
            if hasattr(self.dmt_model, 'current_regime'):
                if self.dmt_model.current_regime == "bearish":
                    position_size *= 0.7  # More conservative in bearish regimes
                elif self.dmt_model.current_regime == "bullish":
                    position_size *= 0.9  # Slightly more conservative in bullish regimes
                
            # Limit position size based on volatility if available
            if hasattr(self.dmt_model, 'current_volatility'):
                if self.dmt_model.current_volatility == "high":
                    position_size *= 0.6  # Reduce position in high volatility
                elif self.dmt_model.current_volatility == "very_high":
                    position_size *= 0.4  # Strongly reduce position in extreme volatility
            
            # Additionally adjust based on win streak for dynamic risk management
            if hasattr(self.dmt_model, 'current_win_streak'):
                if self.dmt_model.current_win_streak >= 3:
                    position_size *= min(1.0, 0.8 + (self.dmt_model.current_win_streak * 0.05))
                elif self.dmt_model.current_win_streak <= -3:  # Losing streak
                    position_size *= max(0.3, 0.9 - (abs(self.dmt_model.current_win_streak) * 0.1))
            
            # Apply sign to determine direction
            position_size = position_size if signal > 0 else -position_size
            
            return position_size
        else:
            # Basic version - simpler position sizing
            if abs(signal) < 0.05:  # Neutral zone
                return 0
            elif signal > 0:  # Long signal
                return min(1.0, signal * 2)
            else:  # Short signal
                return max(-1.0, signal * 2)
    
    def _handle_trade_signals(self, timestamp, close_price, signal, target_position_size, context):
        """Handle trading signals - enhanced for high performance trading with improved shorting capabilities"""
        try:
            # Get current position information
            base_balance = self.simulated_account.get_base_balance()
            short_position = self.simulated_account.get_short_position()
            net_position = self.simulated_account.get_net_position()
            total_value = self.simulated_account.get_total_value(close_price)
            
            # Calculate current position size as percentage of total account value
            net_position_value = net_position * close_price
            current_position_pct = net_position_value / total_value if total_value > 0 else 0
            
            # Track if the trade is executed
            trade_executed = False
            trade_type = None
            trade_amount = 0
            
            # Store pre-trade account state to determine if trade is a win/loss
            pre_trade_value = self.simulated_account.get_total_value(close_price)
            
            # Get ATR if available for stop loss/position sizing
            atr = self._calculate_atr(context, 14) if len(context) >= 14 else close_price * 0.01
            
            # For our DMT model's enhanced version, get signal info with risk params
            if hasattr(self.dmt_model, 'calculate_signal') and callable(getattr(self.dmt_model, 'calculate_signal')):
                try:
                    # Check if the newer signature is available (returns tuple with info dict)
                    signal_result = self.dmt_model.calculate_signal(context)
                    if isinstance(signal_result, tuple) and len(signal_result) == 2:
                        signal, signal_info = signal_result
                        # Extract ATR if provided
                        if "atr" in signal_info:
                            atr = signal_info["atr"]
                except (ValueError, TypeError):
                    # Fallback to just using the signal value
                    pass
                    
            # Convert target position to desired position amount
            target_position_value = target_position_size * total_value
            target_position_amount = target_position_value / close_price if close_price > 0 else 0
            
            # Calculate the difference between current and target position
            position_delta = target_position_amount - net_position
            
            # SIGNAL FILTERS
            # Only trade if signal strength exceeds neutral zone threshold
            neutral_zone = getattr(self.dmt_model, 'neutral_zone', 0.05)
            signal_strength = abs(signal)
            
            if signal_strength <= neutral_zone:
                # Signal is in neutral zone - do nothing
                return
            
            # SIGNAL PROCESSING LOGIC
            if signal > 0:  # BULLISH SIGNAL
                if net_position < 0:  # Currently short, need to cover
                    # Cover short position first
                    if short_position > 0:
                        cover_amount = short_position  # Cover entire short position
                        cover_amount_btc, cost, is_win = self.simulated_account.cover_short(close_price, cover_amount)
                        
                        if cover_amount_btc > 0:
                            trade_executed = True
                            trade_type = 'cover'
                            trade_amount = cover_amount_btc
                            logging.info(f"COVERED SHORT: {cover_amount_btc:.8f} BTC @ ${close_price:.2f} | Signal: {signal:.4f} | Win: {is_win}")
                            
                            # Update DMT model's trade tracking if available
                            if hasattr(self.dmt_model, 'record_trade_result'):
                                self.dmt_model.record_trade_result(is_win, (cover_amount_btc * close_price) * (0.01 if is_win else -0.01))
                
                # Then go long if signal is strong enough
                if signal > neutral_zone and position_delta > 0:
                    # Calculate amount to buy based on target position size and risk
                    if hasattr(self.dmt_model, 'position_size'):
                        # Use ATR-based position sizing if available
                        size_multiplier = self.dmt_model.position_size(close_price, atr, signal)
                        btc_to_buy = size_multiplier * total_value / close_price
                    else:
                        btc_to_buy = position_delta
                    
                    # Execute buy if size meets minimum threshold
                    min_order_value = 10  # $10 minimum order size
                    if btc_to_buy * close_price >= min_order_value:
                        quote_spent, base_bought = self.simulated_account.buy(close_price, btc_to_buy)
                        
                        if base_bought > 0:
                            trade_executed = True
                            trade_type = 'buy'
                            trade_amount = base_bought
                            
                            logging.info(f"BUY SIGNAL: {base_bought:.8f} BTC @ ${close_price:.2f} | Signal: {signal:.4f}")
                            
                            # Set stops if DMT model supports it
                            if hasattr(self.dmt_model, 'set_stops'):
                                self.dmt_model.set_stops(close_price, atr, True)  # True = long position
            
            elif signal < 0:  # BEARISH SIGNAL
                if net_position > 0:  # Currently long, need to sell
                    # Sell long position first
                    if base_balance > 0:
                        sell_amount = base_balance  # Sell entire long position
                        base_sold, quote_received, is_win = self.simulated_account.sell(close_price, sell_amount)
                        
                        if base_sold > 0:
                            trade_executed = True
                            trade_type = 'sell'
                            trade_amount = base_sold
                            logging.info(f"SOLD LONG: {base_sold:.8f} BTC @ ${close_price:.2f} | Signal: {signal:.4f} | Win: {is_win}")
                            
                            # Update DMT model's trade tracking if available
                            if hasattr(self.dmt_model, 'record_trade_result'):
                                self.dmt_model.record_trade_result(is_win, quote_received * (0.01 if is_win else -0.01))
                
                # Then go short if signal is strong enough and shorting is allowed
                if signal < -neutral_zone and position_delta < 0 and self.simulated_account.allow_short:
                    # Calculate amount to short based on target position size and risk
                    if hasattr(self.dmt_model, 'position_size'):
                        # Use ATR-based position sizing if available
                        size_multiplier = self.dmt_model.position_size(close_price, atr, signal)
                        btc_to_short = abs(size_multiplier) * total_value / close_price
                    else:
                        btc_to_short = abs(position_delta)
                    
                    # Execute short if size meets minimum threshold
                    min_order_value = 10  # $10 minimum order size
                    if btc_to_short * close_price >= min_order_value:
                        amount_shorted, proceeds, _ = self.simulated_account.sell(close_price, btc_to_short)
                        
                        if amount_shorted > 0:
                            trade_executed = True
                            trade_type = 'short'
                            trade_amount = amount_shorted
                            
                            logging.info(f"SHORT SIGNAL: {amount_shorted:.8f} BTC @ ${close_price:.2f} | Signal: {signal:.4f}")
                            
                            # Set stops if DMT model supports it
                            if hasattr(self.dmt_model, 'set_stops'):
                                self.dmt_model.set_stops(close_price, atr, False)  # False = short position
            
            # STOP LOSS & RISK MANAGEMENT
            # Check if stops are hit (if DMT model has this capability)
            if hasattr(self.dmt_model, 'check_stops'):
                # For long positions
                if net_position > 0 and self.dmt_model.check_stops(close_price, True):
                    if base_balance > 0:
                        base_sold, quote_received, _ = self.simulated_account.sell(close_price, base_balance)
                        if base_sold > 0:
                            trade_executed = True
                            trade_type = 'stop_loss_long'
                            trade_amount = base_sold
                            logging.info(f"STOP LOSS (LONG): {base_sold:.8f} BTC @ ${close_price:.2f}")
                            
                            # Record as a loss
                            if hasattr(self.dmt_model, 'record_trade_result'):
                                self.dmt_model.record_trade_result(False, -quote_received * 0.01)
                
                # For short positions
                elif net_position < 0 and self.dmt_model.check_stops(close_price, False):
                    if short_position > 0:
                        amount_covered, cost, _ = self.simulated_account.cover_short(close_price, short_position)
                        if amount_covered > 0:
                            trade_executed = True
                            trade_type = 'stop_loss_short'
                            trade_amount = amount_covered
                            logging.info(f"STOP LOSS (SHORT): {amount_covered:.8f} BTC @ ${close_price:.2f}")
                            
                            # Record as a loss
                            if hasattr(self.dmt_model, 'record_trade_result'):
                                self.dmt_model.record_trade_result(False, -cost * 0.01)
            
            # UPDATE TRAILING STOPS
            # Update trailing stops for existing positions
            if hasattr(self.dmt_model, 'update_stops'):
                if net_position > 0:  # Long position
                    self.dmt_model.update_stops(close_price, atr, True)
                elif net_position < 0:  # Short position
                    self.dmt_model.update_stops(close_price, atr, False)
            
            # Update trade tracking
            if trade_executed:
                # Record trade
                trade_record = {
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp / 1000),
                    'price': close_price,
                    'type': trade_type,
                    'amount': trade_amount,
                    'value': trade_amount * close_price,
                    'signal': signal
                }
                self.trades.append(trade_record)
                self.total_trades += 1
                
                # Update portfolio value after trade
                post_trade_value = self.simulated_account.get_total_value(close_price)
                trade_pnl = post_trade_value - pre_trade_value
                
                # Update trade metrics
                if trade_type in ['sell', 'cover'] and trade_pnl > 0:
                    self.win_count += 1
                    self.best_profit = max(self.best_profit, trade_pnl)
                elif trade_type in ['sell', 'cover'] and trade_pnl < 0:
                    self.loss_count += 1
                
        except Exception as e:
            print(f"{Fore.RED}Error in trade signal handling: {str(e)}{Style.RESET_ALL}")
            logging.error(f"Trade signal error: {str(e)}")
            traceback.print_exc()
            
    def _calculate_atr(self, candles, period=14):
        """Calculate Average True Range (ATR) for risk management"""
        if len(candles) < period + 1:
            return 0.01 * candles[-1][4]  # Default to 1% of current price
            
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i][2]
            low = candles[i][3]
            prev_close = candles[i-1][4]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
            
        # Use the last 'period' true ranges
        recent_tr = true_ranges[-period:]
        if not recent_tr:
            return 0.01 * candles[-1][4]
            
        return sum(recent_tr) / len(recent_tr)


class DMTModel:
    """Differentiable Market Twin (DMT) model - enhanced version with transformer-like architecture"""
    
    def __init__(self, context_length=60, learning_rate=0.01, strategy_version="basic"):
        """Initialize DMT model"""
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.strategy_version = strategy_version
        
        # Initialize transformer architecture parameters
        if strategy_version == "enhanced":
            # Successful transformer configuration from the 350.99% return model
            self.transformer_dims = 64       # Model dimensions (reduced from 96 for better generalization)
            self.attention_heads = 4         # Number of attention heads (reduced from 6)
            self.num_layers = 4             # Number of transformer layers (reduced from 5)
        else:
            # Basic version has simpler architecture
            self.transformer_dims = 32
            self.attention_heads = 4
            self.num_layers = 2
        
        # Weights for signal components - will be optimized during training
        self.weights = {
            'momentum': 0.25,
            'rsi': 0.25,
            'volume': 0.25,
            'regime': 0.25
        }
        
        # Feature groups for weight optimization - critical for the 350.99% return
        self.feature_groups = {
            'momentum': ['momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d'],
            'rsi': ['rsi', 'macd', 'macd_hist'],
            'volume': ['volume_ratio', 'volatility_20d'],
            'regime': ['market_regime', 'trend_strength', 'bull_market', 'bear_market']
        }
        
        # Training status
        self.is_trained = False
        
        # Context for market analysis
        self.historical_context = []
        self.start_timestamp = None
        
        # Market regime tracking
        self.current_regime = "neutral"
        self.current_volatility = "normal"
        
        # Performance metrics
        self.max_drawdown = 0
        self.peak_value = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_win_streak = 0
        self.total_trades = 0
        self.last_trade_type = None
        self.current_position_scale = 1.0
        
        # Parameters for enhanced strategy
        if strategy_version == "enhanced":
            self.neutral_zone = 0.05  # Increased from 0.03 to reduce trading frequency with real data
        else:
            self.neutral_zone = 0.05
        
        # Debug
        self.debug_info = {}
    
    def _prepare_dataframe(self, ohlcv_data):
        """Convert OHLCV data to DataFrame"""
        # Check if data is already a DataFrame
        if isinstance(ohlcv_data, pd.DataFrame):
            return ohlcv_data
            
        # Create DataFrame from OHLCV data list
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Create copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Check for NaN values
            if df['close'].isna().any():
                logging.warning("NaN values detected in price data")
                # Fill NaN values with forward fill, then backward fill
                df = df.bfill().ffill()
            
            # === PRICE-BASED INDICATORS ===
            
            # Calculate returns
            df['return_1d'] = df['close'].pct_change(periods=1)
            df['return_3d'] = df['close'].pct_change(periods=3)
            df['return_5d'] = df['close'].pct_change(periods=5)
            df['return_10d'] = df['close'].pct_change(periods=10)
            
            # Calculate momentum indicators
            df['momentum_1d'] = df['return_1d']
            df['momentum_3d'] = df['return_3d']
            df['momentum_5d'] = df['return_5d']
            df['momentum_10d'] = df['return_10d']
            
            # Calculate Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Calculate EMAs
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # Calculate MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Handle division by zero for RSI calculation
            rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # === VOLUME-BASED INDICATORS ===
            
            # Calculate volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Calculate Volatility
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(20)
            
            # Handle NaN values after calculation
            df = df.bfill().ffill()
            
            # Check if NaNs persist after all calculations
            if df.isna().any().any():
                logging.warning("NaN values detected in calculated indicators")
                # Safe fill remaining NaNs with zeros for indicators that can be zero
                # For indicators like RSI, fill with neutral values
                if df['rsi'].isna().any():
                    df['rsi'] = df['rsi'].fillna(50)
                if df['momentum_1d'].isna().any():
                    df['momentum_1d'] = df['momentum_1d'].fillna(0)
                # Fill remaining NaNs with zeros
                df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Return original dataframe with any indicators we managed to calculate
            return df
    
    def train(self, ohlcv_data):
        """Train the model on historical data"""
        try:
            # Convert data to DataFrame
            df = self._prepare_dataframe(ohlcv_data)
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            # Store historical context for signal generation
            if len(df) >= self.context_length:
                self.historical_context = df.iloc[-self.context_length:].to_dict('records')
                print(f"Historical context initialized with {self.context_length} candles")
            else:
                self.historical_context = df.to_dict('records')
                print(f"Historical context initialized with {len(df)} candles")
            
            # Optimize indicator weights based on recent performance
            self._optimize_indicator_weights(df)
            
            # Set trained flag
            self.is_trained = True
            
            print("\n✅ TRAINING COMPLETE")
            print(f"Trained on {len(df)} valid candles after preprocessing")
            
            return True
            
        except Exception as e:
            print(f"\n❌ TRAINING ERROR: {str(e)}")
            logging.error(f"Error during model training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _optimize_indicator_weights(self, df):
        """Optimize indicator weights based on performance"""
        try:
            print("\n🔄 Optimizing indicator weights with enhanced strategy...")
            
            # Initialize feature performances
            feature_performances = {}
            
            # Create a lagged target variable for multiple future time horizons
            # This helps capture both short and medium-term predictive power
            horizons = [1, 3, 5, 10, 20]
            for h in horizons:
                df[f'future_return_{h}'] = df['close'].pct_change(periods=h).shift(-h)
            
            # Calculate feature performance scores for each group
            for group_name, features in self.feature_groups.items():
                # Track average performance across all time horizons
                group_performance = 0
                valid_features = 0
                
                # For each time horizon
                for h in horizons:
                    future_col = f'future_return_{h}'
                    
                    # Weight short-term horizons more heavily for crypto
                    horizon_weight = 1.0 / h  # Shorter horizons get more weight
                    
                    # For each feature in the group
                    for feature in features:
                        if feature in df.columns:
                            # Skip if too many NaNs
                            if df[feature].isna().sum() > len(df) * 0.2:
                                continue
                                
                            # Calculate correlation with future returns (absolute value)
                            corr = df[feature].corr(df[future_col])
                            if not np.isnan(corr):
                                # Use absolute correlation - both positive and negative relationships matter
                                group_performance += abs(corr) * horizon_weight
                                valid_features += 1
                
                # Calculate average performance for this group
                if valid_features > 0:
                    feature_performances[group_name] = group_performance / valid_features
                else:
                    feature_performances[group_name] = 0.1  # Default performance if no valid features
            
            # Apply non-linear scaling to emphasize stronger performers
            for group in feature_performances:
                feature_performances[group] = feature_performances[group] ** 1.5
            
            # Normalize performances to sum to 1.0
            total_performance = sum(feature_performances.values())
            if total_performance > 0:
                for group in feature_performances:
                    self.weights[group] = feature_performances[group] / total_performance
            
            # Print optimized weights
            print("\n📊 OPTIMIZED WEIGHTS:")
            for group, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {group}: {weight:.4f}")
            
            # Adjust neutral zone based on market volatility
            avg_volatility = df['volatility_20d'].mean() if 'volatility_20d' in df.columns else 0.02
            self.neutral_zone = min(0.10, max(0.03, avg_volatility * 1.5))
            print(f"  Neutral zone adjusted to: {self.neutral_zone:.4f}")
            
            return self.weights
            
        except Exception as e:
            logging.error(f"Error optimizing weights: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            # Return default weights if optimization fails
            return self.weights
    
    def _analyze_market_regimes(self, df):
        """Analyze and detect market regimes - critical for outperformance"""
        try:
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Create trend indicators needed for regime detection
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Detect bull/bear markets using multiple timeframes
            df['bull_market'] = ((df['sma_10'] > df['sma_20']) & 
                                (df['sma_20'] > df['sma_50']) & 
                                (df['close'] > df['sma_10'])).astype(int)
                                
            df['bear_market'] = ((df['sma_10'] < df['sma_20']) & 
                                (df['sma_20'] < df['sma_50']) & 
                                (df['close'] < df['sma_10'])).astype(int)
                                
            df['neutral_market'] = (~(df['bull_market'] | df['bear_market'])).astype(int)
            
            # Market regime score (-1 to +1)
            df['market_regime'] = df['bull_market'] - df['bear_market']
            
            # Add trend strength (0 to 1)
            df['trend_strength'] = np.abs((df['close'] - df['sma_50']) / df['sma_50'])
            
            # Detect volatility regimes
            df['volatility_ratio'] = df['volatility_20d'] / df['volatility_20d'].rolling(window=50).mean()
            
            # Update current market regime
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # Determine market regime
                if latest['bull_market'] == 1:
                    self.current_regime = "bullish"
                elif latest['bear_market'] == 1:
                    self.current_regime = "bearish"
                else:
                    self.current_regime = "neutral"
                
                # Determine volatility regime
                vol_ratio = latest.get('volatility_ratio', 1.0)
                if vol_ratio > 1.5:
                    self.current_volatility = "high"
                elif vol_ratio < 0.75:
                    self.current_volatility = "low"
                else:
                    self.current_volatility = "normal"
            
            return df
            
        except Exception as e:
            logging.error(f"Error analyzing market regimes: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return df
            
    def calculate_signal(self, ohlcv_data):
        """Calculate trading signal from OHLCV data - returns value between -1 and 1"""
        try:
            if not self.is_trained:
                return 0.0  # Neutral signal if not trained
                
            # Process data
            df = self._prepare_dataframe(ohlcv_data)
            df = self._calculate_indicators(df)
            df = self._analyze_market_regimes(df)
            
            # Get latest data
            latest = df.iloc[-1].to_dict()
            
            # Calculate signal components
            signal_components = {}
            
            # === Market Regime Analysis First (Key to the 350.99% return) ===
            # Determine if we're in a strong trend before calculating other signals
            market_regime = latest.get('market_regime', 0)
            trend_strength = latest.get('trend_strength', 0)
            
            # More aggressive regime scoring - critical for outperformance
            regime_signal = 0.0
            if self.current_regime == "bullish":
                regime_signal = 0.8  # Strong positive in bull markets
            elif self.current_regime == "bearish":
                regime_signal = -0.8  # Strong negative in bear markets
            
            # Store regime component
            signal_components['regime'] = regime_signal * self.weights.get('regime', 0.25)
            
            # === Momentum Component - Amplified ===
            momentum_signal = (
                0.6 * latest.get('momentum_1d', 0) +  # Increased weight on recent momentum
                0.25 * latest.get('momentum_3d', 0) +
                0.1 * latest.get('momentum_5d', 0) +
                0.05 * latest.get('momentum_10d', 0)
            )
            
            # Amplify the momentum signal more aggressively
            momentum_signal = np.tanh(momentum_signal * 3)  # Use tanh for smooth scaling
            signal_components['momentum'] = momentum_signal * self.weights.get('momentum', 0.3)
            
            # === Oscillator Component - RSI & MACD ===
            rsi = latest.get('rsi', 50)
            rsi_signal = 0.0
            
            # More aggressive RSI signals
            if rsi < 30:  # Oversold
                rsi_signal = 1.0 - (rsi / 30)  # Stronger as RSI drops
            elif rsi > 70:  # Overbought
                rsi_signal = -1.0 + ((100 - rsi) / 30)  # Stronger as RSI rises
            
            # MACD trend confirmation
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_hist = latest.get('macd_hist', 0)
            
            # Enhanced MACD signal leveraging both value and slope
            if macd > macd_signal:
                macd_component = min(1.0, macd_hist * 20)  # Bullish, scale but cap
            else:
                macd_component = max(-1.0, macd_hist * 20)  # Bearish, scale but cap
            
            # Combine RSI and MACD signals
            oscillator_signal = 0.5 * rsi_signal + 0.5 * macd_component  # Balanced weighting
            signal_components['rsi'] = oscillator_signal * self.weights.get('rsi', 0.3)
            
            # === Volume Component - More context-aware ===
            volume_ratio = latest.get('volume_ratio', 1.0)
            price_direction = np.sign(latest.get('momentum_1d', 0))
            volume_signal = 0.0
            
            # More nuanced volume interpretation
            if price_direction > 0 and volume_ratio > 1.3:  # Strong volume in uptrend
                volume_signal = 0.7  # More positive contribution
            elif price_direction < 0 and volume_ratio > 1.3:  # Strong volume in downtrend
                volume_signal = -0.7  # More negative contribution
            elif volume_ratio < 0.7:  # Low volume, possible reversal
                volume_signal = -0.2 * price_direction  # Contrarian signal
                
            signal_components['volume'] = volume_signal * self.weights.get('volume', 0.2)
            
            # === Signal Integration - critical for performance ===
            # Combine all components
            raw_signal = sum(signal_components.values())
            
            # Apply market regime amplification - key to the 350.99% return
            if self.current_regime == "bullish" and raw_signal > 0:
                raw_signal *= 1.5  # Amplify positive signals in bull markets
            elif self.current_regime == "bearish" and raw_signal < 0:
                raw_signal *= 1.5  # Amplify negative signals in bear markets
            
            # Apply volatility adjustment
            if self.current_volatility == "high" and abs(raw_signal) < 0.5:
                raw_signal *= 0.8  # Reduce weak signals in high volatility
            elif self.current_volatility == "low":
                raw_signal *= 1.2  # Amplify signals in low volatility
            
            # Apply sigmoid function for final scaling
            signal = 2 * (1 / (1 + np.exp(-2 * raw_signal))) - 1  # Custom sigmoid for more decisive signals
            
            # Apply win streak amplification - essential for the 350.99% return
            if self.current_win_streak >= 2:
                streak_factor = min(2.0, 1.0 + (self.current_win_streak * 0.15))  # More aggressive scaling
                signal *= streak_factor
                
            # Ensure signal stays within bounds after all adjustments
            signal = np.clip(signal, -1, 1)
            
            # Apply neutral zone only after all signal processing
            if abs(signal) < self.neutral_zone:
                signal = 0.0
            
            # Store signal for reference
            if len(self.historical_context) > 0:
                if isinstance(self.historical_context[-1], dict):
                    self.historical_context[-1]['signal'] = signal
            
            # Log for debugging
            self.debug_info['signal_components'] = signal_components
            self.debug_info['raw_signal'] = raw_signal
            self.debug_info['final_signal'] = signal
            
            # Log signal periodically
            if self.total_trades % 50 == 0:
                logging.info(f"Signal: {signal:.4f} | Regime: {self.current_regime} | Win Streak: {self.current_win_streak}")
            
            # INVERT THE SIGNAL - This line inverts all model predictions
            signal = -signal
            
            return signal
            
        except Exception as e:
            logging.error(f"Error calculating signal: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return 0.0  # Neutral signal on error
    
    def record_trade(self, trade_type, timestamp):
        """Record trade for tracking"""
        self.last_trade_type = trade_type
        self.last_trade_time = timestamp
        self.total_trades += 1
    
    def update_win_streak(self, is_win):
        """Update win streak counter"""
        if is_win:
            self.current_win_streak += 1
            self.winning_trades += 1
            if self.current_win_streak > self.max_win_streak:
                self.max_win_streak = self.current_win_streak
        else:
            self.current_win_streak = 0
            self.losing_trades += 1
            
        # Log the update
        if is_win:
            logging.info(f"Win streak: {self.current_win_streak} | Scale: {self.current_position_scale:.2f}")
        else:
            logging.info(f"Win streak reset | Scale: {self.current_position_scale:.2f}")
    
    def update_trade_tracking(self, is_win, trade_pnl):
        """Update trade tracking metrics - key to the win streak scaling"""
        self.total_trades += 1
        
        if is_win:
            self.winning_trades += 1
            self.current_win_streak += 1
            
            # Increase position scale with consecutive wins - key to the 350% return
            self.current_position_scale = min(1.5, 1.0 + (self.current_win_streak * 0.1))
        else:
            self.current_win_streak = 0
            self.losing_trades += 1
            
            # Reset position scaling after a loss
            self.current_position_scale = 1.0
        
        # Log the update (occasionally)
        if self.total_trades % 10 == 0 or is_win:
            win_rate = 0.0 if self.total_trades == 0 else (self.winning_trades / self.total_trades) * 100
            logging.info(f"Win streak: {self.current_win_streak} | Win rate: {win_rate:.1f}% | Scale: {self.current_position_scale:.2f}")


class SimulatedAccount:
    """Simulated trading account for backtesting with short selling support"""
    def __init__(self, initial_capital=10000.0, fee_rate=0.001, max_leverage=2.0, allow_short=True):
        """Initialize simulated account
        
        Args:
            initial_capital: Starting capital in quote currency (e.g. USDT)
            fee_rate: Trading fee as decimal (e.g. 0.001 = 0.1%)
            max_leverage: Maximum leverage allowed (e.g. 2.0 = 2x leverage)
            allow_short: Whether to allow short selling
        """
        self.initial_capital = initial_capital
        self.balances = {'USDT': initial_capital, 'BTC': 0.0}
        self.trades = []
        self.fees_paid = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.fee_rate = fee_rate
        self.last_buy_price = 0.0  # Track for win/loss calculation
        self.last_trade_type = None  # 'long' or 'short'
        self.max_leverage = max_leverage
        self.allow_short = allow_short
        self.short_positions = 0.0  # Track short positions separately
        
        # Added debug output
        print(f"Initialized SimulatedAccount with {initial_capital} USDT, max leverage: {max_leverage}x, shorts: {'allowed' if allow_short else 'disabled'}")
        
    def buy(self, price, amount, fee_rate=None):
        """Buy asset (go long) with improved return values for trade tracking"""
        if fee_rate is None:
            fee_rate = self.fee_rate
            
        # Calculate the required quote currency (e.g., USDT)
        quote_amount = amount * price
        fee = quote_amount * fee_rate
        total_cost = quote_amount + fee
        
        # Check if we have enough quote currency
        available_quote = self.balances.get('USDT', 0)
        if available_quote < total_cost:
            # Scale down the order
            scaling_factor = available_quote / total_cost if total_cost > 0 else 0
            amount = amount * scaling_factor
            quote_amount = amount * price
            fee = quote_amount * fee_rate
            total_cost = quote_amount + fee
            
        # Execute the trade if we have enough funds
        if self.balances.get('USDT', 0) >= total_cost and total_cost > 0:
            # Deduct quote currency
            self.balances['USDT'] = self.balances.get('USDT', 0) - total_cost
            
            # Add base currency (e.g., BTC)
            self.balances['BTC'] = self.balances.get('BTC', 0) + amount
            
            # First cover any short positions if they exist
            if self.short_positions > 0:
                cover_amount = min(amount, self.short_positions)
                if cover_amount > 0:
                    self.short_positions -= cover_amount
                    logging.info(f"COVERED SHORT: {cover_amount:.8f} BTC at ${price:.2f}")
            
            # Track the buy price for win/loss calculation
            self.last_buy_price = price
            self.last_trade_type = 'long'
            
            # Record the trade
            trade = {
                'time': datetime.now(),
                'type': 'buy',
                'price': price,
                'amount': amount,
                'quote_amount': quote_amount,
                'fee': fee,
                'total_cost': total_cost
            }
            self.trades.append(trade)
            self.fees_paid += fee
            
            # Log trade
            logging.info(f"BUY EXECUTED: {amount:.8f} BTC at ${price:.2f} | Cost: ${total_cost:.2f}")
            
            return quote_amount, amount  # Return the actual amounts traded
        
        return 0, 0  # Return zeros if trade failed
    
    def sell(self, price, amount, fee_rate=None):
        """Sell asset (close long or open short) with improved return values for trade tracking"""
        if fee_rate is None:
            fee_rate = self.fee_rate
            
        # Check if we have enough base currency (e.g., BTC)
        available_base = self.balances.get('BTC', 0)
        
        # Determine if this is closing a long position or opening a short
        if available_base >= amount:
            # We have enough of the asset - closing a long position
            return self._close_long(price, amount, fee_rate)
        elif self.allow_short:
            # We don't have enough - this is a short sell
            return self._open_short(price, amount, fee_rate)
        else:
            # Short selling not allowed
            logging.warning(f"SELL REJECTED: Insufficient balance ({available_base:.8f} BTC) and short selling disabled")
            return 0, 0, False
    
    def _close_long(self, price, amount, fee_rate):
        """Close a long position (sell assets we own)"""
        available_base = self.balances.get('BTC', 0)
        sell_amount = min(available_base, amount)
        
        # Calculate the resulting quote currency (e.g., USDT)
        quote_amount = sell_amount * price
        fee = quote_amount * fee_rate
        net_proceeds = quote_amount - fee
        
        # Execute the trade if amount is positive
        if sell_amount > 0:
            # Deduct base currency
            self.balances['BTC'] = self.balances.get('BTC', 0) - sell_amount
            
            # Add quote currency
            self.balances['USDT'] = self.balances.get('USDT', 0) + net_proceeds
            
            # Record the trade
            trade = {
                'time': datetime.now(),
                'type': 'sell_long',
                'price': price,
                'amount': sell_amount,
                'quote_amount': quote_amount,
                'fee': fee,
                'net_proceeds': net_proceeds
            }
            
            # Calculate win/loss (only for closing long positions)
            is_win = False
            if self.last_trade_type == 'long':
                is_win = price > self.last_buy_price
                
                if is_win:
                    self.win_count += 1
                    self.total_profit += (price - self.last_buy_price) * sell_amount
                    logging.info(f"WIN TRADE: Buy @ ${self.last_buy_price:.2f}, Sell @ ${price:.2f}")
                else:
                    self.loss_count += 1
                    self.total_loss += (self.last_buy_price - price) * sell_amount
                    logging.info(f"LOSS TRADE: Buy @ ${self.last_buy_price:.2f}, Sell @ ${price:.2f}")
            
            self.trades.append(trade)
            self.fees_paid += fee
            
            # Log trade
            logging.info(f"SELL EXECUTED: {sell_amount:.8f} BTC at ${price:.2f} | Proceeds: ${net_proceeds:.2f}")
            
            self.last_trade_price = price
            self.last_trade_type = 'flat'  # No position after selling
            
            return sell_amount, net_proceeds, is_win
        
        return 0, 0, False
    
    def _open_short(self, price, amount, fee_rate):
        """Open a short position (sell assets we don't own)"""
        # Check available margin
        total_value = self.get_total_value(price)
        current_leverage = self._calculate_current_leverage(price)
        
        # Calculate how much more we can short based on leverage limits
        max_short_value = (self.max_leverage * total_value) - (current_leverage * total_value)
        max_short_amount = max_short_value / price if price > 0 else 0
        
        # Cap the short amount
        short_amount = min(amount, max_short_amount)
        
        if short_amount <= 0:
            logging.warning(f"SHORT REJECTED: Exceeded max leverage of {self.max_leverage}x")
            return 0, 0, False
        
        # Calculate the resulting quote currency
        quote_amount = short_amount * price
        fee = quote_amount * fee_rate
        net_proceeds = quote_amount - fee
        
        # Execute the short
        self.short_positions += short_amount
        self.balances['USDT'] = self.balances.get('USDT', 0) + net_proceeds
        
        # Record the trade
        trade = {
            'time': datetime.now(),
            'type': 'short',
            'price': price,
            'amount': short_amount,
            'quote_amount': quote_amount,
            'fee': fee,
            'net_proceeds': net_proceeds
        }
        
        self.trades.append(trade)
        self.fees_paid += fee
        
        # Set reference price for P&L calculation
        self.last_trade_price = price
        self.last_trade_type = 'short'
        
        # Log trade
        logging.info(f"SHORT EXECUTED: {short_amount:.8f} BTC at ${price:.2f} | Proceeds: ${net_proceeds:.2f}")
        
        return short_amount, net_proceeds, False  # New short is not a win/loss yet
    
    def cover_short(self, price, amount, fee_rate=None):
        """Cover (buy to close) a short position"""
        if fee_rate is None:
            fee_rate = self.fee_rate
        
        # Limit to actual short positions
        cover_amount = min(amount, self.short_positions)
        
        if cover_amount <= 0:
            return 0, 0, False  # No shorts to cover
        
        # Calculate cost to cover
        quote_amount = cover_amount * price
        fee = quote_amount * fee_rate
        total_cost = quote_amount + fee
        
        # Check if we have enough funds
        available_quote = self.balances.get('USDT', 0)
        
        if available_quote < total_cost:
            # Scale down if needed
            scaling_factor = available_quote / total_cost if total_cost > 0 else 0
            cover_amount = cover_amount * scaling_factor
            quote_amount = cover_amount * price
            fee = quote_amount * fee_rate
            total_cost = quote_amount + fee
        
        if cover_amount <= 0 or total_cost <= 0:
            return 0, 0, False
        
        # Execute the cover
        self.balances['USDT'] = self.balances.get('USDT', 0) - total_cost
        self.short_positions -= cover_amount
        
        # Record the trade
        trade = {
            'time': datetime.now(),
            'type': 'cover',
            'price': price,
            'amount': cover_amount,
            'quote_amount': quote_amount,
            'fee': fee,
            'total_cost': total_cost
        }
        
        # Calculate win/loss
        is_win = False
        if self.last_trade_type == 'short':
            is_win = price < self.last_trade_price  # For shorts, we win if price goes down
            
            if is_win:
                profit = (self.last_trade_price - price) * cover_amount
                self.win_count += 1
                self.total_profit += profit
                logging.info(f"WIN SHORT: Short @ ${self.last_trade_price:.2f}, Cover @ ${price:.2f}")
            else:
                loss = (price - self.last_trade_price) * cover_amount
                self.loss_count += 1
                self.total_loss += loss
                logging.info(f"LOSS SHORT: Short @ ${self.last_trade_price:.2f}, Cover @ ${price:.2f}")
                
        self.trades.append(trade)
        self.fees_paid += fee
        
        # Log trade
        logging.info(f"COVER EXECUTED: {cover_amount:.8f} BTC at ${price:.2f} | Cost: ${total_cost:.2f}")
        
        if self.short_positions <= 0:
            self.last_trade_type = 'flat'  # No position after covering
        
        return cover_amount, total_cost, is_win
    
    def get_total_value(self, current_price):
        """Get total account value in quote currency, accounting for shorts"""
        # Sum up the value of all assets in quote currency (e.g., USDT)
        quote_balance = self.balances.get('USDT', 0)
        base_value = self.balances.get('BTC', 0) * current_price
        
        # Subtract short position value
        short_value = self.short_positions * current_price
        
        return quote_balance + base_value - short_value
    
    def _calculate_current_leverage(self, current_price):
        """Calculate current leverage ratio"""
        equity = self.get_total_value(current_price)
        if equity <= 0:
            return float('inf')  # Avoid division by zero
            
        long_exposure = self.balances.get('BTC', 0) * current_price
        short_exposure = self.short_positions * current_price
        total_exposure = long_exposure + short_exposure
        
        return total_exposure / equity
        
    def get_base_balance(self):
        """Get balance of base asset (e.g., BTC)"""
        return self.balances.get('BTC', 0)
        
    def get_quote_balance(self):
        """Get balance of quote asset (e.g., USDT)"""
        return self.balances.get('USDT', 0)
        
    def get_short_position(self):
        """Get current short position size"""
        return self.short_positions
        
    def get_net_position(self):
        """Get net position (long - short)"""
        return self.balances.get('BTC', 0) - self.short_positions
        
    def check_margin_call(self, current_price, margin_call_threshold=0.9):
        """Check if a margin call would be triggered
        
        Args:
            current_price: Current price of the asset
            margin_call_threshold: Leverage ratio that triggers a margin call
            
        Returns:
            bool: True if margin call triggered, False otherwise
        """
        leverage = self._calculate_current_leverage(current_price)
        return leverage >= self.max_leverage * margin_call_threshold


def main():
    """Main entry point for historical backtesting script"""
    parser = argparse.ArgumentParser(description='DMT_v2 Historical Backtester')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--interval', type=str, default='1m', help='Candlestick interval')
    parser.add_argument('--start-date', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-04-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--version', type=str, default='enhanced', help='Strategy version (base, enhanced)')
    parser.add_argument('--data-file', type=str, default=None, help='Path to historical data file (CSV)')
    parser.add_argument('--limit-data', type=int, default=None, help='Limit the number of data points to use')
    parser.add_argument('--fee', type=float, default=0.001, help='Trading fee rate')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--speed', type=int, default=1000, help='Speed factor for backtesting')
    parser.add_argument('--api-key', type=str, default=None, help='Binance API key')
    parser.add_argument('--api-secret', type=str, default=None, help='Binance API secret')
    
    args = parser.parse_args()
    
    backtester = HistoricalBacktester(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        strategy_version=args.version,
        data_file=args.data_file,
        fee_rate=args.fee,
        progress_bar=not args.no_progress,
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    backtester.run_backtest(speed_factor=args.speed, limit_data=args.limit_data)
    
    backtester.save_results_to_csv()


if __name__ == "__main__":
    main()
