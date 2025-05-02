#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Backtester Module
======================
Central backtesting framework for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import time
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Base backtester class for running strategy evaluations.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Union[str, Callable],
        strategy_params: Optional[Dict[str, Any]] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        benchmark_key: Optional[str] = None
    ):
        """
        Initialize backtester with data and strategy
        
        Args:
            data: Price data (OHLCV) for backtesting
            strategy: Strategy function or name of built-in strategy
            strategy_params: Parameters for the strategy
            initial_capital: Starting capital
            commission: Commission per trade as percentage
            slippage: Slippage per trade as percentage
            benchmark_key: Name of benchmark for comparison
        """
        self.data = data.copy()
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.benchmark_key = benchmark_key
        
        # Store results
        self.results = None
        self.metrics = None
        self.execution_time = 0
        
    def run(self) -> pd.DataFrame:
        """
        Run the backtest
        
        Returns:
            DataFrame with backtest results
        """
        start_time = time.time()
        logger.info(f"Starting backtest with {len(self.data)} data points")
        
        # Initialize results DataFrame
        self.results = self.data.copy()
        self.results['signal'] = 0.0
        self.results['position'] = 0.0
        self.results['equity'] = self.initial_capital
        self.results['cash'] = self.initial_capital
        
        # Generate signals based on strategy
        if callable(self.strategy):
            # Execute strategy function
            self._execute_strategy_function()
        elif isinstance(self.strategy, str):
            # Execute built-in strategy
            self._execute_builtin_strategy()
        else:
            raise ValueError("Strategy must be a function or name of built-in strategy")
        
        # Calculate returns
        self.results['returns'] = self.results['equity'].pct_change().fillna(0)
        
        # Add benchmark if specified
        if self.benchmark_key and 'Close' in self.results.columns:
            self.results['buy_hold_equity'] = self.initial_capital * (
                self.results['Close'] / self.results['Close'].iloc[0]
            )
            self.results['buy_hold_returns'] = self.results['buy_hold_equity'].pct_change().fillna(0)
        
        # Calculate metrics
        self._calculate_metrics()
        
        self.execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {self.execution_time:.2f} seconds")
        
        return self.results
    
    def _execute_strategy_function(self) -> None:
        """
        Execute the provided strategy function
        """
        try:
            # Call strategy function with data and parameters
            strategy_results = self.strategy(self.data, **self.strategy_params)
            
            # Check if function returns signals or a complete results DataFrame
            if isinstance(strategy_results, pd.DataFrame) and 'signal' in strategy_results.columns:
                self.results = strategy_results
            elif isinstance(strategy_results, pd.Series):
                self.results['signal'] = strategy_results
                self._calculate_positions_and_equity()
            else:
                raise ValueError("Strategy function must return signals or complete results DataFrame")
                
        except Exception as e:
            logger.error(f"Error executing strategy function: {str(e)}")
            raise
    
    def _execute_builtin_strategy(self) -> None:
        """
        Execute a built-in strategy by name
        """
        strategy_name = self.strategy.lower()
        
        try:
            if strategy_name == 'dmt_v2':
                from stock_trader_o3_algo.strategies.dmt_v2_strategy import DMT_v2_Strategy
                
                # Initialize strategy
                dmt_strategy = DMT_v2_Strategy(
                    version=self.strategy_params.get('version', 'original'),
                    asset_type=self.strategy_params.get('asset_type', 'equity'),
                    lookback_period=self.strategy_params.get('lookback_period', 252),
                    initial_capital=self.initial_capital
                )
                
                # Run backtest
                self.results, _ = dmt_strategy.run_backtest(self.data)
                
            elif strategy_name == 'tri_shot':
                # Implement Tri-Shot strategy
                logger.warning("Tri-Shot strategy not fully implemented yet")
                self._execute_placeholder_strategy()
                
            elif strategy_name == 'turbo_qt':
                # Implement TurboQT strategy
                logger.warning("TurboQT strategy not fully implemented yet")
                self._execute_placeholder_strategy()
                
            else:
                logger.warning(f"Unknown strategy: {strategy_name}, using placeholder")
                self._execute_placeholder_strategy()
                
        except Exception as e:
            logger.error(f"Error executing built-in strategy {strategy_name}: {str(e)}")
            raise
    
    def _execute_placeholder_strategy(self) -> None:
        """
        Execute a simple placeholder strategy (moving average crossover)
        """
        # Calculate moving averages
        self.results['ma_short'] = self.results['Close'].rolling(
            window=self.strategy_params.get('ma_short', 10)
        ).mean()
        
        self.results['ma_long'] = self.results['Close'].rolling(
            window=self.strategy_params.get('ma_long', 50)
        ).mean()
        
        # Generate signals: 1 when short MA crosses above long MA, -1 when crossing below
        self.results['signal'] = 0.0
        crossover = (self.results['ma_short'] > self.results['ma_long']) & \
                    (self.results['ma_short'].shift() <= self.results['ma_long'].shift())
        crossunder = (self.results['ma_short'] < self.results['ma_long']) & \
                     (self.results['ma_short'].shift() >= self.results['ma_long'].shift())
                    
        self.results.loc[crossover, 'signal'] = 1.0
        self.results.loc[crossunder, 'signal'] = -1.0
        
        # Calculate positions and equity
        self._calculate_positions_and_equity()
    
    def _calculate_positions_and_equity(self) -> None:
        """
        Calculate positions and equity based on signals
        """
        # Fill NaN signals with 0
        self.results['signal'].fillna(0, inplace=True)
        
        # Convert signals to positions (assuming signals are directional)
        self.results['position'] = self.results['signal'].copy()
        
        # Initialize equity and cash
        self.results['equity'] = self.initial_capital
        self.results['cash'] = self.initial_capital
        
        # Calculate equity changes
        for i in range(1, len(self.results)):
            # Previous position and equity
            prev_position = self.results['position'].iloc[i-1]
            prev_equity = self.results['equity'].iloc[i-1]
            
            # Market return for this period
            market_return = self.results['Close'].iloc[i] / self.results['Close'].iloc[i-1] - 1
            
            # Strategy return based on previous position
            strategy_return = prev_position * market_return
            
            # Apply commission and slippage if position changed
            position_change = abs(self.results['position'].iloc[i] - prev_position)
            if position_change > 0.01:  # Threshold to detect position changes
                strategy_return -= position_change * (self.commission + self.slippage)
            
            # Update equity
            self.results.loc[self.results.index[i], 'equity'] = prev_equity * (1 + strategy_return)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        from stock_trader_o3_algo.backtester.performance import calculate_performance_metrics
        
        self.metrics = calculate_performance_metrics(
            self.results,
            initial_capital=self.initial_capital
        )
        
        return self.metrics
    
    def plot_results(self, title: str = None, filename: Optional[str] = None) -> None:
        """
        Plot backtest results
        
        Args:
            title: Plot title
            filename: File path to save plot
        """
        from stock_trader_o3_algo.backtester.visualization import plot_results
        
        if self.results is None:
            raise ValueError("Run backtest before plotting results")
        
        # Create title if not provided
        if title is None:
            if isinstance(self.strategy, str):
                strategy_name = self.strategy.upper()
            else:
                strategy_name = "Custom Strategy"
            
            title = f"{strategy_name} Backtest Results"
        
        # If benchmark is present, include it in the plot
        results_dict = {'Strategy': self.results}
        
        if self.benchmark_key and 'buy_hold_equity' in self.results.columns:
            # Create a copy with benchmark data
            benchmark_results = self.results.copy()
            benchmark_results['equity'] = self.results['buy_hold_equity']
            results_dict['Buy & Hold'] = benchmark_results
        
        # Plot the results
        plot_results(results_dict, title=title, filename=filename)
    
    def plot_regime_analysis(self, ticker: str, filename: Optional[str] = None) -> None:
        """
        Plot regime analysis of backtest results
        
        Args:
            ticker: Symbol of the asset
            filename: File path to save plot
        """
        from stock_trader_o3_algo.backtester.visualization import plot_regime_positions
        
        if self.results is None:
            raise ValueError("Run backtest before plotting results")
            
        plot_regime_positions(self.results, ticker=ticker, filename=filename)
    
    def save_results(self, filename: str) -> None:
        """
        Save backtest results to CSV
        
        Args:
            filename: File path to save results
        """
        if self.results is None:
            raise ValueError("Run backtest before saving results")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save to CSV
        self.results.to_csv(filename)
        logger.info(f"Results saved to {filename}")
    
    def print_metrics(self) -> None:
        """
        Print performance metrics to console
        """
        from stock_trader_o3_algo.backtester.visualization import create_performance_summary
        
        if self.metrics is None:
            raise ValueError("Run backtest before printing metrics")
        
        # Format single strategy results
        results_dict = {'Strategy': self.results}
        
        if self.benchmark_key and 'buy_hold_equity' in self.results.columns:
            benchmark_results = self.results.copy()
            benchmark_results['equity'] = self.results['buy_hold_equity']
            results_dict['Buy & Hold'] = benchmark_results
        
        # Generate and print summary
        summary = create_performance_summary(
            results_dict, 
            benchmark_key='Buy & Hold' if self.benchmark_key else None
        )
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(summary)
        print("\n" + "=" * 80)
        print(f"Execution time: {self.execution_time:.2f} seconds")
        print("=" * 80)
        

class BatchBacktester:
    """
    Run multiple backtest configurations and compare results
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_configs: List[Dict[str, Any]],
        initial_capital: float = 10000.0,
        benchmark_key: Optional[str] = None
    ):
        """
        Initialize batch backtester
        
        Args:
            data: Price data (OHLCV) for backtesting
            strategy_configs: List of strategy configuration dictionaries
            initial_capital: Starting capital
            benchmark_key: Name of benchmark for comparison
        """
        self.data = data.copy()
        self.strategy_configs = strategy_configs
        self.initial_capital = initial_capital
        self.benchmark_key = benchmark_key
        
        # Store results from all backtest runs
        self.all_results = {}
        self.all_metrics = {}
        self.execution_times = {}
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Run all backtest configurations
        
        Returns:
            Dictionary of results by strategy name
        """
        logger.info(f"Starting batch backtest with {len(self.strategy_configs)} configurations")
        
        for config in self.strategy_configs:
            name = config.get('name', f"Strategy_{len(self.all_results)+1}")
            strategy = config.get('strategy')
            params = config.get('params', {})
            
            logger.info(f"Running backtest for {name}")
            
            try:
                # Create and run backtester
                backtester = Backtester(
                    data=self.data,
                    strategy=strategy,
                    strategy_params=params,
                    initial_capital=self.initial_capital,
                    commission=config.get('commission', 0.0),
                    slippage=config.get('slippage', 0.0),
                    benchmark_key=self.benchmark_key
                )
                
                results = backtester.run()
                
                # Store results and metrics
                self.all_results[name] = results
                self.all_metrics[name] = backtester.metrics
                self.execution_times[name] = backtester.execution_time
                
                logger.info(f"Completed backtest for {name}")
                
            except Exception as e:
                logger.error(f"Error in backtest for {name}: {str(e)}")
        
        # Add benchmark if specified
        if self.benchmark_key and len(self.all_results) > 0:
            first_results = next(iter(self.all_results.values()))
            if 'Close' in first_results.columns:
                # Create benchmark results
                benchmark_results = first_results.copy()
                benchmark_results['equity'] = self.initial_capital * (
                    benchmark_results['Close'] / benchmark_results['Close'].iloc[0]
                )
                
                self.all_results['Buy & Hold'] = benchmark_results
        
        logger.info(f"Batch backtest completed with {len(self.all_results)} successful runs")
        
        return self.all_results
    
    def plot_comparison(
        self, 
        title: str = "Strategy Comparison", 
        filename: Optional[str] = None
    ) -> None:
        """
        Plot comparative results of all strategies
        
        Args:
            title: Plot title
            filename: File path to save plot
        """
        from stock_trader_o3_algo.backtester.visualization import plot_results
        
        if not self.all_results:
            raise ValueError("Run batch backtest before plotting results")
        
        plot_results(self.all_results, title=title, filename=filename)
    
    def print_summary(self) -> None:
        """
        Print comparative performance summary
        """
        from stock_trader_o3_algo.backtester.visualization import create_performance_summary
        
        if not self.all_results:
            raise ValueError("Run batch backtest before printing summary")
        
        summary = create_performance_summary(
            self.all_results, 
            benchmark_key='Buy & Hold' if 'Buy & Hold' in self.all_results else None
        )
        
        print("\n" + "=" * 80)
        print("BATCH BACKTEST SUMMARY")
        print("=" * 80)
        print(summary)
        
        print("\n" + "=" * 40)
        print("Execution Times")
        print("=" * 40)
        for name, time_sec in self.execution_times.items():
            print(f"{name}: {time_sec:.2f} seconds")
            
        print("\n" + "=" * 80)
