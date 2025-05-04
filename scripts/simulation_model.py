"""
Simulation Model for Paper Trading

This module provides a simulation model that runs alongside Binance Testnet,
allowing for full control over initial capital and balances.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

class SimulationAccount:
    """
    A simulation account that mirrors Binance trading but with full control over balances.
    """
    def __init__(
        self,
        initial_capital: float = 10000.0,
        base_asset: str = "BTC",
        quote_asset: str = "USDT",
        fee_rate: float = 0.001  # 0.1% fee by default
    ):
        """
        Initialize the simulation account.
        
        Args:
            initial_capital: Initial capital in quote asset (e.g., USDT)
            base_asset: Base asset symbol (e.g., BTC)
            quote_asset: Quote asset symbol (e.g., USDT)
            fee_rate: Trading fee rate as a decimal (e.g., 0.001 for 0.1%)
        """
        self.initial_capital = initial_capital
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.fee_rate = fee_rate
        
        # Initialize balances
        self.balances = {
            quote_asset: initial_capital,
            base_asset: 0.0
        }
        
        # Track trades and performance
        self.trades = []
        self.equity_history = []
        self.price_history = []
        self.best_profit = 0.0
        self.max_drawdown = 0.0
        self.start_time = datetime.now()
        
        logging.info(f"Simulation account initialized with {initial_capital} {quote_asset}")
    
    def reset(self, initial_capital: Optional[float] = None):
        """
        Reset the simulation account to initial state.
        
        Args:
            initial_capital: New initial capital (if None, use the original)
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        # Reset balances
        self.balances = {
            self.quote_asset: self.initial_capital,
            self.base_asset: 0.0
        }
        
        # Clear trade history but keep price history
        self.trades = []
        self.equity_history = []
        self.best_profit = 0.0
        self.max_drawdown = 0.0
        self.start_time = datetime.now()
        
        logging.info(f"Simulation account reset with {self.initial_capital} {self.quote_asset}")
        
        return True
    
    def execute_market_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Execute a market buy order in the simulation.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            quantity: Quantity to buy in base asset
            price: Current market price
            timestamp: Order timestamp (defaults to now)
        
        Returns:
            Dict with order details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate order value and fees
        order_value = quantity * price
        fee = order_value * self.fee_rate
        total_cost = order_value + fee
        
        # Check if we have enough balance
        if self.balances[self.quote_asset] < total_cost:
            logging.warning(f"Insufficient balance: {self.balances[self.quote_asset]} {self.quote_asset} < {total_cost} {self.quote_asset}")
            # Adjust quantity to available balance
            available_for_purchase = self.balances[self.quote_asset] / (price * (1 + self.fee_rate))
            quantity = available_for_purchase
            order_value = quantity * price
            fee = order_value * self.fee_rate
            total_cost = order_value + fee
        
        # Update balances
        self.balances[self.quote_asset] -= total_cost
        self.balances[self.base_asset] += quantity
        
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'BUY',
            'price': price,
            'quantity': quantity,
            'value': order_value,
            'fee': fee,
            'total_cost': total_cost,
            'balance_after': {
                self.quote_asset: self.balances[self.quote_asset],
                self.base_asset: self.balances[self.base_asset]
            }
        }
        self.trades.append(trade)
        
        # Update equity history
        self._update_equity(price)
        
        logging.info(f"SIM BUY: {quantity:.8f} {self.base_asset} at {price:.2f} {self.quote_asset}")
        
        return trade
    
    def execute_market_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Execute a market sell order in the simulation.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            quantity: Quantity to sell in base asset
            price: Current market price
            timestamp: Order timestamp (defaults to now)
        
        Returns:
            Dict with order details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if we have enough balance
        if self.balances[self.base_asset] < quantity:
            logging.warning(f"Insufficient balance: {self.balances[self.base_asset]} {self.base_asset} < {quantity} {self.base_asset}")
            quantity = self.balances[self.base_asset]
        
        # Calculate order value and fees
        order_value = quantity * price
        fee = order_value * self.fee_rate
        net_proceeds = order_value - fee
        
        # Update balances
        self.balances[self.base_asset] -= quantity
        self.balances[self.quote_asset] += net_proceeds
        
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'SELL',
            'price': price,
            'quantity': quantity,
            'value': order_value,
            'fee': fee,
            'net_proceeds': net_proceeds,
            'balance_after': {
                self.quote_asset: self.balances[self.quote_asset],
                self.base_asset: self.balances[self.base_asset]
            }
        }
        self.trades.append(trade)
        
        # Update equity history
        self._update_equity(price)
        
        logging.info(f"SIM SELL: {quantity:.8f} {self.base_asset} at {price:.2f} {self.quote_asset}")
        
        return trade
    
    def get_balance(self, asset: str) -> float:
        """
        Get the current balance of an asset.
        
        Args:
            asset: Asset symbol
        
        Returns:
            Current balance
        """
        return self.balances.get(asset, 0.0)
    
    def get_portfolio_value(self, current_price: float) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            current_price: Current price of the base asset
        
        Returns:
            Total portfolio value in quote asset
        """
        base_value = self.balances[self.base_asset] * current_price
        quote_value = self.balances[self.quote_asset]
        return base_value + quote_value
    
    def get_pnl(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate profit and loss.
        
        Args:
            current_price: Current price of the base asset
        
        Returns:
            Tuple of (profit_loss, profit_loss_percentage)
        """
        portfolio_value = self.get_portfolio_value(current_price)
        profit_loss = portfolio_value - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return profit_loss, profit_loss_pct
    
    def get_position_size(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate current position size.
        
        Args:
            current_price: Current price of the base asset
        
        Returns:
            Tuple of (position_value, position_percentage)
        """
        portfolio_value = self.get_portfolio_value(current_price)
        position_value = self.balances[self.base_asset] * current_price
        position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        return position_value, position_pct
    
    def get_trade_stats(self) -> Dict:
        """
        Calculate trading statistics.
        
        Returns:
            Dict with trading statistics
        """
        buy_trades = sum(1 for trade in self.trades if trade['side'] == 'BUY')
        sell_trades = sum(1 for trade in self.trades if trade['side'] == 'SELL')
        total_fees = sum(trade['fee'] for trade in self.trades)
        
        # Calculate win/loss if we have completed round trips
        win_count = 0
        loss_count = 0
        total_profit = 0.0
        total_loss = 0.0
        
        # Analyze trades in pairs (buy/sell)
        buy_stack = []
        for trade in self.trades:
            if trade['side'] == 'BUY':
                buy_stack.append(trade)
            elif trade['side'] == 'SELL' and buy_stack:
                # Match this sell with the oldest buy
                buy_trade = buy_stack.pop(0)
                
                # Calculate profit/loss for this round trip
                buy_value = buy_trade['value']
                sell_value = trade['value']
                trade_pl = sell_value - buy_value
                
                if trade_pl > 0:
                    win_count += 1
                    total_profit += trade_pl
                else:
                    loss_count += 1
                    total_loss += abs(trade_pl)
        
        # Calculate win rate and profit factor
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_fees': total_fees,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _update_equity(self, current_price: float):
        """
        Update equity history and track max profit/drawdown.
        
        Args:
            current_price: Current price of the base asset
        """
        portfolio_value = self.get_portfolio_value(current_price)
        self.equity_history.append({
            'timestamp': datetime.now(),
            'value': portfolio_value
        })
        
        # Update price history
        self.price_history.append({
            'timestamp': datetime.now(),
            'price': current_price
        })
        
        # Track best profit and max drawdown
        profit_loss, _ = self.get_pnl(current_price)
        
        if profit_loss > self.best_profit:
            self.best_profit = profit_loss
        
        current_drawdown = self.best_profit - profit_loss
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
