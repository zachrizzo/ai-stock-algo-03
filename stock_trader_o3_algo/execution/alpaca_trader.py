"""
Module for executing trades with Alpaca API.
"""
import os
import datetime as dt
from typing import Dict, List, Optional

import alpaca_trade_api as tradeapi
import pandas as pd
import pytz

from stock_trader_o3_algo.config.settings import (
    API_KEY, API_SECRET, BASE_URL, 
    CASH_ETF
)
from stock_trader_o3_algo.core.strategy import get_portfolio_allocation


class AlpacaTrader:
    """Trading executor for Alpaca API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Alpaca trader.
        
        Args:
            api_key: Alpaca API key (defaults to environment variable)
            api_secret: Alpaca API secret (defaults to environment variable)
            base_url: Alpaca API base URL (defaults to environment variable)
        """
        # Use provided values or defaults from settings
        self.api_key = api_key or API_KEY
        self.api_secret = api_secret or API_SECRET
        self.base_url = base_url or BASE_URL
        
        # Initialize API client
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url)
        
        # Initialize cooldown tracking
        self.cooldown_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cooldown.txt')
        self.peak_equity_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'peak_equity.txt')
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.cooldown_file), exist_ok=True)
    
    def get_account_info(self) -> Dict:
        """
        Get account information from Alpaca.
        
        Returns:
            Dictionary with account information
        """
        account = self.api.get_account()
        return {
            'id': account.id,
            'cash': float(account.cash),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'status': account.status
        }
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions from Alpaca.
        
        Returns:
            Dictionary with symbols as keys and dollar values as values
        """
        positions = {}
        try:
            for position in self.api.list_positions():
                symbol = position.symbol
                market_value = float(position.market_value)
                positions[symbol] = market_value
        except Exception as e:
            print(f"Error getting positions: {e}")
        
        return positions
    
    def liquidate_all_positions(self) -> None:
        """Liquidate all positions."""
        try:
            self.api.close_all_positions()
            print("All positions liquidated successfully")
        except Exception as e:
            print(f"Error liquidating positions: {e}")
    
    def place_fractional_order(self, symbol: str, target_dollars: float) -> Dict:
        """
        Place a fractional order for a given symbol.
        
        Args:
            symbol: Symbol to trade
            target_dollars: Dollar amount to trade
            
        Returns:
            Dictionary with order information
        """
        if target_dollars < 0.01:
            print(f"Skipping {symbol}: amount too small ({target_dollars:.2f})")
            return {}
        
        try:
            # Get current price
            last_trade = self.api.get_latest_trade(symbol)
            price = float(last_trade.price)
            
            # Calculate quantity (fractional shares)
            qty = round(target_dollars / price, 6)
            
            if qty > 0:
                # Place order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                
                print(f"Order placed for {symbol}: {qty} shares (${target_dollars:.2f})")
                
                return {
                    'id': order.id,
                    'symbol': symbol,
                    'qty': qty,
                    'notional': target_dollars,
                    'side': 'buy',
                    'status': order.status
                }
            else:
                print(f"Skipping {symbol}: quantity too small ({qty})")
                return {}
        
        except Exception as e:
            print(f"Error placing order for {symbol}: {e}")
            return {}
    
    def is_cooldown_active(self) -> bool:
        """
        Check if the kill switch cooldown is active.
        
        Returns:
            True if cooldown is active, False otherwise
        """
        if not os.path.exists(self.cooldown_file):
            return False
        
        with open(self.cooldown_file, 'r') as f:
            cooldown_end_str = f.read().strip()
        
        try:
            cooldown_end = dt.datetime.fromisoformat(cooldown_end_str)
            return dt.datetime.now() < cooldown_end
        except (ValueError, TypeError):
            return False
    
    def set_cooldown(self, weeks: int = 4) -> None:
        """
        Set the kill switch cooldown.
        
        Args:
            weeks: Number of weeks for cooldown
        """
        cooldown_end = dt.datetime.now() + dt.timedelta(weeks=weeks)
        
        with open(self.cooldown_file, 'w') as f:
            f.write(cooldown_end.isoformat())
        
        print(f"Kill switch cooldown set until {cooldown_end}")
    
    def get_peak_equity(self) -> float:
        """
        Get the peak equity value.
        
        Returns:
            Peak equity value
        """
        if not os.path.exists(self.peak_equity_file):
            return 0.0
        
        try:
            with open(self.peak_equity_file, 'r') as f:
                return float(f.read().strip())
        except (ValueError, TypeError):
            return 0.0
    
    def update_peak_equity(self, current_equity: float) -> float:
        """
        Update the peak equity value.
        
        Args:
            current_equity: Current equity value
            
        Returns:
            Updated peak equity value
        """
        peak_equity = self.get_peak_equity()
        new_peak = max(peak_equity, current_equity)
        
        with open(self.peak_equity_file, 'w') as f:
            f.write(str(new_peak))
        
        return new_peak
    
    def execute_strategy(self, prices: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> Dict:
        """
        Execute the micro-CTA strategy.
        
        Args:
            prices: DataFrame with price data
            date: Date to use for execution (defaults to latest date in prices)
            
        Returns:
            Dictionary with execution information
        """
        # Use the latest date if not specified
        if date is None:
            date = prices.index[-1]
        
        # Get account information
        account_info = self.get_account_info()
        equity = account_info['equity']
        
        # Update peak equity
        peak_equity = self.update_peak_equity(equity)
        
        # Check if cooldown is active
        if self.is_cooldown_active():
            print("Cooldown is active, skipping execution")
            return {
                'date': date,
                'status': 'COOLDOWN',
                'equity': equity,
                'peak_equity': peak_equity,
                'orders': []
            }
        
        # Get portfolio allocation
        allocation = get_portfolio_allocation(
            prices,
            date=date,
            equity=equity,
            equity_peak=peak_equity
        )
        
        # Check if kill switch should be activated
        if equity < peak_equity * 0.8:
            print("Kill switch activated - moving to cash")
            self.liquidate_all_positions()
            self.set_cooldown()
            
            # Place order for cash ETF
            order = self.place_fractional_order(CASH_ETF, equity)
            
            return {
                'date': date,
                'status': 'KILL_SWITCH',
                'equity': equity,
                'peak_equity': peak_equity,
                'orders': [order] if order else []
            }
        
        # Execute the allocation
        print(f"Executing allocation: {allocation}")
        
        # Liquidate current positions
        self.liquidate_all_positions()
        
        # Place orders for new positions
        orders = []
        for symbol, amount in allocation.items():
            if amount > 0:
                order = self.place_fractional_order(symbol, amount)
                if order:
                    orders.append(order)
        
        return {
            'date': date,
            'status': 'EXECUTED',
            'equity': equity,
            'peak_equity': peak_equity,
            'orders': orders
        }
