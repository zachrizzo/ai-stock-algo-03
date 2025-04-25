"""
Backtesting engine for the micro-CTA strategy.
"""
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stock_trader_o3_algo.config.settings import (
    RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF,
    BACKTEST_RESULTS_DIR, LOOKBACK_DAYS, VOL_LOOK
)
from stock_trader_o3_algo.core.strategy import get_portfolio_allocation
from stock_trader_o3_algo.data.price_data import fetch_prices


class BacktestEngine:
    """Backtesting engine for the micro-CTA strategy."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100.0,
        rebalance_freq: str = "W-MON",  # Weekly on Monday by default
        transaction_cost_pct: float = 0.0003,  # 3 basis points
        cooldown_weeks: int = 4,
        trade_weekdays: Optional[Tuple[int, ...]] = None  # e.g., (0, 2, 4) for Mon/Wed/Fri
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            start_date: Start date for the backtest (YYYY-MM-DD)
            end_date: End date for the backtest (YYYY-MM-DD)
            initial_capital: Initial capital for the backtest
            rebalance_freq: Frequency for rebalancing (pandas frequency string)
            transaction_cost_pct: Transaction cost as a percentage of trade value
            cooldown_weeks: Number of weeks to remain in cash after kill switch
            trade_weekdays: Custom weekdays for rebalancing (e.g., (0, 2, 4) for Mon/Wed/Fri)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_pct = transaction_cost_pct
        self.cooldown_weeks = cooldown_weeks
        self.trade_weekdays = trade_weekdays  # None means use rebalance_freq as before
        
        # Initialize results containers
        self.equity_curve = None
        self.positions_history = None
        self.trade_history = None
        self.portfolio_stats = None
        
        # Fetch price data with extra history for calculations
        self.tickers = [RISK_ON, RISK_OFF, HEDGE_ETF, CASH_ETF]
        
        # Calculate how many days of extra history we need for calculations
        # We need at least LOOKBACK_DAYS for momentum and VOL_LOOK for volatility
        required_history_days = max(LOOKBACK_DAYS, VOL_LOOK) + 20  # Add extra buffer
        
        # Fetch data with enough history before start_date
        self.prices = fetch_prices(self.tickers, days=260*2, end_date=self.end_date)
        
        # Store the full price history for calculations
        self.full_prices = self.prices.copy()
        
        # Trim prices to the backtest period for portfolio valuation
        mask = (self.prices.index >= self.start_date) & (self.prices.index <= self.end_date)
        self.prices = self.prices.loc[mask]
        
        # Verify we have enough data for the strategy calculations
        earliest_available_date = self.full_prices.index[0]
        if earliest_available_date > (self.start_date - pd.Timedelta(days=required_history_days)):
            print(f"Warning: Limited historical data available before backtest start date. " 
                  f"Have data from {earliest_available_date}, but ideally need data from " 
                  f"{self.start_date - pd.Timedelta(days=required_history_days)} for calculations.")
    
    def run_backtest(self) -> pd.Series:
        """
        Run the backtest and return the equity curve.
        
        Returns:
            Series with the equity curve
        """
        # Generate rebalance dates
        if self.trade_weekdays is None:
            # Original behavior – use pandas frequency string
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq=self.rebalance_freq
            )
            rebalance_dates = [d for d in rebalance_dates if d in self.prices.index]
        else:
            # Custom weekdays (e.g., Mon/Wed/Fri)
            rebalance_dates = [d for d in self.prices.index if d.weekday() in self.trade_weekdays]
        
        # Initial portfolio setup
        equity = self.initial_capital
        peak_equity = equity
        current_positions = {CASH_ETF: equity}
        equity_curve = [(self.prices.index[0], equity)]
        positions_history = []
        trade_history = []
        kill_switch_active_until = None
        
        # Track daily equity
        for date in self.prices.index:
            # Check if we need to rebalance
            is_rebalance_day = date in rebalance_dates
            
            # Check if kill switch cooldown is over
            if kill_switch_active_until is not None:
                if date >= kill_switch_active_until:
                    kill_switch_active_until = None
            
            # Calculate current equity value based on positions
            day_equity = 0
            for symbol, amount in current_positions.items():
                # Convert dollar amount to shares at the previous rebalance
                if symbol in self.prices.columns:
                    try:
                        price = self.prices.loc[date, symbol]
                        position_value = amount * price / price  # Amount in dollars
                        day_equity += position_value
                    except KeyError:
                        # If price data is missing, use the previous value
                        day_equity += amount
            
            # Update peak equity
            peak_equity = max(peak_equity, day_equity)
            
            # Check if kill switch should be activated
            if day_equity < peak_equity * 0.8 and kill_switch_active_until is None:
                # Activate kill switch - move to cash and set cooldown period
                current_positions = {CASH_ETF: day_equity}
                kill_switch_active_until = date + pd.Timedelta(weeks=self.cooldown_weeks)
                trade_history.append({
                    'date': date,
                    'action': 'KILL_SWITCH',
                    'trades': [{'symbol': CASH_ETF, 'amount': day_equity, 'price': 1.0}]
                })
            
            # Rebalance if it's a rebalance day and not in cooldown
            if is_rebalance_day and kill_switch_active_until is None:
                # Get historical data up to this date
                historical_data = self.full_prices.loc[:date]
                
                # Get target allocation
                target_allocation = get_portfolio_allocation(
                    historical_data,
                    date=date,
                    equity=day_equity,
                    equity_peak=peak_equity
                )
                
                # Calculate and apply transaction costs
                trades = []
                total_cost = 0
                
                for symbol, target_amount in target_allocation.items():
                    current_amount = current_positions.get(symbol, 0)
                    trade_amount = target_amount - current_amount
                    
                    if abs(trade_amount) > 0.01:  # Only trade if significant change
                        # Calculate transaction cost
                        cost = abs(trade_amount) * self.transaction_cost_pct
                        total_cost += cost
                        
                        # Record the trade
                        trades.append({
                            'symbol': symbol,
                            'amount': trade_amount,
                            'price': self.prices.loc[date, symbol] if symbol in self.prices.columns else 1.0,
                            'cost': cost
                        })
                
                # Apply transaction costs to the equity
                day_equity -= total_cost
                
                # Update positions
                if trades:
                    # Adjust the target allocation to account for transaction costs
                    remaining_equity = day_equity
                    new_positions = {}
                    
                    for symbol, target_amount in target_allocation.items():
                        # Adjust proportionally
                        adjusted_amount = target_amount * (day_equity / sum(target_allocation.values()))
                        new_positions[symbol] = adjusted_amount
                        remaining_equity -= adjusted_amount
                    
                    # Add any remaining equity to cash
                    if abs(remaining_equity) > 0.01:
                        new_positions[CASH_ETF] = new_positions.get(CASH_ETF, 0) + remaining_equity
                    
                    # Update current positions
                    current_positions = new_positions
                    
                    # Record the trades
                    trade_history.append({
                        'date': date,
                        'action': 'REBALANCE',
                        'trades': trades,
                        'total_cost': total_cost
                    })
            
            # Record positions for the day
            positions_history.append({
                'date': date,
                'positions': current_positions.copy(),
                'equity': day_equity
            })
            
            # Update equity curve
            equity_curve.append((date, day_equity))
        
        # Convert equity curve to a Series
        equity_series = pd.Series(dict(equity_curve))
        self.equity_curve = equity_series
        
        # Save positions history and trade history
        self.positions_history = pd.DataFrame(positions_history)
        self.positions_history.set_index('date', inplace=True)
        
        self.trade_history = trade_history
        
        # Calculate and save portfolio statistics
        self.calculate_portfolio_stats()
        
        return equity_series
    
    def calculate_portfolio_stats(self) -> Dict:
        """
        Calculate portfolio statistics.
        
        Returns:
            Dictionary with portfolio statistics
        """
        if self.equity_curve is None:
            raise ValueError("Backtest must be run before calculating statistics")
        
        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()
        
        # Calculate CAGR
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        cagr = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (1 / years) - 1
        
        # Calculate volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (cagr / annual_vol) if annual_vol > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        drawdowns = 1 - cumulative_returns / cumulative_returns.cummax()
        max_drawdown = drawdowns.max()
        
        # Calculate Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (cagr / downside_deviation) if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio (CAGR / Max DD)
        calmar = cagr / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calculate Ulcer Index
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        # Calculate win/loss metrics
        win_months = len(returns.resample('M').sum()[returns.resample('M').sum() > 0])
        loss_months = len(returns.resample('M').sum()[returns.resample('M').sum() <= 0])
        total_months = win_months + loss_months
        win_rate = win_months / total_months if total_months > 0 else 0
        
        # Calculate number of trades
        num_trades = len([t for t in self.trade_history if t['action'] == 'REBALANCE'])
        
        stats = {
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve.iloc[-1],
            'total_return': self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'ulcer_index': ulcer_index,
            'win_rate_monthly': win_rate,
            'num_trades': num_trades,
            'avg_trades_per_year': num_trades / years
        }
        
        self.portfolio_stats = stats
        return stats
    
    def plot_equity_curve(self, benchmark_symbol: str = RISK_ON, save_path: Optional[str] = None) -> None:
        """
        Plot the equity curve with a benchmark for comparison.
        
        Args:
            benchmark_symbol: Symbol to use as benchmark
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if self.equity_curve is None:
            raise ValueError("Backtest must be run before plotting")
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        equity_norm = self.equity_curve / self.equity_curve.iloc[0]
        equity_norm.plot(label='Strategy')
        
        # Plot benchmark
        if benchmark_symbol in self.prices.columns:
            benchmark = self.prices[benchmark_symbol].loc[self.equity_curve.index]
            benchmark_norm = benchmark / benchmark.iloc[0]
            benchmark_norm.plot(label=f'Benchmark ({benchmark_symbol})')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Growth of $1')
        plt.title('Equity Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics as text
        if self.portfolio_stats:
            stats_text = (
                f"CAGR: {self.portfolio_stats['cagr']:.2%}\n"
                f"Sharpe: {self.portfolio_stats['sharpe_ratio']:.2f}\n"
                f"Sortino: {self.portfolio_stats['sortino_ratio']:.2f}\n"
                f"Max DD: {self.portfolio_stats['max_drawdown']:.2%}\n"
                f"Trades/Year: {self.portfolio_stats['avg_trades_per_year']:.1f}"
            )
            plt.figtext(0.01, 0.01, stats_text, fontsize=10)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_asset_allocation(self, freq: str = 'M', save_path: Optional[str] = None) -> None:
        """
        Plot the asset allocation over time.
        
        Args:
            freq: Frequency for resampling positions (e.g., 'M' for monthly)
            save_path: Path to save the plot (if None, plot is displayed)
        """
        if self.positions_history is None:
            raise ValueError("Backtest must be run before plotting")
        
        # Create a DataFrame with positions
        positions_df = pd.DataFrame()
        
        for symbol in self.tickers:
            positions_df[symbol] = self.positions_history['positions'].apply(
                lambda x: x.get(symbol, 0)
            )
        
        # Resample if requested
        if freq:
            positions_df = positions_df.resample(freq).last()
        
        # Ensure all values are positive for stacked area plot
        # Replace any negative values with 0 to avoid plotting errors
        positions_df = positions_df.clip(lower=0)
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Plot stacked area chart
        positions_df.plot.area(stacked=True, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Allocation')
        plt.title('Asset Allocation Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
    
    def save_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save backtest results to CSV files.
        
        Args:
            output_dir: Directory to save results (if None, uses default)
            
        Returns:
            Path to the output directory
        """
        if self.equity_curve is None:
            raise ValueError("Backtest must be run before saving results")
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(
                BACKTEST_RESULTS_DIR,
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save equity curve
        self.equity_curve.to_csv(os.path.join(output_dir, 'equity_curve.csv'))
        
        # Save positions history
        self.positions_history.to_csv(os.path.join(output_dir, 'positions_history.csv'))
        
        # Save trade history
        pd.DataFrame(self.trade_history).to_csv(os.path.join(output_dir, 'trade_history.csv'), index=False)
        
        # Save portfolio stats
        if self.portfolio_stats:
            pd.Series(self.portfolio_stats).to_csv(os.path.join(output_dir, 'portfolio_stats.csv'))
        
        # Save plots
        self.plot_equity_curve(save_path=os.path.join(output_dir, 'equity_curve.png'))
        self.plot_asset_allocation(save_path=os.path.join(output_dir, 'asset_allocation.png'))
        
        return output_dir
