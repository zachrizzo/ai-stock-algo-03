"""
Performance metrics calculation for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple

def calculate_performance_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate common performance metrics from an equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Dictionary of performance metrics
    """
    if isinstance(equity_curve, pd.DataFrame):
        # If passed a DataFrame, convert to Series
        if 'equity' in equity_curve.columns:
            equity_curve = equity_curve['equity']
        else:
            equity_curve = equity_curve.iloc[:, 0]
    
    # Ensure the equity curve has datetime index
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("Equity curve must have a DatetimeIndex")
        
    # Check for empty data
    if len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'final_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else 0.0
        }
        
    # Basic metrics
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    total_return = final_value / initial_value - 1
    
    # Calculate CAGR (Compound Annual Growth Rate)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        days = 1  # Avoid division by zero
    years = days / 365.25
    cagr = (final_value / initial_value) ** (1 / max(years, 0.01)) - 1
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Risk metrics
    daily_std = returns.std()
    annualized_vol = daily_std * np.sqrt(252)  # Assuming 252 trading days per year
    
    # Max drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak - 1)
    max_drawdown = drawdown.min()
    
    # Risk-adjusted returns
    excess_return = cagr - risk_free_rate
    sharpe_ratio = excess_return / max(annualized_vol, 0.0001)
    
    # Sortino ratio (only considering negative returns)
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
    sortino_ratio = excess_return / max(downside_deviation, 0.0001)
    
    # Win rate
    wins = len(returns[returns > 0])
    losses = len(returns[returns < 0])
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Average win and loss
    avg_win = returns[returns > 0].mean() if wins > 0 else 0
    avg_loss = returns[returns < 0].mean() if losses > 0 else 0
    
    # Calmar ratio
    calmar = cagr / max(abs(max_drawdown), 0.0001)
    
    # Return all metrics
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': annualized_vol,
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_value': final_value
    }

def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown statistics from an equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        DataFrame with drawdown statistics
    """
    # Calculate drawdown series
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak - 1) * 100  # Convert to percentage
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    
    # Find start and end of each drawdown period
    drawdown_starts = []
    drawdown_ends = []
    drawdown_depths = []
    in_drawdown = False
    
    for i, (date, is_dd) in enumerate(is_drawdown.items()):
        if is_dd and not in_drawdown:
            # Start of a drawdown period
            drawdown_starts.append(date)
            in_drawdown = True
        elif not is_dd and in_drawdown:
            # End of a drawdown period
            drawdown_ends.append(date)
            # Find the depth of this drawdown
            start_idx = equity_curve.index.get_loc(drawdown_starts[-1])
            end_idx = equity_curve.index.get_loc(date)
            period_drawdown = drawdown.iloc[start_idx:end_idx+1]
            drawdown_depths.append(period_drawdown.min())
            in_drawdown = False
            
    # If still in drawdown at the end
    if in_drawdown:
        drawdown_ends.append(equity_curve.index[-1])
        start_idx = equity_curve.index.get_loc(drawdown_starts[-1])
        period_drawdown = drawdown.iloc[start_idx:]
        drawdown_depths.append(period_drawdown.min())
        
    # Create result DataFrame
    result = pd.DataFrame({
        'start_date': drawdown_starts,
        'end_date': drawdown_ends,
        'depth_pct': drawdown_depths,
        'duration_days': [(end - start).days for start, end in zip(drawdown_starts, drawdown_ends)]
    })
    
    # Sort by depth
    result = result.sort_values('depth_pct')
    
    return result

def calculate_rolling_returns(equity_curve: pd.Series, windows: list = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Calculate rolling returns for various time periods.
    
    Args:
        equity_curve: Series of portfolio values over time
        windows: List of rolling windows in months
        
    Returns:
        DataFrame with rolling returns
    """
    # Convert windows from months to days (approximate)
    window_days = [int(w * 30.4) for w in windows]
    
    # Calculate rolling returns
    result = pd.DataFrame(index=equity_curve.index)
    
    for months, days in zip(windows, window_days):
        if len(equity_curve) > days:
            rolling_return = equity_curve / equity_curve.shift(days) - 1
            result[f'{months}m'] = rolling_return
            
    return result

def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns from the equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        DataFrame with monthly returns
    """
    # Resample to month-end and calculate return
    monthly = equity_curve.resample('M').last()
    monthly_returns = monthly.pct_change().dropna()
    
    # Create a more readable DataFrame
    result = pd.DataFrame({
        'month': monthly_returns.index,
        'return': monthly_returns.values
    })
    
    # Add year and month columns
    result['year'] = result['month'].dt.year
    result['month_name'] = result['month'].dt.strftime('%b')
    
    return result

def compare_strategies(equity_curves: Dict[str, pd.Series], initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Args:
        equity_curves: Dictionary of equity curves with strategy names as keys
        initial_capital: Initial capital for each strategy
        
    Returns:
        DataFrame with performance comparison
    """
    result = {}
    
    for name, curve in equity_curves.items():
        metrics = calculate_performance_metrics(curve)
        result[name] = metrics
        
    # Convert to DataFrame for easy comparison
    comparison = pd.DataFrame(result).T
    
    # Reorder columns for better readability
    column_order = [
        'total_return', 'cagr', 'volatility', 'sharpe', 'sortino',
        'max_drawdown', 'calmar', 'win_rate', 'final_value'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in comparison.columns:
            comparison[col] = np.nan
            
    # Return formatted comparison
    return comparison[column_order]
