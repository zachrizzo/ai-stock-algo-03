"""
Utility functions for calculating performance metrics.
"""
import numpy as np
import pandas as pd


def calculate_cagr(equity_curve: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        equity_curve: Series with equity values indexed by date
        
    Returns:
        CAGR as a decimal
    """
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return 0
    
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def calculate_drawdowns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate a series of drawdowns.
    
    Args:
        equity_curve: Series with equity values indexed by date
        
    Returns:
        Series with drawdown values
    """
    roll_max = equity_curve.cummax()
    drawdowns = 1 - equity_curve / roll_max
    return drawdowns


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series with equity values indexed by date
        
    Returns:
        Maximum drawdown as a decimal
    """
    drawdowns = calculate_drawdowns(equity_curve)
    return drawdowns.max()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series with return values
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Series with return values
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    
    return excess_returns.mean() * periods_per_year / downside_std if downside_std > 0 else 0


def calculate_calmar_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        equity_curve: Series with equity values indexed by date
        risk_free_rate: Annualized risk-free rate
        
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    
    return (cagr - risk_free_rate) / max_dd if max_dd > 0 else float('inf')


def calculate_ulcer_index(equity_curve: pd.Series) -> float:
    """
    Calculate Ulcer Index.
    
    Args:
        equity_curve: Series with equity values indexed by date
        
    Returns:
        Ulcer Index
    """
    drawdowns = calculate_drawdowns(equity_curve)
    return np.sqrt(np.mean(drawdowns ** 2))


def calculate_win_rate(returns: pd.Series, period: str = 'M') -> float:
    """
    Calculate win rate over a specified period.
    
    Args:
        returns: Series with return values
        period: Period to calculate win rate over ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Win rate as a decimal
    """
    period_returns = returns.resample(period).sum()
    wins = len(period_returns[period_returns > 0])
    total = len(period_returns)
    
    return wins / total if total > 0 else 0


def generate_performance_report(equity_curve: pd.Series, benchmark: pd.Series = None) -> pd.DataFrame:
    """
    Generate a performance report.
    
    Args:
        equity_curve: Series with equity values indexed by date
        benchmark: Series with benchmark values indexed by date
        
    Returns:
        DataFrame with performance metrics
    """
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Calculate benchmark returns if provided
    if benchmark is not None:
        benchmark_returns = benchmark.pct_change().dropna()
        # Align benchmark returns with strategy returns
        benchmark_returns = benchmark_returns.reindex(returns.index)
    else:
        benchmark_returns = None
    
    # Calculate performance metrics
    metrics = {
        'Total Return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        'CAGR': calculate_cagr(equity_curve),
        'Annualized Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(equity_curve),
        'Calmar Ratio': calculate_calmar_ratio(equity_curve),
        'Ulcer Index': calculate_ulcer_index(equity_curve),
        'Win Rate (Monthly)': calculate_win_rate(returns, 'M'),
        'Win Rate (Weekly)': calculate_win_rate(returns, 'W')
    }
    
    # Calculate benchmark metrics if provided
    if benchmark_returns is not None:
        benchmark_metrics = {
            'Benchmark Total Return': benchmark.iloc[-1] / benchmark.iloc[0] - 1,
            'Benchmark CAGR': calculate_cagr(benchmark),
            'Benchmark Annualized Volatility': benchmark_returns.std() * np.sqrt(252),
            'Benchmark Max Drawdown': calculate_max_drawdown(benchmark),
            'Beta': returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 0,
            'Alpha': calculate_cagr(equity_curve) - (calculate_cagr(benchmark) * returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 0)
        }
        metrics.update(benchmark_metrics)
    
    # Convert metrics to DataFrame
    return pd.DataFrame({'Value': metrics})
