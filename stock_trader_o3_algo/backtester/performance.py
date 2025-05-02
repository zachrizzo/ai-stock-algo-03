#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtester Performance Metrics
=============================
Functions for calculating performance metrics from backtest results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Tuple


def calculate_performance_metrics(
    results: pd.DataFrame,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from backtest results
    
    Args:
        results: DataFrame with equity curve and returns
        initial_capital: Starting capital for the simulation
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods in a year (252 for daily data)
        
    Returns:
        Dictionary with performance metrics
    """
    equity = results['equity'] if 'equity' in results.columns else results['Close']
    
    # Calculate returns
    returns = equity.pct_change().dropna()
    
    # Basic performance metrics
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    
    # Calculate time in days/years
    days = (results.index[-1] - results.index[0]).days
    years = days / 365.25
    
    # Annualized metrics
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Calculate Sharpe ratio
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    sharpe_ratio = excess_return / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
    
    # Calculate drawdown stats
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Calculate Calmar ratio (CAGR / Max Drawdown)
    calmar_ratio = -cagr / max_drawdown if max_drawdown != 0 else float('inf')
    
    # Calculate Sortino ratio (penalizes only downside volatility)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return * periods_per_year / downside_deviation if downside_deviation > 0 else 0
    
    # Win/loss metrics
    trades = returns[returns != 0]
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def safe_improvement(new_value: float, old_value: float) -> float:
    """
    Calculate improvement percentage safely, avoiding division by zero
    
    Args:
        new_value: New metric value
        old_value: Old metric value
        
    Returns:
        Percentage improvement
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
        
    return (new_value - old_value) / abs(old_value) * 100


def compare_strategies(
    results_dict: Dict[str, pd.DataFrame],
    benchmark_key: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple strategy results and calculate relative performance
    
    Args:
        results_dict: Dictionary of strategy results DataFrames
        benchmark_key: Key of the benchmark strategy (default: None)
        metrics: List of metrics to compare (default: all available)
        
    Returns:
        DataFrame with comparative metrics
    """
    # Calculate metrics for each strategy
    metrics_dict = {}
    for strategy_name, results in results_dict.items():
        metrics_dict[strategy_name] = calculate_performance_metrics(results)
    
    # Create DataFrame from metrics
    comparison = pd.DataFrame(metrics_dict)
    
    # Calculate relative performance if benchmark is provided
    if benchmark_key is not None and benchmark_key in metrics_dict:
        benchmark_metrics = metrics_dict[benchmark_key]
        
        for strategy_name in results_dict.keys():
            if strategy_name != benchmark_key:
                for metric in metrics_dict[strategy_name].keys():
                    relative_metric = f"{strategy_name}_vs_{benchmark_key}_{metric}"
                    comparison.loc[relative_metric] = safe_improvement(
                        metrics_dict[strategy_name][metric],
                        benchmark_metrics[metric]
                    )
    
    return comparison


def calculate_regime_performance(
    results: pd.DataFrame, 
    regime_col: str = 'regime'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics by market regime
    
    Args:
        results: DataFrame with backtest results and regime labels
        regime_col: Name of the column with regime labels
        
    Returns:
        Dictionary with performance metrics by regime
    """
    regime_performance = {}
    
    # Get unique regimes
    if regime_col not in results.columns:
        return regime_performance
    
    regimes = results[regime_col].unique()
    
    # Calculate metrics for each regime
    for regime in regimes:
        regime_data = results[results[regime_col] == regime].copy()
        
        # Skip if not enough data
        if len(regime_data) < 5:
            continue
            
        regime_performance[regime] = calculate_performance_metrics(regime_data)
        regime_performance[regime]['days'] = len(regime_data)
        regime_performance[regime]['pct_days'] = len(regime_data) / len(results)
        
    return regime_performance
