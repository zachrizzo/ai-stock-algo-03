#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtester Visualization
=======================
Functions for visualizing backtest results and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Union
import os


def plot_results(
    results_dict: Dict[str, pd.DataFrame],
    title: str = "Strategy Comparison",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    include_drawdowns: bool = True
) -> plt.Figure:
    """
    Plot comparative results of different strategy versions
    
    Args:
        results_dict: Dictionary of DataFrames with backtest results
        title: Plot title
        filename: File path to save the plot (optional)
        figsize: Figure size as (width, height)
        include_drawdowns: Whether to include drawdown subplot
    
    Returns:
        Matplotlib figure object
    """
    if include_drawdowns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot equity curves
    for name, results in results_dict.items():
        ax1.plot(results.index, results['equity'], label=f"{name}")
    
    # Format equity plot
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Calculate and plot drawdowns if requested
    if include_drawdowns:
        for name, results in results_dict.items():
            returns = results['equity'].pct_change().fillna(0)
            cum_returns = (1 + returns).cumprod()
            drawdowns = cum_returns / cum_returns.expanding().max() - 1
            ax2.plot(results.index, drawdowns, label=f"{name}")
        
        # Format drawdown plot
        ax2.set_title('Drawdowns', fontsize=14)
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 0.1)
        
        # Format dates on x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    else:
        ax1.set_xlabel('Date', fontsize=12)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot if filename is provided
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    return fig


def plot_regime_positions(
    results: pd.DataFrame,
    ticker: str,
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """
    Create detailed position analysis plot by market regime
    
    Args:
        results: DataFrame with backtest results
        ticker: Asset ticker symbol
        filename: File path to save the plot (optional)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=figsize, sharex=True, 
        gridspec_kw={'height_ratios': [3, 1, 1, 1]}
    )
    
    # Plot price and equity curves
    ax1.plot(results.index, results['Close'], label=f"{ticker} Price", color='gray', alpha=0.7)
    ax1.set_ylabel(f"{ticker} Price", fontsize=12)
    
    # Create twin axis for equity
    ax1_twin = ax1.twinx()
    ax1_twin.plot(results.index, results['equity'], label="Strategy", color='green')
    
    if 'buy_hold_equity' in results.columns:
        ax1_twin.plot(
            results.index, results['buy_hold_equity'], 
            label="Buy & Hold", color='blue', linestyle='--'
        )
    
    ax1_twin.set_ylabel('Portfolio Value', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title(f"{ticker} Strategy Performance & Market Regimes", fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Plot positions with regime background
    ax2.plot(results.index, results['position'], color='purple')
    ax2.set_ylabel('Position Size', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.grid(True, alpha=0.3)
    
    # Add regime background colors if regime column exists
    if 'regime' in results.columns:
        # Get regime transitions
        regime_changes = results['regime'].ne(results['regime'].shift()).cumsum()
        regimes = results.groupby(regime_changes).first()
        
        # Define colors for each regime
        regime_colors = {
            'Bull': 'lightgreen',
            'Neutral': 'lightyellow',
            'Bear': 'lightcoral'
        }
        
        # Add colored background for each regime period
        for i in range(len(regimes) - 1):
            start_date = regimes.index[i]
            end_date = regimes.index[i + 1]
            regime = results.loc[start_date, 'regime']
            color = regime_colors.get(regime, 'lightgray')
            
            # Add background color to all subplots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvspan(start_date, end_date, alpha=0.2, color=color)
        
        # Add legend for regimes
        import matplotlib.patches as mpatches
        patches = []
        for regime, color in regime_colors.items():
            patch = mpatches.Patch(color=color, alpha=0.2, label=regime)
            patches.append(patch)
        
        ax2.legend(handles=patches, loc='upper right')
    
    # Plot regime confidences if available
    has_regime_confidences = all(col in results.columns for col in ['bull_confidence', 'bear_confidence', 'neutral_confidence'])
    
    if has_regime_confidences:
        # Plot regime confidences
        ax3.plot(results.index, results['bull_confidence'], color='green', label='Bull')
        ax3.plot(results.index, results['bear_confidence'], color='red', label='Bear')
        ax3.plot(results.index, results['neutral_confidence'], color='gray', label='Neutral')
        ax3.set_ylabel('Regime Confidence', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        # If no regime confidences, plot returns instead
        returns = results['Close'].pct_change() * 100  # as percentage
        ax3.plot(results.index, returns, color='blue', label='Daily Returns')
        ax3.set_ylabel('Daily Returns (%)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    # Plot drawdowns
    returns = results['equity'].pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod()
    drawdowns = cum_returns / cum_returns.expanding().max() - 1
    
    # Plot buy & hold drawdowns if available
    if 'buy_hold_equity' in results.columns:
        bh_returns = results['buy_hold_equity'].pct_change().fillna(0)
        bh_cum_returns = (1 + bh_returns).cumprod()
        bh_drawdowns = bh_cum_returns / bh_cum_returns.expanding().max() - 1
        
        ax4.plot(results.index, bh_drawdowns, color='blue', linestyle='--', label='Buy & Hold')
    
    ax4.plot(results.index, drawdowns, color='red', label='Strategy')
    ax4.set_ylabel('Drawdown', fontsize=12)
    ax4.set_ylim(-1, 0.1)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot if filename is provided
    if filename:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Regime plot saved to: {filename}")
    
    return fig


def create_performance_summary(
    results_dict: Dict[str, pd.DataFrame],
    benchmark_key: Optional[str] = None,
    output_format: str = 'text'
) -> str:
    """
    Create a formatted performance summary from strategy results
    
    Args:
        results_dict: Dictionary of DataFrames with backtest results
        benchmark_key: Key of benchmark strategy for comparison (optional)
        output_format: Output format ('text', 'markdown', or 'html')
    
    Returns:
        Formatted performance summary string
    """
    from stock_trader_o3_algo.backtester.performance import calculate_performance_metrics
    
    # Calculate metrics for each strategy
    metrics = {}
    for name, results in results_dict.items():
        metrics[name] = calculate_performance_metrics(results)
    
    # Define the metrics to display
    display_metrics = [
        ('Total Return', 'total_return', '{:.2%}'),
        ('CAGR', 'cagr', '{:.2%}'),
        ('Volatility', 'volatility', '{:.2%}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
        ('Max Drawdown', 'max_drawdown', '{:.2%}'),
        ('Calmar Ratio', 'calmar_ratio', '{:.2f}'),
        ('Win Rate', 'win_rate', '{:.2%}'),
        ('Profit Factor', 'profit_factor', '{:.2f}')
    ]
    
    # Create headers
    if output_format == 'text':
        # ASCII table
        header = f"{'Metric':<20} " + " ".join(f"{name:<15}" for name in metrics.keys())
        separator = "-" * (20 + 15 * len(metrics))
        lines = [header, separator]
        
        # Add metric rows
        for display_name, metric_key, fmt in display_metrics:
            row = f"{display_name:<20} "
            for strategy in metrics.keys():
                value = metrics[strategy].get(metric_key, 0)
                row += fmt.format(value).ljust(15)
            lines.append(row)
            
        if benchmark_key and benchmark_key in metrics:
            # Add outperformance vs benchmark
            lines.append(separator)
            lines.append(f"{'vs ' + benchmark_key:<20}")
            
            for display_name, metric_key, fmt in display_metrics[:3]:  # Only show a few key metrics
                row = f"{display_name:<20} "
                benchmark_value = metrics[benchmark_key].get(metric_key, 0)
                
                for strategy in metrics.keys():
                    if strategy == benchmark_key:
                        row += "benchmark".ljust(15)
                    else:
                        value = metrics[strategy].get(metric_key, 0) - benchmark_value
                        row += (fmt.format(value) + " Δ").ljust(15)
                lines.append(row)
        
        return "\n".join(lines)
        
    elif output_format == 'markdown':
        # Markdown table
        header = f"| {'Metric':<18} | " + " | ".join(f"{name:<12}" for name in metrics.keys()) + " |"
        separator = f"|:{'-'*18}:|" + "|:".join(['-'*12 for _ in metrics.keys()]) + "|"
        lines = [header, separator]
        
        # Add metric rows
        for display_name, metric_key, fmt in display_metrics:
            row = f"| {display_name:<18} | "
            for strategy in metrics.keys():
                value = metrics[strategy].get(metric_key, 0)
                row += fmt.format(value).ljust(12) + " | "
            lines.append(row)
            
        if benchmark_key and benchmark_key in metrics:
            # Add outperformance vs benchmark
            lines.append(f"| {'**vs ' + benchmark_key + '**':<18} |" + " | ".join(["" for _ in metrics.keys()]) + " |")
            
            for display_name, metric_key, fmt in display_metrics[:3]:  # Only show a few key metrics
                row = f"| {display_name:<18} | "
                benchmark_value = metrics[benchmark_key].get(metric_key, 0)
                
                for strategy in metrics.keys():
                    if strategy == benchmark_key:
                        row += "benchmark".ljust(12) + " | "
                    else:
                        value = metrics[strategy].get(metric_key, 0) - benchmark_value
                        row += (fmt.format(value) + " Δ").ljust(12) + " | "
                lines.append(row)
        
        return "\n".join(lines)
        
    elif output_format == 'html':
        # HTML table
        lines = ['<table class="performance-table">']
        
        # Header row
        lines.append('<tr><th>Metric</th>')
        for name in metrics.keys():
            lines.append(f'<th>{name}</th>')
        lines.append('</tr>')
        
        # Add metric rows
        for display_name, metric_key, fmt in display_metrics:
            lines.append(f'<tr><td>{display_name}</td>')
            for strategy in metrics.keys():
                value = metrics[strategy].get(metric_key, 0)
                lines.append(f'<td>{fmt.format(value)}</td>')
            lines.append('</tr>')
            
        if benchmark_key and benchmark_key in metrics:
            # Add outperformance vs benchmark
            lines.append(f'<tr><td colspan="{len(metrics)+1}" class="separator"></td></tr>')
            lines.append(f'<tr><td><strong>vs {benchmark_key}</strong></td>')
            for _ in metrics.keys():
                lines.append('<td></td>')
            lines.append('</tr>')
            
            for display_name, metric_key, fmt in display_metrics[:3]:
                lines.append(f'<tr><td>{display_name}</td>')
                benchmark_value = metrics[benchmark_key].get(metric_key, 0)
                
                for strategy in metrics.keys():
                    if strategy == benchmark_key:
                        lines.append('<td>benchmark</td>')
                    else:
                        value = metrics[strategy].get(metric_key, 0) - benchmark_value
                        lines.append(f'<td>{fmt.format(value)} Δ</td>')
                lines.append('</tr>')
        
        lines.append('</table>')
        return "\n".join(lines)
        
    else:
        return "Unsupported output format. Use 'text', 'markdown', or 'html'."
