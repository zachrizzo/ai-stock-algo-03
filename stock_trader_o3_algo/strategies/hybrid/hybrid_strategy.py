#!/usr/bin/env python3
"""
Hybrid Strategy that combines Tri-Shot, DMT, and TurboQT strategies.

This strategy allocates capital dynamically between the three strategies
based on their recent performance and market conditions.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import strategy-specific modules
from stock_trader_o3_algo.strategies.tri_shot.tri_shot import TICKERS as TRI_SHOT_TICKERS
from stock_trader_o3_algo.strategies.tri_shot.tri_shot import calculate_vol_weight, check_for_crash
from stock_trader_o3_algo.strategies.tri_shot.tri_shot_model import load_walk_forward_model
from stock_trader_o3_algo.strategies.tri_shot.tri_shot_features import latest_features as generate_features

# Use the core strategy module for DMT and TurboQT since direct imports aren't available
from stock_trader_o3_algo.core.strategy import get_portfolio_allocation, CASH_ETF
from stock_trader_o3_algo.core.performance import calculate_performance_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default strategy weights (can be adjusted dynamically)
DEFAULT_WEIGHTS = {
    'tri_shot': 0.4,
    'dmt': 0.3,
    'turbo_qt': 0.3
}

def detect_market_regime(prices, lookback=60):
    """
    Detect the current market regime based on price action.
    
    Returns:
        str: 'bull', 'bear', or 'neutral'
    """
    if 'QQQ' not in prices.columns or len(prices) < lookback:
        return 'neutral'  # Not enough data
        
    # Get QQQ for market reference
    qqq = prices['QQQ'].iloc[-lookback:]
    returns = qqq.pct_change().dropna()
    
    # Calculate metrics
    trend = qqq.iloc[-1] / qqq.iloc[0] - 1  # Overall trend
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate 50-day and 200-day moving averages
    ma50 = qqq.rolling(min(50, len(qqq))).mean().iloc[-1]
    ma200 = qqq.rolling(min(200, len(qqq))).mean().iloc[-1]
    
    # Calculate VIX if available
    vix_high = False
    if '^VIX' in prices.columns:
        vix = prices['^VIX'].iloc[-5:].mean()  # 5-day average
        vix_high = vix > 25  # High volatility threshold
    
    # Determine regime
    if trend > 0.05 and ma50 > ma200 and not vix_high:
        return 'bull'
    elif trend < -0.05 or (ma50 < ma200 and vix_high):
        return 'bear'
    else:
        return 'neutral'

def get_tri_shot_allocation(prices, date=None, equity=10000.0, equity_peak=None, equity_curve=None):
    """
    Get the Tri-Shot strategy allocation based on the model prediction.
    This implementation follows a similar logic to run_monday_strategy but returns allocation
    instead of placing orders.
    """
    if date is None:
        date = prices.index[-1]
    
    # Get subset of data up to the current date
    data = prices.loc[:date]
    
    # Load the model
    model_path = os.path.join(os.getcwd(), "tri_shot_data/tri_shot_ensemble.pkl")
    try:
        model = load_walk_forward_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {CASH_ETF: equity}  # Default to cash on error
    
    # Generate features
    try:
        X = generate_features(data)
        
        # Check if we have features for the current date
        if X is None or X.empty or date not in X.index:
            logger.warning(f"No features generated for date {date}")
            return {CASH_ETF: equity}
            
        # Get prediction for the current date
        proba_up = model.predict_proba(X.loc[date].values.reshape(1, -1))[0][1]
        logger.info(f"Tri-Shot probability up: {proba_up:.4f}")
        
        # Check for crash protection
        is_crash = check_for_crash(data)
        
        # Determine position based on prediction
        if is_crash:
            # Crash protection: 40% short, 60% cash
            allocation = {
                TRI_SHOT_TICKERS["DN"]: equity * 0.4,  # SQQQ
                CASH_ETF: equity * 0.6
            }
        elif proba_up > 0.60:  # Bullish
            # Calculate position weight
            weight = calculate_vol_weight(data, TRI_SHOT_TICKERS["UP"], proba_up)
            allocation = {
                TRI_SHOT_TICKERS["UP"]: equity * weight,  # TQQQ
                CASH_ETF: equity * (1 - weight)
            }
        elif proba_up < 0.40:  # Bearish
            # Calculate position weight
            weight = calculate_vol_weight(data, TRI_SHOT_TICKERS["DN"], 1 - proba_up)
            allocation = {
                TRI_SHOT_TICKERS["DN"]: equity * weight,  # SQQQ
                CASH_ETF: equity * (1 - weight)
            }
        else:  # Neutral
            # Go to bonds or cash
            if proba_up >= 0.45 and proba_up <= 0.55:
                # More neutral = bonds
                weight = calculate_vol_weight(data, TRI_SHOT_TICKERS["BOND"], 0.5)
                allocation = {
                    TRI_SHOT_TICKERS["BOND"]: equity * weight,  # TMF
                    CASH_ETF: equity * (1 - weight)
                }
            else:
                # Less conviction = cash
                allocation = {CASH_ETF: equity}
        
        return allocation
        
    except Exception as e:
        logger.error(f"Error in tri_shot allocation: {e}")
        return {CASH_ETF: equity}  # Default to cash on error

def get_dmt_allocation(prices, date=None, equity=10000.0, equity_peak=None, equity_curve=None):
    """
    Simplified DMT strategy allocation.
    Since we don't have direct access to the DMT allocation function, we'll adapt the
    core strategy get_portfolio_allocation function to mimic DMT behavior.
    """
    try:
        # For DMT, focus on a momentum-based approach
        # This is a simplified version since we don't have access to the actual DMT model
        
        if date is None:
            date = prices.index[-1]
        
        # Get QQQ data
        if 'QQQ' not in prices.columns:
            return {CASH_ETF: equity}
            
        # Calculate short-term momentum (10 days)
        momentum_10d = prices['QQQ'].pct_change(10).loc[date]
        
        # Calculate medium-term momentum (30 days)
        momentum_30d = prices['QQQ'].pct_change(30).loc[date]
        
        # Determine direction based on momentum
        if momentum_10d > 0 and momentum_30d > 0:
            # Strong uptrend - go long
            weight = min(0.9, 0.5 + (momentum_30d * 5))  # Scale weight by momentum
            allocation = {
                'TQQQ': equity * weight,
                CASH_ETF: equity * (1 - weight)
            }
        elif momentum_10d < 0 and momentum_30d < 0:
            # Strong downtrend - go short
            weight = min(0.8, 0.4 + (abs(momentum_30d) * 4))  # Scale weight by momentum
            allocation = {
                'SQQQ': equity * weight,
                CASH_ETF: equity * (1 - weight)
            }
        elif momentum_10d > 0 and momentum_30d < 0:
            # Possible trend reversal up - small long position
            allocation = {
                'QQQ': equity * 0.3,
                'TLT': equity * 0.3,
                CASH_ETF: equity * 0.4
            }
        elif momentum_10d < 0 and momentum_30d > 0:
            # Possible trend reversal down - defensive
            allocation = {
                'TLT': equity * 0.4,
                CASH_ETF: equity * 0.6
            }
        else:
            # Neutral - go to cash
            allocation = {CASH_ETF: equity}
            
        return allocation
        
    except Exception as e:
        logger.error(f"Error in DMT allocation: {e}")
        return {CASH_ETF: equity}  # Default to cash on error

def get_turbo_qt_allocation(prices, date=None, equity=10000.0, equity_peak=None, equity_curve=None):
    """
    Simplified TurboQT strategy allocation.
    This implements a rotational strategy based on relative strength.
    """
    try:
        if date is None:
            date = prices.index[-1]
            
        # Get a subset of the data up to the current date
        data = prices.loc[:date]
        
        # Define the candidate assets
        candidates = ['QQQ', 'TQQQ', 'TLT', 'TMF']
        
        # Check if we have the required data
        for ticker in candidates:
            if ticker not in data.columns:
                logger.warning(f"Missing ticker {ticker} in data")
                return {CASH_ETF: equity}
                
        # Calculate relative strength (20-day return)
        returns_20d = {}
        for ticker in candidates:
            if len(data[ticker]) >= 20:
                returns_20d[ticker] = data[ticker].pct_change(20).iloc[-1]
            else:
                returns_20d[ticker] = 0
                
        # Calculate volatility
        volatility = {}
        for ticker in candidates:
            if len(data[ticker]) >= 20:
                volatility[ticker] = data[ticker].pct_change().iloc[-20:].std() * (252 ** 0.5)
            else:
                volatility[ticker] = 1.0  # High volatility as default
                
        # Calculate risk-adjusted returns
        risk_adjusted = {}
        for ticker in candidates:
            if volatility[ticker] > 0:
                risk_adjusted[ticker] = returns_20d[ticker] / volatility[ticker]
            else:
                risk_adjusted[ticker] = 0
                
        # Check if markets are under stress (VIX > 30)
        market_stress = False
        if '^VIX' in data.columns:
            vix_level = data['^VIX'].iloc[-1]
            market_stress = vix_level > 30
            
        # Find the best performing asset
        best_ticker = max(risk_adjusted.items(), key=lambda x: x[1])[0]
        best_return = returns_20d[best_ticker]
        
        # Determine allocation
        if market_stress or (best_return < 0 and abs(best_return) > 0.05):
            # Defensive positioning during market stress
            allocation = {
                'TLT': equity * 0.3,
                CASH_ETF: equity * 0.7
            }
        elif best_return > 0:
            # Allocate to the best performing asset with volatility-based sizing
            target_vol = 0.15  # Target portfolio volatility
            weight = min(0.9, target_vol / volatility[best_ticker])
            
            allocation = {
                best_ticker: equity * weight,
                CASH_ETF: equity * (1 - weight)
            }
        else:
            # Default to cash if no clear winners
            allocation = {CASH_ETF: equity}
            
        return allocation
        
    except Exception as e:
        logger.error(f"Error in TurboQT allocation: {e}")
        return {CASH_ETF: equity}  # Default to cash on error

def get_strategy_performance(prices, lookback=60):
    """
    Calculate the recent performance of each strategy.
    
    Returns:
        dict: Dict with performance metrics for each strategy
    """
    end_date = prices.index[-1]
    start_date = prices.index[max(0, len(prices) - lookback)]
    
    # Run each strategy for the lookback period
    performance = {}
    
    # Get historical performance data if available
    performance_file = os.path.join(os.path.dirname(__file__), '../../../tri_shot_data/strategy_performance.csv')
    if os.path.exists(performance_file):
        try:
            perf_df = pd.read_csv(performance_file, index_col=0, parse_dates=True)
            # Filter to lookback period
            perf_df = perf_df.loc[start_date:end_date]
            
            if not perf_df.empty:
                # Calculate performance metrics for each strategy
                for strategy in ['tri_shot', 'dmt', 'turbo_qt']:
                    if strategy in perf_df.columns:
                        equity_curve = perf_df[strategy]
                        metrics = calculate_performance_metrics(equity_curve)
                        performance[strategy] = metrics
                        
        except Exception as e:
            logger.warning(f"Error reading performance data: {e}")
    
    # Default metrics if we couldn't get historical data
    if not performance:
        performance = {
            'tri_shot': {'sharpe': 2.5, 'cagr': 0.5, 'max_drawdown': 0.15},
            'dmt': {'sharpe': 1.5, 'cagr': 0.25, 'max_drawdown': 0.12},
            'turbo_qt': {'sharpe': 1.0, 'cagr': 0.2, 'max_drawdown': 0.2}
        }
        
    return performance

def get_strategy_weights(prices, regime=None, performance=None):
    """
    Calculate optimal weights for each strategy based on market regime and performance.
    
    Returns:
        dict: Dictionary with strategy weights that sum to 1.0
    """
    # Use detected regime if not provided
    if regime is None:
        regime = detect_market_regime(prices)
        
    # Use calculated performance if not provided
    if performance is None:
        performance = get_strategy_performance(prices)
        
    # Base weights by regime
    if regime == 'bull':
        weights = {
            'tri_shot': 0.5,   # Aggressive in bull markets
            'dmt': 0.3,
            'turbo_qt': 0.2
        }
    elif regime == 'bear':
        weights = {
            'tri_shot': 0.2,
            'dmt': 0.3,
            'turbo_qt': 0.5    # More defensive in bear markets
        }
    else:  # neutral
        weights = {
            'tri_shot': 0.33,
            'dmt': 0.34,
            'turbo_qt': 0.33
        }
        
    # Adjust weights based on recent performance
    # Calculate risk-adjusted return scores
    scores = {}
    for strategy, metrics in performance.items():
        # Use Sharpe ratio as primary metric, with CAGR as secondary
        # Penalize for excessive drawdowns
        score = metrics.get('sharpe', 1.0) * 0.6 + metrics.get('cagr', 0.1) * 10 * 0.4
        score *= (1 - min(0.5, metrics.get('max_drawdown', 0) * 2))  # Drawdown penalty
        scores[strategy] = max(0.5, score)  # Minimum score of 0.5
        
    # Normalize scores to sum to 1.0
    total_score = sum(scores.values())
    if total_score > 0:
        normalized_scores = {k: v / total_score for k, v in scores.items()}
        
        # Blend base weights with performance-based weights (50/50)
        for strategy in weights:
            weights[strategy] = 0.5 * weights[strategy] + 0.5 * normalized_scores.get(strategy, 0.0)
            
    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
        
    return weights

def get_hybrid_allocation(prices, date=None, equity=10000.0, equity_peak=None, equity_curve=None):
    """
    Get the portfolio allocation for the hybrid strategy.
    
    Args:
        prices: DataFrame of asset prices
        date: Reference date for allocation
        equity: Current portfolio equity value
        equity_peak: Peak equity value (for drawdown calculation)
        equity_curve: Full equity curve (for stop-loss calculation)
        
    Returns:
        Dictionary with asset symbols as keys and dollar allocations as values
    """
    # Use the latest date if not specified
    if date is None:
        date = prices.index[-1]
        
    # Detect market regime
    regime = detect_market_regime(prices)
    logger.info(f"Detected market regime: {regime}")
    
    # Get strategy performance
    performance = get_strategy_performance(prices)
    
    # Calculate strategy weights
    weights = get_strategy_weights(prices, regime, performance)
    logger.info(f"Strategy weights: {weights}")
    
    # Get allocations from individual strategies
    tri_shot_alloc = get_tri_shot_allocation(
        prices, date, equity=equity * weights['tri_shot'], 
        equity_peak=equity_peak, equity_curve=equity_curve
    )
    
    dmt_alloc = get_dmt_allocation(
        prices, date, equity=equity * weights['dmt'], 
        equity_peak=equity_peak, equity_curve=equity_curve
    )
    
    turbo_qt_alloc = get_turbo_qt_allocation(
        prices, date, equity=equity * weights['turbo_qt'], 
        equity_peak=equity_peak, equity_curve=equity_curve
    )
    
    # Combine allocations
    combined_alloc = {}
    for alloc in [tri_shot_alloc, dmt_alloc, turbo_qt_alloc]:
        for symbol, amount in alloc.items():
            if symbol in combined_alloc:
                combined_alloc[symbol] += amount
            else:
                combined_alloc[symbol] = amount
                
    # Check for risk limits - if current drawdown is severe, reduce exposure
    current_drawdown = 0
    if equity_peak is not None and equity_peak > 0:
        current_drawdown = 1 - (equity / equity_peak)
        
    # Apply drawdown protection
    if current_drawdown > 0.2:  # More than 20% drawdown
        # Move more to cash as drawdown increases
        cash_pct = min(0.9, current_drawdown * 2)  # Max 90% cash
        
        # Calculate current cash allocation
        current_cash = combined_alloc.get(CASH_ETF, 0)
        cash_target = equity * cash_pct
        
        if cash_target > current_cash:
            # Proportionally reduce other positions
            non_cash = {k: v for k, v in combined_alloc.items() if k != CASH_ETF}
            total_non_cash = sum(non_cash.values())
            
            if total_non_cash > 0:
                reduction = cash_target - current_cash
                for symbol, amount in non_cash.items():
                    reduce_amount = (amount / total_non_cash) * reduction
                    combined_alloc[symbol] -= reduce_amount
                
                # Update cash allocation
                combined_alloc[CASH_ETF] = cash_target
    
    return combined_alloc

def run_hybrid_strategy(prices=None, capital=10000.0):
    """
    Run the hybrid strategy with current market data.
    
    Args:
        prices: Prices DataFrame (if None, will fetch latest data)
        capital: Starting capital
        
    Returns:
        dict: Allocation dictionary
    """
    # Get data if not provided
    if prices is None:
        from stock_trader_o3_algo.strategies.tri_shot.tri_shot_features import fetch_data
        tickers = ["QQQ", "TQQQ", "SQQQ", "TMF", "TLT", "^VIX"]
        prices = fetch_data(tickers, days=365)
    
    logger.info("Running hybrid strategy allocation")
    allocation = get_hybrid_allocation(prices, equity=capital)
    
    # Log allocation
    logger.info("Final allocation:")
    for symbol, amount in allocation.items():
        logger.info(f"  {symbol}: ${amount:.2f} ({amount/capital*100:.1f}%)")
        
    return allocation

if __name__ == "__main__":
    # Run the strategy with default parameters
    run_hybrid_strategy()
