"""
Tests for the micro-CTA strategy.
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_trader_o3_algo.core.strategy import (
    select_candidate_asset,
    calculate_position_weight,
    check_crash_conditions,
    get_portfolio_allocation
)
from stock_trader_o3_algo.config.settings import (
    RISK_ON, RISK_OFF, CASH_ETF, HEDGE_ETF,
    WEEKLY_VOL_TARGET, HEDGE_WEIGHT
)


class TestMicroCTAStrategy(unittest.TestCase):
    """Test cases for the micro-CTA strategy."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=100)
        
        # Scenario 1: SPY trending up, TLT flat
        spy_prices1 = np.linspace(100, 120, 100)  # Uptrend
        tlt_prices1 = np.linspace(100, 100, 100)  # Flat
        
        # Scenario 2: SPY down, TLT up
        spy_prices2 = np.linspace(100, 90, 100)   # Downtrend
        tlt_prices2 = np.linspace(100, 110, 100)  # Uptrend
        
        # Scenario 3: Both down
        spy_prices3 = np.linspace(100, 90, 100)   # Downtrend
        tlt_prices3 = np.linspace(100, 95, 100)   # Downtrend
        
        # Create price DataFrames
        self.prices1 = pd.DataFrame({
            RISK_ON: spy_prices1,
            RISK_OFF: tlt_prices1,
            HEDGE_ETF: np.linspace(100, 90, 100),
            CASH_ETF: np.linspace(100, 101, 100)
        }, index=dates)
        
        self.prices2 = pd.DataFrame({
            RISK_ON: spy_prices2,
            RISK_OFF: tlt_prices2,
            HEDGE_ETF: np.linspace(100, 110, 100),
            CASH_ETF: np.linspace(100, 101, 100)
        }, index=dates)
        
        self.prices3 = pd.DataFrame({
            RISK_ON: spy_prices3,
            RISK_OFF: tlt_prices3,
            HEDGE_ETF: np.linspace(100, 110, 100),
            CASH_ETF: np.linspace(100, 101, 100)
        }, index=dates)
        
        # Create a price DataFrame with weekly crash
        self.crash_prices = self.prices1.copy()
        self.crash_prices.iloc[-1, self.crash_prices.columns.get_indexer([RISK_ON])] = 90  # Sharp drop
    
    def test_select_candidate_asset_spy_trend(self):
        """Test asset selection when SPY is trending up."""
        candidate = select_candidate_asset(self.prices1)
        self.assertEqual(candidate, RISK_ON)
    
    def test_select_candidate_asset_tlt_trend(self):
        """Test asset selection when TLT is trending up and SPY is down."""
        candidate = select_candidate_asset(self.prices2)
        self.assertEqual(candidate, RISK_OFF)
    
    def test_select_candidate_asset_defensive(self):
        """Test asset selection when both SPY and TLT are down."""
        candidate = select_candidate_asset(self.prices3)
        self.assertEqual(candidate, CASH_ETF)
    
    def test_calculate_position_weight(self):
        """Test position weight calculation based on volatility."""
        # For a simple test, we expect weight to be capped at 1.0
        weight = calculate_position_weight(self.prices1, RISK_ON)
        self.assertLessEqual(weight, 1.0)
        self.assertGreaterEqual(weight, 0.0)
        
        # Cash ETF should always have weight of 1.0
        cash_weight = calculate_position_weight(self.prices1, CASH_ETF)
        self.assertEqual(cash_weight, 1.0)
    
    def test_check_crash_conditions(self):
        """Test detection of crash conditions."""
        # Normal conditions
        self.assertFalse(check_crash_conditions(self.prices1))
        
        # Crash conditions
        self.assertTrue(check_crash_conditions(self.crash_prices))
    
    def test_get_portfolio_allocation(self):
        """Test portfolio allocation calculation."""
        # Normal conditions
        allocation = get_portfolio_allocation(self.prices1, equity=100)
        self.assertIn(RISK_ON, allocation)
        
        # Ensure correct format
        self.assertIsInstance(allocation, dict)
        self.assertTrue(all(isinstance(v, float) for v in allocation.values()))
        
        # Ensure allocations sum to 100
        self.assertAlmostEqual(sum(allocation.values()), 100, delta=0.1)
        
        # Test crash conditions
        crash_allocation = get_portfolio_allocation(self.crash_prices, equity=100)
        self.assertIn(HEDGE_ETF, crash_allocation)
        self.assertAlmostEqual(crash_allocation[HEDGE_ETF], 15.0, delta=0.1)  # 15% hedge


if __name__ == '__main__':
    unittest.main()
