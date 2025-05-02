#!/usr/bin/env python3
"""
Test the Binance fee structure implementation
===========================================
This script simulates trades with different fee structures to demonstrate
how trading costs affect performance.
"""

import sys
import os
import pandas as pd
from decimal import Decimal

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class FeeTester:
    """Simulates trading fees across different tiers and scenarios"""
    
    def __init__(self):
        # Initialize fee structure
        self.fee_structure = self._initialize_fee_structure()
        
    def _initialize_fee_structure(self):
        """Initialize the fee structure based on Binance's tier system"""
        return {
            # Regular and VIP tiers - format: {'tier': {'maker_fee': X, 'taker_fee': Y, 'bnb_maker_fee': Z, 'bnb_taker_fee': W}}
            'regular': {'maker_fee': 0.001000, 'taker_fee': 0.001000, 'bnb_maker_fee': 0.000750, 'bnb_taker_fee': 0.000750},
            'vip1': {'maker_fee': 0.000900, 'taker_fee': 0.001000, 'bnb_maker_fee': 0.000675, 'bnb_taker_fee': 0.000750},
            'vip2': {'maker_fee': 0.000800, 'taker_fee': 0.001000, 'bnb_maker_fee': 0.000600, 'bnb_taker_fee': 0.000750},
            'vip3': {'maker_fee': 0.000700, 'taker_fee': 0.000800, 'bnb_maker_fee': 0.000450, 'bnb_taker_fee': 0.000550},
            'vip4': {'maker_fee': 0.000500, 'taker_fee': 0.000600, 'bnb_maker_fee': 0.000400, 'bnb_taker_fee': 0.000500},
            'vip5': {'maker_fee': 0.000300, 'taker_fee': 0.000400, 'bnb_maker_fee': 0.000260, 'bnb_taker_fee': 0.000360},
            'vip6': {'maker_fee': 0.000200, 'taker_fee': 0.000300, 'bnb_maker_fee': 0.000180, 'bnb_taker_fee': 0.000280},
            'vip7': {'maker_fee': 0.000150, 'taker_fee': 0.000250, 'bnb_maker_fee': 0.000125, 'bnb_taker_fee': 0.000220},
            'vip8': {'maker_fee': 0.000100, 'taker_fee': 0.000200, 'bnb_maker_fee': 0.000080, 'bnb_taker_fee': 0.000180},
            'vip9': {'maker_fee': 0.000075, 'taker_fee': 0.000150, 'bnb_maker_fee': 0.000060, 'bnb_taker_fee': 0.000120}
        }
    
    def calculate_fee(self, trade_value, tier='regular', is_maker=False, use_bnb=True):
        """Calculate fee for a given trade value and conditions"""
        # Get the base fee rate
        if tier not in self.fee_structure:
            tier = 'regular'  # Default to regular tier if invalid
            
        if use_bnb:
            fee_rate = self.fee_structure[tier]['bnb_maker_fee'] if is_maker else self.fee_structure[tier]['bnb_taker_fee']
        else:
            fee_rate = self.fee_structure[tier]['maker_fee'] if is_maker else self.fee_structure[tier]['taker_fee']
        
        # Calculate fee
        fee_amount = trade_value * fee_rate
        
        return fee_amount
    
    def simulate_trading_scenario(self, initial_capital=10000, trades_per_month=10, 
                             avg_trade_size_pct=0.2, months=12, monthly_return=0.05,
                             tier='regular', is_maker=False, use_bnb=True):
        """
        Simulate a trading scenario over a period
        
        Args:
            initial_capital: Starting capital
            trades_per_month: Number of trades per month
            avg_trade_size_pct: Average trade size as percentage of capital
            months: Number of months to simulate
            monthly_return: Monthly return before fees
            tier: Trading tier
            is_maker: Whether trades are maker orders
            use_bnb: Whether to use BNB for fee discount
            
        Returns:
            Final capital, total fees, and return metrics
        """
        capital = initial_capital
        total_fees = 0
        total_fees_no_discount = 0
        total_volume = 0
        
        for month in range(months):
            # Calculate this month's pre-fee return
            monthly_gain = capital * monthly_return
            capital += monthly_gain
            
            # Calculate trading volume and fees
            for _ in range(trades_per_month):
                trade_size = capital * avg_trade_size_pct
                total_volume += trade_size
                
                # Calculate fee with specified settings
                fee = self.calculate_fee(trade_size, tier, is_maker, use_bnb)
                total_fees += fee
                
                # Also calculate what fees would be at regular tier with no discount (for comparison)
                regular_fee = self.calculate_fee(trade_size, 'regular', is_maker, False)
                total_fees_no_discount += regular_fee
                
                # Deduct fee from capital
                capital -= fee
        
        # Calculate metrics
        net_return = (capital / initial_capital - 1) * 100
        fees_as_pct_of_capital = (total_fees / initial_capital) * 100
        fees_as_pct_of_volume = (total_fees / total_volume) * 100
        fee_savings = total_fees_no_discount - total_fees
        fee_savings_pct = (fee_savings / total_fees_no_discount) * 100
        
        return {
            'final_capital': capital,
            'total_fees': total_fees,
            'total_volume': total_volume,
            'net_return_pct': net_return,
            'fees_as_pct_of_capital': fees_as_pct_of_capital,
            'fees_as_pct_of_volume': fees_as_pct_of_volume,
            'fee_savings': fee_savings,
            'fee_savings_pct': fee_savings_pct
        }
    
    def compare_scenarios(self):
        """Compare different fee scenarios and print results"""
        scenarios = [
            {
                'name': 'Regular tier, No BNB discount',
                'params': {'tier': 'regular', 'use_bnb': False, 'is_maker': False}
            },
            {
                'name': 'Regular tier, With BNB discount',
                'params': {'tier': 'regular', 'use_bnb': True, 'is_maker': False}
            },
            {
                'name': 'VIP3 tier, No BNB discount',
                'params': {'tier': 'vip3', 'use_bnb': False, 'is_maker': False}
            },
            {
                'name': 'VIP3 tier, With BNB discount',
                'params': {'tier': 'vip3', 'use_bnb': True, 'is_maker': False}
            },
            {
                'name': 'VIP3 tier, Maker orders, With BNB',
                'params': {'tier': 'vip3', 'use_bnb': True, 'is_maker': True}
            },
            {
                'name': 'VIP9 tier, Maker orders, With BNB',
                'params': {'tier': 'vip9', 'use_bnb': True, 'is_maker': True}
            }
        ]
        
        # Use parameters based on the DMT_v2 enhanced strategy
        # High target vol (0.35), max_position 2.0, etc.
        base_params = {
            'initial_capital': 10000,
            'trades_per_month': 20,  # Fairly active trading
            'avg_trade_size_pct': 0.5,  # Larger position sizes
            'months': 12,
            'monthly_return': 0.12  # Aggressive target return
        }
        
        results = []
        for scenario in scenarios:
            params = base_params.copy()
            params.update(scenario['params'])
            result = self.simulate_trading_scenario(**params)
            result['scenario'] = scenario['name']
            results.append(result)
        
        # Convert to DataFrame for nice display
        df = pd.DataFrame(results)
        df = df[['scenario', 'final_capital', 'net_return_pct', 'total_fees', 
                'fees_as_pct_of_capital', 'fee_savings_pct']]
        
        # Format columns
        df['final_capital'] = df['final_capital'].map('${:,.2f}'.format)
        df['net_return_pct'] = df['net_return_pct'].map('{:,.2f}%'.format)
        df['total_fees'] = df['total_fees'].map('${:,.2f}'.format)
        df['fees_as_pct_of_capital'] = df['fees_as_pct_of_capital'].map('{:,.2f}%'.format)
        df['fee_savings_pct'] = df['fee_savings_pct'].map('{:,.2f}%'.format)
        
        print("\n=== IMPACT OF BINANCE FEE STRUCTURE ON TURBODMT_V2 PERFORMANCE ===\n")
        print(df.to_string(index=False))
        print("\nAssumptions:")
        print(f"- Initial capital: ${base_params['initial_capital']:,}")
        print(f"- Trading frequency: {base_params['trades_per_month']} trades per month")
        print(f"- Avg position size: {base_params['avg_trade_size_pct']*100}% of capital")
        print(f"- Time period: {base_params['months']} months")
        print(f"- Monthly return before fees: {base_params['monthly_return']*100}%")
        print("\nTurboDMT_v2 parameters:")
        print("- target_annual_vol: 0.35")
        print("- max_position_size: 2.0")
        print("- neutral_zone: 0.03")
        print("- Model: 96 dimensions, 6 attention heads, 5 layers")


if __name__ == '__main__':
    tester = FeeTester()
    tester.compare_scenarios()
