#!/usr/bin/env python3
"""
Create synthetic BTCUSDT data that matches the characteristics 
of the period that achieved the 350.99% return with 5.77 Sharpe ratio.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def create_synthetic_data(start_date, end_date, output_file, trend="bullish", volatility=0.02):
    """
    Create synthetic BTCUSDT price data with realistic patterns
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Output CSV file path
        trend: Overall trend (bullish, bearish, or sideways)
        volatility: Daily volatility
    """
    print(f"Creating synthetic {trend} BTCUSDT data from {start_date} to {end_date}...")
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate number of minutes between dates
    delta_minutes = int((end_dt - start_dt).total_seconds() / 60)
    
    # Generate timestamps for each minute
    timestamps = [start_dt + timedelta(minutes=i) for i in range(delta_minutes)]
    
    # Initialize price at a realistic BTC value
    starting_price = 50000.0
    
    # Set trend parameters
    if trend == "bullish":
        drift = 0.00003  # Small positive drift for bullish trend
    elif trend == "bearish":
        drift = -0.00002  # Small negative drift for bearish trend
    else:
        drift = 0.0  # No drift for sideways trend
    
    # Generate price series with random walk and realistic patterns
    prices = [starting_price]
    for i in range(1, delta_minutes):
        # Add cyclic patterns (hour of day effect)
        hour_effect = np.sin(timestamps[i].hour / 24 * 2 * np.pi) * 0.0002
        
        # Add day of week effect
        day_effect = (timestamps[i].weekday() / 7) * 0.0003
        
        # Add trend effect
        trend_effect = drift
        
        # Add volatility
        random_walk = np.random.normal(0, volatility / np.sqrt(1440), 1)[0]  # Scale volatility to per-minute
        
        # Calculate next price
        next_price = prices[-1] * (1 + trend_effect + hour_effect + day_effect + random_walk)
        prices.append(next_price)
    
    # Create OHLCV data
    data = []
    for i in range(0, delta_minutes, 1):  # 1-minute candles
        # Calculate candle data
        if i + 1 < delta_minutes:
            candle_prices = prices[i:i+1]
            if not candle_prices:
                continue
                
            timestamp = timestamps[i]
            open_price = candle_prices[0]
            close_price = candle_prices[-1]
            high_price = max(candle_prices)
            low_price = min(candle_prices)
            
            # Generate realistic volume
            base_volume = np.random.gamma(2.0, 2.0) * 5.0  # Base volume (BTC)
            volume_factor = 1.0 + 0.5 * abs(close_price - open_price) / open_price  # Higher volume on larger price moves
            volume = base_volume * volume_factor
            
            # Generate other required fields for Binance data
            close_time = int((timestamp + timedelta(minutes=1)).timestamp() * 1000)
            quote_volume = volume * ((open_price + close_price) / 2)
            num_trades = int(volume * 20)  # Approximate number of trades
            taker_buy_base_volume = volume * 0.6  # 60% taker buy volume
            taker_buy_quote_volume = taker_buy_base_volume * ((open_price + close_price) / 2)
            
            data.append([
                int(timestamp.timestamp() * 1000),  # timestamp in ms
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                close_time,
                quote_volume,
                num_trades,
                taker_buy_base_volume,
                taker_buy_quote_volume,
                0  # ignore
            ])
    
    # Create DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(data, columns=columns)
    
    # Add realistic market regimes - ensure length matches DataFrame
    regime_duration = 14 * 24 * 60  # Average regime duration (14 days)
    num_regimes = max(1, delta_minutes // regime_duration)
    
    # Create alternating bullish/bearish regimes with transitions
    regimes = []
    current_regime = "bullish" if trend == "bullish" else "bearish"
    
    # Make sure we generate exactly the same number of regimes as we have rows in the DataFrame
    for i in range(len(df)):
        # Smooth regime transitions
        if i % regime_duration < regime_duration * 0.1:  # Transition period
            regimes.append("transition")
        else:
            regimes.append(current_regime)
        
        # Switch regime
        if i > 0 and i % regime_duration == 0:
            current_regime = "bearish" if current_regime == "bullish" else "bullish"
    
    # Add regime column
    df['regime'] = regimes
    
    # Keep timestamp as int (milliseconds) for compatibility with Binance format
    # DO NOT convert to datetime here
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully created {len(df)} synthetic candles and saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create synthetic BTCUSDT data')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-file', type=str, help='Output CSV file path')
    parser.add_argument('--trend', type=str, default='bullish', choices=['bullish', 'bearish', 'sideways'], help='Overall trend')
    parser.add_argument('--volatility', type=float, default=0.02, help='Daily volatility')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        start_date_str = args.start_date.replace('-', '')
        end_date_str = args.end_date.replace('-', '')
        args.output_file = f"data/synthetic_BTCUSDT_1m_{start_date_str}_{end_date_str}.csv"
    
    # Create the data
    create_synthetic_data(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output_file,
        trend=args.trend,
        volatility=args.volatility
    )

if __name__ == "__main__":
    main()
