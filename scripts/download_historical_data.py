#!/usr/bin/env python3
"""
Download historical BTCUSDT data from Binance for the period 
that achieved the 350.99% return with 5.77 Sharpe ratio.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import time
import argparse

# Load environment variables
load_dotenv()

def download_historical_data(symbol, interval, start_date, end_date, output_file):
    """
    Download historical data from Binance and save to CSV file
    """
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}...")
    
    # Initialize Binance client
    api_key = os.environ.get('BINANCE_API_KEY_TEST')
    api_secret = os.environ.get('BINANCE_API_SECRET_TEST')
    
    if not api_key or not api_secret:
        raise ValueError("Binance API key and secret are required")
    
    client = Client(api_key, api_secret)
    
    # Convert dates to millisecond timestamps
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Download data in chunks to avoid timeouts
    current_start = start_timestamp
    chunk_size = 1000  # Number of candles per chunk
    all_candles = []
    
    while current_start < end_timestamp:
        try:
            # Calculate chunk end time
            if interval.endswith('m'):
                minutes = int(interval[:-1])
                chunk_end = current_start + (chunk_size * minutes * 60 * 1000)
            elif interval.endswith('h'):
                hours = int(interval[:-1])
                chunk_end = current_start + (chunk_size * hours * 60 * 60 * 1000)
            elif interval.endswith('d'):
                days = int(interval[:-1])
                chunk_end = current_start + (chunk_size * days * 24 * 60 * 60 * 1000)
            else:
                chunk_end = current_start + (chunk_size * 60 * 1000)  # Default to 1m
            
            # Ensure we don't exceed the end date
            chunk_end = min(chunk_end, end_timestamp)
            
            # Get klines for this chunk
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=current_start,
                end_str=chunk_end
            )
            
            # Add to the list of candles
            all_candles.extend(klines)
            
            # Print progress
            start_date_str = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m-%d %H:%M:%S')
            end_date_str = datetime.fromtimestamp(chunk_end / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Downloaded {len(klines)} candles for {start_date_str} to {end_date_str}")
            
            # Update start timestamp for next chunk
            if klines:
                current_start = klines[-1][0] + 1  # Start from the next candle after the last one
            else:
                # If no data for this chunk, move forward by the chunk size
                current_start = chunk_end + 1
            
            # Sleep to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_candles, columns=columns)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully downloaded {len(df)} candles and saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Download historical data from Binance')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--interval', type=str, default='1m', help='Candlestick interval')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-file', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        start_date_str = args.start_date.replace('-', '')
        end_date_str = args.end_date.replace('-', '')
        args.output_file = f"data/{args.symbol}_{args.interval}_{start_date_str}_{end_date_str}.csv"
    
    # Download the data
    download_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
