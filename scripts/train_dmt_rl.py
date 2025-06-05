#!/usr/bin/env python
"""
Training script for the DMT-RL Model.
Performs offline training using historical data and creates a model save file.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dmt_model import DMTRLModel, create_dmt_rl_model
from scripts.utils import configure_logging, load_data

# Configure logging
logger = configure_logging("train_dmt_rl")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DMT-RL hybrid model")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1m",
                        help="Trading interval (e.g. 1m, 5m, 15m, 1h, etc.)")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to historical data file")
    parser.add_argument("--offline-steps", type=int, default=10000,
                        help="Number of offline training steps")
    parser.add_argument("--online-episodes", type=int, default=100,
                        help="Number of online training episodes")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directory to save trained models")
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Train/test split ratio for data")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden size for transformer model")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--target-return", type=float, default=0.01,
                        help="Target return for RL agent")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--allow-short", action="store_true", default=True,
                        help="Allow short selling")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    return parser.parse_args()

def load_market_data(data_file):
    """Load market data from file"""
    logger.info(f"Loading data from {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        
        # Handle different timestamp/datetime column formats
        time_cols = ['timestamp', 'datetime', 'time', 'date']
        time_col = next((col for col in time_cols if col in df.columns), None)
        
        if time_col:
            if df[time_col].dtype == np.int64 or df[time_col].dtype == np.float64:
                # Unix timestamp in milliseconds
                df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
            else:
                # String timestamp
                df['datetime'] = pd.to_datetime(df[time_col])
                
            df.set_index('datetime', inplace=True)
        
        logger.info(f"Loaded {len(df)} data points")
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model(model, data, offline_steps, online_episodes, batch_size, train_split):
    """Train the DMT-RL model"""
    # Split data into train/test
    split_idx = int(len(data) * train_split)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points")
    
    # Train model
    logger.info("Starting offline training...")
    model.train_offline(train_data, steps=offline_steps, batch_size=batch_size)
    
    logger.info("Starting online training...")
    model.train_online(train_data, episodes=online_episodes)
    
    # Evaluate model
    logger.info("Evaluating model on test data...")
    returns, sharpe, max_dd = model.evaluate_performance(test_data)
    
    logger.info(f"Test performance - Returns: {returns:.2f}%, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%")
    
    return {
        "returns": returns,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "train_size": len(train_data),
        "test_size": len(test_data)
    }

def save_model_and_results(model, results, model_dir, symbol, interval):
    """Save trained model and results"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{symbol}_{interval}_dmt_rl_{timestamp}.pth"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)
    
    # Save results
    results_filename = f"{symbol}_{interval}_dmt_rl_results_{timestamp}.json"
    results_path = os.path.join(model_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_path}")
    
    return model_path, results_path

def main():
    """Main function"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger("train_dmt_rl").setLevel(getattr(logging, args.log_level))
    
    # Create model
    logger.info(f"Creating DMT-RL model for {args.symbol} {args.interval}")
    model = create_dmt_rl_model(
        symbol=args.symbol,
        interval=args.interval,
        allow_short=args.allow_short,
        max_position=1.0,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        target_return=args.target_return,
        learning_rate=args.learning_rate
    )
    
    # Load data
    data = load_market_data(args.data_file)
    
    # Train model
    results = train_model(
        model=model,
        data=data,
        offline_steps=args.offline_steps,
        online_episodes=args.online_episodes,
        batch_size=args.batch_size,
        train_split=args.train_split
    )
    
    # Save model and results
    model_path, results_path = save_model_and_results(
        model=model,
        results=results,
        model_dir=args.model_dir,
        symbol=args.symbol,
        interval=args.interval
    )
    
    logger.info(f"Training complete. Model saved to {model_path}")
    logger.info(f"Final performance - Returns: {results['returns']:.2f}%, Sharpe: {results['sharpe']:.2f}, Max DD: {results['max_drawdown']:.2f}%")
    
    return model, results

if __name__ == "__main__":
    main()
