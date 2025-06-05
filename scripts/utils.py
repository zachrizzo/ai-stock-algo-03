"""
Utility functions for the DMT trading system
"""
import os
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def configure_logging(name, level=logging.INFO):
    """Configure logging for the given name"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_data(filepath, start_date=None, end_date=None):
    """
    Load data from CSV file with options to filter by date range
    
    Args:
        filepath: Path to CSV file
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        
    Returns:
        DataFrame: Loaded and filtered data
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
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
        
    # Filter by date range if provided
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df.loc[start_dt:end_dt]
    
    return df

def save_results(results, filepath):
    """
    Save backtest results to file
    
    Args:
        results: Dictionary of results to save
        filepath: Path to save results to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Filter out non-serializable data
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_results[key] = value
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            # Skip DataFrames for JSON serialization
            pass
        else:
            # Try to convert to string
            try:
                serializable_results[key] = str(value)
            except:
                pass
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return filepath

def create_results_dir(base_dir="./results"):
    """Create a timestamped results directory"""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"backtest_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def format_as_percentage(value):
    """Format a value as a percentage string"""
    return f"{value * 100:.2f}%"
