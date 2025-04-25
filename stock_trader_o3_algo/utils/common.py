"""
Common utility functions used across different strategies.
"""
import os
import json
import datetime as dt
import pytz
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default timezone
DEFAULT_TZ = pytz.timezone("America/New_York")

def get_alpaca_api() -> tradeapi.REST:
    """Initialize Alpaca API client."""
    api_key = os.getenv("ALPACA_KEY")
    api_secret = os.getenv("ALPACA_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")
    
    return tradeapi.REST(api_key, api_secret, base_url)

def ensure_directory(dir_path: Path) -> Path:
    """Create a directory if it doesn't exist."""
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        print(f"Created directory: {dir_path}")
    return dir_path

def load_json(file_path: Path) -> dict:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return {}

def save_json(data: dict, file_path: Path) -> None:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")

def calculate_drawdown(equity_curve: pd.Series) -> Tuple[float, dt.datetime]:
    """Calculate maximum drawdown and its date from an equity curve."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    return max_drawdown, max_drawdown_date

def parse_date(date_str: str) -> dt.datetime:
    """Parse a date string in YYYY-MM-DD format."""
    return dt.datetime.strptime(date_str, '%Y-%m-%d')
