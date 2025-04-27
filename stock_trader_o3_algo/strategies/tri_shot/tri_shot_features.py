import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Union
import yfinance as yf
import datetime as dt
from datetime import timedelta
import warnings
from stock_trader_o3_algo.config.settings import TZ # Import TZ from settings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def make_feature_matrix(prices: pd.DataFrame, 
                        target_ticker: str = "QQQ", 
                        lookback_days: int = 300,
                        handle_nans: str = 'drop') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix and target labels for the ML model.
    
    Args:
        prices: DataFrame with price data for all tickers
        target_ticker: Ticker to predict (usually QQQ)
        lookback_days: Number of days to use for feature creation
        handle_nans: How to handle NaN values: 'drop' to remove them (default),
                    'fill_zeros' to replace with zeros, or 'fill_means' to use feature means
        
    Returns:
        X: Feature matrix
        y: Target labels (1 for up, 0 for down)
    """
    df = prices.copy()
    
    # Target creation: 5-day forward returns
    df[f'{target_ticker}_fwd_5d_ret'] = df[target_ticker].shift(-5) / df[target_ticker] - 1
    
    # Create binary labels (0 for down, 1 for up)
    df['target'] = np.where(df[f'{target_ticker}_fwd_5d_ret'] >= 0, 1, 0)
    
    # ------ ENHANCE FEATURE SET ------
    
    # 1. Price momentum - standard lookbacks
    for days in [5, 10, 21, 63]:
        df[f'{target_ticker}_mom_{days}d'] = df[target_ticker].pct_change(days)
    
    # 2. Kelly slope - slope of log-price vs sqrt(t)
    # This captures convexity in price movement
    log_price = np.log(df[target_ticker])
    for days in [10, 21]:
        kelly_y = log_price.rolling(window=days).apply(lambda x: np.polyfit(np.sqrt(np.arange(1, len(x)+1)), x, 1)[0], raw=False)
        df[f'{target_ticker}_kelly_{days}d'] = kelly_y
    
    # 3. Efficiency Ratio - measures trend quality
    # High ER means price is moving efficiently, low ER means choppy/noisy
    for days in [10, 21]:
        # Numerator: net directional move
        net_move = (df[target_ticker] - df[target_ticker].shift(days)).abs()
        # Denominator: sum of all price movements (path length)
        path_length = df[target_ticker].diff().abs().rolling(days).sum()
        # ER = directional move / path length
        df[f'{target_ticker}_eff_ratio_{days}d'] = net_move / path_length
    
    # 4. Volatility measures
    df[f'{target_ticker}_vol_5d'] = df[target_ticker].pct_change().rolling(5).std() * np.sqrt(252)
    df[f'{target_ticker}_vol_20d'] = df[target_ticker].pct_change().rolling(20).std() * np.sqrt(252)
    
    # 5. Higher-order moments: skew and kurtosis
    for days in [10, 20]:
        df[f'{target_ticker}_skew_{days}d'] = df[target_ticker].pct_change().rolling(days).skew()
        df[f'{target_ticker}_kurt_{days}d'] = df[target_ticker].pct_change().rolling(days).kurt()
    
    # 6. Correlation-cluster break indicator
    # Positive correlation between QQQ and VIX often signals regime change
    if '^VIX' in df.columns:
        for days in [10, 20]:
            rolling_corr = df[target_ticker].pct_change().rolling(days).corr(df['^VIX'].pct_change())
            df[f'{target_ticker}_vix_corr_{days}d'] = rolling_corr
            # Signal = 1 when correlation turns positive (risk-off)
            df[f'{target_ticker}_vix_corr_pos_{days}d'] = np.where(rolling_corr > 0, 1, 0)
    
    # 7. ATR - measure of volatility that accounts for gaps
    def calc_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    # If we have high/low data
    if isinstance(df, pd.DataFrame) and all(col in df.columns for col in [f'{target_ticker}_High', f'{target_ticker}_Low']):
        df[f'{target_ticker}_atr_14d'] = calc_atr(df[f'{target_ticker}_High'], df[f'{target_ticker}_Low'], df[target_ticker], 14)
    else:
        # Simplified ATR using only close prices
        df[f'{target_ticker}_atr_14d'] = df[target_ticker].pct_change().abs().rolling(14).mean() * df[target_ticker]
    
    # 8. Money flow - volume-weighted price change
    if f'{target_ticker}_Volume' in df.columns:
        price_change = df[target_ticker].pct_change()
        volume = df[f'{target_ticker}_Volume']
        df[f'{target_ticker}_money_flow_1d'] = price_change * volume
        df[f'{target_ticker}_money_flow_5d'] = df[f'{target_ticker}_money_flow_1d'].rolling(5).sum()
    
    # 9. EMAs and regime filters
    for days in [8, 21, 50, 200]:
        df[f'{target_ticker}_ema_{days}'] = df[target_ticker].ewm(span=days).mean()
        df[f'{target_ticker}_vs_ema_{days}'] = df[target_ticker] / df[f'{target_ticker}_ema_{days}'] - 1
    
    # 10. EMA crossovers
    df[f'{target_ticker}_ema_8_21_cross'] = np.where(
        df[f'{target_ticker}_ema_8'] > df[f'{target_ticker}_ema_21'], 1, -1
    )
    df[f'{target_ticker}_ema_50_200_cross'] = np.where(
        df[f'{target_ticker}_ema_50'] > df[f'{target_ticker}_ema_200'], 1, -1
    )
    
    # 11. MACD
    df[f'{target_ticker}_ema_12'] = df[target_ticker].ewm(span=12).mean()
    df[f'{target_ticker}_ema_26'] = df[target_ticker].ewm(span=26).mean()
    df[f'{target_ticker}_macd'] = df[f'{target_ticker}_ema_12'] - df[f'{target_ticker}_ema_26']
    df[f'{target_ticker}_macd_signal'] = df[f'{target_ticker}_macd'].ewm(span=9).mean()
    df[f'{target_ticker}_macd_hist'] = df[f'{target_ticker}_macd'] - df[f'{target_ticker}_macd_signal']
    df[f'{target_ticker}_macd_hist_chg'] = df[f'{target_ticker}_macd_hist'].pct_change(3)
    
    # 12. RSI
    delta = df[target_ticker].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df[f'{target_ticker}_rsi'] = 100 - (100 / (1 + rs))
    
    # RSI divergence
    df[f'{target_ticker}_rsi_div'] = (df[target_ticker].pct_change(5).rolling(5).mean() - 
                                     df[f'{target_ticker}_rsi'].pct_change(5).rolling(5).mean())
    
    # 13. VIX-based features
    if '^VIX' in df.columns:
        df['vix'] = df['^VIX']
        df['vix_5d_chg'] = df['vix'].pct_change(5)
        df['vix_20d_chg'] = df['vix'].pct_change(20)
        df['vix_ma_ratio'] = df['vix'] / df['vix'].rolling(20).mean()
        
        # VIX term slope proxy
        if 'VIXMO' in df.columns:  # If we have VIX Mid-Term futures data
            df['vix_term_slope'] = df['^VIX'] / df['VIXMO'] - 1
        
        # VIX percentile rank (0-100) over last 252 days
        df['vix_rank_252d'] = df['vix'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
    
    # 14. TLT and bond market features
    if 'TLT' in df.columns:
        df['tlt_mom_20d'] = df['TLT'].pct_change(20)
        df['tlt_vol_20d'] = df['TLT'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # TLT vs QQQ relative strength
        df['tlt_qqq_rs_20d'] = df['TLT'].pct_change(20) - df[target_ticker].pct_change(20)
    else:
        # Create placeholder with zeros if TLT is not available
        df['tlt_mom_20d'] = 0.0
    
    # 15. Economic data (basic proxies)
    # USD strength
    if 'UUP' in df.columns:  # Dollar ETF
        df['usd_chg_20d'] = df['UUP'].pct_change(20)
    
    # 16. Temporal features
    # Day of week
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        # One-hot encode day of week
        for i in range(5):  # 0=Monday, 4=Friday
            df[f'day_{i}'] = np.where(df['day_of_week'] == i, 1, 0)
        
        # Month of year
        df['month'] = df.index.month
        # Month seasonality effect (simplified)
        for i in range(1, 13):
            df[f'month_{i}'] = np.where(df['month'] == i, 1, 0)
    
    # 17. Technical "overbought/oversold" signals
    df[f'{target_ticker}_overbought'] = np.where(df[f'{target_ticker}_rsi'] > 70, 1, 0)
    df[f'{target_ticker}_oversold'] = np.where(df[f'{target_ticker}_rsi'] < 30, 1, 0)
    
    # 18. Volatility regime
    df['vol_regime'] = np.where(df[f'{target_ticker}_vol_20d'] > df[f'{target_ticker}_vol_20d'].rolling(63).mean(), 1, 0)
    
    # Drop rows with NaN targets (typically the last 5 rows)
    df = df.dropna(subset=['target'])
    
    # Get all features for modeling
    feature_cols = [col for col in df.columns if col != 'target' and col != f'{target_ticker}_fwd_5d_ret'
                   and not any(c in col for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'date'])]
    
    # Handle NaN values in features based on the selected strategy
    if handle_nans == 'drop':
        # Traditional approach: drop rows with any NaN features
        df_clean = df.dropna(subset=feature_cols)
        print(f"Warning: Dropped {len(df) - len(df_clean)} rows due to NaN values in features")
        df = df_clean
    elif handle_nans == 'fill_zeros':
        # Fill NaNs with zeros - simple but can introduce bias
        df[feature_cols] = df[feature_cols].fillna(0)
        print(f"Filled NaN values with zeros in {len(feature_cols)} features")
    elif handle_nans == 'fill_means':
        # Fill NaNs with feature means - better statistical approach
        for col in feature_cols:
            if df[col].isna().any():
                # Calculate mean of non-NaN values
                col_mean = df[col].mean(skipna=True)
                # Replace NaNs with mean
                df[col] = df[col].fillna(col_mean)
        print(f"Filled NaN values with feature means in {len(feature_cols)} features")
    else:
        # Default to dropping NaNs
        df_clean = df.dropna(subset=feature_cols)
        print(f"Warning: Dropped {len(df) - len(df_clean)} rows due to NaN values in features")
        df = df_clean
    
    # Return features (X) and target (y)
    X = df[feature_cols]
    y = df['target']
    
    return X, y

def latest_features(prices: pd.DataFrame, target_ticker: str = "QQQ") -> pd.DataFrame:
    """
    Create feature vector for the most recent data point.
    
    Args:
        prices: DataFrame with price data for all tickers
        target_ticker: Ticker to predict (usually QQQ)
        
    Returns:
        X: Single row DataFrame with features for latest data point
    """
    X, _ = make_feature_matrix(prices, target_ticker)
    return X.iloc[[-1]]

def fetch_data(tickers: List[str], days: int = 300) -> pd.DataFrame:
    """
    Fetch historical price data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with price data
    """
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"].ffill()
    return df

def fetch_data_from_date(tickers, start_date, end_date=None):
    """
    Fetch historical price data for a list of tickers from a specific start date.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date as datetime object
        end_date: Optional end date as datetime object (defaults to today)
        
    Returns:
        DataFrame with price data
    """
    if end_date is None:
        end_date = dt.datetime.now()
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Add a buffer for feature calculation
    buffer_start = start_date - dt.timedelta(days=30)
    
    # For long-term backtests, we need to be aware that some ETFs like TQQQ didn't exist before certain dates
    # TQQQ and SQQQ inception: Feb 2010
    # TMF inception: April 2009
    
    # Use yfinance's batch download for efficiency
    try:
        # Download all data at once
        data = yf.download(tickers, start=buffer_start, end=end_date, auto_adjust=True, progress=False)
        
        if len(data) == 0:
            print("No data available for the specified date range")
            return pd.DataFrame()
        
        # Extract the Close prices
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close']
        else:
            # If only one ticker, the result won't have MultiIndex
            df = pd.DataFrame(data['Close'])
            df.columns = [tickers[0]]
        
        # Forward fill missing values
        df = df.ffill()
        
        # Make index timezone-aware using the project's timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize(TZ)
        else:
            df.index = df.index.tz_convert(TZ)
            
        print(f"Downloaded data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def train_model(prices: pd.DataFrame, target_ticker: str = "QQQ") -> xgb.XGBClassifier:
    """
    Train an XGBoost model on historical data.
    
    Args:
        prices: DataFrame with price data for all tickers
        target_ticker: Ticker to predict (usually QQQ)
        
    Returns:
        Trained XGBoost classifier
    """
    X, y = make_feature_matrix(prices, target_ticker)
    
    # Split the data into training and validation sets
    # Use the most recent 20% of data for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Initialize and train the model with simplified parameters
    clf = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=100,  # Reduced for testing
        learning_rate=0.1
    )
    
    # Simple fit without eval set or callbacks
    clf.fit(X_train, y_train)
    
    return clf

def save_model(model, filename: str = "tri_shot_model.pkl") -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained XGBoost classifier
        filename: Output filename
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename: str = "tri_shot_model.pkl") -> xgb.XGBClassifier:
    """
    Load a trained model from disk.
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded XGBoost classifier
    """
    return joblib.load(filename)
