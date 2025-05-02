#!/usr/bin/env python3
"""
TurboDMT_v2 Feature Engineering
===============================
Advanced feature engineering for the TurboDMT_v2 trading strategy with 
spectral analysis, multi-timeframe features, and market regime indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy import signal, stats, fft
from sklearn.preprocessing import StandardScaler
import talib


class AdvancedFeatureGenerator:
    """Feature generator for TurboDMT_v2 with enhanced technical indicators"""
    
    def __init__(
        self,
        use_spectral: bool = True,
        use_multi_timeframe: bool = True,
        use_market_regime: bool = True,
        use_volatility_surface: bool = True,
        use_orderflow: bool = True,
        standardize: bool = True,
    ):
        self.use_spectral = use_spectral
        self.use_multi_timeframe = use_multi_timeframe
        self.use_market_regime = use_market_regime
        self.use_volatility_surface = use_volatility_surface
        self.use_orderflow = use_orderflow
        self.standardize = standardize
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate enhanced feature set from OHLCV data
        
        Args:
            data: DataFrame with at least OHLCV columns
            
        Returns:
            DataFrame with all generated features
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure all required columns are present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Original features will be the base
        features = pd.DataFrame(index=df.index)
        
        # Generate all feature groups
        self._add_base_features(df, features)
        
        if self.use_multi_timeframe:
            self._add_multi_timeframe_features(df, features)
        
        if self.use_spectral:
            self._add_spectral_features(df, features)
        
        if self.use_market_regime:
            self._add_regime_features(df, features)
        
        if self.use_volatility_surface:
            self._add_volatility_features(df, features)
        
        if self.use_orderflow:
            self._add_orderflow_features(df, features)
            
        # Add target variable (next day's return)
        features['target'] = df['Close'].pct_change(1).shift(-1)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        # Save feature names
        self.feature_names = features.columns.tolist()
        
        # Standardize features if required
        if self.standardize:
            # Exclude the target from standardization
            feature_cols = [col for col in features.columns if col != 'target']
            features[feature_cols] = self.scaler.fit_transform(features[feature_cols])
        
        return features
    
    def _add_base_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add basic price and volume features"""
        # Price transforms
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        features['ma5'] = df['Close'].rolling(5).mean() / df['Close'] - 1
        features['ma10'] = df['Close'].rolling(10).mean() / df['Close'] - 1
        features['ma20'] = df['Close'].rolling(20).mean() / df['Close'] - 1
        features['ma50'] = df['Close'].rolling(50).mean() / df['Close'] - 1
        
        # Volatility estimators
        features['realized_vol'] = df['returns'].rolling(21).std() * np.sqrt(252)
        features['atr'] = self._calculate_atr(df, 14) / df['Close']
        features['natr'] = talib.NATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14) / 100
        
        # Momentum indicators
        for period in [5, 10, 21, 63]:
            features[f'mom_{period}'] = df['Close'].pct_change(period)
            features[f'roc_{period}'] = talib.ROC(df['Close'].values, timeperiod=period) / 100
        
        # Mean reversion indicators
        for period in [5, 10, 21]:
            features[f'mean_rev_{period}'] = (df['Close'] - df['Close'].rolling(period).mean()) / df['Close'].rolling(period).std()
        
        # Price ratios
        features['high_low_ratio'] = df['High'] / df['Low'] - 1
        features['close_open_ratio'] = df['Close'] / df['Open'] - 1
        
        # Gap indicators
        features['overnight_gap'] = (df['Open'] / df['Close'].shift(1)) - 1
        
        # Range measures
        features['daily_range'] = (df['High'] - df['Low']) / df['Close']
        features['weekly_range'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close']
        
        # Volume indicators
        features['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['vwap'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
        features['vwap_ratio'] = df['Close'] / features['vwap'] - 1
        
        # Technical indicators
        features['rsi'] = talib.RSI(df['Close'].values, timeperiod=14) / 100
        features['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14) / 100
        features['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14) / 100
        features['macd'], features['macd_signal'], _ = talib.MACD(
            df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        features['macd'] = features['macd'] / df['Close']
        features['macd_signal'] = features['macd_signal'] / df['Close']
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['Close'].values - lower) / (upper - lower)
        
        # Stochastic oscillator
        features['stoch_k'], features['stoch_d'] = talib.STOCH(
            df['High'].values, df['Low'].values, df['Close'].values,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        features['stoch_k'] = features['stoch_k'] / 100
        features['stoch_d'] = features['stoch_d'] / 100
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add multi-timeframe features for different lookback periods"""
        # Weekly features
        features['weekly_return'] = df['Close'].pct_change(5)
        features['weekly_volatility'] = features['returns'].rolling(5).std() * np.sqrt(5)
        
        # Biweekly features
        features['biweekly_return'] = df['Close'].pct_change(10)
        features['biweekly_volatility'] = features['returns'].rolling(10).std() * np.sqrt(10)
        
        # Monthly features
        features['monthly_return'] = df['Close'].pct_change(21)
        features['monthly_volatility'] = features['returns'].rolling(21).std() * np.sqrt(21)
        
        # Quarterly features
        features['quarterly_return'] = df['Close'].pct_change(63)
        features['quarterly_volatility'] = features['returns'].rolling(63).std() * np.sqrt(63)
        
        # Cross-timeframe ratios
        features['ma_short_long_ratio'] = features['ma5'] / features['ma50']
        features['vol_short_long_ratio'] = features['weekly_volatility'] / features['monthly_volatility']
        features['mom_short_long_ratio'] = features['mom_5'] / (features['mom_21'] + 1e-8)
        
        # Trend strength across timeframes
        features['trend_alignment'] = (
            np.sign(features['ma5']) + 
            np.sign(features['ma10']) + 
            np.sign(features['ma20']) + 
            np.sign(features['ma50'])
        ) / 4.0
        
        # Relative strength across timeframes
        features['relative_strength'] = (
            features['weekly_return'] * 0.4 + 
            features['biweekly_return'] * 0.3 + 
            features['monthly_return'] * 0.2 + 
            features['quarterly_return'] * 0.1
        )
    
    def _add_spectral_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add spectral analysis features using Fourier transforms"""
        # Extract price series
        close_prices = df['Close'].values
        returns = features['returns'].fillna(0).values
        
        # Function to calculate dominant frequencies
        def get_dominant_frequencies(data, sampling_rate=1.0, n_peaks=3):
            # Detrend the data
            detrended = signal.detrend(data)
            
            # Apply Fourier transform
            freq_spectrum = np.abs(fft.rfft(detrended))
            freqs = fft.rfftfreq(len(detrended), d=1/sampling_rate)
            
            # Find dominant frequencies
            peak_indices = signal.find_peaks(freq_spectrum)[0]
            if len(peak_indices) == 0:
                return np.zeros(n_peaks), np.zeros(n_peaks)
            
            # Sort peaks by amplitude
            sorted_peaks = sorted(zip(freq_spectrum[peak_indices], freqs[peak_indices], peak_indices), 
                                reverse=True)
            
            # Extract top frequencies and their power
            top_freqs = []
            top_powers = []
            
            for i in range(min(n_peaks, len(sorted_peaks))):
                power, freq, _ = sorted_peaks[i]
                top_freqs.append(freq)
                top_powers.append(power / len(detrended))  # Normalize power
            
            # Pad with zeros if needed
            top_freqs = top_freqs + [0] * (n_peaks - len(top_freqs))
            top_powers = top_powers + [0] * (n_peaks - len(top_powers))
            
            return np.array(top_freqs), np.array(top_powers)
        
        # Calculate spectral features for different window sizes
        window_sizes = [30, 60, 120]
        for window in window_sizes:
            # Create arrays to store features
            dom_freq1 = np.zeros(len(df))
            dom_freq2 = np.zeros(len(df))
            dom_power1 = np.zeros(len(df))
            dom_power2 = np.zeros(len(df))
            spectral_ratio = np.zeros(len(df))
            spectral_entropy = np.zeros(len(df))
            
            # Calculate rolling spectral features
            for i in range(window, len(df)):
                window_data = returns[i-window:i]
                
                # Get dominant frequencies
                freqs, powers = get_dominant_frequencies(window_data, n_peaks=2)
                
                # Store results
                dom_freq1[i] = freqs[0]
                dom_freq2[i] = freqs[1]
                dom_power1[i] = powers[0]
                dom_power2[i] = powers[1]
                
                # Spectral ratio (ratio of powers)
                if powers[1] > 0:
                    spectral_ratio[i] = powers[0] / powers[1]
                else:
                    spectral_ratio[i] = 0
                
                # Calculate spectral entropy (measure of randomness)
                freq_spectrum = np.abs(fft.rfft(window_data))
                freq_spectrum = freq_spectrum / (np.sum(freq_spectrum) + 1e-10)
                entropy = -np.sum(freq_spectrum * np.log2(freq_spectrum + 1e-10))
                spectral_entropy[i] = entropy / np.log2(len(freq_spectrum))
            
            # Add features to DataFrame
            features[f'dominant_freq1_{window}'] = dom_freq1
            features[f'dominant_freq2_{window}'] = dom_freq2
            features[f'dominant_power1_{window}'] = dom_power1
            features[f'dominant_power2_{window}'] = dom_power2
            features[f'spectral_ratio_{window}'] = spectral_ratio
            features[f'spectral_entropy_{window}'] = spectral_entropy
    
    def _add_regime_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add market regime detection features"""
        # Trend features
        features['bull_trend'] = ((df['Close'] > df['Close'].shift(1)) & 
                                (df['Close'] > df['Close'].rolling(10).mean())).astype(int)
        features['bear_trend'] = ((df['Close'] < df['Close'].shift(1)) & 
                                (df['Close'] < df['Close'].rolling(10).mean())).astype(int)
        
        # Volatility regime
        vol = features['realized_vol']
        high_vol_threshold = vol.rolling(252).mean() + vol.rolling(252).std()
        low_vol_threshold = vol.rolling(252).mean() - vol.rolling(252).std()
        features['high_vol_regime'] = (vol > high_vol_threshold).astype(int)
        features['low_vol_regime'] = (vol < low_vol_threshold).astype(int)
        
        # Momentum regime
        mom = features['mom_21']
        high_mom_threshold = mom.rolling(252).mean() + mom.rolling(252).std()
        low_mom_threshold = mom.rolling(252).mean() - mom.rolling(252).std()
        features['high_mom_regime'] = (mom > high_mom_threshold).astype(int)
        features['low_mom_regime'] = (mom < low_mom_threshold).astype(int)
        
        # Mean reversion regime
        mean_rev = features['mean_rev_21']
        features['mean_rev_regime'] = ((mean_rev > 1) | (mean_rev < -1)).astype(int)
        
        # Trend strength
        features['trend_strength'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14) / 100
        
        # Market efficiency ratio (Fractal efficiency)
        n = 10
        price_change = np.abs(df['Close'] - df['Close'].shift(n))
        path_length = df['Close'].diff().abs().rolling(n).sum()
        features['efficiency_ratio'] = price_change / (path_length + 1e-10)
        
        # Synthetic VIX (based on realized volatility)
        features['synthetic_vix'] = features['realized_vol'] * 100
        
        # Hurst exponent (proxy for mean-reversion vs. trending)
        def hurst_exponent(prices, lags=20):
            tau = []
            lagvec = []
            
            # Step through the different lags
            for lag in range(2, lags):
                # Construct price difference with lag
                pp = np.array(prices[lag:]) - np.array(prices[:-lag])
                
                # Find the variance
                tau.append(np.sqrt(np.std(pp)))
                lagvec.append(lag)
            
            # Linear fit to a log-log plot
            m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
            hurst = m[0]
            return hurst
        
        # Calculate Hurst exponent for rolling windows
        window_size = 100
        hurst_values = np.zeros(len(df))
        for i in range(window_size, len(df)):
            hurst_values[i] = hurst_exponent(df['Close'].values[i-window_size:i])
        
        features['hurst_exponent'] = hurst_values
        
        # Calculate regime probabilities based on multiple indicators
        # Higher values = more bullish, lower values = more bearish
        features['regime_probability'] = (
            (features['bull_trend'] * 0.2) + 
            ((1 - features['high_vol_regime']) * 0.2) + 
            (features['high_mom_regime'] * 0.2) + 
            (features['ma_short_long_ratio'] > 0) * 0.2 + 
            (features['trend_strength'] > 0.25) * 0.2
        )
    
    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add volatility surface and term structure features"""
        # GARCH-like volatility prediction
        returns = features['returns'].fillna(0).values
        
        # Function to estimate GARCH(1,1) volatility
        def garch_volatility(returns, omega=0.000001, alpha=0.1, beta=0.85):
            n = len(returns)
            h = np.zeros(n)
            h[0] = np.var(returns)
            
            for t in range(1, n):
                h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
            
            return np.sqrt(h)
        
        # Calculate GARCH volatility
        garch_vol = garch_volatility(returns)
        features['garch_vol'] = garch_vol
        
        # Volatility of volatility (vol of realized vol)
        features['vol_of_vol'] = features['realized_vol'].rolling(21).std()
        
        # Volatility term structure (ratio of short-term to long-term vol)
        features['vol_term_structure'] = (
            features['returns'].rolling(5).std() / 
            features['returns'].rolling(21).std()
        )
        
        # Volatility skew (proxy for protection demand)
        high_low_ratio = (df['High'] - df['Close']) / (df['Close'] - df['Low'])
        features['vol_skew'] = high_low_ratio.rolling(10).mean()
        
        # Forward-looking volatility estimate (simple forecast)
        ewma_vol = features['returns'].ewm(span=10).std()
        features['forward_vol_est'] = ewma_vol * (1 + 0.1 * features['vol_term_structure'])
        
        # Volatility risk premium (historical vs implied - using realized as proxy)
        features['vol_risk_premium'] = features['realized_vol'] / features['garch_vol']
        
        # Volatility regime shifts (acceleration)
        features['vol_acceleration'] = features['realized_vol'].diff()
    
    def _add_orderflow_features(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        """Add order flow and liquidity features based on price/volume"""
        # Volume Weighted Average Price (VWAP)
        vwap = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
        features['price_to_vwap'] = df['Close'] / vwap - 1
        
        # Volume Imbalance
        up_volume = df['Volume'].where(df['Close'] > df['Open'], 0)
        down_volume = df['Volume'].where(df['Close'] < df['Open'], 0)
        features['volume_imbalance'] = (up_volume - down_volume) / (up_volume + down_volume + 1e-10)
        
        # Buying/Selling Pressure
        features['buy_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        features['sell_pressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-10)
        
        # Order Flow Index (OFI) - simplified version
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        money_flow_multiplier = ((typical_price - typical_price.shift(1)) / 
                                (df['High'] - df['Low'] + 1e-10))
        money_flow = raw_money_flow * money_flow_multiplier
        features['money_flow_index'] = money_flow.rolling(14).sum() / raw_money_flow.rolling(14).sum()
        
        # Chaikin Money Flow
        cmf = talib.ADOSC(df['High'].values, df['Low'].values, df['Close'].values, 
                        df['Volume'].values, fastperiod=3, slowperiod=10)
        features['chaikin_money_flow'] = cmf / df['Volume'].rolling(10).mean()
        
        # Volume Profile
        features['relative_volume'] = df['Volume'] / df['Volume'].rolling(30).mean()
        
        # Volume Zones
        high_volume_zone = df['Volume'].rolling(20).mean() + df['Volume'].rolling(20).std()
        features['high_volume_zone'] = (df['Volume'] > high_volume_zone).astype(int)
        
        # Money Flow Index
        features['money_flow_index'] = talib.MFI(df['High'].values, df['Low'].values, 
                                              df['Close'].values, df['Volume'].values, timeperiod=14) / 100
        
        # On-Balance Volume normalized
        obv = talib.OBV(df['Close'].values, df['Volume'].values)
        features['obv_normalized'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr


class FeaturePyramid:
    """
    Multi-timeframe feature pyramid for analyzing market patterns at different time scales
    and combining them into a unified feature representation.
    """
    
    def __init__(
        self,
        base_timeframe: str = 'D',
        timeframes: List[str] = ['D', 'W', 'M'],
        use_spectral: bool = True,
        standardize: bool = True,
    ):
        self.base_timeframe = base_timeframe
        self.timeframes = timeframes
        self.use_spectral = use_spectral
        self.standardize = standardize
        self.feature_generators = {
            tf: AdvancedFeatureGenerator(
                use_spectral=use_spectral,
                use_multi_timeframe=(tf == base_timeframe),  # Only use multi-timeframe for base timeframe
                use_market_regime=True,
                use_volatility_surface=True,
                use_orderflow=True,
                standardize=standardize,
            )
            for tf in timeframes
        }
        self.features = {}
        self.combined_features = None
    
    def generate_pyramid(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate feature pyramid from dictionary of dataframes at different timeframes
        
        Args:
            data: Dictionary mapping timeframe to corresponding OHLCV dataframe
            
        Returns:
            DataFrame with combined features from all timeframes
        """
        # Check if all required timeframes are present
        for tf in self.timeframes:
            if tf not in data:
                raise ValueError(f"Missing data for timeframe: {tf}")
        
        # Generate features for each timeframe
        for tf in self.timeframes:
            self.features[tf] = self.feature_generators[tf].generate_features(data[tf])
        
        # Get base timeframe features
        base_features = self.features[self.base_timeframe]
        
        # Create a dataframe to hold the combined features
        combined = base_features.copy()
        
        # Add features from other timeframes, resampled to match base timeframe
        for tf in self.timeframes:
            if tf == self.base_timeframe:
                continue
            
            # Resample to base timeframe
            tf_features = self.features[tf]
            
            # Forward fill to match base timeframe index
            resampled = pd.DataFrame(index=base_features.index)
            
            # For each feature in this timeframe, resample to base timeframe
            for col in tf_features.columns:
                if col == 'target':  # Skip target column
                    continue
                
                # Resample and forward fill
                resampled[f"{col}_{tf}"] = tf_features[col].reindex(
                    resampled.index, method='ffill'
                )
            
            # Add resampled features to combined dataframe
            for col in resampled.columns:
                combined[col] = resampled[col]
        
        # Save combined features
        self.combined_features = combined
        
        return combined


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download some test data
    spy = yf.download("SPY", start="2022-01-01", end="2023-01-01")
    
    # Create feature generator
    feature_gen = AdvancedFeatureGenerator()
    
    # Generate features
    features = feature_gen.generate_features(spy)
    
    # Print shape and first few columns
    print(f"Generated {features.shape[1]} features from {spy.shape[0]} data points")
    print(features.head().iloc[:, :5])  # Show first 5 columns
    
    # Test Feature Pyramid
    # Create dummy weekly and monthly data
    spy_weekly = spy.resample('W').agg({
        'Open': 'first', 
        'High': 'max', 
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum'
    })
    
    spy_monthly = spy.resample('M').agg({
        'Open': 'first', 
        'High': 'max', 
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Create data dictionary
    data_dict = {
        'D': spy,
        'W': spy_weekly,
        'M': spy_monthly
    }
    
    # Create feature pyramid
    pyramid = FeaturePyramid()
    
    # Generate pyramid features
    pyramid_features = pyramid.generate_pyramid(data_dict)
    
    # Print pyramid shape
    print(f"\nGenerated {pyramid_features.shape[1]} pyramid features")
    print(f"First few columns: {pyramid_features.columns[:5].tolist()}")
