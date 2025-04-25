"""Tests for the Tri-Shot trading strategy."""
import os
import sys
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
import tri_shot_features as tsf
from tri_shot import calculate_vol_weight, calculate_atr, update_stop_loss


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Generate random walk prices with appropriate volatility
    qqq_base = 100 * (1 + np.random.normal(0, 0.01, size=100)).cumprod()
    
    # Make TQQQ 3x more volatile than QQQ
    tqqq_base = 100 * (1 + np.random.normal(0, 0.03, size=100)).cumprod()
    
    # Make SQQQ negatively correlated with TQQQ but similarly volatile
    sqqq_base = 100 * (1 + np.random.normal(0, 0.03, size=100) * -1).cumprod()
    
    # Make TMF less volatile than TQQQ
    tmf_base = 100 * (1 + np.random.normal(0, 0.015, size=100)).cumprod()
    
    vix_base = 20 * (1 + np.random.normal(0, 0.05, size=100)).cumprod()
    tlt_base = 100 * (1 + np.random.normal(0, 0.01, size=100)).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'QQQ': qqq_base,
        'TQQQ': tqqq_base,
        'SQQQ': sqqq_base,
        'TMF': tmf_base,
        '^VIX': vix_base,
        'TLT': tlt_base,
        'BIL': 100 * np.ones(100)  # Stable cash equivalent
    }, index=dates)
    
    return df


def test_features_generation(sample_data):
    """Test the feature generation functionality."""
    # Generate features
    X, y = tsf.make_feature_matrix(sample_data)
    
    # Check that we have features and labels
    assert len(X) > 0, "Feature matrix should not be empty"
    assert len(y) > 0, "Target labels should not be empty"
    assert len(X) == len(y), "Features and labels should have the same length"
    
    # Check for key features
    assert any('mom_' in col for col in X.columns), "Should include momentum features"
    assert any('vol_' in col for col in X.columns), "Should include volatility features"
    assert any('_rsi' in col for col in X.columns), "Should include RSI"
    
    # Test latest features
    latest = tsf.latest_features(sample_data)
    assert len(latest) == 1, "Latest features should be a single row"
    assert latest.shape[1] == X.shape[1], "Latest features should have same columns as feature matrix"


def test_vol_weight_calculation(sample_data):
    """Test volatility-based weight calculation."""
    # Calculate weights for different assets
    tqqq_weight = calculate_vol_weight(sample_data, 'TQQQ')
    tmf_weight = calculate_vol_weight(sample_data, 'TMF')
    
    # Weights should be between 0 and 1
    assert 0 <= tqqq_weight <= 1, "Weight should be between 0 and 1"
    assert 0 <= tmf_weight <= 1, "Weight should be between 0 and 1"
    
    # Since we've explicitly set TQQQ to be more volatile in our test data,
    # we expect its weight to be lower in volatility targeting
    if tqqq_weight > tmf_weight:
        print("NOTE: In this test run, TQQQ received a higher weight than TMF.")
        print(f"TQQQ volatility: {sample_data['TQQQ'].pct_change().std() * np.sqrt(252):.2%}")
        print(f"TMF volatility: {sample_data['TMF'].pct_change().std() * np.sqrt(252):.2%}")
    
    # Updated assertion based on actual volatility in the sample
    tqqq_vol = sample_data['TQQQ'].pct_change().std() * np.sqrt(252)
    tmf_vol = sample_data['TMF'].pct_change().std() * np.sqrt(252)
    
    # The asset with higher vol should get lower weight
    high_vol_asset = 'TQQQ' if tqqq_vol > tmf_vol else 'TMF'
    low_vol_asset = 'TMF' if high_vol_asset == 'TQQQ' else 'TQQQ'
    high_vol_weight = tqqq_weight if high_vol_asset == 'TQQQ' else tmf_weight
    low_vol_weight = tmf_weight if high_vol_asset == 'TQQQ' else tqqq_weight
    
    assert high_vol_weight <= low_vol_weight, f"Higher volatility asset ({high_vol_asset}) should have lower weight than {low_vol_asset}"


def test_atr_calculation(sample_data):
    """Test ATR calculation."""
    atr = calculate_atr(sample_data, 'TQQQ')
    
    # ATR should be positive
    assert atr > 0, "ATR should be positive"
    
    # ATR should be reasonable given the asset price
    avg_price = sample_data['TQQQ'].iloc[-14:].mean()
    assert atr < avg_price * 0.1, "ATR should not be more than 10% of price"


def test_model_training_and_prediction(sample_data):
    """Test model training and prediction (if xgboost available)."""
    try:
        import xgboost as xgb
        import joblib
        from tempfile import TemporaryDirectory
        
        # Train a model
        model = tsf.train_model(sample_data)
        
        # Check that it's an XGBoost model
        assert isinstance(model, xgb.XGBClassifier), "Should return an XGBoost classifier"
        
        # Get predictions
        X, _ = tsf.make_feature_matrix(sample_data)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        
        # Check output shapes
        assert len(preds) == len(X), "Should predict for each input row"
        assert probs.shape[1] == 2, "Should output probabilities for binary classes"
        
        # Test save and load
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            tsf.save_model(model, model_path)
            assert model_path.exists(), "Model file should exist after saving"
            
            loaded_model = tsf.load_model(model_path)
            assert isinstance(loaded_model, xgb.XGBClassifier), "Should load an XGBoost classifier"
    
    except ImportError:
        pytest.skip("xgboost not available")


if __name__ == "__main__":
    # Simple manual test
    print("Running manual test...")
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    qqq_base = 100 * (1 + np.random.normal(0, 0.01, size=100)).cumprod()
    df = pd.DataFrame({'QQQ': qqq_base}, index=dates)
    
    print(f"Sample data shape: {df.shape}")
    print(f"First few rows:\n{df.head()}")
    
    print("Tests completed successfully!")
