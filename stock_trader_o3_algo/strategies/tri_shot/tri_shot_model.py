#!/usr/bin/env python3
"""
Walk-Forward Tri-Shot Model

This module implements a walk-forward validation framework for XGBoost models
designed to predict market directional moves while avoiding overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path

# For ML training
try:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    import shap
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# Local imports
from . import tri_shot_features as tsf


class WalkForwardModel:
    """Enhanced model with walk-forward validation and ensemble learning."""
    
    def __init__(self, 
                 state_dir: Path = Path(os.path.join(os.getcwd(), "tri_shot_data")),
                 n_folds: int = 4,
                 lookback_days: int = 1260,  # ~5 years data
                 class_weight: Optional[Dict[int, float]] = None,
                 calibrate: bool = True,
                 use_focal_loss: bool = True,
                 feature_selection: bool = True,
                 top_n_features: int = 25):
        """
        Initialize the model with advanced configuration.
        
        Args:
            state_dir: Directory for model storage
            n_folds: Number of folds for walk-forward CV
            lookback_days: Maximum historical days to use
            class_weight: Weighting for imbalanced classes
            calibrate: Whether to calibrate predicted probabilities
            use_focal_loss: Use focal loss for emphasizing hard examples
            feature_selection: Use SHAP for feature selection
            top_n_features: Number of top features to keep
        """
        self.state_dir = state_dir
        self.n_folds = n_folds
        self.lookback_days = lookback_days
        
        # Use balanced class weights by default - up moves are more common than down moves
        if class_weight is None:
            self.class_weight = {0: 1.2, 1: 0.8}  # Slight overweight on down-moves
        else:
            self.class_weight = class_weight
            
        self.calibrate = calibrate
        self.use_focal_loss = use_focal_loss
        self.feature_selection = feature_selection
        self.top_n_features = top_n_features
        
        # Models
        self.models = {}
        self.feature_importances = None
        self.selected_features = None
        self.metaclassifier = None
    
    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Focal loss implementation for XGBoost.
        Focuses training on hard examples by down-weighting easy ones.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            gamma: Focusing parameter
            alpha: Class balancing parameter
            
        Returns:
            Gradient and hessian for XGBoost custom objective
        """
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # For binary classification
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Focal term
        focal_weight = alpha_t * (1 - p_t) ** gamma
        
        # Gradient and hessian for XGBoost
        grad = -focal_weight * (y_true - y_pred)
        hess = focal_weight * p_t * (1 - p_t)
        
        return grad, hess
    
    def train(self, prices: pd.DataFrame, target_ticker: str = "QQQ"):
        """
        Train the model using walk-forward validation and stacked ensemble.
        
        Args:
            prices: DataFrame with price data
            target_ticker: Ticker to predict
            
        Returns:
            Dictionary with performance metrics
        """
        if not HAS_ML_DEPS:
            raise ImportError("ML dependencies not available. Install xgboost, scikit-learn, and shap.")
        
        print("Preparing data for training...")
        X, y = tsf.make_feature_matrix(prices, target_ticker)
        
        # Initialize TimeSeriesSplit for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        print(f"Training with {self.n_folds}-fold walk-forward validation...")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold+1}/{self.n_folds}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # Check for features that may cause issues (all zeros, all NaN)
            valid_features = ~X_train.isna().all() & (X_train.std() > 0)
            if not valid_features.all():
                print(f"Removing {sum(~valid_features)} invalid features")
                X_train = X_train.loc[:, valid_features]
                X_test = X_test.loc[:, valid_features]
            
            # Feature selection with SHAP if requested
            if self.feature_selection and fold == 0:
                print("Performing feature selection with SHAP...")
                # Train a quick model for feature selection
                feature_selector = xgb.XGBClassifier(
                    max_depth=3,
                    n_estimators=100,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight=self.class_weight
                )
                feature_selector.fit(X_train, y_train)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(feature_selector)
                shap_values = explainer.shap_values(X_train)
                
                # Get feature importance
                feature_importances = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False)
                
                # Select top N features
                self.selected_features = feature_importances.head(self.top_n_features)['feature'].tolist()
                print(f"Selected top {len(self.selected_features)} features")
                
                # Update dataframes to use only selected features
                X_train = X_train[self.selected_features]
                X_test = X_test[self.selected_features]
            elif self.feature_selection and self.selected_features:
                # Use previously selected features for later folds
                X_train = X_train[self.selected_features]
                X_test = X_test[self.selected_features]
            
            # Train base models
            base_models = {
                'xgb': self._train_xgb(X_train, y_train),
                'xgb_light': self._train_xgb_light(X_train, y_train),
            }
            
            # Generate predictions from base models
            base_preds = {}
            for name, model in base_models.items():
                # Use robust helper to avoid IndexError when a fold contains
                # only a single class in the training data.
                preds = self._get_positive_proba(model, X_test)
                base_preds[name] = preds
            
            # Store model for this fold
            self.models[f'fold_{fold+1}'] = base_models
            
            # Calculate metrics for this fold
            y_pred_proba = np.mean([preds for preds in base_preds.values()], axis=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_metric = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'log_loss': log_loss(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Calculate directional accuracy (predicting up/down correctly)
            fold_metric['directional_accuracy'] = (
                (y_test == 1) & (y_pred == 1) | (y_test == 0) & (y_pred == 0)
            ).mean()
            
            print(f"Fold {fold+1} metrics: Accuracy={fold_metric['accuracy']:.4f}, "
                  f"Directional={fold_metric['directional_accuracy']:.4f}")
            
            fold_metrics.append(fold_metric)
            all_predictions.extend(y_pred_proba)
            all_actuals.extend(y_test)
        
        # Train meta-classifier on all out-of-fold predictions
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions).reshape(-1, 1)
        
        self.metaclassifier = LogisticRegression(class_weight=self.class_weight)
        self.metaclassifier.fit(all_predictions, all_actuals)
        
        # Final model metrics
        calibrated_preds = self.metaclassifier.predict_proba(all_predictions)[:, 1]
        final_preds = (calibrated_preds > 0.5).astype(int)
        
        final_metrics = {
            'accuracy': accuracy_score(all_actuals, final_preds),
            'precision': precision_score(all_actuals, final_preds, zero_division=0),
            'recall': recall_score(all_actuals, final_preds, zero_division=0),
            'f1': f1_score(all_actuals, final_preds, zero_division=0),
            'log_loss': log_loss(all_actuals, calibrated_preds),
            'confusion_matrix': confusion_matrix(all_actuals, final_preds).tolist(),
            'fold_metrics': fold_metrics
        }
        
        # Calculate directional accuracy
        final_metrics['directional_accuracy'] = (
            (all_actuals == 1) & (final_preds == 1) | (all_actuals == 0) & (final_preds == 0)
        ).mean()
        
        print(f"Final model metrics: Accuracy={final_metrics['accuracy']:.4f}, "
              f"Directional={final_metrics['directional_accuracy']:.4f}")
        
        return final_metrics
    
    def _get_positive_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        """Return probability of class 1 ("up") in a robust way.

        Some sklearn models trained on a dataset that happens to contain only
        a single class will expose *one* probability column instead of two
        (e.g. shape ``(n_samples, 1)``). Attempting to index ``[:, 1]`` then
        triggers an ``IndexError``. This helper inspects the model's
        ``classes_`` attribute when present and gracefully falls back so that
        the caller always receives a 1-D array of probabilities for the
        positive class.
        """
        proba = model.predict_proba(X)
        # If predict_proba already returns 1-D assume it is the positive class
        if proba.ndim == 1:
            return proba
        # If only one probability column is returned (n_classes == 1)
        if proba.shape[1] == 1:
            return proba[:, 0]
        # Standard two-column case – identify index of class 1 if possible
        if hasattr(model, "classes_"):
            classes = model.classes_
            if 1 in classes:
                pos_idx = int(np.where(classes == 1)[0][0])
                return proba[:, pos_idx]
        # Fallback – take the second column
        return proba[:, 1]
    
    def _train_xgb(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train an XGBoost model with tuned parameters."""
        model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=self.class_weight.get(0, 1.0) / self.class_weight.get(1, 1.0)
        )
        
        if self.use_focal_loss:
            model.fit(X, y, verbose=False)
        else:
            model.fit(X, y)
        
        if self.calibrate:
            # Calibrate probabilities with isotonic regression
            calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrator.fit(X, y)
            return calibrator
        
        return model
    
    def _train_xgb_light(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train a lighter XGBoost model with different parameters."""
        model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0,
            scale_pos_weight=self.class_weight.get(0, 1.0) / self.class_weight.get(1, 1.0)
        )
        
        model.fit(X, y)
        
        if self.calibrate:
            # Calibrate probabilities with Platt scaling
            calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            calibrator.fit(X, y)
            return calibrator
        
        return model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted probabilities
        """
        if not self.models:
            raise ValueError("Model not trained yet.")
        
        # Feature selection if needed
        if self.feature_selection and self.selected_features:
            X = X[self.selected_features]
        
        # Generate predictions from all fold models
        all_fold_preds = []
        for fold_name, fold_models in self.models.items():
            fold_preds = []
            for model_name, model in fold_models.items():
                fold_preds.append(self._get_positive_proba(model, X))
            # Average predictions from models in this fold
            all_fold_preds.append(np.mean(fold_preds, axis=0))
        
        # Average predictions across all folds
        ensemble_preds = np.mean(all_fold_preds, axis=0).reshape(-1, 1)
        
        # Apply meta-classifier if available
        if self.metaclassifier is not None:
            final_preds = self.metaclassifier.predict_proba(ensemble_preds)[:, 1]
        else:
            final_preds = ensemble_preds.flatten()
        
        return final_preds
    
    def save(self, filename: str = "tri_shot_ensemble.pkl"):
        """Save the trained model to disk."""
        model_path = self.state_dir / filename
        joblib.dump(self, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load(cls, filename: str = "tri_shot_ensemble.pkl", state_dir: Path = Path(os.path.expanduser("~/.tri_shot"))):
        """Load a trained model from disk."""
        model_path = state_dir / filename
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def plot_feature_importance(self, save_path: Optional[str] = None):
        """Plot feature importance from the model."""
        if not self.feature_importances:
            print("No feature importance data available.")
            return
        
        plt.figure(figsize=(12, 10))
        plt.barh(self.feature_importances['feature'].head(20), 
                 self.feature_importances['importance'].head(20))
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()


def train_walk_forward_model(prices: pd.DataFrame, target_ticker: str = "QQQ",
                           state_dir: Path = Path(os.path.expanduser("~/.tri_shot")),
                           save_model: bool = True):
    """
    Train a walk-forward model and save it.
    
    Args:
        prices: DataFrame with price data
        target_ticker: Ticker to predict
        state_dir: Directory for model storage
        save_model: Whether to save the trained model
        
    Returns:
        Trained model and metrics
    """
    # Ensure state directory exists
    if not state_dir.exists():
        state_dir.mkdir(parents=True)
    
    # Create and train model
    model = WalkForwardModel(state_dir=state_dir)
    metrics = model.train(prices, target_ticker)
    
    # Save model if requested
    if save_model:
        model.save()
    
    return model, metrics


def load_walk_forward_model(state_dir: Path = Path(os.path.expanduser("~/.tri_shot"))):
    """
    Load a previously saved walk-forward model.
    
    Args:
        state_dir: Directory with the saved model
        
    Returns:
        Loaded model or None if not found
    """
    return WalkForwardModel.load(state_dir=state_dir)
