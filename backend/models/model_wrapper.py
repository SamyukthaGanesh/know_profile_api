"""
Model Wrapper
Wraps any ML model to work seamlessly with the AI governance framework.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelWrapper(BaseModel):
    """
    Universal wrapper for any ML model.
    Works with sklearn, xgboost, lightgbm, catboost, tensorflow, pytorch, etc.
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str = 'classification',
        feature_names: Optional[List[str]] = None,
        model_name: str = 'UnknownModel',
        model_version: str = '1.0'
    ):
        """
        Initialize model wrapper.
        
        Args:
            model: The actual ML model object
            model_type: 'classification' or 'regression'
            feature_names: Names of features
            model_name: Name of the model
            model_version: Version identifier
        """
        super().__init__(model, model_type)
        
        self.model_name = model_name
        self.model_version = model_version
        
        if feature_names:
            self.set_feature_names(feature_names)
        
        # Detect model framework
        self.framework = self._detect_framework()
        logger.info(f"Wrapped {self.framework} model: {model_name} v{model_version}")
    
    def _detect_framework(self) -> str:
        """Detect which ML framework is being used"""
        model_module = type(self.model).__module__
        
        if 'sklearn' in model_module:
            return 'sklearn'
        elif 'xgboost' in model_module:
            return 'xgboost'
        elif 'lightgbm' in model_module:
            return 'lightgbm'
        elif 'catboost' in model_module:
            return 'catboost'
        elif 'tensorflow' in model_module or 'keras' in model_module:
            return 'tensorflow'
        elif 'torch' in model_module:
            return 'pytorch'
        else:
            return 'unknown'
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'ModelWrapper':
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional parameters (e.g., eval_set, early_stopping_rounds)
            
        Returns:
            Self
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.set_feature_names(list(X.columns))
            X_train = X.values
        else:
            X_train = X
        
        # Convert y to array if needed
        if isinstance(y, pd.Series):
            y_train = y.values
        else:
            y_train = y
        
        # Fit the model
        logger.info(f"Fitting {self.model_name}...")
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True
        logger.info(f"Model fitted successfully")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted and not self._check_if_fitted():
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X
        
        predictions = self.model.predict(X_pred)
        return predictions
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict probabilities (for classification).
        
        Args:
            X: Features to predict
            
        Returns:
            Probability predictions
        """
        if self.model_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        if not self.is_fitted and not self._check_if_fitted():
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_pred = X.values
        else:
            X_pred = X
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model doesn't have predict_proba, using predict instead")
            predictions = self.model.predict(X_pred)
            # Convert to pseudo-probabilities
            if predictions.ndim == 1:
                # Binary classification
                proba = np.column_stack([1 - predictions, predictions])
            else:
                proba = predictions
            return proba
        
        probabilities = self.model.predict_proba(X_pred)
        return probabilities
    
    def _check_if_fitted(self) -> bool:
        """Check if model is fitted"""
        # Try common attributes that indicate a fitted model
        fitted_attributes = [
            'coef_', 'feature_importances_', 'n_features_in_',
            'classes_', 'n_classes_', 'tree_', 'estimators_'
        ]
        
        for attr in fitted_attributes:
            if hasattr(self.model, attr):
                self.is_fitted = True
                return True
        
        return False
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        importance = None
        
        # Try different methods based on framework
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (sklearn, xgboost, lightgbm, catboost)
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models (sklearn)
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multi-class: take mean absolute value
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)
        
        if importance is not None and self.feature_names:
            return dict(zip(self.feature_names, importance))
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_type': self.model_type,
            'framework': self.framework,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_classes_'):
            info['n_classes'] = self.model.n_classes_
        
        if hasattr(self.model, 'classes_'):
            info['classes'] = list(self.model.classes_)
        
        return info
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            metrics: List of metric names
            
        Returns:
            Dictionary of metric scores
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
        )
        
        if metrics is None:
            if self.model_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            else:
                metrics = ['mse', 'mae', 'r2']
        
        # Get predictions
        y_pred = self.predict(X)
        
        results = {}
        
        if self.model_type == 'classification':
            # Classification metrics
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(y, y_pred)
            
            if 'precision' in metrics:
                results['precision'] = precision_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted')
            
            if 'recall' in metrics:
                results['recall'] = recall_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted')
            
            if 'f1' in metrics:
                results['f1'] = f1_score(y, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted')
            
            if 'auc' in metrics and hasattr(self.model, 'predict_proba'):
                y_proba = self.predict_proba(X)
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                    # Binary classification
                    results['auc'] = roc_auc_score(y, y_proba[:, 1])
                elif y_proba.ndim > 1:
                    # Multi-class
                    results['auc'] = roc_auc_score(y, y_proba, multi_class='ovr')
        
        else:
            # Regression metrics
            if 'mse' in metrics:
                results['mse'] = mean_squared_error(y, y_pred)
            
            if 'rmse' in metrics:
                results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            
            if 'mae' in metrics:
                results['mae'] = mean_absolute_error(y, y_pred)
            
            if 'r2' in metrics:
                results['r2'] = r2_score(y, y_pred)
        
        return results
    
    def get_prediction_explanation_context(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        instance_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Get context for explaining a prediction.
        Useful for creating ExplanationContext objects.
        
        Args:
            X: Features
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with prediction context
        """
        # Get single instance
        if isinstance(X, pd.DataFrame):
            instance = X.iloc[instance_idx:instance_idx+1]
            feature_values = instance.iloc[0].to_dict()
        else:
            instance = X[instance_idx:instance_idx+1]
            feature_values = {
                f"feature_{i}": float(val) 
                for i, val in enumerate(instance[0])
            }
        
        # Get prediction
        prediction = self.predict(instance)[0]
        
        # Get confidence if available
        confidence = None
        if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.predict_proba(instance)[0]
            confidence = float(np.max(proba))
        
        context = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction': prediction,
            'confidence': confidence,
            'feature_values': feature_values,
            'feature_names': self.feature_names
        }
        
        return context


def create_model_wrapper(
    model: Any,
    model_type: str = 'classification',
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> ModelWrapper:
    """
    Convenience function to create a model wrapper.
    
    Args:
        model: ML model to wrap
        model_type: 'classification' or 'regression'
        feature_names: Names of features
        **kwargs: Additional parameters
        
    Returns:
        ModelWrapper instance
    """
    return ModelWrapper(
        model=model,
        model_type=model_type,
        feature_names=feature_names,
        **kwargs
    )