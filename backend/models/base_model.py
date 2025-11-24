"""
Base Model Interface
Provides a model-agnostic interface for any ML model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    Ensures consistent interface for all model types.
    """
    
    def __init__(self, model: Any, model_type: str = 'classification'):
        """
        Initialize base model.
        
        Args:
            model: The actual ML model object
            model_type: 'classification' or 'regression'
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = None
        self.is_fitted = False
        
        # Validate model
        self._validate_model()
    
    def _validate_model(self):
        """Validate that model has required methods"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
        
        if self.model_type == 'classification' and not hasattr(self.model, 'predict_proba'):
            logger.warning("Classification model doesn't have predict_proba method")
    
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'BaseModel':
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
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
        pass
    
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
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model doesn't support predict_proba")
        
        return self.model.predict_proba(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        if self.feature_names is not None:
            return self.feature_names
        
        # Try to get from model
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        
        return []
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names"""
        self.feature_names = feature_names
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def save(self, path: str):
        """Save model to file"""
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from file"""
        import joblib
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return cls(model)