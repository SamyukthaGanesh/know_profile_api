"""
Base interfaces for fairness optimizers.

This module defines the abstract base classes and interfaces that all
fairness optimization strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseFairnessOptimizer(ABC):
    """Abstract base class for fairness optimization strategies."""
    
    @abstractmethod
    def fit(self, estimator, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], 
            sensitive_features: Union[pd.Series, np.ndarray],
            config: Dict[str, Any]) -> BaseEstimator:
        """
        Fit fairness optimizer with given data.
        
        Args:
            estimator: Base estimator to optimize
            X: Training features
            y: Training labels
            sensitive_features: Sensitive feature values
            config: Configuration parameters
            
        Returns:
            Fitted estimator
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of optimization strategy."""
        pass


class BaseEnsembleOptimizer(BaseFairnessOptimizer):
    """Base class for ensemble-based fairness optimizers."""
    
    @abstractmethod
    def create_ensemble(self, base_estimator, X, y, sensitive_features, 
                       n_estimators: int, config: Dict[str, Any]) -> BaseEstimator:
        """
        Create ensemble of fairness-aware models.
        
        Args:
            base_estimator: Base estimator for ensemble
            X: Training features
            y: Training labels  
            sensitive_features: Sensitive feature values
            n_estimators: Number of estimators in ensemble
            config: Configuration parameters
            
        Returns:
            Fitted ensemble estimator
        """
        pass