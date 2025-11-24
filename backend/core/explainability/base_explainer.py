"""
Base Explainer Abstract Class
Provides a model-agnostic and data-agnostic interface for all explainability methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations supported"""
    LOCAL = "local"  # Single prediction explanation
    GLOBAL = "global"  # Overall model explanation
    BOTH = "both"  # Supports both local and global


class ExplainerOutput(Enum):
    """Output formats for explanations"""
    FEATURE_IMPORTANCE = "feature_importance"
    RULES = "rules"
    COUNTERFACTUAL = "counterfactual"
    ATTRIBUTION = "attribution"


@dataclass
class ExplanationResult:
    """Standardized output for all explainability methods"""
    method: str  # Name of the method (SHAP, LIME, etc.)
    explanation_type: ExplanationType
    
    # Core explanation data
    feature_importance: Optional[Dict[str, float]] = None
    feature_values: Optional[Dict[str, Any]] = None
    
    # Additional explanation formats
    rules: Optional[List[str]] = None
    counterfactuals: Optional[pd.DataFrame] = None
    attributions: Optional[np.ndarray] = None
    
    # Metadata
    base_value: Optional[float] = None  # For SHAP
    prediction: Optional[Union[float, int, np.ndarray]] = None
    confidence: Optional[float] = None
    coverage: Optional[float] = None  # For Anchors
    precision: Optional[float] = None  # For Anchors
    
    # Visualization data
    plot_data: Optional[Dict[str, Any]] = None
    
    # Raw output from the underlying library
    raw_output: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            "method": self.method,
            "explanation_type": self.explanation_type.value,
            "prediction": self.prediction
        }
        
        if self.feature_importance:
            result["feature_importance"] = self.feature_importance
            
        if self.feature_values:
            result["feature_values"] = self.feature_values
            
        if self.rules:
            result["rules"] = self.rules
            
        if self.confidence is not None:
            result["confidence"] = self.confidence
            
        if self.base_value is not None:
            result["base_value"] = self.base_value
            
        return result
    
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n most important features"""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


class BaseExplainer(ABC):
    """
    Abstract base class for all explainability methods.
    Ensures model-agnostic and data-agnostic implementation.
    """
    
    def __init__(
        self,
        model: Any,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        **kwargs
    ):
        """
        Initialize the explainer.
        
        Args:
            model: Any ML model with predict/predict_proba methods
            data: Training or background data for explanation
            feature_names: Names of the features
            class_names: Names of the classes (for classification)
            mode: 'classification' or 'regression'
            **kwargs: Additional method-specific parameters
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names or self._infer_feature_names(data)
        self.class_names = class_names
        self.mode = mode
        self.config = kwargs
        
        # Validate model
        self._validate_model()
        
        # Initialize the specific explainer
        self._initialize()
        
        logger.info(f"Initialized {self.__class__.__name__} for {mode}")
    
    def _infer_feature_names(self, data: Optional[Union[pd.DataFrame, np.ndarray]]) -> List[str]:
        """Infer feature names from data"""
        if data is None:
            return []
        
        if isinstance(data, pd.DataFrame):
            return data.columns.tolist()
        elif isinstance(data, np.ndarray):
            return [f"feature_{i}" for i in range(data.shape[1])]
        else:
            return []
    
    def _validate_model(self):
        """Validate that the model has required methods"""
        if not hasattr(self.model, 'predict'):
            raise ValueError(f"Model must have a 'predict' method")
        
        if self.mode == "classification" and not hasattr(self.model, 'predict_proba'):
            # Try to handle models that only have predict
            logger.warning("Classification model doesn't have predict_proba, will use predict")
    
    @abstractmethod
    def _initialize(self):
        """Initialize the specific explainer (SHAP, LIME, etc.)"""
        pass
    
    @abstractmethod
    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single prediction (local explanation).
        
        Args:
            instance: Single instance to explain
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult object
        """
        pass
    
    @abstractmethod
    def explain_global(
        self,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain the overall model behavior (global explanation).
        
        Args:
            n_samples: Number of samples to use for global explanation
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult object
        """
        pass
    
    @abstractmethod
    def get_feature_importance(
        self,
        method: str = "mean_abs",
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            method: Method to calculate importance ('mean_abs', 'gain', etc.)
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary of feature names to importance scores
        """
        pass
    
    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray, None] = None,
        explanation_type: str = "local",
        **kwargs
    ) -> Union[ExplanationResult, List[ExplanationResult]]:
        """
        Unified interface for all explanation types.
        
        Args:
            X: Data to explain (None for global)
            explanation_type: 'local' or 'global'
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult or list of ExplanationResults
        """
        if explanation_type == "global":
            return self.explain_global(**kwargs)
        elif explanation_type == "local":
            if X is None:
                raise ValueError("Data required for local explanations")
            
            # Handle single instance or multiple instances
            if isinstance(X, pd.DataFrame):
                if len(X) == 1:
                    return self.explain_instance(X, **kwargs)
                else:
                    return [self.explain_instance(X.iloc[[i]], **kwargs) 
                           for i in range(len(X))]
            elif isinstance(X, np.ndarray):
                if X.ndim == 1 or (X.ndim == 2 and len(X) == 1):
                    return self.explain_instance(X, **kwargs)
                else:
                    return [self.explain_instance(X[i:i+1], **kwargs) 
                           for i in range(len(X))]
        else:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
    
    def validate_input(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Validate and convert input to numpy array.
        
        Args:
            X: Input data
            
        Returns:
            Numpy array
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f"Input must be pandas DataFrame or numpy array, got {type(X)}")
    
    def format_explanation_for_api(
        self,
        explanation: ExplanationResult,
        include_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Format explanation for API response.
        
        Args:
            explanation: ExplanationResult object
            include_raw: Whether to include raw output
            
        Returns:
            Dictionary ready for JSON serialization
        """
        result = explanation.to_dict()
        
        # Remove raw output if not needed (usually too large)
        if not include_raw and 'raw_output' in result:
            del result['raw_output']
        
        # Convert numpy types to Python types
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
        
        return result
    
    @property
    def supports_global(self) -> bool:
        """Check if the method supports global explanations"""
        return True
    
    @property
    def supports_local(self) -> bool:
        """Check if the method supports local explanations"""
        return True
    
    @property
    def explanation_type(self) -> ExplanationType:
        """Get the type of explanations this method supports"""
        if self.supports_global and self.supports_local:
            return ExplanationType.BOTH
        elif self.supports_global:
            return ExplanationType.GLOBAL
        else:
            return ExplanationType.LOCAL


class ExplainerFactory:
    """Factory class to create appropriate explainers"""
    
    _explainers = {}
    
    @classmethod
    def register(cls, name: str, explainer_class: type):
        """Register a new explainer type"""
        cls._explainers[name.lower()] = explainer_class
    
    @classmethod
    def create(
        cls,
        method: str,
        model: Any,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        **kwargs
    ) -> BaseExplainer:
        """
        Create an explainer instance.
        
        Args:
            method: Name of the explanation method ('shap', 'lime', etc.)
            model: ML model to explain
            data: Training/background data
            **kwargs: Additional parameters
            
        Returns:
            Explainer instance
        """
        method_lower = method.lower()
        if method_lower not in cls._explainers:
            raise ValueError(f"Unknown explanation method: {method}")
        
        explainer_class = cls._explainers[method_lower]
        return explainer_class(model=model, data=data, **kwargs)
    
    @classmethod
    def available_methods(cls) -> List[str]:
        """Get list of available explanation methods"""
        return list(cls._explainers.keys())