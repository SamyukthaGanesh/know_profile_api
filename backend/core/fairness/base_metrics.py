"""
Base Fairness Metrics Abstract Class
Provides a model-agnostic and data-agnostic interface for fairness evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessMetricType(Enum):
    """Types of fairness metrics"""
    STATISTICAL_PARITY = "statistical_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL = "individual"
    COUNTERFACTUAL = "counterfactual"


class BiasDirection(Enum):
    """Direction of detected bias"""
    NO_BIAS = "no_bias"
    POSITIVE_BIAS = "positive_bias"  # Favors protected group
    NEGATIVE_BIAS = "negative_bias"  # Against protected group
    MIXED_BIAS = "mixed_bias"  # Different across groups


@dataclass
class FairnessResult:
    """Standardized output for fairness evaluation"""
    metric_name: str
    metric_type: FairnessMetricType
    
    # Core metrics
    overall_score: float  # 0 = perfectly fair, higher = more bias
    group_scores: Dict[str, float]  # Scores per group
    
    # Statistical analysis
    disparate_impact_ratio: Optional[float] = None
    statistical_significance: Optional[float] = None  # p-value
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Bias detection
    bias_detected: bool = False
    bias_direction: BiasDirection = BiasDirection.NO_BIAS
    affected_groups: Optional[List[str]] = None
    
    # Thresholds and recommendations
    threshold_used: Optional[float] = None
    passes_threshold: bool = True
    recommendation: Optional[str] = None
    
    # Additional details
    sample_sizes: Optional[Dict[str, int]] = None
    raw_rates: Optional[Dict[str, float]] = None
    confusion_matrices: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "overall_score": self.overall_score,
            "group_scores": self.group_scores,
            "bias_detected": self.bias_detected,
            "bias_direction": self.bias_direction.value,
            "affected_groups": self.affected_groups,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "statistical_significance": self.statistical_significance,
            "passes_threshold": self.passes_threshold,
            "recommendation": self.recommendation
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        if not self.bias_detected:
            return f"No significant bias detected (score: {self.overall_score:.3f})"
        else:
            return (f"Bias detected: {self.bias_direction.value} "
                   f"affecting groups: {', '.join(self.affected_groups or [])} "
                   f"(score: {self.overall_score:.3f})")


class BaseFairnessMetric(ABC):
    """
    Abstract base class for all fairness metrics.
    Ensures model-agnostic and data-agnostic implementation.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        threshold: float = 0.8,  # Default 80% rule
        confidence_level: float = 0.95,
        **kwargs
    ):
        """
        Initialize fairness metric calculator.
        
        Args:
            model: ML model to evaluate (optional, can use predictions directly)
            threshold: Fairness threshold (e.g., 0.8 for 80% rule)
            confidence_level: Confidence level for statistical tests
            **kwargs: Additional metric-specific parameters
        """
        self.model = model
        self.threshold = threshold
        self.confidence_level = confidence_level
        self.config = kwargs
        
        # Validate model if provided
        if model is not None:
            self._validate_model()
    
    def _validate_model(self):
        """Validate that model has required methods"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
    
    @abstractmethod
    def calculate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> FairnessResult:
        """
        Calculate fairness metric.
        
        Args:
            X: Input features
            y_true: True labels
            sensitive_feature: Protected attribute values
            y_pred: Predictions (if None, will use model.predict)
            **kwargs: Additional parameters
            
        Returns:
            FairnessResult object
        """
        pass
    
    @abstractmethod
    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate metrics for each group.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            groups: Group labels
            
        Returns:
            Dictionary of metrics per group
        """
        pass
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_features: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> Union[FairnessResult, Dict[str, FairnessResult]]:
        """
        Evaluate fairness across one or multiple sensitive features.
        
        Args:
            X: Input features
            y_true: True labels
            sensitive_features: One or multiple protected attributes
            y_pred: Predictions (optional)
            **kwargs: Additional parameters
            
        Returns:
            Single FairnessResult or dictionary of results
        """
        # Handle multiple sensitive features
        if isinstance(sensitive_features, dict):
            results = {}
            for feature_name, feature_values in sensitive_features.items():
                results[feature_name] = self.calculate(
                    X, y_true, feature_values, y_pred, **kwargs
                )
            return results
        elif isinstance(sensitive_features, pd.DataFrame):
            results = {}
            for col in sensitive_features.columns:
                results[col] = self.calculate(
                    X, y_true, sensitive_features[col], y_pred, **kwargs
                )
            return results
        else:
            # Single sensitive feature
            return self.calculate(X, y_true, sensitive_features, y_pred, **kwargs)
    
    def _get_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get predictions either from provided y_pred or model.
        
        Args:
            X: Input features
            y_pred: Optional predictions
            
        Returns:
            Numpy array of predictions
        """
        if y_pred is not None:
            return self._to_numpy(y_pred)
        elif self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("Either y_pred or model must be provided")
    
    def _to_numpy(self, data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert data to numpy array"""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _get_unique_groups(self, sensitive_feature: np.ndarray) -> List[Any]:
        """Get unique groups from sensitive feature"""
        return list(np.unique(sensitive_feature))
    
    def _calculate_disparate_impact(
        self,
        group_rates: Dict[str, float],
        reference_group: Optional[str] = None
    ) -> float:
        """
        Calculate disparate impact ratio.
        
        Args:
            group_rates: Dictionary of rates per group
            reference_group: Reference group (if None, uses highest rate)
            
        Returns:
            Disparate impact ratio (min_rate / max_rate)
        """
        if len(group_rates) < 2:
            return 1.0
        
        rates = list(group_rates.values())
        
        if reference_group and reference_group in group_rates:
            ref_rate = group_rates[reference_group]
            other_rates = [r for g, r in group_rates.items() if g != reference_group]
            if ref_rate > 0:
                ratios = [r / ref_rate for r in other_rates]
                return min(ratios) if ratios else 1.0
        
        # Default: min/max ratio
        max_rate = max(rates)
        min_rate = min(rates)
        
        if max_rate > 0:
            return min_rate / max_rate
        return 1.0
    
    def _statistical_parity_test(
        self,
        group_rates: Dict[str, float],
        group_sizes: Dict[str, int]
    ) -> Tuple[float, bool]:
        """
        Perform statistical test for parity.
        
        Args:
            group_rates: Rates per group
            group_sizes: Sample sizes per group
            
        Returns:
            Tuple of (p-value, is_significant)
        """
        if len(group_rates) < 2:
            return 1.0, False
        
        # Chi-square test for independence
        groups = list(group_rates.keys())
        
        # Create contingency table
        positive_counts = [int(group_rates[g] * group_sizes[g]) for g in groups]
        negative_counts = [group_sizes[g] - positive_counts[i] for i, g in enumerate(groups)]
        
        contingency_table = np.array([positive_counts, negative_counts])
        
        # Perform chi-square test
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            is_significant = p_value < (1 - self.confidence_level)
            return p_value, is_significant
        except:
            return 1.0, False
    
    def _calculate_confidence_interval(
        self,
        rate: float,
        n: int,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a rate.
        
        Args:
            rate: Observed rate
            n: Sample size
            confidence_level: Confidence level (default: self.confidence_level)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if n == 0:
            return (0.0, 1.0)
        
        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        denominator = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)
    
    def _detect_bias_direction(
        self,
        group_rates: Dict[str, float],
        reference_group: Optional[str] = None
    ) -> Tuple[BiasDirection, List[str]]:
        """
        Detect direction and affected groups.
        
        Args:
            group_rates: Rates per group
            reference_group: Reference group for comparison
            
        Returns:
            Tuple of (bias_direction, affected_groups)
        """
        if len(group_rates) < 2:
            return BiasDirection.NO_BIAS, []
        
        rates = list(group_rates.values())
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        # Check if rates are similar (no bias)
        if std_rate < 0.05 * mean_rate:  # Less than 5% variation
            return BiasDirection.NO_BIAS, []
        
        # Identify disadvantaged groups
        disadvantaged = []
        advantaged = []
        
        for group, rate in group_rates.items():
            if rate < mean_rate - std_rate:
                disadvantaged.append(group)
            elif rate > mean_rate + std_rate:
                advantaged.append(group)
        
        # Determine bias direction
        if disadvantaged and not advantaged:
            return BiasDirection.NEGATIVE_BIAS, disadvantaged
        elif advantaged and not disadvantaged:
            return BiasDirection.POSITIVE_BIAS, advantaged
        elif disadvantaged and advantaged:
            return BiasDirection.MIXED_BIAS, disadvantaged + advantaged
        else:
            return BiasDirection.NO_BIAS, []
    
    def generate_recommendation(
        self,
        result: FairnessResult
    ) -> str:
        """
        Generate recommendation based on fairness result.
        
        Args:
            result: FairnessResult object
            
        Returns:
            Human-readable recommendation
        """
        if not result.bias_detected:
            return "No action required. The model appears to be fair across groups."
        
        recommendations = []
        
        if result.disparate_impact_ratio and result.disparate_impact_ratio < self.threshold:
            recommendations.append(
                f"Consider rebalancing training data or adjusting decision thresholds. "
                f"Disparate impact ratio ({result.disparate_impact_ratio:.2f}) is below "
                f"the threshold ({self.threshold:.2f})."
            )
        
        if result.affected_groups:
            recommendations.append(
                f"Focus on improving outcomes for groups: {', '.join(result.affected_groups)}"
            )
        
        if result.bias_direction == BiasDirection.NEGATIVE_BIAS:
            recommendations.append(
                "Investigate whether model features or training data contain "
                "historical biases against the affected groups."
            )
        
        return " ".join(recommendations) if recommendations else "Review model for potential improvements."


class FairnessMetricFactory:
    """Factory class to create appropriate fairness metrics"""
    
    _metrics = {}
    
    @classmethod
    def register(cls, name: str, metric_class: type):
        """Register a new fairness metric type"""
        cls._metrics[name.lower()] = metric_class
    
    @classmethod
    def create(
        cls,
        metric: str,
        model: Optional[Any] = None,
        **kwargs
    ) -> BaseFairnessMetric:
        """
        Create a fairness metric instance.
        
        Args:
            metric: Name of the fairness metric
            model: ML model to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Fairness metric instance
        """
        metric_lower = metric.lower()
        if metric_lower not in cls._metrics:
            raise ValueError(f"Unknown fairness metric: {metric}")
        
        metric_class = cls._metrics[metric_lower]
        return metric_class(model=model, **kwargs)
    
    @classmethod
    def available_metrics(cls) -> List[str]:
        """Get list of available fairness metrics"""
        return list(cls._metrics.keys())