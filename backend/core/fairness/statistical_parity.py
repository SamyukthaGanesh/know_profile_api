"""
Statistical Parity Fairness Metric
Measures whether positive outcomes are equally likely across different groups.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from .base_metrics import (
    BaseFairnessMetric, 
    FairnessResult, 
    FairnessMetricType,
    BiasDirection,
    FairnessMetricFactory
)

logger = logging.getLogger(__name__)


class StatisticalParity(BaseFairnessMetric):
    """
    Statistical Parity (Demographic Parity) fairness metric.
    
    Checks if P(Y=1|A=a) = P(Y=1|A=b) for all groups a, b.
    In other words, the probability of positive outcome should be the same across groups.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        threshold: float = 0.8,  # 80% rule
        absolute_threshold: Optional[float] = 0.1,  # Max 10% difference
        use_calibrated: bool = False,
        **kwargs
    ):
        """
        Initialize Statistical Parity metric.
        
        Args:
            model: ML model to evaluate
            threshold: Ratio threshold (default 0.8 for 80% rule)
            absolute_threshold: Maximum absolute difference allowed
            use_calibrated: Whether to use calibrated predictions for binary classification
            **kwargs: Additional parameters
        """
        super().__init__(model=model, threshold=threshold, **kwargs)
        self.absolute_threshold = absolute_threshold
        self.use_calibrated = use_calibrated
        self.metric_name = "Statistical Parity"
    
    def calculate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> FairnessResult:
        """
        Calculate Statistical Parity metric.
        
        Args:
            X: Input features
            y_true: True labels (not used for statistical parity, but kept for consistency)
            sensitive_feature: Protected attribute values
            y_pred: Predictions (if None, will use model.predict)
            sample_weight: Optional sample weights
            **kwargs: Additional parameters
            
        Returns:
            FairnessResult object
        """
        # Get predictions
        predictions = self._get_predictions(X, y_pred)
        
        # Convert to numpy arrays
        predictions = self._to_numpy(predictions)
        sensitive_feature = self._to_numpy(sensitive_feature)
        y_true_np = self._to_numpy(y_true)
        
        # Get binary predictions if needed
        if self.use_calibrated and len(np.unique(predictions)) > 2:
            # Assume probability predictions, use 0.5 threshold
            predictions = (predictions > 0.5).astype(int)
        elif predictions.dtype == float and predictions.max() <= 1 and predictions.min() >= 0:
            # Probability predictions
            predictions = (predictions > 0.5).astype(int)
        
        # Calculate group metrics
        group_metrics = self.calculate_group_metrics(
            y_true_np, predictions, sensitive_feature, sample_weight
        )
        
        # Calculate overall statistical parity score
        overall_score = self._calculate_overall_score(group_metrics['positive_rates'])
        
        # Calculate disparate impact ratio
        disparate_impact = self._calculate_disparate_impact(group_metrics['positive_rates'])
        
        # Statistical significance test
        p_value, is_significant = self._statistical_parity_test(
            group_metrics['positive_rates'],
            group_metrics['group_sizes']
        )
        
        # Detect bias direction
        bias_direction, affected_groups = self._detect_bias_direction(
            group_metrics['positive_rates']
        )
        
        # Check if passes threshold
        passes_threshold = self._check_threshold(
            group_metrics['positive_rates'],
            disparate_impact
        )
        
        # Create result
        result = FairnessResult(
            metric_name=self.metric_name,
            metric_type=FairnessMetricType.STATISTICAL_PARITY,
            overall_score=overall_score,
            group_scores=group_metrics['positive_rates'],
            disparate_impact_ratio=disparate_impact,
            statistical_significance=p_value,
            bias_detected=is_significant and not passes_threshold,
            bias_direction=bias_direction if is_significant else BiasDirection.NO_BIAS,
            affected_groups=affected_groups if is_significant else [],
            threshold_used=self.threshold,
            passes_threshold=passes_threshold,
            sample_sizes=group_metrics['group_sizes'],
            raw_rates=group_metrics['positive_rates'],
            confusion_matrices=group_metrics.get('confusion_matrices')
        )
        
        # Generate recommendation
        result.recommendation = self.generate_recommendation(result)
        
        return result
    
    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate positive rates for each group.
        
        Args:
            y_true: True labels (for confusion matrix)
            y_pred: Predictions
            groups: Group labels
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary with positive rates and other metrics per group
        """
        unique_groups = self._get_unique_groups(groups)
        
        positive_rates = {}
        group_sizes = {}
        confusion_matrices = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            if sample_weight is not None:
                group_weight = sample_weight[group_mask]
                # Weighted positive rate
                positive_count = np.sum((group_pred == 1) * group_weight)
                total_weight = np.sum(group_weight)
                positive_rate = positive_count / total_weight if total_weight > 0 else 0
                group_sizes[str(group)] = int(np.sum(group_weight))
            else:
                # Unweighted positive rate
                positive_rate = np.mean(group_pred == 1)
                group_sizes[str(group)] = int(np.sum(group_mask))
            
            positive_rates[str(group)] = float(positive_rate)
            
            # Calculate confusion matrix
            confusion_matrices[str(group)] = self._calculate_confusion_matrix(
                group_true, group_pred
            )
        
        # Add overall statistics
        overall_positive_rate = np.mean(y_pred == 1)
        
        return {
            'positive_rates': positive_rates,
            'group_sizes': group_sizes,
            'confusion_matrices': confusion_matrices,
            'overall_positive_rate': float(overall_positive_rate),
            'unique_groups': [str(g) for g in unique_groups]
        }
    
    def _calculate_overall_score(self, group_rates: Dict[str, float]) -> float:
        """
        Calculate overall statistical parity score.
        0 = perfect parity, higher = more disparity
        
        Args:
            group_rates: Positive rates per group
            
        Returns:
            Overall score
        """
        if len(group_rates) < 2:
            return 0.0
        
        rates = list(group_rates.values())
        
        # Method 1: Maximum absolute difference
        max_rate = max(rates)
        min_rate = min(rates)
        max_diff = max_rate - min_rate
        
        # Method 2: Standard deviation
        std_dev = np.std(rates)
        
        # Method 3: Average absolute deviation from mean
        mean_rate = np.mean(rates)
        avg_deviation = np.mean([abs(r - mean_rate) for r in rates])
        
        # Combine methods (weighted average)
        overall_score = (max_diff * 0.5 + std_dev * 0.3 + avg_deviation * 0.2)
        
        return float(overall_score)
    
    def _check_threshold(
        self,
        group_rates: Dict[str, float],
        disparate_impact: float
    ) -> bool:
        """
        Check if metric passes fairness thresholds.
        
        Args:
            group_rates: Positive rates per group
            disparate_impact: Disparate impact ratio
            
        Returns:
            True if passes threshold
        """
        # Check disparate impact threshold (e.g., 80% rule)
        if disparate_impact < self.threshold:
            return False
        
        # Check absolute difference threshold if specified
        if self.absolute_threshold is not None:
            rates = list(group_rates.values())
            max_diff = max(rates) - min(rates)
            if max_diff > self.absolute_threshold:
                return False
        
        return True
    
    def _calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix for a group.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            
        Returns:
            2x2 confusion matrix
        """
        # Ensure binary
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        return np.array([[tn, fp], [fn, tp]])
    
    def get_selection_rates(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Get selection rates (positive prediction rates) for each group.
        
        Args:
            X: Input features
            sensitive_feature: Protected attribute
            y_pred: Predictions
            
        Returns:
            Dictionary of selection rates per group
        """
        predictions = self._get_predictions(X, y_pred)
        sensitive_feature = self._to_numpy(sensitive_feature)
        
        unique_groups = self._get_unique_groups(sensitive_feature)
        selection_rates = {}
        
        for group in unique_groups:
            group_mask = sensitive_feature == group
            group_pred = predictions[group_mask]
            
            if len(group_pred) > 0:
                selection_rates[str(group)] = float(np.mean(group_pred == 1))
            else:
                selection_rates[str(group)] = 0.0
        
        return selection_rates
    
    def suggest_mitigation(
        self,
        result: FairnessResult
    ) -> Dict[str, Any]:
        """
        Suggest mitigation strategies for detected bias.
        
        Args:
            result: FairnessResult object
            
        Returns:
            Dictionary with mitigation suggestions
        """
        if not result.bias_detected:
            return {
                'needed': False,
                'message': 'No mitigation needed - model is fair'
            }
        
        suggestions = {
            'needed': True,
            'strategies': []
        }
        
        # Threshold optimization
        if result.disparate_impact_ratio < self.threshold:
            suggestions['strategies'].append({
                'method': 'Threshold Optimization',
                'description': 'Adjust decision thresholds per group to equalize positive rates',
                'expected_impact': 'Can achieve perfect statistical parity',
                'tradeoff': 'May reduce overall accuracy'
            })
        
        # Reweighting
        if result.affected_groups:
            suggestions['strategies'].append({
                'method': 'Sample Reweighting',
                'description': f'Increase weights for underrepresented outcomes in groups: {", ".join(result.affected_groups)}',
                'expected_impact': 'Improves representation during training',
                'tradeoff': 'May overfit to minority cases'
            })
        
        # Data augmentation
        suggestions['strategies'].append({
            'method': 'Synthetic Data Generation',
            'description': 'Generate synthetic samples to balance group representations',
            'expected_impact': 'Increases diversity in training data',
            'tradeoff': 'Quality depends on generation method'
        })
        
        # Feature removal
        suggestions['strategies'].append({
            'method': 'Fairness-Aware Feature Selection',
            'description': 'Remove or transform features correlated with sensitive attributes',
            'expected_impact': 'Reduces indirect discrimination',
            'tradeoff': 'May reduce predictive performance'
        })
        
        return suggestions


# Register the Statistical Parity metric
FairnessMetricFactory.register("statistical_parity", StatisticalParity)
FairnessMetricFactory.register("demographic_parity", StatisticalParity)  # Alias


def calculate_statistical_parity(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    sensitive_feature: Union[pd.Series, np.ndarray],
    y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.8,
    **kwargs
) -> FairnessResult:
    """
    Convenience function to calculate statistical parity.
    
    Args:
        model: ML model
        X: Input features
        sensitive_feature: Protected attribute
        y_pred: Optional predictions
        threshold: Fairness threshold
        **kwargs: Additional parameters
        
    Returns:
        FairnessResult object
    """
    metric = StatisticalParity(model=model, threshold=threshold, **kwargs)
    # Use dummy y_true since statistical parity doesn't need it
    y_dummy = np.zeros(len(X))
    return metric.calculate(X, y_dummy, sensitive_feature, y_pred)