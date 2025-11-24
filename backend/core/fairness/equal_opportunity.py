"""
Equal Opportunity Fairness Metric
Ensures equal true positive rates across different groups.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from .base_metrics import (
    BaseFairnessMetric, 
    FairnessResult, 
    FairnessMetricType,
    BiasDirection,
    FairnessMetricFactory
)

logger = logging.getLogger(__name__)


class EqualOpportunity(BaseFairnessMetric):
    """
    Equal Opportunity fairness metric.
    
    Checks if P(Y_hat=1|Y=1,A=a) = P(Y_hat=1|Y=1,A=b) for all groups a, b.
    In other words, the True Positive Rate (TPR) should be the same across groups.
    
    This ensures that qualified individuals have equal chance of positive outcomes
    regardless of their group membership.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        threshold: float = 0.8,  # 80% rule for TPR ratio
        absolute_threshold: Optional[float] = 0.1,  # Max 10% TPR difference
        check_equalized_odds: bool = False,  # Also check False Positive Rate
        **kwargs
    ):
        """
        Initialize Equal Opportunity metric.
        
        Args:
            model: ML model to evaluate
            threshold: Ratio threshold for TPR (default 0.8)
            absolute_threshold: Maximum absolute TPR difference allowed
            check_equalized_odds: If True, also checks FPR (becomes Equalized Odds)
            **kwargs: Additional parameters
        """
        super().__init__(model=model, threshold=threshold, **kwargs)
        self.absolute_threshold = absolute_threshold
        self.check_equalized_odds = check_equalized_odds
        self.metric_name = "Equal Opportunity" if not check_equalized_odds else "Equalized Odds"
    
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
        Calculate Equal Opportunity metric.
        
        Args:
            X: Input features
            y_true: True labels (REQUIRED for Equal Opportunity)
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
        
        # Ensure binary predictions
        if predictions.dtype == float and predictions.max() <= 1 and predictions.min() >= 0:
            # Probability predictions - convert to binary
            predictions = (predictions > 0.5).astype(int)
        
        # Ensure binary labels
        unique_labels = np.unique(y_true_np)
        if len(unique_labels) > 2:
            logger.warning(f"More than 2 classes found. Converting to binary (1 vs rest).")
            # Assume the positive class is 1 or the max value
            positive_label = 1 if 1 in unique_labels else max(unique_labels)
            y_true_np = (y_true_np == positive_label).astype(int)
        
        # Calculate group metrics
        group_metrics = self.calculate_group_metrics(
            y_true_np, predictions, sensitive_feature, sample_weight
        )
        
        # Focus on TPR for Equal Opportunity
        tpr_scores = group_metrics['true_positive_rates']
        
        # Calculate overall score based on TPR
        overall_score = self._calculate_overall_score(tpr_scores)
        
        # Calculate disparate impact for TPR
        disparate_impact = self._calculate_disparate_impact(tpr_scores)
        
        # Statistical significance test for TPR differences
        p_value, is_significant = self._test_tpr_equality(
            group_metrics['confusion_matrices'],
            group_metrics['group_sizes']
        )
        
        # Detect bias direction based on TPR
        bias_direction, affected_groups = self._detect_tpr_bias(
            tpr_scores,
            group_metrics.get('false_positive_rates', {})
        )
        
        # Check if passes threshold
        passes_threshold = self._check_threshold(tpr_scores, disparate_impact)
        
        # If checking equalized odds, also verify FPR
        if self.check_equalized_odds:
            fpr_scores = group_metrics.get('false_positive_rates', {})
            fpr_disparate_impact = self._calculate_disparate_impact(fpr_scores)
            fpr_passes = self._check_threshold(fpr_scores, fpr_disparate_impact)
            passes_threshold = passes_threshold and fpr_passes
            
            # Combine TPR and FPR for overall score
            fpr_score = self._calculate_overall_score(fpr_scores)
            overall_score = (overall_score + fpr_score) / 2
        
        # Create result
        result = FairnessResult(
            metric_name=self.metric_name,
            metric_type=FairnessMetricType.EQUAL_OPPORTUNITY if not self.check_equalized_odds 
                        else FairnessMetricType.EQUALIZED_ODDS,
            overall_score=overall_score,
            group_scores=tpr_scores,
            disparate_impact_ratio=disparate_impact,
            statistical_significance=p_value,
            bias_detected=is_significant and not passes_threshold,
            bias_direction=bias_direction if is_significant else BiasDirection.NO_BIAS,
            affected_groups=affected_groups if is_significant else [],
            threshold_used=self.threshold,
            passes_threshold=passes_threshold,
            sample_sizes=group_metrics['group_sizes'],
            raw_rates={
                'true_positive_rates': tpr_scores,
                'false_positive_rates': group_metrics.get('false_positive_rates', {}),
                'true_negative_rates': group_metrics.get('true_negative_rates', {}),
                'false_negative_rates': group_metrics.get('false_negative_rates', {})
            },
            confusion_matrices=group_metrics['confusion_matrices']
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
        Calculate TPR, FPR, TNR, FNR for each group.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            groups: Group labels
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary with rates and confusion matrices per group
        """
        unique_groups = self._get_unique_groups(groups)
        
        true_positive_rates = {}
        false_positive_rates = {}
        true_negative_rates = {}
        false_negative_rates = {}
        group_sizes = {}
        confusion_matrices = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # Calculate confusion matrix
            cm = self._calculate_confusion_matrix(group_true, group_pred)
            confusion_matrices[str(group)] = cm
            
            # Extract rates from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate rates with handling for zero denominators
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 1.0  # Specificity
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Miss Rate
            
            true_positive_rates[str(group)] = float(tpr)
            false_positive_rates[str(group)] = float(fpr)
            true_negative_rates[str(group)] = float(tnr)
            false_negative_rates[str(group)] = float(fnr)
            group_sizes[str(group)] = int(np.sum(group_mask))
        
        return {
            'true_positive_rates': true_positive_rates,
            'false_positive_rates': false_positive_rates,
            'true_negative_rates': true_negative_rates,
            'false_negative_rates': false_negative_rates,
            'group_sizes': group_sizes,
            'confusion_matrices': confusion_matrices,
            'unique_groups': [str(g) for g in unique_groups]
        }
    
    def _calculate_overall_score(self, group_rates: Dict[str, float]) -> float:
        """
        Calculate overall equal opportunity score.
        0 = perfect equality, higher = more inequality
        
        Args:
            group_rates: TPR (or other rates) per group
            
        Returns:
            Overall score
        """
        if len(group_rates) < 2:
            return 0.0
        
        # Filter out groups with undefined rates (e.g., no positive samples)
        valid_rates = [r for r in group_rates.values() if r > 0 or r == 0]
        
        if len(valid_rates) < 2:
            return 0.0
        
        # Maximum difference in rates
        max_rate = max(valid_rates)
        min_rate = min(valid_rates)
        max_diff = max_rate - min_rate
        
        # Standard deviation of rates
        std_dev = np.std(valid_rates)
        
        # Weighted score
        overall_score = max_diff * 0.6 + std_dev * 0.4
        
        return float(overall_score)
    
    def _test_tpr_equality(
        self,
        confusion_matrices: Dict[str, np.ndarray],
        group_sizes: Dict[str, int]
    ) -> Tuple[float, bool]:
        """
        Test statistical significance of TPR differences.
        
        Args:
            confusion_matrices: Confusion matrices per group
            group_sizes: Sample sizes per group
            
        Returns:
            Tuple of (p-value, is_significant)
        """
        if len(confusion_matrices) < 2:
            return 1.0, False
        
        # Extract TPRs and sample sizes for positive class
        tprs = []
        positive_samples = []
        
        for group, cm in confusion_matrices.items():
            tn, fp, fn, tp = cm.ravel()
            n_positive = tp + fn
            
            if n_positive > 0:
                tpr = tp / n_positive
                tprs.append(tpr)
                positive_samples.append(n_positive)
        
        if len(tprs) < 2:
            return 1.0, False
        
        # Perform chi-square test
        try:
            # Create contingency table
            successes = [int(tpr * n) for tpr, n in zip(tprs, positive_samples)]
            failures = [n - s for s, n in zip(successes, positive_samples)]
            
            contingency = np.array([successes, failures])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            
            is_significant = p_value < (1 - self.confidence_level)
            return p_value, is_significant
        except:
            return 1.0, False
    
    def _detect_tpr_bias(
        self,
        tpr_scores: Dict[str, float],
        fpr_scores: Dict[str, float]
    ) -> Tuple[BiasDirection, List[str]]:
        """
        Detect bias direction based on TPR (and optionally FPR).
        
        Args:
            tpr_scores: True positive rates per group
            fpr_scores: False positive rates per group
            
        Returns:
            Tuple of (bias_direction, affected_groups)
        """
        if len(tpr_scores) < 2:
            return BiasDirection.NO_BIAS, []
        
        # Analyze TPR disparities
        tpr_values = list(tpr_scores.values())
        mean_tpr = np.mean(tpr_values)
        std_tpr = np.std(tpr_values)
        
        disadvantaged = []
        advantaged = []
        
        for group, tpr in tpr_scores.items():
            # Group is disadvantaged if TPR is significantly below average
            if tpr < mean_tpr - std_tpr:
                disadvantaged.append(group)
            # Group is advantaged if TPR is significantly above average  
            elif tpr > mean_tpr + std_tpr:
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
    
    def _check_threshold(
        self,
        group_rates: Dict[str, float],
        disparate_impact: float
    ) -> bool:
        """
        Check if TPR differences pass fairness thresholds.
        
        Args:
            group_rates: TPR per group
            disparate_impact: Disparate impact ratio
            
        Returns:
            True if passes threshold
        """
        # Check ratio threshold
        if disparate_impact < self.threshold:
            return False
        
        # Check absolute difference threshold
        if self.absolute_threshold is not None:
            valid_rates = [r for r in group_rates.values() if r >= 0]
            if len(valid_rates) >= 2:
                max_diff = max(valid_rates) - min(valid_rates)
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
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        # Ensure binary
        y_true = (y_true == 1).astype(int)
        y_pred = (y_pred == 1).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[tn, fp], [fn, tp]])
    
    def get_error_rate_balance(
        self,
        result: FairnessResult
    ) -> Dict[str, Any]:
        """
        Analyze balance of error rates across groups.
        
        Args:
            result: FairnessResult object
            
        Returns:
            Dictionary with error rate analysis
        """
        if result.raw_rates is None:
            return {}
        
        tpr = result.raw_rates.get('true_positive_rates', {})
        fpr = result.raw_rates.get('false_positive_rates', {})
        
        analysis = {
            'tpr_range': max(tpr.values()) - min(tpr.values()) if tpr else 0,
            'fpr_range': max(fpr.values()) - min(fpr.values()) if fpr else 0,
            'balanced': False
        }
        
        # Check if error rates are balanced (both TPR and FPR similar across groups)
        if analysis['tpr_range'] < 0.1 and analysis['fpr_range'] < 0.1:
            analysis['balanced'] = True
            analysis['message'] = "Error rates are well-balanced across groups"
        else:
            analysis['message'] = "Significant imbalance in error rates detected"
        
        return analysis
    
    def suggest_threshold_adjustment(
        self,
        result: FairnessResult,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Suggest group-specific thresholds to achieve equal opportunity.
        
        Args:
            result: FairnessResult object
            X: Input features
            y_true: True labels
            sensitive_feature: Protected attribute
            
        Returns:
            Dictionary with suggested thresholds per group
        """
        if not result.bias_detected:
            return {'message': 'No threshold adjustment needed'}
        
        # Get probability predictions
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]  # Binary classification positive class
        else:
            return {'message': 'Model does not support probability predictions'}
        
        sensitive_feature = self._to_numpy(sensitive_feature)
        y_true = self._to_numpy(y_true)
        unique_groups = self._get_unique_groups(sensitive_feature)
        
        # Find optimal thresholds for each group
        optimal_thresholds = {}
        target_tpr = np.mean(list(result.raw_rates['true_positive_rates'].values()))
        
        for group in unique_groups:
            group_mask = sensitive_feature == group
            group_proba = y_proba[group_mask]
            group_true = y_true[group_mask]
            
            # Try different thresholds to find one that achieves target TPR
            best_threshold = 0.5
            best_diff = float('inf')
            
            for threshold in np.linspace(0.1, 0.9, 50):
                group_pred = (group_proba >= threshold).astype(int)
                
                # Calculate TPR for this threshold
                tp = np.sum((group_true == 1) & (group_pred == 1))
                fn = np.sum((group_true == 1) & (group_pred == 0))
                
                if (tp + fn) > 0:
                    tpr = tp / (tp + fn)
                    diff = abs(tpr - target_tpr)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = threshold
            
            optimal_thresholds[str(group)] = float(best_threshold)
        
        return optimal_thresholds


# Register the Equal Opportunity metric
FairnessMetricFactory.register("equal_opportunity", EqualOpportunity)
FairnessMetricFactory.register("equalized_odds", 
                               lambda **kwargs: EqualOpportunity(check_equalized_odds=True, **kwargs))


def calculate_equal_opportunity(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
    sensitive_feature: Union[pd.Series, np.ndarray],
    y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.8,
    **kwargs
) -> FairnessResult:
    """
    Convenience function to calculate equal opportunity.
    
    Args:
        model: ML model
        X: Input features
        y_true: True labels (required)
        sensitive_feature: Protected attribute
        y_pred: Optional predictions
        threshold: Fairness threshold
        **kwargs: Additional parameters
        
    Returns:
        FairnessResult object
    """
    metric = EqualOpportunity(model=model, threshold=threshold, **kwargs)
    return metric.calculate(X, y_true, sensitive_feature, y_pred)