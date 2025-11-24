"""
Statistical analysis utilities for fairness testing.

This module provides statistical tests and confidence interval computation
for evaluating fairness and bias in machine learning models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple
from sklearn.utils import resample


class StatisticalAnalyzer:
    """Statistical analysis for fairness metrics."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals and tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def perform_fairness_tests(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              sensitive_features: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical significance tests for fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive feature values
            
        Returns:
            Dictionary of test results
        """
        tests = {}
        groups = np.unique(sensitive_features)
        
        if len(groups) == 2:
            tests.update(self._binary_group_tests(y_true, y_pred, sensitive_features, groups))
        else:
            tests.update(self._multi_group_tests(y_true, y_pred, sensitive_features, groups))
            
        return tests
    
    def _binary_group_tests(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           sensitive_features: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests for binary group comparison."""
        tests = {}
        group1, group2 = groups
        mask1 = sensitive_features == group1
        mask2 = sensitive_features == group2
        
        # Chi-square test for independence
        try:
            contingency_table = pd.crosstab(sensitive_features, y_pred)
            chi2, p_chi2 = stats.chi2_contingency(contingency_table)[:2]
            tests['chi2_test'] = {
                'statistic': float(chi2), 
                'p_value': float(p_chi2),
                'significant': p_chi2 < self.alpha
            }
        except Exception:
            tests['chi2_test'] = {'error': 'Could not compute chi-square test'}
        
        # Two-proportion z-test for selection rates
        try:
            n1, n2 = mask1.sum(), mask2.sum()
            p1 = y_pred[mask1].mean() if n1 > 0 else 0
            p2 = y_pred[mask2].mean() if n2 > 0 else 0
            
            if n1 > 0 and n2 > 0 and p1 != p2:
                pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                z_stat = (p1 - p2) / se if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                tests['proportion_test'] = {
                    'statistic': float(z_stat), 
                    'p_value': float(p_val),
                    'significant': p_val < self.alpha
                }
        except Exception:
            tests['proportion_test'] = {'error': 'Could not compute proportion test'}
            
        # Mann-Whitney U test for distribution comparison
        try:
            if len(y_pred[mask1]) > 0 and len(y_pred[mask2]) > 0:
                u_stat, p_u = stats.mannwhitneyu(
                    y_pred[mask1], y_pred[mask2], alternative='two-sided'
                )
                tests['mannwhitney_test'] = {
                    'statistic': float(u_stat), 
                    'p_value': float(p_u),
                    'significant': p_u < self.alpha
                }
        except Exception:
            tests['mannwhitney_test'] = {'error': 'Could not compute Mann-Whitney test'}
            
        return tests
    
    def _multi_group_tests(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          sensitive_features: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests for multiple group comparison."""
        tests = {}
        
        # ANOVA for multi-group comparison
        try:
            group_metrics = []
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    group_metrics.append(y_pred[mask])
            
            if len(group_metrics) > 2:
                f_stat, p_anova = stats.f_oneway(*group_metrics)
                tests['anova_test'] = {
                    'statistic': float(f_stat), 
                    'p_value': float(p_anova),
                    'significant': p_anova < self.alpha
                }
        except Exception:
            tests['anova_test'] = {'error': 'Could not compute ANOVA test'}
            
        return tests
    
    def compute_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   sensitive_features: np.ndarray, 
                                   n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals for fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            sensitive_features: Sensitive feature values
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of confidence intervals
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        def bootstrap_metric(y_t, y_p, metric_fn, n_samples):
            """Bootstrap a metric function."""
            scores = []
            for _ in range(n_samples):
                indices = resample(range(len(y_t)), random_state=None)
                y_boot_true = y_t[indices]
                y_boot_pred = y_p[indices] 
                try:
                    score = metric_fn(y_boot_true, y_boot_pred)
                    scores.append(score)
                except:
                    continue
            return np.array(scores)
        
        confidence_intervals = {}
        
        # Overall performance intervals
        overall_intervals = {}
        for metric_name, metric_fn in [
            ('accuracy', accuracy_score),
            ('precision', lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0)),
            ('recall', lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0))
        ]:
            scores = bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap)
            if len(scores) > 0:
                lower = np.percentile(scores, 100 * self.alpha/2)
                upper = np.percentile(scores, 100 * (1 - self.alpha/2))
                overall_intervals[metric_name] = (float(lower), float(upper))
        
        confidence_intervals['overall'] = overall_intervals
        
        # Group-wise intervals
        groups = np.unique(sensitive_features)
        by_group_intervals = {}
        
        for group in groups:
            mask = sensitive_features == group
            if mask.sum() > 10:  # Need sufficient samples
                group_intervals = {}
                for metric_name, metric_fn in [
                    ('accuracy', accuracy_score),
                    ('precision', lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0)),
                    ('recall', lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0))
                ]:
                    scores = bootstrap_metric(y_true[mask], y_pred[mask], metric_fn, n_bootstrap)
                    if len(scores) > 0:
                        lower = np.percentile(scores, 100 * self.alpha/2)
                        upper = np.percentile(scores, 100 * (1 - self.alpha/2))
                        group_intervals[metric_name] = (float(lower), float(upper))
                
                by_group_intervals[str(group)] = group_intervals
        
        confidence_intervals['by_group'] = by_group_intervals
        
        return confidence_intervals
    
    def test_bias_significance(self, fairness_metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Test whether observed bias is statistically significant.
        
        Args:
            fairness_metrics: Dictionary of fairness disparity metrics
            
        Returns:
            Dictionary indicating which metrics show significant bias
        """
        significant_bias = {}
        
        # Simple threshold-based significance test
        # In practice, this would use more sophisticated statistical tests
        threshold = 0.05  # 5% disparity threshold
        
        for metric_name, disparity in fairness_metrics.items():
            if metric_name.endswith('_disparity'):
                significant_bias[metric_name] = abs(disparity) > threshold
                
        return significant_bias