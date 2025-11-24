"""
Helper utility functions for fairness optimization.

This module contains common utility functions used across different 
fairness optimization strategies.
"""

import numpy as np
from typing import Union, Callable


def safe_proba(estimator, X) -> np.ndarray:
    """
    Safely extract prediction probabilities from an estimator.
    
    Args:
        estimator: Trained sklearn estimator
        X: Input features
        
    Returns:
        Array of prediction probabilities for positive class
    """
    # Try predict_proba, then decision_function -> convert to [0,1]
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        # Min-max to 0..1 for calibration-agnostic thresholding
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    # Fallback to predictions as 0/1
    return estimator.predict(X).astype(float)


def validate_inputs(X, y, sensitive_features):
    """
    Validate input data for fairness optimization.
    
    Args:
        X: Feature matrix
        y: Target labels
        sensitive_features: Sensitive feature values
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    if len(X) != len(sensitive_features):
        raise ValueError(f"X and sensitive_features must have same length: {len(X)} vs {len(sensitive_features)}")
    
    if len(np.unique(y)) != 2:
        raise ValueError(f"Only binary classification supported, found {len(np.unique(y))} classes")


def get_metric_function(metric_name: str) -> Callable:
    """
    Get sklearn metric function by name.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric function
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, balanced_accuracy_score
    )
    
    metric_map = {
        'accuracy': accuracy_score,
        'precision': precision_score, 
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score,
        'balanced_accuracy': balanced_accuracy_score
    }
    
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return metric_map[metric_name]


def format_results(results: dict) -> dict:
    """
    Format results dictionary for consistent output.
    
    Args:
        results: Raw results dictionary
        
    Returns:
        Formatted results dictionary
    """
    formatted = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            formatted[key] = format_results(value)
        elif isinstance(value, (int, float)):
            formatted[key] = round(float(value), 4)
        else:
            formatted[key] = value
            
    return formatted