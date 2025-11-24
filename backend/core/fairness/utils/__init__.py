"""Fairness optimization utilities."""

from .helpers import safe_proba, validate_inputs, get_metric_function, format_results
from .hyperparameters import HyperparameterOptimizer

__all__ = [
    'safe_proba',
    'validate_inputs', 
    'get_metric_function',
    'format_results',
    'HyperparameterOptimizer'
]