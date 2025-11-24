"""
Fairness Module
Provides fairness metrics, bias detection, optimization, and visualization.
"""

from .base_metrics import (
    BaseFairnessMetric,
    FairnessResult,
    FairnessMetricType,
    BiasDirection,
    FairnessMetricFactory
)
from .statistical_parity import StatisticalParity, calculate_statistical_parity
from .equal_opportunity import EqualOpportunity, calculate_equal_opportunity
from .calibration import CalibrationMetric, calculate_calibration
from .bias_detector import BiasDetector, BiasSeverity, ComprehensiveBiasReport, create_bias_detector

# New advanced components
from .optimizer import FairnessOptimizer
from .visualizer import FairnessVisualizer

# Modular components
from .config import FairnessConfig, EnsembleConfig, MultiObjectiveConfig
from .utils import safe_proba, validate_inputs, get_metric_function, format_results, HyperparameterOptimizer
from .analysis import StatisticalAnalyzer
from .optimizers import BaseFairnessOptimizer, BaseEnsembleOptimizer

__all__ = [
    # Base classes
    'BaseFairnessMetric',
    'FairnessResult',
    'FairnessMetricType',
    'BiasDirection',
    'FairnessMetricFactory',
    
    # Metric implementations
    'StatisticalParity',
    'EqualOpportunity',
    'CalibrationMetric',
    'BiasDetector',
    
    # Advanced optimization and visualization
    'FairnessOptimizer',
    'FairnessConfig',
    'FairnessVisualizer',
    
    # Enums and data classes
    'BiasSeverity',
    'ComprehensiveBiasReport',
    
    # Modular components
    'EnsembleConfig',
    'MultiObjectiveConfig',
    'safe_proba',
    'validate_inputs', 
    'get_metric_function',
    'format_results',
    'HyperparameterOptimizer',
    'StatisticalAnalyzer',
    'BaseFairnessOptimizer',
    'BaseEnsembleOptimizer',
    
    # Utility functions
    'calculate_statistical_parity',
    'calculate_equal_opportunity',
    'calculate_calibration',
    'create_bias_detector'
]