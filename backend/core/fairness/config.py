"""
Configuration classes for fairness optimization.

This module contains all configuration options for fairness optimization,
providing a clean interface for setting up different fairness strategies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FairnessConfig:
    """Configuration for fairness optimization.
    
    Attributes:
        objective: Fairness objective to optimize for
        mitigation: Fairness mitigation strategy
        constraints_eps: Fairness slack for reductions
        postprocess_predict_method: Method for postprocessing predictions
        postprocess_obj: Objective for postprocessing optimization
        fairness_weight: Weight for fairness vs accuracy trade-off
        custom_threshold: Custom decision threshold
        ensemble_size: Number of models for ensemble
        bootstrap_samples: Whether to use bootstrap sampling
        calibration: Whether to apply probability calibration
        hyperparameter_optimization: Enable automatic hyperparameter tuning
        multi_objective_optimization: Optimize for multiple objectives
        robustness_analysis: Perform robustness testing
        statistical_testing: Perform statistical significance tests
        confidence_intervals: Compute confidence intervals
        confidence_level: Confidence level for statistical tests
        n_bootstrap_samples: Number of bootstrap samples
        ensemble_method: Type of ensemble method
        ensemble_voting: Voting method for ensemble
        ensemble_config: Advanced ensemble configuration
        multi_objective_config: Multi-objective optimization configuration
        intersectional_analysis: Analyze intersectional fairness
        temporal_analysis: Analyze fairness over time
        individual_fairness: Measure individual fairness
    """
    objective: str = "equalized_odds"   # "equalized_odds" | "demographic_parity" | "equal_opportunity"
    mitigation: str = "reduction"       # "none" | "postprocess" | "reduction" | "ensemble" | "multi_objective"
    constraints_eps: float = 0.02       # fairness slack for reductions
    postprocess_predict_method: str = "auto"  # "auto"|"predict_proba"|"decision_function"
    postprocess_obj: str = "accuracy"   # "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"
    
    # Advanced options
    fairness_weight: float = 1.0        # Weight for fairness vs accuracy trade-off (0.1-10.0)
    custom_threshold: Optional[float] = None  # Custom decision threshold
    ensemble_size: int = 1              # Number of models for ensemble (if > 1)
    bootstrap_samples: bool = False     # Whether to use bootstrap sampling
    calibration: bool = False           # Whether to apply probability calibration
    
    # New advanced features
    hyperparameter_optimization: bool = False  # Enable automatic hyperparameter tuning
    multi_objective_optimization: bool = False # Optimize for multiple objectives simultaneously
    robustness_analysis: bool = False   # Perform robustness testing
    statistical_testing: bool = False   # Perform statistical significance tests
    confidence_intervals: bool = False  # Compute confidence intervals
    confidence_level: float = 0.95     # Confidence level for statistical tests
    n_bootstrap_samples: int = 1000    # Number of bootstrap samples for uncertainty estimation
    
    # Ensemble configurations  
    ensemble_method: str = "voting"     # "voting" | "bagging" | "boosting" | "stacking"
    ensemble_voting: str = "soft"       # "soft" | "hard" (for voting ensembles)
    ensemble_config: Optional[Dict[str, Any]] = None        # Advanced ensemble configuration
    
    # Multi-objective optimization
    multi_objective_config: Optional[Dict[str, Any]] = None # Multi-objective optimization configuration
    
    # Advanced fairness metrics
    intersectional_analysis: bool = False  # Analyze intersectional fairness
    temporal_analysis: bool = False     # Analyze fairness over time
    individual_fairness: bool = False   # Measure individual fairness
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Set default configurations if not provided
        if self.ensemble_config is None:
            self.ensemble_config = {
                'type': 'voting',
                'n_estimators': 5,
                'search_method': 'grid',
                'cv_folds': 5
            }
        
        if self.multi_objective_config is None:
            self.multi_objective_config = {
                'objectives': ['accuracy', 'fairness'],
                'weights': [0.7, 0.3]
            }
        
        if self.objective not in ["equalized_odds", "demographic_parity", "equal_opportunity"]:
            raise ValueError(f"Invalid objective: {self.objective}")
        
        if self.mitigation not in ["none", "postprocess", "reduction", "ensemble", "multi_objective"]:
            raise ValueError(f"Invalid mitigation strategy: {self.mitigation}")
            
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {self.confidence_level}")
            
        if self.n_bootstrap_samples < 100:
            raise ValueError(f"Bootstrap samples must be >= 100, got {self.n_bootstrap_samples}")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    type: str = "voting"  # "voting" | "bagging" | "boosting"
    n_estimators: int = 5
    search_method: str = "grid"  # "grid" | "random"
    cv_folds: int = 5
    voting: str = "soft"  # "soft" | "hard" (for voting ensembles)
    

@dataclass  
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: list = None
    weights: list = None
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['accuracy', 'fairness']
        if self.weights is None:
            self.weights = [0.7, 0.3]
            
        if len(self.objectives) != len(self.weights):
            raise ValueError("Objectives and weights must have same length")
            
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")