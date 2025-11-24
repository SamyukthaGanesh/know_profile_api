"""
Hyperparameter optimization utilities for fairness-aware models.

This module provides functionality for optimizing hyperparameters
of machine learning models while considering fairness constraints.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


class HyperparameterOptimizer:
    """Hyperparameter optimization for fairness-aware models."""
    
    def __init__(self, search_method='grid', cv_folds=5, scoring='accuracy', n_jobs=-1):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            search_method: 'grid' or 'random' search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
        """
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
    
    def optimize(self, estimator, X, y, sensitive_feature_names=None):
        """
        Optimize hyperparameters for given estimator.
        
        Args:
            estimator: Base estimator to optimize
            X: Training features
            y: Training labels
            sensitive_feature_names: Names of sensitive features to exclude
            
        Returns:
            Optimized estimator
        """
        # Remove sensitive features from training data
        if sensitive_feature_names:
            X_train = X.drop(columns=sensitive_feature_names, errors="ignore")
        else:
            X_train = X
            
        # Get parameter grid for the estimator
        param_grid = self.get_parameter_grid(estimator)
        
        if not param_grid:
            return estimator
            
        # Create search object
        if self.search_method == 'random':
            search = RandomizedSearchCV(
                estimator, param_grid, cv=self.cv_folds, 
                scoring=self.scoring, n_jobs=self.n_jobs, random_state=42
            )
        else:
            search = GridSearchCV(
                estimator, param_grid, cv=self.cv_folds,
                scoring=self.scoring, n_jobs=self.n_jobs
            )
            
        search.fit(X_train, y)
        return search.best_estimator_
    
    def get_parameter_grid(self, estimator):
        """
        Get appropriate parameter grid for hyperparameter optimization.
        
        Args:
            estimator: Estimator to get parameter grid for
            
        Returns:
            Dictionary of parameter ranges
        """
        if isinstance(estimator, RandomForestClassifier):
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(estimator, GradientBoostingClassifier):
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif isinstance(estimator, LogisticRegression):
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
                'max_iter': [1000]
            }
        elif isinstance(estimator, SVC):
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            # For unknown estimators, return empty grid
            return {}
    
    def get_fairness_aware_grid(self, estimator, fairness_constraint='equalized_odds'):
        """
        Get parameter grid optimized for fairness-accuracy trade-offs.
        
        Args:
            estimator: Estimator to get parameter grid for
            fairness_constraint: Type of fairness constraint
            
        Returns:
            Dictionary of parameter ranges optimized for fairness
        """
        base_grid = self.get_parameter_grid(estimator)
        
        # Adjust parameters for better fairness-accuracy trade-offs
        if isinstance(estimator, RandomForestClassifier):
            # Slightly reduce complexity for better fairness
            base_grid['max_depth'] = [3, 5, 8]
            base_grid['min_samples_leaf'] = [2, 5, 10]
            
        elif isinstance(estimator, LogisticRegression):
            # Add stronger regularization options for fairness
            base_grid['C'] = [0.01, 0.1, 1.0, 10.0]
            
        return base_grid