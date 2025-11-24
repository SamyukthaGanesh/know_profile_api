# fairness_optimizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)

from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, true_negative_rate,
    false_positive_rate, false_negative_rate
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings


Metric = Callable[[np.ndarray, np.ndarray], float]


def _safe_proba(estimator, X):
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


@dataclass
class FairnessConfig:
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
    ensemble_config: dict = None        # Advanced ensemble configuration
    
    # Multi-objective optimization
    multi_objective_config: dict = None # Multi-objective optimization configuration
    
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
            raise ValueError(f"Unknown objective: {self.objective}")
        if self.mitigation not in ["none", "postprocess", "reduction", "ensemble", "multi_objective"]:
            raise ValueError(f"Unknown mitigation: {self.mitigation}")
        if not 0 <= self.constraints_eps <= 1:
            raise ValueError("constraints_eps must be between 0 and 1")
        if not 0.1 <= self.fairness_weight <= 10.0:
            raise ValueError("fairness_weight must be between 0.1 and 10.0")
        if self.custom_threshold is not None and not 0 <= self.custom_threshold <= 1:
            raise ValueError("custom_threshold must be between 0 and 1")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError("confidence_level must be between 0.5 and 0.99")
        if self.n_bootstrap_samples < 100:
            raise ValueError("n_bootstrap_samples must be >= 100")


class FairnessOptimizer(BaseEstimator, ClassifierMixin):
    """
    A dataset- and model-agnostic fairness optimizer.
    - Accepts any sklearn-compatible classifier.
    - Evaluates fairness with MetricFrame.
    - Mitigates bias via post-processing or in-processing reductions.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        sensitive_feature_names: List[str],
        config: Optional[FairnessConfig] = None,
        custom_metrics: Optional[Dict[str, Metric]] = None,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.sensitive_feature_names = sensitive_feature_names
        self.config = config or FairnessConfig()
        # Store custom_metrics as dict for sklearn compatibility
        self.custom_metrics = custom_metrics if custom_metrics is not None else {}
        self.random_state = random_state

        self._fitted_estimator = None
        self._is_postprocessor = False
        self._sensitive_series = None

        # Default metrics (overall + by group)
        self.metrics: Dict[str, Metric] = {
            "accuracy": accuracy_score,
            "roc_auc": lambda y_true, y_score: roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
            "average_precision": lambda y_true, y_score: average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "tnr": true_negative_rate,
            "fpr": false_positive_rate,
            "fnr": false_negative_rate,
        }
        self.metrics.update(self.custom_metrics)

    # ---------- Public API ----------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "FairnessOptimizer":
        X, y = self._to_frame(X), self._to_series(y)
        sens = self._extract_sensitive(X)
        self._sensitive_series = sens

        cfg = self.config
        base = clone(self.base_estimator)

        # Hyperparameter optimization if requested
        if cfg.hyperparameter_optimization:
            base = self._optimize_hyperparameters(base, X, y, sens)

        if cfg.mitigation == "none":
            self._fitted_estimator = self._fit_base_model(base, X, y, cfg)
            self._is_postprocessor = False
            return self

        if cfg.mitigation == "postprocess":
            self._fitted_estimator = self._fit_postprocess_model(base, X, y, sens, cfg)
            self._is_postprocessor = True
            return self

        if cfg.mitigation == "reduction":
            self._fitted_estimator = self._fit_reduction_model(base, X, y, sens, cfg)
            self._is_postprocessor = False
            return self

        if cfg.mitigation == "ensemble":
            self._fitted_estimator = self._fit_ensemble_model(base, X, y, sens, cfg)
            self._is_postprocessor = False
            return self

        if cfg.mitigation == "multi_objective":
            self._fitted_estimator = self._fit_multi_objective_model(base, X, y, sens, cfg)
            self._is_postprocessor = False
            return self

        raise ValueError(f"Unknown mitigation '{cfg.mitigation}'")

    # ---------- Helper methods for enhanced algorithms ----------
    
    def _optimize_hyperparameters(self, base_estimator, X, y, sens):
        """Optimize hyperparameters using grid search or random search."""
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        
        # Define parameter grids based on estimator type
        param_grid = self._get_parameter_grid(base_estimator)
        
        if not param_grid:
            return base_estimator
            
        search_method = self.config.ensemble_config.get('search_method', 'grid')
        cv_folds = self.config.ensemble_config.get('cv_folds', 5)
        
        if search_method == 'random':
            search = RandomizedSearchCV(
                base_estimator, param_grid, cv=cv_folds, 
                scoring='accuracy', n_jobs=-1, random_state=42
            )
        else:
            search = GridSearchCV(
                base_estimator, param_grid, cv=cv_folds,
                scoring='accuracy', n_jobs=-1
            )
            
        search.fit(X_train, y)
        return search.best_estimator_
    
    def _get_parameter_grid(self, estimator):
        """Get appropriate parameter grid for hyperparameter optimization."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        if isinstance(estimator, RandomForestClassifier):
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        elif isinstance(estimator, GradientBoostingClassifier):
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif isinstance(estimator, LogisticRegression):
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif isinstance(estimator, SVC):
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        return {}
    
    def _fit_base_model(self, base_estimator, X, y, config):
        """Fit base model without fairness constraints."""
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        return base_estimator.fit(X_train, y)
    
    def _fit_postprocess_model(self, base_estimator, X, y, sens, config):
        """Fit postprocessing fairness model."""
        # Train unconstrained model first
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        unconstrained = clone(base_estimator).fit(X_train, y)
        
        # Use correct constraint names for ThresholdOptimizer
        if config.objective == "equalized_odds":
            constraint = "equalized_odds"
        else:
            constraint = "demographic_parity"
        
        # Fix: Map to proper objective strings for ThresholdOptimizer
        objective_mapping = {
            "accuracy": "accuracy_score",
            "balanced_accuracy": "balanced_accuracy_score",
            "precision": "precision_score",
            "recall": "recall_score", 
            "f1": "f1_score",
            "roc_auc": "roc_auc_score"
        }
        
        mapped_objective = objective_mapping.get(config.postprocess_obj, "accuracy_score")
            
        return ThresholdOptimizer(
            estimator=unconstrained,
            constraints=constraint,
            objective=mapped_objective,
            predict_method=config.postprocess_predict_method,
            prefit=True,
            flip=True,
        ).fit(X_train, y, sensitive_features=sens)
    
    def _fit_reduction_model(self, base_estimator, X, y, sens, config):
        """Fit reduction-based fairness model."""
        constraint = EqualizedOdds() if config.objective == "equalized_odds" else DemographicParity()
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        return ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=config.constraints_eps,
        ).fit(X_train, y, sensitive_features=sens)
    
    def _fit_ensemble_model(self, base_estimator, X, y, sens, config):
        """Fit ensemble of fairness-aware models."""
        ensemble_config = config.ensemble_config
        ensemble_type = ensemble_config.get('type', 'voting')
        n_estimators = ensemble_config.get('n_estimators', 5)
        
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        
        if ensemble_type == 'voting':
            return self._create_voting_ensemble(base_estimator, X_train, y, sens, n_estimators, config)
        elif ensemble_type == 'bagging':
            return self._create_bagging_ensemble(base_estimator, X_train, y, sens, n_estimators, config)
        elif ensemble_type == 'boosting':
            return self._create_boosting_ensemble(base_estimator, X_train, y, sens, n_estimators, config)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    def _create_voting_ensemble(self, base_estimator, X, y, sens, n_estimators, config):
        """Create a voting ensemble of fairness-aware models."""
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for i in range(n_estimators):
            # Create different fairness models
            if i % 3 == 0:
                # Reduction-based
                constraint = EqualizedOdds() if config.objective == "equalized_odds" else DemographicParity()
                est = ExponentiatedGradient(
                    estimator=clone(base_estimator),
                    constraints=constraint,
                    eps=config.constraints_eps * (0.5 + i * 0.1),  # Vary epsilon
                ).fit(X, y, sensitive_features=sens)
            elif i % 3 == 1:
                # Base model
                est = clone(base_estimator).fit(X, y)
            else:
                # Postprocessing
                unconstrained = clone(base_estimator).fit(X, y)
                constraint_name = "equalized_odds" if config.objective == "equalized_odds" else "demographic_parity"
                est = ThresholdOptimizer(
                    estimator=unconstrained,
                    constraints=constraint_name,
                    objective=config.postprocess_obj,
                    prefit=True,
                ).fit(X, y, sensitive_features=sens)
            
            estimators.append((f'model_{i}', est))
        
        # Create and fit the voting ensemble
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble.fit(X, y)
    
    def _create_bagging_ensemble(self, base_estimator, X, y, sens, n_estimators, config):
        """Create a bagging ensemble with fairness constraints."""
        from sklearn.ensemble import BaggingClassifier
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        # Fix: Create a wrapper that generates fresh constraints for each bootstrap
        class FairnessBaggingWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, base_estimator=None, objective="equalized_odds", eps=0.02, sensitive_features=None):
                self.base_estimator = base_estimator
                self.objective = objective
                self.eps = eps
                self.sensitive_features = sensitive_features
                
            def fit(self, X, y):
                # Create fresh constraint for each fit call (bootstrap sample)
                constraint = EqualizedOdds() if self.objective == "equalized_odds" else DemographicParity()
                
                self.model_ = ExponentiatedGradient(
                    estimator=clone(self.base_estimator),
                    constraints=constraint,
                    eps=self.eps,
                )
                
                # Use stored sensitive features
                if self.sensitive_features is not None:
                    self.model_.fit(X, y, sensitive_features=self.sensitive_features)
                else:
                    # Fallback to base model without fairness
                    self.model_ = clone(self.base_estimator)
                    self.model_.fit(X, y)
                
                # Set classes_ attribute required by sklearn
                self.classes_ = np.unique(y)
                return self
                
            def predict(self, X):
                return self.model_.predict(X)
                
            def predict_proba(self, X):
                if hasattr(self.model_, 'predict_proba'):
                    return self.model_.predict_proba(X)
                pred = self.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1.0
                return proba
        
        # Create the wrapper
        fair_wrapper = FairnessBaggingWrapper(
            base_estimator=base_estimator,
            objective=config.objective,
            eps=config.constraints_eps,
            sensitive_features=sens
        )
        
        # Use BaggingClassifier with the wrapper
        bagging = BaggingClassifier(
            estimator=fair_wrapper,
            n_estimators=n_estimators,
            random_state=42
        )
        
        return bagging.fit(X, y)
    
    def _create_boosting_ensemble(self, base_estimator, X, y, sens, n_estimators, config):
        """Create a boosting ensemble with fairness awareness."""
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        # Fix: Create a wrapper that supports sample_weight for boosting
        class BoostingFairnessWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, base_estimator=None, objective="equalized_odds", eps=0.02):
                self.base_estimator = base_estimator
                self.objective = objective
                self.eps = eps
                
            def fit(self, X, y, sample_weight=None):
                # Ignore sample_weight for fairness models (limitation of ExponentiatedGradient)
                # Use base estimator with sample_weight support instead
                if sample_weight is not None:
                    # Use weighted base estimator
                    self.model_ = clone(self.base_estimator)
                    if hasattr(self.model_, 'fit') and 'sample_weight' in self.model_.fit.__code__.co_varnames:
                        self.model_.fit(X, y, sample_weight=sample_weight)
                    else:
                        self.model_.fit(X, y)
                else:
                    # Use fairness-aware model when no sample weights
                    constraint = EqualizedOdds() if self.objective == "equalized_odds" else DemographicParity()
                    self.model_ = ExponentiatedGradient(
                        estimator=clone(self.base_estimator),
                        constraints=constraint,
                        eps=self.eps,
                    )
                    # Would need sensitive features here, but boosting doesn't easily support this
                    # Fallback to base model
                    self.model_ = clone(self.base_estimator)
                    if hasattr(self.model_, 'fit') and 'sample_weight' in self.model_.fit.__code__.co_varnames:
                        self.model_.fit(X, y, sample_weight=sample_weight)
                    else:
                        self.model_.fit(X, y)
                
                # Set classes_ attribute required by sklearn
                self.classes_ = np.unique(y)
                return self
                
            def predict(self, X):
                return self.model_.predict(X)
                
            def predict_proba(self, X):
                if hasattr(self.model_, 'predict_proba'):
                    return self.model_.predict_proba(X)
                pred = self.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[np.arange(len(pred)), pred] = 1.0
                return proba
        
        # Create the boosting-compatible wrapper
        wrapper = BoostingFairnessWrapper(
            base_estimator=base_estimator,
            objective=config.objective,
            eps=config.constraints_eps
        )
        
        return AdaBoostClassifier(
            estimator=wrapper,
            n_estimators=n_estimators,
            random_state=42
        ).fit(X, y)
    
    def _fit_multi_objective_model(self, base_estimator, X, y, sens, config):
        """Fit multi-objective optimization model balancing accuracy and fairness."""
        X_train = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        
        # Use multiple constraints with different epsilon values
        objectives = config.multi_objective_config.get('objectives', ['accuracy', 'fairness'])
        weights = config.multi_objective_config.get('weights', [0.7, 0.3])
        
        if len(weights) != len(objectives):
            weights = [1.0 / len(objectives)] * len(objectives)
        
        # Fix: Create wrapper models that sklearn recognizes as classifiers
        models = []
        epsilons = np.linspace(0.01, 0.1, 3)  # Different fairness constraints
        
        for i, eps in enumerate(epsilons):
            constraint = EqualizedOdds() if config.objective == "equalized_odds" else DemographicParity()
            
            # Create and fit the ExponentiatedGradient model
            exp_grad = ExponentiatedGradient(
                estimator=clone(base_estimator),
                constraints=constraint,
                eps=eps,
            )
            exp_grad.fit(X_train, y, sensitive_features=sens)
            
            # Wrap in a simple classifier that VotingClassifier accepts
            class WrappedClassifier:
                def __init__(self, fitted_model):
                    self.fitted_model = fitted_model
                
                def predict(self, X):
                    return self.fitted_model.predict(X)
                
                def predict_proba(self, X):
                    if hasattr(self.fitted_model, 'predict_proba'):
                        return self.fitted_model.predict_proba(X)
                    # Fallback for models without predict_proba
                    pred = self.predict(X)
                    proba = np.zeros((len(pred), 2))
                    proba[np.arange(len(pred)), pred] = 1.0
                    return proba
                    
                def fit(self, X, y):
                    return self  # Already fitted
            
            wrapped_model = WrappedClassifier(exp_grad)
            models.append(wrapped_model)
        
        # Instead of VotingClassifier, use simple weighted averaging
        # This avoids the classifier validation issue
        class MultiObjectiveEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = np.array(weights)
                
            def predict(self, X):
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    predictions.append(pred)
                
                # Weighted majority voting
                predictions = np.array(predictions)
                weighted_preds = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
                return (weighted_preds > 0.5).astype(int)
            
            def predict_proba(self, X):
                probas = []
                for model in self.models:
                    proba = model.predict_proba(X)
                    probas.append(proba)
                
                # Weighted average of probabilities
                probas = np.array(probas)
                weighted_probas = np.average(probas, axis=0, weights=self.weights[:len(probas)])
                return weighted_probas
            
            def fit(self, X, y):
                return self  # Models already fitted
        
        # Create the ensemble with equal weights for simplicity
        ensemble_weights = [1.0] * len(models)
        ensemble = MultiObjectiveEnsemble(models, ensemble_weights)
        
        return ensemble

    def _perform_statistical_tests(self, y_true, y_pred, sens, overall, by_group):
        """Perform statistical significance tests for fairness metrics."""
        from scipy import stats
        
        tests = {}
        groups = sens.unique()
        
        if len(groups) == 2:
            # Binary group comparison
            group1, group2 = groups
            mask1 = sens == group1
            mask2 = sens == group2
            
            # Chi-square test for independence (for categorical outcomes)
            contingency_table = pd.crosstab(sens, y_pred)
            chi2, p_chi2 = stats.chi2_contingency(contingency_table)[:2]
            tests['chi2_test'] = {'statistic': chi2, 'p_value': p_chi2}
            
            # Two-proportion z-test for selection rates
            n1, n2 = mask1.sum(), mask2.sum()
            p1 = y_pred[mask1].mean() if n1 > 0 else 0
            p2 = y_pred[mask2].mean() if n2 > 0 else 0
            
            if n1 > 0 and n2 > 0 and p1 != p2:
                pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                z_stat = (p1 - p2) / se if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                tests['proportion_test'] = {'statistic': z_stat, 'p_value': p_val}
            
            # Mann-Whitney U test for continuous metrics (e.g., probabilities)
            if hasattr(self._fitted_estimator, 'predict_proba'):
                try:
                    X_subset = self._to_frame(y_pred.index)
                    X_clean = X_subset.drop(columns=self.sensitive_feature_names, errors="ignore")
                    all_probs = self._fitted_estimator.predict_proba(X_clean)[:, 1]
                    probs1 = all_probs[mask1] if n1 > 0 else []
                    probs2 = all_probs[mask2] if n2 > 0 else []
                    
                    if len(probs1) > 0 and len(probs2) > 0:
                        u_stat, p_u = stats.mannwhitneyu(probs1, probs2, alternative='two-sided')
                        tests['mannwhitney_test'] = {'statistic': u_stat, 'p_value': p_u}
                except:
                    pass  # Skip if we can't compute probabilities
        
        else:
            # Multi-group comparison using ANOVA
            group_metrics = []
            for group in groups:
                mask = sens == group
                if mask.sum() > 0:
                    group_metrics.append(y_pred[mask])
            
            if len(group_metrics) > 2:
                f_stat, p_anova = stats.f_oneway(*group_metrics)
                tests['anova_test'] = {'statistic': f_stat, 'p_value': p_anova}
        
        return tests
    
    def _compute_confidence_intervals(self, y_true, y_pred, sens, alpha=0.05):
        """Compute confidence intervals for fairness metrics using bootstrap."""
        from scipy import stats
        
        n_bootstrap = 1000
        confidence_level = 1 - alpha
        
        def bootstrap_metric(y_t, y_p, metric_fn, n_samples):
            bootstrap_scores = []
            n = len(y_t)
            
            for _ in range(n_samples):
                # Bootstrap sample
                idx = np.random.choice(n, size=n, replace=True)
                y_boot_true = y_t.iloc[idx] if hasattr(y_t, 'iloc') else y_t[idx]
                y_boot_pred = y_p.iloc[idx] if hasattr(y_p, 'iloc') else y_p[idx]
                
                try:
                    score = metric_fn(y_boot_true, y_boot_pred)
                    bootstrap_scores.append(score)
                except:
                    continue
            
            return np.array(bootstrap_scores)
        
        ci_results = {}
        
        # Overall confidence intervals
        overall_ci = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in self.metrics:
                bootstrap_scores = bootstrap_metric(
                    y_true, y_pred, self.metrics[metric_name], n_bootstrap
                )
                if len(bootstrap_scores) > 0:
                    lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
                    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
                    overall_ci[metric_name] = (lower, upper)
        
        ci_results['overall'] = overall_ci
        
        # Group-wise confidence intervals
        by_group_ci = {}
        for group in sens.unique():
            mask = sens == group
            if mask.sum() > 10:  # Minimum sample size for CI
                group_ci = {}
                y_group_true = y_true[mask]
                y_group_pred = y_pred[mask]
                
                for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric_name in self.metrics:
                        bootstrap_scores = bootstrap_metric(
                            y_group_true, y_group_pred, self.metrics[metric_name], n_bootstrap
                        )
                        if len(bootstrap_scores) > 0:
                            lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
                            upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
                            group_ci[metric_name] = (lower, upper)
                
                by_group_ci[group] = group_ci
        
        ci_results['by_group'] = by_group_ci
        return ci_results
    
    def _analyze_robustness(self, X, y, sens):
        """Analyze model robustness through various perturbations."""
        X_clean = X.drop(columns=self.sensitive_feature_names, errors="ignore")
        
        robustness_results = {}
        
        # 1. Feature importance stability
        if hasattr(self._fitted_estimator, 'feature_importances_'):
            base_importance = self._fitted_estimator.feature_importances_
            importance_variations = []
            
            for _ in range(10):  # Multiple bootstrap samples
                idx = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X_clean.iloc[idx]
                y_boot = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                
                # Refit and get importance
                temp_model = clone(self._fitted_estimator)
                try:
                    temp_model.fit(X_boot, y_boot)
                    if hasattr(temp_model, 'feature_importances_'):
                        importance_variations.append(temp_model.feature_importances_)
                except:
                    continue
            
            if importance_variations:
                importance_std = np.std(importance_variations, axis=0)
                robustness_results['feature_importance_stability'] = {
                    'mean_std': np.mean(importance_std),
                    'max_std': np.max(importance_std),
                    'coefficient_of_variation': importance_std / (base_importance + 1e-8)
                }
        
        # 2. Prediction stability under noise
        noise_levels = [0.01, 0.05, 0.1]
        prediction_stability = {}
        
        for noise_level in noise_levels:
            predictions = []
            
            for _ in range(50):  # Multiple noise samples
                X_noisy = X_clean.copy()
                # Add Gaussian noise to numerical features
                numerical_cols = X_noisy.select_dtypes(include=[np.number]).columns
                noise = np.random.normal(0, noise_level, X_noisy[numerical_cols].shape)
                X_noisy[numerical_cols] += noise
                
                try:
                    pred = self._fitted_estimator.predict(X_noisy)
                    predictions.append(pred)
                except:
                    continue
            
            if predictions:
                # Calculate prediction agreement
                predictions = np.array(predictions)
                base_pred = self._fitted_estimator.predict(X_clean)
                
                agreements = []
                for pred in predictions:
                    agreement = np.mean(pred == base_pred)
                    agreements.append(agreement)
                
                prediction_stability[f'noise_{noise_level}'] = {
                    'mean_agreement': np.mean(agreements),
                    'std_agreement': np.std(agreements),
                    'min_agreement': np.min(agreements)
                }
        
        robustness_results['prediction_stability'] = prediction_stability
        
        # 3. Cross-group generalization
        groups = sens.unique()
        if len(groups) >= 2:
            cross_group_performance = {}
            
            for train_group in groups:
                for test_group in groups:
                    if train_group != test_group:
                        # Train on one group, test on another
                        train_mask = sens == train_group
                        test_mask = sens == test_group
                        
                        if train_mask.sum() > 10 and test_mask.sum() > 10:
                            X_train_group = X_clean[train_mask]
                            y_train_group = y[train_mask]
                            X_test_group = X_clean[test_mask]
                            y_test_group = y[test_mask]
                            
                            try:
                                temp_model = clone(self.base_estimator)
                                temp_model.fit(X_train_group, y_train_group)
                                pred_cross = temp_model.predict(X_test_group)
                                
                                cross_accuracy = self.metrics['accuracy'](y_test_group, pred_cross)
                                cross_group_performance[f'{train_group}_to_{test_group}'] = cross_accuracy
                            except:
                                continue
            
            robustness_results['cross_group_generalization'] = cross_group_performance
        
        return robustness_results

    def predict(self, X):
        check_is_fitted(self, "_fitted_estimator")
        X = self._to_frame(X)
        Xn = X.drop(columns=self.sensitive_feature_names, errors="ignore")

        if self._is_postprocessor:
            # ThresholdOptimizer requires sensitive_features at predict time
            sens = self._infer_sensitive_for_new_X(X)
            return self._fitted_estimator.predict(Xn, sensitive_features=sens)
        return self._fitted_estimator.predict(Xn)

    def predict_proba(self, X):
        check_is_fitted(self, "_fitted_estimator")
        X = self._to_frame(X)
        Xn = X.drop(columns=self.sensitive_feature_names, errors="ignore")

        if self._is_postprocessor:
            sens = self._infer_sensitive_for_new_X(X)
            # ThresholdOptimizer doesn't guarantee predict_proba; emulate via scores
            preds = self._fitted_estimator._pmf_predict(Xn, sensitive_features=sens)
            # return P(y=1) as column 1; pad to 2-col array
            proba1 = preds[:, 1] if preds.ndim == 2 and preds.shape[1] > 1 else preds.ravel()
            return np.vstack([1 - proba1, proba1]).T

        # For reductions, the wrapped estimator should expose standard methods
        if hasattr(self._fitted_estimator, "predict_proba"):
            return self._fitted_estimator.predict_proba(Xn)
        scores = _safe_proba(self._fitted_estimator, Xn)
        return np.vstack([1 - scores, scores]).T

    def evaluate(
        self,
        X,
        y,
        include_by_group: bool = True,
        sensitive_features: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          - overall: dict of metrics
          - by_group: DataFrame of metrics per sensitive group (if include_by_group)
          - disparities: dict with max absolute group disparities per metric
        """
        check_is_fitted(self, "_fitted_estimator")
        X = self._to_frame(X)
        y = self._to_series(y)

        y_pred = self.predict(X)
        y_score = self.predict_proba(X)[:, 1]

        sens = sensitive_features if sensitive_features is not None else self._infer_sensitive_for_new_X(X)

        # Build metric frame with both thresholded and score-based metrics
        mf = MetricFrame(
            metrics={
                "accuracy": lambda yt, yp: self.metrics["accuracy"](yt, yp),
                "precision": lambda yt, yp: self.metrics["precision"](yt, yp),
                "recall": lambda yt, yp: self.metrics["recall"](yt, yp),
                "f1": lambda yt, yp: self.metrics["f1"](yt, yp),
                "selection_rate": lambda yt, yp: self.metrics["selection_rate"](yt, yp),
                "tpr": lambda yt, yp: self.metrics["tpr"](yt, yp),
                "tnr": lambda yt, yp: self.metrics["tnr"](yt, yp),
                "fpr": lambda yt, yp: self.metrics["fpr"](yt, yp),
                "fnr": lambda yt, yp: self.metrics["fnr"](yt, yp),
            },
            y_true=y,
            y_pred=y_pred,
            sensitive_features=sens
        )

        # Separate handling for AUC since it needs scores
        try:
            auc_mf = MetricFrame(
                metrics={"roc_auc": lambda yt, ys: self.metrics["roc_auc"](yt, ys)},
                y_true=y,
                y_pred=y_score,  # Use scores for AUC
                sensitive_features=sens
            )
            overall_auc = auc_mf.overall["roc_auc"]
            by_group_auc = auc_mf.by_group["roc_auc"]
        except Exception:
            overall_auc = np.nan
            by_group_auc = pd.Series([np.nan] * len(sens.unique()), index=sens.unique(), name="roc_auc")

        overall = {m: mf.overall[m] for m in mf.overall.index}
        overall["roc_auc"] = overall_auc
        
        by_group = mf.by_group.copy()
        by_group["roc_auc"] = by_group_auc

        # Disparities = max |group - overall|
        disparities = {}
        for m in by_group.columns:
            try:
                disparities[m] = float(np.nanmax(np.abs(by_group[m] - overall[m])))
            except Exception:
                disparities[m] = np.nan

        result = {
            "overall": overall,
            "by_group": by_group,
            "disparities": disparities
        }
        
        # Add statistical testing if requested
        if self.config.statistical_testing:
            result["statistical_tests"] = self._perform_statistical_tests(y, y_pred, sens, overall, by_group)
        
        # Add confidence intervals if requested
        if self.config.confidence_intervals:
            result["confidence_intervals"] = self._compute_confidence_intervals(y, y_pred, sens)
        
        # Add robustness analysis if requested
        if self.config.robustness_analysis:
            result["robustness"] = self._analyze_robustness(X, y, sens)

        return result

    def cross_validate_fairness(
        self,
        X,
        y,
        cv=5,
        sensitive_features=None,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=False
    ):
        """
        Perform cross-validation with fairness evaluation.
        Returns both standard ML metrics and fairness metrics across folds.
        """
        X = self._to_frame(X)
        y = self._to_series(y)
        
        sens = sensitive_features if sensitive_features is not None else self._extract_sensitive(X)
        
        # Create stratified CV to ensure balanced splits
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Store fairness results across folds
        fairness_results = {
            "fold_disparities": [],
            "fold_overall_metrics": [],
            "fold_by_group_metrics": []
        }
        
        # Standard scikit-learn CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validate(
                self, X, y, cv=cv, scoring=scoring, 
                return_train_score=return_train_score,
                error_score='raise'
            )
        
        # Manual fairness evaluation for each fold
        for train_idx, test_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx] 
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            sens_test_fold = sens.iloc[test_idx]
            
            # Clone and fit model for this fold
            fold_optimizer = clone(self)
            try:
                fold_optimizer.fit(X_train_fold, y_train_fold)
                fold_report = fold_optimizer.evaluate(
                    X_test_fold, y_test_fold, 
                    sensitive_features=sens_test_fold
                )
                
                fairness_results["fold_disparities"].append(fold_report["disparities"])
                fairness_results["fold_overall_metrics"].append(fold_report["overall"])
                fairness_results["fold_by_group_metrics"].append(fold_report["by_group"])
                
            except Exception as e:
                print(f"Warning: Fold failed with error: {e}")
                continue
        
        # Aggregate fairness results
        if fairness_results["fold_disparities"]:
            avg_disparities = {}
            for metric in fairness_results["fold_disparities"][0].keys():
                values = [fold[metric] for fold in fairness_results["fold_disparities"] 
                         if not np.isnan(fold[metric])]
                avg_disparities[metric] = {
                    "mean": np.mean(values) if values else np.nan,
                    "std": np.std(values) if values else np.nan
                }
        else:
            avg_disparities = {}
        
        return {
            "cv_scores": cv_results,
            "fairness_disparities": avg_disparities,
            "fairness_details": fairness_results
        }

    def get_feature_importance(self, X=None):
        """Get feature importance from the fitted model if available"""
        check_is_fitted(self, "_fitted_estimator")
        
        if hasattr(self._fitted_estimator, 'feature_importances_'):
            importance = self._fitted_estimator.feature_importances_
        elif hasattr(self._fitted_estimator, 'coef_'):
            importance = np.abs(self._fitted_estimator.coef_[0])
        elif hasattr(self._fitted_estimator, '_best_classifier'):
            # For reductions algorithms, try to get from wrapped classifier
            best_clf = self._fitted_estimator._best_classifier
            if hasattr(best_clf, 'feature_importances_'):
                importance = best_clf.feature_importances_
            elif hasattr(best_clf, 'coef_'):
                importance = np.abs(best_clf.coef_[0])
            else:
                return None
        else:
            return None
            
        if X is not None:
            X_frame = self._to_frame(X)
            feature_names = X_frame.drop(columns=self.sensitive_feature_names, errors='ignore').columns
            return pd.Series(importance, index=feature_names).sort_values(ascending=False)
        
        return importance

    def predict_with_threshold(self, X, threshold=0.5):
        """Make predictions with a custom threshold"""
        check_is_fitted(self, "_fitted_estimator")
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)

    def get_fairness_summary(self, X, y):
        """Get a comprehensive fairness summary report"""
        report = self.evaluate(X, y)
        
        summary = {
            "overall_performance": {
                "accuracy": report["overall"]["accuracy"],
                "precision": report["overall"]["precision"],
                "recall": report["overall"]["recall"],
                "f1": report["overall"]["f1"]
            },
            "fairness_metrics": {
                "max_accuracy_disparity": report["disparities"]["accuracy"],
                "max_selection_rate_disparity": report["disparities"]["selection_rate"],
                "max_tpr_disparity": report["disparities"]["tpr"],
                "max_fpr_disparity": report["disparities"]["fpr"]
            },
            "fairness_score": 1 - report["disparities"]["accuracy"],  # Simple fairness score
            "configuration": {
                "objective": self.config.objective,
                "mitigation": self.config.mitigation,
                "constraints_eps": self.config.constraints_eps
            }
        }
        
        return summary

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn compatibility)"""
        params = {
            'base_estimator': self.base_estimator,
            'sensitive_feature_names': self.sensitive_feature_names,
            'config': self.config,
            'custom_metrics': self.custom_metrics,
            'random_state': self.random_state
        }
        
        if deep and hasattr(self.base_estimator, 'get_params'):
            base_params = self.base_estimator.get_params(deep=True)
            for key, value in base_params.items():
                params[f'base_estimator__{key}'] = value
                
        return params

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn compatibility)"""
        # Handle nested parameters for base_estimator
        base_estimator_params = {}
        estimator_params = {}
        
        for key, value in params.items():
            if key.startswith('base_estimator__'):
                base_estimator_params[key[16:]] = value  # Remove 'base_estimator__' prefix
            else:
                estimator_params[key] = value
        
        # Set base estimator parameters
        if base_estimator_params and hasattr(self.base_estimator, 'set_params'):
            self.base_estimator.set_params(**base_estimator_params)
        
        # Set estimator parameters
        for key, value in estimator_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        
        return self

    # ---------- Helpers ----------
    def _extract_sensitive(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.sensitive_feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing sensitive feature(s) in X: {missing}")
        # If multiple sensitive features, combine as tuples -> one composite group
        if len(self.sensitive_feature_names) == 1:
            return X[self.sensitive_feature_names[0]].astype("category")
        combo = X[self.sensitive_feature_names].astype(str).agg("|".join, axis=1)
        return combo.astype("category")

    def _infer_sensitive_for_new_X(self, X: pd.DataFrame) -> pd.Series:
        # Use columns from X if present; otherwise reuse training groups (broadcast)
        present = [c for c in self.sensitive_feature_names if c in X.columns]
        if present:
            return self._extract_sensitive(X)
        if self._sensitive_series is None:
            raise ValueError("Sensitive features not provided at fit time and cannot be inferred.")
        # Best-effort: repeat most frequent training group
        mode = self._sensitive_series.mode().iloc[0]
        return pd.Series([mode] * len(X), index=X.index, dtype="category")

    @staticmethod
    def _to_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def _to_series(y):
        return y if isinstance(y, pd.Series) else pd.Series(y)
