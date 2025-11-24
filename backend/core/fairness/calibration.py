"""
Calibration Fairness Metric
Ensures that predicted probabilities are well-calibrated across different groups.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.calibration import calibration_curve
from .base_metrics import (
    BaseFairnessMetric, 
    FairnessResult, 
    FairnessMetricType,
    BiasDirection,
    FairnessMetricFactory
)

logger = logging.getLogger(__name__)


class CalibrationMetric(BaseFairnessMetric):
    """
    Calibration fairness metric.
    
    Checks if P(Y=1|score=s,A=a) = P(Y=1|score=s,A=b) for all groups a, b.
    In other words, a predicted probability of 0.7 should mean 70% actual positive rate
    across all demographic groups.
    
    Well-calibrated models are essential for fair decision-making, as they ensure
    that risk scores mean the same thing regardless of group membership.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        threshold: float = 0.1,  # Max calibration error difference
        n_bins: int = 10,  # Number of bins for calibration curve
        strategy: str = 'uniform',  # 'uniform' or 'quantile'
        **kwargs
    ):
        """
        Initialize Calibration metric.
        
        Args:
            model: ML model to evaluate (must support predict_proba)
            threshold: Maximum acceptable calibration error difference between groups
            n_bins: Number of bins for calibration analysis
            strategy: Binning strategy ('uniform' or 'quantile')
            **kwargs: Additional parameters
        """
        super().__init__(model=model, threshold=threshold, **kwargs)
        self.n_bins = n_bins
        self.strategy = strategy
        self.metric_name = "Calibration"
        
    def _validate_model(self):
        """Validate that model supports probability predictions"""
        super()._validate_model()
        if self.model is not None and not hasattr(self.model, 'predict_proba'):
            raise ValueError("Calibration requires a model with predict_proba method")
    
    def calculate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> FairnessResult:
        """
        Calculate Calibration metric.
        
        Args:
            X: Input features
            y_true: True labels (REQUIRED for calibration)
            sensitive_feature: Protected attribute values
            y_pred: Predictions (not used for calibration)
            y_proba: Predicted probabilities (if None, will use model.predict_proba)
            **kwargs: Additional parameters
            
        Returns:
            FairnessResult object
        """
        # Get probability predictions
        if y_proba is not None:
            probabilities = self._to_numpy(y_proba)
        elif self.model is not None:
            proba_full = self.model.predict_proba(X)
            # For binary classification, use positive class probability
            if proba_full.ndim > 1 and proba_full.shape[1] > 1:
                probabilities = proba_full[:, 1]
            else:
                probabilities = proba_full.flatten()
        else:
            raise ValueError("Either y_proba or model with predict_proba must be provided")
        
        # Convert to numpy arrays
        y_true_np = self._to_numpy(y_true)
        sensitive_feature = self._to_numpy(sensitive_feature)
        
        # Ensure binary labels
        unique_labels = np.unique(y_true_np)
        if len(unique_labels) > 2:
            logger.warning("More than 2 classes found. Converting to binary (1 vs rest).")
            positive_label = 1 if 1 in unique_labels else max(unique_labels)
            y_true_np = (y_true_np == positive_label).astype(int)
        
        # Calculate group metrics
        group_metrics = self.calculate_group_metrics(
            y_true_np, probabilities, sensitive_feature
        )
        
        # Calculate overall calibration scores
        calibration_errors = group_metrics['calibration_errors']
        
        # Calculate overall score (max difference in calibration errors)
        overall_score = self._calculate_overall_score(calibration_errors)
        
        # Calculate disparate calibration (ratio of calibration errors)
        disparate_calibration = self._calculate_disparate_impact(calibration_errors)
        
        # Statistical significance test
        p_value, is_significant = self._test_calibration_equality(
            group_metrics['calibration_curves'],
            group_metrics['group_sizes']
        )
        
        # Detect bias direction
        bias_direction, affected_groups = self._detect_calibration_bias(
            calibration_errors,
            group_metrics['brier_scores']
        )
        
        # Check if passes threshold
        passes_threshold = overall_score <= self.threshold
        
        # Create result
        result = FairnessResult(
            metric_name=self.metric_name,
            metric_type=FairnessMetricType.CALIBRATION,
            overall_score=overall_score,
            group_scores=calibration_errors,
            disparate_impact_ratio=disparate_calibration,
            statistical_significance=p_value,
            bias_detected=is_significant and not passes_threshold,
            bias_direction=bias_direction if is_significant else BiasDirection.NO_BIAS,
            affected_groups=affected_groups if is_significant else [],
            threshold_used=self.threshold,
            passes_threshold=passes_threshold,
            sample_sizes=group_metrics['group_sizes'],
            raw_rates={
                'calibration_errors': calibration_errors,
                'brier_scores': group_metrics['brier_scores'],
                'expected_calibration_errors': group_metrics['ece_scores'],
                'max_calibration_errors': group_metrics['mce_scores']
            }
        )
        
        # Add calibration curves to raw output
        result.raw_output = {
            'calibration_curves': group_metrics['calibration_curves'],
            'bin_edges': group_metrics['bin_edges']
        }
        
        # Generate recommendation
        result.recommendation = self.generate_recommendation(result)
        
        return result
    
    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate calibration metrics for each group.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            groups: Group labels
            
        Returns:
            Dictionary with calibration metrics per group
        """
        unique_groups = self._get_unique_groups(groups)
        
        calibration_errors = {}
        brier_scores = {}
        ece_scores = {}  # Expected Calibration Error
        mce_scores = {}  # Maximum Calibration Error
        calibration_curves = {}
        group_sizes = {}
        
        # Global bin edges for consistency
        if self.strategy == 'uniform':
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
        else:
            # Quantile-based bins
            bin_edges = np.percentile(y_proba, np.linspace(0, 100, self.n_bins + 1))
            bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        for group in unique_groups:
            group_mask = groups == group
            group_proba = y_proba[group_mask]
            group_true = y_true[group_mask]
            
            group_sizes[str(group)] = int(np.sum(group_mask))
            
            # Calculate calibration curve
            if len(group_proba) > 0:
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        group_true,
                        group_proba,
                        n_bins=min(self.n_bins, len(group_proba) // 2),
                        strategy=self.strategy
                    )
                    
                    calibration_curves[str(group)] = {
                        'fraction_of_positives': fraction_of_positives,
                        'mean_predicted_value': mean_predicted_value
                    }
                    
                    # Calculate calibration error (mean absolute difference)
                    calib_error = np.mean(np.abs(
                        fraction_of_positives - mean_predicted_value
                    ))
                    calibration_errors[str(group)] = float(calib_error)
                    
                except Exception as e:
                    logger.warning(f"Could not calculate calibration for group {group}: {e}")
                    calibration_errors[str(group)] = 0.0
                    calibration_curves[str(group)] = {
                        'fraction_of_positives': np.array([]),
                        'mean_predicted_value': np.array([])
                    }
            else:
                calibration_errors[str(group)] = 0.0
                calibration_curves[str(group)] = {
                    'fraction_of_positives': np.array([]),
                    'mean_predicted_value': np.array([])
                }
            
            # Calculate Brier Score
            brier = self._calculate_brier_score(group_true, group_proba)
            brier_scores[str(group)] = float(brier)
            
            # Calculate Expected Calibration Error (ECE)
            ece = self._calculate_ece(group_true, group_proba, bin_edges)
            ece_scores[str(group)] = float(ece)
            
            # Calculate Maximum Calibration Error (MCE)
            mce = self._calculate_mce(group_true, group_proba, bin_edges)
            mce_scores[str(group)] = float(mce)
        
        return {
            'calibration_errors': calibration_errors,
            'brier_scores': brier_scores,
            'ece_scores': ece_scores,
            'mce_scores': mce_scores,
            'calibration_curves': calibration_curves,
            'group_sizes': group_sizes,
            'bin_edges': bin_edges,
            'unique_groups': [str(g) for g in unique_groups]
        }
    
    def _calculate_brier_score(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> float:
        """
        Calculate Brier Score (mean squared error of probabilities).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Brier score
        """
        if len(y_true) == 0:
            return 0.0
        
        return float(np.mean((y_proba - y_true) ** 2))
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        bin_edges: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE is the weighted average of calibration errors across bins.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            bin_edges: Bin boundaries
            
        Returns:
            ECE score
        """
        if len(y_true) == 0:
            return 0.0
        
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(len(bin_edges) - 1):
            # Find samples in this bin
            in_bin = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            
            # Handle last bin inclusively
            if i == len(bin_edges) - 2:
                in_bin = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(y_true[in_bin])
                bin_confidence = np.mean(y_proba[in_bin])
                
                # Weighted calibration error for this bin
                ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    
    def _calculate_mce(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        bin_edges: np.ndarray
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        MCE is the maximum calibration error across all bins.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            bin_edges: Bin boundaries
            
        Returns:
            MCE score
        """
        if len(y_true) == 0:
            return 0.0
        
        max_error = 0.0
        
        for i in range(len(bin_edges) - 1):
            # Find samples in this bin
            in_bin = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            
            # Handle last bin inclusively
            if i == len(bin_edges) - 2:
                in_bin = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(y_true[in_bin])
                bin_confidence = np.mean(y_proba[in_bin])
                
                bin_error = np.abs(bin_accuracy - bin_confidence)
                max_error = max(max_error, bin_error)
        
        return float(max_error)
    
    def _calculate_overall_score(self, calibration_errors: Dict[str, float]) -> float:
        """
        Calculate overall calibration score.
        0 = perfect calibration across groups, higher = worse
        
        Args:
            calibration_errors: Calibration errors per group
            
        Returns:
            Overall score (max difference in calibration errors)
        """
        if len(calibration_errors) < 2:
            return 0.0
        
        errors = list(calibration_errors.values())
        
        # Maximum difference in calibration errors
        max_diff = max(errors) - min(errors)
        
        return float(max_diff)
    
    def _test_calibration_equality(
        self,
        calibration_curves: Dict[str, Dict],
        group_sizes: Dict[str, int]
    ) -> Tuple[float, bool]:
        """
        Test statistical significance of calibration differences.
        
        Args:
            calibration_curves: Calibration curves per group
            group_sizes: Sample sizes per group
            
        Returns:
            Tuple of (p-value, is_significant)
        """
        if len(calibration_curves) < 2:
            return 1.0, False
        
        # Use Kolmogorov-Smirnov test for distribution comparison
        try:
            groups = list(calibration_curves.keys())
            
            # Get calibration errors for each group
            errors_by_group = []
            for group in groups:
                curve = calibration_curves[group]
                if len(curve['fraction_of_positives']) > 0:
                    errors = np.abs(
                        curve['fraction_of_positives'] - curve['mean_predicted_value']
                    )
                    errors_by_group.append(errors)
            
            if len(errors_by_group) < 2:
                return 1.0, False
            
            # Perform KS test between first two groups
            statistic, p_value = stats.ks_2samp(errors_by_group[0], errors_by_group[1])
            
            is_significant = p_value < (1 - self.confidence_level)
            return p_value, is_significant
            
        except Exception as e:
            logger.warning(f"Could not perform statistical test: {e}")
            return 1.0, False
    
    def _detect_calibration_bias(
        self,
        calibration_errors: Dict[str, float],
        brier_scores: Dict[str, float]
    ) -> Tuple[BiasDirection, List[str]]:
        """
        Detect calibration bias direction and affected groups.
        
        Args:
            calibration_errors: Calibration errors per group
            brier_scores: Brier scores per group
            
        Returns:
            Tuple of (bias_direction, affected_groups)
        """
        if len(calibration_errors) < 2:
            return BiasDirection.NO_BIAS, []
        
        errors = list(calibration_errors.values())
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Check if calibration is similar across groups
        if std_error < 0.02:  # Less than 2% variation
            return BiasDirection.NO_BIAS, []
        
        # Identify poorly calibrated groups
        poorly_calibrated = []
        well_calibrated = []
        
        for group, error in calibration_errors.items():
            if error > mean_error + std_error:
                poorly_calibrated.append(group)
            elif error < mean_error - std_error:
                well_calibrated.append(group)
        
        # Determine bias direction
        if poorly_calibrated and not well_calibrated:
            return BiasDirection.NEGATIVE_BIAS, poorly_calibrated
        elif well_calibrated and not poorly_calibrated:
            return BiasDirection.POSITIVE_BIAS, well_calibrated
        elif poorly_calibrated and well_calibrated:
            return BiasDirection.MIXED_BIAS, poorly_calibrated
        else:
            return BiasDirection.NO_BIAS, []
    
    def plot_calibration_curves(
        self,
        result: FairnessResult,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate calibration curve plots for visualization.
        
        Args:
            result: FairnessResult object
            save_path: Optional path to save plot
            
        Returns:
            Dictionary with plot data
        """
        if 'calibration_curves' not in result.raw_output:
            return {}
        
        calibration_curves = result.raw_output['calibration_curves']
        
        plot_data = {
            'groups': [],
            'curves': []
        }
        
        for group, curve_data in calibration_curves.items():
            plot_data['groups'].append(group)
            plot_data['curves'].append({
                'predicted': curve_data['mean_predicted_value'].tolist(),
                'actual': curve_data['fraction_of_positives'].tolist()
            })
        
        # Add perfect calibration line
        plot_data['perfect_calibration'] = {
            'x': [0, 1],
            'y': [0, 1]
        }
        
        return plot_data
    
    def suggest_recalibration(
        self,
        result: FairnessResult
    ) -> Dict[str, Any]:
        """
        Suggest recalibration methods for poorly calibrated groups.
        
        Args:
            result: FairnessResult object
            
        Returns:
            Dictionary with recalibration suggestions
        """
        if not result.bias_detected:
            return {
                'needed': False,
                'message': 'Model is well-calibrated across groups'
            }
        
        suggestions = {
            'needed': True,
            'methods': []
        }
        
        # Platt Scaling
        suggestions['methods'].append({
            'method': 'Platt Scaling',
            'description': 'Fit a logistic regression on the model outputs',
            'best_for': 'Models that output uncalibrated scores',
            'implementation': 'sklearn.calibration.CalibratedClassifierCV with method="sigmoid"'
        })
        
        # Isotonic Regression
        suggestions['methods'].append({
            'method': 'Isotonic Regression',
            'description': 'Non-parametric calibration using isotonic regression',
            'best_for': 'Non-monotonic calibration errors',
            'implementation': 'sklearn.calibration.CalibratedClassifierCV with method="isotonic"'
        })
        
        # Beta Calibration
        suggestions['methods'].append({
            'method': 'Beta Calibration',
            'description': 'Calibration using beta distribution',
            'best_for': 'Probabilities that don\'t span [0,1]',
            'implementation': 'Custom implementation or betacal package'
        })
        
        # Group-specific calibration
        if result.affected_groups:
            suggestions['methods'].append({
                'method': 'Group-Specific Calibration',
                'description': f'Apply separate calibration for groups: {", ".join(result.affected_groups)}',
                'best_for': 'When different groups have different calibration errors',
                'implementation': 'Train separate calibrators per group'
            })
        
        return suggestions
    
    def recalibrate_predictions(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        method: str = 'isotonic'
    ) -> np.ndarray:
        """
        Recalibrate predictions using specified method.
        
        Args:
            y_proba: Predicted probabilities
            y_true: True labels
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Recalibrated probabilities
        """
        from sklearn.calibration import calibration_curve
        from sklearn.isotonic import IsotonicRegression
        from scipy.optimize import minimize
        
        if method == 'isotonic':
            # Isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip')
            calibrated = ir.fit_transform(y_proba, y_true)
            
        elif method == 'sigmoid':
            # Platt scaling (logistic regression)
            def platt_loss(params):
                a, b = params
                calibrated = 1 / (1 + np.exp(-(a * y_proba + b)))
                return np.mean((calibrated - y_true) ** 2)
            
            result = minimize(platt_loss, [1, 0], method='BFGS')
            a, b = result.x
            calibrated = 1 / (1 + np.exp(-(a * y_proba + b)))
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return calibrated


# Register the Calibration metric
FairnessMetricFactory.register("calibration", CalibrationMetric)


def calculate_calibration(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
    sensitive_feature: Union[pd.Series, np.ndarray],
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.1,
    **kwargs
) -> FairnessResult:
    """
    Convenience function to calculate calibration metric.
    
    Args:
        model: ML model with predict_proba
        X: Input features
        y_true: True labels (required)
        sensitive_feature: Protected attribute
        y_proba: Optional predicted probabilities
        threshold: Maximum acceptable calibration difference
        **kwargs: Additional parameters
        
    Returns:
        FairnessResult object
    """
    metric = CalibrationMetric(model=model, threshold=threshold, **kwargs)
    return metric.calculate(X, y_true, sensitive_feature, y_proba=y_proba)