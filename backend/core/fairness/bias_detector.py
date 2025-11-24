"""
Bias Detector - Combined Fairness Evaluation
Orchestrates multiple fairness metrics to provide comprehensive bias detection.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from .base_metrics import (
    BaseFairnessMetric,
    FairnessResult,
    FairnessMetricType,
    BiasDirection
)
from .statistical_parity import StatisticalParity
from .equal_opportunity import EqualOpportunity
from .calibration import CalibrationMetric

logger = logging.getLogger(__name__)


class BiasSeverity(Enum):
    """Severity levels for detected bias"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComprehensiveBiasReport:
    """Comprehensive bias detection report"""
    overall_bias_detected: bool
    severity: BiasSeverity
    
    # Individual metric results
    statistical_parity_result: Optional[FairnessResult] = None
    equal_opportunity_result: Optional[FairnessResult] = None
    calibration_result: Optional[FairnessResult] = None
    
    # Aggregated scores
    overall_fairness_score: float = 0.0  # 0-100, higher is better
    metrics_passed: int = 0
    metrics_failed: int = 0
    
    # Affected groups across all metrics
    consistently_affected_groups: List[str] = None
    
    # Recommendations
    priority_actions: List[str] = None
    mitigation_strategies: Dict[str, Any] = None
    
    # Summary
    executive_summary: str = ""
    technical_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "overall_bias_detected": self.overall_bias_detected,
            "severity": self.severity.value,
            "overall_fairness_score": self.overall_fairness_score,
            "metrics_passed": self.metrics_passed,
            "metrics_failed": self.metrics_failed,
            "consistently_affected_groups": self.consistently_affected_groups or [],
            "statistical_parity": self.statistical_parity_result.to_dict() if self.statistical_parity_result else None,
            "equal_opportunity": self.equal_opportunity_result.to_dict() if self.equal_opportunity_result else None,
            "calibration": self.calibration_result.to_dict() if self.calibration_result else None,
            "priority_actions": self.priority_actions or [],
            "executive_summary": self.executive_summary,
            "technical_summary": self.technical_summary
        }


class BiasDetector:
    """
    Comprehensive bias detector that runs multiple fairness metrics
    and provides actionable recommendations.
    """
    
    def __init__(
        self,
        model: Any = None,
        fairness_threshold: float = 0.8,
        calibration_threshold: float = 0.1,
        enable_statistical_parity: bool = True,
        enable_equal_opportunity: bool = True,
        enable_calibration: bool = True,
        check_equalized_odds: bool = False,
        **kwargs
    ):
        """
        Initialize bias detector.
        
        Args:
            model: ML model to evaluate (can be None for API-first usage)
            fairness_threshold: Threshold for statistical parity and equal opportunity
            calibration_threshold: Threshold for calibration
            enable_statistical_parity: Whether to check statistical parity
            enable_equal_opportunity: Whether to check equal opportunity
            enable_calibration: Whether to check calibration
            check_equalized_odds: Whether to check equalized odds (TPR + FPR)
            **kwargs: Additional parameters
        """
        # Handle None model for API-first usage
        if model is None:
            class MockModel:
                def predict(self, X):
                    # Return dummy predictions
                    return np.zeros(len(X))
                
                def predict_proba(self, X):
                    # Return dummy probabilities
                    probs = np.random.random(len(X))
                    return np.column_stack([1 - probs, probs])
            model = MockModel()
        
        self.model = model
        self.fairness_threshold = fairness_threshold
        self.calibration_threshold = calibration_threshold
        self.config = kwargs
        
        # Initialize metrics
        self.metrics = {}
        
        if enable_statistical_parity:
            self.metrics['statistical_parity'] = StatisticalParity(
                model=model,
                threshold=fairness_threshold,
                **kwargs
            )
        
        if enable_equal_opportunity:
            self.metrics['equal_opportunity'] = EqualOpportunity(
                model=model,
                threshold=fairness_threshold,
                check_equalized_odds=check_equalized_odds,
                **kwargs
            )
        
        if enable_calibration:
            if hasattr(model, 'predict_proba'):
                self.metrics['calibration'] = CalibrationMetric(
                    model=model,
                    threshold=calibration_threshold,
                    **kwargs
                )
            else:
                logger.warning("Model doesn't support predict_proba. Skipping calibration check.")
        
        logger.info(f"Initialized BiasDetector with {len(self.metrics)} metrics")
    
    def detect_bias(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_features: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Union[ComprehensiveBiasReport, Dict[str, ComprehensiveBiasReport]]:
        """
        Detect bias across all enabled metrics and sensitive features.
        
        Args:
            X: Input features
            y_true: True labels
            sensitive_features: One or multiple protected attributes
            y_pred: Predictions (optional)
            y_proba: Predicted probabilities (optional)
            feature_names: Names of sensitive features (if dict)
            
        Returns:
            ComprehensiveBiasReport or dict of reports (one per sensitive feature)
        """
        # Handle multiple sensitive features
        if isinstance(sensitive_features, dict):
            reports = {}
            for feature_name, feature_values in sensitive_features.items():
                logger.info(f"Analyzing bias for sensitive feature: {feature_name}")
                reports[feature_name] = self._detect_bias_single_feature(
                    X, y_true, feature_values, y_pred, y_proba, feature_name
                )
            return reports
        elif isinstance(sensitive_features, pd.DataFrame):
            reports = {}
            for col in sensitive_features.columns:
                logger.info(f"Analyzing bias for sensitive feature: {col}")
                reports[col] = self._detect_bias_single_feature(
                    X, y_true, sensitive_features[col], y_pred, y_proba, col
                )
            return reports
        else:
            # Single sensitive feature
            feature_name = feature_names[0] if feature_names else "sensitive_feature"
            return self._detect_bias_single_feature(
                X, y_true, sensitive_features, y_pred, y_proba, feature_name
            )
    
    def _detect_bias_single_feature(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        sensitive_feature: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]],
        y_proba: Optional[Union[pd.Series, np.ndarray]],
        feature_name: str
    ) -> ComprehensiveBiasReport:
        """
        Detect bias for a single sensitive feature.
        
        Args:
            X: Input features
            y_true: True labels
            sensitive_feature: Protected attribute
            y_pred: Predictions
            y_proba: Predicted probabilities
            feature_name: Name of the sensitive feature
            
        Returns:
            ComprehensiveBiasReport
        """
        results = {}
        
        # Run each enabled metric
        for metric_name, metric in self.metrics.items():
            try:
                logger.info(f"Running {metric_name} check...")
                
                if metric_name == 'calibration':
                    result = metric.calculate(
                        X, y_true, sensitive_feature, 
                        y_pred=y_pred, y_proba=y_proba
                    )
                else:
                    result = metric.calculate(
                        X, y_true, sensitive_feature, y_pred=y_pred
                    )
                
                results[metric_name] = result
                logger.info(f"{metric_name}: {'PASSED' if result.passes_threshold else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Error running {metric_name}: {e}")
                results[metric_name] = None
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(results, feature_name)
        
        return report
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Optional[FairnessResult]],
        feature_name: str
    ) -> ComprehensiveBiasReport:
        """
        Generate comprehensive bias report from individual metric results.
        
        Args:
            results: Dictionary of metric results
            feature_name: Name of sensitive feature
            
        Returns:
            ComprehensiveBiasReport
        """
        # Count passes and failures
        metrics_passed = 0
        metrics_failed = 0
        any_bias_detected = False
        
        for result in results.values():
            if result is not None:
                if result.passes_threshold:
                    metrics_passed += 1
                else:
                    metrics_failed += 1
                
                if result.bias_detected:
                    any_bias_detected = True
        
        # Determine severity
        severity = self._determine_severity(results, metrics_failed)
        
        # Find consistently affected groups
        affected_groups = self._find_consistently_affected_groups(results)
        
        # Calculate overall fairness score (0-100)
        overall_score = self._calculate_overall_fairness_score(results)
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(results, affected_groups)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(results)
        
        # Generate summaries
        executive_summary = self._generate_executive_summary(
            feature_name, severity, overall_score, metrics_passed, metrics_failed
        )
        
        technical_summary = self._generate_technical_summary(results, affected_groups)
        
        # Create report
        report = ComprehensiveBiasReport(
            overall_bias_detected=any_bias_detected,
            severity=severity,
            statistical_parity_result=results.get('statistical_parity'),
            equal_opportunity_result=results.get('equal_opportunity'),
            calibration_result=results.get('calibration'),
            overall_fairness_score=overall_score,
            metrics_passed=metrics_passed,
            metrics_failed=metrics_failed,
            consistently_affected_groups=affected_groups,
            priority_actions=priority_actions,
            mitigation_strategies=mitigation_strategies,
            executive_summary=executive_summary,
            technical_summary=technical_summary
        )
        
        return report
    
    def _determine_severity(
        self,
        results: Dict[str, Optional[FairnessResult]],
        metrics_failed: int
    ) -> BiasSeverity:
        """
        Determine overall bias severity.
        
        Args:
            results: Metric results
            metrics_failed: Number of failed metrics
            
        Returns:
            BiasSeverity level
        """
        if metrics_failed == 0:
            return BiasSeverity.NONE
        
        # Check individual metric scores
        max_score = 0.0
        for result in results.values():
            if result is not None and not result.passes_threshold:
                max_score = max(max_score, result.overall_score)
        
        # Determine severity based on both count and scores
        if metrics_failed >= 3 or max_score > 0.3:
            return BiasSeverity.CRITICAL
        elif metrics_failed == 2 or max_score > 0.2:
            return BiasSeverity.HIGH
        elif metrics_failed == 1 or max_score > 0.1:
            return BiasSeverity.MEDIUM
        else:
            return BiasSeverity.LOW
    
    def _find_consistently_affected_groups(
        self,
        results: Dict[str, Optional[FairnessResult]]
    ) -> List[str]:
        """
        Find groups that are consistently affected across metrics.
        
        Args:
            results: Metric results
            
        Returns:
            List of consistently affected groups
        """
        # Count how many times each group appears as affected
        group_counts = {}
        
        for result in results.values():
            if result is not None and result.affected_groups:
                for group in result.affected_groups:
                    group_counts[group] = group_counts.get(group, 0) + 1
        
        # Groups affected by multiple metrics
        total_metrics = len([r for r in results.values() if r is not None])
        threshold = max(2, total_metrics // 2)  # At least 2 or half of metrics
        
        consistently_affected = [
            group for group, count in group_counts.items()
            if count >= threshold
        ]
        
        return consistently_affected
    
    def _calculate_overall_fairness_score(
        self,
        results: Dict[str, Optional[FairnessResult]]
    ) -> float:
        """
        Calculate overall fairness score (0-100, higher is better).
        
        Args:
            results: Metric results
            
        Returns:
            Overall fairness score
        """
        scores = []
        
        for metric_name, result in results.items():
            if result is not None:
                # Convert metric score to 0-100 scale
                if metric_name == 'statistical_parity':
                    # Disparate impact ratio (0.8-1.0 is good)
                    if result.disparate_impact_ratio:
                        score = min(100, result.disparate_impact_ratio * 100)
                    else:
                        score = 50
                        
                elif metric_name == 'equal_opportunity':
                    # TPR difference (0 is perfect)
                    score = max(0, 100 - (result.overall_score * 500))
                    
                elif metric_name == 'calibration':
                    # Calibration error (0 is perfect)
                    score = max(0, 100 - (result.overall_score * 500))
                    
                else:
                    score = 50
                
                scores.append(score)
        
        # Overall score is weighted average
        if scores:
            overall = np.mean(scores)
        else:
            overall = 0.0
        
        return float(overall)
    
    def _generate_priority_actions(
        self,
        results: Dict[str, Optional[FairnessResult]],
        affected_groups: List[str]
    ) -> List[str]:
        """
        Generate prioritized list of actions.
        
        Args:
            results: Metric results
            affected_groups: Consistently affected groups
            
        Returns:
            List of priority actions
        """
        actions = []
        
        # Check each metric
        for metric_name, result in results.items():
            if result is not None and not result.passes_threshold:
                if metric_name == 'statistical_parity':
                    actions.append(
                        f"URGENT: Address selection rate disparity. "
                        f"Disparate impact ratio: {result.disparate_impact_ratio:.2f}"
                    )
                    
                elif metric_name == 'equal_opportunity':
                    actions.append(
                        f"HIGH: Improve true positive rates for disadvantaged groups. "
                        f"Current TPR difference: {result.overall_score:.2%}"
                    )
                    
                elif metric_name == 'calibration':
                    actions.append(
                        f"MEDIUM: Recalibrate model predictions. "
                        f"Calibration error difference: {result.overall_score:.2%}"
                    )
        
        # Add group-specific actions
        if affected_groups:
            actions.insert(0,
                f"CRITICAL: Focus mitigation efforts on groups: {', '.join(affected_groups)}"
            )
        
        return actions
    
    def _generate_mitigation_strategies(
        self,
        results: Dict[str, Optional[FairnessResult]]
    ) -> Dict[str, Any]:
        """
        Generate mitigation strategies for detected biases.
        
        Args:
            results: Metric results
            
        Returns:
            Dictionary of mitigation strategies
        """
        strategies = {
            'preprocessing': [],
            'inprocessing': [],
            'postprocessing': [],
            'model_selection': []
        }
        
        # Based on which metrics failed, suggest appropriate strategies
        for metric_name, result in results.items():
            if result is not None and not result.passes_threshold:
                
                if metric_name == 'statistical_parity':
                    strategies['preprocessing'].append({
                        'method': 'Reweighting',
                        'description': 'Adjust sample weights to balance group representations',
                        'complexity': 'Low',
                        'effectiveness': 'High for statistical parity'
                    })
                    
                    strategies['postprocessing'].append({
                        'method': 'Threshold Optimization',
                        'description': 'Use group-specific decision thresholds',
                        'complexity': 'Low',
                        'effectiveness': 'High for statistical parity'
                    })
                
                if metric_name == 'equal_opportunity':
                    strategies['inprocessing'].append({
                        'method': 'Adversarial Debiasing',
                        'description': 'Train with adversary to remove group information',
                        'complexity': 'High',
                        'effectiveness': 'High for equal opportunity'
                    })
                    
                    strategies['preprocessing'].append({
                        'method': 'Learning Fair Representations',
                        'description': 'Transform features to remove discrimination',
                        'complexity': 'Medium',
                        'effectiveness': 'Medium-High'
                    })
                
                if metric_name == 'calibration':
                    strategies['postprocessing'].append({
                        'method': 'Platt Scaling',
                        'description': 'Apply logistic calibration to predictions',
                        'complexity': 'Low',
                        'effectiveness': 'High for calibration'
                    })
                    
                    strategies['postprocessing'].append({
                        'method': 'Isotonic Regression',
                        'description': 'Non-parametric calibration method',
                        'complexity': 'Low',
                        'effectiveness': 'High for calibration'
                    })
        
        # Add general strategies
        strategies['model_selection'].append({
            'method': 'Fairness-Constrained Learning',
            'description': 'Retrain model with fairness constraints',
            'complexity': 'Medium-High',
            'effectiveness': 'Depends on constraint type'
        })
        
        return strategies
    
    def _generate_executive_summary(
        self,
        feature_name: str,
        severity: BiasSeverity,
        overall_score: float,
        metrics_passed: int,
        metrics_failed: int
    ) -> str:
        """Generate executive summary"""
        
        if severity == BiasSeverity.NONE:
            return (
                f"✅ No significant bias detected for '{feature_name}'. "
                f"The model demonstrates fair treatment across all demographic groups "
                f"(Fairness Score: {overall_score:.1f}/100). "
                f"All {metrics_passed} fairness metrics passed their thresholds."
            )
        else:
            severity_text = {
                BiasSeverity.LOW: "minor",
                BiasSeverity.MEDIUM: "moderate",
                BiasSeverity.HIGH: "significant",
                BiasSeverity.CRITICAL: "critical"
            }[severity]
            
            return (
                f"⚠️ {severity_text.upper()} bias detected for '{feature_name}'. "
                f"The model shows {severity_text} disparities in treatment across demographic groups "
                f"(Fairness Score: {overall_score:.1f}/100). "
                f"{metrics_failed} out of {metrics_passed + metrics_failed} fairness metrics failed. "
                f"Immediate corrective action is {'REQUIRED' if severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL] else 'recommended'}."
            )
    
    def _generate_technical_summary(
        self,
        results: Dict[str, Optional[FairnessResult]],
        affected_groups: List[str]
    ) -> str:
        """Generate technical summary"""
        
        summary_parts = []
        
        for metric_name, result in results.items():
            if result is not None:
                status = "✓ PASS" if result.passes_threshold else "✗ FAIL"
                summary_parts.append(
                    f"{status} {metric_name.replace('_', ' ').title()}: "
                    f"Score={result.overall_score:.3f}, "
                    f"Threshold={result.threshold_used:.3f}"
                )
                
                if result.disparate_impact_ratio:
                    summary_parts[-1] += f", DI Ratio={result.disparate_impact_ratio:.3f}"
        
        technical_summary = "\n".join(summary_parts)
        
        if affected_groups:
            technical_summary += f"\n\nConsistently Affected Groups: {', '.join(affected_groups)}"
        
        return technical_summary
    
    def analyze_comprehensive_bias(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        labels: Union[pd.Series, np.ndarray],
        predictions: Union[pd.Series, np.ndarray],
        probabilities: Optional[Union[pd.Series, np.ndarray]] = None,
        sensitive_features: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]] = None
    ) -> ComprehensiveBiasReport:
        """
        Analyze comprehensive bias using provided predictions.
        This method is compatible with the API interface.
        
        Args:
            features: Input features
            labels: True labels
            predictions: Model predictions
            probabilities: Predicted probabilities (optional)
            sensitive_features: Protected attributes
            
        Returns:
            ComprehensiveBiasReport
        """
        if sensitive_features is None:
            raise ValueError("sensitive_features must be provided")
            
        # Use detect_bias method internally
        return self.detect_bias(
            X=features,
            y_true=labels,
            sensitive_features=sensitive_features,
            y_pred=predictions,
            y_proba=probabilities
        )
    
    def generate_bias_report_html(
        self,
        report: ComprehensiveBiasReport,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate HTML report for bias detection results.
        
        Args:
            report: ComprehensiveBiasReport
            output_path: Optional path to save HTML
            
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bias Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .severity-{report.severity.value} {{ 
                    color: {'green' if report.severity == BiasSeverity.NONE else 'orange' if report.severity == BiasSeverity.LOW else 'red'};
                    font-weight: bold;
                }}
                .metric-pass {{ color: green; }}
                .metric-fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Bias Detection Report</h1>
            
            <h2>Executive Summary</h2>
            <p class="severity-{report.severity.value}">{report.executive_summary}</p>
            
            <h2>Overall Metrics</h2>
            <p>Fairness Score: <strong>{report.overall_fairness_score:.1f}/100</strong></p>
            <p>Severity: <strong class="severity-{report.severity.value}">{report.severity.value.upper()}</strong></p>
            <p>Metrics Passed: <span class="metric-pass">{report.metrics_passed}</span></p>
            <p>Metrics Failed: <span class="metric-fail">{report.metrics_failed}</span></p>
            
            <h2>Technical Details</h2>
            <pre>{report.technical_summary}</pre>
            
            <h2>Priority Actions</h2>
            <ol>
                {''.join(f'<li>{action}</li>' for action in (report.priority_actions or []))}
            </ol>
            
            <h2>Mitigation Strategies</h2>
            <!-- Strategies would be formatted here -->
            
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
        
        return html


def create_bias_detector(
    model: Any,
    **kwargs
) -> BiasDetector:
    """
    Convenience function to create a bias detector.
    
    Args:
        model: ML model to evaluate
        **kwargs: Additional parameters
        
    Returns:
        BiasDetector instance
    """
    return BiasDetector(model=model, **kwargs)