"""
Anchors Explainer Implementation
Provides rule-based explanations with high precision guarantees.
Anchors are IF-THEN rules that locally approximate the model's behavior.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from .base_explainer import BaseExplainer, ExplanationResult, ExplanationType, ExplainerFactory

logger = logging.getLogger(__name__)


class AnchorsExplainer(BaseExplainer):
    """
    Anchors-based explainer for high-precision rule-based explanations.
    
    Anchors are IF-THEN rules that "anchor" a prediction, meaning that
    instances satisfying these rules are predicted the same way with high confidence.
    
    Example: "IF income > 50000 AND credit_score > 700 THEN loan=approved (precision: 0.95)"
    """
    
    def _initialize(self):
        """Initialize Anchors explainer"""
        self.threshold = self.config.get('threshold', 0.95)  # Precision threshold
        self.delta = self.config.get('delta', 0.1)  # Confidence level
        self.tau = self.config.get('tau', 0.15)  # Tolerance parameter
        self.batch_size = self.config.get('batch_size', 100)
        self.max_anchor_size = self.config.get('max_anchor_size', None)
        self.beam_size = self.config.get('beam_size', 4)
        
        # Discretization for continuous features
        self.discretization_method = self.config.get('discretization', 'quartile')
        self.n_bins = self.config.get('n_bins', 4)
        
        # Store discretizers
        self.discretizers = {}
        
        # Setup discretization if data is available
        if self.data is not None:
            self._setup_discretization()
        
        logger.info(f"Initialized Anchors explainer with precision threshold {self.threshold}")
    
    def _setup_discretization(self):
        """Setup discretizers for continuous features"""
        if self.data is None:
            return
        
        data_array = self.validate_input(self.data)
        
        for i in range(data_array.shape[1]):
            feature_values = data_array[:, i]
            
            # Check if feature needs discretization
            if self._is_continuous_feature(feature_values):
                self.discretizers[i] = self._create_discretizer(
                    feature_values, 
                    self.discretization_method
                )
    
    def _is_continuous_feature(self, values: np.ndarray) -> bool:
        """Check if feature is continuous"""
        unique_values = np.unique(values)
        
        # If more than n_bins unique values and not all integers, consider continuous
        if len(unique_values) > self.n_bins:
            if not np.all(values == values.astype(int)):
                return True
        
        return False
    
    def _create_discretizer(
        self, 
        values: np.ndarray, 
        method: str
    ) -> Dict[str, Any]:
        """
        Create discretizer for a continuous feature.
        
        Args:
            values: Feature values
            method: Discretization method ('quartile', 'uniform', 'kmeans')
            
        Returns:
            Discretizer dictionary with bins and labels
        """
        if method == 'quartile':
            # Quantile-based discretization
            bins = np.percentile(values, np.linspace(0, 100, self.n_bins + 1))
        elif method == 'uniform':
            # Uniform-width bins
            bins = np.linspace(np.min(values), np.max(values), self.n_bins + 1)
        elif method == 'kmeans':
            # K-means based discretization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
            kmeans.fit(values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # Create bins from cluster centers
            bins = [np.min(values)]
            for i in range(len(centers) - 1):
                bins.append((centers[i] + centers[i+1]) / 2)
            bins.append(np.max(values))
            bins = np.array(bins)
        else:
            # Default to quartile
            bins = np.percentile(values, np.linspace(0, 100, self.n_bins + 1))
        
        # Ensure unique bins
        bins = np.unique(bins)
        
        # Create labels
        labels = []
        for i in range(len(bins) - 1):
            labels.append(f"[{bins[i]:.2f}, {bins[i+1]:.2f})")
        
        return {
            'bins': bins,
            'labels': labels,
            'method': method
        }
    
    def _discretize_instance(self, instance: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Discretize a continuous instance.
        
        Args:
            instance: Instance to discretize
            
        Returns:
            Tuple of (discretized_instance, feature_descriptions)
        """
        discretized = instance.copy()
        descriptions = {}
        
        for feature_idx, discretizer in self.discretizers.items():
            value = instance[feature_idx]
            bins = discretizer['bins']
            labels = discretizer['labels']
            
            # Find which bin the value falls into
            bin_idx = np.digitize(value, bins) - 1
            bin_idx = max(0, min(bin_idx, len(labels) - 1))
            
            discretized[feature_idx] = bin_idx
            descriptions[feature_idx] = labels[bin_idx]
        
        return discretized, descriptions
    
    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None,
        beam_size: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single prediction using Anchors.
        
        Args:
            instance: Single instance to explain
            threshold: Precision threshold (default: self.threshold)
            beam_size: Beam search size (default: self.beam_size)
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with anchor rules
        """
        # Convert to numpy
        instance_array = self.validate_input(instance)
        if instance_array.ndim == 1:
            instance_array = instance_array.reshape(1, -1)
        
        # Get prediction for this instance
        prediction = self.model.predict(instance_array)[0]
        
        # Get confidence
        confidence = None
        if self.mode == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance_array)[0]
            confidence = float(np.max(proba))
        
        # Use provided threshold or default
        threshold = threshold or self.threshold
        beam_size = beam_size or self.beam_size
        
        # Discretize instance
        discretized_instance, feature_descriptions = self._discretize_instance(
            instance_array[0]
        )
        
        # Find anchor using beam search
        anchor_rules, anchor_precision, anchor_coverage = self._find_anchor(
            instance_array[0],
            discretized_instance,
            prediction,
            threshold,
            beam_size
        )
        
        # Convert anchor rules to human-readable format
        rules = []
        feature_importance = {}
        feature_values = {}
        
        for feature_idx in anchor_rules:
            feature_name = self.feature_names[feature_idx]
            feature_value = instance_array[0][feature_idx]
            
            # Get description
            if feature_idx in feature_descriptions:
                rule_text = f"{feature_name} is {feature_descriptions[feature_idx]}"
            else:
                # For categorical or already discrete features
                rule_text = f"{feature_name} = {feature_value:.2f}"
            
            rules.append(rule_text)
            
            # Feature importance based on inclusion in anchor
            feature_importance[feature_name] = 1.0  # Binary: in anchor or not
            feature_values[feature_name] = float(feature_value)
        
        # Add zero importance for features not in anchor
        for i, feature_name in enumerate(self.feature_names):
            if i not in anchor_rules:
                feature_importance[feature_name] = 0.0
                feature_values[feature_name] = float(instance_array[0][i])
        
        # Create summary rule
        summary_rule = f"IF {' AND '.join(rules)} THEN prediction={prediction}"
        
        return ExplanationResult(
            method="Anchors",
            explanation_type=ExplanationType.LOCAL,
            feature_importance=feature_importance,
            feature_values=feature_values,
            rules=[summary_rule] + rules,
            prediction=float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            confidence=confidence,
            precision=anchor_precision,
            coverage=anchor_coverage,
            raw_output={
                'anchor_features': anchor_rules,
                'anchor_precision': anchor_precision,
                'anchor_coverage': anchor_coverage,
                'discretized_instance': discretized_instance,
                'feature_descriptions': feature_descriptions
            }
        )
    
    def _find_anchor(
        self,
        instance: np.ndarray,
        discretized_instance: np.ndarray,
        prediction: Any,
        threshold: float,
        beam_size: int
    ) -> Tuple[List[int], float, float]:
        """
        Find anchor rules using beam search.
        
        Args:
            instance: Original instance
            discretized_instance: Discretized instance
            prediction: Model prediction for instance
            threshold: Precision threshold
            beam_size: Beam search size
            
        Returns:
            Tuple of (anchor_features, precision, coverage)
        """
        n_features = len(instance)
        
        # Start with empty anchor
        candidates = [set()]
        best_anchor = set()
        best_precision = 0.0
        
        # Beam search
        for depth in range(min(n_features, self.max_anchor_size or n_features)):
            new_candidates = []
            
            for candidate in candidates:
                # Try adding each feature not in candidate
                for feature_idx in range(n_features):
                    if feature_idx not in candidate:
                        new_candidate = candidate | {feature_idx}
                        
                        # Evaluate this candidate
                        precision, coverage = self._evaluate_anchor(
                            instance,
                            discretized_instance,
                            new_candidate,
                            prediction
                        )
                        
                        # Check if this anchor meets threshold
                        if precision >= threshold:
                            if len(new_candidate) < len(best_anchor) or not best_anchor:
                                best_anchor = new_candidate
                                best_precision = precision
                            # Don't expand further if we found an anchor
                            continue
                        
                        # Add to candidates for next iteration
                        new_candidates.append((new_candidate, precision, coverage))
            
            # If we found an anchor, stop
            if best_anchor:
                break
            
            # Keep top beam_size candidates
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [c[0] for c in new_candidates[:beam_size]]
            
            # If no candidates left, stop
            if not candidates:
                break
        
        # If no anchor found meeting threshold, return best candidate
        if not best_anchor and candidates:
            best_anchor = candidates[0]
            best_precision, coverage = self._evaluate_anchor(
                instance, discretized_instance, best_anchor, prediction
            )
        else:
            coverage = self._calculate_coverage(
                discretized_instance, best_anchor
            )
        
        return list(best_anchor), best_precision, coverage
    
    def _evaluate_anchor(
        self,
        instance: np.ndarray,
        discretized_instance: np.ndarray,
        anchor_features: set,
        prediction: Any
    ) -> Tuple[float, float]:
        """
        Evaluate precision of an anchor.
        
        Args:
            instance: Original instance
            discretized_instance: Discretized instance
            anchor_features: Features in the anchor
            prediction: Expected prediction
            
        Returns:
            Tuple of (precision, coverage)
        """
        # Generate samples that satisfy the anchor
        samples = self._generate_samples_with_anchor(
            instance,
            discretized_instance,
            anchor_features
        )
        
        # Get predictions for samples
        predictions = self.model.predict(samples)
        
        # Calculate precision
        if len(predictions) == 0:
            return 0.0, 0.0
        
        if self.mode == 'classification':
            # For classification, check if prediction matches
            matches = (predictions == prediction).sum()
        else:
            # For regression, check if predictions are within tolerance
            tolerance = self.config.get('regression_tolerance', 0.1)
            matches = (np.abs(predictions - prediction) <= tolerance * abs(prediction)).sum()
        
        precision = matches / len(predictions)
        
        # Calculate coverage (how often anchor applies)
        coverage = self._calculate_coverage(discretized_instance, anchor_features)
        
        return precision, coverage
    
    def _generate_samples_with_anchor(
        self,
        instance: np.ndarray,
        discretized_instance: np.ndarray,
        anchor_features: set
    ) -> np.ndarray:
        """
        Generate samples that satisfy the anchor conditions.
        
        Args:
            instance: Original instance
            discretized_instance: Discretized instance
            anchor_features: Features in the anchor
            
        Returns:
            Array of samples
        """
        samples = []
        
        # Generate batch_size samples
        for _ in range(self.batch_size):
            sample = instance.copy()
            
            # For features in anchor, keep the same discretized value
            # For other features, sample from the data distribution
            for feature_idx in range(len(instance)):
                if feature_idx not in anchor_features:
                    # Sample from training data
                    if self.data is not None:
                        data_array = self.validate_input(self.data)
                        random_idx = np.random.randint(len(data_array))
                        sample[feature_idx] = data_array[random_idx, feature_idx]
                    else:
                        # If no training data, perturb slightly
                        sample[feature_idx] = instance[feature_idx] + np.random.randn() * 0.1
            
            samples.append(sample)
        
        return np.array(samples)
    
    def _calculate_coverage(
        self,
        discretized_instance: np.ndarray,
        anchor_features: set
    ) -> float:
        """
        Calculate coverage of the anchor (how often it applies).
        
        Args:
            discretized_instance: Discretized instance
            anchor_features: Features in the anchor
            
        Returns:
            Coverage score
        """
        if self.data is None or not anchor_features:
            return 1.0
        
        data_array = self.validate_input(self.data)
        
        # Discretize all data
        matches = 0
        for i in range(len(data_array)):
            discretized, _ = self._discretize_instance(data_array[i])
            
            # Check if this sample satisfies the anchor
            satisfies = True
            for feature_idx in anchor_features:
                if discretized[feature_idx] != discretized_instance[feature_idx]:
                    satisfies = False
                    break
            
            if satisfies:
                matches += 1
        
        coverage = matches / len(data_array)
        return coverage
    
    def explain_global(
        self,
        n_samples: Optional[int] = 100,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate global explanation by finding common anchor patterns.
        
        Args:
            n_samples: Number of samples to analyze
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with common rules
        """
        if self.data is None:
            raise ValueError("Data required for global explanations")
        
        data_array = self.validate_input(self.data)
        
        # Sample if needed
        if n_samples and n_samples < len(data_array):
            indices = np.random.choice(len(data_array), n_samples, replace=False)
            data_array = data_array[indices]
        
        # Collect anchors from multiple instances
        all_rules = []
        feature_frequencies = {fname: 0 for fname in self.feature_names}
        
        logger.info(f"Finding anchors for {len(data_array)} samples...")
        
        for i in range(min(len(data_array), n_samples or 100)):
            try:
                result = self.explain_instance(data_array[i])
                all_rules.extend(result.rules[1:])  # Skip summary rule
                
                # Count feature frequencies
                for fname, importance in result.feature_importance.items():
                    if importance > 0:
                        feature_frequencies[fname] += 1
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")
                continue
        
        # Calculate global feature importance as frequency
        total = len(data_array)
        feature_importance = {
            fname: freq / total 
            for fname, freq in feature_frequencies.items()
        }
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Find most common rules
        from collections import Counter
        rule_counts = Counter(all_rules)
        common_rules = [rule for rule, _ in rule_counts.most_common(10)]
        
        return ExplanationResult(
            method="Anchors",
            explanation_type=ExplanationType.GLOBAL,
            feature_importance=feature_importance,
            rules=common_rules,
            raw_output={
                'rule_frequencies': dict(rule_counts),
                'n_samples': len(data_array)
            }
        )
    
    def get_feature_importance(
        self,
        method: str = "frequency",
        n_samples: Optional[int] = 100,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance based on anchor frequency.
        
        Args:
            method: Method to calculate importance ('frequency')
            n_samples: Number of samples
            
        Returns:
            Dictionary of feature importance
        """
        global_result = self.explain_global(n_samples=n_samples)
        return global_result.feature_importance
    
    @property
    def supports_global(self) -> bool:
        """Anchors supports global through aggregation"""
        return True
    
    @property
    def supports_local(self) -> bool:
        """Anchors is primarily for local explanations"""
        return True


# Register the Anchors explainer
ExplainerFactory.register("anchors", AnchorsExplainer)
ExplainerFactory.register("anchor", AnchorsExplainer)  # Alias


def create_anchors_explainer(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.95,
    **kwargs
) -> AnchorsExplainer:
    """
    Convenience function to create an Anchors explainer.
    
    Args:
        model: ML model to explain
        data: Training data (required for Anchors)
        feature_names: Names of features
        threshold: Precision threshold
        **kwargs: Additional parameters
        
    Returns:
        AnchorsExplainer instance
    """
    return AnchorsExplainer(
        model=model,
        data=data,
        feature_names=feature_names,
        threshold=threshold,
        **kwargs
    )