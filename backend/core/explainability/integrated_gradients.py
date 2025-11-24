"""
Integrated Gradients Explainer Implementation
Provides attribution-based explanations for neural networks and differentiable models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
from .base_explainer import BaseExplainer, ExplanationResult, ExplanationType, ExplainerFactory

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explainer for attribution-based explanations.
    Works best with neural networks and other differentiable models.
    """
    
    def _initialize(self):
        """Initialize Integrated Gradients explainer"""
        self.baseline = None
        self.n_steps = self.config.get('n_steps', 50)
        self.method = self.config.get('method', 'gausslegendre')  # or 'riemann'
        
        # Check if we have gradient computation capability
        self.framework = self._detect_framework()
        
        # Set up baseline
        self._setup_baseline()
        
        logger.info(f"Initialized Integrated Gradients with {self.n_steps} steps using {self.framework}")
    
    def _detect_framework(self) -> str:
        """Detect the deep learning framework being used"""
        module_name = type(self.model).__module__ if self.model else ""
        
        if 'torch' in module_name or 'pytorch' in module_name:
            return 'pytorch'
        elif 'tensorflow' in module_name or 'keras' in module_name:
            return 'tensorflow'
        elif 'sklearn' in module_name:
            # For sklearn, we'll use numerical gradients
            return 'sklearn'
        else:
            return 'numpy'  # Fallback to numerical gradients
    
    def _setup_baseline(self):
        """Set up the baseline for integrated gradients"""
        if self.data is not None:
            data_array = self.validate_input(self.data)
            
            # Different baseline strategies
            baseline_type = self.config.get('baseline_type', 'zeros')
            
            if baseline_type == 'zeros':
                self.baseline = np.zeros(data_array.shape[1])
            elif baseline_type == 'mean':
                self.baseline = np.mean(data_array, axis=0)
            elif baseline_type == 'median':
                self.baseline = np.median(data_array, axis=0)
            elif baseline_type == 'random':
                idx = np.random.randint(len(data_array))
                self.baseline = data_array[idx]
            else:
                self.baseline = np.zeros(data_array.shape[1])
        else:
            logger.warning("No data provided for baseline. Using zeros.")
    
    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        baseline: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single prediction using Integrated Gradients.
        
        Args:
            instance: Single instance to explain
            baseline: Custom baseline (if None, uses self.baseline)
            target_class: Target class for classification (if None, uses predicted class)
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with attribution scores
        """
        # Convert to numpy
        instance_array = self.validate_input(instance)
        if instance_array.ndim == 1:
            instance_array = instance_array.reshape(1, -1)
        
        # Get baseline
        if baseline is not None:
            baseline_array = self.validate_input(baseline)
        else:
            baseline_array = self.baseline
            
        if baseline_array is None:
            baseline_array = np.zeros_like(instance_array[0])
        
        # Compute integrated gradients based on framework
        if self.framework == 'pytorch':
            attributions = self._compute_ig_pytorch(
                instance_array[0], baseline_array, target_class
            )
        elif self.framework == 'tensorflow':
            attributions = self._compute_ig_tensorflow(
                instance_array[0], baseline_array, target_class
            )
        else:
            # Fallback to numerical gradients
            attributions = self._compute_ig_numerical(
                instance_array[0], baseline_array, target_class
            )
        
        # Get prediction
        prediction = self.model.predict(instance_array)[0]
        
        # Get confidence if classification
        confidence = None
        if self.mode == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance_array)[0]
            confidence = float(np.max(proba))
        
        # Create feature importance dictionary
        feature_importance = {}
        feature_values = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = float(attributions[i])
            feature_values[feature_name] = float(instance_array[0][i])
        
        # Calculate attribution statistics
        total_attribution = np.sum(attributions)
        positive_attribution = np.sum(attributions[attributions > 0])
        negative_attribution = np.sum(attributions[attributions < 0])
        
        return ExplanationResult(
            method="Integrated Gradients",
            explanation_type=ExplanationType.LOCAL,
            feature_importance=feature_importance,
            feature_values=feature_values,
            attributions=attributions,
            prediction=float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            confidence=confidence,
            base_value=float(self._get_baseline_prediction(baseline_array)),
            raw_output={
                'attributions': attributions,
                'baseline': baseline_array,
                'total_attribution': total_attribution,
                'positive_attribution': positive_attribution,
                'negative_attribution': negative_attribution,
                'n_steps': self.n_steps
            }
        )
    
    def _compute_ig_pytorch(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute Integrated Gradients using PyTorch"""
        try:
            import torch
            
            # Convert to tensors
            instance_tensor = torch.FloatTensor(instance).unsqueeze(0)
            baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0)
            
            # Enable gradients
            instance_tensor.requires_grad = True
            
            # Generate interpolated inputs
            alphas = np.linspace(0, 1, self.n_steps)
            gradients = []
            
            for alpha in alphas:
                # Interpolate between baseline and instance
                interpolated = baseline_tensor + alpha * (instance_tensor - baseline_tensor)
                interpolated.requires_grad = True
                
                # Forward pass
                output = self.model(interpolated)
                
                # Get target class output
                if target_class is not None:
                    target_output = output[0, target_class]
                else:
                    target_output = output[0, torch.argmax(output[0])]
                
                # Compute gradients
                grad = torch.autograd.grad(
                    outputs=target_output,
                    inputs=interpolated,
                    create_graph=False
                )[0]
                
                gradients.append(grad.detach().numpy()[0])
            
            # Integrate gradients
            gradients = np.array(gradients)
            avg_gradients = np.mean(gradients, axis=0)
            integrated_gradients = (instance - baseline) * avg_gradients
            
            return integrated_gradients
            
        except ImportError:
            logger.warning("PyTorch not available. Falling back to numerical gradients.")
            return self._compute_ig_numerical(instance, baseline, target_class)
    
    def _compute_ig_tensorflow(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute Integrated Gradients using TensorFlow"""
        try:
            import tensorflow as tf
            
            # Convert to tensors
            instance_tensor = tf.constant(instance.reshape(1, -1), dtype=tf.float32)
            baseline_tensor = tf.constant(baseline.reshape(1, -1), dtype=tf.float32)
            
            # Generate interpolated inputs
            alphas = tf.linspace(0.0, 1.0, self.n_steps)
            gradients = []
            
            for alpha in alphas:
                # Interpolate
                interpolated = baseline_tensor + alpha * (instance_tensor - baseline_tensor)
                
                with tf.GradientTape() as tape:
                    tape.watch(interpolated)
                    
                    # Forward pass
                    output = self.model(interpolated)
                    
                    # Get target class output
                    if target_class is not None:
                        target_output = output[0, target_class]
                    else:
                        target_output = tf.reduce_max(output[0])
                
                # Compute gradients
                grad = tape.gradient(target_output, interpolated)
                gradients.append(grad.numpy()[0])
            
            # Integrate gradients
            gradients = np.array(gradients)
            avg_gradients = np.mean(gradients, axis=0)
            integrated_gradients = (instance - baseline) * avg_gradients
            
            return integrated_gradients
            
        except ImportError:
            logger.warning("TensorFlow not available. Falling back to numerical gradients.")
            return self._compute_ig_numerical(instance, baseline, target_class)
    
    def _compute_ig_numerical(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Integrated Gradients using numerical gradients.
        Works with any model but slower than automatic differentiation.
        """
        # Generate interpolation steps
        if self.method == 'gausslegendre':
            # Gauss-Legendre quadrature for better accuracy
            from scipy.special import roots_legendre
            points, weights = roots_legendre(self.n_steps)
            alphas = (points + 1) / 2  # Map from [-1, 1] to [0, 1]
            weights = weights / 2  # Adjust weights for interval change
        else:
            # Simple Riemann sum
            alphas = np.linspace(0, 1, self.n_steps)
            weights = np.ones(self.n_steps) / self.n_steps
        
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and instance
            interpolated = baseline + alpha * (instance - baseline)
            
            # Compute numerical gradient
            grad = self._numerical_gradient(interpolated, target_class)
            gradients.append(grad)
        
        # Weighted integration
        gradients = np.array(gradients)
        if self.method == 'gausslegendre':
            integrated_gradients = np.sum(gradients * weights[:, np.newaxis], axis=0)
        else:
            integrated_gradients = np.mean(gradients, axis=0)
        
        # Multiply by (input - baseline)
        integrated_gradients = (instance - baseline) * integrated_gradients
        
        return integrated_gradients
    
    def _numerical_gradient(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """
        Compute numerical gradient using finite differences.
        
        Args:
            x: Input point
            target_class: Target class for classification
            epsilon: Small perturbation for finite differences
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(x)
        
        # Get base prediction
        base_pred = self._get_model_output(x.reshape(1, -1), target_class)
        
        # Compute gradient for each feature
        for i in range(len(x)):
            # Positive perturbation
            x_plus = x.copy()
            x_plus[i] += epsilon
            pred_plus = self._get_model_output(x_plus.reshape(1, -1), target_class)
            
            # Negative perturbation
            x_minus = x.copy()
            x_minus[i] -= epsilon
            pred_minus = self._get_model_output(x_minus.reshape(1, -1), target_class)
            
            # Central difference
            grad[i] = (pred_plus - pred_minus) / (2 * epsilon)
        
        return grad
    
    def _get_model_output(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None
    ) -> float:
        """Get model output for given input and target class"""
        if self.mode == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(x)[0]
            if target_class is not None:
                return proba[target_class]
            else:
                return np.max(proba)
        else:
            return self.model.predict(x)[0]
    
    def _get_baseline_prediction(self, baseline: np.ndarray) -> float:
        """Get model prediction for baseline"""
        return self._get_model_output(baseline.reshape(1, -1))
    
    def explain_global(
        self,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate global explanation by aggregating local explanations.
        
        Args:
            n_samples: Number of samples to use
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with aggregated attributions
        """
        if self.data is None:
            raise ValueError("Data required for global explanations")
        
        data_array = self.validate_input(self.data)
        
        # Sample if needed
        if n_samples and n_samples < len(data_array):
            indices = np.random.choice(len(data_array), n_samples, replace=False)
            data_array = data_array[indices]
        
        # Compute attributions for each sample
        all_attributions = []
        
        logger.info(f"Computing integrated gradients for {len(data_array)} samples...")
        
        for i in range(len(data_array)):
            result = self.explain_instance(data_array[i])
            all_attributions.append(result.attributions)
        
        # Aggregate attributions
        all_attributions = np.array(all_attributions)
        mean_abs_attributions = np.mean(np.abs(all_attributions), axis=0)
        
        # Create feature importance
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = float(mean_abs_attributions[i])
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return ExplanationResult(
            method="Integrated Gradients",
            explanation_type=ExplanationType.GLOBAL,
            feature_importance=feature_importance,
            attributions=mean_abs_attributions,
            raw_output={
                'all_attributions': all_attributions,
                'mean_attributions': np.mean(all_attributions, axis=0),
                'std_attributions': np.std(all_attributions, axis=0)
            }
        )
    
    def get_feature_importance(
        self,
        method: str = "mean_abs",
        n_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Get feature importance scores using Integrated Gradients"""
        global_result = self.explain_global(n_samples=n_samples)
        return global_result.feature_importance
    
    def validate_completeness(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        baseline: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate attribution completeness (how well attributions sum to output difference).
        
        Args:
            instance: Instance to validate
            baseline: Baseline for comparison
            threshold: Acceptable difference threshold
            
        Returns:
            Dictionary with validation results
        """
        result = self.explain_instance(instance, baseline)
        
        # Get predictions
        instance_array = self.validate_input(instance)
        baseline_array = baseline if baseline is not None else self.baseline
        
        instance_output = self._get_model_output(instance_array.reshape(1, -1))
        baseline_output = self._get_baseline_prediction(baseline_array)
        
        # Calculate completeness
        output_diff = instance_output - baseline_output
        attribution_sum = np.sum(result.attributions)
        completeness_error = abs(output_diff - attribution_sum)
        
        return {
            'output_difference': output_diff,
            'attribution_sum': attribution_sum,
            'completeness_error': completeness_error,
            'is_complete': completeness_error < threshold,
            'relative_error': completeness_error / (abs(output_diff) + 1e-10)
        }


# Register the Integrated Gradients explainer
ExplainerFactory.register("integrated_gradients", IntegratedGradientsExplainer)
ExplainerFactory.register("ig", IntegratedGradientsExplainer)  # Alias


def create_ig_explainer(
    model: Any,
    data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    baseline_type: str = 'zeros',
    n_steps: int = 50,
    **kwargs
) -> IntegratedGradientsExplainer:
    """
    Convenience function to create an Integrated Gradients explainer.
    
    Args:
        model: Model to explain
        data: Background data for baseline
        baseline_type: Type of baseline ('zeros', 'mean', 'median', 'random')
        n_steps: Number of integration steps
        **kwargs: Additional parameters
        
    Returns:
        IntegratedGradientsExplainer instance
    """
    return IntegratedGradientsExplainer(
        model=model,
        data=data,
        baseline_type=baseline_type,
        n_steps=n_steps,
        **kwargs
    )