"""
LIME Explainer Implementation
Provides model-agnostic LIME explanations for any ML model.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import lime
import lime.lime_tabular
import lime.lime_text
import logging
from .base_explainer import BaseExplainer, ExplanationResult, ExplanationType, ExplainerFactory

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME-based explainer for model-agnostic local explanations.
    Focuses on explaining individual predictions by approximating the model locally.
    """
    
    def _initialize(self):
        """Initialize LIME explainer based on data type"""
        self.lime_explainer = None
        
        # Convert data to numpy if needed
        if self.data is not None:
            self.training_data = self.validate_input(self.data)
            
            # Get basic statistics for continuous features
            self.data_stats = {
                'mean': np.mean(self.training_data, axis=0),
                'std': np.std(self.training_data, axis=0),
                'min': np.min(self.training_data, axis=0),
                'max': np.max(self.training_data, axis=0)
            }
        else:
            self.training_data = None
            self.data_stats = None
        
        # Initialize LIME tabular explainer
        self._initialize_tabular_explainer()
    
    def _initialize_tabular_explainer(self):
        """Initialize LIME tabular explainer"""
        if self.training_data is None:
            logger.warning("No training data provided. LIME will use random perturbations.")
            return
        
        # Determine feature types (continuous vs categorical)
        feature_types = self._infer_feature_types()
        
        # Get categorical features indices
        categorical_features = [i for i, ftype in enumerate(feature_types) if ftype == 'categorical']
        
        # Determine mode
        lime_mode = 'classification' if self.mode == 'classification' else 'regression'
        
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=categorical_features if categorical_features else None,
            mode=lime_mode,
            discretize_continuous=self.config.get('discretize_continuous', True),
            discretizer=self.config.get('discretizer', 'quartile'),
            sample_around_instance=True,
            random_state=self.config.get('random_state', 42),
            verbose=False
        )
        
        logger.info(f"Initialized LIME tabular explainer with {len(self.feature_names)} features")
    
    def _infer_feature_types(self) -> List[str]:
        """
        Infer whether features are continuous or categorical.
        
        Returns:
            List of 'continuous' or 'categorical' for each feature
        """
        if self.training_data is None:
            return ['continuous'] * len(self.feature_names)
        
        feature_types = []
        
        for i in range(self.training_data.shape[1]):
            unique_values = np.unique(self.training_data[:, i])
            
            # Heuristics for categorical detection
            if len(unique_values) <= 10:  # Few unique values
                feature_types.append('categorical')
            elif np.all(unique_values == unique_values.astype(int)):  # All integers
                if len(unique_values) < 0.05 * len(self.training_data):  # Less than 5% unique
                    feature_types.append('categorical')
                else:
                    feature_types.append('continuous')
            else:
                feature_types.append('continuous')
        
        return feature_types
    
    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        num_features: Optional[int] = None,
        num_samples: int = 5000,
        distance_metric: str = 'euclidean',
        model_regressor: Optional[Any] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single prediction using LIME.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME to generate
            distance_metric: Distance metric for weights
            model_regressor: Local model to use (default: Ridge)
            **kwargs: Additional LIME parameters
            
        Returns:
            ExplanationResult with LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Provide training data.")
        
        # Convert to numpy
        instance_array = self.validate_input(instance)
        
        # Ensure 1D array for LIME
        if instance_array.ndim == 2:
            instance_array = instance_array[0]
        
        # Default number of features
        if num_features is None:
            num_features = min(10, len(self.feature_names))
        
        # Get prediction function
        if self.mode == 'classification' and hasattr(self.model, 'predict_proba'):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            data_row=instance_array,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor
        )
        
        # Get prediction
        prediction = self.model.predict(instance_array.reshape(1, -1))[0]
        
        # Get confidence if classification
        confidence = None
        if self.mode == 'classification' and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance_array.reshape(1, -1))[0]
            confidence = float(np.max(proba))
        
        # Extract feature importance and create rules
        feature_importance = {}
        feature_values = {}
        rules = []
        
        # Get explanation as list
        explanation_list = explanation.as_list()
        
        for feature_condition, importance in explanation_list:
            # Parse feature name and condition
            feature_name = self._parse_feature_name(feature_condition)
            
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                feature_importance[feature_name] = float(importance)
                feature_values[feature_name] = float(instance_array[feature_idx])
                
                # Create human-readable rule
                rule = self._create_rule(feature_condition, importance)
                rules.append(rule)
            else:
                # Handle discretized features
                for fname in self.feature_names:
                    if fname in feature_condition:
                        feature_name = fname
                        break
                
                if feature_name:
                    feature_idx = self.feature_names.index(feature_name)
                    feature_importance[feature_name] = float(importance)
                    feature_values[feature_name] = float(instance_array[feature_idx])
                    
                    rule = self._create_rule(feature_condition, importance)
                    rules.append(rule)
        
        # Add remaining features with zero importance
        for fname in self.feature_names:
            if fname not in feature_importance:
                idx = self.feature_names.index(fname)
                feature_values[fname] = float(instance_array[idx])
                # Don't add zero importance to keep explanation sparse
        
        # Get local model metrics
        local_model_r2 = explanation.score if hasattr(explanation, 'score') else None
        local_model_coverage = explanation.local_pred if hasattr(explanation, 'local_pred') else None
        
        return ExplanationResult(
            method="LIME",
            explanation_type=ExplanationType.LOCAL,
            feature_importance=feature_importance,
            feature_values=feature_values,
            rules=rules,
            prediction=float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            confidence=confidence,
            coverage=local_model_r2,  # R² of local model
            raw_output={
                'explanation': explanation,
                'as_list': explanation_list,
                'local_model_r2': local_model_r2
            }
        )
    
    def _parse_feature_name(self, feature_condition: str) -> str:
        """
        Parse feature name from LIME condition string.
        
        Args:
            feature_condition: String like 'feature_name > 0.5' or 'feature_name'
            
        Returns:
            Feature name
        """
        # Remove operators and values
        for op in ['<=', '>=', '<', '>', '=', '!=']:
            if op in feature_condition:
                return feature_condition.split(op)[0].strip()
        
        # If no operator, might be categorical
        return feature_condition.strip()
    
    def _create_rule(self, feature_condition: str, importance: float) -> str:
        """
        Create human-readable rule from LIME condition.
        
        Args:
            feature_condition: LIME condition string
            importance: Feature importance value
            
        Returns:
            Human-readable rule
        """
        impact = "increases" if importance > 0 else "decreases"
        strength = abs(importance)
        
        if strength > 0.5:
            impact_level = "strongly " + impact
        elif strength > 0.2:
            impact_level = "moderately " + impact
        else:
            impact_level = "slightly " + impact
        
        return f"When {feature_condition}, it {impact_level} the prediction by {strength:.3f}"
    
    def explain_global(
        self,
        n_samples: Optional[int] = 100,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate global explanation by aggregating local explanations.
        LIME is primarily a local method, so this aggregates multiple local explanations.
        
        Args:
            n_samples: Number of samples to explain and aggregate
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with aggregated importance
        """
        if self.training_data is None:
            raise ValueError("Training data required for global explanations")
        
        # Sample instances to explain
        if n_samples > len(self.training_data):
            n_samples = len(self.training_data)
        
        indices = np.random.choice(len(self.training_data), n_samples, replace=False)
        sample_data = self.training_data[indices]
        
        # Collect feature importance from each instance
        all_importances = {fname: [] for fname in self.feature_names}
        
        logger.info(f"Generating {n_samples} local explanations for global view...")
        
        for i in range(n_samples):
            try:
                result = self.explain_instance(
                    sample_data[i],
                    num_features=len(self.feature_names)
                )
                
                for fname in self.feature_names:
                    if fname in result.feature_importance:
                        all_importances[fname].append(result.feature_importance[fname])
                    else:
                        all_importances[fname].append(0.0)
                        
            except Exception as e:
                logger.warning(f"Failed to explain instance {i}: {e}")
                continue
        
        # Aggregate importances (mean absolute value)
        global_importance = {}
        for fname, values in all_importances.items():
            if values:
                global_importance[fname] = float(np.mean(np.abs(values)))
            else:
                global_importance[fname] = 0.0
        
        # Sort by importance
        global_importance = dict(sorted(
            global_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return ExplanationResult(
            method="LIME",
            explanation_type=ExplanationType.GLOBAL,
            feature_importance=global_importance,
            raw_output={
                'n_samples': n_samples,
                'aggregation_method': 'mean_absolute',
                'all_importances': all_importances
            }
        )
    
    def get_feature_importance(
        self,
        method: str = "mean_abs",
        n_samples: Optional[int] = 100,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores using LIME.
        
        Args:
            method: Aggregation method ('mean_abs', 'frequency')
            n_samples: Number of samples for global importance
            
        Returns:
            Dictionary of feature importance
        """
        global_explanation = self.explain_global(n_samples=n_samples)
        return global_explanation.feature_importance
    
    def explain_with_anchors(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get anchor-based explanation (high-precision rules).
        This is an alternative to standard LIME explanation.
        
        Args:
            instance: Instance to explain
            threshold: Precision threshold for anchors
            
        Returns:
            Dictionary with anchor rules
        """
        # This would require anchor-exp library
        # Placeholder for anchor functionality
        logger.info("Anchor explanations require additional setup")
        return {
            'message': 'Anchor explanations not implemented in this version',
            'alternative': 'Use standard LIME explanation'
        }
    
    @property
    def supports_global(self) -> bool:
        """LIME supports global through aggregation"""
        return True
    
    @property
    def supports_local(self) -> bool:
        """LIME is primarily for local explanations"""
        return True


# Register the LIME explainer
ExplainerFactory.register("lime", LIMEExplainer)


def create_lime_explainer(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> LIMEExplainer:
    """
    Convenience function to create a LIME explainer.
    
    Args:
        model: ML model to explain
        data: Training data (required for LIME)
        feature_names: Names of features
        **kwargs: Additional parameters
        
    Returns:
        LIMEExplainer instance
    """
    return LIMEExplainer(
        model=model,
        data=data,
        feature_names=feature_names,
        **kwargs
    )