"""
SHAP Explainer Implementation
Provides model-agnostic SHAP explanations for any ML model.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import shap
import logging
from .base_explainer import BaseExplainer, ExplanationResult, ExplanationType, ExplainerFactory

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainer for model-agnostic explanations.
    Automatically selects the best SHAP explainer based on the model type.
    """
    
    def _initialize(self):
        """Initialize the appropriate SHAP explainer based on model type"""
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Convert data to numpy if needed
        if self.data is not None:
            self.background_data = self.validate_input(self.data)
        else:
            self.background_data = None
        
        # Try to determine the best explainer
        self._select_explainer()
    
    def _select_explainer(self):
        """Automatically select the best SHAP explainer for the model"""
        model_type = type(self.model).__name__
        
        try:
            # Try TreeExplainer first (fastest for tree-based models)
            if self._is_tree_model():
                logger.info(f"Using TreeExplainer for {model_type}")
                self.explainer = shap.TreeExplainer(self.model)
                
            # Try LinearExplainer for linear models
            elif self._is_linear_model():
                logger.info(f"Using LinearExplainer for {model_type}")
                if self.background_data is not None:
                    self.explainer = shap.LinearExplainer(
                        self.model, 
                        self.background_data,
                        feature_dependence="independent"
                    )
                else:
                    raise ValueError("LinearExplainer requires background data")
                    
            # Try DeepExplainer for neural networks
            elif self._is_deep_model():
                logger.info(f"Using DeepExplainer for {model_type}")
                if self.background_data is not None:
                    # Sample background data if too large
                    background = self._sample_background(100)
                    self.explainer = shap.DeepExplainer(self.model, background)
                else:
                    raise ValueError("DeepExplainer requires background data")
                    
            # Default to KernelExplainer (works with any model but slower)
            else:
                logger.info(f"Using KernelExplainer for {model_type} (this may be slow)")
                if self.background_data is not None:
                    # Sample background data for efficiency
                    background = self._sample_background(100)
                    
                    # Create prediction function
                    if self.mode == "classification" and hasattr(self.model, 'predict_proba'):
                        predict_fn = self.model.predict_proba
                    else:
                        predict_fn = self.model.predict
                    
                    self.explainer = shap.KernelExplainer(predict_fn, background)
                else:
                    # Try Explainer (unified API)
                    self.explainer = shap.Explainer(self.model)
                    
        except Exception as e:
            logger.warning(f"Could not initialize specialized explainer: {e}")
            logger.info("Falling back to generic Explainer")
            # Fallback to generic explainer
            self.explainer = shap.Explainer(
                self.model,
                self.background_data if self.background_data is not None else shap.sample(self.data, 100)
            )
    
    def _is_tree_model(self) -> bool:
        """Check if model is tree-based"""
        tree_models = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'XGBClassifier', 'XGBRegressor', 'XGBRFClassifier', 'XGBRFRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor'
        ]
        return type(self.model).__name__ in tree_models
    
    def _is_linear_model(self) -> bool:
        """Check if model is linear"""
        linear_models = [
            'LinearRegression', 'LogisticRegression',
            'Ridge', 'RidgeClassifier', 'Lasso', 'ElasticNet',
            'SGDClassifier', 'SGDRegressor'
        ]
        return type(self.model).__name__ in linear_models
    
    def _is_deep_model(self) -> bool:
        """Check if model is a deep learning model"""
        # Check for common deep learning frameworks
        module_name = type(self.model).__module__
        return any(framework in module_name for framework in ['torch', 'tensorflow', 'keras'])
    
    def _sample_background(self, n_samples: int = 100) -> np.ndarray:
        """Sample background data for efficiency"""
        if self.background_data is None:
            return None
        
        if len(self.background_data) > n_samples:
            indices = np.random.choice(len(self.background_data), n_samples, replace=False)
            return self.background_data[indices]
        return self.background_data
    
    def explain_instance(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        check_additivity: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single prediction using SHAP.
        
        Args:
            instance: Single instance to explain
            check_additivity: Whether to check SHAP additivity
            **kwargs: Additional SHAP parameters
            
        Returns:
            ExplanationResult with SHAP values
        """
        # Convert to numpy
        instance_array = self.validate_input(instance)
        
        # Ensure 2D array
        if instance_array.ndim == 1:
            instance_array = instance_array.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance_array)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class classification
            if self.mode == "classification" and len(shap_values) > 1:
                # Get prediction to determine which class to explain
                prediction = self.model.predict(instance_array)[0]
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(instance_array)[0]
                    predicted_class = np.argmax(proba)
                    confidence = float(proba[predicted_class])
                    # Use SHAP values for predicted class
                    shap_values = shap_values[predicted_class][0]
                else:
                    predicted_class = int(prediction)
                    confidence = None
                    shap_values = shap_values[predicted_class][0]
            else:
                shap_values = shap_values[0]
        elif shap_values.ndim > 1:
            shap_values = shap_values[0]
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.explainer.expected_value[0])
            else:
                base_value = float(self.explainer.expected_value)
        else:
            base_value = None
        
        # Get prediction
        prediction = self.model.predict(instance_array)[0]
        
        # Create feature importance dictionary
        feature_importance = {}
        feature_values = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Handle both scalar and array SHAP values
            shap_val = shap_values[i]
            if isinstance(shap_val, (np.ndarray, list)) and len(shap_val) > 0:
                feature_importance[feature_name] = float(shap_val[0] if hasattr(shap_val, '__len__') else shap_val)
            else:
                feature_importance[feature_name] = float(shap_val)
            feature_values[feature_name] = float(instance_array[0][i])
        
        # Check additivity if requested
        if check_additivity and base_value is not None:
            shap_sum = base_value + sum(shap_values)
            if hasattr(self.model, 'predict_proba') and self.mode == "classification":
                model_output = self.model.predict_proba(instance_array)[0]
                if isinstance(model_output, np.ndarray):
                    model_output = model_output[predicted_class]
            else:
                model_output = self.model.predict(instance_array)[0]
            
            additivity_check = np.abs(shap_sum - model_output) < 0.01
            logger.info(f"SHAP additivity check: {additivity_check} (diff: {np.abs(shap_sum - model_output)})")
        
        return ExplanationResult(
            method="SHAP",
            explanation_type=ExplanationType.LOCAL,
            feature_importance=feature_importance,
            feature_values=feature_values,
            base_value=base_value,
            prediction=float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            confidence=confidence if 'confidence' in locals() else None,
            raw_output={
                'shap_values': shap_values,
                'expected_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None
            }
        )
    
    def explain_global(
        self,
        n_samples: Optional[int] = None,
        plot: bool = False,
        **kwargs
    ) -> ExplanationResult:
        """
        Explain the overall model behavior using SHAP.
        
        Args:
            n_samples: Number of samples to use
            plot: Whether to generate plot data
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult with global SHAP importance
        """
        if self.background_data is None:
            raise ValueError("Background data required for global explanations")
        
        # Sample data if specified
        if n_samples and n_samples < len(self.background_data):
            indices = np.random.choice(len(self.background_data), n_samples, replace=False)
            data_sample = self.background_data[indices]
        else:
            data_sample = self.background_data
        
        # Calculate SHAP values for all samples
        logger.info(f"Calculating SHAP values for {len(data_sample)} samples...")
        shap_values = self.explainer.shap_values(data_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class - use first class or average
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            # Handle both scalar and array SHAP values
            shap_val = mean_abs_shap[i]
            if isinstance(shap_val, (np.ndarray, list)) and len(shap_val) > 0:
                feature_importance[feature_name] = float(shap_val[0] if hasattr(shap_val, '__len__') else shap_val)
            else:
                feature_importance[feature_name] = float(shap_val)
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        # Generate plot data if requested
        plot_data = None
        if plot:
            plot_data = {
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'feature_names': self.feature_names,
                'data': data_sample.tolist() if isinstance(data_sample, np.ndarray) else data_sample
            }
        
        return ExplanationResult(
            method="SHAP",
            explanation_type=ExplanationType.GLOBAL,
            feature_importance=feature_importance,
            raw_output={
                'shap_values': shap_values,
                'mean_abs_shap': mean_abs_shap,
                'expected_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None
            },
            plot_data=plot_data
        )
    
    def get_feature_importance(
        self,
        method: str = "mean_abs",
        n_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores using SHAP.
        
        Args:
            method: 'mean_abs' for mean absolute SHAP values
            n_samples: Number of samples to use
            
        Returns:
            Dictionary of feature importance
        """
        global_explanation = self.explain_global(n_samples=n_samples)
        return global_explanation.feature_importance
    
    def get_interaction_values(
        self,
        instance: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get SHAP interaction values (if supported by the explainer).
        
        Args:
            instance: Instance to explain
            
        Returns:
            Interaction values matrix
        """
        if not hasattr(self.explainer, 'shap_interaction_values'):
            logger.warning("This explainer doesn't support interaction values")
            return None
        
        instance_array = self.validate_input(instance)
        if instance_array.ndim == 1:
            instance_array = instance_array.reshape(1, -1)
        
        return self.explainer.shap_interaction_values(instance_array)
    
    def get_waterfall_data(
        self,
        instance: Union[pd.DataFrame, np.ndarray],
        max_features: int = 10
    ) -> Dict[str, Any]:
        """
        Get data for SHAP waterfall plot.
        
        Args:
            instance: Instance to explain
            max_features: Maximum number of features to include
            
        Returns:
            Dictionary with waterfall plot data
        """
        explanation = self.explain_instance(instance)
        
        # Sort features by absolute importance
        sorted_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]
        
        return {
            'features': [f[0] for f in sorted_features],
            'values': [f[1] for f in sorted_features],
            'base_value': explanation.base_value,
            'prediction': explanation.prediction,
            'feature_values': {k: explanation.feature_values[k] for k, _ in sorted_features}
        }


# Register the SHAP explainer
ExplainerFactory.register("shap", SHAPExplainer)


def create_shap_explainer(
    model: Any,
    data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    **kwargs
) -> SHAPExplainer:
    """
    Convenience function to create a SHAP explainer.
    
    Args:
        model: ML model to explain
        data: Background data
        **kwargs: Additional parameters
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model=model, data=data, **kwargs)