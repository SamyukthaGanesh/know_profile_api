"""
Explainability Module
Provides SHAP, LIME, Integrated Gradients, and Anchors explanations.
"""

from .base_explainer import (
    BaseExplainer,
    ExplanationResult,
    ExplanationType,
    ExplainerOutput,
    ExplainerFactory
)
from .shap_explainer import SHAPExplainer, create_shap_explainer
from .lime_explainer import LIMEExplainer, create_lime_explainer
from .integrated_gradients import IntegratedGradientsExplainer, create_ig_explainer
from .anchors_explainer import AnchorsExplainer, create_anchors_explainer

__all__ = [
    # Base classes
    'BaseExplainer',
    'ExplanationResult',
    'ExplanationType',
    'ExplainerOutput',
    'ExplainerFactory',
    
    # Explainer implementations
    'SHAPExplainer',
    'LIMEExplainer',
    'IntegratedGradientsExplainer',
    'AnchorsExplainer',
    
    # Convenience functions
    'create_shap_explainer',
    'create_lime_explainer',
    'create_ig_explainer',
    'create_anchors_explainer',
]