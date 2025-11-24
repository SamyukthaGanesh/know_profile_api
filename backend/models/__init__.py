"""
Models Module
Provides model-agnostic wrappers for any ML model.
"""

from .base_model import BaseModel
from .model_wrapper import ModelWrapper, create_model_wrapper

__all__ = [
    'BaseModel',
    'ModelWrapper',
    'create_model_wrapper',
]