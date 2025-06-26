"""
Core classes and interfaces for WMT25 Model Compression Framework

This module contains the fundamental abstractions and base classes
that define the compression framework architecture.
"""

from .base_models import BaseModel, HuggingFaceModel
from .base_compressor import BaseCompressor
from .experiment_config import ExperimentConfig, ExperimentResults, create_experiment_config
from .experiment_runner import ExperimentRunner

__all__ = [
    "BaseModel",
    "HuggingFaceModel", 
    "BaseCompressor",
    "ExperimentConfig",
    "ExperimentResults",
    "ExperimentRunner",
    "create_experiment_config"
] 