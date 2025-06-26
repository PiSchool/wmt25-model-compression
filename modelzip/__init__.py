"""
WMT25 Model Compression Framework

A comprehensive framework for model compression research and experiments,
specifically designed for the WMT25 constrained model compression task.

The framework follows clean architecture principles with modular components
organized by responsibility and technique.
"""

# Core framework components
from .core import (
    BaseModel, 
    HuggingFaceModel,
    BaseCompressor,
    ExperimentConfig,
    ExperimentResults,
    ExperimentRunner
)

# Compression techniques
from .compression import (
    QuantizationCompressor,
    PruningCompressor,
    DistillationCompressor,
    MixedPrecisionCompressor,
    get_compressor
)

# Evaluation components
from .evaluation import TranslationEvaluator

# Utilities
from .utils import setup_model, verify_model

# Data management
from .data_manager import DataManager

# Configuration
from .constrained_config import CONSTRAINED_TASK, COMPRESSION_CONFIG

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "BaseModel",
    "HuggingFaceModel", 
    "BaseCompressor",
    "ExperimentConfig",
    "ExperimentResults",
    "ExperimentRunner",
    
    # Compression techniques
    "QuantizationCompressor",
    "PruningCompressor",
    "DistillationCompressor", 
    "MixedPrecisionCompressor",
    "get_compressor",
    
    # Evaluation
    "TranslationEvaluator",
    
    # Utilities
    "setup_model",
    "verify_model",
    
    # Data management
    "DataManager",
    
    # Configuration
    "CONSTRAINED_TASK",
    "COMPRESSION_CONFIG"
]
