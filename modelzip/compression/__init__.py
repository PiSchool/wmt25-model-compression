"""
Compression Techniques for WMT25 Model Compression Framework

This module contains all available compression methods, organized by technique.
Each compression method is in its own focused module for clarity and maintainability.
"""

from .quantization.quantization_compressor import QuantizationCompressor
from .pruning.pruning_compressor import PruningCompressor
from .distillation.distillation_compressor import DistillationCompressor
from .mixed_precision.mixed_precision_compressor import MixedPrecisionCompressor

# Import factory function
from .compression_factory import get_compressor, list_available_methods

__all__ = [
    "QuantizationCompressor",
    "PruningCompressor", 
    "DistillationCompressor",
    "MixedPrecisionCompressor",
    "get_compressor",
    "list_available_methods"
] 