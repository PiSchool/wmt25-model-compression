"""
Layer Pruning Module for WMT25 Model Compression

This module provides layer-based pruning functionality using PruneMe's
similarity analysis and mergekit's layer merging capabilities.
"""

from .layer_similarity_analyzer import LayerSimilarityAnalyzer
from .layer_merger import LayerMerger
from .layer_pruning_compressor import LayerPruningCompressor, PruningConfig, PruningResult
from .pruning_compressor import PruningCompressor
from .config_loader import ConfigLoader, config_loader

__all__ = [
    "LayerSimilarityAnalyzer",
    "LayerMerger", 
    "LayerPruningCompressor",
    "PruningCompressor",
    "PruningConfig",
    "PruningResult",
    "ConfigLoader",
    "config_loader"
]

__version__ = "1.0.0" 