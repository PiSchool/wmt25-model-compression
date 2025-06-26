#!/usr/bin/env python
"""
Compression Factory for WMT25 Model Compression

Factory pattern implementation for creating compression instances.
Provides clean interface for obtaining different compression techniques.
"""

import logging as LOG
from typing import Union

from ..core.base_compressor import BaseCompressor, NoCompressionCompressor
from ..core.experiment_config import ExperimentConfig

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_compressor(config: ExperimentConfig) -> BaseCompressor:
    """Factory function to create appropriate compressor instance
    
    Args:
        config: Experiment configuration containing compression method
        
    Returns:
        BaseCompressor: Appropriate compressor instance
        
    Raises:
        ValueError: If compression method is not supported
    """
    method = config.compression_method.lower()
    
    # Quantization methods
    if method.startswith("quantization") or method.endswith(("8bit", "4bit")):
        from .quantization.quantization_compressor import QuantizationCompressor
        return QuantizationCompressor(config)
    
    # Pruning methods
    elif method.startswith("pruning") or "prune" in method:
        from .pruning.pruning_compressor import PruningCompressor
        return PruningCompressor(config)
    
    # Distillation methods
    elif method.startswith("distillation") or "distill" in method:
        from .distillation.distillation_compressor import DistillationCompressor
        return DistillationCompressor(config)
    
    # Mixed precision methods
    elif method.startswith("mixed_precision") or method in ["fp16", "bf16", "half"]:
        from .mixed_precision.mixed_precision_compressor import MixedPrecisionCompressor
        return MixedPrecisionCompressor(config)
    
    # Baseline (no compression)
    elif method in ["baseline", "none", "no_compression"]:
        return NoCompressionCompressor(config)
    
    else:
        # Try to map common method names
        method_mapping = {
            "8bit": "quantization_8bit",
            "4bit": "quantization_4bit", 
            "magnitude": "pruning_magnitude",
            "structured": "pruning_structured",
            "knowledge": "distillation_response",
            "fp16": "mixed_precision_fp16",
            "bf16": "mixed_precision_bf16"
        }
        
        mapped_method = method_mapping.get(method)
        if mapped_method:
            LOG.info(f"Mapping '{method}' to '{mapped_method}'")
            # Create new config with mapped method
            new_config = ExperimentConfig(
                name=config.name,
                compression_method=mapped_method,
                lang_pair=config.lang_pair,
                base_model=config.base_model,
                compression_params=config.compression_params,
                training_params=config.training_params,
                eval_params=config.eval_params
            )
            return get_compressor(new_config)
        
        raise ValueError(
            f"Unsupported compression method: {method}. "
            f"Supported methods: quantization_*, pruning_*, distillation_*, "
            f"mixed_precision_*, baseline"
        )


def list_available_methods() -> dict:
    """List all available compression methods with descriptions
    
    Returns:
        dict: Dictionary mapping method names to descriptions
    """
    return {
        "baseline": "No compression (baseline comparison)",
        "quantization_8bit": "8-bit quantization using BitsAndBytes",
        "quantization_4bit": "4-bit quantization using BitsAndBytes", 
        "pruning_magnitude": "Magnitude-based unstructured pruning",
        "pruning_structured": "Structured pruning (channel/filter removal)",
        "distillation_response": "Response-based knowledge distillation",
        "distillation_feature": "Feature-based knowledge distillation",
        "mixed_precision_fp16": "Half precision (FP16) compression",
        "mixed_precision_bf16": "Brain floating point (BF16) compression"
    }


def validate_compression_method(method: str) -> bool:
    """Validate if a compression method is supported
    
    Args:
        method: Compression method name
        
    Returns:
        bool: True if method is supported, False otherwise
    """
    try:
        # Create dummy config to test method validation
        from ..core.experiment_config import ExperimentConfig
        dummy_config = ExperimentConfig(
            name="validation_test",
            compression_method=method,
            lang_pair="ces-deu"
        )
        get_compressor(dummy_config)
        return True
    except ValueError:
        return False


def get_method_requirements(method: str) -> dict:
    """Get requirements and dependencies for a compression method
    
    Args:
        method: Compression method name
        
    Returns:
        dict: Requirements including hardware, software dependencies
    """
    requirements = {
        "baseline": {
            "hardware": "Any",
            "dependencies": ["transformers", "torch"],
            "gpu_memory": "Model size",
            "special_notes": "No compression applied"
        },
        "quantization_8bit": {
            "hardware": "CUDA GPU",
            "dependencies": ["transformers", "torch", "bitsandbytes"],
            "gpu_memory": "~50% of original model",
            "special_notes": "Requires BitsAndBytes library"
        },
        "quantization_4bit": {
            "hardware": "CUDA GPU", 
            "dependencies": ["transformers", "torch", "bitsandbytes"],
            "gpu_memory": "~25% of original model",
            "special_notes": "Requires BitsAndBytes library"
        },
        "pruning_magnitude": {
            "hardware": "Any",
            "dependencies": ["transformers", "torch"],
            "gpu_memory": "Model size (during compression)",
            "special_notes": "Removes weights based on magnitude"
        },
        "pruning_structured": {
            "hardware": "Any",
            "dependencies": ["transformers", "torch"],
            "gpu_memory": "Model size (during compression)",
            "special_notes": "Removes entire channels/filters"
        },
        "distillation_response": {
            "hardware": "High memory for teacher + student",
            "dependencies": ["transformers", "torch", "datasets"],
            "gpu_memory": "2x model size (teacher + student)",
            "special_notes": "Requires training data and time"
        },
        "mixed_precision_fp16": {
            "hardware": "Any (optimized for modern GPUs)",
            "dependencies": ["transformers", "torch"],
            "gpu_memory": "~50% of original model",
            "special_notes": "Native support on most hardware"
        },
        "mixed_precision_bf16": {
            "hardware": "Modern GPUs with BF16 support",
            "dependencies": ["transformers", "torch"],
            "gpu_memory": "~50% of original model",
            "special_notes": "Better numerical stability than FP16"
        }
    }
    
    return requirements.get(method, {
        "hardware": "Unknown",
        "dependencies": ["Unknown"],
        "gpu_memory": "Unknown",
        "special_notes": "Method not found in requirements database"
    }) 