#!/usr/bin/env python
"""
Experiment Configuration Classes for WMT25 Model Compression

Defines the data structures used to configure and store results
from compression experiments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from ..constrained_config import CONSTRAINED_TASK, TRAINING_CONFIG, EVAL_CONFIG


@dataclass
class ExperimentConfig:
    """Configuration for a compression experiment
    
    This dataclass contains all the parameters needed to run
    a complete compression experiment, including model settings,
    compression parameters, and evaluation configuration.
    """
    
    # Required fields
    name: str
    compression_method: str
    lang_pair: str
    
    # Optional fields with defaults
    base_model: str = CONSTRAINED_TASK["base_model"]
    compression_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    eval_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default parameters if not provided"""
        if not self.compression_params:
            self.compression_params = {}
        if not self.training_params:
            self.training_params = TRAINING_CONFIG.copy()
        if not self.eval_params:
            self.eval_params = EVAL_CONFIG.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "name": self.name,
            "compression_method": self.compression_method,
            "lang_pair": self.lang_pair,
            "base_model": self.base_model,
            "compression_params": self.compression_params,
            "training_params": self.training_params,
            "eval_params": self.eval_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary"""
        return cls(**data)


@dataclass 
class ExperimentResults:
    """Results from a compression experiment
    
    Contains all metrics and metadata from running a compression
    experiment, including performance metrics and quality scores.
    """
    
    # Experiment configuration
    config: ExperimentConfig
    
    # Performance metrics
    model_size_mb: float
    memory_usage_mb: float
    inference_time_ms: float
    compression_ratio: float
    
    # Quality metrics
    quality_scores: Dict[str, float]
    
    # Metadata
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            "config": self.config.to_dict(),
            "model_size_mb": self.model_size_mb,
            "memory_usage_mb": self.memory_usage_mb,
            "inference_time_ms": self.inference_time_ms,
            "compression_ratio": self.compression_ratio,
            "quality_scores": self.quality_scores,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResults":
        """Create results from dictionary"""
        config_data = data.pop("config")
        config = ExperimentConfig.from_dict(config_data)
        return cls(config=config, **data)
    
    def get_efficiency_score(self) -> float:
        """Calculate a combined efficiency score"""
        # Combine compression ratio and inference speed
        speed_factor = 1000 / self.inference_time_ms  # Higher is better
        return self.compression_ratio * speed_factor
    
    def get_quality_score(self) -> float:
        """Get primary quality score (CHRF if available)"""
        return self.quality_scores.get("chrf", 0.0)


def create_experiment_config(
    name: str, 
    compression_method: str, 
    lang_pair: str,
    base_model: str = None,
    **kwargs
) -> ExperimentConfig:
    """Helper function to create experiment configurations
    
    Args:
        name: Unique name for the experiment
        compression_method: Type of compression to apply
        lang_pair: Language pair for translation task
        base_model: Model to compress (defaults to constrained task model)
        **kwargs: Additional parameters (compression_params, etc.)
        
    Returns:
        ExperimentConfig: Configured experiment
    """
    if base_model is None:
        base_model = CONSTRAINED_TASK["base_model"]
    
    return ExperimentConfig(
        name=name,
        compression_method=compression_method,
        lang_pair=lang_pair,
        base_model=base_model,
        **kwargs
    ) 