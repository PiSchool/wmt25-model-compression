#!/usr/bin/env python
"""
Base Compressor Interface for WMT25 Model Compression

Defines the abstract interface that all compression techniques must implement.
This ensures consistency and extensibility across different compression methods.
"""

import abc
import logging as LOG
from pathlib import Path
from .base_models import BaseModel

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseCompressor(abc.ABC):
    """Abstract base class for all model compression techniques
    
    This interface ensures that all compression methods follow the same
    contract and can be used interchangeably in the experiment framework.
    """
    
    def __init__(self, config):
        """Initialize compressor with experiment configuration
        
        Args:
            config: ExperimentConfig containing compression parameters
        """
        self.config = config
        
    @abc.abstractmethod
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Compress the model and save to output_path
        
        Args:
            model: The model to compress
            output_path: Directory where compressed model should be saved
            
        Returns:
            BaseModel: Wrapper for the compressed model
        """
        pass
    
    @abc.abstractmethod
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio between original and compressed models
        
        Args:
            original_model: The original uncompressed model
            compressed_model: The compressed model
            
        Returns:
            float: Compression ratio (original_size / compressed_size)
        """
        pass
    
    def _copy_run_script(self, output_path: Path):
        """Copy run.sh script for submission compatibility
        
        This is a common utility method that most compressors will need
        to ensure their output is compatible with the submission format.
        """
        run_script = output_path / "run.sh"
        baseline_script = Path(__file__).parent.parent / "run.sh"
        if baseline_script.exists():
            run_script.write_text(baseline_script.read_text())
            LOG.info(f"Copied run.sh to {run_script}")
        else:
            LOG.warning(f"Could not find baseline run.sh script at {baseline_script}")
    
    def _save_compression_metadata(self, output_path: Path, metadata: dict):
        """Save compression metadata for analysis and debugging
        
        Args:
            output_path: Directory where compressed model is saved
            metadata: Dictionary containing compression-specific metadata
        """
        import json
        metadata_file = output_path / "compression_metadata.json"
        
        # Add common metadata
        full_metadata = {
            "compression_method": self.config.compression_method,
            "compression_params": self.config.compression_params,
            "framework_version": "1.0.0",
            **metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        LOG.info(f"Saved compression metadata to {metadata_file}")


class NoCompressionCompressor(BaseCompressor):
    """Baseline compressor that applies no compression
    
    This is used for baseline comparisons and ensures that
    even "uncompressed" models follow the same interface.
    """
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Create a baseline copy of the model for comparison"""
        LOG.info("No compression applied - baseline model")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to output directory for consistency
        if str(model.model_path) != str(output_path):
            model.model.save_pretrained(output_path)
            model.tokenizer.save_pretrained(output_path)
            
            # Copy submission script
            self._copy_run_script(output_path)
            
            # Save metadata
            self._save_compression_metadata(output_path, {
                "original_size_mb": model.get_model_size(),
                "compression_applied": False
            })
            
            # Return new model wrapper for consistency
            from .base_models import HuggingFaceModel
            return HuggingFaceModel(output_path, self.config)
        
        return model
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Baseline has 1.0x compression ratio (no compression)"""
        return 1.0 