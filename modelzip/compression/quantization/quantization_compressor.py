#!/usr/bin/env python
"""
Quantization Compression for WMT25 Model Compression

Implements quantization-based compression using BitsAndBytesConfig
with support for 8-bit and 4-bit quantization for CUDA GPUs.
"""

import torch
import logging as LOG
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from ...core.base_compressor import BaseCompressor
from ...core.base_models import BaseModel, HuggingFaceModel
from ...core.experiment_config import ExperimentConfig
from ...constrained_config import COMPRESSION_CONFIG

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class QuantizationCompressor(BaseCompressor):
    """Quantization-based compression using BitsAndBytesConfig
    
    Supports both 8-bit and 4-bit quantization optimized for CUDA GPUs.
    """
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Apply quantization compression to the model
        
        Args:
            model: Model to compress
            output_path: Directory to save compressed model
            
        Returns:
            BaseModel: Compressed model wrapper
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy tokenizer first
        model.tokenizer.save_pretrained(output_path)
        
        # Determine quantization type from compression method
        compression_type = self.config.compression_method.split("_")[-1]  # "8bit" or "4bit"
        
        # Apply quantization
        quantized_model = self._apply_quantization(model, compression_type)
        
        # Save quantized model
        quantized_model.save_pretrained(output_path)
        
        # Copy submission script and save metadata
        self._copy_run_script(output_path)
        self._save_compression_metadata(output_path, {
            "quantization_type": compression_type,
            "original_size_mb": model.get_model_size()
        })
        
        # Create new model wrapper
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_quantized",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair,
            compression_params=self._get_applied_params(compression_type)
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def _apply_quantization(self, model: BaseModel, compression_type: str):
        """Apply quantization using BitsAndBytesConfig
        
        Args:
            model: Model to quantize
            compression_type: "8bit" or "4bit"
            
        Returns:
            Quantized model using bitsandbytes
        """
        try:
            quant_config = COMPRESSION_CONFIG["quantization"][compression_type]
            
            quantized_model = AutoModelForCausalLM.from_pretrained(
                str(model.model_path),
                quantization_config=BitsAndBytesConfig(**quant_config),
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            LOG.info(f"Applied {compression_type} quantization using BitsAndBytesConfig")
            return quantized_model
            
        except ImportError as e:
            raise RuntimeError(
                "BitsAndBytes is required for quantization. "
                "Install with: pip install bitsandbytes"
            ) from e
    
    def _get_applied_params(self, compression_type: str) -> dict:
        """Get the parameters that were actually applied"""
        params = {"compression_type": compression_type}
        params.update(COMPRESSION_CONFIG["quantization"][compression_type])
        return params
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio for quantized model
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Quantized model
            
        Returns:
            float: Compression ratio (original_size / compressed_size)
        """
        original_size = original_model.get_model_size()
        compressed_size = compressed_model.get_model_size()
        return original_size / compressed_size if compressed_size > 0 else 0.0 