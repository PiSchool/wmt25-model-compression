#!/usr/bin/env python
"""
Mixed Precision Compression for WMT25 Model Compression

Implements mixed precision compression using different floating point
formats (FP16, BF16) for model size reduction with minimal quality loss.
"""

import torch
import logging as LOG
from pathlib import Path
from transformers import AutoModelForCausalLM

from ...core.base_compressor import BaseCompressor
from ...core.base_models import BaseModel, HuggingFaceModel
from ...core.experiment_config import ExperimentConfig
from ...constrained_config import COMPRESSION_CONFIG

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MixedPrecisionCompressor(BaseCompressor):
    """Mixed precision compression using different floating point formats
    
    Supports FP16 (half precision) and BF16 (bfloat16) compression
    with automatic optimization for different hardware platforms.
    """
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Apply mixed precision compression to the model
        
        Args:
            model: Model to compress
            output_path: Directory to save compressed model
            
        Returns:
            BaseModel: Compressed model wrapper
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy tokenizer first
        model.tokenizer.save_pretrained(output_path)
        
        # Extract precision type from compression method (e.g., "mixed_precision_fp16" -> "fp16")
        precision_type = self.config.compression_method.split("_")[-1]  # "fp16" or "bf16"
        
        # Get mixed precision configuration
        precision_config = self.config.compression_params.get(
            "mixed_precision",
            COMPRESSION_CONFIG["mixed_precision"].get(precision_type, COMPRESSION_CONFIG["mixed_precision"]["fp16"])
        )
        
        # Apply mixed precision compression
        compressed_model = self._apply_mixed_precision(model, precision_config)
        
        # Save compressed model
        compressed_model.save_pretrained(output_path)
        
        # Copy submission script and save metadata
        self._copy_run_script(output_path)
        self._save_compression_metadata(output_path, {
            "precision_config": precision_config,
            "precision_type": precision_config.get("dtype", "fp16"),
            "original_size_mb": model.get_model_size(),
            "hardware_optimization": self._get_hardware_optimization()
        })
        
        # Create new model wrapper
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_mixed_precision",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair,
            compression_params={"mixed_precision": precision_config}
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def _apply_mixed_precision(self, model: BaseModel, precision_config: dict):
        """Apply mixed precision compression based on configuration
        
        Args:
            model: Model to compress
            precision_config: Precision configuration parameters
            
        Returns:
            Model with mixed precision applied
        """
        dtype_str = precision_config.get("dtype", "fp16")
        
        # Map string to torch dtype
        dtype_mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        
        target_dtype = dtype_mapping.get(dtype_str, torch.float16)
        
        LOG.info(f"Applying mixed precision compression with {dtype_str}")
        
        # Load model with specified precision
        compressed_model = AutoModelForCausalLM.from_pretrained(
            str(model.model_path),
            torch_dtype=target_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Apply additional optimizations if specified
        if precision_config.get("optimize_for_inference", True):
            compressed_model = self._optimize_for_inference(compressed_model, precision_config)
        
        return compressed_model
    
    def _optimize_for_inference(self, model, precision_config: dict):
        """Apply additional optimizations for inference
        
        Args:
            model: Model to optimize
            precision_config: Optimization configuration
            
        Returns:
            Optimized model
        """
        # Apply torch.jit compilation if requested
        if precision_config.get("use_torch_compile", False) and hasattr(torch, 'compile'):
            try:
                LOG.info("Applying torch.compile optimization")
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                LOG.warning(f"torch.compile failed: {e}")
        
        # Apply ONNX optimization if requested
        if precision_config.get("convert_to_onnx", False):
            model = self._convert_to_onnx(model, precision_config)
        
        # Set model to evaluation mode for inference
        model.eval()
        
        return model
    
    def _convert_to_onnx(self, model, precision_config: dict):
        """Convert model to ONNX format for optimization
        
        Args:
            model: Model to convert
            precision_config: Configuration parameters
            
        Returns:
            ONNX-optimized model or original model if conversion fails
        """
        try:
            # This is a placeholder for ONNX conversion
            # In practice, this would require more complex setup
            LOG.info("ONNX conversion placeholder - returning original model")
            return model
        except Exception as e:
            LOG.warning(f"ONNX conversion failed: {e}")
            return model
    
    def _get_hardware_optimization(self) -> dict:
        """Get hardware-specific optimization information
        
        Returns:
            Dictionary with hardware optimization details
        """
        optimization_info = {
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            optimization_info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            })
        
        return optimization_info
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio for mixed precision model
        
        Mixed precision typically provides 2x compression for FP16
        and varies for other precision formats.
        
        Args:
            original_model: Original model (assumed FP32)
            compressed_model: Mixed precision model
            
        Returns:
            float: Compression ratio based on precision format
        """
        # Get actual file sizes
        original_size = original_model.get_model_size()
        compressed_size = compressed_model.get_model_size()
        
        if compressed_size > 0:
            actual_ratio = original_size / compressed_size
        else:
            # Estimate based on precision type if file size not available
            precision_config = self.config.compression_params.get("mixed_precision", {})
            dtype_str = precision_config.get("dtype", "fp16")
            
            if dtype_str == "fp16":
                actual_ratio = 2.0  # FP16 is half the size of FP32
            elif dtype_str == "bf16":
                actual_ratio = 2.0  # BF16 is also half the size of FP32
            else:
                actual_ratio = 1.0  # FP32 or unknown
        
        LOG.info(f"Mixed precision compression ratio: {actual_ratio:.2f}x")
        return actual_ratio 