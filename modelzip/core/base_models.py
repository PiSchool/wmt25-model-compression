#!/usr/bin/env python
"""
Base Model Classes for WMT25 Model Compression

Defines abstract interfaces and implementations for model wrappers
that can be used across different compression techniques.
"""

import abc
import torch
import psutil
import logging as LOG
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constrained_config import MODEL_CONFIG

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseModel(abc.ABC):
    """Abstract base class for all model wrappers"""
    
    def __init__(self, model_path: Path, config=None):
        self.model_path = Path(model_path)
        self.config = config
        self._model = None
        self._tokenizer = None
        
    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            self._model = self.load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer
    
    @abc.abstractmethod
    def load_model(self):
        """Load the model implementation"""
        pass
    
    @abc.abstractmethod 
    def load_tokenizer(self):
        """Load the tokenizer implementation"""
        pass
    
    def get_model_size(self) -> float:
        """Get model size in MB"""
        if self.model_path.is_dir():
            total_size = sum(f.stat().st_size for f in self.model_path.rglob("*") if f.is_file())
        else:
            total_size = self.model_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().rss / (1024 * 1024)


class HuggingFaceModel(BaseModel):
    """HuggingFace Transformers model wrapper"""
    
    def __init__(self, model_path: Path, config=None):
        super().__init__(model_path, config)
        self._is_hf_model_id = not Path(str(model_path)).exists()
        
    def load_model(self):
        """Load HuggingFace model with appropriate configuration"""
        # Build model loading arguments
        model_args = {
            "torch_dtype": MODEL_CONFIG.get("torch_dtype", "auto"),
            "device_map": MODEL_CONFIG.get("device_map", "auto"),
        }
        
        # Add valid model loading parameters from compression config
        if self.config and hasattr(self.config, 'compression_params'):
            # Only pass parameters that are valid for AutoModelForCausalLM.from_pretrained()
            valid_model_params = {
                "load_in_8bit", "load_in_4bit", "quantization_config", 
                "torch_dtype", "device_map", "attn_implementation",
                "low_cpu_mem_usage", "trust_remote_code"
            }
            
            for key, value in self.config.compression_params.items():
                if key in valid_model_params:
                    model_args[key] = value
            
        return AutoModelForCausalLM.from_pretrained(str(self.model_path), **model_args)
    
    def load_tokenizer(self):
        """Load HuggingFace tokenizer"""
        return AutoTokenizer.from_pretrained(str(self.model_path), use_fast=True)
    
    def get_model_size(self) -> float:
        """Get model size in MB - handles both local and HF model IDs"""
        if self._is_hf_model_id:
            # For HuggingFace model IDs, estimate from parameters
            try:
                model = self.model
                total_params = sum(p.numel() for p in model.parameters())
                # Estimate: 4 bytes per param (float32) or 2 bytes (float16)
                bytes_per_param = 2 if model.dtype == torch.float16 else 4
                size_mb = (total_params * bytes_per_param) / (1024 * 1024)
                return size_mb
            except Exception as e:
                LOG.warning(f"Could not determine model size for {self.model_path}: {e}")
                return 0.0
        else:
            # For local paths, use file system size
            return super().get_model_size()
    
    def get_model_params(self) -> int:
        """Get total number of parameters in the model
        
        Returns:
            int: Total number of parameters
        """
        try:
            model = self.model
            total_params = sum(p.numel() for p in model.parameters())
            return total_params
        except Exception as e:
            LOG.warning(f"Could not determine model parameters for {self.model_path}: {e}")
            return 0 