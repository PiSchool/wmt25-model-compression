#!/usr/bin/env python
"""
Pruning Compression for WMT25 Model Compression

Implements structured and unstructured pruning methods for model compression.
Supports various sparsity levels and pruning strategies.
"""

import torch
import torch.nn.utils.prune as prune
import logging as LOG
from pathlib import Path

from ...core.base_compressor import BaseCompressor
from ...core.base_models import BaseModel, HuggingFaceModel
from ...core.experiment_config import ExperimentConfig
from ...constrained_config import COMPRESSION_CONFIG

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PruningCompressor(BaseCompressor):
    """Pruning-based compression using structured/unstructured pruning
    
    Supports magnitude-based pruning with configurable sparsity levels
    and different pruning strategies (global, layer-wise, etc.).
    """
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Apply pruning compression to the model
        
        Args:
            model: Model to compress
            output_path: Directory to save compressed model
            
        Returns:
            BaseModel: Compressed model wrapper
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy tokenizer first
        model.tokenizer.save_pretrained(output_path)
        
        # Get pruning configuration
        pruning_config = self.config.compression_params.get(
            "pruning", 
            COMPRESSION_CONFIG["pruning"]["magnitude"]
        )
        
        # Apply pruning
        pruned_model = self._apply_pruning(model, pruning_config)
        
        # Save pruned model
        pruned_model.save_pretrained(output_path)
        
        # Copy submission script and save metadata
        self._copy_run_script(output_path)
        self._save_compression_metadata(output_path, {
            "pruning_config": pruning_config,
            "sparsity_level": pruning_config.get("amount", 0.1),
            "pruning_method": pruning_config.get("method", "magnitude"),
            "original_size_mb": model.get_model_size()
        })
        
        # Create new model wrapper
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_pruned",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair,
            compression_params={"pruning": pruning_config}
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def _apply_pruning(self, model: BaseModel, pruning_config: dict):
        """Apply pruning to model based on configuration
        
        Args:
            model: Model to prune
            pruning_config: Pruning configuration parameters
            
        Returns:
            Pruned model
        """
        pruned_model = model.model
        sparsity = pruning_config.get("amount", 0.1)
        method = pruning_config.get("method", "magnitude")
        
        LOG.info(f"Applying {method} pruning with {sparsity:.1%} sparsity")
        
        if method == "magnitude":
            self._apply_magnitude_pruning(pruned_model, sparsity)
        elif method == "structured":
            self._apply_structured_pruning(pruned_model, sparsity)
        else:
            LOG.warning(f"Unknown pruning method: {method}, using magnitude")
            self._apply_magnitude_pruning(pruned_model, sparsity)
        
        return pruned_model
    
    def _apply_magnitude_pruning(self, model, sparsity: float):
        """Apply magnitude-based unstructured pruning
        
        Args:
            model: Model to prune
            sparsity: Fraction of weights to prune (0.0 to 1.0)
        """
        parameters_to_prune = []
        
        # Collect all linear layers for pruning
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global magnitude pruning
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            LOG.info(f"Applied magnitude pruning to {len(parameters_to_prune)} layers")
        else:
            LOG.warning("No linear layers found for pruning")
    
    def _apply_structured_pruning(self, model, sparsity: float):
        """Apply structured pruning (channel/filter pruning)
        
        Args:
            model: Model to prune
            sparsity: Fraction of structures to prune (0.0 to 1.0)
        """
        # For structured pruning, we remove entire channels/filters
        # This is more complex and requires careful handling of dimensions
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Apply structured pruning to linear layers
                # Prune entire neurons (output dimensions)
                if hasattr(module, 'out_features'):
                    n_prune = int(module.out_features * sparsity)
                    if n_prune > 0:
                        prune.structured(
                            module, 
                            name='weight', 
                            amount=n_prune, 
                            dim=0,  # Prune output dimensions
                            method=prune.LnStructured(n=2, dim=1)
                        )
                        # Make permanent
                        prune.remove(module, 'weight')
        
        LOG.info(f"Applied structured pruning with {sparsity:.1%} sparsity")
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio for pruned model
        
        For pruning, we calculate based on the number of zero parameters
        vs. total parameters, since file size might not change significantly.
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Pruned model
            
        Returns:
            float: Compression ratio based on sparsity
        """
        # Count non-zero parameters in compressed model
        total_params = 0
        non_zero_params = 0
        
        for param in compressed_model.model.parameters():
            total_params += param.numel()
            non_zero_params += torch.count_nonzero(param).item()
        
        # Calculate effective compression ratio
        sparsity = 1 - (non_zero_params / total_params) if total_params > 0 else 0
        effective_ratio = 1 / (1 - sparsity) if sparsity < 1.0 else 1.0
        
        LOG.info(f"Model sparsity: {sparsity:.1%}, effective compression: {effective_ratio:.2f}x")
        return effective_ratio 