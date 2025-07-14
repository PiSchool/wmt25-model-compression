#!/usr/bin/env python
"""
Layer Pruning Compressor for WMT25 Model Compression

Main orchestrator that combines layer similarity analysis and layer merging
to perform complete layer-based pruning of transformer models.
"""

import os
import sys
import logging as LOG
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .layer_similarity_analyzer import LayerSimilarityAnalyzer
from .layer_merger import LayerMerger
from .config_loader import config_loader

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class PruningConfig:
    """Configuration for layer pruning process"""
    model_path: str
    output_path: str
    dataset: str = None  # Will be set to config default
    dataset_column: str = None  # Will be set to config default
    batch_size: int = None  # Will be set to config default
    max_length: int = None  # Will be set to config default
    layers_to_skip: int = None  # Will be set to config default
    dataset_size: int = None  # Will be set to config default
    dataset_subset: str = None  # Will be set to config default
    merge_method: str = None  # Will be set to config default
    temp_dir: Optional[str] = None
    
    def __post_init__(self):
        """Set default values from config if not provided"""
        if self.dataset is None:
            self.dataset = config_loader.get_default_dataset()
        if self.dataset_column is None:
            self.dataset_column = config_loader.get_default_dataset_column()
        if self.batch_size is None:
            self.batch_size = config_loader.get_default_batch_size()
        if self.max_length is None:
            self.max_length = config_loader.get_default_max_length()
        if self.layers_to_skip is None:
            self.layers_to_skip = config_loader.get_default_layers_to_skip()
        if self.dataset_size is None:
            self.dataset_size = config_loader.get_default_dataset_size()
        if self.dataset_subset is None:
            self.dataset_subset = config_loader.get_default_dataset_subset()
        if self.merge_method is None:
            self.merge_method = config_loader.get_default_merge_method()
        if self.temp_dir is None:
            self.temp_dir = config_loader.get_temp_dir()


@dataclass
class PruningResult:
    """Results from the layer pruning process"""
    success: bool
    original_model_path: str
    pruned_model_path: Optional[str]
    removed_layers: Optional[Tuple[int, int]]
    layer_distance: Optional[float]
    model_size_reduction: Optional[float]
    analysis_results: Optional[Dict]
    merge_results: Optional[Dict]
    error_message: Optional[str] = None


class LayerPruningCompressor:
    """Main orchestrator for layer-based pruning using similarity analysis and merging
    
    This class provides a complete workflow for:
    1. Analyzing layer similarity to identify redundant layers
    2. Merging the model by removing the identified layers
    3. Validating the pruned model
    4. Providing comprehensive results and metrics
    """
    
    def __init__(self, pruneme_path: str = None):
        """Initialize the layer pruning compressor
        
        Args:
            pruneme_path: Path to the PruneMe repository (uses config default if None)
        """
        if pruneme_path is None:
            pruneme_path = config_loader.get_pruneme_path()
        self.pruneme_path = Path(pruneme_path)
        self.analyzer = LayerSimilarityAnalyzer(pruneme_path)
        self.merger = LayerMerger(pruneme_path)
        
        LOG.info("LayerPruningCompressor initialized successfully")
    
    def compress_model(self, config: PruningConfig) -> PruningResult:
        """Perform complete layer pruning workflow
        
        Args:
            config: Configuration for the pruning process
            
        Returns:
            PruningResult: Complete results of the pruning process
        """
        LOG.info(f"Starting layer pruning compression for model: {config.model_path}")
        
        # Create temporary directory if not specified
        temp_dir = Path(config.temp_dir) if config.temp_dir else Path("temp_pruning")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Analyze layer similarity
            LOG.info("Step 1: Analyzing layer similarity...")
            analysis_success, analysis_results = self.analyzer.analyze_layers(
                model_path=config.model_path,
                dataset=config.dataset,
                dataset_column=config.dataset_column,
                batch_size=config.batch_size,
                max_length=config.max_length,
                layers_to_skip=config.layers_to_skip,
                dataset_size=config.dataset_size,
                dataset_subset=config.dataset_subset,
                output_dir=temp_dir / "analysis"
            )
            
            if not analysis_success:
                return PruningResult(
                    success=False,
                    original_model_path=config.model_path,
                    pruned_model_path=None,
                    removed_layers=None,
                    layer_distance=None,
                    model_size_reduction=None,
                    analysis_results=None,
                    merge_results=None,
                    error_message="Layer similarity analysis failed"
                )
            
            # Step 2: Extract layer recommendation
            start_layer, end_layer, distance = self.analyzer.get_layer_recommendation(analysis_results)
            LOG.info(f"Recommended to remove layers {start_layer} to {end_layer} (distance: {distance:.4f})")
            
            # Step 3: Merge layers (remove the identified layers)
            LOG.info("Step 2: Merging layers (removing redundant layers)...")
            merge_success, merge_results = self.merger.merge_layers(
                model_path=config.model_path,
                output_path=config.output_path,
                start_layer=start_layer,
                end_layer=end_layer,
                merge_method=config.merge_method,
                temp_dir=temp_dir / "merge"
            )
            
            if not merge_success:
                return PruningResult(
                    success=False,
                    original_model_path=config.model_path,
                    pruned_model_path=None,
                    removed_layers=(start_layer, end_layer),
                    layer_distance=distance,
                    model_size_reduction=None,
                    analysis_results=analysis_results,
                    merge_results=None,
                    error_message="Layer merging failed"
                )
            
            # Step 4: Calculate model size reduction
            size_reduction = self._calculate_size_reduction(config.model_path, config.output_path)
            
            LOG.info(f"Layer pruning completed successfully!")
            LOG.info(f"Removed layers {start_layer} to {end_layer}")
            LOG.info(f"Model size reduction: {size_reduction:.2f}%")
            
            return PruningResult(
                success=True,
                original_model_path=config.model_path,
                pruned_model_path=config.output_path,
                removed_layers=(start_layer, end_layer),
                layer_distance=distance,
                model_size_reduction=size_reduction,
                analysis_results=analysis_results,
                merge_results=merge_results
            )
            
        except Exception as e:
            LOG.error(f"Unexpected error during layer pruning: {e}")
            return PruningResult(
                success=False,
                original_model_path=config.model_path,
                pruned_model_path=None,
                removed_layers=None,
                layer_distance=None,
                model_size_reduction=None,
                analysis_results=None,
                merge_results=None,
                error_message=str(e)
            )
        finally:
            # Clean up temporary directory if it was auto-created
            if not config.temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                LOG.info(f"Cleaned up temporary directory: {temp_dir}")
    
    def _calculate_size_reduction(self, original_path: str, pruned_path: str) -> float:
        """Calculate the percentage reduction in model size
        
        Args:
            original_path: Path to original model
            pruned_path: Path to pruned model
            
        Returns:
            float: Percentage reduction in model size
        """
        try:
            original_size = self._get_model_size(original_path)
            pruned_size = self._get_model_size(pruned_path)
            
            if original_size == 0:
                return 0.0
            
            reduction = ((original_size - pruned_size) / original_size) * 100
            return round(reduction, 2)
            
        except Exception as e:
            LOG.warning(f"Could not calculate size reduction: {e}")
            return 0.0
    
    def _get_model_size(self, model_path: str) -> int:
        """Get the total size of a model in bytes
        
        Args:
            model_path: Path to the model
            
        Returns:
            int: Total size in bytes
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            return 0
        
        total_size = 0
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def get_compression_summary(self, result: PruningResult) -> Dict:
        """Generate a comprehensive summary of the compression process
        
        Args:
            result: Results from compress_model()
            
        Returns:
            Dict: Summary of the compression process
        """
        if not result.success:
            return {
                "status": "failed",
                "error": result.error_message,
                "original_model": result.original_model_path
            }
        
        summary = {
            "status": "success",
            "original_model": result.original_model_path,
            "pruned_model": result.pruned_model_path,
            "compression_details": {
                "removed_layers": f"{result.removed_layers[0]} to {result.removed_layers[1]}",
                "layer_distance": f"{result.layer_distance:.4f}",
                "size_reduction": f"{result.model_size_reduction:.2f}%"
            },
            "analysis_summary": result.analysis_results.get("recommendation", ""),
            "merge_summary": result.merge_results.get("summary", "") if result.merge_results else ""
        }
        
        return summary
    
    def validate_pruned_model(self, result: PruningResult) -> bool:
        """Validate that the pruned model is usable
        
        Args:
            result: Results from compress_model()
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        if not result.success or not result.pruned_model_path:
            return False
        
        try:
            # Check if the model directory exists
            model_dir = Path(result.pruned_model_path)
            if not model_dir.exists():
                LOG.error(f"Pruned model directory does not exist: {result.pruned_model_path}")
                return False
            
            # Check for essential model files
            essential_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = []
            
            for file_name in essential_files:
                if not (model_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                LOG.error(f"Missing essential model files: {missing_files}")
                return False
            
            # Try to load the model (basic validation)
            try:
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(result.pruned_model_path)
                model = AutoModel.from_pretrained(result.pruned_model_path)
                LOG.info("Pruned model loaded successfully for validation")
                return True
            except Exception as e:
                LOG.error(f"Failed to load pruned model for validation: {e}")
                return False
                
        except Exception as e:
            LOG.error(f"Error during model validation: {e}")
            return False


def main():
    """Example usage of the LayerPruningCompressor"""
    # Example configuration
    config = PruningConfig(
        model_path="CohereLabs/aya-expanse-8b",
        output_path="./pruned_aya8b",
        dataset="arcee-ai/sec-data-mini",
        dataset_size=1000,
        batch_size=4
    )
    
    # Initialize compressor
    compressor = LayerPruningCompressor()
    
    # Run compression
    result = compressor.compress_model(config)
    
    # Print results
    if result.success:
        print("✅ Layer pruning completed successfully!")
        summary = compressor.get_compression_summary(result)
        print(f"Removed layers: {summary['compression_details']['removed_layers']}")
        print(f"Size reduction: {summary['compression_details']['size_reduction']}")
        print(f"Pruned model saved to: {result.pruned_model_path}")
    else:
        print(f"❌ Layer pruning failed: {result.error_message}")


if __name__ == "__main__":
    main() 