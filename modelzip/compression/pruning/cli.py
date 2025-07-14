#!/usr/bin/env python
"""
Command Line Interface for Layer Pruning Compressor

Provides a simple CLI to run layer-based pruning on transformer models.
"""

import argparse
import json
import logging as LOG
from pathlib import Path
from typing import Optional

from .layer_pruning_compressor import LayerPruningCompressor, PruningConfig, PruningResult
from .config_loader import config_loader

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Layer Pruning Compressor for WMT25 Model Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python -m modelzip.compression.pruning.cli --model CohereLabs/aya-expanse-8b --output ./pruned_model

  # Custom dataset and parameters
  python -m modelzip.compression.pruning.cli \\
    --model CohereLabs/aya-expanse-8b \\
    --output ./pruned_model \\
    --dataset arcee-ai/sec-data-mini \\
    --dataset-size 2000 \\
    --batch-size 4 \\
    --layers-to-skip 28

  # Save detailed results to JSON
  python -m modelzip.compression.pruning.cli \\
    --model CohereLabs/aya-expanse-8b \\
    --output ./pruned_model \\
    --results results.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path or name of the model to compress"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the pruned model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--dataset", "-d",
        default=config_loader.get_default_dataset(),
        help=f"Dataset to use for layer similarity analysis (default: {config_loader.get_default_dataset()})"
    )
    
    parser.add_argument(
        "--dataset-column",
        default=config_loader.get_default_dataset_column(),
        help=f"Column name containing text data (default: {config_loader.get_default_dataset_column()})"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=config_loader.get_default_batch_size(),
        help=f"Batch size for processing (default: {config_loader.get_default_batch_size()})"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=config_loader.get_default_max_length(),
        help=f"Maximum sequence length (default: {config_loader.get_default_max_length()})"
    )
    
    parser.add_argument(
        "--layers-to-skip",
        type=int,
        default=config_loader.get_default_layers_to_skip(),
        help=f"Number of layers to skip (block size) (default: {config_loader.get_default_layers_to_skip()})"
    )
    
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=config_loader.get_default_dataset_size(),
        help=f"Number of dataset samples to process (default: {config_loader.get_default_dataset_size()})"
    )
    
    parser.add_argument(
        "--dataset-subset",
        default=config_loader.get_default_dataset_subset(),
        help=f"Dataset subset to use (default: {config_loader.get_default_dataset_subset()})"
    )
    
    parser.add_argument(
        "--merge-method",
        default=config_loader.get_default_merge_method(),
        choices=config_loader.get_merging_config().get("supported_methods", ["slicing", "weighted"]),
        help=f"Merge method to use (default: {config_loader.get_default_merge_method()})"
    )
    
    parser.add_argument(
        "--temp-dir",
        help="Temporary directory for intermediate files (auto-created if not specified)"
    )
    
    parser.add_argument(
        "--pruneme-path",
        default=config_loader.get_pruneme_path(),
        help=f"Path to PruneMe repository (default: {config_loader.get_pruneme_path()})"
    )
    
    parser.add_argument(
        "--results", "-r",
        help="Path to save detailed results as JSON"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the pruned model after compression"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def save_results(result: PruningResult, output_path: str):
    """Save detailed results to JSON file
    
    Args:
        result: Pruning results
        output_path: Path to save results
    """
    # Convert result to dictionary
    result_dict = {
        "success": result.success,
        "original_model_path": result.original_model_path,
        "pruned_model_path": result.pruned_model_path,
        "removed_layers": result.removed_layers,
        "layer_distance": result.layer_distance,
        "model_size_reduction": result.model_size_reduction,
        "error_message": result.error_message,
        "analysis_results": result.analysis_results,
        "merge_results": result.merge_results
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    LOG.info(f"Detailed results saved to: {output_path}")


def main():
    """Main CLI function"""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        LOG.getLogger().setLevel(LOG.DEBUG)
    
    # Create configuration
    config = PruningConfig(
        model_path=args.model,
        output_path=args.output,
        dataset=args.dataset,
        dataset_column=args.dataset_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layers_to_skip=args.layers_to_skip,
        dataset_size=args.dataset_size,
        dataset_subset=args.dataset_subset,
        merge_method=args.merge_method,
        temp_dir=args.temp_dir
    )
    
    # Initialize compressor
    try:
        compressor = LayerPruningCompressor(args.pruneme_path)
    except Exception as e:
        LOG.error(f"Failed to initialize LayerPruningCompressor: {e}")
        return 1
    
    # Run compression
    LOG.info("Starting layer pruning compression...")
    result = compressor.compress_model(config)
    
    # Print results
    if result.success:
        print("\n" + "="*60)
        print("✅ LAYER PRUNING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        summary = compressor.get_compression_summary(result)
        print(f"Original Model: {summary['original_model']}")
        print(f"Pruned Model:   {summary['pruned_model']}")
        print(f"Removed Layers: {summary['compression_details']['removed_layers']}")
        print(f"Layer Distance: {summary['compression_details']['layer_distance']}")
        print(f"Size Reduction: {summary['compression_details']['size_reduction']}")
        print(f"Analysis:       {summary['analysis_summary']}")
        print(f"Merge Method:   {summary['merge_summary']}")
        
        # Validate if requested
        if args.validate:
            print("\n🔍 Validating pruned model...")
            is_valid = compressor.validate_pruned_model(result)
            if is_valid:
                print("✅ Pruned model validation passed!")
            else:
                print("❌ Pruned model validation failed!")
        
        # Save detailed results if requested
        if args.results:
            save_results(result, args.results)
        
        print("\n" + "="*60)
        return 0
        
    else:
        print("\n" + "="*60)
        print("❌ LAYER PRUNING FAILED!")
        print("="*60)
        print(f"Error: {result.error_message}")
        print(f"Original Model: {result.original_model_path}")
        
        # Save error results if requested
        if args.results:
            save_results(result, args.results)
        
        print("\n" + "="*60)
        return 1


if __name__ == "__main__":
    exit(main()) 