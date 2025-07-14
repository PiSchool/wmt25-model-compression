#!/usr/bin/env python
"""
Layer Similarity Analyzer for WMT25 Model Compression

Wrapper around PruneMe's layer similarity analysis functionality.
Integrates with the WMT25 framework for layer-based pruning.
"""

import os
import sys
import subprocess
import pandas as pd
import logging as LOG
from pathlib import Path
from typing import Dict, Tuple, Optional

from .config_loader import config_loader

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LayerSimilarityAnalyzer:
    """Analyzes layer similarity using PruneMe's layer_similarity.py
    
    This class provides a clean interface to PruneMe's layer similarity
    analysis, integrating it with the WMT25 framework architecture.
    """
    
    def __init__(self, pruneme_path: str = None):
        """Initialize the layer similarity analyzer
        
        Args:
            pruneme_path: Path to the PruneMe repository (uses config default if None)
        """
        if pruneme_path is None:
            pruneme_path = config_loader.get_pruneme_path()
        self.pruneme_path = Path(pruneme_path)
        self._validate_pruneme_setup()
    
    def _validate_pruneme_setup(self):
        """Validate that PruneMe is properly set up"""
        layer_similarity_script = self.pruneme_path / "compute_block_similarity" / "layer_similarity.py"
        
        if not layer_similarity_script.exists():
            raise FileNotFoundError(
                f"PruneMe layer_similarity.py not found at: {layer_similarity_script}\n"
                f"Please ensure PruneMe is cloned in the parent directory and properly installed."
            )
        
        LOG.info(f"PruneMe layer similarity script found at: {layer_similarity_script}")
    
    def analyze_layers(self, 
                      model_path: str,
                      dataset: str = None,
                      dataset_column: str = None,
                      batch_size: int = None,
                      max_length: int = None,
                      layers_to_skip: int = None,
                      dataset_size: int = None,
                      dataset_subset: str = None,
                      output_dir: Path = None) -> Tuple[bool, Optional[Dict]]:
        """Run layer similarity analysis using PruneMe
        
        Args:
            model_path: Path or name of the model to analyze
            dataset: Dataset name for analysis
            dataset_column: Column name containing text data
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            layers_to_skip: Number of layers to skip (block size)
            dataset_size: Number of dataset samples to process
            dataset_subset: Dataset subset to use
            output_dir: Directory to save results (optional)
            
        Returns:
            Tuple[bool, Optional[Dict]]: (success, results_dict)
        """
        LOG.info(f"Starting layer similarity analysis for model: {model_path}")
        
        # Set defaults from config if not provided
        if dataset is None:
            dataset = config_loader.get_default_dataset()
        if dataset_column is None:
            dataset_column = config_loader.get_default_dataset_column()
        if batch_size is None:
            batch_size = config_loader.get_default_batch_size()
        if max_length is None:
            max_length = config_loader.get_default_max_length()
        if layers_to_skip is None:
            layers_to_skip = config_loader.get_default_layers_to_skip()
        if dataset_size is None:
            dataset_size = config_loader.get_default_dataset_size()
        if dataset_subset is None:
            dataset_subset = config_loader.get_default_dataset_subset()
        
        # Create output directory if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct path to PruneMe layer_similarity.py
        layer_similarity_script = self.pruneme_path / "compute_block_similarity" / "layer_similarity.py"
        
        # Change to the PruneMe compute_block_similarity directory
        original_dir = os.getcwd()
        pruneme_compute_dir = self.pruneme_path / "compute_block_similarity"
        
        try:
            os.chdir(pruneme_compute_dir)
            LOG.info(f"Changed to directory: {pruneme_compute_dir}")
            
            # Build the command
            cmd = [
                sys.executable, "layer_similarity.py",
                "--model_path", model_path,
                "--dataset", dataset,
                "--dataset_column", dataset_column,
                "--batch_size", str(batch_size),
                "--max_length", str(max_length),
                "--layers_to_skip", str(layers_to_skip),
                "--dataset_size", str(dataset_size),
                "--dataset_subset", dataset_subset
            ]
            
            LOG.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=False)
            
            # Read and process results
            results = self._process_results(output_dir)
            
            LOG.info("Layer similarity analysis completed successfully")
            return True, results
            
        except subprocess.CalledProcessError as e:
            LOG.error(f"Error running layer similarity analysis: {e}")
            return False, None
        except Exception as e:
            LOG.error(f"Unexpected error during layer similarity analysis: {e}")
            return False, None
        finally:
            # Change back to original directory
            os.chdir(original_dir)
    
    def _process_results(self, output_dir: Path = None) -> Dict:
        """Process the layer similarity analysis results
        
        Args:
            output_dir: Directory to save processed results
            
        Returns:
            Dict: Processed results including layer distances and recommendations
        """
        # Read the CSV file
        csv_file = Path("layer_distances.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"Layer distances CSV not found at: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Find the most similar layers
        most_similar_row = df.loc[df['average_distance'].idxmin()]
        start_layer = int(most_similar_row['block_start'])
        end_layer = int(most_similar_row['block_end'])
        min_distance = float(most_similar_row['average_distance'])
        
        # Calculate statistics
        avg_distance = df['average_distance'].mean()
        std_distance = df['average_distance'].std()
        
        results = {
            "layer_distances": df.to_dict('records'),
            "most_similar_layers": {
                "start": start_layer,
                "end": end_layer,
                "distance": min_distance
            },
            "statistics": {
                "average_distance": avg_distance,
                "std_distance": std_distance,
                "total_blocks": len(df)
            },
            "recommendation": f"Remove layers {start_layer} to {end_layer} (distance: {min_distance:.4f})"
        }
        
        # Save results to output directory if specified
        if output_dir:
            # Copy the CSV file
            import shutil
            target_csv = output_dir / "layer_distances.csv"
            shutil.copy2(csv_file, target_csv)
            LOG.info(f"Layer distances saved to: {target_csv}")
            
            # Save processed results as JSON
            import json
            results_file = output_dir / "layer_analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            LOG.info(f"Analysis results saved to: {results_file}")
        
        return results
    
    def get_layer_recommendation(self, results: Dict) -> Tuple[int, int, float]:
        """Extract layer removal recommendation from results
        
        Args:
            results: Results from analyze_layers()
            
        Returns:
            Tuple[int, int, float]: (start_layer, end_layer, distance)
        """
        if not results or "most_similar_layers" not in results:
            raise ValueError("Invalid results format")
        
        similar = results["most_similar_layers"]
        return similar["start"], similar["end"], similar["distance"] 