#!/usr/bin/env python
"""
Layer Merger for WMT25 Model Compression

Wrapper around mergekit functionality for layer-based model merging.
Integrates with the WMT25 framework for layer pruning operations.
"""

import os
import sys
import subprocess
import logging as LOG
from pathlib import Path
from typing import Tuple, Optional, Dict

from .config_loader import config_loader

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LayerMerger:
    """Merges model layers using mergekit functionality
    
    This class provides a clean interface to mergekit for layer merging,
    integrating it with the WMT25 framework architecture.
    """
    
    def __init__(self, pruneme_path: str = None):
        """Initialize the layer merger
        
        Args:
            pruneme_path: Path to the PruneMe repository (uses config default if None)
        """
        if pruneme_path is None:
            pruneme_path = config_loader.get_pruneme_path()
        self.pruneme_path = Path(pruneme_path)
        self._validate_mergekit_setup()
    
    def _validate_mergekit_setup(self):
        """Validate that mergekit is properly set up"""
        merge_me_script = self.pruneme_path / "slice_with_mergekit" / "merge_me.py"
        
        if not merge_me_script.exists():
            raise FileNotFoundError(
                f"PruneMe merge_me.py not found at: {merge_me_script}\n"
                f"Please ensure PruneMe is cloned in the parent directory and mergekit is installed."
            )
        
        LOG.info(f"PruneMe merge_me.py script found at: {merge_me_script}")
    
    def create_merge_config(self, 
                           model_path: str,
                           output_path: str,
                           start_layer: int,
                           end_layer: int,
                           merge_method: str = None) -> str:
        """Create mergekit YAML configuration for layer slicing
        
        Args:
            model_path: Path or name of the model to slice
            output_path: Path for the output model
            start_layer: Start layer index (inclusive)
            end_layer: End layer index (inclusive)
            merge_method: Merge method to use (slicing or weighted)
            
        Returns:
            str: YAML configuration content
        """
        LOG.info(f"Creating merge config for layers {start_layer} to {end_layer}")
        
        # Set default merge method from config if not provided
        if merge_method is None:
            merge_method = config_loader.get_default_merge_method()
        
        # Create YAML content based on merge method
        if merge_method == "slicing":
            yaml_content = f"""\
slices:
  - sources:
      - model: {model_path}
        layer_range: [{start_layer},{end_layer}]

merge_method: passthrough
dtype: bfloat16
"""
        elif merge_method == "weighted":
            yaml_content = f"""\
slices:
  - sources:
      - model: {model_path}
        layer_range: [{start_layer},{end_layer}]

merge_method: linear
dtype: bfloat16
"""
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")
        
        LOG.info(f"Merge config created for {merge_method} method")
        return yaml_content
    
    def merge_layers(self, 
                    model_path: str,
                    output_path: str,
                    start_layer: int,
                    end_layer: int,
                    merge_method: str = None,
                    temp_dir: Path = None) -> Tuple[bool, Optional[Dict]]:
        """Merge model layers using mergekit
        
        Args:
            model_path: Path or name of the model to slice
            output_path: Path for the output model
            start_layer: Start layer index (inclusive)
            end_layer: End layer index (inclusive)
            merge_method: Merge method to use (slicing or weighted)
            temp_dir: Temporary directory for processing (optional)
            
        Returns:
            Tuple[bool, Optional[Dict]]: (success, results_dict)
        """
        LOG.info(f"Starting layer merging for layers {start_layer} to {end_layer}")
        
        # Set default merge method from config if not provided
        if merge_method is None:
            merge_method = config_loader.get_default_merge_method()
        
        # Convert output_path to Path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use temp_dir if provided, otherwise create one in workdir/results
        work_dir = temp_dir if temp_dir else Path("workdir/results/temp_merge")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create merge configuration
        config_content = self.create_merge_config(model_path, output_path, start_layer, end_layer, merge_method)
        
        # Save config to work directory
        config_path = work_dir / "slice.yaml"
        with open(config_path, "w") as f:
            f.write(config_content)
        
        # Copy config to PruneMe slice_with_mergekit directory
        slice_dir = self.pruneme_path / "slice_with_mergekit"
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        target_config = slice_dir / "slice.yaml"
        shutil.copy2(config_path, target_config)
        LOG.info(f"Config copied to: {target_config}")
        
        # Run the merge_me.py script
        success = self._run_merge_script(slice_dir, output_path)
        
        if success:
            LOG.info(f"Layer merging completed successfully. Model saved to: {output_path}")
            
            # Generate results dictionary
            results = {
                "merged_model_path": str(output_path),
                "summary": f"{merge_method.capitalize()} merge completed successfully",
                "config_file": str(config_path),
                "layers_removed": {
                    "start": start_layer,
                    "end": end_layer,
                    "count": end_layer - start_layer + 1
                }
            }
            
            return True, results
        else:
            LOG.error("Layer merging failed")
            return False, None
    
    def _run_merge_script(self, slice_dir: Path, output_path: Path) -> bool:
        """Run the merge_me.py script from PruneMe
        
        Args:
            slice_dir: Directory containing merge_me.py
            output_path: Directory to save results
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Change to the slice_with_mergekit directory
        original_dir = os.getcwd()
        import shutil
        import subprocess
        
        try:
            os.chdir(slice_dir)
            LOG.info(f"Changed to directory: {slice_dir}")
            
            # Run the merge_me.py script
            cmd = [sys.executable, "merge_me.py"]
            LOG.info(f"Running command: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True, capture_output=False)
            
            # Copy merged model to output directory
            merged_model_path = Path("merged")
            if merged_model_path.exists():
                # Copy directly to output_path, not to a subdirectory
                try:
                    if output_path.exists():
                        shutil.rmtree(output_path)
                    shutil.copytree(merged_model_path, output_path)
                    LOG.info(f"Merged model copied to: {output_path}")
                except Exception as e:
                    LOG.error(f"Failed to copy merged model: {e}")
                    # Try alternative copy method
                    try:
                        subprocess.run(["cp", "-r", str(merged_model_path), str(output_path)], check=True)
                        LOG.info(f"Merged model copied to: {output_path} (using cp)")
                    except Exception as e2:
                        LOG.error(f"Failed to copy merged model with cp: {e2}")
                        return False
            else:
                LOG.error(f"Merged model not found at: {merged_model_path}")
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            LOG.error(f"Error running merge_me.py: {e}")
            return False
        except Exception as e:
            LOG.error(f"Unexpected error during merging: {e}")
            return False
        finally:
            # Change back to original directory
            os.chdir(original_dir)
    
    def validate_merged_model(self, merged_model_path: Path) -> bool:
        """Validate that the merged model is properly created
        
        Args:
            merged_model_path: Path to the merged model
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        if not merged_model_path.exists():
            LOG.error(f"Merged model path does not exist: {merged_model_path}")
            return False
        
        # Check for essential files
        required_files = ["config.json", "pytorch_model.bin"]
        for file_name in required_files:
            file_path = merged_model_path / file_name
            if not file_path.exists():
                LOG.error(f"Required file missing: {file_path}")
                return False
        
        LOG.info(f"Merged model validation passed: {merged_model_path}")
        return True
    
    def get_merge_summary(self, 
                         original_model: str,
                         start_layer: int,
                         end_layer: int,
                         merged_model_path: Path) -> dict:
        """Generate a summary of the merging operation
        
        Args:
            original_model: Original model name/path
            start_layer: Start layer that was removed
            end_layer: End layer that was removed
            merged_model_path: Path to the merged model
            
        Returns:
            dict: Summary of the merging operation
        """
        layers_removed = end_layer - start_layer + 1
        
        summary = {
            "original_model": original_model,
            "layers_removed": {
                "start": start_layer,
                "end": end_layer,
                "count": layers_removed
            },
            "merged_model_path": str(merged_model_path),
            "operation": "layer_pruning",
            "method": "mergekit_passthrough"
        }
        
        # Add file size information if available
        if merged_model_path.exists():
            try:
                total_size = sum(f.stat().st_size for f in merged_model_path.rglob("*") if f.is_file())
                summary["merged_model_size_mb"] = total_size / (1024 * 1024)
            except Exception as e:
                LOG.warning(f"Could not calculate merged model size: {e}")
        
        return summary 