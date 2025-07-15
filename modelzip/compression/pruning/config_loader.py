#!/usr/bin/env python
"""
Configuration Loader for Layer Pruning Module

Loads configuration from YAML files and provides default values.
"""

import yaml
import logging as LOG
from pathlib import Path
from typing import Dict, Any, Optional

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ConfigLoader:
    """Loads and manages configuration for the layer pruning module"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file
        
        Returns:
            Dict: Configuration dictionary
        """
        if not self.config_path.exists():
            LOG.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            LOG.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            LOG.error(f"Error loading config from {self.config_path}: {e}")
            LOG.info("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Dict: Default configuration
        """
        return {
            "model": {
                "default_path": "CohereLabs/aya-expanse-8b",
                "output_suffix": "pruned"
            },
            "dataset": {
                "default_name": "arcee-ai/sec-data-mini",
                "default_column": "text",
                "default_subset": "train",
                "default_size": 4000
            },
            "processing": {
                "batch_size": 8,
                "max_length": 1024,
                "layers_to_skip": 28
            },
            "merging": {
                "default_method": "slicing",
                "supported_methods": ["slicing", "weighted"]
            },
            "paths": {
                "pruneme_path": "../PruneMe",
                "temp_dir": None
            },
            "output": {
                "save_results": False,
                "validate_model": False,
                "verbose_logging": False
            }
        }
    
    def get_model_config(self) -> Dict[str, str]:
        """Get model configuration
        
        Returns:
            Dict: Model configuration
        """
        return self.config.get("model", {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration
        
        Returns:
            Dict: Dataset configuration
        """
        return self.config.get("dataset", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration
        
        Returns:
            Dict: Processing configuration
        """
        return self.config.get("processing", {})
    
    def get_merging_config(self) -> Dict[str, Any]:
        """Get merging configuration
        
        Returns:
            Dict: Merging configuration
        """
        return self.config.get("merging", {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration
        
        Returns:
            Dict: Paths configuration
        """
        return self.config.get("paths", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration
        
        Returns:
            Dict: Output configuration
        """
        return self.config.get("output", {})
    
    def get_default_model_path(self) -> str:
        """Get default model path
        
        Returns:
            str: Default model path
        """
        return self.get_model_config().get("default_path", "CohereLabs/aya-expanse-8b")
    
    def get_default_dataset(self) -> str:
        """Get default dataset name
        
        Returns:
            str: Default dataset name
        """
        return self.get_dataset_config().get("default_name", "arcee-ai/sec-data-mini")
    
    def get_default_dataset_column(self) -> str:
        """Get default dataset column
        
        Returns:
            str: Default dataset column
        """
        return self.get_dataset_config().get("default_column", "text")
    
    def get_default_batch_size(self) -> int:
        """Get default batch size
        
        Returns:
            int: Default batch size
        """
        return self.get_processing_config().get("batch_size", 8)
    
    def get_default_max_length(self) -> int:
        """Get default max length
        
        Returns:
            int: Default max length
        """
        return self.get_processing_config().get("max_length", 1024)
    
    def get_default_layers_to_skip(self) -> int:
        """Get default layers to skip
        
        Returns:
            int: Default layers to skip
        """
        return self.get_processing_config().get("layers_to_skip", 28)
    
    def get_default_dataset_size(self) -> int:
        """Get default dataset size
        
        Returns:
            int: Default dataset size
        """
        return self.get_dataset_config().get("default_size", 4000)
    
    def get_default_dataset_subset(self) -> str:
        """Get default dataset subset
        
        Returns:
            str: Default dataset subset
        """
        return self.get_dataset_config().get("default_subset", "train")
    
    def get_default_merge_method(self) -> str:
        """Get default merge method
        
        Returns:
            str: Default merge method
        """
        return self.get_merging_config().get("default_method", "slicing")
    
    def get_pruneme_path(self) -> str:
        """Get PruneMe path
        
        Returns:
            str: PruneMe path
        """
        return self.get_paths_config().get("pruneme_path", "../PruneMe")
    
    def get_temp_dir(self) -> Optional[str]:
        """Get temp directory
        
        Returns:
            Optional[str]: Temp directory path
        """
        temp_dir = self.get_paths_config().get("temp_dir")
        if temp_dir is None or temp_dir == "/results/temp_pruning":
            return "workdir/results/temp_pruning"
        if temp_dir.startswith("/results/"):
            return temp_dir.replace("/results/", "workdir/results/")
        return temp_dir
    
    def should_save_results(self) -> bool:
        """Check if results should be saved
        
        Returns:
            bool: True if results should be saved
        """
        return self.get_output_config().get("save_results", False)
    
    def should_validate_model(self) -> bool:
        """Check if model should be validated
        
        Returns:
            bool: True if model should be validated
        """
        return self.get_output_config().get("validate_model", False)
    
    def should_verbose_logging(self) -> bool:
        """Check if verbose logging should be enabled
        
        Returns:
            bool: True if verbose logging should be enabled
        """
        return self.get_output_config().get("verbose_logging", False)
    
    def create_pruning_config(self, 
                            model_path: Optional[str] = None,
                            output_path: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """Create a pruning configuration with defaults from config file
        
        Args:
            model_path: Model path (uses default if None)
            output_path: Output path (uses default if None)
            **kwargs: Additional configuration overrides
            
        Returns:
            Dict: Complete pruning configuration
        """
        config = {
            "model_path": model_path or self.get_default_model_path(),
            "output_path": output_path or f"workdir/results/{self.get_model_config().get('output_suffix', 'pruned')}",
            "dataset": kwargs.get("dataset", self.get_default_dataset()),
            "dataset_column": kwargs.get("dataset_column", self.get_default_dataset_column()),
            "batch_size": kwargs.get("batch_size", self.get_default_batch_size()),
            "max_length": kwargs.get("max_length", self.get_default_max_length()),
            "layers_to_skip": kwargs.get("layers_to_skip", self.get_default_layers_to_skip()),
            "dataset_size": kwargs.get("dataset_size", self.get_default_dataset_size()),
            "dataset_subset": kwargs.get("dataset_subset", self.get_default_dataset_subset()),
            "merge_method": kwargs.get("merge_method", self.get_default_merge_method()),
            "temp_dir": kwargs.get("temp_dir", self.get_temp_dir())
        }
        
        return config


# Global config loader instance
config_loader = ConfigLoader() 