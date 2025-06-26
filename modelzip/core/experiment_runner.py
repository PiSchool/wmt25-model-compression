#!/usr/bin/env python
"""
Experiment Runner for WMT25 Model Compression

Orchestrates the execution of compression experiments including
model loading, compression application, evaluation, and result storage.
"""

import json
import time
import logging as LOG
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base_models import BaseModel, HuggingFaceModel
from .experiment_config import ExperimentConfig, ExperimentResults
from ..compression import get_compressor
from ..evaluation.evaluator import TranslationEvaluator
from ..utils.model_utils import setup_model
from ..constrained_config import WORKDIR

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ExperimentRunner:
    """Orchestrates compression experiments from start to finish
    
    Handles the complete experiment pipeline including model setup,
    compression, evaluation, and result storage with proper error handling.
    """
    
    def __init__(self, workdir: Path = None, hf_token: str = None):
        """Initialize experiment runner
        
        Args:
            workdir: Working directory for experiments (defaults to configured WORKDIR)
            hf_token: HuggingFace token for gated models
        """
        self.workdir = workdir or WORKDIR
        self.experiments_dir = self.workdir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.hf_token = hf_token
        # Results storage
        self.results: List[ExperimentResults] = []
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResults:
        """Run a single compression experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResults: Results of the experiment
        """
        LOG.info(f"Starting experiment: {config.name}")
        
        try:
            # Setup model
            model_path = setup_model(config.base_model, token=self.hf_token)
            model = HuggingFaceModel(model_path, config)
            
            # Create experiment directory
            exp_dir = self.experiments_dir / config.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply compression
            compressor = get_compressor(config)
            compressed_model_dir = exp_dir / "compressed_model"
            
            start_time = time.time()
            compressed_model = compressor.compress(model, compressed_model_dir)
            compression_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Evaluate compressed model
            evaluator = TranslationEvaluator(config.lang_pair)
            
            # Performance metrics
            model_size = compressed_model.get_model_size()
            memory_usage = compressed_model.get_memory_usage()
            compression_ratio = compressor.get_compression_ratio(model, compressed_model)
            
            # Quality evaluation
            quality_scores = {}
            try:
                quality_scores = evaluator.evaluate_model(compressed_model, config.eval_params)
            except Exception as e:
                LOG.warning(f"Quality evaluation failed: {e}")
                quality_scores = {"error": str(e)}
            
            # Create results
            results = ExperimentResults(
                config=config,
                model_size_mb=model_size,
                memory_usage_mb=memory_usage,
                inference_time_ms=compression_time,  # Using compression time as placeholder
                compression_ratio=compression_ratio,
                quality_scores=quality_scores
            )
            
            # Save results
            self._save_experiment_results(results, exp_dir)
            self.results.append(results)
            
            LOG.info(f"Experiment {config.name} completed successfully")
            return results
            
        except Exception as e:
            LOG.error(f"Experiment {config.name} failed: {e}")
            # Create failure result
            failure_result = ExperimentResults(
                config=config,
                model_size_mb=0.0,
                memory_usage_mb=0.0,
                inference_time_ms=0.0,
                compression_ratio=0.0,
                quality_scores={"error": str(e)}
            )
            return failure_result
    
    def run_experiments(self, configs: List[ExperimentConfig]) -> List[ExperimentResults]:
        """Run multiple experiments
        
        Args:
            configs: List of experiment configurations
            
        Returns:
            List[ExperimentResults]: Results from all experiments
        """
        results = []
        
        for config in configs:
            result = self.run_experiment(config)
            results.append(result)
        
        return results
    
    def _save_experiment_results(self, results: ExperimentResults, exp_dir: Path):
        """Save experiment results to file
        
        Args:
            results: Results to save
            exp_dir: Experiment directory
        """
        results_file = exp_dir / "results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        LOG.info(f"Results saved to {results_file}")
    
    def load_experiment_results(self, exp_name: str) -> Optional[ExperimentResults]:
        """Load experiment results from file
        
        Args:
            exp_name: Name of the experiment
            
        Returns:
            ExperimentResults or None if not found
        """
        results_file = self.experiments_dir / exp_name / "results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file) as f:
                data = json.load(f)
            return ExperimentResults.from_dict(data)
        except Exception as e:
            LOG.error(f"Failed to load results for {exp_name}: {e}")
            return None
    
    def list_experiments(self) -> List[str]:
        """List all completed experiments
        
        Returns:
            List of experiment names
        """
        if not self.experiments_dir.exists():
            return []
        
        experiments = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "results.json").exists():
                experiments.append(exp_dir.name)
        
        return sorted(experiments)
    
    def get_all_results(self) -> List[ExperimentResults]:
        """Load all experiment results from disk
        
        Returns:
            List of all experiment results
        """
        all_results = []
        
        for exp_name in self.list_experiments():
            result = self.load_experiment_results(exp_name)
            if result:
                all_results.append(result)
        
        return all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze all experiment results
        
        Returns:
            Dictionary containing analysis of all results
        """
        all_results = self.get_all_results()
        
        if not all_results:
            return {"error": "No experiment results found"}
        
        # Group results by compression method
        by_method = {}
        for result in all_results:
            method = result.config.compression_method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        # Calculate statistics for each method
        method_stats = {}
        for method, results in by_method.items():
            # Filter out failed experiments
            successful_results = [r for r in results if "error" not in r.quality_scores]
            
            if successful_results:
                method_stats[method] = {
                    "count": len(successful_results),
                    "avg_compression_ratio": sum(r.compression_ratio for r in successful_results) / len(successful_results),
                    "avg_model_size_mb": sum(r.model_size_mb for r in successful_results) / len(successful_results),
                    "avg_inference_time_ms": sum(r.inference_time_ms for r in successful_results) / len(successful_results),
                    "avg_quality_score": sum(r.get_quality_score() for r in successful_results) / len(successful_results)
                }
            else:
                method_stats[method] = {"count": 0, "note": "All experiments failed"}
        
        # Find best performing methods
        best_compression = max(
            (method for method, stats in method_stats.items() if stats.get("count", 0) > 0),
            key=lambda m: method_stats[m]["avg_compression_ratio"],
            default=None
        )
        
        best_quality = max(
            (method for method, stats in method_stats.items() if stats.get("count", 0) > 0),
            key=lambda m: method_stats[m]["avg_quality_score"],
            default=None
        )
        
        return {
            "total_experiments": len(all_results),
            "successful_experiments": len([r for r in all_results if "error" not in r.quality_scores]),
            "by_method": method_stats,
            "best_compression_method": best_compression,
            "best_quality_method": best_quality,
            "analysis_timestamp": time.time()
        }
    
    def cleanup_experiments(self, keep_recent: int = 5):
        """Clean up old experiment directories
        
        Args:
            keep_recent: Number of recent experiments to keep
        """
        experiments = self.list_experiments()
        
        if len(experiments) <= keep_recent:
            return
        
        # Sort by modification time and remove old ones
        exp_dirs = [
            (self.experiments_dir / name, (self.experiments_dir / name).stat().st_mtime)
            for name in experiments
        ]
        exp_dirs.sort(key=lambda x: x[1], reverse=True)
        
        for exp_dir, _ in exp_dirs[keep_recent:]:
            LOG.info(f"Removing old experiment: {exp_dir.name}")
            import shutil
            shutil.rmtree(exp_dir) 