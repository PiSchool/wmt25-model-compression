#!/usr/bin/env python
"""
Experiment Runner for WMT25 Model Compression

Orchestrates the execution of compression experiments including
model loading, compression application, evaluation, and result storage.
"""

import json
import time
import logging as LOG
from datetime import datetime
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
            
            # Create experiment directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name_with_timestamp = f"{config.name}_{timestamp}"
            exp_dir = self.experiments_dir / exp_name_with_timestamp
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
    
    def analyze_results(self, save_reports: bool = True) -> Dict[str, Any]:
        """Analyze all experiment results and generate comprehensive reports
        
        Args:
            save_reports: Whether to save reports to results directory
            
        Returns:
            Dictionary containing analysis of all results
        """
        all_results = self.get_all_results()
        
        if not all_results:
            return {"error": "No experiment results found"}
        
        # Group results by compression method
        by_method = {}
        by_lang_pair = {}
        
        for result in all_results:
            method = result.config.compression_method
            lang_pair = result.config.lang_pair
            
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
            
            if lang_pair not in by_lang_pair:
                by_lang_pair[lang_pair] = []
            by_lang_pair[lang_pair].append(result)
        
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
        
        # Calculate statistics for each language pair
        lang_pair_stats = {}
        for lang_pair, results in by_lang_pair.items():
            successful_results = [r for r in results if "error" not in r.quality_scores]
            
            if successful_results:
                lang_pair_stats[lang_pair] = {
                    "count": len(successful_results),
                    "avg_compression_ratio": sum(r.compression_ratio for r in successful_results) / len(successful_results),
                    "avg_model_size_mb": sum(r.model_size_mb for r in successful_results) / len(successful_results),
                    "avg_inference_time_ms": sum(r.inference_time_ms for r in successful_results) / len(successful_results),
                    "avg_quality_score": sum(r.get_quality_score() for r in successful_results) / len(successful_results),
                    "best_method": max(successful_results, key=lambda r: r.get_quality_score()).config.compression_method
                }
            else:
                lang_pair_stats[lang_pair] = {"count": 0, "note": "All experiments failed"}
        
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
        
        analysis = {
            "total_experiments": len(all_results),
            "successful_experiments": len([r for r in all_results if "error" not in r.quality_scores]),
            "by_method": method_stats,
            "by_lang_pair": lang_pair_stats,
            "best_compression_method": best_compression,
            "best_quality_method": best_quality,
            "analysis_timestamp": time.time(),
            "all_results": all_results
        }
        
        # Generate reports if requested
        if save_reports:
            self._generate_comprehensive_reports(analysis)
        
        return analysis
    
    def _generate_comprehensive_reports(self, analysis: Dict[str, Any]):
        """Generate comprehensive reports in the results directory
        
        Args:
            analysis: Analysis results dictionary
        """
        # Create results directory
        results_dir = self.workdir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON analysis report
        json_report_file = results_dir / f"analysis_report_{timestamp}.json"
        with open(json_report_file, 'w') as f:
            # Create a serializable version of analysis (without results objects)
            serializable_analysis = {k: v for k, v in analysis.items() if k != "all_results"}
            json.dump(serializable_analysis, f, indent=2)
        
        # Generate CSV comparison report
        csv_report_file = results_dir / f"comparison_{timestamp}.csv"
        self._generate_csv_report(analysis["all_results"], csv_report_file)
        
        # Generate markdown report
        md_report_file = results_dir / f"final_report_{timestamp}.md"
        self._generate_markdown_report(analysis, md_report_file)
        
        # Create "latest" symlinks for easy access
        self._create_latest_symlinks(results_dir, timestamp)
        
        LOG.info(f"📊 Reports generated in {results_dir}:")
        LOG.info(f"   📄 Markdown: {md_report_file.name}")
        LOG.info(f"   📊 CSV: {csv_report_file.name}")
        LOG.info(f"   📋 JSON: {json_report_file.name}")
    
    def _generate_csv_report(self, results: List, csv_file: Path):
        """Generate CSV comparison report"""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                "experiment", "compression_method", "lang_pair", "base_model",
                "model_size_mb", "compression_ratio", "inference_time_ms", 
                "memory_usage_mb", "quality_chrf", "quality_comet", "timestamp"
            ]
            writer.writerow(header)
            
            # Write data rows
            for result in results:
                if "error" not in result.quality_scores:
                    row = [
                        result.config.name,
                        result.config.compression_method,
                        result.config.lang_pair,
                        result.config.base_model,
                        f"{result.model_size_mb:.1f}",
                        f"{result.compression_ratio:.2f}",
                        f"{result.inference_time_ms:.1f}",
                        f"{result.memory_usage_mb:.1f}",
                        f"{result.quality_scores.get('chrf', 0.0):.2f}",
                        f"{result.quality_scores.get('comet', 0.0):.2f}",
                        result.timestamp
                    ]
                    writer.writerow(row)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], md_file: Path):
        """Generate comprehensive markdown report"""
        
        with open(md_file, 'w') as f:
            f.write("# WMT25 Model Compression Constrained Task Results\n\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview section
            f.write("## 📊 Overview\n\n")
            f.write(f"- **Total Experiments:** {analysis['total_experiments']}\n")
            f.write(f"- **Successful Experiments:** {analysis['successful_experiments']}\n")
            f.write(f"- **Language Pairs:** {', '.join(analysis['by_lang_pair'].keys())}\n")
            f.write(f"- **Compression Methods:** {', '.join(analysis['by_method'].keys())}\n\n")
            
            # Best performers section
            f.write("## 🏆 Best Performers\n\n")
            f.write(f"- **Best Compression Ratio:** {analysis.get('best_compression_method', 'N/A')}\n")
            f.write(f"- **Best Quality Score:** {analysis.get('best_quality_method', 'N/A')}\n\n")
            
            # Results by method
            f.write("## 📈 Results by Compression Method\n\n")
            f.write("| Method | Experiments | Avg Compression | Avg Size (MB) | Avg Time (ms) | Avg Quality |\n")
            f.write("|--------|-------------|----------------|---------------|---------------|--------------|\n")
            
            for method, stats in analysis['by_method'].items():
                if stats.get('count', 0) > 0:
                    f.write(f"| {method} | {stats['count']} | "
                           f"{stats['avg_compression_ratio']:.2f}x | "
                           f"{stats['avg_model_size_mb']:.1f} | "
                           f"{stats['avg_inference_time_ms']:.1f} | "
                           f"{stats['avg_quality_score']:.2f} |\n")
                else:
                    f.write(f"| {method} | 0 | - | - | - | Failed |\n")
            
            f.write("\n")
            
            # Results by language pair
            f.write("## 🌐 Results by Language Pair\n\n")
            f.write("| Language Pair | Experiments | Best Method | Avg Compression | Avg Quality |\n")
            f.write("|---------------|-------------|-------------|----------------|--------------|\n")
            
            for lang_pair, stats in analysis['by_lang_pair'].items():
                if stats.get('count', 0) > 0:
                    f.write(f"| {lang_pair} | {stats['count']} | "
                           f"{stats.get('best_method', 'N/A')} | "
                           f"{stats['avg_compression_ratio']:.2f}x | "
                           f"{stats['avg_quality_score']:.2f} |\n")
                else:
                    f.write(f"| {lang_pair} | 0 | - | - | Failed |\n")
            
            f.write("\n")
            
            # Individual results table
            f.write("## 📋 All Experiment Results\n\n")
            f.write("| Experiment | Method | Language | Compression | Size (MB) | Time (ms) | CHRF | COMET |\n")
            f.write("|------------|--------|----------|-------------|-----------|-----------|------|-------|\n")
            
            for result in analysis['all_results']:
                if "error" not in result.quality_scores:
                    f.write(f"| {result.config.name} | "
                           f"{result.config.compression_method} | "
                           f"{result.config.lang_pair} | "
                           f"{result.compression_ratio:.2f}x | "
                           f"{result.model_size_mb:.1f} | "
                           f"{result.inference_time_ms:.1f} | "
                           f"{result.quality_scores.get('chrf', 0.0):.2f} | "
                           f"{result.quality_scores.get('comet', 0.0):.2f} |\n")
                else:
                    f.write(f"| {result.config.name} | "
                           f"{result.config.compression_method} | "
                           f"{result.config.lang_pair} | ❌ Failed | - | - | - | - |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            for lang_pair, stats in analysis['by_lang_pair'].items():
                if stats.get('count', 0) > 0:
                    f.write(f"### {lang_pair}\n")
                    f.write(f"- **Recommended method:** {stats.get('best_method', 'N/A')}\n")
                    f.write(f"- **Expected compression:** {stats['avg_compression_ratio']:.2f}x\n")
                    f.write(f"- **Expected quality:** {stats['avg_quality_score']:.2f} CHRF\n\n")
    
    def _create_latest_symlinks(self, results_dir: Path, timestamp: str):
        """Create 'latest' symlinks for easy access to most recent reports"""
        try:
            # Remove old symlinks
            for pattern in ["final_report_latest.md", "comparison_latest.csv", "analysis_report_latest.json"]:
                symlink = results_dir / pattern
                if symlink.is_symlink():
                    symlink.unlink()
            
            # Create new symlinks
            (results_dir / "final_report_latest.md").symlink_to(f"final_report_{timestamp}.md")
            (results_dir / "comparison_latest.csv").symlink_to(f"comparison_{timestamp}.csv")
            (results_dir / "analysis_report_latest.json").symlink_to(f"analysis_report_{timestamp}.json")
            
        except Exception as e:
            LOG.warning(f"Could not create latest symlinks: {e}")
    
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