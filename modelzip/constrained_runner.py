#!/usr/bin/env python
"""
Constrained Task Runner for WMT25 Model Compression

Main entry point for running the WMT25 constrained model compression task.
Provides CLI interface and orchestrates experiments using the modular framework.
"""

import argparse
import json
import logging as LOG
from pathlib import Path
from typing import List

# Import from new modular structure
from .core import ExperimentConfig, ExperimentRunner, create_experiment_config
from .compression import get_compressor, list_available_methods
from .data_manager import DataManager
from .utils import setup_model
from .constrained_config import CONSTRAINED_TASK, COMPRESSION_CONFIG, WORKDIR

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ConstrainedTaskRunner:
    """Main runner for WMT25 constrained model compression experiments
    
    Orchestrates the complete pipeline from data preparation to final analysis.
    Uses the modular framework components for clean separation of concerns.
    """
    
    def __init__(self, workdir: Path = None):
        """Initialize the constrained task runner
        
        Args:
            workdir: Working directory for experiments and data
        """
        self.workdir = workdir or WORKDIR
        self.data_manager = DataManager(self.workdir)
        
        # Check for HuggingFace token for gated models
        import os
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        self.experiment_runner = ExperimentRunner(self.workdir, hf_token=self.hf_token)
        
        if not self.hf_token and "CohereLabs/aya-expanse-8b" in str(CONSTRAINED_TASK.get("base_model", "")):
            LOG.warning("No HuggingFace token found. Aya Expanse model requires gated access.")
            LOG.warning("Please visit https://huggingface.co/CohereLabs/aya-expanse-8b and accept the license agreement.")
            LOG.warning("Then set: export HF_TOKEN=your_token_here")
        
        LOG.info(f"Initialized ConstrainedTaskRunner with workdir: {self.workdir}")
    
    def setup_data(self):
        """Setup training and test data for all language pairs"""
        LOG.info("Setting up data for WMT25 constrained task...")
        
        for lang_pair in CONSTRAINED_TASK["language_pairs"]:
            LOG.info(f"Setting up data for {lang_pair}")
            try:
                # Download training data
                train_file = self.data_manager.download_training_data(lang_pair)
                LOG.info(f"Training data: {train_file}")
                
                # Download test data
                test_file = self.data_manager.download_test_data(lang_pair)
                LOG.info(f"Test data: {test_file}")
                
            except Exception as e:
                LOG.error(f"Failed to setup data for {lang_pair}: {e}")
        
        # Print data statistics
        self.print_data_statistics()
    
    def setup_models(self):
        """Setup base models for compression experiments"""
        LOG.info("Setting up models for WMT25 constrained task...")
        
        base_model = CONSTRAINED_TASK["base_model"]
        LOG.info(f"Setting up base model: {base_model}")
        
        try:
            model_path = setup_model(base_model, token=self.hf_token)
            LOG.info(f"Base model ready at: {model_path}")
        except Exception as e:
            LOG.error(f"Failed to setup base model {base_model}: {e}")
            if "gated" in str(e).lower():
                LOG.error("This appears to be a gated model. Please ensure you have:")
                LOG.error("1. Accepted the license agreement on HuggingFace")
                LOG.error("2. Set your HF_TOKEN environment variable")
                LOG.error("3. Or use a different model with --base-model parameter")
    
    def run_all_experiments(self, quick_test: bool = False):
        """Run all compression experiments for the constrained task
        
        Args:
            quick_test: If True, run minimal experiments for testing
        """
        LOG.info("Running WMT25 constrained task experiments...")
        
        # Create experiment configurations
        configs = self._create_experiment_configs(quick_test)
        
        if not configs:
            LOG.error("No experiment configurations created")
            return
        
        LOG.info(f"Created {len(configs)} experiment configurations")
        
        # Run experiments
        results = self.experiment_runner.run_experiments(configs)
        
        # Print summary
        successful = sum(1 for r in results if "error" not in r.quality_scores)
        LOG.info(f"Completed {len(results)} experiments, {successful} successful")
        
        return results
    
    def _create_experiment_configs(self, quick_test: bool = False) -> List[ExperimentConfig]:
        """Create experiment configurations for the constrained task
        
        Args:
            quick_test: If True, create minimal configs for testing
            
        Returns:
            List of experiment configurations
        """
        configs = []
        
        # Select language pairs and methods based on test mode
        if quick_test:
            lang_pairs = [CONSTRAINED_TASK["language_pairs"][0]]  # Just first language pair
            compression_methods = ["baseline", "quantization_8bit"]  # Minimal methods
        else:
            lang_pairs = CONSTRAINED_TASK["language_pairs"]
            compression_methods = list(COMPRESSION_CONFIG.keys())
        
        # Create configurations for each combination
        for lang_pair in lang_pairs:
            for method in compression_methods:
                # Skip invalid combinations
                if not self._validate_method_for_pair(method, lang_pair):
                    continue
                
                config_name = f"{method}_{lang_pair}"
                if quick_test:
                    config_name = f"quick_test_{config_name}"
                
                # Create configuration
                config = create_experiment_config(
                    name=config_name,
                    compression_method=method,
                    lang_pair=lang_pair,
                    base_model=CONSTRAINED_TASK["base_model"]
                )
                
                configs.append(config)
        
        return configs
    
    def _validate_method_for_pair(self, method: str, lang_pair: str) -> bool:
        """Validate if a compression method can be used with a language pair
        
        Args:
            method: Compression method name
            lang_pair: Language pair
            
        Returns:
            bool: True if combination is valid
        """
        # For now, allow all combinations
        # In the future, this could check for specific requirements
        return True
    
    def analyze_results(self):
        """Analyze all experiment results"""
        LOG.info("Analyzing experiment results...")
        
        analysis = self.experiment_runner.analyze_results()
        
        if "error" in analysis:
            LOG.error(f"Analysis failed: {analysis['error']}")
            return
        
        # Print analysis results
        self._print_analysis(analysis)
        
        # Save analysis to file
        analysis_file = self.workdir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        LOG.info(f"Analysis saved to {analysis_file}")
    
    def _print_analysis(self, analysis: dict):
        """Print analysis results in a readable format
        
        Args:
            analysis: Analysis results dictionary
        """
        print("\n" + "="*60)
        print("WMT25 CONSTRAINED TASK ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nTotal Experiments: {analysis['total_experiments']}")
        print(f"Successful: {analysis['successful_experiments']}")
        
        print(f"\nBest Compression Method: {analysis.get('best_compression_method', 'N/A')}")
        print(f"Best Quality Method: {analysis.get('best_quality_method', 'N/A')}")
        
        print("\nResults by Compression Method:")
        print("-" * 40)
        
        for method, stats in analysis.get("by_method", {}).items():
            if stats.get("count", 0) > 0:
                print(f"\n{method}:")
                print(f"  Experiments: {stats['count']}")
                print(f"  Avg Compression: {stats['avg_compression_ratio']:.2f}x")
                print(f"  Avg Model Size: {stats['avg_model_size_mb']:.1f} MB")
                print(f"  Avg Inference: {stats['avg_inference_time_ms']:.1f} ms")
                print(f"  Avg Quality: {stats['avg_quality_score']:.2f}")
            else:
                print(f"\n{method}: {stats.get('note', 'No data')}")
        
        print("\n" + "="*60)
    
    def print_data_statistics(self):
        """Print data statistics for all language pairs"""
        LOG.info("Data statistics for WMT25 constrained task:")
        
        for lang_pair in CONSTRAINED_TASK["language_pairs"]:
            stats = self.data_manager.get_data_statistics(lang_pair)
            
            print(f"\n{lang_pair}:")
            print(f"  Training samples: {stats.get('train_samples', 'N/A')}")
            print(f"  Test samples: {stats.get('test_samples', 'N/A')}")
    
    def list_available_methods(self):
        """List all available compression methods"""
        methods = list_available_methods()
        
        print("\nAvailable Compression Methods:")
        print("-" * 40)
        
        for method, description in methods.items():
            print(f"{method:20} : {description}")
        
        return methods
    
    def run_single_experiment(self, method: str, lang_pair: str = None, base_model: str = None, name: str = None):
        """Run a single experiment with specified parameters
        
        Args:
            method: Compression method to use
            lang_pair: Language pair to test (if None, runs on all pairs)
            base_model: Custom base model (if None, uses default)
            name: Optional custom name for the experiment
        """
        if base_model is None:
            base_model = CONSTRAINED_TASK["base_model"]
        
        # If no language pair specified, run on all language pairs
        if lang_pair is None:
            lang_pairs = CONSTRAINED_TASK["language_pairs"]
            LOG.info(f"Running {method} compression on all language pairs: {lang_pairs}")
            
            results = []
            for pair in lang_pairs:
                exp_name = f"single_{method}_{pair}" if name is None else f"{name}_{pair}"
                result = self._run_single_experiment_for_pair(method, pair, base_model, exp_name)
                results.append((pair, result))
            
            # Print summary
            self._print_multi_pair_results(method, results)
            return results
        else:
            # Run on single language pair
            exp_name = f"single_{method}_{lang_pair}" if name is None else name
            result = self._run_single_experiment_for_pair(method, lang_pair, base_model, exp_name)
            return [(lang_pair, result)]
    
    def _run_single_experiment_for_pair(self, method: str, lang_pair: str, base_model: str, name: str):
        """Run a single experiment for a specific language pair"""
        config = create_experiment_config(
            name=name,
            compression_method=method,
            lang_pair=lang_pair,
            base_model=base_model
        )
        
        LOG.info(f"Running experiment: {name}")
        result = self.experiment_runner.run_experiment(config)
        
        # Print result for this pair
        if "error" not in result.quality_scores:
            print(f"\nExperiment Results for {name}:")
            print(f"  Compression Ratio: {result.compression_ratio:.2f}x")
            print(f"  Model Size: {result.model_size_mb:.1f} MB")
            print(f"  Inference Time: {result.inference_time_ms:.1f} ms")
            print(f"  Quality Score: {result.get_quality_score():.2f}")
        else:
            print(f"\nExperiment {name} failed: {result.quality_scores['error']}")
        
        return result
    
    def _print_multi_pair_results(self, method: str, results: list):
        """Print summary results for multiple language pairs"""
        print(f"\n{'='*60}")
        print(f"SUMMARY: {method.upper()} COMPRESSION ACROSS ALL LANGUAGE PAIRS")
        print(f"{'='*60}")
        
        successful = 0
        total_compression = 0
        total_quality = 0
        
        for lang_pair, result in results:
            if "error" not in result.quality_scores:
                successful += 1
                total_compression += result.compression_ratio
                total_quality += result.get_quality_score()
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            print(f"{lang_pair:12} | {status} | "
                  f"Ratio: {result.compression_ratio:.2f}x | "
                  f"Size: {result.model_size_mb:.1f}MB | "
                  f"Quality: {result.get_quality_score():.2f}")
        
        if successful > 0:
            avg_compression = total_compression / successful
            avg_quality = total_quality / successful
            print(f"\nAVERAGE RESULTS ({successful}/{len(results)} successful):")
            print(f"  Average Compression Ratio: {avg_compression:.2f}x")
            print(f"  Average Quality Score: {avg_quality:.2f}")
        
        print(f"{'='*60}")
    
    def run_quick_test_with_method(self, method: str, lang_pair: str = None, base_model: str = None):
        """Run quick test with specified method and optional language pair
        
        Args:
            method: Compression method to use
            lang_pair: Language pair to test (if None, uses ces-deu as default)
            base_model: Custom base model (if None, uses default)
        """
        if base_model is None:
            base_model = CONSTRAINED_TASK["base_model"]
        
        if lang_pair is None:
            lang_pair = "ces-deu"  # Default for quick test
        
        LOG.info(f"Running quick test: {method} on {lang_pair} with {base_model}")
        
        config = create_experiment_config(
            name=f"quick_test_{method}_{lang_pair}",
            compression_method=method,
            lang_pair=lang_pair,
            base_model=base_model,
            eval_params={"max_samples": 10}  # Quick test with limited samples
        )
        
        result = self.experiment_runner.run_experiment(config)
        
        # Print result
        print(f"\n{'='*50}")
        print(f"QUICK TEST RESULTS")
        print(f"{'='*50}")
        print(f"Method: {method}")
        print(f"Language Pair: {lang_pair}")
        print(f"Base Model: {base_model}")
        print(f"{'-'*50}")
        
        if "error" not in result.quality_scores:
            print(f"✅ Test completed successfully!")
            print(f"  Compression Ratio: {result.compression_ratio:.2f}x")
            print(f"  Model Size: {result.model_size_mb:.1f} MB")
            print(f"  Inference Time: {result.inference_time_ms:.1f} ms")
            print(f"  Quality Score: {result.get_quality_score():.2f}")
        else:
            print(f"❌ Test failed: {result.quality_scores['error']}")
        
        print(f"{'='*50}")
        return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="WMT25 Constrained Model Compression Task Runner")
    
    # Main pipeline options
    parser.add_argument("--setup-data", action="store_true", help="Setup training and test data")
    parser.add_argument("--setup-models", action="store_true", help="Setup base models")
    parser.add_argument("--run-experiments", action="store_true", help="Run compression experiments")
    parser.add_argument("--analyze", action="store_true", help="Analyze experiment results")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    
    # Testing and targeted experiment options
    parser.add_argument("--quick-test", action="store_true", help="Run minimal experiments for testing")
    parser.add_argument("--single-experiment", action="store_true", help="Run single method experiment")
    
    # Method and configuration options
    parser.add_argument("--method", help="Compression method to use (required for --quick-test and --single-experiment)")
    parser.add_argument("--lang-pair", help="Language pair (optional, defaults: ces-deu for quick-test, all pairs for single-experiment)")
    parser.add_argument("--base-model", help="Custom base model (optional, uses Aya Expanse 8B by default)")
    
    # Utility options
    parser.add_argument("--list-methods", action="store_true", help="List available compression methods")
    parser.add_argument("--stats", action="store_true", help="Print data statistics")
    parser.add_argument("--workdir", type=Path, help="Working directory for experiments")
    
    # Legacy single experiment option (maintained for compatibility)
    parser.add_argument("--single", help="Legacy: Run single experiment with method:lang_pair format")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ConstrainedTaskRunner(args.workdir)
    
    # Validate method requirement for quick-test and single-experiment
    if (args.quick_test or args.single_experiment) and not args.method:
        parser.error("--method is required when using --quick-test or --single-experiment")
    
    # Execute requested actions
    if args.setup_data:
        runner.setup_data()
    
    if args.setup_models:
        runner.setup_models()
    
    if args.run_experiments:
        runner.run_all_experiments(args.quick_test)
    
    if args.analyze:
        runner.analyze_results()
    
    if args.all:
        runner.setup_data()
        runner.setup_models()
        runner.run_all_experiments(args.quick_test)
        runner.analyze_results()
    
    if args.quick_test:
        if args.method:
            runner.run_quick_test_with_method(args.method, args.lang_pair, args.base_model)
        else:
            runner.run_all_experiments(quick_test=True)
    
    if args.single_experiment:
        runner.run_single_experiment(args.method, args.lang_pair, args.base_model)
    
    if args.list_methods:
        runner.list_available_methods()
    
    if args.stats:
        runner.print_data_statistics()
    
    # Legacy support for --single
    if args.single:
        try:
            method, lang_pair = args.single.split(":")
            runner.run_single_experiment(method, lang_pair)
        except ValueError:
            print("Error: --single format should be method:lang_pair (e.g., baseline:ces-deu)")


if __name__ == "__main__":
    main() 