#!/usr/bin/env python
"""
Test Script for New Modular WMT25 Model Compression Framework

Tests all components of the restructured framework to ensure
proper modularization and functionality.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        # Core imports
        from modelzip.core import (
            BaseModel, HuggingFaceModel, BaseCompressor,
            ExperimentConfig, ExperimentResults, ExperimentRunner
        )
        print("✅ Core modules imported successfully")
        
        # Compression techniques
        from modelzip.compression import (
            QuantizationCompressor, PruningCompressor,
            DistillationCompressor, MixedPrecisionCompressor,
            get_compressor, list_available_methods
        )
        print("✅ Compression modules imported successfully")
        
        # Evaluation
        from modelzip.evaluation import TranslationEvaluator
        print("✅ Evaluation module imported successfully")
        
        # Utilities
        from modelzip.utils import setup_model, verify_model
        print("✅ Utilities imported successfully")
        
        # Data management
        from modelzip.data_manager import DataManager
        print("✅ Data manager imported successfully")
        
        # Configuration
        from modelzip.constrained_config import CONSTRAINED_TASK, COMPRESSION_CONFIG
        print("✅ Configuration imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_factory_pattern():
    """Test compression factory pattern"""
    print("\nTesting compression factory pattern...")
    
    try:
        from modelzip.compression import get_compressor, list_available_methods
        from modelzip.core import ExperimentConfig
        
        # Test listing methods
        methods = list_available_methods()
        print(f"Available methods: {list(methods.keys())}")
        
        # Test creating compressors
        test_configs = [
            ("baseline", "ces-deu"),
            ("quantization_8bit", "jpn-zho"),
            ("pruning_magnitude", "eng-ara"),
            ("mixed_precision_fp16", "ces-deu")
        ]
        
        for method, lang_pair in test_configs:
            config = ExperimentConfig(
                name=f"test_{method}",
                compression_method=method,
                lang_pair=lang_pair
            )
            
            compressor = get_compressor(config)
            print(f"✅ Created {type(compressor).__name__} for {method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test experiment configuration"""
    print("\nTesting experiment configuration...")
    
    try:
        from modelzip.core import ExperimentConfig, create_experiment_config
        
        # Test manual creation
        config1 = ExperimentConfig(
            name="test_exp",
            compression_method="quantization_8bit",
            lang_pair="ces-deu"
        )
        
        print(f"✅ Created config: {config1.name}")
        print(f"   Method: {config1.compression_method}")
        print(f"   Lang pair: {config1.lang_pair}")
        
        # Test helper function
        config2 = create_experiment_config(
            name="test_exp2",
            compression_method="pruning_magnitude",
            lang_pair="jpn-zho"
        )
        
        print(f"✅ Created config via helper: {config2.name}")
        
        # Test serialization
        config_dict = config1.to_dict()
        config_restored = ExperimentConfig.from_dict(config_dict)
        
        assert config_restored.name == config1.name
        print("✅ Configuration serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_model_abstractions():
    """Test model abstraction layer"""
    print("\nTesting model abstractions...")
    
    try:
        from modelzip.core import HuggingFaceModel
        from modelzip.constrained_config import CONSTRAINED_TASK
        
        # Test model wrapper creation (don't actually load model)
        model_path = Path("dummy_model_path")
        model = HuggingFaceModel(model_path)
        
        print(f"✅ Created model wrapper for path: {model.model_path}")
        print(f"   Is HF model ID: {model._is_hf_model_id}")
        
        # Test with actual HF model ID
        hf_model = HuggingFaceModel(Path(CONSTRAINED_TASK["base_model"]))
        print(f"✅ Created model wrapper for HF ID: {hf_model.model_path}")
        print(f"   Is HF model ID: {hf_model._is_hf_model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model abstraction test failed: {e}")
        traceback.print_exc()
        return False


def test_compression_techniques():
    """Test individual compression technique classes"""
    print("\nTesting compression technique classes...")
    
    try:
        from modelzip.compression.quantization import QuantizationCompressor
        from modelzip.compression.pruning import PruningCompressor
        from modelzip.compression.distillation import DistillationCompressor
        from modelzip.compression.mixed_precision import MixedPrecisionCompressor
        from modelzip.core import ExperimentConfig
        
        # Create test config
        config = ExperimentConfig(
            name="test_compression",
            compression_method="quantization_8bit",
            lang_pair="ces-deu"
        )
        
        # Test each compressor class
        compressors = [
            QuantizationCompressor(config),
            PruningCompressor(config),
            DistillationCompressor(config),
            MixedPrecisionCompressor(config)
        ]
        
        for compressor in compressors:
            print(f"✅ Created {type(compressor).__name__}")
            
            # Test that they inherit from BaseCompressor
            from modelzip.core import BaseCompressor
            assert isinstance(compressor, BaseCompressor)
        
        print("✅ All compressor classes inherit from BaseCompressor")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression techniques test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation_system():
    """Test evaluation system"""
    print("\nTesting evaluation system...")
    
    try:
        from modelzip.evaluation import TranslationEvaluator
        
        # Test evaluator creation
        evaluator = TranslationEvaluator("ces-deu")
        print(f"✅ Created evaluator for language pair: {evaluator.lang_pair}")
        print(f"   Prompts loaded: {bool(evaluator.prompts)}")
        
        # Test metric calculation methods
        references = ["Hello world", "How are you?"]
        hypotheses = ["Hello world", "How are you doing?"]
        
        chrf_score = evaluator._calculate_chrf(references, hypotheses)
        bleu_score = evaluator._calculate_bleu(references, hypotheses)
        
        print(f"✅ CHRF score calculated: {chrf_score:.2f}")
        print(f"✅ BLEU score calculated: {bleu_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation system test failed: {e}")
        traceback.print_exc()
        return False


def test_runner_framework():
    """Test experiment runner framework"""
    print("\nTesting experiment runner framework...")
    
    try:
        from modelzip.core import ExperimentRunner, ExperimentConfig
        from pathlib import Path
        
        # Create test runner with temporary directory
        test_workdir = Path("test_workdir")
        runner = ExperimentRunner(test_workdir)
        
        print(f"✅ Created experiment runner with workdir: {runner.workdir}")
        print(f"   Experiments dir: {runner.experiments_dir}")
        
        # Test configuration creation
        configs = []
        for method in ["baseline", "quantization_8bit"]:
            for lang_pair in ["ces-deu", "jpn-zho"]:
                config = ExperimentConfig(
                    name=f"test_{method}_{lang_pair}",
                    compression_method=method,
                    lang_pair=lang_pair
                )
                configs.append(config)
        
        print(f"✅ Created {len(configs)} test configurations")
        
        # Test analysis methods (without actual experiments)
        analysis = runner.analyze_results()
        print(f"✅ Analysis method callable: {type(analysis)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Runner framework test failed: {e}")
        traceback.print_exc()
        return False


def test_main_runner():
    """Test main constrained runner"""
    print("\nTesting main constrained runner...")
    
    try:
        from modelzip.constrained_runner import ConstrainedTaskRunner
        
        # Create runner
        runner = ConstrainedTaskRunner()
        print(f"✅ Created ConstrainedTaskRunner")
        print(f"   Workdir: {runner.workdir}")
        
        # Test method availability
        methods = [
            'setup_data', 'setup_models', 'run_all_experiments',
            'analyze_results', 'list_available_methods'
        ]
        
        for method_name in methods:
            assert hasattr(runner, method_name), f"Missing method: {method_name}"
            print(f"✅ Method available: {method_name}")
        
        # Test configuration creation
        configs = runner._create_experiment_configs(quick_test=True)
        print(f"✅ Created {len(configs)} quick test configurations")
        
        return True
        
    except Exception as e:
        print(f"❌ Main runner test failed: {e}")
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that the new directory structure is correct"""
    print("\nTesting directory structure...")
    
    try:
        expected_structure = {
            "modelzip/core/": [
                "__init__.py", "base_models.py", "base_compressor.py",
                "experiment_config.py", "experiment_runner.py"
            ],
            "modelzip/compression/": [
                "__init__.py", "compression_factory.py"
            ],
            "modelzip/compression/quantization/": [
                "__init__.py", "quantization_compressor.py"
            ],
            "modelzip/compression/pruning/": [
                "__init__.py", "pruning_compressor.py"
            ],
            "modelzip/compression/distillation/": [
                "__init__.py", "distillation_compressor.py"
            ],
            "modelzip/compression/mixed_precision/": [
                "__init__.py", "mixed_precision_compressor.py"
            ],
            "modelzip/evaluation/": [
                "__init__.py", "evaluator.py"
            ],
            "modelzip/utils/": [
                "__init__.py", "model_utils.py"
            ]
        }
        
        for directory, files in expected_structure.items():
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"❌ Missing directory: {directory}")
                return False
                
            for file_name in files:
                file_path = dir_path / file_name
                if not file_path.exists():
                    print(f"❌ Missing file: {file_path}")
                    return False
        
        print("✅ All expected directories and files exist")
        
        # Check that old files are removed
        old_files = [
            "modelzip/experiment_base.py",
            "modelzip/model_utils.py"
        ]
        
        for old_file in old_files:
            if Path(old_file).exists():
                print(f"❌ Old file still exists: {old_file}")
                return False
        
        print("✅ Old files properly removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 Testing New Modular WMT25 Framework Structure")
    print("="*60)
    
    tests = [
        test_directory_structure,
        test_imports,
        test_configuration,
        test_model_abstractions,
        test_factory_pattern,
        test_compression_techniques,
        test_evaluation_system,
        test_runner_framework,
        test_main_runner
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The new modular structure is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main() 