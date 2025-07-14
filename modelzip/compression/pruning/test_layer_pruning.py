#!/usr/bin/env python
"""
Test script for Layer Pruning Module

This script tests the basic functionality of the layer pruning components
without requiring actual model processing (which would be expensive).
"""

import sys
import logging as LOG
from pathlib import Path
from unittest.mock import Mock, patch

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelzip.compression.pruning import (
    LayerSimilarityAnalyzer,
    LayerMerger,
    LayerPruningCompressor,
    PruningConfig,
    PruningResult
)

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_layer_similarity_analyzer():
    """Test LayerSimilarityAnalyzer initialization and basic functionality"""
    print("Testing LayerSimilarityAnalyzer...")
    
    try:
        # Test initialization
        analyzer = LayerSimilarityAnalyzer()
        print("✅ LayerSimilarityAnalyzer initialized successfully")
        
        # Test get_layer_recommendation with mock data
        mock_results = {
            "most_similar_layers": {
                "start": 12,
                "end": 15,
                "distance": 0.0234
            }
        }
        
        start, end, distance = analyzer.get_layer_recommendation(mock_results)
        assert start == 12
        assert end == 15
        assert distance == 0.0234
        print("✅ get_layer_recommendation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ LayerSimilarityAnalyzer test failed: {e}")
        return False


def test_layer_merger():
    """Test LayerMerger initialization and basic functionality"""
    print("Testing LayerMerger...")
    
    try:
        # Test initialization
        merger = LayerMerger()
        print("✅ LayerMerger initialized successfully")
        
        # Test create_merge_config
        config_content = merger.create_merge_config(
            model_path="test_model",
            output_path="test_output",
            start_layer=12,
            end_layer=15,
            merge_method="slicing"
        )
        
        assert "test_model" in config_content
        assert "passthrough" in config_content  # merge_method for slicing
        assert "bfloat16" in config_content
        assert "layer_range: [12,15]" in config_content
        print("✅ create_merge_config works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ LayerMerger test failed: {e}")
        return False


def test_layer_pruning_compressor():
    """Test LayerPruningCompressor initialization and basic functionality"""
    print("Testing LayerPruningCompressor...")
    
    try:
        # Test initialization
        compressor = LayerPruningCompressor()
        print("✅ LayerPruningCompressor initialized successfully")
        
        # Test PruningConfig
        config = PruningConfig(
            model_path="test_model",
            output_path="test_output",
            dataset="test_dataset",
            batch_size=4
        )
        
        assert config.model_path == "test_model"
        assert config.output_path == "test_output"
        assert config.dataset == "test_dataset"
        assert config.batch_size == 4
        print("✅ PruningConfig works correctly")
        
        # Test PruningResult
        result = PruningResult(
            success=True,
            original_model_path="test_model",
            pruned_model_path="test_output",
            removed_layers=(12, 15),
            layer_distance=0.0234,
            model_size_reduction=12.5,
            analysis_results={},
            merge_results={}
        )
        
        assert result.success is True
        assert result.removed_layers == (12, 15)
        assert result.layer_distance == 0.0234
        print("✅ PruningResult works correctly")
        
        # Test get_compression_summary
        summary = compressor.get_compression_summary(result)
        assert summary["status"] == "success"
        assert "12 to 15" in summary["compression_details"]["removed_layers"]
        print("✅ get_compression_summary works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ LayerPruningCompressor test failed: {e}")
        return False


def test_mock_compression_workflow():
    """Test the complete compression workflow with mocked components"""
    print("Testing complete compression workflow (mocked)...")
    
    try:
        # Create mock analyzer and merger
        mock_analyzer = Mock()
        mock_merger = Mock()
        
        # Mock analyzer results
        mock_analysis_results = {
            "most_similar_layers": {
                "start": 12,
                "end": 15,
                "distance": 0.0234
            },
            "recommendation": "Remove layers 12 to 15 (distance: 0.0234)"
        }
        
        mock_analyzer.analyze_layers.return_value = (True, mock_analysis_results)
        mock_analyzer.get_layer_recommendation.return_value = (12, 15, 0.0234)
        
        # Mock merger results
        mock_merge_results = {
            "merged_model_path": "test_output",
            "summary": "Slicing merge completed successfully"
        }
        mock_merger.merge_layers.return_value = (True, mock_merge_results)
        
        # Create compressor with mocked components
        compressor = LayerPruningCompressor()
        compressor.analyzer = mock_analyzer
        compressor.merger = mock_merger
        
        # Test compression
        config = PruningConfig(
            model_path="test_model",
            output_path="test_output"
        )
        
        with patch.object(compressor, '_calculate_size_reduction', return_value=12.5):
            result = compressor.compress_model(config)
        
        assert result.success is True
        assert result.removed_layers == (12, 15)
        assert result.layer_distance == 0.0234
        assert result.model_size_reduction == 12.5
        
        print("✅ Complete compression workflow works correctly (mocked)")
        return True
        
    except Exception as e:
        print(f"❌ Compression workflow test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TESTING LAYER PRUNING MODULE")
    print("=" * 60)
    
    tests = [
        test_layer_similarity_analyzer,
        test_layer_merger,
        test_layer_pruning_compressor,
        test_mock_compression_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n{'='*40}")
        if test():
            passed += 1
        print(f"{'='*40}")
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The layer pruning module is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main()) 