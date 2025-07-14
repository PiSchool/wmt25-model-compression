# Layer Pruning Module

A comprehensive layer-based pruning solution for transformer models that combines similarity analysis with intelligent layer merging.

## Core Components

### 1. LayerSimilarityAnalyzer
- **Purpose**: Wraps PruneMe's layer similarity analysis
- **Key Methods**: `analyze_layers()`, `get_layer_recommendation()`
- **Output**: Layer distances and similarity recommendations

### 2. LayerMerger  
- **Purpose**: Wraps mergekit functionality for layer removal
- **Key Methods**: `merge_layers()`, `create_merge_config()`
- **Support**: Both slicing and weighted merging methods

### 3. LayerPruningCompressor
- **Purpose**: Main orchestrator for the complete workflow
- **Key Methods**: `compress_model()`, `get_compression_summary()`
- **Features**: Automatic layer detection, size reduction calculation, validation

### 4. CLI Interface
- **Purpose**: Command-line tool for easy usage
- **Features**: All parameters configurable, verbose logging, results export
- **Usage**: `python -m modelzip.compression.pruning.cli --help`

### 5. Configuration System
- **Purpose**: Manages default settings via YAML configuration
- **Features**: Configurable defaults for models, datasets, processing parameters
- **File**: `config.yaml` in the module directory

## Installation

### Prerequisites

1. **PruneMe Repository**: Clone PruneMe in the parent directory
   ```bash
   cd ..
   git clone https://github.com/your-org/PruneMe.git
   cd PruneMe
   pip install -r requirements.txt
   pip install -U datasets
   ```

2. **mergekit**: Install mergekit for layer merging
   ```bash
   git clone https://github.com/cg123/mergekit.git
   cd mergekit
   pip install -e .
   ```

3. **PyYAML**: Install PyYAML for configuration file support
   ```bash
   pip install PyYAML
   ```

### Setup

The module automatically detects and validates the PruneMe setup on initialization.

### Configuration

The module uses a YAML configuration file (`config.yaml`) to manage default settings. You can customize this file to change default models, datasets, and processing parameters.

```yaml
# Default configuration for Layer Pruning Module
model:
  default_path: "CohereLabs/aya-expanse-8b"
  output_suffix: "pruned"

dataset:
  default_name: "arcee-ai/sec-data-mini"
  default_column: "text"
  default_subset: "train"
  default_size: 4000

processing:
  batch_size: 8
  max_length: 1024
  layers_to_skip: 28

merging:
  default_method: "slicing"
  supported_methods: ["slicing", "weighted"]

paths:
  pruneme_path: "../PruneMe"
  temp_dir: null  # Auto-created if null

output:
  save_results: false
  validate_model: false
  verbose_logging: false
```

## Quick Start

### Basic Usage
```bash
# Command line
python -m modelzip.compression.pruning.cli \
  --model microsoft/DialoGPT-medium \
  --output ./pruned_dialogpt

# Python API
from modelzip.compression.pruning import LayerPruningCompressor, PruningConfig

config = PruningConfig(
    model_path="CohereLabs/aya-expanse-8b",
    output_path="./pruned_aya8b"
)

compressor = LayerPruningCompressor()
result = compressor.compress_model(config)
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
python -m modelzip.compression.pruning.cli \
  --model CohereLabs/aya-expanse-8b \
  --output ./pruned_aya8b
```

#### Advanced Usage

```bash
python -m modelzip.compression.pruning.cli \
  --model CohereLabs/aya-expanse-8b \
  --output ./pruned_aya8b \
  --dataset arcee-ai/sec-data-mini \
  --dataset-size 2000 \
  --batch-size 4 \
  --layers-to-skip 28 \
  --merge-method slicing \
  --validate \
  --results results.json
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model path or name | Required |
| `--output`, `-o` | Output path for pruned model | Required |
| `--dataset`, `-d` | Dataset for analysis | `arcee-ai/sec-data-mini` |
| `--dataset-column` | Text column name | `text` |
| `--batch-size`, `-b` | Processing batch size | `8` |
| `--max-length` | Maximum sequence length | `1024` |
| `--layers-to-skip` | Layer block size | `28` |
| `--dataset-size` | Number of samples to process | `4000` |
| `--dataset-subset` | Dataset subset | `train` |
| `--merge-method` | Merge method (slicing/weighted) | `slicing` |
| `--temp-dir` | Temporary directory | Auto-created |
| `--pruneme-path` | PruneMe repository path | `../PruneMe` |
| `--results`, `-r` | Save detailed results to JSON | None |
| `--validate` | Validate pruned model | False |
| `--verbose`, `-v` | Enable verbose logging | False |

### Python API

#### Basic Usage

```python
from modelzip.compression.pruning import LayerPruningCompressor, PruningConfig, config_loader

# Create configuration with defaults from config file
config = PruningConfig(
    model_path="CohereLabs/aya-expanse-8b",
    output_path="./pruned_aya8b"
    # All other parameters will use defaults from config.yaml
)

# Or use the config loader to create configuration
config_dict = config_loader.create_pruning_config(
    model_path="CohereLabs/aya-expanse-8b",
    output_path="./pruned_aya8b",
    dataset_size=2000,
    batch_size=4
)
config = PruningConfig(**config_dict)
```

# Initialize compressor
compressor = LayerPruningCompressor()

# Run compression
result = compressor.compress_model(config)

# Check results
if result.success:
    print(f"✅ Compression successful!")
    print(f"Removed layers: {result.removed_layers}")
    print(f"Size reduction: {result.model_size_reduction}%")
    print(f"Pruned model: {result.pruned_model_path}")
else:
    print(f"❌ Compression failed: {result.error_message}")
```

#### Advanced Usage

```python
from modelzip.compression.pruning import (
    LayerPruningCompressor, 
    PruningConfig,
    LayerSimilarityAnalyzer,
    LayerMerger
)

# Step 1: Analyze layer similarity
analyzer = LayerSimilarityAnalyzer()
success, analysis_results = analyzer.analyze_layers(
    model_path="CohereLabs/aya-expanse-8b",
    dataset="arcee-ai/sec-data-mini",
    dataset_size=2000,
    batch_size=4
)

if success:
    # Get layer recommendation
    start_layer, end_layer, distance = analyzer.get_layer_recommendation(analysis_results)
    print(f"Recommended to remove layers {start_layer} to {end_layer}")
    
    # Step 2: Merge layers
    merger = LayerMerger()
    merge_success, merge_results = merger.merge_layers(
        model_path="CohereLabs/aya-expanse-8b",
        output_path="./pruned_aya8b",
        start_layer=start_layer,
        end_layer=end_layer,
        merge_method="slicing"
    )
    
    if merge_success:
        print("✅ Layer merging completed successfully!")
```

## API Reference

### LayerPruningCompressor

Main orchestrator class that combines analysis and merging.

#### Methods

- `compress_model(config: PruningConfig) -> PruningResult`
- `get_compression_summary(result: PruningResult) -> Dict`
- `validate_pruned_model(result: PruningResult) -> bool`

### LayerSimilarityAnalyzer

Analyzes layer similarity to identify redundant layers.

#### Methods

- `analyze_layers(...) -> Tuple[bool, Optional[Dict]]`
- `get_layer_recommendation(results: Dict) -> Tuple[int, int, float]`

### LayerMerger

Merges models by removing specified layers.

#### Methods

- `merge_layers(...) -> Tuple[bool, Optional[Dict]]`
- `create_merge_config(...) -> str`
- `validate_merged_model(...) -> bool`

### Data Classes

#### PruningConfig

Configuration for the pruning process. All parameters except `model_path` and `output_path` use defaults from the configuration file.

```python
@dataclass
class PruningConfig:
    model_path: str
    output_path: str
    dataset: str = None  # Uses config default
    dataset_column: str = None  # Uses config default
    batch_size: int = None  # Uses config default
    max_length: int = None  # Uses config default
    layers_to_skip: int = None  # Uses config default
    dataset_size: int = None  # Uses config default
    dataset_subset: str = None  # Uses config default
    merge_method: str = None  # Uses config default
    temp_dir: Optional[str] = None  # Uses config default
```

#### ConfigLoader

Manages configuration loading and provides default values.

```python
from modelzip.compression.pruning import ConfigLoader, config_loader

# Use global instance
default_model = config_loader.get_default_model_path()
default_dataset = config_loader.get_default_dataset()

# Or create custom instance
custom_loader = ConfigLoader("path/to/custom/config.yaml")
```

#### PruningResult

Results from the pruning process.

```python
@dataclass
class PruningResult:
    success: bool
    original_model_path: str
    pruned_model_path: Optional[str]
    removed_layers: Optional[Tuple[int, int]]
    layer_distance: Optional[float]
    model_size_reduction: Optional[float]
    analysis_results: Optional[Dict]
    merge_results: Optional[Dict]
    error_message: Optional[str] = None
```

## Examples

### Example 1: Basic Compression

```python
from modelzip.compression.pruning import LayerPruningCompressor, PruningConfig

config = PruningConfig(
    model_path="CohereLabs/aya-expanse-8b",
    output_path="./pruned_aya8b"
)

compressor = LayerPruningCompressor()
result = compressor.compress_model(config)

if result.success:
    summary = compressor.get_compression_summary(result)
    print(f"Size reduction: {summary['compression_details']['size_reduction']}")
```

### Example 2: Custom Analysis

```python
from modelzip.compression.pruning import LayerSimilarityAnalyzer

analyzer = LayerSimilarityAnalyzer()
success, results = analyzer.analyze_layers(
    model_path="CohereLabs/aya-expanse-8b",
    dataset="arcee-ai/sec-data-mini",
    dataset_column="text",
    dataset_size=1000,
    batch_size=2,
    layers_to_skip=24
)

if success:
    start, end, distance = analyzer.get_layer_recommendation(results)
    print(f"Most similar layers: {start} to {end} (distance: {distance:.4f})")
```

### Example 3: Custom Merging

```python
from modelzip.compression.pruning import LayerMerger

merger = LayerMerger()
success, results = merger.merge_layers(
    model_path="CohereLabs/aya-expanse-8b",
    output_path="./custom_pruned",
    start_layer=12,
    end_layer=15,
    merge_method="weighted"
)

if success:
    print(f"Model merged successfully: {results['merged_model_path']}")
```

## Output Format

### Successful Compression

```
============================================================
✅ LAYER PRUNING COMPLETED SUCCESSFULLY!
============================================================
Original Model: CohereLabs/aya-expanse-8b
Pruned Model:   ./pruned_aya8b
Removed Layers: 12 to 15
Layer Distance: 0.0234
Size Reduction: 12.50%
Analysis:       Remove layers 12 to 15 (distance: 0.0234)
Merge Method:   Slicing merge completed successfully
============================================================
```

### Detailed Results JSON

```json
{
  "success": true,
  "original_model_path": "CohereLabs/aya-expanse-8b",
  "pruned_model_path": "./pruned_aya8b",
  "removed_layers": [12, 15],
  "layer_distance": 0.0234,
  "model_size_reduction": 12.5,
  "analysis_results": {
    "layer_distances": [...],
    "most_similar_layers": {
      "start": 12,
      "end": 15,
      "distance": 0.0234
    },
    "statistics": {...},
    "recommendation": "Remove layers 12 to 15 (distance: 0.0234)"
  },
  "merge_results": {
    "merged_model_path": "./pruned_aya8b",
    "summary": "Slicing merge completed successfully",
    "config_file": "temp_merge_config.yaml"
  }
}
```

## Configuration Management

### Using Configuration Files

The module supports configuration through YAML files. The default configuration is loaded from `config.yaml` in the module directory.

#### Custom Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
model:
  default_path: "your-model/name"
  output_suffix: "compressed"

dataset:
  default_name: "your-dataset"
  default_column: "content"
  default_size: 1000

processing:
  batch_size: 4
  max_length: 512
```

Use custom configuration:

```python
from modelzip.compression.pruning import ConfigLoader, LayerPruningCompressor

# Load custom config
custom_loader = ConfigLoader("custom_config.yaml")

# Create compressor with custom config
compressor = LayerPruningCompressor(
    pruneme_path=custom_loader.get_pruneme_path()
)

# Use custom defaults
config = PruningConfig(
    model_path="your-model",
    output_path="./output"
    # All other parameters use custom defaults
)
```

#### Environment-Specific Configuration

You can create different configuration files for different environments:

- `config_dev.yaml` - Development settings
- `config_prod.yaml` - Production settings
- `config_test.yaml` - Testing settings

## Troubleshooting

### Common Issues

1. **PruneMe not found**: Ensure PruneMe is cloned in the parent directory
2. **mergekit not installed**: Install mergekit with `pip install mergekit`
3. **PyYAML not installed**: Install PyYAML with `pip install PyYAML`
4. **Out of memory**: Reduce batch size or dataset size
5. **Model loading errors**: Check model path and format
6. **Configuration errors**: Check YAML syntax in config files

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python -m modelzip.compression.pruning.cli \
  --model your-model \
  --output ./output \
  --verbose
```

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive error handling
3. Include docstrings for all public methods
4. Add tests for new functionality
5. Update this README for new features

## Future Work

### WMT25 Framework Integration

The layer pruning module is currently standalone and ready for use. To fully integrate with the WMT25 framework, the following enhancements are planned:

#### 1. Extend BaseCompressor Interface
```python
class LayerPruningCompressor(BaseCompressor):
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        # Integration with WMT25 BaseCompressor interface
        # This would require adapting the current implementation
```

#### 2. Update Compression Factory
```python
def get_compressor(config: ExperimentConfig) -> BaseCompressor:
    # ... existing logic ...
    elif method == "layer_pruning":
        from .pruning.layer_pruning_compressor import LayerPruningCompressor
        return LayerPruningCompressor(config)
```

#### 3. Add to Configuration System
```python
COMPRESSION_CONFIG = {
    # ... existing config ...
    "layer_pruning": {
        "dataset": "arcee-ai/sec-data-mini",
        "dataset_column": "text",
        "batch_size": 8,
        "max_length": 1024,
        "layers_to_skip": 28,
        "dataset_size": 4000,
        "dataset_subset": "train"
    }
}
```

### Additional Enhancements

1. **Additional Merge Methods** - Support for more mergekit methods
2. **Batch Processing** - Process multiple models simultaneously
3. **Performance Optimization** - GPU acceleration and memory optimization
4. **Advanced Analysis** - More sophisticated layer similarity metrics
5. **Configuration Files** - Support for YAML/JSON configuration files

## License

This module is part of the WMT25 Model Compression framework. 