# Layer Pruning Module

A layer-based pruning solution for transformer models that removes redundant layers using similarity analysis and intelligent merging.

## Overview

This module combines PruneMe's layer similarity analysis with mergekit's layer merging capabilities to compress transformer models by removing redundant layers while maintaining performance.

## Quick Start

### Prerequisites

1. **PruneMe**: Clone in parent directory and install requirements
   ```bash
   cd ..
   git clone https://github.com/your-org/PruneMe.git
   cd PruneMe && pip install -r requirements.txt
   ```

2. **mergekit**: Install for layer merging
   ```bash
   git clone https://github.com/cg123/mergekit.git
   cd mergekit && pip install -e .
   ```

3. **PyYAML**: Install for configuration
   ```bash
   pip install PyYAML
   ```

### Basic Usage

```bash
# Basic compression
python -m modelzip.compression.pruning.cli \
  --model CohereLabs/aya-expanse-8b \
  --output pruned_aya8b

# With custom parameters
python -m modelzip.compression.pruning.cli \
  --model CohereLabs/aya-expanse-8b \
  --output pruned_aya8b \
  --dataset arcee-ai/sec-data-mini \
  --dataset-size 2000 \
  --batch-size 4 \
  --merge-method slicing \
  --validate
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model path or name | Required |
| `--output`, `-o` | Output path for pruned model | Required |
| `--dataset` | Dataset for analysis | `arcee-ai/sec-data-mini` |
| `--dataset-size` | Number of samples to process | `4000` |
| `--batch-size` | Processing batch size | `8` |
| `--merge-method` | Merge method (slicing/weighted) | `slicing` |
| `--validate` | Validate pruned model | False |
| `--verbose`, `-v` | Enable verbose logging | False |

## Module Files

### Core Components

- **`layer_pruning_compressor.py`** - Main orchestrator that combines analysis and merging
- **`layer_similarity_analyzer.py`** - Wraps PruneMe to analyze layer similarity and identify redundant layers
- **`layer_merger.py`** - Wraps mergekit to remove specified layers from the model

### Interface & Configuration

- **`cli.py`** - Command-line interface for easy usage
- **`config_loader.py`** - Manages YAML configuration and default settings
- **`config.yaml`** - Default configuration file with model, dataset, and processing parameters

### Integration & Testing

- **`__init__.py`** - Module initialization and exports
- **`pruning_compressor.py`** - Legacy interface (for backward compatibility)
- **`test_layer_pruning.py`** - Unit tests for the module

## Output

Successful compression produces:
- Pruned model in the specified output directory
- Summary showing removed layers, size reduction, and layer distance
- Optional detailed results JSON file
- Optional model validation

Example output:
```
✅ LAYER PRUNING COMPLETED SUCCESSFULLY!
Original Model: CohereLabs/aya-expanse-8b
Pruned Model:   workdir/results/pruned_aya8b
Removed Layers: 12 to 15
Layer Distance: 0.0234
Size Reduction: 12.50%
```

## Configuration

The module uses `config.yaml` for default settings. You can customize models, datasets, and processing parameters by editing this file or using CLI options to override defaults.

## Output Directory Structure

All outputs are automatically organized under `workdir/results/` (relative to the project root):
- **Pruned models**: `workdir/results/{output_name}/` (e.g., `workdir/results/pruned_aya8b/`)
- **Temporary files**: `workdir/results/temp_pruning/` and `workdir/results/temp_merge/`
- **Analysis results**: `workdir/results/temp_pruning/analysis/`
- **Detailed results**: JSON files saved to specified path with `--results` flag 