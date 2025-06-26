# WMT25 Model Compression Constrained Task Framework

This framework provides a comprehensive solution for conducting model compression experiments for the WMT25 Model Compression Shared Task, specifically focused on the constrained track. **Optimized for CUDA Linux environments.**

## 🎯 Constrained Task Overview

The constrained track focuses on:
- **Base Model**: Aya Expanse 8B (`CohereLabs/aya-expanse-8b`)
- **Language Pairs**: Czech→German, Japanese→Chinese, English→Arabic
- **Evaluation Criteria**: Quality, Model Size, Inference Speed
- **Compression Methods**: Quantization, Pruning, Distillation, Mixed Precision
- **Hardware**: CUDA GPU Linux systems

## 📁 Project Structure

```
wmt25-model-compression/
├── modelzip/
│   ├── core/                      # 🔧 Core framework components
│   │   ├── base_models.py         # Model abstractions
│   │   ├── base_compressor.py     # Compressor interfaces
│   │   ├── experiment_config.py   # Configuration dataclasses
│   │   └── experiment_runner.py   # Experiment orchestration
│   ├── compression/               # 🗜️ Compression techniques
│   │   ├── quantization/          # 8-bit/4-bit quantization
│   │   ├── pruning/               # Structured/magnitude pruning
│   │   ├── distillation/          # Knowledge distillation
│   │   ├── mixed_precision/       # FP16/BF16 compression
│   │   └── compression_factory.py # Factory pattern
│   ├── evaluation/                # 📊 Evaluation system
│   │   └── evaluator.py           # CHRF/BLEU metrics
│   ├── utils/                     # 🛠️ Utilities
│   │   └── model_utils.py         # Model management
│   ├── constrained_config.py      # Configuration for constrained task
│   ├── data_manager.py            # Data downloading and preprocessing
│   └── constrained_runner.py      # Main orchestration script
├── workdir/                       # Generated during setup
│   ├── data/                      # Training data
│   ├── tests/                     # Test data
│   ├── models/                    # Base and compressed models
│   ├── experiments/               # Experiment outputs
│   └── results/                   # Final results and reports
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install the package
pip install -e .

# Install additional dependencies for constrained task
pip install pandas psutil unbabel-comet tabulate bitsandbytes
```

### 2. Quick Test

First, test that the framework is properly set up:

```bash
# Comprehensive framework validation (recommended)
python test_new_structure.py
```

This should show:
```
🎉 All tests passed! The new modular structure is working correctly.
```

Then run a quick test to verify everything works:

```bash
python -m modelzip.constrained_runner --quick-test
```

This will:
- Download and cache the Aya Expanse 8B model locally
- Download test data for Czech→German
- Compress the model using 8-bit quantization
- Evaluate on a small test set
- Report compression ratio, model size, and quality scores

### 3. Full Pipeline

Run the complete pipeline for all constrained language pairs and methods:

```bash
python -m modelzip.constrained_runner --all
```

This will:
1. Download and preprocess all training and test data
2. Run compression experiments for all combinations
3. Generate a comprehensive comparison report

### 4. Single Experiments & Reports

For targeted experiments and immediate reporting:

```bash
# Quick test (10 samples, fast iteration)
python -m modelzip.constrained_runner --quick-test --method baseline

# Full single experiment
python -m modelzip.constrained_runner --single-experiment --lang-pair ces-deu --method quantization_8bit

# Generate report from any existing results
python -m modelzip.constrained_runner --analyze
```

**Available methods:** `baseline`, `quantization_8bit`, `quantization_4bit`, `pruning_magnitude`, `pruning_structured`, `distillation_response`, `distillation_feature`, `mixed_precision_fp16`, `mixed_precision_bf16`

**Report outputs:**
- 📄 `workdir/results/final_report.md` - Markdown report with analysis
- 📊 `workdir/results/comparison.csv` - CSV data for further analysis
- 📋 `workdir/results/*_results.json` - Individual experiment details

## 🖥️ Hardware Requirements

This framework is optimized for **CUDA Linux environments**:

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA capability 7.0+ 
- **CUDA**: CUDA 11.8+ or CUDA 12.x
- **RAM**: 32GB+ system memory
- **VRAM**: 24GB+ GPU memory for Aya Expanse 8B
- **Storage**: 100GB+ free space

### Recommended Configuration
- **GPU**: RTX 4090, A6000, or H100
- **VRAM**: 48GB+ for comfortable experimentation
- **RAM**: 64GB+ for large-scale experiments
- **Storage**: SSD with 500GB+ free space

### CUDA Setup Verification
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

## 📊 Individual Components

### Data Management

```bash
# Download and preprocess data
python -c "from modelzip.data_manager import DataManager; dm = DataManager(); dm.setup_training_data()"

# Get data statistics
python -c "from modelzip.data_manager import DataManager; dm = DataManager(); print(dm.get_data_statistics())"
```

#### Data Cleaning and Preprocessing

The framework applies comprehensive data cleaning techniques to ensure high-quality training data:

**1. Data Type Normalization**
- Converts all data to string type to avoid pandas processing issues

**2. Null and Empty Value Removal**
- Removes rows with null/NaN values
- Removes rows with empty strings (after trimming whitespace)
- Removes rows where conversion resulted in "nan" strings

**3. Length-Based Filtering**
- **Minimum length**: Removes very short sentences (< 5 characters)
- **Maximum length**: Removes very long sentences (> 512 characters)
- Applied to both source and target sentences

**4. Length Ratio Filtering**
- Removes sentence pairs with extreme length differences
- **Ratio bounds**: Source/target length ratio must be between 0.3 and 3.0
- Prevents keeping misaligned pairs like "Hello" → "Very long detailed translation..."

**5. Deduplication**
- Removes exact duplicate sentence pairs after all other cleaning

**Cleaning Results:**

| Language Pair | Original Size | After Cleaning | Data Loss |
|---------------|---------------|----------------|-----------|
| Czech-German  | 243,816       | 241,159        | 1.1% |
| Japanese-Chinese | 1,643      | 1,626          | 1.0% |
| English-Arabic | 163,198      | 135,809        | 16.8% |

**Quality Benefits:**
- Improved translation quality through aligned sentence pairs
- Better model training stability with consistent input lengths
- More reliable evaluation metrics with clean test data
- Enhanced efficiency by removing noisy data

The cleaning parameters can be customized in the `preprocess_training_data()` method:
```python
# Customize cleaning parameters
processed_files = data_manager.preprocess_training_data(
    train_files, 
    max_length=1024,  # Increase max length
    min_length=10     # Increase min length
)
```

### Running Experiments

```bash
# Setup data only
python -m modelzip.constrained_runner --setup-data

# Setup models only (download and cache Aya Expanse 8B)
python -m modelzip.constrained_runner --setup-models

# Run experiments only (requires data and model setup)
python -m modelzip.constrained_runner --run-all-experiments

# Analyze results and generate report
python -m modelzip.constrained_runner --analyze
```

### Single Experiment Mode

For targeted testing or research, you can run individual experiments:

#### **Quick Test (10 samples)**
```bash
# Test with default settings (ces-deu, quantization_8bit)
python -m modelzip.constrained_runner --quick-test

# Test specific language pair and method
python -m modelzip.constrained_runner --quick-test --lang-pair jpn-zho --method baseline

# Test with different base model
python -m modelzip.constrained_runner --quick-test --base-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

#### **Full Single Experiment**
```bash
# Run full test data for one experiment
python -m modelzip.constrained_runner --single-experiment --lang-pair ces-deu --method quantization_8bit

# Run across multiple language pairs with one method
python -m modelzip.constrained_runner --single-experiment --method baseline

# Use custom base model
python -m modelzip.constrained_runner --single-experiment --base-model "your-model/path"
```

#### **Single Experiment Options**
- **`--lang-pair`**: Specify language pair (ces-deu, jpn-zho, eng-ara) or leave empty for all pairs
- **`--method`**: Compression method (baseline, quantization_8bit, quantization_4bit, etc.)
- **`--base-model`**: Override base model (useful for testing with smaller models)
- **`--quick-test`**: Use only 10 test samples for fast iteration
- **`--single-experiment`**: Use full test data for accurate evaluation

**Example Workflows:**

```bash
# Research iteration: Fast testing with small model
python -m modelzip.constrained_runner --quick-test --base-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --method baseline

# Production evaluation: Full test with target model  
python -m modelzip.constrained_runner --single-experiment --lang-pair ces-deu --method quantization_8bit

# Method comparison: Test all pairs with specific method
python -m modelzip.constrained_runner --single-experiment --method quantization_4bit
```

### Model Management

The framework includes comprehensive model utilities for caching and management:

```python
from modelzip.model_utils import (
    setup_base_model, 
    get_model_cache_path, 
    is_model_cached,
    verify_model_access,
    prepare_constrained_models
)

# Check if model is already cached
if is_model_cached("CohereLabs/aya-expanse-8b"):
    cache_path = get_model_cache_path("CohereLabs/aya-expanse-8b")
    print(f"Model cached at: {cache_path}")

# Setup and verify model access
model_path = setup_base_model("CohereLabs/aya-expanse-8b")
if verify_model_access(str(model_path)):
    print("✅ Model ready for use")

# Prepare all models for constrained task
models = prepare_constrained_models()
print(f"Base model ready: {models['base']}")
```

## 🔧 Configuration

### Main Configuration File: `modelzip/constrained_config.py`

Key configurations:
- **Language pairs**: Only constrained pairs
- **Compression methods**: Quantization, pruning, distillation
- **Model settings**: Batch size, max length, device mapping
- **Training parameters**: Learning rate, epochs, gradient accumulation
- **Evaluation settings**: Beam size, temperature, max tokens

### Compression Methods

The framework supports several compression techniques:

1. **Quantization**
   - 8-bit quantization using BitsAndBytes
   - 4-bit quantization with compute dtype optimization

2. **Pruning** (extensible)
   - Structured pruning with configurable sparsity
   - Unstructured pruning

3. **Distillation** (extensible)
   - Knowledge distillation with temperature scaling

## 🧪 Framework Testing

The framework includes two complementary test suites:

### 1. Framework Component Tests (Unit/Integration - No External Dependencies)

Test the framework code itself without requiring internet, models, or data:

```bash
# Run comprehensive framework tests (fast, offline)
python test_new_structure.py
```

This comprehensive test suite validates:
- ✅ **All imports and dependencies** - Ensures all packages are properly installed
- ✅ **Configuration validation** - Tests all configuration files and settings
- ✅ **Data manager functionality** - Validates data handling without downloading
- ✅ **Experiment configuration** - Tests experiment setup and config creation
- ✅ **Compressor class creation** - Validates all compression method classes
- ✅ **Model utilities** - Tests model cache paths and setup functions
- ✅ **Constrained runner** - Validates main orchestration class
- ✅ **Custom compression examples** - Tests extensibility framework

**Expected Output:**
```
🎯 TEST SUMMARY
==========================================
Imports              ✅ PASS
Configuration        ✅ PASS
Data Manager         ✅ PASS
Experiment Config    ✅ PASS
Compressor Classes   ✅ PASS
Experiment Runner    ✅ PASS
Constrained Runner   ✅ PASS
Model Utils          ✅ PASS
Custom Compression   ✅ PASS
TOTAL: 9/9 tests passed (100.0%)
🎉 All tests passed! The new modular structure is working correctly.
```

### 2. Environment/Setup Tests (External Dependencies Required)

Test external connections and environment setup:

```bash
# Test environment setup (requires internet access)
python test_setup.py
```

This test suite validates:
- ✅ **Configuration setup** - Tests config loading and directory creation
- ✅ **Model access** - Tests connection to HuggingFace model repository
- ✅ **Data manager with real data** - Tests data statistics on actual files
- ✅ **Experiment configuration** - Tests experiment setup with real config

**Expected Output:**
```
🎉 All tests completed!

📋 Next steps:
1. Run: python -m modelzip.constrained_runner --setup-models
2. Run: python -m modelzip.constrained_runner --quick-test
3. Run: python -m modelzip.constrained_runner --all
```

### Recommended Testing Order

1. **Run framework tests first** (fast, no external dependencies):
   ```bash
   python test_new_structure.py
   ```

2. **If tests pass, proceed with data and model setup**:
   ```bash
   python -m modelzip.constrained_runner --setup-data
   python -m modelzip.constrained_runner --setup-models
   ```

## 🔧 Developer Guide

### Framework Architecture

**Core Components:**
- **`BaseModel`**: Abstract interface for model wrappers (HuggingFace, ONNX, etc.)
- **`BaseCompressor`**: Abstract interface for compression methods
- **`BaseEvaluator`**: Abstract interface for evaluation methods
- **`ExperimentRunner`**: Orchestrates complete experiments
- **`DataManager`**: Handles data downloading and preprocessing

### 1. Creating Custom Compression Methods

#### Step 1: Inherit from BaseCompressor

```python
from modelzip.core.base_compressor import BaseCompressor
from modelzip.core.base_models import BaseModel
from pathlib import Path
import torch

class PruningCompressor(BaseCompressor):
    """Example: Structured pruning compressor"""
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        """Implement your compression logic"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Example: Structured pruning
        import torch.nn.utils.prune as prune
        
        # Get the actual PyTorch model
        pytorch_model = model.model
        
        # Apply pruning to Linear layers
        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_structured(
                    module, 
                    name='weight', 
                    amount=self.config.compression_params.get('sparsity', 0.3),
                    dim=0  # Prune output channels
                )
                prune.remove(module, 'weight')  # Make pruning permanent
        
        # Save compressed model
        pytorch_model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        # Return new model wrapper
        from modelzip.core.base_models import HuggingFaceModel
        from modelzip.core.experiment_config import ExperimentConfig
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_pruned",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair,
            compression_params=self.config.compression_params
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def get_compression_ratio(self, original_model: BaseModel, compressed_model: BaseModel) -> float:
        """Calculate compression ratio"""
        original_size = original_model.get_model_size()
        compressed_size = compressed_model.get_model_size()
        return original_size / compressed_size if compressed_size > 0 else 1.0
```

#### Step 2: Register Your Compressor

Update `get_compressor()` in `modelzip/compression/compression_factory.py`:

```python
def get_compressor(config: ExperimentConfig) -> BaseCompressor:
    """Factory function to create appropriate compressor instance"""
    method = config.compression_method.lower()
    
    # Add your new compression method
    if method.startswith("pruning_custom"):
        from .pruning.custom_pruning_compressor import CustomPruningCompressor
        return CustomPruningCompressor(config)
    elif method.startswith("quantization"):
        from .quantization.quantization_compressor import QuantizationCompressor
        return QuantizationCompressor(config)
    # ... rest of factory logic
```

### 2. Knowledge Distillation Example

```python
class DistillationCompressor(BaseCompressor):
    """Knowledge distillation compressor"""
    
    def compress(self, model: BaseModel, output_path: Path) -> BaseModel:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Create smaller student model
        teacher_config = model.model.config
        student_config = AutoConfig.from_pretrained(model.model_path)
        
        # Reduce model size (example: half the layers)
        student_config.num_hidden_layers = teacher_config.num_hidden_layers // 2
        student_config.num_attention_heads = teacher_config.num_attention_heads // 2
        
        # Initialize student model
        student_model = AutoModelForCausalLM.from_config(student_config)
        
        # Training loop for distillation (simplified)
        self._distill_knowledge(model.model, student_model)
        
        # Save student model
        output_path.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        # Return compressed model
        from modelzip.core.experiment_config import ExperimentConfig
        from modelzip.core.base_models import HuggingFaceModel
        compressed_config = ExperimentConfig(
            name=f"{self.config.name}_distilled",
            compression_method=self.config.compression_method,
            lang_pair=self.config.lang_pair
        )
        
        return HuggingFaceModel(output_path, compressed_config)
    
    def _distill_knowledge(self, teacher_model, student_model):
        """Implement knowledge distillation training"""
        # This would contain your distillation training loop
        # Using temperature scaling, loss functions, etc.
        pass
```

### 3. Using Fine-tuned Models

#### Option A: Replace Base Model

```python
# Update constrained_config.py
CONSTRAINED_TASK = {
    "base_model": "your-username/aya-expanse-8b-finetuned-ces-deu",  # Your fine-tuned model
    # ... rest of config
}
```

#### Option B: Use Different Models per Language Pair

```python
from modelzip.core.experiment_config import create_experiment_config

# Create configs with different base models
configs = [
    create_experiment_config(
        name="ces_deu_finetuned",
        compression_method="quantization_8bit",
        lang_pair="ces-deu",
        base_model="your-username/aya-expanse-ces-deu-ft"
    ),
    create_experiment_config(
        name="jpn_zho_finetuned", 
        compression_method="quantization_8bit",
        lang_pair="jpn-zho",
        base_model="your-username/aya-expanse-jpn-zho-ft"
    )
]
```

### 4. Custom Model Types

```python
class ONNXModel(BaseModel):
    """ONNX Runtime model wrapper"""
    
    def load_model(self):
        import onnxruntime as ort
        return ort.InferenceSession(str(self.model_path))
    
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(self.model_path.parent))
    
    def get_model_size(self) -> float:
        """Get ONNX model size in MB"""
        if self.model_path.exists():
            return self.model_path.stat().st_size / (1024 * 1024)
        return 0.0
```

### 5. Custom Evaluation Metrics

```python
class CustomEvaluator(TranslationEvaluator):
    """Evaluator with custom metrics"""
    
    def evaluate(self, model: BaseModel, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        # Get standard metrics
        scores = super().evaluate(model, test_data)
        
        # Add custom metrics
        scores["custom_fluency"] = self._calculate_fluency(model, test_data)
        scores["custom_adequacy"] = self._calculate_adequacy(model, test_data)
        
        return scores
    
    def _calculate_fluency(self, model: BaseModel, test_data: List[Dict[str, str]]) -> float:
        """Your custom fluency metric"""
        # Implement your metric
        return 0.85
    
    def _calculate_adequacy(self, model: BaseModel, test_data: List[Dict[str, str]]) -> float:
        """Your custom adequacy metric"""
        # Implement your metric
        return 0.78
```

### 6. Running Custom Experiments

```python
from modelzip.experiment_base import create_experiment_config
from modelzip.constrained_runner import ConstrainedTaskRunner

# Setup runner
runner = ConstrainedTaskRunner()
runner.setup_data(download_test=True, download_train=False)

# Create custom experiments
experiments = [
    create_experiment_config(
        name="pruning_30_ces_deu",
        compression_method="pruning_structured",
        lang_pair="ces-deu",
        compression_params={"sparsity": 0.3}
    ),
    create_experiment_config(
        name="distillation_ces_deu",
        compression_method="distillation",
        lang_pair="ces-deu",
        compression_params={"temperature": 4.0, "alpha": 0.7}
    )
]

# Run experiments
results = []
for config in experiments:
    test_data = runner._load_test_data_for_pair(config.lang_pair)
    result = runner.experiment_runner.run_experiment(config, test_data[:50])  # Small test
    results.append(result)

# Analyze results
comparison = runner.experiment_runner.compare_experiments([
    runner.results_dir / f"{config.name}_results.json" for config in experiments
])
print(comparison)
```

### 7. Batch Processing Multiple Models

```python
def run_compression_sweep():
    """Run experiments across multiple compression methods and parameters"""
    
    compression_methods = [
        ("quantization_8bit", {}),
        ("quantization_4bit", {}), 
        ("pruning_structured", {"sparsity": 0.2}),
        ("pruning_structured", {"sparsity": 0.5}),
        ("distillation", {"temperature": 3.0}),
        ("distillation", {"temperature": 5.0}),
    ]
    
    lang_pairs = ["ces-deu", "jpn-zho", "eng-ara"]
    
    runner = ConstrainedTaskRunner()
    runner.setup_data()
    
    all_configs = []
    for lang_pair in lang_pairs:
        for method, params in compression_methods:
            config = create_experiment_config(
                name=f"{method}_{lang_pair}_{hash(str(params)) % 1000}",
                compression_method=method,
                lang_pair=lang_pair,
                compression_params=params
            )
            all_configs.append(config)
    
    # Run all experiments
    runner.run_all_experiments(all_configs)
    
    # Generate comprehensive report
    runner.analyze_results()

if __name__ == "__main__":
    run_compression_sweep()
```

## 📈 Results and Analysis

### 📊 Report Generation

The framework automatically generates comprehensive reports after experiments. Use any of these methods:

#### **Method 1: Generate Report from Existing Results**
```bash
# Analyze any existing experiments and generate report
python -m modelzip.constrained_runner --analyze
```

#### **Method 2: Complete Pipeline with Report**
```bash
# Run everything and generate final report
python -m modelzip.constrained_runner --all
```

#### **Method 3: After Any Individual Experiment**
```bash
# Run a quick test
python -m modelzip.constrained_runner --quick-test

# Generate report immediately
python -m modelzip.constrained_runner --analyze
```

### 📋 Report Contents

The framework generates comprehensive reports including:

#### **📄 Markdown Report** (`workdir/results/final_report.md`)
```markdown
# WMT25 Model Compression Constrained Task Results

## Overview
- Total experiments: 3
- Language pairs: ces-deu, jpn-zho, eng-ara
- Compression methods: baseline, quantization_8bit, quantization_4bit

## Results Summary

### All Results
| experiment | compression_method | lang_pair | model_size_mb | compression_ratio | inference_time_ms | quality_chrf |
|------------|-------------------|-----------|---------------|-------------------|-------------------|--------------|
| baseline_ces-deu | baseline | ces-deu | 8192.5 | 1.00x | 1250.3 | 45.2 |
| quant8_ces-deu | quantization_8bit | ces-deu | 4096.2 | 2.00x | 1050.1 | 43.8 |

### Results by Language Pair
#### ces-deu
- Best compression ratio: 2.00x (quantization_8bit)
- Best quality (CHRF): 45.2 (baseline)
- Fastest inference: 1050.1ms (quantization_8bit)

## Recommendations
- **ces-deu**: quantization_8bit (compression: 2.00x, quality: 43.8)
```

#### **📊 CSV Data** (`workdir/results/comparison.csv`)
Machine-readable format for further analysis:
```csv
experiment,compression_method,lang_pair,model_size_mb,compression_ratio,inference_time_ms,memory_usage_mb,quality_chrf
baseline_ces-deu,baseline,ces-deu,8192.5,1.0,1250.3,2048.1,45.2
quant8_ces-deu,quantization_8bit,ces-deu,4096.2,2.0,1050.1,1024.0,43.8
```

#### **📋 Individual Results** (`workdir/results/*_results.json`)
Detailed JSON files for each experiment with full configuration and metrics.

### 🎯 Report Features

#### **Automatic Analysis**
- **Best Compression Ratio**: Highest compression while maintaining quality
- **Best Quality**: Highest translation quality scores
- **Fastest Inference**: Lowest inference time per sentence
- **Balanced Recommendations**: Optimal trade-offs between quality and efficiency

#### **Multi-Experiment Comparison**
```bash
# Run multiple experiments
python -m modelzip.constrained_runner --single-experiment --method baseline
python -m modelzip.constrained_runner --single-experiment --method quantization_8bit  
python -m modelzip.constrained_runner --single-experiment --method quantization_4bit

# Generate comparative report
python -m modelzip.constrained_runner --analyze
```

#### **Report Customization**
Modify report generation in `constrained_runner.py`:
```python
def generate_report(self, output_file: Path = None):
    # Customize report format, metrics, and analysis
    # Add custom scoring functions
    # Include additional visualizations
```

### Output File Locations

All reports are saved in `workdir/results/`:

- **📄 `final_report.md`**: Comprehensive markdown report with analysis
- **📊 `comparison.csv`**: CSV data for spreadsheet analysis  
- **📋 `*_results.json`**: Individual experiment results with full details
- **📈 Additional files**: Any custom reports or visualizations you add

### Metrics Tracked

#### **Efficiency Metrics:**
- **Model size (MB)**: Compressed model file size
- **Memory usage (MB)**: GPU memory during inference
- **Inference time (ms)**: Time per sentence translation
- **Compression ratio**: Original size / Compressed size

#### **Quality Metrics:**
- **CHRF score**: Character-level translation quality
- **COMET score**: Neural quality assessment (when available)
- **Custom metrics**: Any additional metrics you implement

### 📈 Using Reports for Decision Making

#### **For Submission Selection**
```bash
# Generate comprehensive report
python -m modelzip.constrained_runner --analyze

# Review results in final_report.md to choose:
# - Best overall performers
# - Language-specific specialists  
# - Speed vs. quality trade-offs
```

#### **For Research Analysis**
```python
import pandas as pd

# Load results for analysis
df = pd.read_csv('workdir/results/comparison.csv')

# Custom analysis
efficiency_score = df['compression_ratio'] * (1000 / df['inference_time_ms'])
quality_score = df['quality_chrf'] / 100
overall_score = efficiency_score * quality_score

df['overall_score'] = overall_score
best_models = df.groupby('lang_pair')['overall_score'].idxmax()
print(df.loc[best_models])
```

## 🎛️ Advanced Usage

### Custom Data Sources

Modify `CONSTRAINED_TASK["data_sources"]` in `constrained_config.py` to use different datasets:

```python
"data_sources": {
    "ces-deu": {
        "test": {"custom_test": "your_command_here"},
        "train": "your_training_data_command",
    }
}
```

### Custom Evaluation Metrics

Extend `TranslationEvaluator` to add new metrics:

```python
class CustomEvaluator(TranslationEvaluator):
    def evaluate(self, model: BaseModel, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        scores = super().evaluate(model, test_data)
        # Add your custom metrics
        scores["custom_metric"] = self.calculate_custom_metric(model, test_data)
        return scores
```

### Batch Processing

For multiple experiments:

```python
configs = [
    create_experiment_config("exp1", "quantization_8bit", "ces-deu"),
    create_experiment_config("exp2", "quantization_4bit", "ces-deu"),
    # ... more configs
]

runner = ConstrainedTaskRunner()
runner.setup_data()
runner.run_all_experiments(configs)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `MODEL_CONFIG`
   - Use gradient checkpointing
   - Try smaller models first

2. **Data Download Failures**
   - Check internet connection
   - Verify mtdata installation: `pip install mtdata==0.4.3`
   - Check dataset availability with: `mtdata list -l ces-deu`

3. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`
   - For COMET: `pip install unbabel-comet`

4. **Model Path Not Found Errors**
   ```bash
   # Setup models first
   python -m modelzip.constrained_runner --setup-models
   
   # Verify model cache
   python -c "from modelzip.model_utils import is_model_cached; print(is_model_cached('CohereLabs/aya-expanse-8b'))"
   ```

5. **Test Framework Failures**
   ```bash
   # Run with verbose output to diagnose
   python test_new_structure.py
   
   # Check specific component
   python -c "from modelzip.constrained_config import CONSTRAINED_TASK; print('Config OK')"
   ```

6. **Data Cleaning Results in Empty Dataset**
   - Check original data quality: may need to adjust cleaning parameters
   - Modify cleaning thresholds in `data_manager.preprocess_training_data()`
   - Some language pairs (like eng-ara) have higher data loss due to quality issues

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [WMT25 Model Compression Task](https://www2.statmt.org/wmt25/model-compression.html)
- [WMT25 Training Data](https://www2.statmt.org/wmt25/mtdata/)
- [Aya Expanse 8B Model](https://huggingface.co/CohereLabs/aya-expanse-8b)

## 🤝 Contributing

To extend the framework:

1. Add new compression methods by extending `BaseCompressor`
2. Add new evaluation metrics by extending `BaseEvaluator`
3. Add new model types by extending `BaseModel`
4. Update configuration files as needed
5. Add tests for new functionality

## 🚀 Submission Instructions

### 🎯 Quick Submission Workflow

**Complete commands for end-to-end submission:**

```bash
# 1. Test framework (no GPU needed)
python test_new_structure.py

# 2. Setup data and models
python -m modelzip.constrained_runner --setup-data
python -m modelzip.constrained_runner --setup-models  

# 3. Run all experiments (creates submission-ready models)
python -m modelzip.constrained_runner --all

# 4. Analyze results to choose best models
python -m modelzip.constrained_runner --analyze

# 5. Verify submission structure
ls workdir/experiments/*/compressed_model/run.sh

# 6. Build Docker submission
docker build -t yourteam-wmt25-submission .

# 7. Save and package Docker image
docker save --output yourteam-wmt25-submission.tar yourteam-wmt25-submission
sha512sum yourteam-wmt25-submission.tar > yourteam-wmt25-submission.tar.sha512

# 8. Upload yourteam-wmt25-submission.tar and .sha512 files to your hosting service
```

### Prerequisites for Submission

1. **Complete the constrained task pipeline**:
   ```bash
   # Test framework
   python test_new_structure.py
   
   # Setup data and models 
   python -m modelzip.constrained_runner --setup-data
   python -m modelzip.constrained_runner --setup-models
   
   # Run experiments
   python -m modelzip.constrained_runner --all
   ```

2. **Generate submission-ready models**:
   ```bash
   # Run experiments - this creates models with run.sh scripts
   python -m modelzip.constrained_runner --run-all-experiments
   ```

### Submission Format Integration

Our framework **fully integrates** with the organizers' submission requirements:

#### **`run.sh` Integration** ✅
- **Automatic**: All compressed models get `run.sh` scripts copied automatically
- **Compatible**: Uses organizers' original `run.sh` that calls `baseline.py`
- **Interface**: Follows required format: `run.sh $lang_pair $batch_size < input.txt > output.txt`

#### **`baseline.py` Integration** ✅  
- **Preserved**: Original `baseline.py` with `LLMWrapper` class remains functional
- **Extended**: Our framework adds `TranslationEvaluator` with similar functionality
- **Batching**: Original batching optimizations preserved in `baseline.py`

#### **Model Directory Structure**
After running experiments, you'll have submission-ready models:
```
workdir/experiments/
├── quantization_8bit_ces-deu/
│   └── compressed_model/
│       ├── pytorch_model.bin
│       ├── config.json  
│       ├── tokenizer.json
│       └── run.sh          # ✅ Submission-ready
├── quantization_4bit_jpn-zho/
│   └── compressed_model/
│       ├── pytorch_model.bin
│       ├── config.json
│       ├── tokenizer.json
│       └── run.sh          # ✅ Submission-ready
└── baseline_eng-ara/
    └── compressed_model/
        ├── pytorch_model.bin
        ├── config.json
        ├── tokenizer.json
        └── run.sh          # ✅ Submission-ready
```

### Creating Docker Submission

1. **Choose your best model(s)**:
   ```bash
   # Analyze results to pick best performers
   python -m modelzip.constrained_runner --analyze
   ```

2. **Prepare Dockerfile**:
   ```dockerfile
   # Use the existing Dockerfile as base
   FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
   
   # ... (existing setup) ...
   
   # Copy your best compressed model(s)
   COPY workdir/experiments/quantization_8bit_ces-deu/compressed_model /model/team-ces-deu-8bit
   COPY workdir/experiments/quantization_4bit_jpn-zho/compressed_model /model/team-jpn-zho-4bit  
   COPY workdir/experiments/baseline_eng-ara/compressed_model /model/team-eng-ara-baseline
   
   # Test the models
   RUN bash /model/team-ces-deu-8bit/run.sh ces-deu 1 <<< "Test sentence."
   ```

3. **Build and save Docker image**:
   ```bash
   # Build image
   docker build -t yourteam-wmt25-submission .
   
   # Save image
   docker save --output yourteam-wmt25-submission.tar yourteam-wmt25-submission
   
   # Calculate checksum
   sha512sum yourteam-wmt25-submission.tar > yourteam-wmt25-submission.tar.sha512
   ```

### Submission Requirements Compliance

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **Dockerfile with team name** | ✅ | Modify existing Dockerfile |
| **Model at `/model/$submission_id`** | ✅ | Copy from `workdir/experiments/*/compressed_model/` |
| **`run.sh` script with correct interface** | ✅ | Automatically generated by framework |
| **Works offline** | ✅ | All models cached locally |
| **Input/output via stdin/stdout** | ✅ | Original `baseline.py` handles this |
| **Language pair and batch size args** | ✅ | Original `run.sh` handles this |

### Testing Your Submission

#### **Local Testing (Memory Requirements)**
The Aya Expanse 8B model requires significant GPU memory (~16-20GB). For local testing:

```bash
# 1. Framework testing (no model loading)
python test_new_structure.py

# 2. Test interface structure (from root directory)
cd /path/to/wmt25-model-compression
ls workdir/experiments/*/compressed_model/run.sh  # Verify run.sh exists

# 3. Test with smaller test data (if you have sufficient GPU memory)
echo "This is a test sentence." | bash workdir/experiments/quantization_8bit_ces-deu/compressed_model/run.sh ces-deu 1
```

#### **Production Testing (Docker Environment)**
For full testing, use the Docker environment with appropriate GPU resources:

```bash
# Build Docker image with GPU support
docker build -t yourteam-submission .

# Test with Docker (requires NVIDIA Docker)
echo "Test sentence" | docker run --gpus all -i yourteam-submission /model/your-model-id/run.sh ces-deu 1
```

**Note**: Memory errors on local machines are expected due to the 8B model size. The submission will work correctly in the evaluation environment with appropriate hardware.

### Integration Summary

**✅ Full Compatibility**: Our framework builds on and extends the organizers' scripts without replacing them:

- **`setup.py`**: Integrated via `model_utils.py`
- **`baseline.py`**: Used directly by `run.sh` 
- **`evaluate.py`**: Extended via `TranslationEvaluator`
- **`compress.py`**: Extended via `QuantizationCompressor`
- **`run.sh`**: Copied to all compressed models
- **`config.py`**: Extended via `constrained_config.py`
- **`report.py`**: Extended via framework reporting

## 📄 License

MIT License - see LICENSE file for details. 

## Aya Expanse Model Setup (Gated Model)

The default Aya Expanse 8B model requires gated access. If you encounter authentication issues:

### Option 1: Setup Aya Expanse with Authentication

1. **Accept License Agreement**:
   - Visit https://huggingface.co/CohereLabs/aya-expanse-8b
   - Accept the license agreement terms

2. **Get HuggingFace Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" access
   - Copy the token

3. **Set Environment Variable**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

### Option 2: Use Alternative Models

For immediate testing without gated access, use these alternatives:

**Small Models for Testing:**
```bash
# Quick test with TinyLlama (1.1B)
python -m modelzip.constrained_runner --quick-test --method mixed_precision_fp16 --base-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Single experiment across all pairs
python -m modelzip.constrained_runner --single-experiment --method mixed_precision_fp16 --base-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Larger Open Models:**
```bash
# Mistral 7B (no gating)
python -m modelzip.constrained_runner --single-experiment --method mixed_precision_fp16 --base-model "mistralai/Mistral-7B-v0.1"

# Llama 3.2 3B (no gating)  
python -m modelzip.constrained_runner --single-experiment --method mixed_precision_fp16 --base-model "meta-llama/Llama-3.2-3B"
```

### Error Resolution

If you see errors like:
```
Unrecognized model in workdir/models/CohereLabs_aya-expanse-8b. Should have a `model_type` key in its config.json
```

This indicates:
1. **Gated Access Issue**: Model download failed due to missing authentication
2. **Empty Model Directory**: The model folder exists but is empty
3. **Network Issues**: Download was interrupted

**Solution:**
1. Remove the empty model directory:
   ```bash
   rm -rf workdir/models/CohereLabs_aya-expanse-8b
   ```

2. Set your HF token and try again:
   ```bash
   export HF_TOKEN=your_token_here
   python -m modelzip.constrained_runner --single-experiment --method mixed_precision_fp16
   ```

3. Or use an alternative model:
   ```bash
   python -m modelzip.constrained_runner --single-experiment --method mixed_precision_fp16 --base-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ``` 