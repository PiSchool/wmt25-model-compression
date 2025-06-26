# WMT25 Model Compression Framework

A modular compression framework for the WMT25 Model Compression Shared Task constrained track, optimized for the **Aya Expanse 8B model** on CUDA Linux systems.

## 🎯 Overview

- **Target Model**: Aya Expanse 8B (`CohereLabs/aya-expanse-8b`)
- **Language Pairs**: Czech→German, Japanese→Chinese, English→Arabic  
- **Compression Methods**: Quantization, Pruning, Distillation, Mixed Precision
- **Hardware**: CUDA GPU Linux environments

## 📁 Project Structure

```
modelzip/
├── core/                      # Framework foundation
│   ├── base_models.py         # Model abstractions
│   ├── base_compressor.py     # Compressor interfaces  
│   ├── experiment_config.py   # Configuration management
│   └── experiment_runner.py   # Experiment orchestration
├── compression/               # Compression techniques
│   ├── quantization/          # 8-bit/4-bit quantization
│   ├── pruning/               # Magnitude/structured pruning
│   ├── distillation/          # Knowledge distillation
│   ├── mixed_precision/       # FP16/BF16 compression
│   └── compression_factory.py # Method factory
├── evaluation/                # Evaluation system
└── utils/                     # Utilities and helpers
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Setup Authentication (Required for Aya Expanse)

The Aya Expanse 8B model requires gated access:

1. Visit https://huggingface.co/CohereLabs/aya-expanse-8b and accept the license
2. Get your HF token from https://huggingface.co/settings/tokens  
3. Set environment variable:
```bash
export HF_TOKEN=your_token_here
```

### 3. Test the Framework

```bash
# Validate framework setup
python test_new_structure.py

# Quick test (10 samples)
python -m modelzip.constrained_runner --quick-test

# Full pipeline (all language pairs and methods)
python -m modelzip.constrained_runner --all
```

## 🖥️ Hardware Requirements

**Minimum:**
- Ubuntu 18.04+, CUDA 11.8+
- NVIDIA GPU with 24GB+ VRAM
- 32GB+ RAM, 100GB+ storage

**Recommended:**  
- RTX 4090/A6000/H100 with 48GB+ VRAM
- 64GB+ RAM, 500GB+ SSD storage

## 🧪 Usage Examples

### Quick Testing
```bash
# Test with default settings (ces-deu, quantization_8bit)
python -m modelzip.constrained_runner --quick-test

# Test specific method
python -m modelzip.constrained_runner --quick-test --method mixed_precision_fp16
```

### Single Experiments
```bash
# One language pair
python -m modelzip.constrained_runner --single-experiment --lang-pair ces-deu --method quantization_8bit

# All language pairs with one method
python -m modelzip.constrained_runner --single-experiment --method baseline
```

### Available Methods
- `baseline` - Uncompressed model
- `quantization_8bit`, `quantization_4bit` - BitsAndBytes quantization
- `pruning_magnitude`, `pruning_structured` - Weight pruning
- `distillation_response`, `distillation_feature` - Knowledge distillation  
- `mixed_precision_fp16`, `mixed_precision_bf16` - Mixed precision

### Analysis and Reports
```bash
# Generate comprehensive report
python -m modelzip.constrained_runner --analyze
```

Results saved to:
- `workdir/results/final_report.md` - Analysis report
- `workdir/results/comparison.csv` - Data export
- `workdir/experiments/*/` - Individual model outputs

## 🔧 Configuration

Main settings in `modelzip/constrained_config.py`:
- **Language pairs** and prompts
- **Compression parameters** for each method
- **Evaluation settings** (batch size, max samples)
- **Model configuration** (device mapping, precision)

## 🏗️ Extending the Framework

### Adding New Compression Methods

1. Create compressor class inheriting from `BaseCompressor`:
```python
from modelzip.core.base_compressor import BaseCompressor

class MyCompressor(BaseCompressor):
    def compress(self, model, config):
        # Your compression logic
        pass
```

2. Register in `compression_factory.py`:
```python
def get_compressor(method: str):
    methods = {
        # ... existing methods ...
        "my_method": MyCompressor,
    }
```

3. Add configuration in `constrained_config.py`:
```python
COMPRESSION_CONFIG = {
    # ... existing config ...
    "my_method": {"param1": "value1"},
}
```

## 🧪 Testing

```bash
# Framework validation (fast, offline)
python test_new_structure.py

# CUDA verification
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📊 Evaluation Features

- **Progress bars** for all evaluation phases
- **CHRF and BLEU** quality metrics
- **Compression ratio** and model size tracking
- **Inference speed** measurement
- **Multi-language** result aggregation

## 🐳 Submission Integration

The framework generates submission-ready models:

```bash
# Run experiments
python -m modelzip.constrained_runner --all

# Check submission structure
ls workdir/experiments/*/compressed_model/run.sh
```

Each compressed model includes:
- `pytorch_model.bin`, `config.json`, `tokenizer.json`
- `run.sh` script compatible with evaluation requirements

## 🔍 Troubleshooting

**Gated Model Issues:**
```bash
# Clear empty model cache
rm -rf workdir/models/CohereLabs_aya-expanse-8b

# Verify token and retry
export HF_TOKEN=your_token_here
python -m modelzip.constrained_runner --quick-test
```

**Memory Issues:**
- Ensure 24GB+ VRAM for Aya Expanse 8B
- Use `--quick-test` for faster iteration
- Monitor GPU memory with `nvidia-smi`

**Import Errors:**
```bash
# Reinstall dependencies
pip install -e .
pip install bitsandbytes accelerate
```

## 📄 License

MIT License - see LICENSE file for details. 