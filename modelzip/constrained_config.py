#!/usr/bin/env python
"""
Configuration for WMT25 Model Compression Constrained Track

Focus on Aya Expanse 8B model compression for the three constrained language pairs:
- Czech to German (ces-deu)
- Japanese to Chinese (jpn-zho) 
- English to Arabic (eng-ara)
"""

import logging as LOG
import os
from pathlib import Path
from typing import Dict, List, Optional

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directories
HF_CACHE = Path(os.getenv("HF_HOME", default=Path.home() / ".cache" / "huggingface")) / "hub"
WORKDIR = Path("./workdir")  # Renamed for consistency
WORK_DIR = WORKDIR  # Backward compatibility
DATA_DIR = WORKDIR / "data"
MODELS_DIR = WORKDIR / "models"
EXPERIMENTS_DIR = WORKDIR / "experiments"
RESULTS_DIR = WORKDIR / "results"

# Constrained Task Configuration
CONSTRAINED_TASK = {
    # Base model for constrained track
    "base_model": "CohereLabs/aya-expanse-8b",
    
    # Constrained language pairs only
    "language_pairs": ["ces-deu", "jpn-zho", "eng-ara"],
    
    # Data sources for each language pair
    "data_sources": {
        "ces-deu": {
            "test": {"wmt19": "sacrebleu -t wmt19 -l cs-de --echo src ref"},
            "train": "Statmt-news_commentary-18-ces-deu",
        },
        "jpn-zho": {
            "test": {"wmt24": "sacrebleu -t wmt24 -l ja-zh --echo src ref:refA"},
            "train": "Statmt-news_commentary-18-jpn-zho", 
        },
        "eng-ara": {
            "test": {"wmt24pp": "mtdata echo Google-wmt24pp-1-eng-ara_SA | sed 's/\\r//g'"},
            "train": "Statmt-news_commentary-18-ara-eng",
        },
    },
    
    # Evaluation metrics
    "metrics": {
        "quality": ["chrf", "comet"],
        "efficiency": ["model_size", "inference_speed", "memory_usage"],
    },
    
    # COMET configuration
    "comet_model": "wmt22-comet-da",  # Default COMET model for quality assessment
    
    # Compression techniques to experiment with
    "compression_methods": [
        "baseline",
        "quantization_8bit",
        "quantization_4bit", 
        "pruning_structured",
        "pruning_unstructured",
        "distillation",
        "mixed_precision",
    ],
    
    # Translation prompts for each language pair
    "prompts": {
        "ces-deu": {
            "template": "Translate the following Czech text to German:\n{text}\n",
            "src_lang": "Czech",
            "tgt_lang": "German"
        },
        "jpn-zho": {
            "template": "Translate the following Japanese text to Chinese:\n{text}\n",
            "src_lang": "Japanese", 
            "tgt_lang": "Chinese"
        },
        "eng-ara": {
            "template": "Translate the following English text to Arabic:\n{text}\n",
            "src_lang": "English",
            "tgt_lang": "Arabic"
        }
    },
}

# Language mappings
LANGS_MAP = {
    "ces": "Czech", "cs": "Czech",
    "deu": "German", "de": "German", 
    "jpn": "Japanese", "ja": "Japanese",
    "zho": "Chinese", "zh": "Chinese",
    "eng": "English", "en": "English",
    "ara": "Arabic", "ar": "Arabic",
}

# Model configuration
MODEL_CONFIG = {
    "max_length": 4096,
    "batch_size": 16,
    "torch_dtype": "auto",
    "device_map": "auto",
}

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "save_steps": 1000,
    "eval_steps": 500,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
}

# Evaluation configuration  
EVAL_CONFIG = {
    "beam_size": 1,
    "temperature": 0.0,
    "max_new_tokens": 512,
    "do_sample": False,
}

# Translation prompts
TRANSLATION_PROMPTS = {
    "default": "Translate the following text from {src} to {tgt}.\n{text}\n",
    "few_shot": "Translate the following text from {src} to {tgt}.\n\nExamples:\n{examples}\n\nText to translate:\n{text}\n",
    "instruction": "You are a professional translator. Translate the following {src} text to {tgt}. Provide only the translation without any additional text.\n\n{text}",
}

# Compression configuration
COMPRESSION_CONFIG = {
    "quantization": {
        "8bit": {"load_in_8bit": True},
        "4bit": {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"},
    },
    "pruning": {
        "structured": {"sparsity": 0.3, "type": "structured"},
        "unstructured": {"sparsity": 0.3, "type": "unstructured"},
    },
    "distillation": {
        "temperature": 4.0,
        "alpha": 0.7,
        "student_layers": None,  # Auto-determined (half of teacher)
        "reduction_factor": 0.75,
    },
    "mixed_precision": {
        "fp16": {
            "dtype": "fp16",
            "optimize_for_inference": True,
            "use_torch_compile": False
        },
        "bf16": {
            "dtype": "bf16", 
            "optimize_for_inference": True,
            "use_torch_compile": False
        }
    },
}

def get_lang_name(lang_code: str) -> str:
    """Get full language name from code"""
    return LANGS_MAP.get(lang_code, lang_code)

def get_constrained_pairs() -> List[str]:
    """Get constrained task language pairs"""
    return CONSTRAINED_TASK["language_pairs"].copy()

def setup_directories():
    """Create all necessary directories"""
    for dir_path in [WORKDIR, DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Created directory: {dir_path}") 