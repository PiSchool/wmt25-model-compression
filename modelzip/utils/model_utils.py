#!/usr/bin/env python
"""
Model Utilities for WMT25 Model Compression

Provides utilities for model downloading, caching, verification,
and management across different compression experiments.
"""

import os
import logging as LOG
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constrained_config import WORKDIR

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_model(model_id: str, force_download: bool = False, token: str = None) -> Path:
    """Setup model by downloading and caching locally
    
    Args:
        model_id: HuggingFace model identifier or local path
        force_download: Whether to re-download if already cached
        token: Optional HuggingFace token for gated models
        
    Returns:
        Path: Local path to the cached model
    """
    # Check if it's already a local path
    if Path(model_id).exists():
        LOG.info(f"Model {model_id} is already local")
        return Path(model_id)
    
    # Check cache first
    cache_path = get_model_cache_path(model_id)
    if cache_path.exists() and not force_download:
        LOG.info(f"Model {model_id} already cached at {cache_path}")
        return cache_path
    
    # Download model
    LOG.info(f"Downloading model {model_id}...")
    download_model(model_id, cache_path, token)
    
    # Verify the download
    if verify_model(cache_path):
        LOG.info(f"Model {model_id} successfully downloaded to {cache_path}")
        return cache_path
    else:
        raise RuntimeError(f"Failed to verify downloaded model at {cache_path}")


def download_model(model_id: str, output_path: Path, token: str = None) -> None:
    """Download model and tokenizer to specified path
    
    Args:
        model_id: HuggingFace model identifier
        output_path: Local directory to save the model
        token: Optional HuggingFace token for gated models
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get token from environment if not provided
        if token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        download_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if token:
            download_kwargs["token"] = token
        
        # Download model with authentication if needed
        LOG.info(f"Downloading model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, **download_kwargs)
        model.save_pretrained(output_path)
        
        # Download tokenizer
        LOG.info(f"Downloading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, **download_kwargs)
        tokenizer.save_pretrained(output_path)
        
        LOG.info(f"Successfully downloaded {model_id} to {output_path}")
        
    except Exception as e:
        error_msg = str(e)
        LOG.error(f"Failed to download model {model_id}: {error_msg}")
        
        # Provide specific guidance for gated models
        if "gated" in error_msg.lower() or "access" in error_msg.lower() or "agreement" in error_msg.lower():
            LOG.error(f"Model {model_id} appears to be gated. Please:")
            LOG.error("1. Visit https://huggingface.co/{} and accept the license agreement".format(model_id))
            LOG.error("2. Set your HuggingFace token: export HF_TOKEN=your_token_here")
            LOG.error("3. Or use a publicly available model instead")
        
        # Clean up partial download
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        raise


def verify_model(model_path: Path) -> bool:
    """Verify that a model directory contains required files
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    if not model_path.exists():
        return False
    
    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]
    
    # Check for model weights (various formats)
    model_weight_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model-00001-of-*.bin",
        "model-*.safetensors"
    ]
    
    # Check required files
    for file_name in required_files:
        if not (model_path / file_name).exists():
            LOG.warning(f"Missing required file: {file_name}")
            return False
    
    # Check for at least one model weight file
    has_weights = any(
        list(model_path.glob(pattern)) 
        for pattern in model_weight_files
    )
    
    if not has_weights:
        LOG.warning("No model weight files found")
        return False
    
    # Verify config.json contains model_type for Transformers compatibility
    config_file = model_path / "config.json"
    try:
        import json
        with open(config_file) as f:
            config = json.load(f)
        
        if "model_type" not in config:
            LOG.warning(f"config.json missing model_type field")
            return False
            
        # Check if it's a supported architecture
        model_type = config["model_type"]
        if model_type not in ["llama", "mistral", "cohere", "cohere2", "gemma", "gemma2", "qwen2"]:
            LOG.warning(f"Model type '{model_type}' may not be fully supported")
            
    except Exception as e:
        LOG.warning(f"Could not validate config.json: {e}")
        return False
    
    LOG.info(f"Model at {model_path} verified successfully")
    return True


def get_model_cache_path(model_id: str) -> Path:
    """Get the local cache path for a model
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Path: Local cache directory for the model
    """
    # Sanitize model ID for filesystem
    safe_model_id = model_id.replace("/", "_").replace(":", "_")
    return WORKDIR / "models" / safe_model_id


def is_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        bool: True if model is cached and valid, False otherwise
    """
    cache_path = get_model_cache_path(model_id)
    return cache_path.exists() and verify_model(cache_path)


def list_cached_models() -> list:
    """List all cached models
    
    Returns:
        list: List of cached model identifiers
    """
    models_dir = WORKDIR / "models"
    if not models_dir.exists():
        return []
    
    cached_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and verify_model(model_dir):
            # Convert back from safe filename
            model_id = model_dir.name.replace("_", "/", 1)  # Only first underscore
            cached_models.append(model_id)
    
    return cached_models


def cleanup_model_cache(keep_recent: int = 3) -> None:
    """Clean up model cache, keeping only recent models
    
    Args:
        keep_recent: Number of recent models to keep
    """
    models_dir = WORKDIR / "models"
    if not models_dir.exists():
        return
    
    # Get all model directories with their modification times
    model_dirs = [
        (d, d.stat().st_mtime) 
        for d in models_dir.iterdir() 
        if d.is_dir()
    ]
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old models
    for model_dir, _ in model_dirs[keep_recent:]:
        LOG.info(f"Removing old cached model: {model_dir.name}")
        import shutil
        shutil.rmtree(model_dir)


def get_model_info(model_path: Path) -> dict:
    """Get information about a model
    
    Args:
        model_path: Path to the model
        
    Returns:
        dict: Model information including size, config, etc.
    """
    if not verify_model(model_path):
        return {"error": "Invalid model"}
    
    # Get model size
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    # Try to load config
    config_path = model_path / "config.json"
    config_info = {}
    if config_path.exists():
        import json
        try:
            with open(config_path) as f:
                config = json.load(f)
                config_info = {
                    "model_type": config.get("model_type", "unknown"),
                    "hidden_size": config.get("hidden_size", "unknown"),
                    "num_layers": config.get("num_hidden_layers", "unknown"),
                    "vocab_size": config.get("vocab_size", "unknown")
                }
        except Exception as e:
            LOG.warning(f"Could not parse config: {e}")
    
    return {
        "size_mb": round(size_mb, 2),
        "num_files": len(list(model_path.rglob("*"))),
        "config": config_info,
        "verified": True
    } 