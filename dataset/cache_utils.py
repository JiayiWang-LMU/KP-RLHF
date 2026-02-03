"""
Cache management utilities for datasets and models.
Stores cached data in the workspace to avoid re-downloading.
"""
from pathlib import Path
from typing import Optional

# Workspace root (go up one level from dataset/ folder)
WORKSPACE_ROOT = Path(__file__).parent.parent
CACHE_DIR = WORKSPACE_ROOT / "dataset" / "cache"
MODELS_CACHE_DIR = WORKSPACE_ROOT / "models_cache"

# Create cache directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_cache_dir() -> str:
    """Get the cache directory for datasets."""
    return str(CACHE_DIR)


def get_model_cache_dir() -> str:
    """Get the cache directory for models."""
    return str(MODELS_CACHE_DIR)


def load_model_cached(
    model_name: str,
    cache_dir: Optional[str] = None,
    **model_kwargs
):
    """
    Load a HuggingFace model and tokenizer with local caching.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
        cache_dir: Optional cache directory. If None, uses workspace cache.
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if cache_dir is None:
        cache_dir = get_model_cache_dir()
    
    print(f"Loading model: {model_name} (cache: {cache_dir})...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        **model_kwargs
    )
    
    print(f"✓ Model and tokenizer loaded")
    return model, tokenizer


def load_dataset_cached(dataset_name: str, split: Optional[str] = None, cache_dir: Optional[str] = None):
    """
    Load a HuggingFace dataset with local caching.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "openbmb/UltraFeedback")
        split: Dataset split to load (e.g., "train", "test"). If None, loads all splits.
        cache_dir: Optional cache directory. If None, uses workspace cache.
        
    Returns:
        Dataset or DatasetDict object
    """
    from datasets import load_dataset
    
    if cache_dir is None:
        cache_dir = get_dataset_cache_dir()
    
    print(f"Loading {dataset_name} (cache: {cache_dir})...")
    
    if split:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    print(f"✓ Dataset loaded")
    return dataset


def get_cache_info():
    """Print information about cached data."""
    print("="*80)
    print("CACHE INFORMATION")
    print("="*80)
    
    # Dataset cache
    print(f"\nDataset cache: {CACHE_DIR}")
    if CACHE_DIR.exists():
        dataset_size = sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file())
        dataset_files = len(list(CACHE_DIR.rglob('*')))
        print(f"  Size: {dataset_size / 1e9:.2f} GB")
        print(f"  Files: {dataset_files}")
    else:
        print("  Empty")
    
    # Model cache
    print(f"\nModel cache: {MODELS_CACHE_DIR}")
    if MODELS_CACHE_DIR.exists():
        model_size = sum(f.stat().st_size for f in MODELS_CACHE_DIR.rglob('*') if f.is_file())
        model_files = len(list(MODELS_CACHE_DIR.rglob('*')))
        print(f"  Size: {model_size / 1e9:.2f} GB")
        print(f"  Files: {model_files}")
    else:
        print("  Empty")
    
    print("="*80)


if __name__ == "__main__":
    # Display cache information
    get_cache_info()
    
    print("\nTo use caching in your scripts:")
    print("  from dataset.cache_utils import get_dataset_cache_dir, load_dataset_cached, load_model_cached")
    print("  cache_dir = get_dataset_cache_dir()")
    print("  dataset = load_dataset_cached('openbmb/UltraFeedback', split='train')")
    print("  model, tokenizer = load_model_cached('meta-llama/Llama-2-7b-hf')")
    print("\nFor SFT/RLHF datasets (Alpaca), use:")
    print("  from rlhf.download_sft import load_sft_dataset, load_sft_dataset_tokenized")

