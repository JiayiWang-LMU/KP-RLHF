#!/usr/bin/env python3
"""
SFT/RLHF Dataset Loader.

This module handles loading and preprocessing of datasets used for SFT and RLHF 
(PPO/GRPO) training. Currently supports Alpaca-style instruction datasets.

Features:
- Loads tatsu-lab/alpaca dataset (52k instruction-response pairs)
- Formats prompts for policy optimization (PPO/GRPO)
- Optional tokenization for trainers that require pre-tokenized data
- Uses workspace cache for efficient re-loading

Usage:
    from dataset.download_sft import load_sft_dataset, load_sft_dataset_tokenized
    
    # For GRPO (needs 'prompt' column)
    dataset = load_sft_dataset(num_samples=10000)
    
    # For PPO (needs 'input_ids' column)
    dataset = load_sft_dataset_tokenized(tokenizer, num_samples=10000, max_length=256)
"""

import sys
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset

# Add dataset directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'dataset'))
from cache_utils import get_dataset_cache_dir


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Format an Alpaca-style instruction into a prompt.
    
    Args:
        instruction: The instruction/task description
        input_text: Optional additional input context
    
    Returns:
        Formatted prompt string
    """
    if input_text:
        return f"{instruction}\n\nInput: {input_text}\n\nResponse:"
    else:
        return f"{instruction}\n\nResponse:"


def load_sft_dataset(
    dataset_name: str = "tatsu-lab/alpaca",
    split: str = "train",
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load and format an SFT dataset for RLHF training.
    
    This function loads the dataset and formats it with a 'prompt' column
    suitable for policy optimization trainers (GRPO, etc.).
    
    Args:
        dataset_name: HuggingFace dataset name (default: tatsu-lab/alpaca)
        split: Dataset split to load (default: train)
        num_samples: Number of samples to use (None = all)
        cache_dir: Cache directory (None = use workspace cache)
    
    Returns:
        Dataset with 'prompt' column
    """
    if cache_dir is None:
        cache_dir = get_dataset_cache_dir()
    
    print(f"Loading SFT dataset: {dataset_name} (cache: {cache_dir})")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    print(f"[OK]Loaded {len(dataset):,} examples")
    
    # Filter to specified number of samples
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"[OK]Selected {len(dataset):,} samples")
    
    # Format prompts
    def format_prompts(examples):
        prompts = []
        instructions = examples['instruction']
        inputs = examples.get('input', [''] * len(instructions))
        
        for instruction, input_text in zip(instructions, inputs):
            prompt = format_alpaca_prompt(instruction, input_text)
            prompts.append(prompt)
        
        return {'prompt': prompts}
    
    dataset = dataset.map(
        format_prompts, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Formatting prompts"
    )
    
    print(f"[OK]Dataset formatted with 'prompt' column")
    return dataset


def load_sft_dataset_tokenized(
    tokenizer,
    dataset_name: str = "tatsu-lab/alpaca",
    split: str = "train",
    num_samples: Optional[int] = None,
    max_length: int = 256,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load, format, and tokenize an SFT dataset for RLHF training.
    
    This function loads the dataset, formats prompts, and tokenizes them.
    Required for trainers that expect pre-tokenized data (PPO).
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: HuggingFace dataset name (default: tatsu-lab/alpaca)
        split: Dataset split to load (default: train)
        num_samples: Number of samples to use (None = all)
        max_length: Maximum sequence length for tokenization
        cache_dir: Cache directory (None = use workspace cache)
    
    Returns:
        Dataset with 'prompt', 'input_ids', 'attention_mask' columns
    """
    # First load and format
    dataset = load_sft_dataset(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        cache_dir=cache_dir,
    )
    
    # Tokenize prompts
    def tokenize_prompts(examples):
        return tokenizer(
            examples['prompt'],
            truncation=True,
            max_length=max_length,
            padding=False,  # Let DataCollator handle padding
            return_tensors=None,
        )
    
    dataset = dataset.map(
        tokenize_prompts,
        batched=True,
        desc=f"Tokenizing prompts (max_length={max_length})"
    )
    
    print(f"[OK]Dataset tokenized with 'input_ids' column")
    return dataset


def get_dataset_info():
    """Return metadata about supported SFT datasets."""
    return {
        'alpaca': {
            'name': 'Alpaca',
            'source': 'tatsu-lab/alpaca',
            'description': 'Stanford Alpaca instruction-following dataset (52k examples)',
            'columns': ['instruction', 'input', 'output'],
        },
    }


if __name__ == '__main__':
    # Test loading
    print("\nTesting SFT dataset loader...")
    
    # Load without tokenization (for GRPO)
    dataset = load_sft_dataset(num_samples=100)
    print(f"\n[OK]Dataset loaded: {len(dataset)} examples")
    print(f"[OK]Columns: {dataset.column_names}")
    
    # Show sample
    sample = dataset[0]
    print(f"\nSample prompt:")
    print(f"  {sample['prompt'][:200]}...")
