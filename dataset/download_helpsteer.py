#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HelpSteer dataset loader.

This module handles loading and preprocessing of the NVIDIA HelpSteer dataset.
Similar structure to ultrafeedback.py for consistency.

Features:
- Loads nvidia/HelpSteer dataset (~35k training examples)
- Groups responses by prompt (typically 4 responses per prompt)
- Computes total score (sum of all dimensions) for ranking
- Returns processed dataset ready for splitting

Usage:
    from dataset.helpsteer import load_helpsteer
    dataset = load_helpsteer()
"""

import re
from datasets import load_dataset, Dataset
from collections import defaultdict
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from cache_utils import get_dataset_cache_dir


def normalize_text(text):
    """Remove newlines and normalize spaces."""
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_total_score(example):
    """Compute total score by summing all dimensions."""
    dimensions = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    total = 0.0
    for dim in dimensions:
        try:
            total += float(example.get(dim, 0))
        except (ValueError, TypeError):
            pass
    return total


def group_by_prompt(dataset):
    """Group HelpSteer examples by prompt (each prompt has ~4 responses)."""
    print("  Grouping responses by prompt...")
    prompt_groups = defaultdict(list)
    
    for example in dataset:
        prompt = example['prompt']
        prompt_groups[prompt].append(example)
    
    print(f"  Found {len(prompt_groups):,} unique prompts")
    
    # Count distribution of responses per prompt
    response_counts = defaultdict(int)
    for prompt, responses in prompt_groups.items():
        response_counts[len(responses)] += 1
    
    print(f"  Response distribution:")
    for count in sorted(response_counts.keys()):
        print(f"    {count} responses: {response_counts[count]:,} prompts")
    
    return prompt_groups


def load_helpsteer(cache_dir=None, require_4_responses=True):
    """
    Load and preprocess NVIDIA HelpSteer dataset.
    
    Args:
        cache_dir: Optional cache directory. If None, uses workspace cache.
        require_4_responses: If True, only include prompts with exactly 4 responses
    
    Returns:
        Processed HuggingFace Dataset with columns:
        - prompt: Normalized prompt text
        - response_A, response_B, response_C, response_D: Response texts
        - model_A, model_B, model_C, model_D: Model names
        - score_A, score_B, score_C, score_D: Total scores (sum of dimensions)
    """
    print("=" * 80)
    print("LOADING NVIDIA/HELPSTEER DATASET")
    print("=" * 80)
    
    if cache_dir is None:
        cache_dir = get_dataset_cache_dir()
    
    print(f"Cache directory: {cache_dir}")
    
    # Try loading HelpSteer
    try:
        ds = load_dataset("nvidia/HelpSteer", cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading nvidia/HelpSteer: {e}")
        print("Trying nvidia/HelpSteer2...")
        ds = load_dataset("nvidia/HelpSteer2", cache_dir=cache_dir)
    
    print(f"[OK]Loaded dataset with splits: {list(ds.keys())}")
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data):,} examples")
    
    # Group by prompt
    train_groups = group_by_prompt(ds['train'])
    
    # Filter to prompts with required number of responses
    if require_4_responses:
        valid_groups = {p: resps for p, resps in train_groups.items() if len(resps) == 4}
        print(f"  Prompts with 4 responses: {len(valid_groups):,}")
    else:
        valid_groups = train_groups
    
    # Convert to k-wise format
    kwise_examples = []
    letters = ['A', 'B', 'C', 'D']
    
    for prompt, responses in valid_groups.items():
        prompt_text = normalize_text(prompt)
        
        example = {
            'prompt': prompt_text,
        }
        
        for i, (letter, resp) in enumerate(zip(letters, responses[:4])):
            response_text = normalize_text(resp['response'])
            total_score = compute_total_score(resp)
            
            example[f'response_{letter}'] = response_text
            example[f'model_{letter}'] = f'helpsteer_{i}'
            example[f'score_{letter}'] = total_score
        
        kwise_examples.append(example)
    
    processed = Dataset.from_list(kwise_examples)
    print(f"[OK]Processed with columns: {processed.column_names}")
    print(f"[OK]Created {len(processed):,} examples")
    
    return processed


def get_dataset_info():
    """Return metadata about the HelpSteer dataset."""
    return {
        'name': 'HelpSteer',
        'source': 'nvidia/HelpSteer',
        'description': 'NVIDIA HelpSteer dataset with 5-dimension scores',
        'num_responses': 4,
        'score_type': 'sum(helpfulness, correctness, coherence, complexity, verbosity)',
        'output_dir': 'dataset/HelpSteer',
    }


if __name__ == '__main__':
    # Test loading
    print("\nTesting HelpSteer loader...")
    ds = load_helpsteer()
    
    print(f"\n[OK]Dataset loaded: {len(ds):,} examples")
    print(f"[OK]Columns: {ds.column_names}")
    
    # Show sample
    sample = ds[0]
    print(f"\nSample:")
    print(f"  Prompt: {sample['prompt'][:100]}...")
    print(f"  Score A: {sample['score_A']}, Model: {sample['model_A']}")
    print(f"  Score B: {sample['score_B']}, Model: {sample['model_B']}")
    print(f"  Score C: {sample['score_C']}, Model: {sample['model_C']}")
    print(f"  Score D: {sample['score_D']}, Model: {sample['model_D']}")
