#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UltraFeedback dataset loader.

This module handles loading and preprocessing of the OpenBMB UltraFeedback dataset.
Similar structure to helpsteer.py for consistency.

Features:
- Loads openbmb/UltraFeedback dataset (60k prompts with 4 responses each)
- Normalizes text to single-line format
- Extracts response scores for ranking
- Returns processed dataset ready for splitting

Usage:
    from dataset.ultrafeedback import load_ultrafeedback
    dataset = load_ultrafeedback()
"""

import re
import json
from datasets import load_dataset
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


def load_ultrafeedback(cache_dir=None):
    """
    Load and preprocess OpenBMB UltraFeedback dataset.
    
    Args:
        cache_dir: Optional cache directory. If None, uses workspace cache.
    
    Returns:
        Processed HuggingFace Dataset with columns:
        - prompt: Normalized prompt text
        - response_A, response_B, response_C, response_D: Response texts
        - model_A, model_B, model_C, model_D: Model names
        - score_A, score_B, score_C, score_D: Fine-grained scores
    """
    print("=" * 80)
    print("LOADING OPENBMB ULTRAFEEDBACK DATASET")
    print("=" * 80)
    
    if cache_dir is None:
        cache_dir = get_dataset_cache_dir()
    
    print(f"Cache directory: {cache_dir}")
    ds = load_dataset("openbmb/UltraFeedback", split="train", cache_dir=cache_dir)
    print(f"[OK]Loaded {len(ds):,} examples")
    
    def process_example(example):
        """Process a single UltraFeedback example."""
        prompt = normalize_text(example.get('instruction', example.get('prompt', '')))
        
        result = {
            'prompt': prompt,
            'response_A': '', 'response_B': '', 'response_C': '', 'response_D': '',
            'model_A': '', 'model_B': '', 'model_C': '', 'model_D': '',
            'score_A': 0.0, 'score_B': 0.0, 'score_C': 0.0, 'score_D': 0.0,
        }
        
        completions = example.get('completions', [])
        if isinstance(completions, str):
            try:
                completions = json.loads(completions)
            except:
                completions = []
        
        if isinstance(completions, list):
            letters = ['A', 'B', 'C', 'D']
            for idx_comp, comp in enumerate(completions[:4]):
                if isinstance(comp, dict):
                    response = normalize_text(comp.get('response', ''))
                    score = comp.get('fine-grained_score', 0)
                    model = comp.get('model', 'unknown')
                    
                    try:
                        score = float(score) if score is not None else 0.0
                    except:
                        score = 0.0
                    
                    letter = letters[idx_comp]
                    result[f'response_{letter}'] = response
                    result[f'model_{letter}'] = str(model)
                    result[f'score_{letter}'] = score
        
        return result
    
    processed = ds.map(process_example, remove_columns=ds.column_names, desc="Processing UltraFeedback")
    print(f"[OK]Processed with columns: {processed.column_names}")
    
    return processed


def get_dataset_info():
    """Return metadata about the UltraFeedback dataset."""
    return {
        'name': 'UltraFeedback',
        'source': 'openbmb/UltraFeedback',
        'description': 'OpenBMB UltraFeedback dataset with 4 responses per prompt',
        'num_responses': 4,
        'score_type': 'fine-grained_score',
        'output_dir': 'dataset/UltraFeedback',
    }


if __name__ == '__main__':
    # Test loading
    print("\nTesting UltraFeedback loader...")
    ds = load_ultrafeedback()
    
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
