#!/usr/bin/env python3
"""
Dataset preparation pipeline for preference learning.

This is the central caller script that:
1. Downloads datasets using dataset loaders (ultrafeedback.py, helpsteer.py)
2. Adds unique prompt_id for tracking
3. Adds preference rankings with ties preserved (using score_to_rank.py)
4. Splits at prompt level: 90% train, 10% test (seed=42)
5. Saves datasets to disk

After running this script, use preprocess_dataset.py for tokenization.

Usage:
    python dataset/prepare_data.py                      # Prepare all datasets
    python dataset/prepare_data.py --dataset ultrafeedback
    python dataset/prepare_data.py --dataset helpsteer
"""

import argparse
import csv
from datasets import DatasetDict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from score_to_rank import process_dataset
from cache_utils import get_dataset_cache_dir


def add_prompt_ids(dataset):
    """Add unique prompt_id to each example for tracking."""
    def add_id(example, idx):
        example['prompt_id'] = idx
        return example
    
    return dataset.map(add_id, with_indices=True, desc="Adding prompt IDs")


def split_dataset(dataset, train_ratio=0.9, test_ratio=0.1, seed=42):
    """
    Split dataset at prompt level into train/test.
    
    Args:
        dataset: HuggingFace Dataset to split
        train_ratio: Fraction for training (default 0.9)
        test_ratio: Fraction for testing (default 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print("=" * 80)
    print("PROMPT-LEVEL SPLITTING")
    print("=" * 80)
    
    print(f"Splitting (train={train_ratio:.0%}, test={test_ratio:.0%}, seed={seed})...")
    print("Note: All responses from the same prompt stay together in one split.")
    
    train_test = dataset.train_test_split(test_size=test_ratio, seed=seed, shuffle=True)
    
    result = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })
    
    print(f"[OK]Train: {len(result['train']):,} prompts")
    print(f"[OK]Test: {len(result['test']):,} prompts")
    
    return result


def save_dataset(dataset_dict, output_path):
    """Save DatasetDict to disk."""
    output_path = Path(output_path)
    print(f"\nSaving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print("[OK]Saved successfully")
    return output_path


def export_to_csv(dataset_dict, output_dir="dataset/csv", dataset_name="dataset"):
    """Export dataset splits to CSV for inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in dataset_dict.items():
        csv_path = output_dir / f"{dataset_name}_{split_name}.csv"
        print(f"  Exporting {csv_path}...")
        
        # Get column names
        columns = split_data.column_names
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for example in split_data:
                writer.writerow(example)
        
        print(f"    [OK]{len(split_data):,} rows")


def prepare_ultrafeedback(cache_dir=None):
    """Prepare UltraFeedback dataset."""
    from download_ultrafeedback import load_ultrafeedback
    
    print("\n" + "=" * 80)
    print("PREPARING ULTRAFEEDBACK DATASET")
    print("=" * 80 + "\n")
    
    # Step 1: Load dataset
    dataset = load_ultrafeedback(cache_dir=cache_dir)
    
    # Step 2: Add prompt IDs
    print("\nAdding prompt IDs...")
    dataset = add_prompt_ids(dataset)
    
    # Step 3: Add preference rankings (with ties preserved)
    print("\nComputing preference rankings (ties preserved)...")
    dataset = process_dataset(dataset)
    
    # Step 4: Split at prompt level (90/10)
    dataset_dict = split_dataset(dataset, train_ratio=0.9, test_ratio=0.1, seed=42)
    
    # Step 5: Save to disk
    output_path = save_dataset(dataset_dict, "dataset/UltraFeedback")
    
    # Step 6: Export to CSV
    print("\n" + "=" * 80)
    print("EXPORTING TO CSV")
    print("=" * 80)
    export_to_csv(dataset_dict, dataset_name="UltraFeedback")
    
    return dataset_dict, output_path


def prepare_helpsteer(cache_dir=None, target_size=None):
    """
    Prepare HelpSteer dataset.
    
    Args:
        cache_dir: Cache directory for downloads
        target_size: Optional target size for sampling (to match UltraFeedback test size)
    """
    from download_helpsteer import load_helpsteer
    
    print("\n" + "=" * 80)
    print("PREPARING HELPSTEER DATASET")
    print("=" * 80 + "\n")
    
    # Step 1: Load dataset
    dataset = load_helpsteer(cache_dir=cache_dir)
    
    # Optional: Sample to target size
    if target_size and len(dataset) > target_size:
        print(f"\nSampling to target size: {target_size:,}")
        dataset = dataset.shuffle(seed=42).select(range(target_size))
        print(f"[OK]Sampled to {len(dataset):,} examples")
    
    # Step 2: Add prompt IDs
    print("\nAdding prompt IDs...")
    dataset = add_prompt_ids(dataset)
    
    # Step 3: Add preference rankings (with ties preserved)
    print("\nComputing preference rankings (ties preserved)...")
    dataset = process_dataset(dataset)
    
    # Step 4: Split at prompt level (90/10)
    dataset_dict = split_dataset(dataset, train_ratio=0.9, test_ratio=0.1, seed=42)
    
    # Step 5: Save to disk
    output_path = save_dataset(dataset_dict, "dataset/HelpSteer")
    
    # Step 6: Export to CSV
    print("\n" + "=" * 80)
    print("EXPORTING TO CSV")
    print("=" * 80)
    export_to_csv(dataset_dict, dataset_name="HelpSteer")
    
    return dataset_dict, output_path


def print_summary(dataset_dict, dataset_name):
    """Print summary of prepared dataset."""
    print("\n" + "=" * 80)
    print(f"SUMMARY: {dataset_name}")
    print("=" * 80)
    
    total = sum(len(split) for split in dataset_dict.values())
    print(f"[OK]Total prompts:        {total:,}")
    print(f"[OK]Train split:          {len(dataset_dict['train']):,} prompts (90%)")
    print(f"[OK]Test split:           {len(dataset_dict['test']):,} prompts (10%)")
    print(f"[OK]Text format:          Single-line (no newlines)")
    print(f"[OK]Split seed:           42")
    print(f"[OK]Split level:          Prompt-level (all 4 responses per prompt stay together)")
    print(f"[OK]Ties:                 Preserved (decomposition happens during batching)")
    print(f"\n[OK]Columns:")
    print(f"    {dataset_dict['train'].column_names}")
    
    # Show sample
    print("\nSAMPLE (First example):")
    print("-" * 80)
    k = dataset_dict['train'][0]
    print(f"Prompt ID: {k.get('prompt_id', 'N/A')}")
    print(f"Prompt: {k['prompt'][:100]}...")
    print(f"\nK-wise Rankings: [{k['rank_A']}, {k['rank_B']}, {k['rank_C']}, {k['rank_D']}]")
    scores = {l: k[f'score_{l}'] for l in ['A', 'B', 'C', 'D'] if k.get(f'score_{l}', 0) > 0}
    print(f"K-wise Scores: {scores}")
    print(f"\nResponse A ({k['model_A']}): {k['response_A'][:80]}...")
    print(f"Response B ({k['model_B']}): {k['response_B'][:80]}...")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for preference learning"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['ultrafeedback', 'helpsteer', 'all'],
        default='all',
        help='Dataset to prepare (default: all)'
    )
    parser.add_argument(
        '--helpsteer-target-size',
        type=int,
        default=None,
        help='Target size for HelpSteer sampling (to match UltraFeedback test size)'
    )
    args = parser.parse_args()
    
    cache_dir = get_dataset_cache_dir()
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print("=" * 80 + "\n")
    
    if args.dataset in ['ultrafeedback', 'all']:
        uf_dict, _ = prepare_ultrafeedback(cache_dir=cache_dir)
        print_summary(uf_dict, "UltraFeedback")
        
        # Get test size for HelpSteer sampling if not specified
        if args.helpsteer_target_size is None:
            args.helpsteer_target_size = len(uf_dict['test'])
    
    if args.dataset in ['helpsteer', 'all']:
        hs_dict, _ = prepare_helpsteer(
            cache_dir=cache_dir, 
            target_size=args.helpsteer_target_size
        )
        print_summary(hs_dict, "HelpSteer")
    
    print("\n" + "=" * 80)
    print("NEXT STEP: Run tokenization")
    print("=" * 80)
    print("Run preprocess_dataset.py to tokenize the prepared datasets:")
    print("  python preprocess_dataset.py --config configs/config_M_pair.yaml --output dataset/UltraFeedback_tokenized")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
