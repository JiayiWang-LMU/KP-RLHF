#!/usr/bin/env python3
"""
Dataset Tokenization Script

This is the central caller for tokenization that:
1. Loads prepared datasets (from prepare_data.py output)
2. Tokenizes all responses with the standard tokenizer
3. Saves the tokenized dataset for reuse across all trainers

This script should be run AFTER prepare_data.py has been executed.

Usage:
    python preprocess_dataset.py --config configs/config_M_pair.yaml --output dataset/UltraFeedback_tokenized
    python preprocess_dataset.py --config configs/config_M_pair.yaml --output dataset/HelpSteer_tokenized --dataset helpsteer

This ensures all trainers use the same tokenized data, saving computation time
and ensuring consistency.
"""

import argparse
import sys
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def tokenize_dataset(dataset_path: Path, tokenizer_name: str, max_length: int, output_path: Path):
    """
    Tokenize a prepared dataset for all trainers.

    Args:
        dataset_path: Path to the prepared dataset (output from prepare_data.py)
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        output_path: Where to save the tokenized dataset
    """
    print(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(str(dataset_path))

    # Check if single dataset or DatasetDict
    from datasets import Dataset
    if isinstance(ds, Dataset):
        # Single dataset (processed HelpSteer eval)
        print("Loading single dataset (processed HelpSteer eval)...")
        train_dataset = ds
        eval_dataset = ds
        available_splits = ['test']  # Force to use 'test' for saving
    else:
        # DatasetDict
        # Check available splits
        available_splits = list(ds.keys())
        print(f"Available splits: {available_splits}")
        
        # Handle different dataset structures
        if 'train' in available_splits and 'validation' in available_splits:
            # HelpSteer structure: has train and validation splits
            print("Loading HelpSteer dataset (train/validation)...")
            train_dataset = ds['train']
            eval_dataset = ds['validation']
        elif 'train' in available_splits and 'test' in available_splits:
            # UltraFeedback structure: has train and test splits (we use test as validation)
            print("Loading UltraFeedback dataset (train/test, using test as validation)...")
            train_dataset = ds['train']
            eval_dataset = ds['test']
        else:
            raise ValueError(f"Expected 'train'+'validation' (HelpSteer) or 'train'+'test' (UltraFeedback) splits. Available: {available_splits}")

    # Verify dataset is in k-wise format (should have been prepared by prepare_data.py)
    if 'response_A' not in train_dataset.column_names:
        raise ValueError(
            "Dataset is not in k-wise format. Please run prepare_data.py first.\n"
            "Expected columns: response_A, response_B, response_C, response_D, rank_A, rank_B, rank_C, rank_D"
        )

    # Use fast tokenizer for better performance
    print(f"Loading tokenizer: {tokenizer_name}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*byte fallback.*")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_responses(examples):
        """Tokenize all responses for a batch of examples."""
        result = {
            'prompt_id': examples['prompt_id'],
            'prompt': examples['prompt'],
            'rank_A': examples['rank_A'],
            'rank_B': examples['rank_B'],
            'rank_C': examples['rank_C'],
            'rank_D': examples['rank_D'],
        }

        # Tokenize each response
        for label in ['A', 'B', 'C', 'D']:
            prompts = examples['prompt']
            responses = examples[f'response_{label}']
            texts = [(p + '\n\n###\n\n' + r).strip() for p, r in zip(prompts, responses)]
            tok = tokenizer(texts, truncation=True, max_length=max_length)
            result[f'input_ids_{label}'] = tok['input_ids']
            result[f'attention_mask_{label}'] = tok['attention_mask']

        return result

    # Tokenize train dataset
    print(f"Tokenizing train dataset (max_length={max_length})...")
    train_dataset = train_dataset.map(
        tokenize_responses,
        batched=True,
        load_from_cache_file=True,
        desc=f"Tokenizing train responses (max_length={max_length})"
    )

    # Tokenize eval dataset
    print(f"Tokenizing eval dataset (max_length={max_length})...")
    eval_dataset = eval_dataset.map(
        tokenize_responses,
        batched=True,
        load_from_cache_file=True,
        desc=f"Tokenizing eval responses (max_length={max_length})"
    )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenized datasets
    print(f"Saving tokenized dataset to {output_path}")
    # Preserve original split naming: 'test' for UltraFeedback, 'validation' for HelpSteer
    if 'test' in available_splits:
        # UltraFeedback: use 'test' split name
        tokenized_ds = {
            'train': train_dataset,
            'test': eval_dataset
        }
    else:
        # HelpSteer: use 'validation' split name
        tokenized_ds = {
            'train': train_dataset,
            'validation': eval_dataset
        }

    # Save the dataset
    from datasets import DatasetDict
    dataset_dict = DatasetDict(tokenized_ds)
    dataset_dict.save_to_disk(str(output_path))

    # Also save the tokenizer
    tokenizer_path = output_path / "tokenizer"
    tokenizer_path.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_path))

    print("Preprocessing completed successfully!")
    print(f"Tokenized dataset saved to: {output_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return output_path, tokenizer_path

def main():
    parser = argparse.ArgumentParser(description="Tokenize prepared datasets for all trainers")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file (used for tokenizer and max_length)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for tokenized dataset')
    parser.add_argument('--dataset', type=str, choices=['ultrafeedback', 'helpsteer'],
                       default='ultrafeedback',
                       help='Dataset to tokenize (default: ultrafeedback)')
    args = parser.parse_args()

    # Add root directory to path so we can import configs
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Load configuration
    from configs.config_loader import load_config
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Determine input dataset based on argument
    output_path = Path(args.output)
    if args.dataset == 'helpsteer':
        dataset_path = Path("dataset/HelpSteer")
    else:
        dataset_path = Path("dataset/UltraFeedback")
    
    tokenizer_name = config.model.model_name  # Use model name as tokenizer
    max_length = config.model.max_length

    print("=" * 80)
    print("DATASET TOKENIZATION")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max length: {max_length}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Run tokenization
    tokenize_dataset(dataset_path, tokenizer_name, max_length, output_path)


if __name__ == '__main__':
    main()