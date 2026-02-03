#!/usr/bin/env python3
"""
Evaluate distribution shift for reward models on HelpSteer dataset.

Goal: Evaluate whether existing reward models generalize to OOD data.

Setup:
- Models: Frozen reward models trained on UltraFeedback
- Dataset: HelpSteer (similar prompts + ranking pairs derived from scores)
- Metrics: Per-prompt Kendall's τ-b between predicted ranking and ground-truth

This script supports two modes:
1. Model-based: --model_name M_pair --seed 42 (finds checkpoint automatically)
2. Path-based:  --model_path reward_model_out/M_pair/seed_42/checkpoint-1800

Usage:
    python evaluation/evaluate_distribution_shift.py --model_name M_pair --seed 42
    python evaluation/evaluate_distribution_shift.py --model_path reward_model_out/M_pair/seed_42/checkpoint-1800
"""

import sys
import io
# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import kendalltau


def compute_kendall_tau_b(ranks, rewards):
    """
    Compute Kendall's τ-b correlation between ground truth ranks and predicted rewards.
    
    Args:
        ranks: List of ground truth ranks [rank_A, rank_B, rank_C, rank_D] (lower is better, 1 = best)
        rewards: List of model rewards (higher is better)
    
    Returns:
        float: Kendall's τ-b, or None if insufficient valid data
    """
    # Filter valid responses (rank > 0)
    valid_pairs = [(ranks[i], rewards[i]) for i in range(len(ranks)) if ranks[i] > 0]
    
    if len(valid_pairs) < 2:
        return None
    
    valid_ranks, valid_rewards = zip(*valid_pairs)
    
    # Negate ranks because lower rank = better (want positive correlation with higher reward)
    tau_b, _ = kendalltau([-r for r in valid_ranks], valid_rewards, variant='b')
    
    if np.isnan(tau_b):
        return None
    
    return tau_b


def evaluate_on_helpsteer(model, tokenizer, helpsteer_dataset, device='cuda', max_length=1024):
    """
    Evaluate reward model on HelpSteer dataset.
    
    Args:
        model: Trained reward model
        tokenizer: Model tokenizer
        helpsteer_dataset: HelpSteer dataset
        device: Device to use
        max_length: Max sequence length
    
    Returns:
        dict: {
            'per_prompt_tau_b': list of Kendall's τ-b values,
            'mean_tau_b': mean τ-b,
            'median_tau_b': median τ-b,
            'std_tau_b': standard deviation,
            'n_prompts': number of prompts evaluated
        }
    """
    model.eval()
    model.to(device)
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    per_prompt_tau_b = []
    
    print(f"\nEvaluating on {len(helpsteer_dataset):,} prompts...")
    
    for i, example in enumerate(helpsteer_dataset):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(helpsteer_dataset)} examples...")
            
        try:
            # Get ground truth ranks
            ranks = [example['rank_A'], example['rank_B'], example['rank_C'], example['rank_D']]
            
            # Get model rewards for all 4 responses
            rewards = []
            for response_key in ['A', 'B', 'C', 'D']:
                prompt = example['prompt']
                response = example[f'response_{response_key}']
                
                # Format: prompt + separator + response (same as training)
                text = (prompt + '\n\n###\n\n' + response).strip()
                
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length,
                    padding='max_length'
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    reward = outputs.logits.squeeze(-1).item()
                
                rewards.append(reward)
            
            # Compute Kendall's τ-b for this prompt
            tau_b = compute_kendall_tau_b(ranks, rewards)
            if tau_b is not None:
                per_prompt_tau_b.append(tau_b)
                
        except Exception as e:
            print(f"  Warning: Failed to process example {i}: {e}")
            continue
            
        # Periodic memory cleanup
        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        'per_prompt_tau_b': per_prompt_tau_b,
        'mean_tau_b': float(np.mean(per_prompt_tau_b)) if per_prompt_tau_b else 0.0,
        'median_tau_b': float(np.median(per_prompt_tau_b)) if per_prompt_tau_b else 0.0,
        'std_tau_b': float(np.std(per_prompt_tau_b)) if per_prompt_tau_b else 0.0,
        'n_prompts': len(per_prompt_tau_b)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate distribution shift on HelpSteer (OOD)')
    
    # Model selection (either by name or by path)
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (e.g., M_pair, M_kw_cox)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Direct path to checkpoint (e.g., reward_model_out/M_pair/seed_42/checkpoint-1800)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Training seed (used with --model_name)')
    parser.add_argument('--models_dir', type=str, default='reward_model_out',
                       help='Base directory containing trained models')
    
    # Dataset and evaluation settings
    parser.add_argument('--helpsteer_data', type=str, default='dataset/HelpSteer',
                       help='Path to prepared HelpSteer dataset (uses test split)')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Max sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to evaluate (for quick testing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_name and not args.model_path:
        print("[ERROR] Either --model_name or --model_path must be provided")
        sys.exit(1)
    
    if args.model_name and args.model_path:
        print("[ERROR] Cannot specify both --model_name and --model_path")
        sys.exit(1)
    
    # Determine checkpoint path
    if args.model_path:
        # Direct path mode
        checkpoint_dir = Path(args.model_path)
        if not checkpoint_dir.exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint_dir}")
            sys.exit(1)
        
        # Extract model name from path for display
        model_name = checkpoint_dir.parent.parent.name if 'seed_' in str(checkpoint_dir.parent) else checkpoint_dir.name
        print("\n" + "="*80)
        print(f"DISTRIBUTION SHIFT EVALUATION (OOD)")
        print("="*80)
        print(f"Checkpoint: {checkpoint_dir}")
        print(f"Device: {args.device}")
    else:
        # Model name mode - find checkpoint automatically
        model_name = args.model_name
        seed_dir = Path(args.models_dir) / args.model_name / f'seed_{args.seed}'
        
        if not seed_dir.exists():
            print(f"[ERROR] Seed directory not found: {seed_dir}")
            sys.exit(1)
        
        # Get the last checkpoint (highest step number = epoch 3.0)
        checkpoints = sorted(seed_dir.glob('checkpoint-*'), key=lambda x: int(x.name.split('-')[-1]))
        
        if not checkpoints:
            print(f"[ERROR] No checkpoints found in: {seed_dir}")
            sys.exit(1)
        
        checkpoint_dir = checkpoints[-1]
        
        print("\n" + "="*80)
        print(f"DISTRIBUTION SHIFT EVALUATION: {args.model_name}")
        print("="*80)
        print(f"Model: {args.model_name}")
        print(f"Seed: {args.seed}")
        print(f"Checkpoint: {checkpoint_dir}")
        print(f"Device: {args.device}")
    
    # Load HelpSteer dataset
    print(f"\nLoading HelpSteer dataset from: {args.helpsteer_data}")
    try:
        helpsteer_full = load_from_disk(args.helpsteer_data)
        # Handle both DatasetDict (with splits) and single Dataset formats
        if hasattr(helpsteer_full, 'keys') and 'test' in helpsteer_full.keys():
            helpsteer_ds = helpsteer_full['test']
            print(f"[OK] Using 'test' split: {len(helpsteer_ds):,} examples")
        else:
            helpsteer_ds = helpsteer_full
            print(f"[OK] Loaded {len(helpsteer_ds):,} examples")
        
        # Optionally limit samples for quick testing
        if args.max_samples and args.max_samples < len(helpsteer_ds):
            helpsteer_ds = helpsteer_ds.select(range(args.max_samples))
            print(f"[OK] Limited to {len(helpsteer_ds):,} examples (--max_samples)")
    except Exception as e:
        print(f"[ERROR] Failed to load HelpSteer dataset: {e}")
        print(f"\n[HINT] Run 'python dataset/prepare_data.py --dataset helpsteer' to prepare the dataset")
        sys.exit(1)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {checkpoint_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
        print(f"[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate on HelpSteer
    print("\n" + "="*80)
    print("EVALUATING ON HELPSTEER (OOD)")
    print("="*80)
    
    results = evaluate_on_helpsteer(
        model=model,
        tokenizer=tokenizer,
        helpsteer_dataset=helpsteer_ds,
        device=args.device,
        max_length=args.max_length
    )
    
    print(f"\n[OK] Evaluation complete")
    print(f"  Prompts evaluated: {results['n_prompts']:,}")
    print(f"  Mean τ-b:   {results['mean_tau_b']:.4f}")
    print(f"  Median τ-b: {results['median_tau_b']:.4f}")
    print(f"  Std τ-b:    {results['std_tau_b']:.4f}")
    
    print("\n" + "="*80)
    print("DISTRIBUTION SHIFT EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
    
    print("\n" + "="*80)
    print("DISTRIBUTION SHIFT EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
