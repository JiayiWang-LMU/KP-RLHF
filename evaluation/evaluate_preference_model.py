#!/usr/bin/env python3
"""
Evaluate a reward model on a saved checkpoint.

Goal: Evaluate reward model performance on held-out test data with bootstrap CIs.

Setup:
- Models: Trained reward models (final checkpoint at epoch 3.0)
- Dataset: UltraFeedback test split (same distribution as training)
- Metrics: Per-prompt Kendall's τ-b with bootstrap confidence intervals

This script supports two modes:
1. Model-based: --model_name M_pair (finds checkpoint automatically)
2. Path-based:  --model_path reward_model_out/M_pair/seed_42/checkpoint-1800

Usage:
    python evaluation/evaluate_preference_model.py --model_name M_pair --n_bootstrap 1000
    python evaluation/evaluate_preference_model.py --model_path reward_model_out/M_pair/seed_42/checkpoint-1800
"""

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import kendalltau
from tqdm import tqdm

# Fixed seed for dataset splitting to ensure consistency with training
DATA_SEED = 42

def compute_kendall_tau_b(ranks, rewards):
    """
    Compute Kendall's τ-b correlation between ground truth ranks and predicted rewards.
    
    Kendall's τ-b naturally handles ties in both rankings.
    Returns τ-b in [-1, 1] where:
      1.0 = perfect agreement
      0.0 = no correlation
     -1.0 = perfect disagreement
    
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


def evaluate_checkpoint(model, tokenizer, test_dataset, device='cuda', max_length=1024):
    """
    Evaluate a single checkpoint on the test set.
    
    Returns:
        dict: {
            'per_prompt_tau_b': list of Kendall's τ-b values (one per prompt),
            'mean_tau_b': mean Kendall's τ-b across all prompts,
            'n_prompts': number of prompts evaluated
        }
    """
    model.eval()
    model.to(device)
    
    per_prompt_tau_b = []
    
    for example in tqdm(test_dataset, desc="Evaluating", leave=False):
        # Get ground truth ranks
        ranks = [example['rank_A'], example['rank_B'], example['rank_C'], example['rank_D']]
        
        # Get model rewards for all 4 responses
        rewards = []
        for response_key in ['A', 'B', 'C', 'D']:
            prompt = example['prompt']
            response = example[f'response_{response_key}']
            
            # Format: prompt + separator + response
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
    
    return {
        'per_prompt_tau_b': per_prompt_tau_b,
        'mean_tau_b': np.mean(per_prompt_tau_b) if per_prompt_tau_b else 0.0,
        'n_prompts': len(per_prompt_tau_b)
    }


def bootstrap_confidence_interval(per_prompt_tau_b, n_bootstrap=1000, confidence=0.95, seed=42):
    """
    Compute bootstrap confidence interval for mean Kendall's τ-b.
    
    Args:
        per_prompt_tau_b: List of Kendall's τ-b values (one per prompt)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        dict: {
            'mean': mean τ-b,
            'lower': lower bound of CI,
            'upper': upper bound of CI,
            'std': standard error
        }
    """
    rng = np.random.RandomState(seed)
    n_prompts = len(per_prompt_tau_b)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample prompts with replacement
        bootstrap_sample = rng.choice(per_prompt_tau_b, size=n_prompts, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'mean': np.mean(per_prompt_tau_b),
        'lower': np.percentile(bootstrap_means, lower_percentile),
        'upper': np.percentile(bootstrap_means, upper_percentile),
        'std': np.std(bootstrap_means)
    }


def find_checkpoints(model_dir):
    """
    Find the checkpoint for a preference model.
    
    Expected structure:
        reward_model_out/M_pair/
            seed_42/
                checkpoint-*/ (only the final checkpoint at epoch 3.0 remains)
    
    Returns:
        List of tuples: [(seed, epoch, checkpoint_path), ...]
        Note: epoch is set to 'Final' for the final checkpoint.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    
    checkpoints = []
    for seed_dir in sorted(model_dir.glob('seed_*')):
        if not seed_dir.is_dir():
            continue
        
        seed = seed_dir.name.split('_')[1]
        
        # Find all checkpoint directories and get the final one (highest step number)
        checkpoint_dirs = sorted([d for d in seed_dir.glob('checkpoint-*') if d.is_dir()], 
                                key=lambda x: int(x.name.split('-')[-1]))
        
        if checkpoint_dirs:
            final_checkpoint = checkpoint_dirs[-1]  # Last checkpoint = epoch 3.0
            step = final_checkpoint.name.split('-')[-1]
            checkpoints.append((seed, f'Final-Step-{step}', str(final_checkpoint)))
        
        # Legacy: Check if seed directory itself contains the model
        has_model = (seed_dir / 'model.safetensors').exists() or \
                   (seed_dir / 'pytorch_model.bin').exists()
        has_config = (seed_dir / 'config.json').exists()
        
        if has_model and has_config:
            checkpoints.append((seed, 'Legacy', str(seed_dir)))
    
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser(description='Evaluate preference model with bootstrapping')
    parser.add_argument('--model_name', type=str, required=False, 
                       help='Name of preference model (e.g., M_pair, M_kw_scratch)')
    parser.add_argument('--model_path', type=str, required=False,
                       help='Direct path to a specific checkpoint to evaluate')
    parser.add_argument('--models_dir', type=str, default='reward_model_out',
                       help='Base directory containing trained models')
    parser.add_argument('--test_data', type=str, default='dataset/UltraFeedback',
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for bootstrap CI')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Max sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for bootstrapping')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to evaluate (for quick testing)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_name and not args.model_path:
        print("❌ Error: Either --model_name or --model_path must be provided")
        sys.exit(1)
    
    if args.model_name and args.model_path:
        print("❌ Error: Cannot specify both --model_name and --model_path")
        sys.exit(1)
    
    if args.model_path:
        # Single checkpoint mode
        checkpoint_path = Path(args.model_path)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        
        # Extract model name from path for output directory
        # Expected path: reward_model_out/MODEL_NAME/seed_X/checkpoint-Y
        path_parts = checkpoint_path.parts
        if len(path_parts) >= 3 and 'reward_model_out' in path_parts:
            model_name_idx = path_parts.index('reward_model_out') + 1
            if model_name_idx < len(path_parts):
                model_name = path_parts[model_name_idx]
            else:
                model_name = "unknown_model"
        else:
            model_name = "unknown_model"
            
        # Extract seed and epoch from path if possible
        seed = "unknown"
        epoch = "unknown"
        if "seed_" in str(checkpoint_path):
            try:
                seed_part = [p for p in path_parts if p.startswith('seed_')][0]
                seed = seed_part.split('_')[1]
            except (IndexError, ValueError):
                pass
                
        if "checkpoint-" in str(checkpoint_path):
            try:
                checkpoint_part = [p for p in path_parts if p.startswith('checkpoint-')][0]
                epoch = checkpoint_part.split('-')[1]
            except (IndexError, ValueError):
                pass
        
        checkpoints = [(seed, epoch, str(checkpoint_path))]
        
        print("\n" + "="*80)
        print(f"EVALUATING SINGLE CHECKPOINT: {checkpoint_path.name}")
        print("="*80)
        print(f"✓ Model: {model_name}")
        print(f"✓ Checkpoint: {checkpoint_path}")
        
    else:
        # Multiple checkpoints mode (existing behavior)
        model_name = args.model_name
        print("\n" + "="*80)
        print(f"EVALUATING PREFERENCE MODEL: {args.model_name}")
        print("="*80)
        
        # Find all checkpoints
        model_dir = Path(args.models_dir) / args.model_name
        checkpoints = find_checkpoints(model_dir)
        
        if not checkpoints:
            print(f"❌ No checkpoints found in {model_dir}")
            print(f"Expected structure: {model_dir}/seed_*/ (containing model files)")
            sys.exit(1)
        
        print(f"✓ Found checkpoints:")
        for seed, epoch, path in checkpoints:
            print(f"  - Seed {seed}, Epoch {epoch}: {path}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.test_data}...")
    dataset_path = Path(args.test_data)
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        sys.exit(1)
        
    ds = load_from_disk(str(dataset_path))
    
    if isinstance(ds, dict) and 'test' in ds:
        test_dataset = ds['test']
    else:
        print("❌ Dataset does not have a 'test' split. Please run prepare_data.py first.")
        sys.exit(1)
    
    # Optionally limit samples for quick testing
    if args.max_samples and args.max_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_samples))
        print(f"✓ Limited to {len(test_dataset)} test prompts (--max_samples)")
    else:
        print(f"✓ Loaded {len(test_dataset)} test prompts")
    
    # Evaluate each checkpoint
    results = []
    
    print(f"\nEvaluating checkpoint (using Kendall's τ-b)...")
    for seed, epoch, checkpoint_path in checkpoints:
        print(f"\n{'='*80}")
        print(f"Checkpoint: Seed {seed}, Epoch {epoch}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*80}")
        
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        print("✓ Model loaded")
        
        # Evaluate
        print("Computing Kendall's τ-b for each prompt...")
        eval_result = evaluate_checkpoint(
            model, tokenizer, test_dataset, 
            device=args.device, max_length=args.max_length
        )
        
        # Bootstrap confidence interval
        print("Computing bootstrap confidence interval...")
        ci_result = bootstrap_confidence_interval(
            eval_result['per_prompt_tau_b'],
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed
        )
        
        # Store results
        result = {
            'seed': seed,
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'mean_tau_b': ci_result['mean'],
            'ci_lower': ci_result['lower'],
            'ci_upper': ci_result['upper'],
            'std_error': ci_result['std'],
            'n_prompts': eval_result['n_prompts'],
            'n_bootstrap': args.n_bootstrap,
            'confidence_level': args.confidence
        }
        results.append(result)
        
        # Print summary
        print(f"\n✓ Results:")
        print(f"  Mean Kendall's τ-b: {result['mean_tau_b']:.4f}")
        print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  Std Error: {result['std_error']:.4f}")
        print(f"  Evaluated on {result['n_prompts']} prompts")
        
        # Clean up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # Save all results
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    # Find best checkpoint and print final summary
    if len(results) > 1:
        best_checkpoint = max(results, key=lambda x: x['mean_tau_b'])
        print(f"\n{'='*80}")
        print("BEST CHECKPOINT")
        print(f"{'='*80}")
        print(f"Seed: {best_checkpoint['seed']}")
        print(f"Epoch: {best_checkpoint['epoch']}")
        print(f"Mean Kendall's τ-b: {best_checkpoint['mean_tau_b']:.4f}")
        print(f"95% CI: [{best_checkpoint['ci_lower']:.4f}, {best_checkpoint['ci_upper']:.4f}]")
        print(f"Path: {best_checkpoint['checkpoint_path']}")
        
        # Save best checkpoint info for pipeline use
        import datetime
        best_info = {
            'model_name': model_name,
            'best_checkpoint': {
                'seed': best_checkpoint['seed'],
                'epoch': best_checkpoint['epoch'],
                'path': best_checkpoint['checkpoint_path'],
                'mean_tau_b': best_checkpoint['mean_tau_b'],
                'ci_lower': best_checkpoint['ci_lower'],
                'ci_upper': best_checkpoint['ci_upper'],
                'std_error': best_checkpoint['std_error']
            },
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        best_info_file = output_path / 'best_checkpoint.json'
        with open(best_info_file, 'w') as f:
            json.dump(best_info, f, indent=2)
        print(f"Best checkpoint info saved to: {best_info_file}")
    else:
        single_result = results[0]
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Seed: {single_result['seed']}")
        print(f"Epoch: {single_result['epoch']}")
        print(f"Mean Kendall's τ-b: {single_result['mean_tau_b']:.4f}")
        print(f"95% CI: [{single_result['ci_lower']:.4f}, {single_result['ci_upper']:.4f}]")
        print(f"Std Error: {single_result['std_error']:.4f}")
        print(f"Path: {single_result['checkpoint_path']}")
        
        # No need to save checkpoint info for single seed - summary reads from results.json
    
    # Skip saving summary.txt to avoid duplication - use create_pipeline_summary.py instead
    """
    summary_file = output_path / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"EVALUATION SUMMARY: {args.model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test dataset: {args.test_data}\n")
        f.write(f"Number of test prompts: {len(test_dataset)}\n")
        f.write(f"Bootstrap samples: {args.n_bootstrap}\n")
        f.write(f"Confidence level: {args.confidence*100:.0f}%\n")
        f.write(f"Metric: Kendall's τ-b (handles ties naturally)\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST CHECKPOINT\n")
        f.write("="*80 + "\n")
        f.write(f"{'Seed':<8} {'Epoch':<8} {'Mean τ-b':<12} {'95% CI':<25} {'Std Error':<12}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            f.write(f"{r['seed']:<8} {r['epoch']:<8} {r['mean_tau_b']:<12.4f} {ci_str:<25} {r['std_error']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        if len(results) > 1:
            f.write("BEST CHECKPOINT\n")
            f.write("="*80 + "\n")
            f.write(f"Seed: {best_checkpoint['seed']}\n")
            f.write(f"Epoch: {best_checkpoint['epoch']}\n")
            f.write(f"Mean Kendall's τ-b: {best_checkpoint['mean_tau_b']:.4f}\n")
            f.write(f"95% CI: [{best_checkpoint['ci_lower']:.4f}, {best_checkpoint['ci_upper']:.4f}]\n")
            f.write(f"Std Error: {best_checkpoint['std_error']:.4f}\n")
            f.write(f"Path: {best_checkpoint['checkpoint_path']}\n")
        else:
            f.write("CHECKPOINT\n")
            f.write("="*80 + "\n")
            single_result = results[0]
            f.write(f"Seed: {single_result['seed']}\n")
            f.write(f"Epoch: {single_result['epoch']}\n")
            f.write(f"Mean Kendall's τ-b: {single_result['mean_tau_b']:.4f}\n")
            f.write(f"95% CI: [{single_result['ci_lower']:.4f}, {single_result['ci_upper']:.4f}]\n")
            f.write(f"Std Error: {single_result['std_error']:.4f}\n")
            f.write(f"Path: {single_result['checkpoint_path']}\n")
    """
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    # print(f"Summary saved to: {summary_file}")  # Commented out to avoid duplication
    print()


if __name__ == '__main__':
    main()
