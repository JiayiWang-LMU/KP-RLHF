#!/usr/bin/env python3
"""
Run RewardBench 2 evaluation on trained reward models.

Goal: Evaluate reward models on standardized benchmark (RewardBench 2).

Setup:
- Models: Trained reward models
- Dataset: allenai/reward-bench-2 (Best-of-4 selection, ties, multiple categories)
- Metrics: Per-category accuracy (Chat, Safety, Reasoning, etc.)

This script supports one mode:
1. Path-based: --model_path reward_model_out/M_pair/seed_42/checkpoint-1800

Note: RewardBench requires specific package versions.
      Use --install to install rewardbench and --restore to restore main packages.

Usage:
    python evaluation/run_rewardbench.py --model_path reward_model_out/M_pair/seed_42/checkpoint-1800
    python evaluation/run_rewardbench.py --model_path reward_model_out/M_pair/seed_42/checkpoint-1800 --install --restore
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch


def check_rewardbench_installed():
    """Check if rewardbench package is installed."""
    try:
        import rewardbench
        return True
    except ImportError:
        return False


def install_rewardbench():
    """Install rewardbench package (will downgrade transformers/trl)."""
    print("\n[INSTALL] Installing rewardbench (will temporarily downgrade transformers/trl)...")
    requirements_file = Path(__file__).parent.parent / "requirements_rewardbench.txt"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"
    ])
    print("[OK] rewardbench installed successfully!")


def restore_main_packages():
    """Restore main packages after rewardbench evaluation."""
    print("\n[RESTORE] Restoring main packages (transformers, trl)...")
    requirements_file = Path(__file__).parent.parent / "requirements_windows.txt"
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"
    ])
    print("[OK] Main packages restored!")


def run_rewardbench_v2_evaluation(
    model_path: str, 
    output_dir: str = None, 
    batch_size: int = 8,
    debug: bool = False
):
    """
    Run RewardBench 2 evaluation

    Args:
        model_path: Path to the reward model checkpoint (absolute path)
        output_dir: Optional output directory for results
        batch_size: Batch size for evaluation
        debug: Run on small subset for testing
        
    Returns:
        Dictionary with evaluation results
    """

    from accelerate import Accelerator
    from accelerate.logging import get_logger
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from tqdm import tqdm
    from datasets import Dataset, load_dataset
    import pandas as pd
    
    from rewardbench import (
        REWARD_MODEL_CONFIG,
        check_tokenizer_chat_template,
    )
    
    CORE_EVAL_SET_V2 = "allenai/reward-bench-2"
    
    # Convert to absolute path
    model_path = str(Path(model_path).resolve())
    
    # Prepare output directory
    if output_dir is None:
        model_name = Path(model_path).parent.parent.name
        output_dir = Path("evaluation_results") / model_name / "rewardbench2"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("REWARDBENCH 2 EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Dataset: {CORE_EVAL_SET_V2}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Debug mode: {debug}")
    
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index
    
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Running reward model on {model_path}")
    
    # Use default_v2 config for our model (Seq. Classifier)
    config = REWARD_MODEL_CONFIG.get("default_v2", REWARD_MODEL_CONFIG["default"])
    logger.info(f"Using reward model config: {config}")
    
    # Get model type and pipeline builder
    model_builder = config.get("model_builder", AutoModelForSequenceClassification.from_pretrained)
    quantized = config.get("quantized", False)
    model_type = config.get("model_type", "Seq. Classifier")
    
    ############################
    # Load tokenizer
    ############################
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    
    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    
    # Load raw dataset
    raw_dataset = load_dataset(CORE_EVAL_SET_V2, split="test")
    
    # Store metadata before unrolling
    total_completions = raw_dataset["total_completions"]
    num_correct = raw_dataset["num_correct"]
    logger.info(f"Total completions: {sum(total_completions)}")
    logger.info(f"Num prompts: {len(raw_dataset)}")
    
    # Unroll each row: each prompt has multiple chosen and rejected responses
    # We need to score each response individually
    def unroll_output(idx, row):
        rows = []
        options = list(row["chosen"]) + list(row["rejected"])
        
        for i, output in enumerate(options):
            new_row = {
                "id": idx,
                "subset": row.get("subset", "unknown"),
                "prompt": row["prompt"],
                "response": output,
                "text": f"{row['prompt']}\n\n{output}",
            }
            rows.append(new_row)
        return rows
    
    new_dataset = []
    for idx, row in enumerate(raw_dataset):
        new_dataset.extend(unroll_output(idx, row))
    
    dataset = Dataset.from_pandas(pd.DataFrame(data=new_dataset))
    
    # Store subset info before removing
    subsets = dataset["subset"]
    ids = dataset["id"]
    
    # Debug: use only first 10 prompts (40 rows in unrolled)
    if debug:
        # Find where prompt id changes to limit to first 10 prompts
        max_rows = min(40, len(dataset))
        dataset = dataset.select(range(max_rows))
        subsets = subsets[:max_rows]
        ids = ids[:max_rows]
        # Limit metadata too
        unique_ids = list(set(ids[:max_rows]))
        max_prompts = len(unique_ids)
        total_completions = total_completions[:max_prompts]
        num_correct = num_correct[:max_prompts]
    
    logger.info(f"Unrolled dataset size: {len(dataset)}")
    
    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = batch_size
    logger.info("*** Load reward model ***")
    
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    
    # Load model
    model_kwargs = {"torch_dtype": torch.float16}
    if quantized:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = {"": current_device}
    else:
        # Put model on GPU
        model_kwargs["device_map"] = {"": current_device}
    
    model = model_builder(
        model_path,
        **model_kwargs,
        trust_remote_code=False,
    )
    
    # Build the pipeline
    reward_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        **reward_pipeline_kwargs,
    )
    
    # Set pad token if needed
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id
    
    ############################
    # Run inference
    ############################
    logger.info("*** Running inference ***")
    results_all = []
    
    texts = dataset["text"]
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch_texts = texts[i:i + BATCH_SIZE]
        outputs = reward_pipe(batch_texts)
        scores = [out["score"] for out in outputs]
        results_all.extend(scores)
    
    ############################
    # Reroll scores back to per-prompt
    ############################
    logger.info("*** Processing results ***")
    
    # Group scores by prompt id
    from collections import defaultdict
    prompt_scores = defaultdict(list)
    prompt_subsets = {}
    for i, (pid, score, subset) in enumerate(zip(ids, results_all, subsets)):
        prompt_scores[pid].append(score)
        prompt_subsets[pid] = subset
    
    # For each prompt, check if best score is from a "correct" response
    # In RewardBench 2, chosen responses are correct, rejected are incorrect
    # The order in unrolling is: chosen first, then rejected
    correct_count = 0
    results_per_prompt = []
    
    for pid in sorted(prompt_scores.keys()):
        scores = prompt_scores[pid]
        n_total = total_completions[pid]
        n_correct = num_correct[pid]
        
        # Best score wins
        best_idx = np.argmax(scores)
        
        # If best_idx < n_correct, it's a correct (chosen) response
        is_correct = best_idx < n_correct
        if is_correct:
            correct_count += 1
        
        results_per_prompt.append({
            "id": int(pid),
            "subset": prompt_subsets[pid],
            "correct": bool(is_correct),
            "n_options": int(n_total),
            "n_correct": int(n_correct),
            "best_idx": int(best_idx),
        })
    
    ############################
    # Calculate overall accuracy
    ############################
    accuracy = correct_count / len(results_per_prompt)
    logger.info(f"\n*** Results ***")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{len(results_per_prompt)})")
    
    # Group by subset for per-category results
    subset_correct = defaultdict(list)
    for r in results_per_prompt:
        subset_correct[r["subset"]].append(r["correct"])
    
    category_results = {}
    for subset, correct_list in sorted(subset_correct.items()):
        cat_acc = np.mean(correct_list)
        category_results[subset] = cat_acc
        logger.info(f"  {subset}: {cat_acc:.4f} ({sum(correct_list)}/{len(correct_list)})")
    
    ############################
    # Save results
    ############################
    results = {
        "model": model_path,
        "dataset": CORE_EVAL_SET_V2,
        "overall_accuracy": float(accuracy),
        "num_prompts": len(results_per_prompt),
        "categories": {k: float(v) for k, v in category_results.items()},
        "per_prompt_results": results_per_prompt,
    }
    
    results_file = output_dir / "rewardbench2_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[SAVED] Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RewardBench 2 evaluation")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to reward model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on small subset for testing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install rewardbench package without prompting (downgrades transformers/trl)"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore main packages (transformers/trl) after evaluation"
    )
    
    args = parser.parse_args()
    
    # Install rewardbench if requested
    if args.install:
        install_rewardbench()
    
    # Check if rewardbench is available
    if not check_rewardbench_installed():
        print("[ERROR] rewardbench package not found. Use --install to install it.")
        sys.exit(1)
    
    # Run evaluation
    # If max_samples is set, enable debug mode for quick testing
    debug_mode = args.debug or (args.max_samples is not None and args.max_samples < 100)
    results = run_rewardbench_v2_evaluation(
        args.model_path, 
        args.output_dir, 
        args.batch_size,
        debug_mode
    )
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        if "categories" in results:
            for cat, score in results["categories"].items():
                print(f"  {cat}: {score:.4f}")
    
    # Restore main packages if requested
    if args.restore:
        restore_main_packages()


if __name__ == "__main__":
    main()
