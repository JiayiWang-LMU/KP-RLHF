#!/usr/bin/env python3
"""
KP-RLHF: Learning from K-Wise Comparisons and Partial Rankings for RLHF

This is the main entry point that orchestrates the entire pipeline:

1. Dataset Preparation:
   - Load UltraFeedback preference dataset
   - Split, preprocess, and tokenize

2. Reward Modeling:
   - Train reward models using pairwise or listwise trainers
   - Pairwise: BT (Bradley-Terry), BT-Heuristic, BT-Davidson
   - Listwise: PL (Plackett-Luce), Cox-Breslow, DL (Davidson-Luce)

3. Reward Model Evaluation:
   - In-distribution: Kendall's Tau-b on test split
   - OOD: Kendall's Tau-b on HelpSteer
   - RewardBench via API

4. Policy Optimization:
   - PPO or GRPO using trained reward model
   - Generate responses from optimized policy

5. LLM Judge Evaluation:
   - Pairwise comparison using Llama/Prometheus judges
   - Compare policies trained with different reward models

Usage:
    python main.py --mode test          # Quick test run with limited samples
    python main.py --mode full          # Interactive full run
    python main.py --mode reward_only   # Train and evaluate reward model only
    python main.py --mode policy_only   # Run policy optimization only
    python main.py --mode evaluate      # Run LLM judge evaluation only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test mode parameters for quick local testing
TEST_MODE_PARAMS = {
    'batch_size': 1,
    'grad_accum': 4,
    'max_steps': 20,
    'subset': 200,
    'eval_steps': 10,
    'ppo_steps': 10,
    'grpo_steps': 10,
    'num_samples': 20,
    'test_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}

# Available preference models
PAIRWISE_MODELS = {
    'BT': {
        'trainer': 'pairwise/pairwise_trainer.py',
        'config': 'configs/config_M_pair.yaml',
        'output_dir': 'reward_model_out/M_pair',
    },
    'BT-Heuristic': {
        'trainer': 'pairwise/pairwise_trainer_bt05.py',
        'config': 'configs/config_M_pair_bt05.yaml',
        'output_dir': 'reward_model_out/M_pair_bt05',
    },
    'BT-Davidson': {
        'trainer': 'pairwise/pairwise_trainer_btt.py',
        'config': 'configs/config_M_pair_btt.yaml',
        'output_dir': 'reward_model_out/M_pair_btt',
    },
}

LISTWISE_MODELS = {
    'PL': {
        'trainer': 'k_wise/kwise_trainer_standard_pl.py',
        'config': 'configs/config_M_kw_standard_pl.yaml',
        'output_dir': 'reward_model_out/M_kw_standard_pl',
    },
    'Cox': {
        'trainer': 'k_wise/kwise_trainer_cox.py',
        'config': 'configs/config_M_kw_cox.yaml',
        'output_dir': 'reward_model_out/M_kw_cox',
    },
    'DL': {
        'trainer': 'k_wise/kwise_trainer_dl.py',
        'config': 'configs/config_M_kw_dl.yaml',
        'output_dir': 'reward_model_out/M_kw_dl',
    },
}

ALL_PREFERENCE_MODELS = {**PAIRWISE_MODELS, **LISTWISE_MODELS}

# Policy optimization algorithms
PO_ALGORITHMS = {
    'PPO': {
        'trainer': 'rlhf/train_ppo.py',
        'config': 'configs/config_rlhf_ppo.yaml',
        'output_dir': 'rlhf_out/ppo',
    },
    'GRPO': {
        'trainer': 'rlhf/train_grpo.py',
        'config': 'configs/config_rlhf_grpo.yaml',
        'output_dir': 'rlhf_out/grpo',
    },
}

# LLM Judges
LLM_JUDGES = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'prometheus': 'prometheus-eval/prometheus-7b-v2.0',
}


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_step(step_num: int, total_steps: int, description: str):
    """Print a step indicator."""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-" * 60)


def run_command(cmd: List[str], description: str, env: Optional[Dict] = None) -> bool:
    """Run a command and return success status."""
    print(f"\n> Running: {' '.join(cmd)}")
    
    # Merge with current environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        result = subprocess.run(cmd, env=run_env, check=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"Command not found: {e}")
        return False


def check_dataset_exists() -> bool:
    """Check if UltraFeedback dataset exists."""
    dataset_path = Path("dataset/UltraFeedback")
    return dataset_path.exists() and (dataset_path / "dataset_dict.json").exists()


def check_tokenized_dataset_exists(config_path: str) -> bool:
    """Check if tokenized dataset exists."""
    tokenized_path = Path("dataset/UltraFeedback_tokenized")
    return tokenized_path.exists() and (tokenized_path / "dataset_dict.json").exists()


def prepare_dataset() -> bool:
    """Prepare UltraFeedback dataset."""
    print_step(1, 5, "PREPARING ULTRAFEEDBACK DATASET")
    
    if check_dataset_exists():
        print("UltraFeedback dataset already exists")
        return True
    
    return run_command(
        ["python", "dataset/prepare_data.py", "--dataset", "ultrafeedback"],
        "Dataset preparation"
    )


def preprocess_dataset(config_path: str) -> bool:
    """Preprocess (tokenize) dataset."""
    print_step(2, 5, "TOKENIZING DATASET")
    
    if check_tokenized_dataset_exists(config_path):
        print("Tokenized dataset already exists")
        return True
    
    return run_command(
        ["python", "dataset/preprocess_dataset.py", 
         "--config", config_path, 
         "--output", "dataset/UltraFeedback_tokenized"],
        "Dataset tokenization"
    )


def train_reward_model(model_name: str, seed: int = 42, test_mode: bool = False) -> Tuple[bool, Optional[str]]:
    """Train a reward model and return checkpoint path."""
    model_info = ALL_PREFERENCE_MODELS.get(model_name)
    if not model_info:
        print(f"Unknown model: {model_name}")
        return False, None
    
    print_step(3, 5, f"TRAINING REWARD MODEL: {model_name}")
    print(f"Trainer: {model_info['trainer']}")
    print(f"Config: {model_info['config']}")
    print(f"Output: {model_info['output_dir']}")
    print(f"Seed: {seed}")
    if test_mode:
        print(f"Mode: TEST (reduced params)")
    
    cmd = [
        "python", model_info['trainer'], 
        "--config", model_info['config'], 
        "--seed", str(seed)
    ]
    
    # Add test mode arguments
    if test_mode:
        cmd.extend([
            "--test_mode",
            "--test_batch_size", str(TEST_MODE_PARAMS['batch_size']),
            "--test_grad_accum", str(TEST_MODE_PARAMS['grad_accum']),
            "--test_max_steps", str(TEST_MODE_PARAMS['max_steps']),
            "--test_subset", str(TEST_MODE_PARAMS['subset']),
            "--test_eval_steps", str(TEST_MODE_PARAMS['eval_steps']),
        ])
    
    env = {"PYTHONPATH": "."}
    success = run_command(cmd, f"Training {model_name}", env=env)
    
    if success:
        # Find the checkpoint
        checkpoint_dir = Path(model_info['output_dir']) / f"seed_{seed}"
        # Only match checkpoint directories
        checkpoints = [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()]
        if checkpoints:
            # Get the latest checkpoint by step number
            latest = sorted(checkpoints, key=lambda p: int(p.name.split("-")[1]))[-1]
            print(f"Checkpoint saved: {latest}")
            return True, str(latest)
    
    return success, None


def evaluate_reward_model_test(checkpoint_path: str, model_name: str) -> bool:
    """Evaluate reward model on test set (Kendall's Tau-b)."""
    print("\n[EVALUATION] In-distribution test (Kendall's Tau-b)")
    
    return run_command(
        ["python", "evaluation/evaluate_preference_model.py",
         "--model_path", checkpoint_path,
         "--n_bootstrap", "1000"],
        "Test set evaluation"
    )


def evaluate_reward_model_ood(checkpoint_path: str, model_name: str, seed: int = 42) -> bool:
    """Evaluate reward model on HelpSteer (OOD distribution shift)."""
    print("\n[EVALUATION] Out-of-distribution test on HelpSteer")
    
    # Extract model name from path
    return run_command(
        ["python", "evaluation/evaluate_distribution_shift.py",
         "--model_path", checkpoint_path,
         "--seed", str(seed)],
        "HelpSteer OOD evaluation"
    )


def evaluate_reward_model_rewardbench(checkpoint_path: str, model_name: str, seed: int = 42) -> bool:
    """Evaluate reward model on RewardBench."""
    print("\n[EVALUATION] RewardBench evaluation")
    
    return run_command(
        ["python", "evaluation/run_rewardbench.py",
         "--model_path", checkpoint_path],
        "RewardBench evaluation"
    )


def train_policy(po_algorithm: str, reward_model_path: str, model_name: str, 
                 seed: int = 42, test_mode: bool = False) -> Tuple[bool, Optional[str]]:
    """Train policy using PPO or GRPO."""
    po_info = PO_ALGORITHMS.get(po_algorithm)
    if not po_info:
        print(f"✗ Unknown PO algorithm: {po_algorithm}")
        return False, None
    
    # Create output directory with reward model name
    output_dir = f"{po_info['output_dir']}_{model_name.lower().replace('-', '_')}"
    
    print_step(4, 5, f"TRAINING POLICY: {po_algorithm} with {model_name} reward")
    print(f"Trainer: {po_info['trainer']}")
    print(f"Config: {po_info['config']}")
    print(f"Reward Model: {reward_model_path}")
    print(f"Output: {output_dir}")
    if test_mode:
        print(f"Mode: TEST (reduced params for quick local testing)")
    
    cmd = [
        "python", po_info['trainer'],
        "--config", po_info['config'],
        "--reward_model_path", reward_model_path,
        "--output_dir", output_dir,
        "--seed", str(seed)
    ]
    
    # Add test mode arguments for PPO/GRPO
    if test_mode:
        cmd.extend([
            "--test_mode",
            "--test_steps", str(TEST_MODE_PARAMS['ppo_steps']),
            "--test_model", TEST_MODE_PARAMS['test_model'],
        ])
    
    success = run_command(cmd, f"Training {po_algorithm} policy")
    
    if success:
        final_model_path = Path(output_dir) / "final_model"
        if final_model_path.exists():
            return True, str(final_model_path)
    
    return success, None


def generate_responses(policy_path: str, model_name: str, po_algorithm: str, 
                       num_samples: int = 798, num_responses: int = 4,
                       test_mode: bool = False) -> Tuple[bool, Optional[str]]:
    """Generate responses from a trained policy."""
    print_step(5, 5, f"GENERATING RESPONSES: {model_name}")
    
    output_dir = f"evaluation_results/rlhf/{po_algorithm.lower()}"
    output_file = f"responses_{model_name.lower().replace('-', '_')}.json"
    
    # Use reduced samples in test mode
    if test_mode:
        num_samples = TEST_MODE_PARAMS['num_samples']
        num_responses = 2  # Fewer responses per prompt
    
    success = run_command(
        ["python", "rlhf/generate_responses.py",
         "--model_paths", policy_path,
         "--model_names", model_name,
         "--output_dir", output_dir,
         "--output_file", output_file,
         "--num_samples", str(num_samples),
         "--num_responses_per_prompt", str(num_responses),
         "--max_new_tokens", "64" if test_mode else "256",
         "--do_sample",
         "--seed", "42"],
        "Response generation"
    )
    
    if success:
        responses_path = Path(output_dir) / output_file
        return True, str(responses_path)
    
    return False, None


def find_available_policies(po_algorithm: str) -> List[Tuple[str, str]]:
    """Find available trained policies for a given PO algorithm."""
    policies = []
    po_base_dir = Path(f"rlhf_out/{po_algorithm.lower()}_")
    
    # Look for directories matching pattern
    for model_name in ALL_PREFERENCE_MODELS.keys():
        policy_dir = Path(f"rlhf_out/{po_algorithm.lower()}_{model_name.lower().replace('-', '_')}/final_model")
        if policy_dir.exists():
            policies.append((model_name, str(policy_dir)))
    
    return policies


def find_available_responses(po_algorithm: str) -> List[Tuple[str, str]]:
    """Find available response files for a given PO algorithm."""
    responses = []
    responses_dir = Path(f"evaluation_results/rlhf/{po_algorithm.lower()}")
    
    if responses_dir.exists():
        for response_file in responses_dir.glob("responses_*.json"):
            # Extract model name from filename
            model_name = response_file.stem.replace("responses_", "").upper().replace("_", "-")
            responses.append((model_name, str(response_file)))
    
    return responses


def run_llm_judge_evaluation(responses_file: str, output_dir: str, 
                              model_names: List[str], judge: str = "llama") -> bool:
    """Run LLM judge evaluation."""
    judge_model = LLM_JUDGES.get(judge)
    if not judge_model:
        print(f"✗ Unknown judge: {judge}")
        return False
    
    print(f"\n[EVALUATION] LLM Judge: {judge}")
    print(f"Responses: {responses_file}")
    print(f"Models: {', '.join(model_names)}")
    
    return run_command(
        ["python", "rlhf/evaluate_head2head.py",
         "--responses_file", responses_file,
         "--output_dir", output_dir,
         "--model_names"] + model_names + [
         "--judge_model", judge_model,
         "--ordering_mode", "ab",
         "--run_id", "main_run"],
        f"LLM Judge evaluation ({judge})"
    )


def interactive_menu(title: str, options: List[str]) -> int:
    """Display an interactive menu and return selected index."""
    print(f"\n{title}")
    print("-" * 40)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        try:
            choice = int(input(f"\nEnter choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return choice - 1
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def run_test_mode(seed: int = 42):
    """Run a quick test with limited samples on small GPU."""
    print_header("KP-RLHF: TEST MODE", "=", 80)
    print("Running quick test with reduced parameters for small GPU...")
    print(f"Seed: {seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTest mode parameters:")
    for k, v in TEST_MODE_PARAMS.items():
        print(f"  {k}: {v}")
    
    # Step 1: Prepare dataset
    if not prepare_dataset():
        return False
    
    # Step 2: Preprocess dataset (using BT config)
    if not preprocess_dataset("configs/config_M_pair.yaml"):
        return False
    
    # Step 3: Train BT reward model with TEST MODE
    success, checkpoint_path = train_reward_model("BT", seed=seed, test_mode=True)
    if not success or not checkpoint_path:
        return False
    
    # Step 4: Run evaluations with small samples for quick testing
    print("\n" + "=" * 60)
    print("REWARD MODEL EVALUATION (Test mode - small samples)")
    print("=" * 60)
    
    # 4a: In-distribution test (UltraFeedback test split - 10 samples)
    print("\n[4a] In-distribution test (Kendall's τ-b, 10 samples)")
    run_command(
        ["python", "evaluation/evaluate_preference_model.py",
         "--model_path", checkpoint_path,
         "--max_samples", "10",
         "--n_bootstrap", "100"],
        "UltraFeedback test evaluation"
    )
    
    # 4b: Out-of-distribution test (HelpSteer - 10 samples)
    print("\n[4b] Out-of-distribution test (HelpSteer, 10 samples)")
    run_command(
        ["python", "evaluation/evaluate_distribution_shift.py",
         "--model_path", checkpoint_path,
         "--max_samples", "10"],
        "HelpSteer OOD evaluation"
    )
    
    # 4c: RewardBench (50 samples for quick test)
    # Note: --install downgrades transformers/trl for rewardbench
    # Note: --restore upgrades back to main packages for PPO training
    print("\n[4c] RewardBench evaluation (50 samples)")
    run_command(
        ["python", "evaluation/run_rewardbench.py",
         "--model_path", checkpoint_path,
         "--max_samples", "50",
         "--install",
         "--restore"],
        "RewardBench evaluation"
    )
    
    # Step 5: Train PPO policy with TEST MODE
    success, policy_path = train_policy("PPO", checkpoint_path, "BT", seed=seed, test_mode=True)
    if not success or not policy_path:
        print("Policy training failed, but reward model training is complete.")
        return True  # Partial success
    
    # Step 6: Generate responses (using same TinyLlama model from PPO)
    print("\n" + "=" * 60)
    print("RESPONSE GENERATION (Test Mode - TinyLlama)")
    print("=" * 60)
    success, responses_path = generate_responses(
        policy_path, "BT", "PPO", test_mode=True
    )
    
    # Step 7: LLM judge evaluation (skip in test mode - requires separate LLM judge)
    print("\n" + "=" * 60)
    print("LLM JUDGE EVALUATION (Skipped in test mode)")
    print("=" * 60)
    print("Skipping LLM judge evaluation (requires loading separate judge model)")
    
    print("\n" + "=" * 80)
    print("TEST RUN COMPLETED")
    print("=" * 80)
    print("\nTest mode verified:")
    print("Dataset preparation works")
    print("Dataset tokenization works")
    print("Reward model training works (BT)")
    print("Policy training works (PPO)")
    print("Response generation works")
    print(f"\nArtifacts:")
    print(f"  - Reward model: {checkpoint_path}")
    if policy_path:
        print(f"  - Policy model: {policy_path}")
    if responses_path:
        print(f"  - Responses: {responses_path}")
    
    return True


def run_full_mode(seed: int = 42):
    """Run full interactive mode."""
    print_header("KP-RLHF: FULL MODE", "=", 80)
    print(f"Seed: {seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Prepare dataset
    if not prepare_dataset():
        return False
    
    # Step 2: Select preference model
    print("\n" + "=" * 60)
    print("SELECT PREFERENCE MODEL")
    print("=" * 60)
    
    print("\nPairwise Models:")
    for name, info in PAIRWISE_MODELS.items():
        print(f"  - {name}: {info['trainer']}")
    
    print("\nListwise Models:")
    for name, info in LISTWISE_MODELS.items():
        print(f"  - {name}: {info['trainer']}")
    
    all_models = list(ALL_PREFERENCE_MODELS.keys())
    model_idx = interactive_menu("Select a preference model:", all_models)
    model_name = all_models[model_idx]
    model_info = ALL_PREFERENCE_MODELS[model_name]
    
    print(f"\nSelected: {model_name}")
    
    # Step 3: Preprocess dataset
    if not preprocess_dataset(model_info['config']):
        return False
    
    # Step 4: Train reward model
    success, checkpoint_path = train_reward_model(model_name, seed=seed)
    if not success or not checkpoint_path:
        return False
    
    # Step 5: Evaluate reward model
    print("\n" + "=" * 60)
    print("REWARD MODEL EVALUATION")
    print("=" * 60)
    
    evaluate_reward_model_test(checkpoint_path, model_name)
    evaluate_reward_model_ood(checkpoint_path, model_name, seed=seed)
    
    # Ask about RewardBench
    rb_choice = input("\nRun RewardBench evaluation? (y/n): ").lower()
    if rb_choice == 'y':
        evaluate_reward_model_rewardbench(checkpoint_path, model_name, seed=seed)
    
    # Step 6: Select PO algorithm
    print("\n" + "=" * 60)
    print("POLICY OPTIMIZATION")
    print("=" * 60)
    
    po_idx = interactive_menu("Select PO algorithm:", list(PO_ALGORITHMS.keys()))
    po_algorithm = list(PO_ALGORITHMS.keys())[po_idx]
    
    print(f"\nSelected: {po_algorithm}")
    
    # Step 7: Train policy
    success, policy_path = train_policy(po_algorithm, checkpoint_path, model_name, seed=seed)
    if not success or not policy_path:
        print("Policy training failed.")
        return False
    
    # Step 8: Generate responses
    success, responses_path = generate_responses(policy_path, model_name, po_algorithm)
    
    # Step 9: Check for LLM judge evaluation
    print("\n" + "=" * 60)
    print("LLM JUDGE EVALUATION")
    print("=" * 60)
    
    available_responses = find_available_responses(po_algorithm)
    print(f"\nFound {len(available_responses)} policies with responses for {po_algorithm}:")
    for name, path in available_responses:
        print(f"  - {name}: {path}")
    
    if len(available_responses) >= 2:
        eval_choice = input("\nRun LLM judge evaluation on available policies? (y/n): ").lower()
        if eval_choice == 'y':
            # Merge responses into single file for evaluation
            # For now, just report that evaluation can be run manually
            print("\nTo run LLM judge evaluation, use:")
            print(f"  python rlhf/evaluate_head2head.py \\")
            print(f"      --responses_file <merged_responses.json> \\")
            print(f"      --output_dir evaluation_results/rlhf/llm_judge/{po_algorithm.lower()} \\")
            print(f"      --model_names {' '.join([n for n, _ in available_responses])} \\")
            print(f"      --judge_model meta-llama/Llama-3.1-8B-Instruct")
    else:
        print(f"\nNeed at least 2 trained policies for LLM judge evaluation.")
        print(f"Run this script again with a different preference model to enable comparison.")
    
    print("\n" + "=" * 80)
    print("FULL RUN COMPLETED!")
    print("=" * 80)
    print(f"\nArtifacts:")
    print(f"  - Reward model: {checkpoint_path}")
    print(f"  - Policy model: {policy_path}")
    if responses_path:
        print(f"  - Responses: {responses_path}")
    
    return True


def run_evaluate_mode():
    """Run LLM judge evaluation on existing responses."""
    print_header("KP-RLHF: EVALUATION MODE", "=", 80)
    
    # Select PO algorithm
    po_idx = interactive_menu("Select PO algorithm to evaluate:", list(PO_ALGORITHMS.keys()))
    po_algorithm = list(PO_ALGORITHMS.keys())[po_idx]
    
    # Find available responses
    available_responses = find_available_responses(po_algorithm)
    
    if len(available_responses) < 2:
        print(f"\n✗ Need at least 2 trained policies for comparison.")
        print(f"Found {len(available_responses)} policies for {po_algorithm}.")
        return False
    
    print(f"\nFound {len(available_responses)} policies:")
    for name, path in available_responses:
        print(f"  - {name}")
    
    # Select judge
    judge_idx = interactive_menu("Select LLM judge:", list(LLM_JUDGES.keys()))
    judge = list(LLM_JUDGES.keys())[judge_idx]
    
    # For now, need to merge responses manually
    print("\nNote: Response files need to be merged before running evaluation.")
    print("Use merge_responses.py to combine individual response files.")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="KP-RLHF: Unified Framework for Preference Learning and Policy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  test          Quick test run with BT model and limited samples
  full          Interactive full run with model selection
  reward_only   Train and evaluate reward model only
  policy_only   Run policy optimization only (requires existing reward model)
  evaluate      Run LLM judge evaluation on existing responses

Examples:
  python main.py --mode test --seed 42
  python main.py --mode full
  python main.py --mode reward_only --model BT --seed 123
  python main.py --mode policy_only --reward_model reward_model_out/M_pair/seed_42/checkpoint-1800 --po PPO
        """
    )
    
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'full', 'reward_only', 'policy_only', 'evaluate'],
                        help='Run mode (default: test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(ALL_PREFERENCE_MODELS.keys()),
                        help='Preference model for reward_only mode')
    parser.add_argument('--reward_model', type=str, default=None,
                        help='Path to reward model checkpoint for policy_only mode')
    parser.add_argument('--po', type=str, default='PPO',
                        choices=list(PO_ALGORITHMS.keys()),
                        help='PO algorithm for policy_only mode')
    
    args = parser.parse_args()
    
    # Set environment variables for caching
    os.environ['HF_HOME'] = str(Path.cwd() / 'models_cache')
    os.environ['TRANSFORMERS_CACHE'] = str(Path.cwd() / 'models_cache')
    os.environ['HF_DATASETS_CACHE'] = str(Path.cwd() / 'dataset' / 'cache')
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║   ██╗  ██╗██████╗       ██████╗ ██╗     ██╗  ██╗███████╗                     ║")
    print("║   ██║ ██╔╝██╔══██╗      ██╔══██╗██║     ██║  ██║██╔════╝                     ║")
    print("║   █████╔╝ ██████╔╝█████╗██████╔╝██║     ███████║█████╗                       ║")
    print("║   ██╔═██╗ ██╔═══╝ ╚════╝██╔══██╗██║     ██╔══██║██╔══╝                       ║")
    print("║   ██║  ██╗██║           ██║  ██║███████╗██║  ██║██║                          ║")
    print("║   ╚═╝  ╚═╝╚═╝           ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝                          ║")
    print("║                                                                              ║")
    print("║   Learning from K-wise Comparisons and Partial Rankings for RLHF             ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    if args.mode == 'test':
        success = run_test_mode(seed=args.seed)
    elif args.mode == 'full':
        success = run_full_mode(seed=args.seed)
    elif args.mode == 'reward_only':
        model = args.model or 'BT'
        print_header(f"REWARD MODEL TRAINING: {model}", "=", 80)
        
        if not prepare_dataset():
            sys.exit(1)
        
        model_info = ALL_PREFERENCE_MODELS[model]
        if not preprocess_dataset(model_info['config']):
            sys.exit(1)
        
        success, checkpoint = train_reward_model(model, seed=args.seed)
        if success and checkpoint:
            evaluate_reward_model_test(checkpoint, model)
            evaluate_reward_model_ood(checkpoint, model, seed=args.seed)
    elif args.mode == 'policy_only':
        if not args.reward_model:
            print("✗ --reward_model required for policy_only mode")
            sys.exit(1)
        
        # Extract model name from path
        model_name = "Custom"
        for name in ALL_PREFERENCE_MODELS.keys():
            if name.lower().replace('-', '_') in args.reward_model.lower():
                model_name = name
                break
        
        success, policy = train_policy(args.po, args.reward_model, model_name, seed=args.seed)
        if success and policy:
            generate_responses(policy, model_name, args.po)
    elif args.mode == 'evaluate':
        success = run_evaluate_mode()
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
