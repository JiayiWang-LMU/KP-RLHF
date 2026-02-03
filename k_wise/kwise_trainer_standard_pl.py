"""Standard Plackett-Luce (PL) trainer.

This script:
- Loads UltraFeedback datasets
- Trains with 3 seeds
- Uses YAML configuration files for hyperparameter management

Standard Plackett-Luce Model:
- For ranking σ = (σ[OK][OK]σ[OK][OK]... [OK]σ[OK] where k [OK]{2, 3, 4}
- P(σ | θ) = ∏ᵢ₌₁ᵏ⁻¹ [exp(θ_σ[OK] / Σⱼ₌ᵢᵏ exp(θ_σ[OK]]
- Log-likelihood: Σᵢ₌₁ᵏ⁻[OK][θ_σ[OK]- logsumexp(θ_σ[OK] ..., θ_σ[OK]]
- Loss: -log P(σ | θ)

Computational Complexity: O(k) per ranking
- Uses reverse cumulative sum trick
- Precompute cum_sum[i] = exp(r[i]) + cum_sum[i+1] backwards
- Avoids O(k²) repeated logsumexp computations
"""

import argparse
import sys
import os
import math
import random
import itertools
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np
import logging
from scipy.stats import kendalltau
from tqdm import tqdm
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradientNormCallback(TrainerCallback):
    """Track gradient norm and its rolling variance for optimization stability analysis."""
    
    def __init__(self, window_size=150):
        self.window_size = window_size
        self.grad_norms = deque(maxlen=window_size)
        
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is not None and state.global_step % args.gradient_accumulation_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)
            rolling_var = np.var(list(self.grad_norms)) if len(self.grad_norms) >= 2 else 0.0
            if hasattr(state, 'log_history'):
                state.log_history.append({
                    'gradient_norm': total_norm,
                    'gradient_norm_rolling_var': rolling_var,
                    'step': state.global_step
                })



def compute_kendall_tau_per_prompt(ranks, rewards):
    """Compute Kendall's Tau-b correlation between ground truth ranks and predicted rewards."""
    valid_pairs = [(ranks[i], rewards[i]) for i in range(len(ranks)) if ranks[i] > 0]
    if len(valid_pairs) < 2:
        return None
    valid_ranks, valid_rewards = zip(*valid_pairs)
    tau, _ = kendalltau([-r for r in valid_ranks], valid_rewards, variant='b')
    if np.isnan(tau):
        return None
    return tau


def load_tokenized_dataset(dataset_path: Path, tokenizer_path: Path):
    """
    Load pre-tokenized dataset and tokenizer.

    Args:
        dataset_path: Path to the tokenized dataset
        tokenizer_path: Path to the saved tokenizer

    Returns:
        train_dataset, tokenizer
    """
    print(f"Loading tokenized dataset from {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    
    train_dataset = ds['train']
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    
    return train_dataset, tokenizer


def build_model(model_name):
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to YAML config file')
    parser.add_argument('--seed', type=int, required=True, help='random seed for reproducibility')
    args = parser.parse_args()
    
    # Add parent directory to path so we can import configs
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Load configuration
    from configs.config_loader import load_config
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Using seed: {args.seed}")
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Clear GPU cache to avoid memory fragmentation from previous runs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Extract config values
    dataset_path = Path(config.dataset.dataset_path)
    tokenizer_path = Path(config.dataset.tokenizer_path) if hasattr(config.dataset, 'tokenizer_path') else dataset_path / "tokenizer"
    model_name = config.model.model_name
    base_output_dir = Path(config.training.output_dir)
    output_dir = base_output_dir / f"seed_{args.seed}"
    max_length = config.model.max_length
    train_subset = config.dataset.train_subset
    
    # Training hyperparameters from config
    per_device_batch_size = config.training.per_device_train_batch_size
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    bf16 = config.training.bf16
    fp16 = config.training.fp16

    # Setup output and logging directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.training.log_dir) / f"seed_{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file handler for logging
    log_file = log_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f'Logging to {log_file}')

    # device info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU required for training.")
    device = torch.device('cuda')

    logger.info('Loading tokenized dataset...')
    
    # Load pre-tokenized dataset
    train_dataset, tokenizer = load_tokenized_dataset(dataset_path, tokenizer_path)
    
    if train_subset > 0:
        train_dataset = train_dataset.select(range(min(train_subset, len(train_dataset))))
    
    # Compute steps per epoch
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_epochs = 3
    max_steps = steps_per_epoch * total_epochs
    save_steps = steps_per_epoch // 4  # Save 12 checkpoints across 3 epochs (every 0.25 epochs)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Save steps: {save_steps}")
    print(f"Total epochs: {total_epochs}")
    print(f"Max steps: {max_steps}")
    print(f"Eval steps: {save_steps}")
    
    # Build model
    model = build_model(model_name)
    model.to(device)

    # Collate function: expand ties into multiple strict orderings
    class PLCollator:
        def __init__(self, tokenizer, max_length):
            """
            Args:
                tokenizer: HuggingFace tokenizer
                max_length: Maximum sequence length
            """
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, batch):
            """Expand ties into multiple strict orderings.
            
            For each prompt with ranks [rank_A, rank_B, rank_C, rank_D]:
            - Group responses by rank (tied buckets)
            - Generate all valid strict orderings by taking the Cartesian product of tied buckets
            - Similar to 'A=B>C>D' becoming 'A>C>D' and 'B>C>D'
            
            Returns:
                input_ids: Flattened input_ids for ALL responses
                prompt_boundaries: List of (start, end) indices for each prompt
                prompt_rankings: List of lists of relative indices (0..K-1) for each prompt
            """
            from torch.nn.utils.rnn import pad_sequence
            
            all_input_ids = []
            all_attention_mask = []
            prompt_boundaries = []  # Track which responses belong to which prompt
            prompt_rankings = []    # Store list of strict orderings (indices) for each prompt
            
            current_idx = 0
            
            for example in batch:
                ranks = {
                    'A': example['rank_A'],
                    'B': example['rank_B'],
                    'C': example['rank_C'],
                    'D': example['rank_D']
                }
                labels = ['A', 'B', 'C', 'D']
                
                # Add all K responses for this prompt
                prompt_start = current_idx
                for label in labels:
                    input_ids_tensor = torch.tensor(example[f'input_ids_{label}'], dtype=torch.long)
                    attention_mask_tensor = torch.tensor(example[f'attention_mask_{label}'], dtype=torch.long)
                    
                    # Truncate to max_length if necessary
                    if input_ids_tensor.size(0) > self.max_length:
                        input_ids_tensor = input_ids_tensor[:self.max_length]
                        attention_mask_tensor = attention_mask_tensor[:self.max_length]
                    
                    all_input_ids.append(input_ids_tensor)
                    all_attention_mask.append(attention_mask_tensor)
                    current_idx += 1
                
                prompt_boundaries.append((prompt_start, current_idx))
                
                # Calculate strict orderings via Cartesian product of tied buckets
                # Group relative indices (0, 1, 2, 3) by rank
                rank_buckets = {}
                for i, label in enumerate(labels):
                    rank = ranks[label]
                    if rank not in rank_buckets:
                        rank_buckets[rank] = []
                    rank_buckets[rank].append(i) # Append relative index 0-3
                
                # Sort buckets by rank (ascending = better first)
                sorted_ranks = sorted(rank_buckets.keys())
                
                # List of buckets in order: e.g., [[0, 1], [2, 3]] for A=B>C=D
                ordered_buckets = [rank_buckets[r] for r in sorted_ranks]
                
                # Cartesian product to get all combinations
                # e.g., A=B>C=D -> [0,1] x [2,3] -> (0,2), (0,3), (1,2), (1,3)
                # These are tuples of indices
                strict_orderings = list(itertools.product(*ordered_buckets))
                
                # Convert tuples to lists
                prompt_rankings.append([list(ordering) for ordering in strict_orderings])
            
            if len(all_input_ids) == 0:
                return {
                    'input_ids': torch.empty(0, 0, dtype=torch.long),
                    'attention_mask': torch.empty(0, 0, dtype=torch.long),
                    'prompt_boundaries': [],
                    'prompt_rankings': []
                }
            
            # Pad all sequences
            input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt_boundaries': prompt_boundaries,
                'prompt_rankings': prompt_rankings
            }
    
    # Initialize collator
    collate_fn = PLCollator(tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=None,
        load_best_model_at_end=False,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        report_to=config.training.report_to,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=True,
        max_grad_norm=config.training.max_grad_norm,
        optim='adamw_torch_fused',
        seed=args.seed,
    )
    
    # Implement a Trainer subclass that computes Standard Plackett-Luce loss
    class PlackettLuceTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Compute Standard Plackett-Luce log-likelihood loss.
            
            Key design:
            - Compute scores for all K responses in one pass (O(K))
            - Iterate over all expanded strict orderings per prompt
            - Compute PL loss for each strict ordering
            - Average losses over orderings per prompt, then average over batch
            
            For strict ranking σ = (σ[OK][OK]σ[OK][OK]... [OK]σ[OK]:
            P(σ | θ) = ∏ᵢ₌₁ᵏ⁻¹ [exp(θ_σ[OK] / Σⱼ₌ᵢᵏ exp(θ_σ[OK]]
            
            Log-likelihood:
            log P(σ | θ) = Σᵢ₌₁ᵏ⁻[OK][θ_σ[OK]- logsumexp(θ_σ[OK] ..., θ_σ[OK]]
            """
            prompt_boundaries = inputs['prompt_boundaries']
            prompt_rankings = inputs['prompt_rankings']
            model_inputs = {k: v for k, v in inputs.items() if k not in ['prompt_boundaries', 'prompt_rankings']}

            if len(prompt_boundaries) == 0:
                return torch.tensor(0.0, requires_grad=True, device=model.device)

            # Single forward pass for all responses
            outputs = model(**model_inputs, return_dict=True)
            all_rewards = outputs.logits.view(-1)  # [total_responses]
            
            total_loss = 0.0
            num_prompts = 0
            
            for (start_idx, end_idx), rankings in zip(prompt_boundaries, prompt_rankings):
                # Extract scores for this prompt's K responses
                prompt_rewards = all_rewards[start_idx:end_idx]  # [K]
                
                # Clamp rewards
                prompt_rewards = torch.clamp(prompt_rewards, -20, 20)
                
                if not rankings:
                    continue
                
                prompt_loss = 0.0
                num_rankings = 0
                
                for ranking_indices in rankings:
                    # ranking_indices is a list of relative indices (e.g. [0, 2, 3])
                    # corresponding to a strict ordering
                    k = len(ranking_indices)
                    if k < 2:
                        continue # Cannot compute PL loss for < 2 items
                    
                    # Extract rewards for this specific ranking
                    # Use indices to pick from prompt_rewards
                    ranking_rewards = prompt_rewards[ranking_indices]
                    
                    # Efficient O(k) Plackett-Luce via cumulative sum trick
                    exp_rewards = torch.exp(ranking_rewards)
                    
                    # Compute cumulative sum from right to left
                    cum_sum = torch.zeros(k, device=ranking_rewards.device)
                    cum_sum[-1] = exp_rewards[-1]
                    for i in range(k - 2, -1, -1):
                        cum_sum[i] = exp_rewards[i] + cum_sum[i + 1]
                    
                    # Compute Log-likelihood
                    log_likelihood = torch.sum(ranking_rewards[:-1] - torch.log(cum_sum[:-1]))
                    
                    prompt_loss += (-log_likelihood)
                    num_rankings += 1
                
                if num_rankings > 0:
                    # Average over the multiple valid rankings for this prompt
                    total_loss += (prompt_loss / num_rankings)
                    num_prompts += 1
            
            # Average over prompts
            if num_prompts > 0:
                loss = total_loss / num_prompts
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=model.device)

            if return_outputs:
                return loss, outputs
            return loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Override prediction_step to handle custom loss computation."""
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss.detach(), None, None)
            else:
                # Filter inputs for model calls in other prediction modes
                # remove custom keys
                model_inputs = {k: v for k, v in inputs.items() if k not in ['prompt_boundaries', 'prompt_rankings']}
                return super().prediction_step(model, model_inputs, prediction_loss_only, ignore_keys)

    trainer = PlackettLuceTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        callbacks=[GradientNormCallback(window_size=150)]
    )

    print('Starting training via Trainer.train()')
    logger.info('Tracking gradient norm and rolling variance (window=150 steps) for optimization stability analysis.')

    # Track training time
    import time
    training_start_time = time.time()

    train_result = trainer.train()

    # Calculate training time
    training_elapsed_time = time.time() - training_start_time
    hours = int(training_elapsed_time // 3600)
    minutes = int((training_elapsed_time % 3600) // 60)
    seconds = int(training_elapsed_time % 60)

    print(f'Training finished successfully.')
    print(f'Training time: {hours}h {minutes}m {seconds}s ({training_elapsed_time:.2f} seconds)')
    print(f'Train results: {train_result}')
    print(f'All checkpoints saved to {output_dir}')
    logger.info(f'Training complete. Final checkpoint saved.')


if __name__ == '__main__':
    main()
