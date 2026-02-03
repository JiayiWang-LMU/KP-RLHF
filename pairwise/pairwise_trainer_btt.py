"""BT-Davidson trainer.

This script:
- Loads UltraFeedback dataset
- Decomposes prompts into pairwise comparisons (keeps ties)
- Trains with 3 seeds
- Uses YAML configuration files for hyperparameter management

Bradley-Terry-Davidson Model:
- For items i and j with reward scores r(x_i) and r(x_j):
  - P(i [OK]j) = exp(r_i) / (exp(r_i) + exp(r_j) + γ[OK]exp(r_i)·exp(r_j)))
  - P(i ~ j) = γ[OK]exp(r_i)·exp(r_j)) / (exp(r_i) + exp(r_j) + γ[OK]exp(r_i)·exp(r_j)))
- γ: tie parameter (γ = 1 as default)
- Loss = -Σ log P(observed outcome | θ, γ)
"""

import argparse
import sys
import os
import math
import random
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
    
    # BT-Davidson specific parameters
    gamma = config.bt_davidson.gamma

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

    print('Loading tokenized dataset...')
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

    # Collate function: prepare K responses per prompt for efficient scoring
    class BTTDavidsonCollator:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, batch):
            """Prepare batch for efficient k-wise BT-Davidson training.
            
            For each prompt with K=4 responses and ranks [rank_A, rank_B, rank_C, rank_D]:
            - Tokenize all K responses once
            - Store ranking information for later pairwise loss computation (including ties)
            - This enables O(K) forward passes instead of O(K²)
            """
            from torch.nn.utils.rnn import pad_sequence
        
            all_input_ids = []
            all_attention_mask = []
            prompt_boundaries = []  # Track which responses belong to which prompt
            prompt_ranks = []  # Store ranking info for each prompt
        
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
            
                prompt_boundaries.append((prompt_start, current_idx))  # [start, end)
                prompt_ranks.append([ranks[label] for label in labels])  # Store ranks in order
        
            if len(all_input_ids) == 0:
                return {
                    'input_ids': torch.empty(0, 0, dtype=torch.long),
                    'attention_mask': torch.empty(0, 0, dtype=torch.long),
                    'prompt_boundaries': [],
                    'prompt_ranks': []
                }
        
            # Pad all sequences together
            input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0)
        
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt_boundaries': prompt_boundaries,
                'prompt_ranks': prompt_ranks
            }

    # Initialize collator
    collate_fn = BTTDavidsonCollator(tokenizer, max_length)

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

    # Implement a Trainer subclass for BT-Davidson loss
    class BTDavidsonTrainer(Trainer):
        def __init__(self, *args, gamma=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.gamma = gamma
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Efficient k-wise BT-Davidson loss with InstructGPT-style normalization.
            
            Key design: Following InstructGPT (Ouyang et al., 2022), we normalize by the number
            of pairwise comparisons to ensure equal prompt-level contribution.
            
            For each prompt with K=4 responses:
            1. Compute K scores with single forward pass (O(K) not O(K²))
            2. Generate all C(K,2)=6 pairwise losses (including ties)
            3. Average over all 6 pairs to normalize prompt contribution
            
            Loss per prompt: L_i = -(1/6) * Σ log P(observed outcome)
            where P includes win/tie/lose probabilities modeled with γ parameter.
            
            Unlike standard BT, n_i = 6 is constant (ties are modeled, not removed).
            """
            prompt_boundaries = inputs['prompt_boundaries']
            prompt_ranks = inputs['prompt_ranks']
            model_inputs = {k: v for k, v in inputs.items() if k not in ['prompt_boundaries', 'prompt_ranks']}

            if len(prompt_boundaries) == 0:
                return torch.tensor(0.0, requires_grad=True, device=model.device)

            # Single forward pass for all responses across all prompts
            outputs = model(**model_inputs, return_dict=True)
            all_scores = outputs.logits.view(-1)  # [total_responses]
            
            # Compute loss for each prompt separately, then average
            prompt_losses = []
            
            for (start_idx, end_idx), ranks in zip(prompt_boundaries, prompt_ranks):
                # Extract scores for this prompt's K responses
                prompt_scores = all_scores[start_idx:end_idx]  # [K]
                K = len(prompt_scores)
                
                # Clamp scores to prevent overflow
                prompt_scores = torch.clamp(prompt_scores, -20, 20)
                
                # Generate all pairwise comparisons (KEEP ties)
                pairwise_losses = []
                for i in range(K):
                    for j in range(i + 1, K):
                        rank_i = ranks[i]
                        rank_j = ranks[j]
                        
                        # Determine relationship: tie or preference
                        is_tie = (rank_i == rank_j)
                        
                        # Determine which has better rank
                        if rank_i < rank_j:  # i is better (lower rank = better)
                            r_w = prompt_scores[i]
                            r_l = prompt_scores[j]
                        else:
                            r_w = prompt_scores[j]
                            r_l = prompt_scores[i]
                        
                        # BT-Davidson loss with γ
                        # P(i [OK]j) = exp(r_i) / (exp(r_i) + exp(r_j) + γ[OK]exp(r_i)·exp(r_j)))
                        # P(i ~ j) = γ[OK]exp(r_i)·exp(r_j)) / (exp(r_i) + exp(r_j) + γ[OK]exp(r_i)·exp(r_j)))
                        
                        # Geometric mean term with gamma: log(γ) + (r_w + r_l) / 2
                        r_mean = (r_w + r_l) / 2.0
                        log_gamma_geom = torch.log(torch.tensor(self.gamma, device=r_mean.device)) + r_mean
                        
                        # Denominator: log(exp(r_w) + exp(r_l) + γ·exp((r_w + r_l)/2))
                        log_denom = torch.logsumexp(
                            torch.stack([r_w, r_l, log_gamma_geom], dim=0),
                            dim=0
                        )
                        
                        if is_tie:
                            # For ties: log P(i ~ j) = log(γ) + (r_w + r_l)/2 - log_denom
                            log_prob = log_gamma_geom - log_denom
                        else:
                            # For preference: log P(i [OK]j) = r_w - log_denom
                            log_prob = r_w - log_denom
                        
                        # Negative log-likelihood
                        pairwise_losses.append(-log_prob)
                
                # Average pairwise losses to normalize by number of pairs
                # This ensures all prompts contribute equally regardless of ties
                if len(pairwise_losses) > 0:
                    prompt_loss = torch.stack(pairwise_losses).mean()
                    prompt_losses.append(prompt_loss)
            
            # Average loss across all prompts in batch
            if len(prompt_losses) == 0:
                loss = torch.tensor(0.0, requires_grad=True, device=model.device)
            else:
                loss = torch.stack(prompt_losses).mean()

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
                model_inputs = {k: v for k, v in inputs.items() if k not in ['rankings_list']}
                return super().prediction_step(model, model_inputs, prediction_loss_only, ignore_keys)

    trainer = BTDavidsonTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        gamma=gamma,
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
    print(f'All 12 checkpoints saved to {output_dir}')
    logger.info(f'Training complete. Final checkpoint saved at epoch 3.0.')


if __name__ == '__main__':
    main()
