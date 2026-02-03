"""Cox-Breslow trainer.

This script:
- Loads UltraFeedback dataset
- Keeps ties and models them using Breslow's approximation to Cox partial likelihood
- Trains with 3 seeds
- Uses YAML configuration files for hyperparameter management

Cox-Breslow Model:
- For ranking π = (B[OK] B[OK] ..., B[OK] where B[OK]are tie buckets
- Breslow partial likelihood:
  L = ∏ₜ₌₁[OK][∏ᵢ∈B[OK]exp(θ[OK]] / [Σⱼ∈R[OK]exp(θ[OK]]^|Bₜ|
  
- Log-likelihood (per ranking):
  [OK]= Σₜ₌₁ᵐ [Σᵢ∈B[OK]θ[OK]- |Bₜ| × log(Σⱼ∈R[OK]exp(θ[OK])]
  
- Loss: -[OK](negative log-likelihood)

"""

import argparse
import sys
import os
import math
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import torch
import numpy as np
import random
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

    # Collate function: keep ties and structure them as buckets
    class CoxBreslowCollator:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, batch):
            """Keep ties and structure rankings as ordered buckets.
            
            For each prompt with ranks [rank_A, rank_B, rank_C, rank_D]:
            - Group responses by rank (tied buckets)
            - Create ordered sequence of buckets B[OK][OK]B[OK][OK]... [OK]B[OK]
            """
            from torch.nn.utils.rnn import pad_sequence
            
            # Store rankings for each prompt
            rankings_list = []  # List of bucket structures
            input_ids_list = []
            attention_mask_list = []
            
            for example in batch:
                ranks = {
                    'A': example['rank_A'],
                    'B': example['rank_B'],
                    'C': example['rank_C'],
                    'D': example['rank_D']
                }
                labels = ['A', 'B', 'C', 'D']
                
                # Group by rank (create buckets)
                rank_buckets = {}
                for label in labels:
                    rank = ranks[label]
                    if rank not in rank_buckets:
                        rank_buckets[rank] = []
                    rank_buckets[rank].append(label)
                
                # Sort buckets by rank (ascending = better first)
                sorted_ranks = sorted(rank_buckets.keys())
                
                # Create ordered buckets
                buckets = [rank_buckets[rank] for rank in sorted_ranks]
                
                # Store the bucket structure
                rankings_list.append(buckets)
                
                # Collect input_ids and attention_masks for all labels
                for label in labels:
                    input_ids_tensor = torch.tensor(example[f'input_ids_{label}'], dtype=torch.long)
                    attention_mask_tensor = torch.tensor(example[f'attention_mask_{label}'], dtype=torch.long)
                    
                    # Truncate to max_length if necessary
                    if input_ids_tensor.size(0) > self.max_length:
                        input_ids_tensor = input_ids_tensor[:self.max_length]
                        attention_mask_tensor = attention_mask_tensor[:self.max_length]
                    
                    input_ids_list.append(input_ids_tensor)
                    attention_mask_list.append(attention_mask_tensor)
            
            if len(input_ids_list) == 0:
                return {
                    'input_ids': torch.empty(0, 0, dtype=torch.long),
                    'attention_mask': torch.empty(0, 0, dtype=torch.long),
                    'rankings': [],
                    'batch_size': 0
                }
            
            # Pad all sequences
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'rankings': rankings_list,  # List of bucket structures
                'batch_size': len(batch)
            }
    
    # Initialize collator
    collate_fn = CoxBreslowCollator(tokenizer, max_length)

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
    
    # Implement a Trainer subclass that computes Cox-Breslow loss
    class CoxBreslowTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Compute Cox-Breslow log-likelihood loss (Breslow's approximation to Cox model).
            
            For ranking π = (B[OK] B[OK] ..., B[OK] with tie buckets:
            Breslow partial likelihood:
            L = ∏ₜ₌₁[OK][∏ᵢ∈B[OK]exp(θ[OK]] / [Σⱼ∈R[OK]exp(θ[OK]]^|Bₜ|
            
            Log-likelihood (per ranking):
            [OK]= Σₜ₌₁ᵐ [Σᵢ∈B[OK]θ[OK]- |Bₜ| × log(Σⱼ∈R[OK]exp(θ[OK])]
            
            We minimize negative log-likelihood: -[OK]
            """
            batch_size = inputs['batch_size']
            rankings = inputs['rankings']
            model_inputs = {k: v for k, v in inputs.items() if k not in ['batch_size', 'rankings']}

            # Handle empty batch
            if batch_size == 0:
                return torch.tensor(0.0, requires_grad=True, device=model.device)

            # Single forward pass for all responses (4 per prompt)
            outputs = model(**model_inputs, return_dict=True)
            all_rewards = outputs.logits.view(-1)  # [batch_size * 4]
            
            # Clamp rewards to prevent overflow
            all_rewards = torch.clamp(all_rewards, -20, 20)
            
            # Process each prompt
            total_loss = 0.0
            
            for prompt_idx, buckets in enumerate(rankings):
                # Get rewards for this prompt's 4 responses (A, B, C, D)
                prompt_rewards = all_rewards[prompt_idx * 4:(prompt_idx + 1) * 4]
                
                # Create label-to-reward mapping
                labels = ['A', 'B', 'C', 'D']
                label_to_idx = {label: idx for idx, label in enumerate(labels)}
                
                # Efficient O(k) Cox-Breslow via cumulative sum trick
                # Precompute exp(rewards) once for all items
                exp_rewards = torch.exp(prompt_rewards)
                
                # Initialize cumulative sum with all items
                # S_t = sum of exp(rewards) for remaining items at stage t
                cumulative_sum = torch.sum(exp_rewards)
                
                log_likelihood = 0.0
                
                for bucket in buckets:
                    # Numerator: Σᵢ∈B[OK]θ[OK](sum of rewards in bucket)
                    bucket_reward_sum = sum(prompt_rewards[label_to_idx[label]] for label in bucket)
                    
                    # Denominator: log(S_t) where S_t = Σⱼ∈R[OK]exp(θ[OK]
                    log_cumulative_sum = torch.log(cumulative_sum)
                    
                    # Bucket size
                    bucket_size = len(bucket)
                    
                    # Stage contribution: Σᵢ∈B[OK]θ[OK]- |Bₜ| × log(S_t)
                    stage_log_likelihood = bucket_reward_sum - bucket_size * log_cumulative_sum
                    
                    log_likelihood += stage_log_likelihood
                    
                    # Update cumulative sum: S_{t+1} = S_t - Σᵢ∈B[OK]exp(θ[OK]
                    for label in bucket:
                        cumulative_sum = cumulative_sum - exp_rewards[label_to_idx[label]]
                
                total_loss += (-log_likelihood)
            
            # Average over prompts
            loss = total_loss / batch_size

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

    trainer = CoxBreslowTrainer(
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
