"""GRPO-based RLHF trainer using Hugging Face TRL 0.27.

This script:
- Loads a base LLM policy (e.g., TinyLlama-1.1B or Alpaca-7B)
- Loads a trained reward model from checkpoint
- Trains the policy using GRPO (Group Relative Policy Optimization)
- Logs KL divergence and reward metrics during training
- Saves checkpoints for evaluation

Memory Optimization Strategy:
- LoRA: Train only ~0.2% of parameters
- Gradient checkpointing: Trade compute for memory on activations
- bf16: Use bfloat16 precision for efficiency

TRL 0.27 GRPOTrainer architecture:
- model: policy model (REQUIRED)
- reward_funcs: reward function or model (REQUIRED)
- args: GRPOConfig (optional)
- train_dataset: dataset with 'prompt' column (optional)
- processing_class: tokenizer (optional)
"""

import argparse
import yaml
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Any, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model

# TRL imports for 0.27
from trl import GRPOConfig, GRPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Small model for test mode
TEST_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


class RewardModelWrapper(nn.Module):
    """Wrapper for encoder-based reward models (like DeBERTa) to work with TRL.
    
    This wrapper makes DeBERTa compatible by filtering out unsupported arguments
    that are meant for causal LMs (use_cache, past_key_values, etc.).
    """
    
    def __init__(self, reward_model):
        super().__init__()
        self.reward_model = reward_model
        self.config = reward_model.config
        
        # Get the backbone model name (e.g., 'deberta' for DebertaV2)
        if hasattr(reward_model, 'base_model_prefix'):
            self.base_model_prefix = reward_model.base_model_prefix
        else:
            self.base_model_prefix = 'deberta'
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass - filter out unsupported kwargs for encoder models."""
        # Encoder models don't support these causal LM arguments
        kwargs.pop('use_cache', None)
        kwargs.pop('past_key_values', None)
        kwargs.pop('position_ids', None)
        
        return self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """Prepare the dataset for RLHF training.
    
    Uses rlhf/download_sft.py for loading and formatting.
    GRPO needs a 'prompt' column (no pre-tokenization required).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from download_sft import load_sft_dataset
    
    dataset_config = config['dataset']
    num_samples = dataset_config.get('num_train_prompts', dataset_config.get('num_samples', 0))
    
    logger.info(f"Loading dataset {dataset_config['dataset_name']}")
    
    dataset = load_sft_dataset(
        dataset_name=dataset_config['dataset_name'],
        split=dataset_config['dataset_split'],
        num_samples=num_samples if num_samples > 0 else None,
    )
    
    logger.info(f"Dataset prepared with 'prompt' field for GRPOTrainer")
    
    return dataset


def create_reward_function(reward_model, reward_tokenizer, device='cuda', max_length=512):
    """Create a reward function for the GRPO trainer.
    
    TRL 0.27 GRPOTrainer passes additional kwargs like completion_ids,
    so we accept **kwargs to handle them gracefully.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for prompt-completion pairs."""
        # TRL may pass extra args like completion_ids - we ignore them
        # Concatenate prompts and completions
        texts = [p + c for p, c in zip(prompts, completions)]
        
        # Tokenize
        inputs = reward_tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)
        
        # Get reward scores
        with torch.no_grad():
            outputs = reward_model(**inputs)
            rewards = outputs.logits.squeeze(-1)
        
        # Handle single sample case
        if rewards.dim() == 0:
            return [rewards.cpu().item()]
        
        return rewards.cpu().tolist()
    
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description='Train policy using GRPO with TRL 0.27')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--reward_model_path', type=str, default=None, help='Override reward model path')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    # Test mode arguments
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode with reduced parameters')
    parser.add_argument('--test_steps', type=int, default=10, help='Max training steps for test mode')
    parser.add_argument('--test_model', type=str, default=TEST_MODEL, help='Small model to use in test mode')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args if provided
    if args.reward_model_path:
        config['model']['reward_model_path'] = args.reward_model_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # Test mode overrides
    if args.test_mode:
        print("=" * 60)
        print("TEST MODE ENABLED - Using reduced parameters for GRPO")
        print("=" * 60)
        # Use small model that fits on 8GB GPU
        config['model']['base_model_name'] = args.test_model
        config['training']['max_steps'] = 10  # Limit training steps
        config['training']['per_device_train_batch_size'] = 1
        config['training']['gradient_accumulation_steps'] = 1
        config['training']['logging_steps'] = 1
        config['training']['save_steps'] = 10
        config['dataset']['max_samples'] = 100  # Small dataset
        print(f"  Model: {args.test_model} (small model for testing)")
        print(f"  Max steps: 10")
        print(f"  Batch size: 1")
        print(f"  Max samples: 100")
        print("=" * 60)
    
    # Override with command line args if provided
    if args.reward_model_path:
        config['model']['reward_model_path'] = args.reward_model_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # Create output directories
    output_dir = Path(config['training']['output_dir'])
    log_dir = Path(config['training']['log_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Training with seed {seed}")
    logger.info(f"Output directory: {output_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if config['training'].get('bf16', True) else torch.float32
    
    # Memory optimization settings
    use_lora = config['model'].get('use_lora', True)
    gradient_checkpointing = config['training'].get('gradient_checkpointing', True)
    
    # ========== LOAD TOKENIZER ==========
    logger.info(f"Loading tokenizer: {config['model']['base_model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # ========== PREPARE DATASET ==========
    dataset = prepare_dataset(config, tokenizer)
    logger.info(f"Dataset prepared with {len(dataset)} samples")
    
    # Split into train and eval
    eval_size = min(100, len(dataset) // 10)  # 10% or max 100 samples for eval
    train_dataset = dataset.select(range(eval_size, len(dataset)))
    eval_dataset = dataset.select(range(eval_size))
    logger.info(f"Train dataset: {len(train_dataset)} samples, Eval dataset: {len(eval_dataset)} samples")
    
    # ========== LOAD REWARD MODEL ==========
    reward_model_path = config['model']['reward_model_path']
    logger.info("=" * 60)
    logger.info("LOADING REWARD MODEL")
    logger.info(f"Loading reward model from: {reward_model_path}")
    
    reward_model_base = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        torch_dtype=dtype,
    )
    reward_model_base.to(device)
    reward_model_base.eval()
    
    # Wrap reward model for compatibility (filters unsupported kwargs)
    reward_model = RewardModelWrapper(reward_model_base)
    
    # Load reward tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    logger.info(f"✓ Reward model loaded successfully from {reward_model_path}")
    logger.info(f"✓ Reward model type: {type(reward_model_base).__name__} (wrapped)")
    logger.info("=" * 60)
    
    # Create reward function
    max_length = config['model'].get('max_length', 512)
    reward_fn = create_reward_function(reward_model, reward_tokenizer, device, max_length)
    logger.info(f"✓ Reward function created using loaded reward model")
    
    # ========== CREATE GRPO CONFIG ==========
    training_config = config['training']
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config.get('prompts_per_update', 4),
        num_generations=training_config.get('group_size', 2),
        steps_per_generation=training_config.get('steps_per_generation', 1),
        num_iterations=training_config.get('num_iterations', 2),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        max_steps=training_config.get('num_updates', 1500),
        learning_rate=training_config.get('learning_rate', 1e-6),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        beta=training_config.get('beta', 0.1),
        max_completion_length=training_config.get('max_new_tokens', 256),
        temperature=training_config.get('temperature', 0.7),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        bf16=training_config.get('bf16', True),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_freq', 500),
        seed=seed,
        report_to=['tensorboard'],
        # Eval configuration
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=training_config.get('save_freq', 500),
    )
    
    logger.info(f"GRPOConfig created")
    logger.info(f"  - per_device_train_batch_size: {grpo_config.per_device_train_batch_size}")
    logger.info(f"  - steps_per_generation: {grpo_config.steps_per_generation}")
    logger.info(f"  - num_generations (group_size): {grpo_config.num_generations}")
    logger.info(f"  - num_iterations (≈PPO epochs): {grpo_config.num_iterations}")
    logger.info(f"  - gradient_accumulation_steps: {grpo_config.gradient_accumulation_steps}")
    logger.info(f"  - max_steps: {grpo_config.max_steps}")
    logger.info(f"  - learning_rate: {grpo_config.learning_rate}")
    logger.info(f"  - beta (KL coef): {grpo_config.beta}")
    
    # Calculate actual generation budget and compare to PPO
    prompts_per_step = grpo_config.per_device_train_batch_size * grpo_config.steps_per_generation
    completions_per_step = prompts_per_step * grpo_config.num_generations
    total_prompts = prompts_per_step * grpo_config.max_steps
    total_completions = completions_per_step * grpo_config.max_steps
    total_optimizer_steps = grpo_config.max_steps * grpo_config.num_iterations
    logger.info(f"")
    logger.info(f"  BUDGET COMPARISON (GRPO vs PPO):")
    logger.info(f"  ├─ Prompts generated: {total_prompts:,} ")
    logger.info(f"  ├─ Completions: {total_completions:,} ")
    logger.info(f"  ├─ Optimizer reuse: num_iterations={grpo_config.num_iterations}")
    logger.info(f"  ├─ Total optimizer steps: {total_optimizer_steps:,}")
    logger.info(f"  └─ Reward-scored tokens: ~{total_completions * 256:,}")
    logger.info(f"")
    
    # ========== LOAD BASE POLICY MODEL ==========
    logger.info(f"Loading base policy model: {config['model']['base_model_name']}")
    
    # Check if flash attention is available
    use_flash_attn = config['training'].get('use_flash_attention', False)
    attn_implementation = None
    if use_flash_attn:
        try:
            import flash_attn
            attn_implementation = 'flash_attention_2'
            logger.info("✓ Flash Attention 2 is available and will be used")
        except ImportError:
            logger.warning("Flash Attention requested but not installed. Using default attention.")
            attn_implementation = None
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model_name'],
        torch_dtype=dtype,
        device_map='auto',
        attn_implementation=attn_implementation,
    )
    model.config.return_dict = True
    
    # Enable gradient checkpointing (reduces activation memory at compute cost)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")
    
    # ========== APPLY LoRA TO POLICY ==========
    if use_lora:
        lora_config = LoraConfig(
            r=config['model'].get('lora_r', 8),
            lora_alpha=config['model'].get('lora_alpha', 16),
            lora_dropout=config['model'].get('lora_dropout', 0.05),
            target_modules=config['model'].get('lora_target_modules', ['q_proj', 'v_proj']),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ LoRA applied: {trainable_params:,} trainable / {total_params:,} total params ({100*trainable_params/total_params:.2f}%)")
    
    # ========== CREATE GRPO TRAINER ==========
    logger.info("Initializing GRPO trainer...")
    
    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,  # Our custom reward function using loaded RM
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    logger.info("✓ GRPO trainer initialized with reward model from checkpoint")
    
    # ========== TRAIN ==========
    logger.info("Starting GRPO training...")
    logger.info(f"  - Total steps: {grpo_config.max_steps}")
    logger.info(f"  - Effective batch size: {grpo_config.per_device_train_batch_size * grpo_config.num_generations * grpo_config.gradient_accumulation_steps}")
    
    grpo_trainer.train()
    
    # ========== SAVE FINAL MODEL ==========
    final_path = output_dir / 'final_model'
    grpo_trainer.save_model(str(final_path))
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == '__main__':
    main()
