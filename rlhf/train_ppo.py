"""PPO-based RLHF trainer using Hugging Face TRL 0.27.

This script:
- Loads a base LLM policy (e.g., TinyLlama-1.1B or Alpaca-7B)
- Loads a trained reward model from checkpoint
- Trains the policy using PPO (Proximal Policy Optimization)
- Logs KL divergence and reward metrics during training
- Saves checkpoints for evaluation

Memory Optimization Strategy:
- LoRA: Train only ~0.2% of parameters
- Gradient checkpointing: Trade compute for memory on activations
- bf16: Use bfloat16 precision for efficiency

TRL 0.27 experimental PPOTrainer architecture:
- model: policy model (REQUIRED)
- ref_model: frozen copy for KL divergence (REQUIRED)
- reward_model: sequence classifier for reward scoring (REQUIRED)
- value_model: model with .score() method for advantage estimation (REQUIRED)
- train_dataset: dataset with 'input_ids' column (optional)
- processing_class: tokenizer (optional)
"""

import argparse
import yaml
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# TRL imports for 0.27 experimental
from trl.experimental.ppo import PPOConfig, PPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Small model for test mode
TEST_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


class ValueModel(nn.Module):
    """Value model for PPO that predicts state values.
    
    TRL 0.27's PolicyAndValueWrapper expects:
    - base_model_prefix: attribute to get the backbone
    - score(hidden_states): method that returns value predictions
    
    This wraps a causal LM and adds a value head.
    """
    
    def __init__(self, base_model, dtype=None):
        super().__init__()
        self.base_model = base_model
        self.base_model_prefix = base_model.base_model_prefix
        
        # Expose the backbone for PolicyAndValueWrapper.critic_backbone
        backbone = getattr(base_model, self.base_model_prefix)
        setattr(self, self.base_model_prefix, backbone)
        
        # Value head - match dtype of base model
        hidden_size = base_model.config.hidden_size
        self.score = nn.Linear(hidden_size, 1, bias=False)
        
        # Convert score layer to match base model dtype
        if dtype is not None:
            self.score = self.score.to(dtype)
        
        # Required for PolicyAndValueWrapper
        self.is_gradient_checkpointing = getattr(base_model, 'is_gradient_checkpointing', False)
        self.config = base_model.config
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.base_model.gradient_checkpointing_enable()
        self.is_gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False
        
    def forward(self, **kwargs):
        """Forward pass - returns base model outputs."""
        return self.base_model(**kwargs)


class RewardModelWrapper(nn.Module):
    """Wrapper for encoder-based reward models (like DeBERTa) to work with TRL PPO.
    
    TRL's get_reward expects a causal LM-like interface, but DeBERTa is an encoder.
    This wrapper makes DeBERTa compatible by filtering out unsupported arguments.
    """
    
    class BackboneWrapper(nn.Module):
        """Wrapper for the backbone that filters out causal LM kwargs."""
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # Filter out causal LM arguments that encoders don't support
            kwargs.pop('use_cache', None)
            kwargs.pop('past_key_values', None)
            kwargs.pop('position_ids', None)
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def __init__(self, reward_model):
        super().__init__()
        self.reward_model = reward_model
        self.config = reward_model.config
        
        # Get the backbone model name (e.g., 'deberta' for DebertaV2)
        if hasattr(reward_model, 'base_model_prefix'):
            self.base_model_prefix = reward_model.base_model_prefix
        else:
            # Fallback for DebertaV2ForSequenceClassification
            self.base_model_prefix = 'deberta'
        
        # Wrap and expose the backbone
        if hasattr(reward_model, self.base_model_prefix):
            backbone = getattr(reward_model, self.base_model_prefix)
            wrapped_backbone = self.BackboneWrapper(backbone)
            setattr(self, self.base_model_prefix, wrapped_backbone)
        
        # Expose the score head (classification layer)
        if hasattr(reward_model, 'score'):
            self.score = reward_model.score
        elif hasattr(reward_model, 'classifier'):
            self.score = reward_model.classifier
    
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
    
    TRL 0.27's PPOTrainer.train() expects pre-tokenized data with 'input_ids'.
    The train loop does: queries = data["input_ids"].to(device)
    So we must tokenize the prompts beforehand.
    
    Uses rlhf/download_sft.py for loading and formatting.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from download_sft import load_sft_dataset_tokenized
    
    dataset_config = config['dataset']
    num_samples = dataset_config.get('num_samples', dataset_config.get('num_train_prompts', 0))
    max_prompt_length = config['training'].get('max_prompt_length', 256)
    
    logger.info(f"Loading dataset {dataset_config['dataset_name']}")
    
    dataset = load_sft_dataset_tokenized(
        tokenizer=tokenizer,
        dataset_name=dataset_config['dataset_name'],
        split=dataset_config['dataset_split'],
        num_samples=num_samples if num_samples > 0 else None,
        max_length=max_prompt_length,
    )
    
    logger.info(f"Dataset prepared with 'input_ids' for PPOTrainer")
    
    # Set format to PyTorch tensors
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Train policy using PPO with TRL 0.27')
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
        print("TEST MODE ENABLED - Using reduced parameters for PPO")
        print("=" * 60)
        # Use small model that fits on 8GB GPU
        config['model']['base_model_name'] = args.test_model
        config['training']['total_episodes'] = 80  # 10 batches * 8 samples = 80 episodes
        config['training']['per_device_train_batch_size'] = 1
        config['training']['gradient_accumulation_steps'] = 1
        config['training']['logging_steps'] = 1
        config['training']['save_steps'] = 80
        config['training']['ref_quantization'] = 'none'  # Small model doesn't need quantization
        config['dataset']['max_samples'] = 100  # Small dataset
        print(f"  Model: {args.test_model} (small model for testing)")
        print(f"  Total episodes: 80")
        print(f"  Batch size: 8")
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
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float32
    
    # Memory optimization settings
    use_lora = config['model'].get('use_lora', True)
    ref_quantization = config['training'].get('ref_quantization', '4bit')  # '4bit', '8bit', 'none', 'cpu'
    gradient_checkpointing = config['training'].get('gradient_checkpointing', True)
    
    # ========== LOAD TOKENIZER ==========
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # ========== LOAD POLICY MODEL ==========
    logger.info(f"Loading policy model: {config['model']['base_model_name']}")
    
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
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model_name'],
        torch_dtype=dtype,
        device_map='auto',
        attn_implementation=attn_implementation,
    )
    policy_model.config.return_dict = True
    
    # Enable gradient checkpointing (reduces activation memory at compute cost)
    if gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
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
        policy_model = get_peft_model(policy_model, lora_config)
        trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy_model.parameters())
        logger.info(f"✓ LoRA applied: {trainable_params:,} trainable / {total_params:,} total params ({100*trainable_params/total_params:.2f}%)")
    
    # ========== LOAD REFERENCE MODEL (QUANTIZED) ==========
    # Reference is frozen - use quantization to save memory
    logger.info(f"Loading reference model (quantization: {ref_quantization})...")
    
    if ref_quantization == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model_name'],
            quantization_config=bnb_config,
            device_map='auto',
        )
        logger.info("✓ Reference model loaded in 4-bit (saves ~10GB)")
    elif ref_quantization == '8bit':
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model_name'],
            load_in_8bit=True,
            device_map='auto',
        )
        logger.info("✓ Reference model loaded in 8-bit (saves ~7GB)")
    elif ref_quantization == 'cpu':
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model_name'],
            torch_dtype=dtype,
            device_map='cpu',
        )
        logger.info("✓ Reference model offloaded to CPU (saves ~14GB, slower)")
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model']['base_model_name'],
            torch_dtype=dtype,
            device_map='auto',
        )
    
    ref_model.config.return_dict = True
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # ========== CREATE VALUE MODEL ==========
    # Value head attached to policy backbone
    # This is the memory-efficient approach for PPO
    logger.info("Creating value head on policy backbone...")
    
    # Get the base model from PEFT wrapper if LoRA is used
    if use_lora:
        base_for_value = policy_model.get_base_model()
    else:
        base_for_value = policy_model
    
    # Value model shares the policy backbone, only adds a small value head
    value_model = ValueModel(base_for_value, dtype=dtype)
    logger.info(f"✓ Value head created on shared backbone (hidden_size={base_for_value.config.hidden_size})")
    logger.info("✓ No separate 7B value model loaded - using shared backbone")
    
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
    
    # Wrap reward model for TRL compatibility (filters unsupported kwargs)
    reward_model = RewardModelWrapper(reward_model_base)
    
    logger.info(f"✓ Reward model loaded successfully from {reward_model_path}")
    logger.info(f"✓ Reward model type: {type(reward_model_base).__name__} (wrapped)")
    logger.info("=" * 60)
    
    # ========== PREPARE DATASET ==========
    dataset = prepare_dataset(config, tokenizer)
    logger.info(f"Dataset prepared with {len(dataset)} samples")
    
    # Split dataset into train and eval for PPOTrainer
    # TRL's generate_completions requires eval_dataset
    eval_size = min(100, len(dataset) // 10)  # 10% or max 100 samples for eval
    train_dataset = dataset.select(range(eval_size, len(dataset)))
    eval_dataset = dataset.select(range(eval_size))
    logger.info(f"Train dataset: {len(train_dataset)} samples, Eval dataset: {len(eval_dataset)} samples")
    
    # ========== CREATE PPO CONFIG ==========
    training = config['training']
    
    ppo_config = PPOConfig(
        output_dir=str(output_dir),
        learning_rate=training.get('learning_rate', 1e-6),
        batch_size=training.get('batch_size', 32),
        mini_batch_size=training.get('mini_batch_size', 8),
        gradient_accumulation_steps=training.get('gradient_accumulation_steps', 2),
        total_episodes=training.get('total_episodes', 20000),
        num_ppo_epochs=training.get('ppo_epochs', 2),
        kl_coef=training.get('init_kl_coef', 0.02),
        cliprange=training.get('cliprange', 0.2),
        cliprange_value=training.get('cliprange_value', 0.2),
        vf_coef=training.get('vf_coef', 0.1),
        gamma=training.get('gamma', 1.0),
        lam=training.get('lam', 0.95),
        response_length=training.get('max_new_tokens', 256),
        temperature=training.get('temperature', 0.7),
        logging_steps=training.get('logging_steps', 10),
        save_steps=training.get('save_freq', 500),
        bf16=training.get('bf16', True),
        seed=seed,
        report_to=['tensorboard'],
        num_sample_generations=0,  # Suppress verbose rich table output during training
    )
    
    logger.info(f"PPOConfig created (experimental PPO)")
    logger.info(f"  - batch_size: {ppo_config.batch_size}")
    logger.info(f"  - total_episodes: {ppo_config.total_episodes}")
    logger.info(f"  - num_ppo_epochs: {ppo_config.num_ppo_epochs}")
    logger.info(f"  - gradient_accumulation_steps: {ppo_config.gradient_accumulation_steps}")
    
    # Calculate actual training budget
    total_prompts = ppo_config.batch_size * ppo_config.total_episodes
    total_completions = total_prompts * 1  # PPO generates 1 completion per prompt
    total_optimizer_steps = (ppo_config.total_episodes // ppo_config.gradient_accumulation_steps) * ppo_config.num_ppo_epochs
    logger.info(f"")
    logger.info(f"  BUDGET (PPO vs GRPO):")
    logger.info(f"  ├─ Prompts generated: {total_prompts:,}")
    logger.info(f"  ├─ Completions: {total_completions:,}")
    logger.info(f"  ├─ Optimizer reuse: num_ppo_epochs={ppo_config.num_ppo_epochs}")
    logger.info(f"  ├─ Total optimizer steps: {total_optimizer_steps:,}")
    logger.info(f"  └─ Reward-scored tokens: ~{total_completions * 256:,}")
    logger.info(f"")
    
    # ========== CREATE PPO TRAINER ==========
    # TRL 0.27 experimental PPOTrainer:
    # - model: raw AutoModelForCausalLM (policy)
    # - ref_model: frozen copy of policy for KL
    # - value_model: model with .score() method for value prediction
    # - reward_model: our trained DeBERTa
    logger.info("Initializing PPO trainer...")
    
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        value_model=value_model,
    )
    
    # ========== MONKEY-PATCH PolicyAndValueWrapper ==========
    # TRL's PolicyAndValueWrapper doesn't have gradient_checkpointing_disable/enable methods
    # but unwrap_model_for_generation calls them. We need to add these methods.
    def _gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on both policy and value model."""
        if hasattr(self.policy, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs:
                self.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.policy.gradient_checkpointing_enable()
        if hasattr(self.value_model, 'gradient_checkpointing_enable'):
            if gradient_checkpointing_kwargs:
                self.value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.value_model.gradient_checkpointing_enable()
        self.is_gradient_checkpointing = True
    
    def _gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on both policy and value model."""
        if hasattr(self.policy, 'gradient_checkpointing_disable'):
            self.policy.gradient_checkpointing_disable()
        if hasattr(self.value_model, 'gradient_checkpointing_disable'):
            self.value_model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False
    
    # Patch the model (which is a PolicyAndValueWrapper after accelerate.prepare)
    import types
    if hasattr(ppo_trainer, 'model'):
        model_to_patch = ppo_trainer.model
        # Handle accelerate wrapped models
        if hasattr(model_to_patch, 'module'):
            model_to_patch = model_to_patch.module
        if not hasattr(model_to_patch, 'gradient_checkpointing_disable'):
            model_to_patch.gradient_checkpointing_disable = types.MethodType(_gradient_checkpointing_disable, model_to_patch)
            model_to_patch.gradient_checkpointing_enable = types.MethodType(_gradient_checkpointing_enable, model_to_patch)
            logger.info("✓ Patched PolicyAndValueWrapper with gradient checkpointing methods")
    
    logger.info("✓ PPO trainer initialized with reward model from checkpoint")
    
    # ========== TRAIN ==========
    logger.info("Starting PPO training...")
    ppo_trainer.train()
    
    # ========== SAVE FINAL MODEL ==========
    final_path = output_dir / 'final_model'
    ppo_trainer.save_model(str(final_path))
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == '__main__':
    main()
