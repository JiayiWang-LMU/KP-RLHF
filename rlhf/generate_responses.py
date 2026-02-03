"""
Generate responses from multiple fine-tuned models for head-to-head comparison.
Uses prompts from tatsu-lab/alpaca dataset (evaluation set, separate from training).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path for cache utilities
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dataset.cache_utils import get_dataset_cache_dir, get_model_cache_dir


def load_xalphacaeval_prompts(
    num_samples: int = 798,
    seed: int = 42,
    cache_dir: str = None
) -> List[Dict]:
    """
    Load prompts from viyer98/XL-AlpacaEval dataset (English split).
    
    Args:
        num_samples: Number of evaluation prompts to load (default: 798, all English prompts)
        seed: Random seed for shuffling
        cache_dir: Cache directory for dataset (uses default if None)
    
    Returns:
        List of prompt dictionaries
    """
    if cache_dir is None:
        cache_dir = get_dataset_cache_dir()
    
    print("Loading viyer98/XL-AlpacaEval dataset (English split) for evaluation...")
    print(f"Cache directory: {cache_dir}")
    
    # Load English split from XL-AlpacaEval
    # Structure: viyer98/XL-AlpacaEval has "default" config with "english" split
    dataset = load_dataset("viyer98/XL-AlpacaEval", name="default", split="english", cache_dir=cache_dir)
    
    print(f"Total prompts in dataset: {len(dataset)}")
    
    # Shuffle with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    
    # Use all prompts (typically 798 in English split)
    eval_size = min(num_samples, len(dataset))
    eval_dataset = dataset.select(range(eval_size))
    
    prompts = []
    for i, item in enumerate(eval_dataset):
        # XL-AlpacaEval uses 'instruction' column
        instruction = item.get('instruction', '').strip()
        
        if instruction:
            prompts.append({
                'prompt': instruction,
                'source': 'xl_alpacaeval',
                'prompt_id': i,
                'instruction': instruction,
                'input': ''
            })
    
    print(f"Loaded {len(prompts)} evaluation prompts from XL-AlpacaEval English split")
    
    return prompts


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True
) -> str:
    """Generate a response from the model using Alpaca format.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (default: 0.7 for stochastic)
        top_p: Top-p sampling (default: 0.95 for stochastic)
        do_sample: Whether to sample (default: True for stochastic)
    """
    # Use Alpaca prompt format
    formatted_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:"
    )
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    with torch.no_grad():
        # For deterministic decoding, use greedy (do_sample=False)
        # For stochastic, use temperature and top_p with do_sample=True
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        
        outputs = model.generate(**inputs, **gen_kwargs)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate responses from multiple models")
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to the fine-tuned models")
    parser.add_argument("--model_names", nargs="+", required=True,
                        help="Names for the models (e.g., BT BT-0.5 Cox)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for responses")
    parser.add_argument("--output_file", type=str, default="responses.json",
                        help="Output filename (default: responses.json)")
    parser.add_argument("--num_samples", type=int, default=798,
                        help="Number of evaluation prompts to use (default: 798 for all English XL-AlpacaEval)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling (default: 0.95)")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable stochastic sampling (default: deterministic/greedy)")
    parser.add_argument("--num_responses_per_prompt", type=int, default=1,
                        help="Number of responses to generate per prompt (default: 1, use 4 for RM evaluation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling and generation")
    args = parser.parse_args()
    
    # Set random seed for reproducibility in sampling
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    assert len(args.model_paths) == len(args.model_names), \
        "Number of model paths must match number of model names"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get cache directories
    dataset_cache = get_dataset_cache_dir()
    model_cache = get_model_cache_dir()
    print(f"Dataset cache: {dataset_cache}")
    print(f"Model cache: {model_cache}")
    print()
    
    # Load prompts from XL-AlpacaEval evaluation set (English)
    prompts = load_xalphacaeval_prompts(
        num_samples=args.num_samples,
        seed=args.seed,
        cache_dir=dataset_cache
    )
    
    # Store all responses
    all_responses = []
    
    # Get cache directory
    model_cache_dir = get_model_cache_dir()
    print(f"Model cache directory: {model_cache_dir}")
    print()
    
    # Load and generate responses from each model
    for model_path, model_name in zip(args.model_paths, args.model_names):
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} from {model_path}")
        print('='*60)
        
        # Load model and tokenizer with caching
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=model_cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=model_cache_dir
        )
        model.eval()
        
        print(f"Generating responses for {len(prompts)} prompts...")
        print(f"  Responses per prompt: {args.num_responses_per_prompt}")
        print(f"  Decoding: {'Stochastic (do_sample=True, temp={}, top_p={})'.format(args.temperature, args.top_p) if args.do_sample else 'Greedy (deterministic)'}")
        
        for i, prompt_data in enumerate(tqdm(prompts, desc=f"Generating ({model_name})")):
            prompt = prompt_data['prompt']
            
            # Generate N responses per prompt (default N=1, use N=4 for RM evaluation)
            if args.num_responses_per_prompt > 1:
                responses_list = []
                for n in range(args.num_responses_per_prompt):
                    # Use deterministic seed for each response: base_seed + prompt_id + response_idx
                    # This ensures reproducible generation while getting diverse samples
                    response_seed = args.seed + i + n
                    torch.manual_seed(response_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(response_seed)
                    
                    response = generate_response(
                        model, tokenizer, prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample
                    )
                    responses_list.append(response)
                response_value = responses_list  # List of N responses
            else:
                # Single response: use base seed + prompt_id for reproducibility
                single_seed = args.seed + i
                torch.manual_seed(single_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(single_seed)
                
                response_value = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample
                )
            
            # Find or create entry for this prompt
            if i >= len(all_responses):
                all_responses.append({
                    'prompt_id': i,
                    'prompt': prompt,
                    'source': prompt_data['source'],
                    'responses': {}
                })
            
            all_responses[i]['responses'][model_name] = response_value
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save responses
    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK]Responses saved to: {output_file}")
    print(f"  Total prompts: {len(all_responses)}")
    print(f"  Models: {args.model_names}")
    print(f"  Responses per prompt: {args.num_responses_per_prompt}")
    if args.do_sample:
        print(f"  Decoding: Stochastic (temperature={args.temperature}, top_p={args.top_p})")
        print(f"  Seeding: Deterministic schedule (base_seed={args.seed} + prompt_id + response_idx)")
    else:
        print(f"  Decoding: Deterministic (greedy)")
        print(f"  Seeding: Deterministic (base_seed={args.seed} + prompt_id)")


if __name__ == "__main__":
    main()
