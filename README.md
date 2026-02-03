# KP-RLHF: Learning from K-Wise Comparisons and Partial Rankings for RLHF

A framework for training reward models using pairwise and listwise models and integration into whole RLHF pipelines.

## Overview

This project implements multiple reward modeling approaches for Reinforcement Learning from Human Feedback:

### Pairwise Methods
- **BT (Bradley-Terry)**: Standard pairwise comparison model
- **BT-Heuristic**: Bradley-Terry with heuristic tie handling
- **BT-Davidson**: Bradley-Terry with Davidson extension for ties

### Listwise (K-Wise) Methods
- **PL (Plackett-Luce)**: Standard listwise ranking model
- **Cox-Breslow**: Partial likelihood approach for rankings
- **DL (Davidson-Luce)**: Extended Plackett-Luce with Davidson-style tie modeling

### Models
- **Reward Model Encoder**: `microsoft/deberta-v3-base` (184M parameters)
- **Policy Model**: `chavinlo/alpaca-native` (Alpaca-7B) with LoRA fine-tuning
- **LLM Judges**: `meta-llama/Llama-3.1-8B-Instruct`, `prometheus-eval/prometheus-7b-v2.0`

## Project Structure

```
├── main.py                 # Main entry point for the pipeline
├── configs/                # YAML configuration files
│   ├── config_M_pair*.yaml       # Pairwise model configs
│   ├── config_M_kw_*.yaml        # Listwise model configs
│   └── config_rlhf_*.yaml        # PPO/GRPO configs
├── dataset/                # Data preparation and preprocessing
├── pairwise/               # Pairwise reward model trainers
├── k_wise/                 # Listwise reward model trainers
├── evaluation/             # Model evaluation scripts
└── rlhf/                   # PPO/GRPO training and response generation
```

## Installation

```bash
pip install -r requirements.txt

# For Windows
pip install -r requirements_windows.txt

# For RewardBench evaluation (separate environment recommended)
pip install -r requirements_rewardbench.txt
```

## Usage

```bash
# Quick test run (uses TinyLlama for fast iteration)
python main.py --mode test

# Full training pipeline
python main.py --mode full

# Train reward model only
python main.py --mode reward_only

# Run policy optimization only
python main.py --mode policy_only

# Run evaluation only
python main.py --mode evaluate
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | Run mode: `test`, `full`, `reward_only`, `policy_only`, `evaluate` |
| `--model` | Select reward model: `BT`, `BT-Heuristic`, `BT-Davidson`, `PL`, `Cox`, `DL` |
| `--po` | Policy optimization algorithm: `PPO` or `GRPO` |
| `--reward_model` | Path to existing reward model checkpoint |
| `--seed` | Random seed for reproducibility |

## Pipeline

1. **Dataset Preparation**: Load and preprocess UltraFeedback preference dataset
2. **Reward Modeling**: Train reward models using pairwise or listwise methods
3. **Evaluation of Reward Models**: Kendall's Tau-b on test/OOD data, RewardBench
4. **Policy Optimization**: PPO or GRPO using trained reward models
5. **LLM Judge Evaluation**: Compare policies via pairwise LLM judging

## Evaluation Metrics

- **In-Distribution**: Kendall's τ-b with bootstrap confidence intervals on UltraFeedback test split
- **Out-of-Distribution**: Kendall's τ-b on HelpSteer dataset
- **RewardBench**: Best-of-4 selection accuracy across chat, safety, and reasoning categories
- **LLM Judge**: Pairwise win rates using Llama-3.1-8B or Prometheus-7B judges

## Datasets

- **Training**: UltraFeedback
- **OOD Evaluation**: HelpSteer
- **SFT Dataset/PO**: Alpaca
- **PO Evaluation with LLM judges**: XL-AlpacaEval

## Requirements

- Python 3.8+
- PyTorch 2.6.0 (CUDA 12.4)
- Transformers 4.57.1
- TRL 0.27.0
- Datasets 4.1.1
- Accelerate 1.10.1
- PEFT 0.18.1
- BitsAndBytes 0.49.1
- SciPy 1.13.1
- NumPy 1.26.4
- Pandas 2.3.3
- TensorBoard 2.20.0
- RewardBench 0.1.4 (separate environment)

## Notes

- **Test Mode**: Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for quick local testing with reduced batch sizes and steps
- **RewardBench**: Requires temporary package downgrades (`transformers==4.51.0`). Use a separate environment or restore packages after evaluation
- **Caching**: Models and datasets are cached locally via `dataset/cache_utils.py` for faster subsequent runs