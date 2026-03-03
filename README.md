# SGTR-RL

**Training for Self-Generated Text Recognition**

Train language models to recognize their own writing. The core task: given text, can a model tell whether it wrote it? Currently training Llama-3.1-8B-Instruct via **SFT** and **GRPO** (Group Relative Policy Optimization) with LoRA fine-tuning on [Tinker](https://tinker-docs.thinkingmachines.ai) (managed GPU platform).

Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline, which generates pairwise and individual SGTR prompts from model outputs.

## Quick Start

```bash
git clone https://github.com/jesse-st-amand/SGTR-RL.git
cd SGTR-RL
uv sync
cp .env.template .env           # fill in TINKER_API_KEY, WANDB_API_KEY
```

## How It Works

**1. Prepare data** — extract SGTR prompts from eval files:
```bash
# Download from HuggingFace and extract training data:
python -m sgtr_rl.scripts.download_hf_data \
    --evaluator ll-3.1-8b --dataset sharegpt --name llama8b --extract

# Or from local .eval files:
python -m sgtr_rl.scripts.extract_from_eval \
    --eval_dir data/original/llama8b \
    --output data/training_data/sharegpt_pw/ \
    --format pw
```

**2. Train** — SFT or GRPO with LoRA on Tinker:
```bash
python -m sgtr_rl.scripts.train \
    --config experiments/15_sft_pw_rec_vs_qwen/config.yaml
```

**3. Monitor** — metrics logged to [Weights & Biases](https://wandb.ai):
- Val accuracy + NLL computed at each epoch boundary
- Epoch 0 baseline (untrained model) for reference
- MMLU and cross-domain SGTR benchmarks tracked throughout training
- Per-sample predictions saved to `val_predictions/` and `benchmark_predictions/`

## Current Results

SFT experiments 15-22 training Llama-3.1-8B on pairwise and individual SGTR across multiple "other" models (Qwen-2.5-7B, Haiku-3.5, GPT-4o, Llama-70B, Claude Opus):
- **90%+ val accuracy** on held-out UUIDs across all model pairs
- Untrained baseline: ~45% (below chance)
- Cross-format generalisation: training on pairwise transfers to individual and vice versa
- Cross-domain generalisation: improvements transfer across ShareGPT, WikiSum, PKU, BigCode
- See `results/batch_15-22_sft_cross_evals/` for plots and analysis

## Architecture

Composed via a single YAML config per experiment:

| Dimension          | Options                                    |
|--------------------|--------------------------------------------|
| Training algorithm | `sft` (implemented) \| `grpo` (implemented) |
| Training backend   | `tinker` (implemented) \| `local` (TRL)    |
| Data format        | `pw` (pairwise) \| `ind` (individual) \| `ind_cot` (individual + CoT) |

## Project Structure

```
SGTR-RL/
├── sgtr_rl/
│   ├── training/
│   │   ├── sft_trainer.py          # TinkerSFTTrainer (cross-entropy loss)
│   │   ├── grpo_trainer.py         # TinkerRLTrainer + LocalGRPOTrainer
│   │   ├── eval.py                 # Shared val evaluation (accuracy, NLL, predictions)
│   │   ├── train_config.py         # TrainingConfig dataclass + YAML loader
│   │   ├── reward.py               # Binary reward (extract "1"/"2", compare to target)
│   │   ├── benchmark_eval.py       # MMLU + SGTR cross-eval during training
│   │   ├── plot_summary.py         # Auto-generated training summary plots
│   │   ├── run_dir.py              # Structured run directory creation
│   │   └── logging_setup.py        # Dual logging (terminal + file)
│   ├── data_processing/
│   │   ├── validate_data.py        # Train/val integrity checks (UUID overlap, schema)
│   │   └── prompt_builder.py       # Build SGTR prompts from raw generation data
│   ├── evaluation/
│   │   └── evaluator.py            # Checkpoint evaluation via inspect-ai
│   └── scripts/
│       ├── train.py                # Main training entry point
│       ├── extract_from_eval.py    # Extract training data from .eval files
│       ├── download_hf_data.py     # Download eval data from HuggingFace
│       ├── eval_baseline.py        # Evaluate untrained model on any dataset
│       ├── prepare_mmlu.py         # Prepare MMLU benchmark data
│       ├── plot_cross_evals.py     # Cross-eval analysis plots
│       ├── dry_run.py              # Simulate training locally (no GPU)
│       ├── prepare_data.py         # Build prompts from raw generation data
│       └── evaluate.py             # Run eval tasks on trained checkpoint
├── experiments/                    # One YAML config per experiment (01-22)
├── analysis/                       # Log parsing, plotting, Jupyter notebook
├── data/                           # Training data (gitignored)
├── results/                        # Run outputs: logs, metrics, predictions (gitignored)
└── tests/                          # Unit tests
```

## Training Data Formats

**Pairwise (PW):**
```json
{"prompt": "Below are two responses... Which one did you write?", "target": "1", "metadata": {"uuid": "...", "format": "pw"}}
```

**Individual (IND):**
```json
{"prompt": "Did you write this response? Answer 1 for yes, 2 for no.", "target": "1", "metadata": {"uuid": "...", "format": "ind"}}
```

Targets are always `"1"` or `"2"`. For PW format, each UUID has exactly 2 records (both response orderings), and train/val splits are done at the UUID level to prevent leakage.

## Dependencies

| Package | Purpose |
|---------|---------|
| [tinker](https://tinker-docs.thinkingmachines.ai) + [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | Managed GPU training (LoRA, sampling, checkpoints, logging) |
| [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) | SGTR experiment configs, prompts, evaluation tasks |
| [inspect-ai](https://inspect.aisi.org.uk/) | LLM evaluation framework (.eval file format) |
| [wandb](https://wandb.ai) | Experiment tracking and metric visualization |

All installed automatically by `uv sync`.

## Environment Variables

Copy `.env.template` to `.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | For Tinker backend | Managed GPU training |
| `WANDB_API_KEY` | For W&B logging | Experiment tracking |
| `TOGETHER_API_KEY` | For Together eval backend | Model inference for evaluation |

## Documentation

See [`docs/CODEBASE.md`](docs/CODEBASE.md) for detailed codebase reference including config schema, data pipeline, and development guide.
