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
python -m scripts.prepare_data --evaluator ll-3.1-8b
```

**2. Train** — SFT or GRPO with LoRA on Tinker:
```bash
python -m scripts.train \
    --config experiments/15_sft_pw_rec_vs_qwen/config.yaml
```

**3. Monitor** — metrics logged to [Weights & Biases](https://wandb.ai):
- Val accuracy + NLL computed at each epoch boundary
- Epoch 0 baseline (untrained model) for reference
- MMLU and cross-domain SGTR benchmarks tracked throughout training
- Per-sample predictions saved to `val_predictions/` and `benchmark_predictions/`

## Current Results

SFT experiments 15-22 training Llama-3.1-8B on pairwise and individual SGTR across multiple "other" models (Qwen-2.5-7B, Haiku-3.5, GPT-4o, Llama-70B, Claude Opus):
- **90%+ val accuracy** on held-out IDs across all model pairs
- Untrained baseline: ~45% (below chance)
- See `results/batch_15-22_sft_cross_evals/` for plots and analysis

## Project Structure

```
SGTR-RL/
├── sgtr_rl/                       # Flat, training-focused package
│   ├── config.py                  # TrainingConfig + YAML loader
│   ├── pipeline.py                # run_training() orchestration
│   ├── sft.py                     # train_sft() function
│   ├── grpo.py                    # train_grpo() function
│   ├── tinker.py                  # TinkerContext + setup_tinker()
│   ├── answer.py                  # extract_answer (1/2 extraction)
│   ├── reward.py                  # Binary reward function
│   ├── data.py                    # Data loading + validation
│   ├── eval.py                    # Val evaluation (accuracy, NLL)
│   ├── benchmarks.py              # MMLU + SGTR cross-eval
│   ├── runs.py                    # Run directory management
│   ├── plotting.py                # Summary plot generation
│   └── logging_setup.py           # Dual logging (terminal + file)
├── scripts/                       # CLI entry points
│   ├── train.py                   # Main training entry point
│   ├── prepare_data.py            # Download + extract training data
│   ├── prepare_mmlu.py            # Prepare MMLU benchmark data
│   └── plot_cross_evals.py        # Cross-eval analysis plots
├── experiments/                   # One YAML config per experiment (14-22)
├── data/                          # Training data (gitignored)
├── results/                       # Run outputs: logs, metrics, predictions (gitignored)
└── tests/                         # Unit and integration tests
```

## Training Data Format

Flat JSON schema:
```json
{"prompt": "...", "target": "1", "id": "abc-123", "format": "pw", "opponent_model": "qwen-2.5-7b"}
```

Targets are always `"1"` or `"2"`. For PW format, each ID has exactly 2 records (both response orderings), and train/val splits are done at the ID level to prevent leakage.

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
