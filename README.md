# SGTR-RL

**Training for Self-Generated Text Recognition**

Train language models to recognize their own writing. The core task: given text, can a model tell whether it wrote it? The repo now supports two training paths:

- **Tinker backend** for the existing managed-GPU workflow
- **Local backend** for single-node HF/PEFT LoRA SFT on local GPUs or RunPod

Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline, which generates pairwise and individual SGTR prompts from model outputs.

## Quick Start

```bash
git clone https://github.com/jesse-st-amand/SGTR-RL.git
cd SGTR-RL
uv sync
cp .env.template .env           # fill in backend/provider tokens as needed
```

## How It Works

**1. Prepare data** — extract SGTR prompts from eval files:
```bash
# Download from HuggingFace and extract training data:
uv run sgtr-prepare-data --evaluator ll-3.1-8b
```

**2. Train** — choose an experiment config and runtime:
```bash
# Existing Tinker path:
uv run sgtr-train \
    --config experiments/01_sft_pw_vs_qwen/config.yaml

# Local GPU path:
uv run sgtr-train \
    --config experiments/01_sft_pw_vs_qwen/config.yaml \
    --runtime runtimes/local_gpu.yaml
```

**3. Launch on RunPod** — one-shot pod creation, training, wait, and teardown:
```bash
uv run sgtr-runpod-launch \
    --config experiments/01_sft_pw_vs_qwen/config.yaml \
    --runtime runtimes/runpod_a100.yaml
```

**4. Monitor** — metrics logged to [Weights & Biases](https://wandb.ai) and written locally:
- Val accuracy + NLL computed at each epoch boundary
- Epoch 0 baseline (untrained model) for reference
- MMLU and cross-domain SGTR benchmarks tracked throughout training
- Per-sample predictions saved to `val_predictions/` and `benchmark_predictions/`
- `status.json` updated as the run progresses
- Final local checkpoints saved as PEFT adapter weights in `checkpoints/final/`

## CLI Commands

All scripts are registered as entry points and can be invoked via `uv run`:

| Command | Description |
|---------|-------------|
| `sgtr-train` | Run SFT/GRPO training with a config file |
| `sgtr-runpod-launch` | Launch a one-shot training job on RunPod |
| `sgtr-prepare-data` | Download eval results from HuggingFace and extract training data |
| `sgtr-prepare-mmlu` | Prepare MMLU benchmark data |
| `sgtr-plot-summary` | Generate per-model summary plots for SFT results |
| `sgtr-plot-cross-evals` | Generate cross-evaluation analysis plots |
| `sgtr-manage-checkpoints` | List and clean up persistent Tinker checkpoints |

Use `--help` with any command for full usage details, e.g. `uv run sgtr-train --help`.

## Current Results

SFT experiments 15-22 training Llama-3.1-8B on pairwise and individual SGTR across multiple "other" models (Qwen-2.5-7B, Haiku-3.5, GPT-4o, Llama-70B, Claude Opus):
- **90%+ val accuracy** on held-out IDs across all model pairs
- Untrained baseline: ~45% (below chance)
- See `results/batch_15-22_sft_cross_evals/` for plots and analysis

## Project Structure

```
SGTR-RL/
├── sgtr_rl/                       # Installable package
│   ├── config.py                  # TrainingConfig + experiment YAML loader
│   ├── runtime_config.py          # RuntimeConfig + runtime YAML loader
│   ├── artifacts.py               # status.json + JSONL metric helpers
│   ├── pipeline.py                # run_training() orchestration + backend dispatch
│   ├── sft.py                     # Tinker SFT training loop
│   ├── local_sft.py               # Local HF/PEFT SFT training loop + checkpoint save
│   ├── local_eval.py              # Local validation + benchmark evaluation
│   ├── grpo.py                    # Tinker GRPO training loop
│   ├── tinker.py                  # TinkerContext + setup_tinker()
│   ├── answer.py                  # extract_answer (1/2 extraction)
│   ├── reward.py                  # Binary reward function
│   ├── data.py                    # Data loading + validation
│   ├── tinker_eval.py             # Val + benchmark evaluation on Tinker
│   ├── benchmarks.py              # Pure benchmark formatting/scheduling logic
│   ├── metrics.py                 # Metric logging + prediction saving
│   ├── runs.py                    # Run directory management
│   ├── logging_setup.py           # Dual logging (terminal + file)
│   └── scripts/                   # CLI entry points (registered in pyproject.toml)
│       ├── train.py               # Main training entry point
│       ├── runpod_launch.py       # Create/poll/delete one-shot RunPod jobs
│       ├── runpod_utils.py        # RunPod request/startup-script helpers
│       ├── prepare_data.py        # Download + extract training data
│       ├── prepare_mmlu.py        # Prepare MMLU benchmark data
│       ├── plot_cross_evals.py    # Cross-eval analysis plots
│       ├── plotting_utils.py      # Shared plotting helpers
│       └── plot_summary.py        # Per-run summary charts for SFT results
├── runtimes/                      # Machine/provider runtime configs
│   ├── local_gpu.yaml             # Example local single-node runtime
│   └── runpod_a100.yaml           # Example RunPod runtime
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
| [transformers](https://huggingface.co/docs/transformers/index) + [peft](https://huggingface.co/docs/peft/index) | Local single-node LoRA training/eval |
| [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) | SGTR experiment configs, prompts, evaluation tasks |
| [inspect-ai](https://inspect.aisi.org.uk/) | LLM evaluation framework (.eval file format) |
| [wandb](https://wandb.ai) | Experiment tracking and metric visualization |

All installed automatically by `uv sync`.

## Environment Variables

Copy `.env.template` to `.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | For Tinker backend | Managed GPU training |
| `RUNPOD_API_KEY` | For RunPod launcher | Pod create/poll/delete |
| `HF_TOKEN` | For local/RunPod gated models | Model download/auth |
| `WANDB_API_KEY` | For W&B logging | Experiment tracking |
| `TOGETHER_API_KEY` | For Together eval backend | Model inference for evaluation |

## Runtime Configs

Experiment YAMLs in `experiments/` stay focused on model/data/algorithm choices. Machine-specific settings live in `runtimes/`.

- `runtimes/local_gpu.yaml`: local single-node HF/PEFT SFT
- `runtimes/runpod_a100.yaml`: example RunPod config with a mounted network volume

For local/RunPod v1, only **SFT** is supported on the local backend. GRPO remains Tinker-only.

## Documentation

See [`docs/CODEBASE.md`](docs/CODEBASE.md) for detailed codebase reference including config schema, data pipeline, and development guide.
