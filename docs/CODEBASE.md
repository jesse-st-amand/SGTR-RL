# SGTR-RL Codebase Guide

Reference for developers and AI agents working on the SGTR-RL training pipeline.

## What This Repo Does

Trains language models to improve **Self-Generated Text Recognition (SGTR)** using reinforcement learning. The core task: given a piece of text, can a model tell whether it wrote it?

The model (Llama-3.1-8B-Instruct) is trained via **SFT** (supervised fine-tuning) and **GRPO** (Group Relative Policy Optimization) with **LoRA** fine-tuning on a managed GPU platform (**Tinker**). Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline.

## Repository Layout

```
SGTR-RL/
├── sgtr_rl/                      # Main Python package
│   ├── config/                   # Data path resolution
│   ├── data_processing/          # Data loading and prompt construction
│   ├── training/                 # Training loop, reward, config, logging
│   ├── evaluation/               # Checkpoint evaluation via inspect-ai
│   └── scripts/                  # CLI entry points
├── experiments/                  # Experiment configs (one YAML per experiment)
├── analysis/                     # Log parsing, plotting, and Jupyter notebook
├── baselines/                    # Baseline evaluation results (JSON)
├── config/                       # Data path configuration
├── data/                         # Data directory (gitignored, see below)
├── results/                      # Training run outputs (gitignored)
├── _external/                    # Git submodules (reference only, not used at runtime)
├── docs/                         # Documentation (this file)
├── pyproject.toml                # Package definition and dependencies
└── uv.lock                       # Locked dependency versions
```

## Key Directories in Detail

### `sgtr_rl/training/` — Training Pipeline

| File | Purpose |
|------|---------|
| `grpo_trainer.py` | `TinkerRLTrainer` (Tinker API) and `LocalGRPOTrainer` (TRL/local GPU). Implements the GRPO loop: sample rollouts, compute rewards, center advantages within groups, build Tinker datums, call `forward_backward` + `optim_step`. |
| `sft_trainer.py` | `TinkerSFTTrainer` — supervised fine-tuning via Tinker's `cross_entropy` loss. Trains on labeled (prompt, target) pairs with loss weights only on assistant tokens. Simpler than GRPO: no rollouts or advantages. |
| `eval.py` | Shared validation evaluation logic. `evaluate_val()` (greedy accuracy), `compute_val_nll()` (forward-pass NLL), `log_val_result()`, `log_val_metrics()`, `save_val_predictions()`. Used by both SFT and GRPO trainers. |
| `train_config.py` | `TrainingConfig` dataclass + `load_training_config()` YAML parser. All hyperparameters flow through here. |
| `reward.py` | `sgtr_binary_reward()` — extracts "1" or "2" from model output, returns 1.0 if correct, 0.0 otherwise. `_extract_answer()` handles both bare digits and "Answer: N" patterns. |
| `run_dir.py` | Creates structured run directories under `results/`. Handles run naming, config freezing, and existing-run policies. |
| `logging_setup.py` | Dual logging to terminal + file. |

### `sgtr_rl/scripts/` — CLI Entry Points

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training entry point | `python -m sgtr_rl.scripts.train --config experiments/03_.../config.yaml` |
| `extract_from_eval.py` | Extract training data from `.eval` files | `python -m sgtr_rl.scripts.extract_from_eval --eval_dir data/original/llama8b --output data/training_data/sharegpt_ind/ --format ind` |
| `prepare_data.py` | Build prompts from raw generation data (alternative to extract_from_eval) | `python -m sgtr_rl.scripts.prepare_data --evaluator_model ll-3.1-8b ...` |
| `eval_baseline.py` | Evaluate base model accuracy on any JSONL dataset via Tinker | `python sgtr_rl/scripts/eval_baseline.py --data data/training_data/sharegpt_ind_cot/val.jsonl` |
| `evaluate.py` | Run eval tasks from experiment config on a trained checkpoint | `python -m sgtr_rl.scripts.evaluate --checkpoint path/to/ckpt --config experiments/.../config.yaml` |
| `dry_run.py` | Simulate training locally (no GPU/API) to validate reward/advantage logic | `python sgtr_rl/scripts/dry_run.py --config experiments/.../config.yaml` |

### `sgtr_rl/data_processing/` — Data Preparation

| File | Purpose | Status |
|------|---------|--------|
| `prompt_builder.py` | Build SGTR prompts from raw generation data via self-rec-framework | Active (used by `prepare_data.py`) |
| `validate_data.py` | Validate train/val JSONL integrity: schema, target values, UUID overlap, PW format checks | Active |
| `eval_loader.py` | Parse `.eval` files into `EvalSample` objects for DPO | Stale (DPO not in use) |
| `triple_generator.py` | Generate DPO (prompt, chosen, rejected) triples | Stale (DPO not in use) |

### `sgtr_rl/evaluation/` — Checkpoint Evaluation

| File | Purpose |
|------|---------|
| `evaluator.py` | `evaluate_checkpoint()` — runs SGTR and standard inspect-ai benchmarks. `EvalCallback` for periodic eval during training. |

### `analysis/` — Log Parsing and Visualization

| File | Purpose |
|------|---------|
| `utils.py` | Parse `train.log` files into structured `RunData` objects. `load_run()`, `list_runs()`. |
| `plotting.py` | Matplotlib plots: dashboard, accuracy, reward, datums, timing. |
| `plot_run.py` | CLI to generate all plots for a run: `python -m analysis.plot_run results/my_run/` |
| `analysis.ipynb` | Interactive Jupyter notebook for analysis. |

### `experiments/` — Experiment Configs

Each experiment is a YAML file defining model, data, hyperparameters, and evaluation tasks.

| Experiment | Description | Backend |
|-----------|-------------|---------|
| `01_RL_grpo_IND_WikiSum` | WikiSum individual recognition | local |
| `02_RL_grpo_IND_ShareGPT` | ShareGPT individual recognition | local |
| `03_RL_grpo_IND_ShareGPT_CoT` | ShareGPT with chain-of-thought prompts | tinker |
| `04_overfit_debug` | Overfit 16 prompts (debugging) | tinker |
| `05_trivial_sanity` | Trivial task (all targets="1") — pipeline validation | tinker |
| `06_trivial_sanity_2` | Trivial task (all targets="2") — pipeline validation | tinker |
| `07_overfit_high_temp` | Overfit with sampling_temperature=1.5 | tinker |
| `08_sft_ind_cot` | SFT warmup on IND_COT (320 train, 80 val) | tinker |
| `09_sft_pairwise` | SFT warmup on PW (184 train, 62 val) — sample-level split (leaked) | tinker |
| `14_sft_pw_uuid_split` | SFT on PW with UUID-level split (160 train, 40 val, no leakage) | tinker |

### `data/` — Data Directory (gitignored)

```
data/
├── original/              # Source .eval files (inspect-ai zip archives)
│   └── llama8b/           # Llama-3.1-8B eval data
├── raw/                   # Pristine copies of processed data (backup)
├── training_data/         # JSONL datasets for training
│   ├── sharegpt_ind/      # Individual format (320 train, 80 val)
│   ├── sharegpt_ind_cot/  # Individual + CoT (320 train, 80 val)
│   ├── sharegpt_pw/       # Pairwise format (UUID-level split, both orderings)
│   ├── sharegpt_ind_cot_overfit/  # 16-sample debug subset
│   ├── trivial_test/      # Debug: all targets="1"
│   └── trivial_test_2/    # Debug: all targets="2"
├── cached_generations/    # Symlink to self-rec-framework/data/results
├── checkpoints/           # Model checkpoints
└── README.md
```

### `results/` — Training Run Outputs (gitignored)

Each run creates a directory under `results/`:
```
results/{experiment_name}__{overrides}__{timestamp}/
├── config.yaml          # Frozen config (including CLI overrides)
├── extraction_meta.json # Data extraction metadata
├── train.log            # Full training log
├── metrics/             # W&B and JSON metric logs
├── val_predictions/     # Per-epoch val predictions (epoch_0.json, epoch_1.json, ...)
├── checkpoints/         # Saved model checkpoints
└── eval/                # Evaluation results
```

View with: W&B dashboard (project configured via `wandb_project` in config)

## Training Data Formats

**Individual (IND/IND_COT):**
```json
{"prompt": "Below is a user request followed by a response. Did you write this response? Answer 1 for yes, 2 for no...", "target": "1", "metadata": {"uuid": "...", "format": "ind", ...}}
```

**Pairwise (PW):**
```json
{"prompt": "Below are two responses... Which one did you write? ...", "target": "1", "metadata": {"uuid": "...", "format": "pw", ...}}
```

Targets are always `"1"` or `"2"`.

## How Training Works (TinkerRLTrainer)

1. Load JSONL prompts and connect to Tinker API
2. For each batch of prompts:
   - Sample `num_rollouts_per_prompt` completions per prompt via Tinker
   - Score each completion with `sgtr_binary_reward` (1.0 if correct, 0.0 otherwise)
   - Compute GRPO advantages: center rewards within each group (prompt)
   - Skip groups where all rollouts gave the same answer (zero signal)
   - Build Tinker `Datum` objects with importance sampling loss inputs
   - Call `forward_backward` + `optim_step`
3. At each epoch boundary: run greedy eval on validation set + compute val NLL
4. Log metrics to W&B and `train.log`

Key insight: GRPO needs **within-group variance** to learn. If all rollouts for a prompt give the same answer, that group provides zero gradient signal.

## How to Run Things

### Prerequisites

```bash
uv sync                     # Install all dependencies
cp .env.template .env       # Add TINKER_API_KEY
```

### Training

```bash
# Run an experiment
python -m sgtr_rl.scripts.train --config experiments/03_RL_grpo_IND_ShareGPT_CoT/config.yaml

# With CLI overrides
python -m sgtr_rl.scripts.train --config experiments/03_.../config.yaml \
    --learning_rate 1e-4 --num_epochs 5

# Skip if run already exists
python -m sgtr_rl.scripts.train --config ... --exists skip
```

### Preparing Data

```bash
# From .eval files (primary method):
python -m sgtr_rl.scripts.extract_from_eval \
    --eval_dir data/original/llama8b \
    --output data/training_data/sharegpt_ind/ \
    --format ind

# With chain-of-thought:
python -m sgtr_rl.scripts.extract_from_eval \
    --eval_dir data/original/llama8b \
    --output data/training_data/sharegpt_ind_cot/ \
    --format ind --cot

# Pairwise format (UUID-level split keeps both orderings together):
python -m sgtr_rl.scripts.extract_from_eval \
    --eval_dir data/original/llama8b \
    --output data/training_data/sharegpt_pw/ \
    --format pw
```

### Baseline Evaluation

```bash
python sgtr_rl/scripts/eval_baseline.py \
    --data data/training_data/sharegpt_ind_cot/val.jsonl
```

### Analysis

```bash
# Generate plots for a run
python -m analysis.plot_run results/my_run_dir/

# Or use the Jupyter notebook
jupyter lab analysis/analysis.ipynb

# W&B (metrics are logged automatically during training)
```

### Dry Run (No GPU)

```bash
python sgtr_rl/scripts/dry_run.py \
    --config experiments/05_trivial_sanity/config.yaml \
    --base-acc 0.5
```

## Development

### Environment

```bash
uv sync                    # Install/update dependencies
uv add <package>           # Add a new dependency (updates pyproject.toml + uv.lock)
uv run ruff check .        # Lint
uv run ruff format .       # Format
```

### Branch

Working branch: check `git branch` (development has been on feature branches off `main`).

### Adding a New Experiment

1. Create `experiments/NN_description/config.yaml` (copy from an existing one)
2. Prepare training data if needed (see above)
3. Run: `python -m sgtr_rl.scripts.train --config experiments/NN_.../config.yaml`

### Adding a New Training Algorithm

1. Add the algorithm name to `TrainingConfig.algorithm` options
2. Create a new trainer class (follow `TinkerRLTrainer` pattern)
3. Register it in `sgtr_rl/scripts/train.py` `TRAINERS` dict

### Experiment Config Schema

```yaml
experiment_name: "descriptive_name"
description: "What this experiment tests"

algorithm: grpo          # grpo | dpo | sft
backend: tinker          # tinker | local

data:
  train_file: data/training_data/sharegpt_ind_cot/train.jsonl
  val_file: data/training_data/sharegpt_ind_cot/val.jsonl

model:
  name: meta-llama/Llama-3.1-8B-Instruct
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.05

wandb_project: sgtr-rl       # Optional: W&B project name (omit to disable)

hyperparameters:
  learning_rate: 5.0e-5
  num_epochs: 3
  per_device_train_batch_size: 4
  num_rollouts_per_prompt: 8
  max_completion_length: 512
  sampling_temperature: 1.0    # Controls rollout diversity
  seed: 42

checkpointing:
  save_steps: 20
  eval_steps: 20
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `tinker` + `tinker-cookbook` | Managed GPU training (LoRA fine-tuning, sampling, checkpoints, ml_log) |
| `self-rec-framework` | SGTR experiment configs, prompts, evaluation tasks |
| `inspect-ai` | LLM evaluation framework (.eval file format) |
| `trl` + `peft` + `transformers` | Local GPU training path (HuggingFace stack) |
| `torch` | Tensor operations for Tinker datum construction |
| `wandb` | Experiment tracking and metric visualization |

## Known Issues / Technical Debt

- **DPO infrastructure** (`eval_loader.py`, `triple_generator.py`, `test_data_loading.py`) is unused but kept for potential future use
- `PROJECT_SETUP.md` and `SETUP_COMPLETE.md` are stale planning documents from early project phases
- `_external/` submodules (`inspect_ai/`, `self-rec-framework/`) are mostly empty — packages are installed via pip from git URLs
- `config/paths.py` has helper functions (`get_checkpoints_path()`, etc.) that aren't used by the current training pipeline
- `data/README.md` references DPO-era terminology and paths
- **Small dataset**: Current PW dataset is 100 unique pairs (200 records). Results are promising (90% val accuracy with SFT) but more data would improve robustness
