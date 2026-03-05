# SGTR-RL Codebase Guide

Reference for developers and AI agents working on the SGTR-RL training pipeline.

## What This Repo Does

Trains language models to improve **Self-Generated Text Recognition (SGTR)** using reinforcement learning. The core task: given a piece of text, can a model tell whether it wrote it?

The model (Llama-3.1-8B-Instruct) is trained via **SFT** (supervised fine-tuning) and **GRPO** (Group Relative Policy Optimization) with **LoRA** fine-tuning on a managed GPU platform (**Tinker**). Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline.

## Repository Layout

```
SGTR-RL/
├── sgtr_rl/                      # Flat, training-focused Python package
│   ├── config.py                 # TrainingConfig + YAML loader
│   ├── answer.py                 # extract_answer (paradigm-agnostic)
│   ├── reward.py                 # sgtr_binary_reward (uses answer.py)
│   ├── tinker.py                 # TinkerContext dataclass + setup_tinker()
│   ├── pipeline.py               # run_training() orchestration
│   ├── sft.py                    # train_sft() function
│   ├── grpo.py                   # train_grpo() function
│   ├── data.py                   # load_jsonl, validate_training_data, build_conversation
│   ├── tinker_eval.py            # Tinker-based eval (val accuracy/NLL, MMLU, SGTR cross-eval)
│   ├── benchmarks.py             # Pure benchmark logic (prompt formatting, scheduling)
│   ├── metrics.py                # Metric logging and prediction saving
│   ├── runs.py                   # Run directory creation
│   ├── logging_setup.py          # Dual logging setup
│   └── plotting.py               # Summary plot generation
├── scripts/                      # Top-level CLI entry points
│   ├── train.py                  # Main training entry point
│   ├── prepare_data.py           # Download + extract training data
│   ├── prepare_mmlu.py           # Prepare MMLU benchmark data
│   └── plot_cross_evals.py       # Cross-eval analysis plots
├── experiments/                  # Experiment configs (one YAML per experiment)
├── data/                         # Data directory (gitignored, see below)
├── results/                      # Training run outputs (gitignored)
├── docs/                         # Documentation (this file)
├── tests/                        # Unit and integration tests
├── pyproject.toml                # Package definition and dependencies
└── uv.lock                       # Locked dependency versions
```

## Package Modules (`sgtr_rl/`)

| File | Purpose |
|------|---------|
| `config.py` | `TrainingConfig` dataclass + `load_training_config()` YAML parser. All hyperparameters flow through here. Includes `BenchmarkEvalConfig`. |
| `answer.py` | `extract_answer()` — extracts "1" or "2" from model output text. Handles both bare digits and "Answer: N" patterns. |
| `reward.py` | `sgtr_binary_reward()` — extracts answer from model output, returns 1.0 if correct, 0.0 otherwise. |
| `tinker.py` | `TinkerContext` dataclass (shared Tinker state) + `setup_tinker()` (creates ServiceClient, training client, tokenizer, renderer, params) + `save_checkpoint()`. |
| `pipeline.py` | `run_training()` — full pipeline orchestration: load data → validate → setup tinker → baseline eval → train → checkpoint → plot → close. |
| `sft.py` | `train_sft(config, ctx, prompts, val_prompts)` — SFT training loop. Cross-entropy loss on (prompt, target) pairs with assistant-token masking. |
| `grpo.py` | `train_grpo(config, ctx, prompts, val_prompts)` — GRPO training loop: sample rollouts, compute rewards, center advantages within groups, build Tinker datums, call `forward_backward` + `optim_step`. |
| `data.py` | `load_jsonl()`, `validate_training_data()`, `build_conversation()` — data loading, integrity validation, and conversation construction. |
| `tinker_eval.py` | Tinker-based evaluation. Orchestrators `run_val_eval(prompts, ctx, ...)` and `run_benchmark_evals(configs, ctx, ...)` accept `TinkerContext`. Lower-level: `evaluate_val()`, `compute_val_nll()`, `evaluate_benchmark()` (MMLU), `evaluate_sgtr_benchmark()` (cross-eval). |
| `benchmarks.py` | Pure benchmark logic (no Tinker deps). `format_mmlu_prompt()`, `extract_mmlu_answer()`, `should_run_benchmark()`, `_subsample()`, `load_benchmark_data()`. |
| `metrics.py` | Metric logging and prediction saving. `log_val_result()`, `log_val_metrics()`, `save_val_predictions()`. |
| `runs.py` | Creates structured run directories under `results/`. Handles run naming, config freezing, and existing-run policies. |
| `plotting.py` | `generate_summary_plot()` — 3-subplot summary figure (loss, accuracy, benchmarks) from `metrics/metrics.jsonl`. |
| `logging_setup.py` | Dual logging to terminal + file. |

## Scripts (`scripts/`)

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training entry point | `python -m scripts.train --config experiments/01_.../config.yaml` |
| `prepare_data.py` | Download from HuggingFace + extract training data | `python -m scripts.prepare_data --evaluator ll-3.1-8b` |
| `prepare_mmlu.py` | Download MMLU and prepare benchmark JSONL | `python -m scripts.prepare_mmlu` |
| `plot_cross_evals.py` | Plot cross-eval results across experiments | `python -m scripts.plot_cross_evals` |

## Training Data Format (Flat Schema)

All training records use a flat JSON schema:

```json
{"prompt": "...", "target": "1", "id": "abc-123", "format": "pw", "opponent_model": "qwen-2.5-7b"}
```

Core fields (required by training): `prompt`, `target`, `id`
Optional fields (metadata): `format`, `opponent_model`, `is_control`, `system_prompt`

**Pairwise (PW):** Each ID has exactly 2 records (both response orderings). Train/val splits are done at the ID level to prevent leakage.

**Individual (IND):** Each ID has 1 record.

Targets are always `"1"` or `"2"`.

### Extraction Metadata

Each extracted training data directory includes a `metadata.json`:

```json
{
  "evaluator": "ll-3.1-8b",
  "experiment": "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst",
  "opponent": "qwen-2.5-7b",
  "format": "pw",
  "extraction": {
    "cot": false,
    "train_ratio": 0.8,
    "seed": 42,
    "train_size": 396,
    "val_size": 100,
    "eval_dirs": ["..."]
  }
}
```

## How Training Works

### Architecture

Training uses plain functions + a shared `TinkerContext` dataclass:

```python
# pipeline.py orchestrates the full flow:
def run_training(config):
    prompts = load_prompts(config)
    val_prompts = load_val_prompts(config)
    validate_training_data(prompts, val_prompts)
    ctx = setup_tinker(config)
    run_baseline_eval(ctx, ...)       # Epoch 0 baseline
    train_fn = {"sft": train_sft, "grpo": train_grpo}[config.algorithm]
    global_step = train_fn(config, ctx, prompts, val_prompts)
    save_checkpoint(ctx, config, global_step)
```

### SFT (primary approach for experiments 14-22)
1. Load JSONL prompts and connect to Tinker API
2. For each batch: compute cross-entropy loss on (prompt, target) pairs with assistant-token masking
3. At each epoch boundary: run greedy eval on val set + compute val NLL + run benchmark evals
4. Log metrics to W&B and `train.log`

### GRPO
1. Load JSONL prompts and connect to Tinker API
2. For each batch:
   - Sample `num_rollouts_per_prompt` completions per prompt via Tinker
   - Score each completion with `sgtr_binary_reward` (1.0 if correct, 0.0 otherwise)
   - Compute GRPO advantages: center rewards within each group (prompt)
   - Skip groups where all rollouts gave the same answer (zero signal)
   - Build Tinker `Datum` objects with importance sampling loss inputs
   - Call `forward_backward` + `optim_step`
3. At each epoch boundary: run greedy eval on validation set + compute val NLL

Key insight: GRPO needs **within-group variance** to learn. If all rollouts for a prompt give the same answer, that group provides zero gradient signal.

## Experiments

Each experiment is a YAML file defining model, data, hyperparameters, and evaluation tasks.

## How to Run Things

### Prerequisites

```bash
uv sync                     # Install all dependencies
cp .env.template .env       # Add TINKER_API_KEY
```

### Training

```bash
python -m scripts.train --config experiments/01_sft_pw_vs_qwen/config.yaml

# With CLI overrides
python -m scripts.train --config experiments/01_.../config.yaml \
    --learning_rate 1e-4 --num_epochs 5

# Skip if run already exists
python -m scripts.train --config ... --exists skip
```

### Preparing Data

```bash
# Download from HuggingFace and extract in one step:
python -m scripts.prepare_data --evaluator ll-3.1-8b

# Download MMLU benchmark data:
python -m scripts.prepare_mmlu
```

### Analysis

```bash
python -m scripts.plot_cross_evals
```

## Development

### Environment

```bash
uv sync                    # Install/update dependencies
uv add <package>           # Add a new dependency
uv run ruff check .        # Lint
uv run ruff format .       # Format
```

### Testing

```bash
uv run pytest                # Run all tests
uv run pytest -m datasci     # Data integrity tests only
uv run pytest --co           # Dry-run: confirm test discovery
```

### Experiment Config Schema

```yaml
experiment_name: "descriptive_name"
description: "What this experiment tests"

algorithm: sft               # grpo | sft (default: sft)

data:
  generator_models: [qwen-2.5-7b]       # "Other" model(s) in the SGTR data
  dataset: sharegpt                      # Dataset source (used for plot titles)
  train_file: data/training_data/.../train.jsonl
  val_file: data/training_data/.../val.jsonl
  use_system_prompt: false               # Prepend system_prompt from records

model:
  name: meta-llama/Llama-3.1-8B-Instruct
  lora_rank: 32

wandb_project: sgtr-rl       # Optional: W&B project name (omit to disable)

hyperparameters:
  learning_rate: 5.0e-5
  num_epochs: 20
  batch_size: 16
  max_completion_length: 512
  sampling_temperature: 1.0    # Controls rollout diversity (GRPO only)
  seed: 42

benchmark_evals:
  mmlu_canary:
    type: mmlu                       # "mmlu" | "sgtr"
    data_file: data/benchmarks/mmlu.jsonl
    num_samples: 20                  # Deterministic subsample (omit for all)
    schedule: every_epoch            # "every_epoch" | "every_N_epochs" | "end_only"
    cot: false                       # Chain-of-thought (non-CoT uses max_tokens=128)
  cross_ind_val:
    type: sgtr
    data_file: data/training_data/.../val.jsonl
    schedule: every_5_epochs
    frequency: 5                     # For every_N_epochs schedule
    num_samples: 78
```

`num_samples` subsamples benchmark data deterministically (seed=42).

### MMLU Prompt Format

MMLU prompts use the inspect-ai 0-shot template:
- Choices formatted as `A) text` (not `A. text`)
- Instruction uses `'ANSWER: $LETTER'` format
- Non-CoT: `max_tokens=128` to prevent hidden reasoning
- CoT: full token budget with "Think step by step" instruction

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `tinker` + `tinker-cookbook` | Managed GPU training (LoRA fine-tuning, sampling, checkpoints, ml_log) |
| `self-rec-framework` | SGTR experiment configs, prompts, evaluation tasks |
| `inspect-ai` | LLM evaluation framework (.eval file format) |
| `huggingface-hub` | Dataset downloads from HuggingFace |
| `torch` | Tensor operations for Tinker datum construction |
| `wandb` | Experiment tracking and metric visualization |
