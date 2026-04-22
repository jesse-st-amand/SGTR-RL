# SGTR-RL Codebase Guide

Reference for developers and AI agents working on the SGTR-RL training pipeline.

## What This Repo Does

Trains language models to improve **Self-Generated Text Recognition (SGTR)** using reinforcement learning. The core task: given a piece of text, can a model tell whether it wrote it?

The model (Llama-3.1-8B-Instruct) is trained via **SFT** (supervised fine-tuning) and **GRPO** (Group Relative Policy Optimization) with **LoRA** fine-tuning. There are now two runtime paths:

- **Tinker** for the original managed-GPU workflow
- **Local HF/PEFT** for single-node GPU training, including RunPod-hosted runs

Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline.

## Repository Layout

```
SGTR-RL/
├── sgtr_rl/                      # Training/runtime Python package
│   ├── config.py                 # TrainingConfig + experiment YAML loader
│   ├── runtime_config.py         # RuntimeConfig + runtime YAML loader
│   ├── artifacts.py              # status.json + JSONL metric helpers
│   ├── answer.py                 # extract_answer (paradigm-agnostic)
│   ├── reward.py                 # sgtr_binary_reward (uses answer.py)
│   ├── tinker.py                 # TinkerContext dataclass + setup_tinker()
│   ├── pipeline.py               # run_training() orchestration + backend dispatch
│   ├── sft.py                    # Tinker SFT function
│   ├── local_sft.py              # Local HF/PEFT SFT function + checkpoint save
│   ├── local_eval.py             # Local validation + benchmark evaluation
│   ├── grpo.py                   # train_grpo() function
│   ├── data.py                   # load_jsonl, validate_training_data, build_conversation
│   ├── tinker_eval.py            # Tinker-based eval (val accuracy/NLL, MMLU, SGTR cross-eval)
│   ├── benchmarks.py             # Pure benchmark logic (prompt formatting, scheduling)
│   ├── metrics.py                # Metric logging and prediction saving
│   ├── runs.py                   # Run directory creation
│   ├── logging_setup.py          # Dual logging setup
│   └── __init__.py               # Package marker
├── scripts/                      # Top-level CLI entry points
│   ├── train.py                  # Main training entry point
│   ├── runpod_launch.py          # Create/poll/delete one-shot RunPod jobs
│   ├── runpod_utils.py           # RunPod payload/startup-script helpers
│   ├── prepare_data.py           # Download + extract training data
│   ├── prepare_mmlu.py           # Prepare MMLU benchmark data
│   ├── plot_cross_evals.py       # Cross-eval analysis plots
│   ├── plotting_utils.py         # Shared plotting helpers
│   └── plot_summary.py           # Per-run summary charts for SFT results
├── experiments/                  # Experiment configs (one YAML per experiment)
├── runtimes/                     # Runtime configs (backend, artifact paths, RunPod)
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
| `config.py` | `TrainingConfig` dataclass + `load_training_config()` YAML parser. All experiment hyperparameters flow through here. Includes `BenchmarkEvalConfig`. |
| `runtime_config.py` | `RuntimeConfig` + `load_runtime_config()` for backend choice, artifact paths, local GPU settings, and RunPod launch settings. |
| `artifacts.py` | `status.json` writer, atomic JSON writes, append-only JSONL metrics logger for local runs. |
| `answer.py` | `extract_answer()` — extracts "1" or "2" from model output text. Handles both bare digits and "Answer: N" patterns. |
| `reward.py` | `sgtr_binary_reward()` — extracts answer from model output, returns 1.0 if correct, 0.0 otherwise. |
| `tinker.py` | `TinkerContext` dataclass (shared Tinker state) + `setup_tinker()` (creates ServiceClient, training client, tokenizer, renderer, params) + `save_checkpoint()`. |
| `pipeline.py` | `run_training()` — full pipeline orchestration: load data → validate → dispatch to the selected backend → update run status. |
| `sft.py` | `train_sft(config, ctx, prompts, val_prompts)` — Tinker SFT loop. |
| `local_sft.py` | `train_local_sft(config, runtime, prompts, val_prompts)` — local HF/PEFT LoRA SFT loop with final adapter checkpoint save. |
| `local_eval.py` | Local-model validation, SGTR cross-evals, MMLU evals, and metric/prediction saving. |
| `grpo.py` | `train_grpo(config, ctx, prompts, val_prompts)` — GRPO training loop: sample rollouts, compute rewards, center advantages within groups, build Tinker datums, call `forward_backward` + `optim_step`. |
| `data.py` | `load_jsonl()`, `validate_training_data()`, `build_conversation()` — data loading, integrity validation, and conversation construction. |
| `tinker_eval.py` | Tinker-based evaluation. Orchestrators `run_val_eval(prompts, ctx, ...)` and `run_benchmark_evals(configs, ctx, ...)` accept `TinkerContext`. Lower-level: `evaluate_val()`, `compute_val_nll()`, `evaluate_benchmark()` (MMLU), `evaluate_sgtr_benchmark()` (cross-eval). |
| `benchmarks.py` | Pure benchmark logic (no Tinker deps). `format_mmlu_prompt()`, `extract_mmlu_answer()`, `should_run_benchmark()`, `_subsample()`, `load_benchmark_data()`. |
| `metrics.py` | Metric logging and prediction saving. `log_val_result()`, `log_val_metrics()`, `save_val_predictions()`. |
| `runs.py` | Creates structured run directories under `results/`. Handles run naming, config freezing, and existing-run policies. |
| `logging_setup.py` | Dual logging to terminal + file. |

## Scripts (`scripts/`)

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training entry point | `python -m scripts.train --config experiments/01_.../config.yaml` |
| `runpod_launch.py` | Launch one-shot RunPod jobs for the local backend | `python -m scripts.runpod_launch --config ... --runtime runtimes/runpod_a100.yaml` |
| `runpod_utils.py` | Internal RunPod request/startup-script builder | Imported by `scripts.runpod_launch` |
| `prepare_data.py` | Download from HuggingFace + extract training data | `python -m scripts.prepare_data --evaluator ll-3.1-8b` |
| `prepare_mmlu.py` | Download MMLU and prepare benchmark JSONL | `python -m scripts.prepare_mmlu` |
| `run_sanity_sweeps.py` | Launch small train-size / label-randomization sanity sweeps from a single base config | `python -m scripts.run_sanity_sweeps --dry-run` |
| `plot_cross_evals.py` | Plot cross-eval results across experiments | `python -m scripts.plot_cross_evals` |
| `plotting_utils.py` | Shared helpers for training summary plots | Imported by `scripts.train` |
| `plot_summary.py` | Plot per-run SFT summaries for selected experiment outputs | `python -m scripts.plot_summary` |

## Training Data Format (Flat Schema)

All training records use a flat JSON schema:

```json
{"prompt": "...", "target": "1", "id": "abc-123", "format": "pw", "opponent_model": "qwen-2.5-7b", "dataset": "sharegpt", "data_subset": "english_26"}
```

Core fields (required by training): `prompt`, `target`, `id`
Optional fields (metadata): `format`, `opponent_model`, `is_control`, `system_prompt`, `dataset`, `data_subset`

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
  "dataset": "sharegpt",
  "format": "pw",
  "extraction": {
    "cot": false,
    "train_ratio": 0.8,
    "seed": 42,
    "train_size": 160,
    "val_size": 40,
    "eval_dirs": ["..."]
  }
}
```

## How Training Works

### Architecture

Training uses plain functions with backend-specific runtime state:

```python
# pipeline.py orchestrates the full flow:
def run_training(config, runtime):
    prompts = load_prompts(config)
    val_prompts = load_val_prompts(config)
    validate_training_data(prompts, val_prompts)
    if runtime.backend == "tinker":
        ...
    elif runtime.backend == "local":
        ...
```

### SFT on Tinker
1. Load JSONL prompts and connect to Tinker API
2. For each batch: compute cross-entropy loss on (prompt, target) pairs with assistant-token masking
3. At each epoch boundary: run greedy eval on val set + compute val NLL + run benchmark evals
4. Log metrics to W&B and `train.log`

### SFT on local GPU / RunPod
1. Load tokenizer/model with `transformers`, apply LoRA with `peft`
2. Build masked chat-template training examples so only assistant target tokens contribute to loss
3. Train with a plain PyTorch loop and evaluate at epoch boundaries
4. Write metrics incrementally to `metrics/metrics.jsonl`, update `status.json`, and save a final adapter checkpoint under `checkpoints/final/`

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

### Handy Sweep Knobs

For sanity checks and controls, experiment YAMLs and `scripts.train` now support a few train-data transforms:

- `data.max_train_ids`: sample a deterministic subset of unique training IDs before training
- `data.subset_seed`: separate seed for that ID-level subset
- `data.randomize_train_labels`: flip binary labels by training ID as a control
- `data.randomize_train_labels_seed`: separate seed for label randomization
- `hyperparameters.max_steps`: optional early stop after a fixed number of optimizer steps
- `evaluation.trigger`: run evals on `epoch` boundaries or by optimizer `step`
- `evaluation.frequency`: run evals every N epochs or N steps, always including the final point

These are intended for quick size sweeps and negative controls without cloning lots of YAML configs.

## Experiments

Each experiment is a YAML file defining model, data, hyperparameters, and evaluation tasks.

Machine/provider-specific settings live separately in `runtimes/*.yaml`.

## How to Run Things

### Prerequisites

```bash
uv sync                     # Install all dependencies
cp .env.template .env       # Add provider tokens as needed
```

### Training

```bash
python -m scripts.train --config experiments/01_sft_pw_vs_qwen/config.yaml

# Local GPU
python -m scripts.train \
    --config experiments/01_sft_pw_vs_qwen/config.yaml \
    --runtime runtimes/local_gpu.yaml

# With CLI overrides
python -m scripts.train --config experiments/01_.../config.yaml \
    --learning_rate 1e-4 --num_epochs 5

# Skip if run already exists
python -m scripts.train --config ... --exists skip
```

```bash
# RunPod
python -m scripts.runpod_launch \
    --config experiments/01_sft_pw_vs_qwen/config.yaml \
    --runtime runtimes/runpod_a100.yaml
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
python -m scripts.plot_summary
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

### Runtime Config Schema

```yaml
backend: local                # tinker | local

artifacts:
  root_dir: /runpod-volume/sgtr-rl/results

local:
  device: cuda                # auto | cuda | cpu
  dtype: bfloat16             # auto | bfloat16 | float16 | float32
  max_seq_length: 4096
  eval_batch_size: 8
  gradient_checkpointing: true
  load_in_4bit: false
  attention_implementation: sdpa
  cache_dir: /runpod-volume/sgtr-rl/hf-cache

runpod:
  image_name: runpod/pytorch:...
  gpu_type_ids: [NVIDIA A100 80GB PCIe]
  network_volume_id: CHANGEME
  volume_mount_path: /runpod-volume
  env_passthrough: [HF_TOKEN, WANDB_API_KEY]
```

For v1, the local backend supports **SFT only**. GRPO remains Tinker-only.

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
