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
│   ├── data.py                   # load_jsonl, flip_target, validate_training_data
│   ├── eval.py                   # Val eval (accuracy + NLL)
│   ├── benchmarks.py             # MMLU + SGTR cross-eval
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
| `data.py` | `load_jsonl()`, `flip_target()`, `validate_training_data()` — data loading and integrity validation (schema, targets, ID overlap, PW ordering). |
| `eval.py` | Shared validation evaluation logic. `evaluate_val()` (greedy accuracy), `compute_val_nll()` (forward-pass NLL), `run_val_eval()`. |
| `benchmarks.py` | Benchmark evaluation during training. `type: mmlu` (inspect-ai compatible prompt format) and `type: sgtr` (cross-eval with `flip_targets`, `num_samples`). |
| `runs.py` | Creates structured run directories under `results/`. Handles run naming, config freezing, and existing-run policies. |
| `plotting.py` | `generate_summary_plot()` — 3-subplot summary figure (loss, accuracy, benchmarks) from `metrics/metrics.jsonl`. |
| `logging_setup.py` | Dual logging to terminal + file. |

## Scripts (`scripts/`)

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training entry point | `python -m scripts.train --config experiments/15_.../config.yaml` |
| `prepare_data.py` | Download from HuggingFace + extract training data | `python -m scripts.prepare_data --evaluator ll-3.1-8b` |
| `prepare_mmlu.py` | Download MMLU and prepare benchmark JSONL files | `python -m scripts.prepare_mmlu` |
| `plot_cross_evals.py` | Plot cross-eval results across experiments | `python -m scripts.plot_cross_evals` |

## Training Data Format (Flat Schema)

All training records use a flat JSON schema:

```json
{"prompt": "...", "target": "1", "id": "abc-123", "format": "pw", "opponent_model": "qwen-2.5-7b"}
```

Core fields (required by training): `prompt`, `target`, `id`
Optional fields (used by benchmark filtering): `format`, `opponent_model`, `is_control`

Field names are configurable in TrainingConfig via `prompt_field`, `target_field`, `id_field`.

**Pairwise (PW):** Each ID has exactly 2 records (both response orderings). Train/val splits are done at the ID level to prevent leakage.

**Individual (IND):** Each ID has 1 record.

Targets are always `"1"` or `"2"`.

## How Training Works

### Architecture

Training uses plain functions + a shared `TinkerContext` dataclass:

```python
# pipeline.py orchestrates the full flow:
def run_training(config):
    prompts = load_prompts(config)
    val_prompts = load_val_prompts(config)
    validate_training_data(config.train_file, config.val_file)
    ctx = setup_tinker(config)        # TinkerContext with training_client, renderer, etc.
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

| Experiment | Description | Status |
|-----------|-------------|--------|
| `14_sft_pw_uuid_split` | SFT on PW with UUID-level split (160 train, 40 val) | Completed |
| `15_sft_pw_rec_vs_qwen` | SFT PW recognition, Llama vs Qwen (158 train, 38 val) | Completed |
| `16_sft_ind_rec_vs_qwen` | SFT IND recognition, Llama vs Qwen | Completed |
| `17_sft_pw_rec_flipped_vs_qwen` | SFT PW recognition flipped (anti-self), Llama vs Qwen | Completed |
| `18_sft_ind_rec_flipped_vs_qwen` | SFT IND recognition flipped (anti-self), Llama vs Qwen | Completed |
| `19_sft_pw_rec_vs_haiku` | SFT PW recognition, Llama vs Haiku-3.5 | Completed |
| `20_sft_pw_rec_vs_gpt4o` | SFT PW recognition, Llama vs GPT-4o | Completed |
| `21_sft_pw_rec_vs_ll70b` | SFT PW recognition, Llama vs Llama-3.1-70B | Completed |
| `22_sft_pw_rec_vs_opus` | SFT PW recognition, Llama vs Claude Opus | Completed |

Experiments 15-22 share a common structure: they train Llama-3.1-8B on SGTR with cross-domain and cross-format evals (MMLU, cross-format SGTR, wikisum, pku, bigcode benchmarks).

## How to Run Things

### Prerequisites

```bash
uv sync                     # Install all dependencies
cp .env.template .env       # Add TINKER_API_KEY
```

### Training

```bash
# Run an experiment
python -m scripts.train --config experiments/15_sft_pw_rec_vs_qwen/config.yaml

# With CLI overrides
python -m scripts.train --config experiments/15_.../config.yaml \
    --learning_rate 1e-4 --num_epochs 5

# Skip if run already exists
python -m scripts.train --config ... --exists skip
```

### Preparing Data

```bash
# Download from HuggingFace and extract in one step:
python -m scripts.prepare_data --evaluator ll-3.1-8b
```

### Analysis

```bash
# Cross-eval analysis across experiments
python -m scripts.plot_cross_evals

# W&B (metrics are logged automatically during training)
```

## Development

### Environment

```bash
uv sync                    # Install/update dependencies
uv add <package>           # Add a new dependency (updates pyproject.toml + uv.lock)
uv run ruff check .        # Lint
uv run ruff format .       # Format
```

### Testing

```bash
uv run pytest                # Run all tests
uv run pytest -m datasci     # Data integrity tests only
uv run pytest --co           # Dry-run: confirm test discovery
```

Tests live in `tests/` and cover:

| File | What it tests |
|------|--------------|
| `test_validate_data.py` | Data validation — ID leakage, schema, targets, PW ordering |
| `test_reward.py` | Answer extraction and binary reward |
| `test_benchmark_eval.py` | MMLU prompt formatting, answer extraction, schedule logic |
| `test_run_dir.py` | Run naming, override computation, directory creation |
| `test_train_config.py` | YAML config loading, defaults |
| `test_plot_summary.py` | Title building, smoothing, summary plot generation |
| `test_data_integrity.py` | Validates actual data files on disk (marked `@datasci`) |
| `test_download_hf_data.py` | HF data download: filename parsing, format detection, filtering |
| `integration/test_sft_lowlevel.py` | SFT pipeline: cross-entropy loss, accuracy threshold, eval schedule, batching |
| `integration/test_grpo_lowlevel.py` | GRPO pipeline: importance sampling loss, advantage centering, zero-signal skipping, reward wiring, datum construction |
| `integration/test_highlevel.py` | Loop structure for both trainers: eval/benchmark schedules, step counts, sampling order |

Integration tests mock Tinker at the `sys.modules` level (see `integration/conftest.py`) so the full training loop runs without GPU access.

### Experiment Config Schema

```yaml
experiment_name: "descriptive_name"
description: "What this experiment tests"

algorithm: sft               # grpo | sft

data:
  evaluator_model: ll-3.1-8b            # Short name of the model being trained
  generator_models: [qwen-2.5-7b]       # "Other" model(s) in the SGTR data
  dataset: sharegpt                      # Dataset source
  subsets: [english_26, english2_74]     # Dataset subsets
  train_file: data/training_data/.../train.jsonl
  val_file: data/training_data/.../val.jsonl

model:
  name: meta-llama/Llama-3.1-8B-Instruct
  lora_rank: 32

wandb_project: sgtr-rl       # Optional: W&B project name (omit to disable)

hyperparameters:
  learning_rate: 5.0e-5
  num_epochs: 20
  per_device_train_batch_size: 16
  max_completion_length: 512
  sampling_temperature: 1.0    # Controls rollout diversity (GRPO only)
  seed: 42

benchmark_evals:
  mmlu_canary:
    type: mmlu                       # "mmlu" | "sgtr"
    data_file: data/benchmarks/mmlu_500.jsonl
    num_samples: 20                  # Deterministic subsample (omit for all)
    schedule: every_epoch            # "every_epoch" | "every_N_epochs" | "end_only"
    cot: false                       # Chain-of-thought (non-CoT uses max_tokens=16)
  cross_ind_val:
    type: sgtr
    data_file: data/training_data/ll-3.1-8b_ICML_02_UT_IND-Q_Rec_NPr_FA_Inst_vs_qwen-2.5-7b-treatment/val.jsonl
    schedule: every_5_epochs
    frequency: 5                     # For every_N_epochs schedule
    flip_targets: false              # Swap "1"<->"2" at eval time
    num_samples: 78
```

#### Cross-Eval and Label Flipping

`flip_targets` in `benchmark_evals` swaps target labels "1" and "2" at eval time — useful for cross-evaluation where the label mapping differs between datasets.

`num_samples` subsamples benchmark data deterministically (seed=42).

#### MMLU Prompt Format

MMLU prompts use the inspect-ai 0-shot template:
- Choices formatted as `A) text` (not `A. text`)
- Instruction uses `'ANSWER: $LETTER'` format
- Non-CoT: `max_tokens=16` to prevent hidden reasoning
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
