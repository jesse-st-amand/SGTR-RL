# SGTR-RL Codebase Guide

Reference for developers and AI agents working on the SGTR-RL training pipeline.

## What This Repo Does

Trains language models to improve **Self-Generated Text Recognition (SGTR)** using reinforcement learning. The core task: given a piece of text, can a model tell whether it wrote it?

The model (Llama-3.1-8B-Instruct) is trained via **SFT** (supervised fine-tuning) and **GRPO** (Group Relative Policy Optimization) with **LoRA** fine-tuning on a managed GPU platform (**Tinker**). Training data comes from the [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) evaluation pipeline.

## Repository Layout

```
SGTR-RL/
├── sgtr_rl/                      # Main Python package
│   ├── data_processing/          # Data loading and prompt construction
│   ├── training/                 # Training loop, reward, config, logging
│   ├── evaluation/               # Checkpoint evaluation via inspect-ai
│   └── scripts/                  # CLI entry points
├── experiments/                  # Experiment configs (one YAML per experiment)
├── analysis/                     # Log parsing, plotting, and Jupyter notebook
├── data/                         # Data directory (gitignored, see below)
├── results/                      # Training run outputs (gitignored)
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
| `train_config.py` | `TrainingConfig` dataclass + `load_training_config()` YAML parser. All hyperparameters flow through here. Includes `BenchmarkEvalConfig` with `filter_model` support. |
| `reward.py` | `sgtr_binary_reward()` — extracts "1" or "2" from model output, returns 1.0 if correct, 0.0 otherwise. `_extract_answer()` handles both bare digits and "Answer: N" patterns. |
| `run_dir.py` | Creates structured run directories under `results/`. Handles run naming, config freezing, and existing-run policies. |
| `plot_summary.py` | `generate_summary_plot()` — generates a 3-subplot summary figure (loss, accuracy, benchmarks) from `metrics/metrics.jsonl`. Auto-called at end of training. |
| `benchmark_eval.py` | Benchmark evaluation during training. Supports `type: mmlu` (inspect-ai compatible prompt format, max_tokens=16 for non-CoT) and `type: sgtr` (cross-eval with `filter_model` for model-specific filtering, `flip_targets` for label flipping, `num_samples` for subsampling). |
| `logging_setup.py` | Dual logging to terminal + file. |

### `sgtr_rl/scripts/` — CLI Entry Points

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training entry point | `python -m sgtr_rl.scripts.train --config experiments/15_.../config.yaml` |
| `extract_from_eval.py` | Extract training data from `.eval` files | `python -m sgtr_rl.scripts.extract_from_eval --eval_dir data/original/llama8b --output data/training_data/sharegpt_ind/ --format ind` |
| `prepare_data.py` | Build prompts from raw generation data (alternative to extract_from_eval) | `python -m sgtr_rl.scripts.prepare_data --evaluator_model ll-3.1-8b ...` |
| `eval_baseline.py` | Evaluate base model accuracy on any JSONL dataset via Tinker | `python sgtr_rl/scripts/eval_baseline.py --data data/training_data/sharegpt_ind_cot/val.jsonl` |
| `evaluate.py` | Run eval tasks from experiment config on a trained checkpoint | `python -m sgtr_rl.scripts.evaluate --checkpoint path/to/ckpt --config experiments/.../config.yaml` |
| `dry_run.py` | Simulate training locally (no GPU/API) to validate reward/advantage logic | `python sgtr_rl/scripts/dry_run.py --config experiments/.../config.yaml` |
| `prepare_mmlu.py` | Download MMLU from HuggingFace and prepare 20-sample + 500-sample benchmark JSONL files | `python -m sgtr_rl.scripts.prepare_mmlu` |
| `download_hf_data.py` | Download SGTR eval results from HuggingFace, reorganize, and optionally extract training JSONL | `python -m sgtr_rl.scripts.download_hf_data --evaluator ll-3.1-8b --dataset sharegpt --name llama8b --extract` |
| `plot_cross_evals.py` | Plot cross-eval results across experiments 15-22 (overview, heatmaps, format comparisons) | `python -m sgtr_rl.scripts.plot_cross_evals` |

### `sgtr_rl/data_processing/` — Data Preparation

| File | Purpose |
|------|---------|
| `prompt_builder.py` | Build SGTR prompts from raw generation data via self-rec-framework |
| `validate_data.py` | Validate train/val JSONL integrity: schema, target values, UUID overlap, PW format checks |

### `sgtr_rl/evaluation/` — Checkpoint Evaluation

| File | Purpose |
|------|---------|
| `evaluator.py` | `evaluate_checkpoint()` — runs SGTR and standard inspect-ai benchmarks via `get_model_str()`. |

### `analysis/` — Log Parsing and Visualization

| File | Purpose |
|------|---------|
| `utils.py` | Parse `train.log` files into structured `RunData` objects. `load_run()`, `list_runs()`. |
| `plotting.py` | Matplotlib plots: dashboard, accuracy, reward, datums, timing. |
| `plot_run.py` | CLI to generate all plots for a run: `python -m analysis.plot_run results/my_run/` |
| `analysis.ipynb` | Interactive Jupyter notebook for analysis. |

### `experiments/` — Experiment Configs

Each experiment is a YAML file defining model, data, hyperparameters, and evaluation tasks.

| Experiment | Description | Status |
|-----------|-------------|--------|
| `01_RL_grpo_IND_WikiSum` | WikiSum individual recognition (GRPO, local) | Early prototype |
| `02_RL_grpo_IND_ShareGPT` | ShareGPT individual recognition (GRPO, local) | Early prototype |
| `03_RL_grpo_IND_ShareGPT_CoT` | ShareGPT with chain-of-thought prompts (GRPO, tinker) | Early prototype |
| `04–13` | Debug/sanity experiments (overfit, trivial tasks, format checks) | Debug |
| `14_sft_pw_uuid_split` | SFT on PW with UUID-level split (160 train, 40 val) | Completed |
| `15_sft_pw_rec_vs_qwen` | SFT PW recognition, Llama vs Qwen (158 train, 38 val) | Completed |
| `16_sft_ind_rec_vs_qwen` | SFT IND recognition, Llama vs Qwen | Completed |
| `17_sft_pw_rec_flipped_vs_qwen` | SFT PW recognition flipped (anti-self), Llama vs Qwen | Completed |
| `18_sft_ind_rec_flipped_vs_qwen` | SFT IND recognition flipped (anti-self), Llama vs Qwen | Completed |
| `19_sft_pw_rec_vs_haiku` | SFT PW recognition, Llama vs Haiku-3.5 | Completed |
| `20_sft_pw_rec_vs_gpt4o` | SFT PW recognition, Llama vs GPT-4o | Completed |
| `21_sft_pw_rec_vs_ll70b` | SFT PW recognition, Llama vs Llama-3.1-70B | Completed |
| `22_sft_pw_rec_vs_opus` | SFT PW recognition, Llama vs Claude Opus | Completed |

Experiments 15-22 share a common structure: they train Llama-3.1-8B on SGTR with cross-domain and cross-format evals (MMLU, cross-format SGTR, wikisum, pku, bigcode benchmarks). Experiments 15-18 compare formats (PW vs IND) and self vs anti-self against Qwen. Experiments 19-22 vary the "other" model.

### `data/` — Data Directory (gitignored)

```
data/
├── original/                         # Source .eval files (inspect-ai zip archives)
│   └── llama8b/                      # Llama-3.1-8B eval data
├── training_data/                    # JSONL datasets for training
│   ├── llama8b_pw_rec_qwen_pw/       # PW recognition, Llama vs Qwen
│   ├── llama8b_ind_rec_qwen_ind/     # IND recognition, Llama vs Qwen
│   ├── llama8b_pw_pref_pw/           # PW preference
│   ├── llama8b_ind_pref_ind/         # IND preference
│   ├── xeval_wikisum_pw_rec_pw/      # Cross-domain: WikiSum PW rec
│   ├── xeval_pku_pw_rec_pw/          # Cross-domain: PKU SafeRLHF PW rec
│   ├── xeval_bigcode_pw_rec_pw/      # Cross-domain: BigCodeBench PW rec
│   ├── sharegpt_ind/                 # Individual format (legacy)
│   ├── sharegpt_ind_cot/             # Individual + CoT (legacy)
│   └── sharegpt_pw/                  # Pairwise format (legacy, exp 14)
├── benchmarks/                       # Benchmark data
│   ├── mmlu_20.jsonl                 # MMLU 20-sample canary
│   └── mmlu_500.jsonl                # MMLU 500-sample evaluation
├── checkpoints/                      # Model checkpoints
└── README.md
```

### `results/` — Training Run Outputs (gitignored)

Each run creates a directory under `results/`:
```
results/{experiment_name}__{timestamp}/
├── config.yaml              # Frozen config (including CLI overrides)
├── train.log                # Full training log
├── summary.png              # Auto-generated summary figure
├── metrics/                 # W&B and JSON metric logs
├── val_predictions/         # Per-epoch val predictions (epoch_0.json, ...)
├── benchmark_predictions/   # Per-epoch benchmark predictions
└── checkpoints/             # Saved model checkpoints
```

Cross-eval batch results are in `results/batch_15-22_sft_cross_evals/`.

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

## How Training Works (TinkerSFTTrainer / TinkerRLTrainer)

### SFT (primary approach for experiments 14-22)
1. Load JSONL prompts and connect to Tinker API
2. For each batch of prompts: compute cross-entropy loss on (prompt, target) pairs with assistant-token masking
3. At each epoch boundary: run greedy eval on val set + compute val NLL + run benchmark evals
4. Log metrics to W&B and `train.log`

### GRPO
1. Load JSONL prompts and connect to Tinker API
2. For each batch of prompts:
   - Sample `num_rollouts_per_prompt` completions per prompt via Tinker
   - Score each completion with `sgtr_binary_reward` (1.0 if correct, 0.0 otherwise)
   - Compute GRPO advantages: center rewards within each group (prompt)
   - Skip groups where all rollouts gave the same answer (zero signal)
   - Build Tinker `Datum` objects with importance sampling loss inputs
   - Call `forward_backward` + `optim_step`
3. At each epoch boundary: run greedy eval on validation set + compute val NLL

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
python -m sgtr_rl.scripts.train --config experiments/15_sft_pw_rec_vs_qwen/config.yaml

# With CLI overrides
python -m sgtr_rl.scripts.train --config experiments/15_.../config.yaml \
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

# Download from HuggingFace and extract in one step:
python -m sgtr_rl.scripts.download_hf_data \
    --evaluator ll-3.1-8b --dataset sharegpt --name llama8b --extract
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

# Cross-eval analysis across experiments 15-22
python -m sgtr_rl.scripts.plot_cross_evals

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

### Testing

```bash
uv run pytest                # Run all tests
uv run pytest -m datasci     # Data integrity tests only
uv run pytest --co           # Dry-run: confirm test discovery
```

Tests live in `tests/` and cover:

| File | What it tests |
|------|--------------|
| `test_validate_data.py` | Data validation — UUID leakage, schema, targets, PW ordering |
| `test_reward.py` | Answer extraction and binary reward |
| `test_benchmark_eval.py` | MMLU prompt formatting (inspect-ai format), answer extraction, schedule logic, model filtering |
| `test_run_dir.py` | Run naming, override computation, directory creation |
| `test_train_config.py` | YAML config loading, defaults, filter_model resolution |
| `test_plot_summary.py` | Title building, smoothing, summary plot generation |
| `test_data_integrity.py` | Validates actual data files on disk (marked `@datasci`) |
| `test_download_hf_data.py` | HF data download: filename parsing, format detection, filtering |
| `integration/test_sft_lowlevel.py` | SFT pipeline: cross-entropy loss, accuracy threshold, eval schedule, flip_targets, batching, checkpoints |
| `integration/test_grpo_lowlevel.py` | GRPO pipeline: importance sampling loss, advantage centering, zero-signal skipping, reward wiring, datum construction |
| `integration/test_highlevel.py` | Loop structure for both trainers: eval/benchmark schedules, step counts, sampling order |

Integration tests mock Tinker at the `sys.modules` level (see `integration/conftest.py`) so the full training loop runs without GPU access.

### Adding a New Experiment

1. Create `experiments/NN_description/config.yaml` (copy from experiment 15 as a template)
2. Prepare training data if needed (see above)
3. Run: `python -m sgtr_rl.scripts.train --config experiments/NN_.../config.yaml`

### Experiment Config Schema

```yaml
experiment_name: "descriptive_name"
description: "What this experiment tests"

algorithm: sft               # grpo | sft
backend: tinker              # tinker | local

data:
  evaluator_model: ll-3.1-8b            # Short name of the model being trained
  generator_models: [qwen-2.5-7b]       # "Other" model(s) in the SGTR data
  dataset: sharegpt                      # Dataset source
  subsets: [english_26, english2_74]     # Dataset subsets
  train_file: data/training_data/.../train.jsonl
  val_file: data/training_data/.../val.jsonl
  flip_targets: false        # Swap "1"<->"2" in training/val data (anti-self)

model:
  name: meta-llama/Llama-3.1-8B-Instruct
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.05

wandb_project: sgtr-rl       # Optional: W&B project name (omit to disable)

hyperparameters:
  learning_rate: 5.0e-5
  num_epochs: 20
  per_device_train_batch_size: 16
  max_completion_length: 512
  sampling_temperature: 1.0    # Controls rollout diversity (GRPO only)
  seed: 42

checkpointing:
  save_steps: 20
  eval_steps: 20

benchmark_evals:
  mmlu_canary:
    type: mmlu                       # "mmlu" | "sgtr"
    data_file: data/benchmarks/mmlu_500.jsonl
    num_samples: 20                  # Deterministic subsample (omit for all)
    schedule: every_epoch            # "every_epoch" | "every_N_epochs" | "end_only"
    cot: false                       # Chain-of-thought (non-CoT uses max_tokens=16)
  cross_ind_val:
    type: sgtr
    data_file: data/training_data/llama8b_ind_rec_qwen_ind/val.jsonl
    schedule: every_5_epochs
    frequency: 5                     # For every_N_epochs schedule
    filter_model: auto               # Filter to this "other" model ("auto" = from generator_models)
    flip_targets: false              # Swap "1"<->"2" at eval time
    num_samples: 78
```

#### Cross-Eval and Label Flipping

`flip_targets` swaps target labels "1" and "2" at runtime without duplicating data files:
- In `data:` section: flips training and validation targets (for anti-self experiments)
- In `benchmark_evals:`: flips comparison targets at eval time

`num_samples` subsamples benchmark data deterministically (seed=42).

#### filter_model

`filter_model` filters SGTR benchmark data to only samples involving a specific "other" model. Without this, cross-eval data files that contain multiple generator models would evaluate on a mixture.
- `filter_model: auto` resolves to the single model in `data.generator_models` (requires exactly one entry)
- `filter_model: "gpt-4o"` filters explicitly to that model
- Omit for MMLU or single-model data files

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
| `trl` + `peft` + `transformers` | Local GPU training path (HuggingFace stack) |
| `torch` | Tensor operations for Tinker datum construction |
| `wandb` | Experiment tracking and metric visualization |

## Known Issues / Technical Debt

- `PROJECT_SETUP.md` and `SETUP_COMPLETE.md` are stale planning documents from early project phases
- `data/README.md` references DPO-era terminology and paths
- **Small dataset**: Current PW dataset is ~160 train / ~40 val unique pairs per experiment. SFT reaches 90%+ val accuracy but more data would improve robustness
- Experiments 04-13 are debug/sanity configs that could be archived
- `evaluator.py` has not been used in the recent experiment runs (benchmark evals are handled by `benchmark_eval.py` within the training loop)
