# SGTR-RL

**Reinforcement Learning for Self-Generated Text Recognition**

Train language models to improve self-generated text recognition (SGTR) using RL, and test whether improvements generalize across experimental operationalizations (e.g., train on Individual, test on Pairwise) and task domains.

## Quick Start

```bash
git clone https://github.com/jesse-st-amand/SGTR-RL.git
cd SGTR-RL
uv sync
cp .env.template .env  # fill in TINKER_API_KEY
```

Run training:
```bash
python sgtr_rl/scripts/train.py --config experiments/02_RL_grpo_IND_ShareGPT/config.yaml
```

## How It Works

The pipeline has three stages:

**1. Prepare data** -- extract SGTR prompts from eval files or build from raw generation data:
```bash
# From .eval files (recommended if you have them):
python sgtr_rl/scripts/extract_from_eval.py \
    --eval_dir /path/to/eval_data \
    --output data/training_data/sharegpt_ind/ \
    --format ind

# From raw generation data:
python sgtr_rl/scripts/prepare_data.py \
    --evaluator_model ll-3.1-8b \
    --generator_models ll-3.1-8b qwen-2.5-7b \
    --dataset sharegpt --subset english2_74 \
    --experiment_config experiments/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/config.yaml \
    --output data/training_data/sharegpt_ind/
```

**2. Train** -- GRPO with LoRA, on Tinker (managed GPU) or local GPU:
```bash
python sgtr_rl/scripts/train.py \
    --config experiments/02_RL_grpo_IND_ShareGPT/config.yaml
```

**3. Evaluate** -- run SGTR + auxiliary evals on the trained checkpoint:
```bash
python sgtr_rl/scripts/evaluate.py \
    --checkpoint data/checkpoints/02_RL_grpo_IND_ShareGPT/checkpoint-final \
    --config experiments/02_RL_grpo_IND_ShareGPT/config.yaml
```

## Architecture

Three independent dimensions, composed via a single YAML config:

| Dimension          | Options                                            |
|--------------------|----------------------------------------------------|
| Training algorithm | `grpo` (implemented) \| `dpo`, `sft` (planned)    |
| Training backend   | `tinker` (implemented) \| `local` (TRL, implemented) |
| Eval backend       | `hf` \| `vllm` \| `together` \| any inspect-ai provider |

New experiments only need a config file -- see `experiments/02_RL_grpo_IND_ShareGPT/config.yaml`.

## Project Structure

```
SGTR-RL/
├── sgtr_rl/
│   ├── config/paths.py                 # Data path resolution
│   ├── data_processing/
│   │   ├── eval_loader.py              # Parse .eval files (for DPO)
│   │   ├── triple_generator.py         # DPO triple creation (planned)
│   │   └── prompt_builder.py           # Build SGTR prompts from raw data
│   ├── training/
│   │   ├── train_config.py             # TrainingConfig dataclass + YAML loader
│   │   ├── reward.py                   # Binary reward (extract "1"/"2", compare to target)
│   │   ├── grpo_trainer.py             # LocalGRPOTrainer (TRL) + TinkerRLTrainer
│   │   └── logging_setup.py            # Dual logging (terminal + file)
│   ├── evaluation/
│   │   └── evaluator.py                # evaluate_checkpoint, EvalCallback
│   └── scripts/
│       ├── train.py                    # CLI: dispatch to trainer
│       ├── prepare_data.py             # CLI: build prompts from raw generation data
│       ├── extract_from_eval.py        # CLI: extract training data from .eval files
│       └── evaluate.py                 # CLI: run eval tasks from config
├── experiments/
│   ├── 01_RL_grpo_IND_WikiSum/         # WikiSum experiment config
│   └── 02_RL_grpo_IND_ShareGPT/       # ShareGPT de-risking experiment config
├── data/                               # Data (gitignored)
│   ├── training_data/                  # JSONL prompt datasets
│   ├── checkpoints/                    # Model checkpoints
│   └── results/                        # Eval results
├── logs/                               # Training logs (gitignored)
└── _external/                          # Git submodules (for reference only)
    ├── self-rec-framework/
    ├── tinker-cookbook/
    └── inspect_ai/
```

## Training Data Format

Training JSONL (`data/training_data/*/train.jsonl`):
```json
{"prompt": "Below is a user request followed by a response...", "target": "1", "metadata": {...}}
```

Each prompt is a complete SGTR question with model-generated text embedded. The target is `"1"` or `"2"` -- the correct answer to the recognition question.

## Dependencies

| Package | Purpose |
|---------|---------|
| [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) | SGTR experiment configs, prompts, evaluation tasks |
| [tinker](https://tinker-docs.thinkingmachines.ai) | Managed GPU training (LoRA fine-tuning via API) |
| [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | Renderers, tokenizers, checkpoint utils for Tinker |
| [inspect-ai](https://inspect.aisi.org.uk/) | LLM evaluation framework |

All installed automatically by `uv sync`. The `_external/` submodules are for reference only.

## Environment Variables

Copy `.env.template` to `.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | For Tinker backend | Managed GPU training |
| `TOGETHER_API_KEY` | For Together eval backend | Model inference for evaluation |

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run ruff check .
uv run ruff format .
```
