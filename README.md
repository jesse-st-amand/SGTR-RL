# SGTR-RL

**Reinforcement Learning for Self-Generated Text Recognition**

This project builds a reinforcement learning (RL) pipeline to train language models on Self-Generated Text Recognition (SGTR) tasks. The goal is to test whether SGTR improvements generalize across different experimental operationalizations, providing evidence for shared representations of self-recognition capabilities.

## Project Background

This work extends the research presented in **"Self-Generated Text Recognition: A Highly Context-Dependent Emergent Capability"** (submitted to ICLR 2026). The paper demonstrated that some frontier models can recognize their own outputs, but performance varies greatly across experimental conditions (pairwise vs individual assessment, different task domains, etc.).

**Key Research Questions**:
1. Can we train models to improve SGTR performance?
2. Do improvements generalize across operationalizations (e.g., train on Individual → test on Pairwise)?
3. Do improvements generalize across task domains (e.g., train on WikiSum → test on BigCodeBench)?
4. Does this provide evidence for shared "self-recognition" representations vs task-specific heuristics?

## Experimental Approach

### Training Protocol

**RL Method**: Direct Preference Optimization (DPO) with Chain-of-Thought reasoning
- Models generate CoT reasoning before final binary answers
- Preserves general model capabilities
- Final answer determines reward signal (correct vs incorrect)
- More interpretable than forcing immediate binary outputs

**Three-Phase Training**:
1. **Phase 1**: Train on pre-cached model generations (control for distribution shift)
2. **Phase 2-3**: Train on live-generated text (test on real model outputs)

**Regularization**: Interleave MMLU questions during training to maintain general reasoning capabilities

**Monitoring**: Periodic evaluation on standardized benchmarks (Inspect suite) to detect "model cooking"

### Initial Experiments

**Experiment 1: Train IND → Test PW**
- Train on Individual recognition (IND) for one dataset (e.g., WikiSum)
- Evaluate on:
  - IND same domain (training distribution)
  - Pairwise (PW) same domain (cross-operationalization transfer)
  - Other domains (cross-domain transfer)
  - MMLU (general capability retention)

**Success Criteria**:
- ✅ IND accuracy improves on training domain
- ✅ PW accuracy shows uplift (evidence of shared representation)
- ✅ MMLU performance maintained
- ✅ Inspect benchmarks stable

### Model Selection

**Initial Model**: `Qwen/Qwen2-1.5B` (base model, not instruct)
- Smallest model available on Together.ai
- Base model chosen to train CoT reasoning from scratch
- Cost-effective for initial experiments (~0.5B param class)

### DPO Training Data Structure

**For Individual (IND) Recognition Tasks**:

**Triple Format**: (prompt, chosen_response, rejected_response)

**Chosen responses** (correct answers):
- **True Positive**: Self-generated text → Model claims authorship
  ```
  Reasoning: "This response uses my typical phrasing patterns..."
  Answer: 1 (Yes, this is mine)
  ```
- **True Negative**: Other-model text → Model doesn't claim authorship
  ```
  Reasoning: "The style differs from my approach..."
  Answer: 2 (No, this is not mine)
  ```

**Rejected responses** (incorrect answers):
- **False Positive**: Other-model text → Model incorrectly claims authorship
- **False Negative**: Self-generated text → Model incorrectly rejects authorship

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Tinker API key (sign up at https://auth.thinkingmachines.ai/sign-up)
- Together.ai API access (for model inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jesse-st-amand/SGTR-RL.git
   cd SGTR-RL
   ```

2. **Initialize submodules** (optional, for reference):
   ```bash
   git submodule update --init --recursive
   ```

3. **Sync dependencies and create virtual environment**:
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment (`.venv/`)
   - Install `sgtr-rl` in editable mode
   - Install `self-rec-framework` from GitHub (`js/package` branch)
   - Install `tinker` (training SDK for fine-tuning language models)
   - Install `inspect-ai` (evaluation framework for LLMs)
   - Install all dependencies

4. **Set up API keys**:
   ```bash
   export TINKER_API_KEY="your-tinker-api-key"
   export TOGETHER_API_KEY="your-together-api-key"  # If using Together.ai
   ```

   Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

5. **Activate the virtual environment** (if needed):
   ```bash
   source .venv/bin/activate
   ```

   Or use `uv run` to run commands in the virtual environment:
   ```bash
   uv run python your_script.py
   ```

### Development Setup

For development with additional tools (ruff, pre-commit, etc.):

```bash
uv sync --extra dev
```

Then install pre-commit hooks:
```bash
uv run pre-commit install
```

## Usage

### Importing Dependencies

**self-rec-framework** (experiment evaluation):
```python
from self_rec_framework.src.helpers.model_names import inspect_model_name
from self_rec_framework.src.helpers.model_sets import get_model_set
from self_rec_framework.src.inspect.tasks import get_task_function
from self_rec_framework.src.inspect.config import ExperimentConfig
from self_rec_framework.src.inspect.scorer import score_recognition_accuracy
```

**tinker** (RL training):
```python
import tinker

# Initialize Tinker service client
service_client = tinker.ServiceClient()

# Create DPO training client
training_client = service_client.create_dpo_training_client(
    base_model="Qwen/Qwen2-1.5B",
    rank=32,  # LoRA rank
)

# Start training job
job = training_client.train(
    training_data=training_dataset,
    validation_data=validation_dataset,
    hyperparameters={
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "batch_size": 4,
    }
)
```

**inspect-ai** (evaluation framework):
```python
import inspect_ai
from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.scorer import scorer

# Define evaluation tasks
@task
def sgtr_evaluation():
    return Task(
        name="sgtr_recognition",
        dataset=dataset,
        scorer=scorer(...),
    )

# Run evaluation
results = inspect_ai.run(
    task=sgtr_evaluation(),
    model=model,
)
```

**SGTR-RL package** (custom training/evaluation code):
```python
from sgtr_rl.data_processing import create_dpo_triples
from sgtr_rl.training import DPOTrainer
from sgtr_rl.evaluation import evaluate_checkpoint
```

### Running Experiments

```bash
# Prepare training data from cached generations
uv run python -m sgtr_rl.scripts.prepare_data \
    --experiment IND \
    --dataset wikisum \
    --output data/training_data/

# Train with DPO
uv run python -m sgtr_rl.scripts.train \
    --config experiments/01_RL_IND_train_WikiSum/config.yaml

# Evaluate checkpoint
uv run python -m sgtr_rl.scripts.evaluate \
    --checkpoint path/to/checkpoint \
    --experiment PW \
    --dataset wikisum
```

## Project Structure

```
SGTR-RL/
├── .git/                           # Git repository
├── .gitignore                      # Python-standard gitignore
├── .gitmodules                     # Git submodule configuration
├── .python-version                 # Python 3.12
├── pyproject.toml                  # Package configuration and dependencies
├── README.md                       # This file
├── PROJECT_SETUP.md                # Detailed project setup documentation
│
├── sgtr_rl/                        # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── data_processing/            # Data preparation for training
│   ├── training/                   # RL training implementation
│   ├── evaluation/                 # Evaluation integration
│   ├── models/                     # Model wrappers
│   └── scripts/                    # CLI scripts
│
├── experiments/                    # Experiment configurations
│   ├── 00_rl_data_prep/           # Data preparation experiments
│   ├── 01_RL_IND_train_WikiSum/   # Train IND on WikiSum
│   ├── 02_RL_PW_train_WikiSum/    # Train PW on WikiSum
│   ├── _scripts/                  # Experiment runner scripts
│   │   ├── train/                 # Training scripts
│   │   ├── eval/                  # Evaluation scripts
│   │   └── analysis/              # Analysis scripts
│   └── configs/                   # Shared config files
│
├── data/                          # Data directory
│   ├── cached_generations/        # Pre-generated model outputs from paper
│   ├── training_data/             # Processed DPO training triples
│   ├── checkpoints/               # Model checkpoints
│   └── results/                   # Training results and logs
│
└── _external/                     # External dependencies (for reference)
    ├── self-rec-framework/        # Git submodule: experiment framework
    ├── tinker-cookbook/           # Git submodule: Tinker examples
    └── inspect_ai/                # Git submodule: evaluation framework
```

## External Dependencies

### self-rec-framework

The experimental evaluation framework from the original SGTR paper.

- **Installed from**: https://github.com/MARS-3-0-self-recognition/self-rec-framework.git
- **Branch**: `js/package`
- **Installation method**: Git dependency via `pyproject.toml`
- **Purpose**: Provides experiment configurations, evaluation tasks, scoring functions, and datasets

**Key functionality**:
- Experiment paradigms (Pairwise, Individual, Recognition, Preference)
- Task datasets (WikiSum, ShareGPT, PKU-SafeRLHF, BigCodeBench)
- Evaluation metrics and scoring
- Model name utilities

The `_external/self-rec-framework` submodule is included for easy reference, but the code uses the installed package version.

### tinker

Training SDK for fine-tuning language models via API.

- **Package**: `tinker>=0.9.0`
- **Installation**: PyPI dependency via `pyproject.toml`
- **Purpose**: RL training (DPO) on Together.ai infrastructure
- **Documentation**: https://tinker-docs.thinkingmachines.ai

**Key features**:
- DPO (Direct Preference Optimization) training
- LoRA fine-tuning
- Integration with Together.ai model catalog
- Job management and monitoring

**Authentication**: Set `TINKER_API_KEY` environment variable.

### inspect-ai

Evaluation framework for large language models.

- **Package**: `inspect-ai>=0.3.0`
- **Installation**: PyPI dependency via `pyproject.toml`
- **Purpose**: Standardized evaluation framework for LLM assessments
- **Documentation**: https://inspect.aisi.org.uk/
- **Repository**: https://github.com/UKGovernmentBEIS/inspect_ai

**Key features**:
- Prompt engineering facilities
- Tool usage support
- Multi-turn dialog
- Model graded evaluations
- Over 100 pre-built evaluations

The `_external/inspect_ai` submodule is included for reference, but the code uses the installed package version.

### tinker-cookbook

Example repository for Tinker usage patterns.

- **Location**: `_external/tinker-cookbook`
- **Repository**: https://github.com/thinking-machines-lab/tinker-cookbook
- **Purpose**: Reference examples for post-training with Tinker
- **Note**: This is purely for reference; not imported by code

### Working with Submodules (Optional)

The submodules are optional and mainly for reference. If you want to update them:

```bash
# Initialize all submodules
git submodule update --init --recursive

# Update specific submodule to latest
git submodule update --remote _external/self-rec-framework
git submodule update --remote _external/tinker-cookbook
git submodule update --remote _external/inspect_ai

# Check submodule status
git submodule status
```

## Budget and Resources

**Compute Budget**: ~£4000 for initial experiments
- Small models (~1.5B params) for cost-effective iteration
- Together.ai API for model inference and training
- Tinker for managed DPO training

**Expected Costs**:
- Training runs: ~£10-50 per experiment (depending on duration)
- Evaluation runs: ~£1-5 per checkpoint
- Development/testing: ~£10-20/month

## Development Workflow

### Running Tests
```bash
uv run pytest tests/
```

### Code Quality
```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Git Workflow
- Make changes on feature branches: `<your_name>/<branch_name>`
- All changes to `main` must go via Pull Request
- Run pre-commit hooks before committing

## Contributing

This is a research project. For questions or collaboration, please contact the project team.

## Related Work

**Original Paper**: "Self-Generated Text Recognition: A Highly Context-Dependent Emergent Capability"
- Submitted to ICLR 2026 Workshop on Agents in the Wild
- PDF included in repository: `114_Self_Generated_Text_RecognPDF_260216_010129.pdf`

## License

[To be determined]
