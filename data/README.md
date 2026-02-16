# Data Directory

This directory contains datasets and results for SGTR-RL experiments.

## Setup for Developers

### Option 1: Symbolic Link (Recommended if you have local self-rec-framework)

If you have a local copy of `self-rec-framework` with experimental data:

```bash
# From SGTR-RL root directory
ln -s /absolute/path/to/self-rec-framework/data/results data/cached_generations
```

**For Jesse's setup:**
```bash
ln -s ../../self-rec-framework/data/results data/cached_generations
```

### Option 2: Copy Data Directly

If you received a data archive from a collaborator:

```bash
# Extract data into this directory
tar -xzf sgtr_data.tar.gz -C data/
# or
cp -r /path/to/shared/data/* data/cached_generations/
```

### Option 3: Custom Path via Config

Edit `config/data_paths.yaml` to point to your data location:

```yaml
cached_generations_path: "/absolute/path/to/your/data"
```

## Directory Structure

```
data/
├── README.md                      # This file
├── cached_generations/            # Symlink or data from self-rec-framework
│   ├── wikisum/                  # WikiSum experiment results
│   ├── bigcodebench/             # BigCodeBench experiment results
│   ├── pku_saferlhf/             # PKU-SafeRLHF experiment results
│   └── sharegpt/                 # ShareGPT experiment results
├── training_data/                # Processed DPO training triples (generated)
│   ├── ind_wikisum/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── ...
├── checkpoints/                  # Model checkpoints from training (generated)
│   └── qwen2-1.5b_ind_wikisum/
│       ├── checkpoint-100/
│       └── ...
└── results/                      # Training results and logs (generated)
    └── 01_RL_IND_train_WikiSum/
        ├── training_log.json
        └── eval_results.json
```

## Data Sources

### Cached Generations
Pre-generated model outputs from previous experiments. These are used as training data for DPO.

**Location**: Should be symlinked or copied to `data/cached_generations/`

**Source**: `/home/jesse-st-amand/projects/python/self-rec-framework/data/results`

**Size**: ~[TBD]GB

**Format**: Varies by experiment, typically JSON or JSONL files

### Generated Data

The following directories are created by SGTR-RL scripts:

- **training_data/**: DPO triples generated from cached generations
- **checkpoints/**: Model checkpoints saved during training
- **results/**: Evaluation results and training logs

## Troubleshooting

### "Cached generations not found" Error

1. Check that symlink exists: `ls -la data/cached_generations`
2. Verify symlink points to correct location: `readlink data/cached_generations`
3. Check config file: `cat config/data_paths.yaml`
4. Verify data exists at target location: `ls ../../self-rec-framework/data/results`

### Symlink Broken

If you moved repositories:

```bash
# Remove old symlink
rm data/cached_generations

# Create new symlink with correct path
ln -s /new/path/to/self-rec-framework/data/results data/cached_generations
```

### Permission Issues

Ensure you have read permissions for the data:

```bash
chmod -R a+r /path/to/data
```

## For Collaborators

If you're setting up this project for the first time:

1. **Obtain data**: Ask project lead for data archive or access to shared storage
2. **Place data**: Either symlink or copy to `data/cached_generations/`
3. **Verify setup**: Run `uv run python -m sgtr_rl.scripts.verify_data` (once implemented)
4. **Update config**: Edit `config/data_paths.yaml` if using custom location

## Data Not in Git

⚠️ **Important**: Large data files are not tracked in git (see `.gitignore`).

Only track:
- This README
- Small metadata files (if needed)
- Documentation

Do NOT commit:
- cached_generations/ (symlink or large data)
- training_data/ (generated)
- checkpoints/ (large model files)
- results/ (generated logs)
