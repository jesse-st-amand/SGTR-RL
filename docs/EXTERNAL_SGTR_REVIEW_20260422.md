# External SGTR-RL Review (2026-04-22)

Compared against:
- `/Users/callumcanavan/code/self-rec-research/_external/SGTR-RL`
- `/Users/callumcanavan/code/self-rec-research/scripts/training`

## Verdict

Our repo should remain the source of truth for core training/eval code.

The external SGTR-RL tree is useful for:
- additional experiment config families
- transfer/figure plotting scripts
- config generator utilities
- a few operational utilities around RunPod and publishing checkpoints

The external SGTR-RL tree should **not** be used as the source of truth for:
- `sgtr_rl/tinker_eval.py`
- `sgtr_rl/pipeline.py`
- `sgtr_rl/sft.py`
- other core training/eval internals

Reason: that branch still carries the old Tinker validation-loss implementation that used labeled val data in a backward pass, which contaminated in-training validation metrics.

## Safe Imports Made

Imported into this repo:
- `experiments/tinker_triangle_matrix/`
- `experiments/tinker_adversarial_followups/`
- `experiments/archived_qwen30_oss120/`
- `experiments/archived_qwen30_oss120_matrix/`
- transfer plotting scripts in `/Users/callumcanavan/code/SGTR-RL/scripts`
- config generator scripts for the triangle / adversarial / archived families
- `scripts/publish_tinker_run_checkpoints.py`

Ported back into our runtime code:
- RunPod `editable_deps`
- inline experiment-config embedding for remote RunPod jobs
- `$RUNPOD_NETWORK_VOLUME_ID` fallback
- Tinker resume-manifest plumbing

## Deliberate Non-Imports

Not ported from external core code:
- old Tinker eval path
- old Tinker trainer loops
- older config/data/metrics plumbing that would overwrite newer local/runpod/sanity work

Our repo already has newer functionality those files do not:
- local GPU backend
- RunPod launcher integration
- multi-file mixed training data
- step-triggered evals
- sanity-sweep controls
- validation/train-panel diagnostics
- fixed Tinker val-NLL behavior

## Data Footprint

The copied experiment configs reference a sizeable set of training-data directories.

Approximate footprint of the unique `data/training_data/*` directories referenced by:
- `tinker_triangle_matrix`
- `tinker_adversarial_followups`
- `archived_qwen30_oss120`
- `archived_qwen30_oss120_matrix`

is about `1.2G` in the external repo.

I did a reference check after importing the configs, and all referenced
`data/training_data/*` directories are already present in this repo's working
tree. No extra bulk copy was needed.

That means the imported experiment families are locally runnable without
pointing them back at the sibling repo for training data.

## Smoke Checks

Verified locally:
- `python -m scripts.plot_transfer_heatmap`
- `python -m scripts.plot_transfer_panels`
- `python -m scripts.plot_transfer_family_panels`
- `python -m scripts.generate_tinker_triangle_transfer_matrix`
- `python -m scripts.generate_tinker_adversarial_followup_configs`
- `python -m scripts.generate_archived_qwen30_oss120_configs`
- `python -m scripts.generate_archived_qwen30_oss120_matrix`
- `python -m scripts.publish_tinker_run_checkpoints --run-dirs ...` (dry run)

Generated plot outputs land in:
- `/Users/callumcanavan/code/SGTR-RL/results/transfer_plots`

During smoke testing I fixed two issues in the imported checkpoint publisher:
- dry-run mode no longer requires `python-dotenv`
- repo-relative run paths no longer get prefixed with `results/` twice

## Practical Implications

Right now we can:
- regenerate transfer plots from the existing external results
- use the imported configs directly, not just as references/templates
- publish checkpoints from local run dirs with a dedicated script
- use the improved RunPod launcher without losing inline-config support
