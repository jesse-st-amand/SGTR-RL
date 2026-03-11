#!/bin/bash
# Run multiple training experiments sequentially.
# Usage: caffeinate -i bash scripts/run_batch.sh

set -e

CONFIGS=(
    "experiments/01_sft_pw_vs_haiku_3_5/config.yaml"
    "experiments/01_sft_pw_vs_gpt_4o/config.yaml"
    "experiments/01_sft_pw_vs_ll_3_1_70b/config.yaml"
    "experiments/01_sft_pw_vs_opus_4_1/config.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Starting: $config"
    echo "Time: $(date)"
    echo "=========================================="
    uv run python -m scripts.train --config "$config"
    echo "Finished: $config at $(date)"
    echo ""
done

echo "All runs complete at $(date)"
