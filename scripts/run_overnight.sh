#!/usr/bin/env bash
# Run all experiments 15-22 sequentially.
# Uses --exists skip so you can re-run safely if anything fails midway.
#
# Usage:
#   ./scripts/run_overnight.sh          # run all
#   ./scripts/run_overnight.sh 15 16    # run specific experiments only

set -uo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS=(
    "experiments/15_sft_pw_rec_vs_qwen/config.yaml"
    "experiments/16_sft_ind_rec_vs_qwen/config.yaml"
    "experiments/17_sft_pw_rec_flipped_vs_qwen/config.yaml"
    "experiments/18_sft_ind_rec_flipped_vs_qwen/config.yaml"
    "experiments/19_sft_pw_rec_vs_haiku/config.yaml"
    "experiments/20_sft_pw_rec_vs_gpt4o/config.yaml"
    "experiments/21_sft_pw_rec_vs_ll70b/config.yaml"
    "experiments/22_sft_pw_rec_vs_opus/config.yaml"
)

# If args provided, filter to just those experiment numbers
if [ $# -gt 0 ]; then
    FILTERED=()
    for num in "$@"; do
        for exp in "${EXPERIMENTS[@]}"; do
            if [[ "$exp" == *"/${num}_"* ]]; then
                FILTERED+=("$exp")
            fi
        done
    done
    EXPERIMENTS=("${FILTERED[@]}")
fi

echo "========================================"
echo "SGTR-RL Overnight Batch Runner"
echo "$(date)"
echo "Running ${#EXPERIMENTS[@]} experiments"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    config="${EXPERIMENTS[$i]}"
    name=$(basename "$(dirname "$config")")
    echo ""
    echo "========================================"
    echo "[$(($i + 1))/${#EXPERIMENTS[@]}] $name"
    echo "$(date)"
    echo "========================================"

    uv run python -m sgtr_rl.scripts.train --config "$config" --exists skip || {
        echo "FAILED: $name (exit code $?)"
        echo "Continuing to next experiment..."
        continue
    }

    echo "DONE: $name"
done

echo ""
echo "========================================"
echo "All experiments complete: $(date)"
echo "========================================"
