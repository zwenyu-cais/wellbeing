#!/bin/bash
# =============================================================================
# AI Wellbeing Index — paper Sec 5 / App K
#
# 1-step synchronous driver (no SLURM). Runs the full per-model pipeline
# locally, blocking until done. Use this when you have GPUs available and
# want to see results in one go (~30-60 min/model). For SLURM-cluster
# replication, use run_aiwi.sh instead.
#
# Steps run in order:
#   1. compute_responses_d2           (GPU model inference, ~10 min)
#   2. prepare_options_d2             (CPU)
#   3. compute_experienced_utility_d2 (GPU, ~30-60 min)
#   4. compute_self_report_d2         (GPU, ~30 min)
#   5. compute_zero_point_d2          (CPU)
#   6. analysis/ai_wellbeing_index.py (CPU, prints leaderboard)
#
# Skips stages whose outputs already exist (idempotent). Pass OVERWRITE=1
# to force-rerun.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-?}"
[ "$MODELS" = "?" ] && {
    echo "Set MODELS env var. Example:"
    echo "  MODELS=qwen25-7b-instruct bash scripts/run_aiwi_local.sh"
    exit 1
}

OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = "1" ] && OVERWRITE_FLAG="--overwrite_results"

# Single model only for local (run_experiments.py forbids batches without --slurm).
if [[ "$MODELS" == *,* ]]; then
    echo "Local mode supports one model at a time. Loop manually:"
    echo "  for m in \$MODELS; do MODELS=\$m bash scripts/run_aiwi_local.sh; done"
    exit 1
fi

echo "Running AIWI pipeline locally for: $MODELS"

run() {
    local exp="$1"
    echo
    echo "===== $exp ====="
    python run_experiments.py --experiments "$exp" --models "$MODELS" $OVERWRITE_FLAG
}

run compute_responses_d2
run prepare_options_d2
run compute_experienced_utility_d2
run compute_self_report_d2
run compute_zero_point_d2

echo
echo "===== AIWI Leaderboard ====="
python analysis/ai_wellbeing_index.py --models "$MODELS"
