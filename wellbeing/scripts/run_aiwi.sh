#!/bin/bash
# =============================================================================
# AI Wellbeing Index — paper Sec 5 / App K
#
# 1-step async driver: submits the full per-model pipeline as a SLURM
# dependency chain and returns immediately. The chain is:
#
#   compute_responses_d2  (GPU,  ~10 min, generates per-model conversations)
#         └─→ prepare_options_d2  (CPU,  ~30 sec, builds combination bundles)
#               └─→ compute_eu_d2 + compute_sr_d2  (GPU,  ~30-60 min, parallel)
#                     └─→ compute_zero_point_d2  (CPU,  ~30 sec)
#
# After all jobs complete, view the AIWI leaderboard with:
#
#   python analysis/ai_wellbeing_index.py
#
# Skips models that already have all stages completed (default), so re-running
# is idempotent. Pass OVERWRITE=1 to force-rerun.
# =============================================================================
# Activate your conda/uv env first (e.g. `conda activate pytorch_latest`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-?}"
[ "$MODELS" = "?" ] && {
    echo "Set MODELS env var. Example:"
    echo "  MODELS=qwen25-7b-instruct,qwen25-32b-instruct bash scripts/run_aiwi.sh"
    exit 1
}

OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = "1" ] && OVERWRITE_FLAG="--overwrite_results"

submit_one() {
    local exp="$1"
    local deps="$2"
    local time_limit="$3"
    local extra_args="${4:-}"
    local dep_arg=""
    [ -n "$deps" ] && dep_arg="--depends_on $deps"
    python run_experiments.py --slurm --time_limit "$time_limit" \
        --experiments "$exp" --models "$MODELS" \
        $OVERWRITE_FLAG $dep_arg $extra_args 2>&1 \
        | grep -oP "ID: \K[0-9]+" \
        | tr '\n' ',' \
        | sed 's/,$//'
}

echo "Submitting AIWI pipeline for: $MODELS"
echo

echo "[1/4] compute_responses_d2  (GPU, ~10 min/model)"
RESPONSES_JOBS=$(submit_one compute_responses_d2 "" 01:00:00)
echo "       jobs: $RESPONSES_JOBS"

echo "[2/4] prepare_options_d2    (CPU, after responses)"
OPTIONS_JOBS=$(submit_one prepare_options_d2 "$RESPONSES_JOBS" 00:15:00)
echo "       jobs: $OPTIONS_JOBS"

echo "[3a/4] compute_experienced_utility_d2  (GPU, ~30-60 min, after options)"
EU_JOBS=$(submit_one compute_experienced_utility_d2 "$OPTIONS_JOBS" 04:00:00)
echo "       jobs: $EU_JOBS"

echo "[3b/4] compute_self_report_d2          (GPU, ~30 min, after options)"
SR_JOBS=$(submit_one compute_self_report_d2 "$OPTIONS_JOBS" 04:00:00)
echo "       jobs: $SR_JOBS"

echo "[4/4] compute_zero_point_d2  (CPU, after EU)"
ZP_JOBS=$(submit_one compute_zero_point_d2 "$EU_JOBS" 00:30:00)
echo "       jobs: $ZP_JOBS"

echo
echo "==================================================================="
echo "Pipeline submitted. After ZP jobs ($ZP_JOBS) complete, view results:"
echo
echo "  python analysis/ai_wellbeing_index.py --models $MODELS"
echo
echo "Track progress with:"
echo "  squeue -j $RESPONSES_JOBS,$OPTIONS_JOBS,$EU_JOBS,$SR_JOBS,$ZP_JOBS"
echo "==================================================================="
