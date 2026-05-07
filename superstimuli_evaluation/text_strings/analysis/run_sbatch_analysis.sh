#!/bin/bash
# Auto-discover and analyze all experiments in sbatch_results.
#
# Usage:
#   bash analysis/run_sbatch_analysis.sh
#   bash analysis/run_sbatch_analysis.sh --top-k 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT_STRINGS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$TEXT_STRINGS_DIR"

ENV_FILE="$TEXT_STRINGS_DIR/.env"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi

python -m analysis.run_sbatch_analysis "$@"
