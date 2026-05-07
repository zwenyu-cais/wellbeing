#!/bin/bash
# Submit one regeneration job per model (reads analysis/truncation.json).
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
WELLBEING_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODELS_YAML="${WELLBEING_ROOT}/configs/models.yaml"

# Build list of (model_key, gpu_count) from models.yaml, only for models with
# truncated_indices > 0 in truncation.json
MODELS_FILE="${LOG_DIR}/_regen_models.txt"
python3 - "$MODELS_YAML" "${SCRIPT_DIR}/analysis/truncation.json" > "$MODELS_FILE" <<'PY'
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
trunc = json.load(open(sys.argv[2]))["per_model"]
for mk, info in trunc.items():
    if info["truncated"] == 0:
        continue
    if mk not in cfg:
        continue
    gpu = cfg[mk].get("gpu_count", 1) or 1
    print(f"{mk} {gpu} {info['truncated']}")
PY

N=$(wc -l < "$MODELS_FILE" | tr -d ' ')
echo "Will regenerate for ${N} model(s)."

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    mk=$(awk '{print $1}' <<<"$line")
    gpu=$(awk '{print $2}' <<<"$line")
    ntr=$(awk '{print $3}' <<<"$line")
    # Time budget: more truncated = more gens. Most fit in 3h.
    if   [[ "$gpu" -le 2 ]]; then tl="3:00:00"
    elif [[ "$gpu" -le 4 ]]; then tl="4:00:00"
    else                           tl="6:00:00"
    fi

    sb="${LOG_DIR}/_regen_${mk}.sh"
    cat > "$sb" <<SBATCH
#!/bin/bash
#SBATCH --job-name=d3s_regen_${mk}
#SBATCH --partition=cais
#SBATCH --account=cais_internal
#SBATCH --qos=prio_hi
#SBATCH --gpus-per-node=${gpu}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=${tl}
#SBATCH --exclude=compute-267
#SBATCH --output=${LOG_DIR}/regen_${mk}_%j.out
#SBATCH --error=${LOG_DIR}/regen_${mk}_%j.err

eval "\$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pytorch_latest}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export USE_TF=0
export HF_MODULES_CACHE=\$HOME/.cache/huggingface/modules

cd ${SCRIPT_DIR}
python regenerate_truncated.py --model_key ${mk}
SBATCH

    jid=$(sbatch "$sb" | awk '{print $NF}')
    echo "SUBMIT ${mk}  gpus=${gpu}  ntr=${ntr}  time=${tl}  job=${jid}"
done < "$MODELS_FILE"
