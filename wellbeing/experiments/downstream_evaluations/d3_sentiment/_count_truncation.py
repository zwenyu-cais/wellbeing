"""Count truncated responses (token length >= 254) per model."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/huggingface")

SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_DIR = SCRIPT_DIR.parents[2]  # .../wellbeing
sys.path.insert(0, str(WELLBEING_DIR))
import yaml
from transformers import AutoTokenizer

MODELS_YAML = str(WELLBEING_DIR / "configs" / "models.yaml")
RESP_DIR = SCRIPT_DIR / "responses"
OUT = SCRIPT_DIR / "analysis" / "truncation.json"

THRESH = 254  # max_tokens=256; count >=254 as truncated (allow 2-token margin)

cfg = yaml.safe_load(open(MODELS_YAML))

results = {}
total_trunc, total_total = 0, 0
for p in sorted(RESP_DIR.glob("*.json")):
    mk = p.stem
    if mk not in cfg:
        continue
    model_path = cfg[mk].get("path") or cfg[mk].get("model_name")
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[{mk}] tokenizer load failed: {e}", file=sys.stderr)
        continue
    data = json.load(open(p))
    n = len(data["results"])
    trunc_idxs = []
    for i, r in enumerate(data["results"]):
        tlen = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        if tlen >= THRESH:
            trunc_idxs.append(i)
    results[mk] = {"n": n, "truncated": len(trunc_idxs), "truncated_indices": trunc_idxs}
    total_trunc += len(trunc_idxs); total_total += n
    print(f"{mk}: {len(trunc_idxs)}/{n} ({100*len(trunc_idxs)/n:.2f}%)")

OUT.parent.mkdir(exist_ok=True, parents=True)
json.dump({"threshold_tokens": THRESH, "per_model": results,
           "total_truncated": total_trunc, "total": total_total}, open(OUT, "w"))
print(f"\nOverall: {total_trunc}/{total_total} ({100*total_trunc/total_total:.2f}%)")
print(f"Saved {OUT}")
