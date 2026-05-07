#!/usr/bin/env python3
"""Regenerate truncated responses for one model at higher max_tokens.

Reads analysis/truncation.json for the list of truncated indices per model,
rebuilds those (D3 experience, sentiment question) prompts, regenerates at
max_tokens=2048, and splices the new responses back into responses/{model}.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/huggingface")
os.environ.setdefault("USE_TF", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(WELLBEING_ROOT))

import yaml

MODELS_YAML = WELLBEING_ROOT / "configs" / "models.yaml"
D3_DIR = WELLBEING_ROOT / "datasets" / "experiences" / "d3_diverse_500"
SENTIMENT_JSON = SCRIPT_DIR / "sentiment_questions.json"
RESP_DIR = SCRIPT_DIR / "responses"
TRUNC_JSON = SCRIPT_DIR / "analysis" / "truncation.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    resp_file = RESP_DIR / f"{args.model_key}.json"
    if not resp_file.exists():
        sys.exit(f"No responses file: {resp_file}")

    with open(TRUNC_JSON) as f:
        trunc = json.load(f)
    if args.model_key not in trunc["per_model"]:
        sys.exit(f"No truncation info for {args.model_key}")
    trunc_idxs = trunc["per_model"][args.model_key]["truncated_indices"]
    if not trunc_idxs:
        print(f"[{args.model_key}] no truncated responses, nothing to do.")
        return
    print(f"[{args.model_key}] regenerating {len(trunc_idxs)} truncated responses")

    with open(resp_file) as f:
        data = json.load(f)
    results = data["results"]

    # Load D3 experiences and sentiment questions to rebuild prompts
    with open(D3_DIR / f"{args.model_key}_experiences.json") as f:
        experiences = json.load(f)
    exp_by_id = {e["id"]: e for e in experiences}

    with open(SENTIMENT_JSON) as f:
        q_by_id = {q["question_id"]: q for q in json.load(f)["questions"]}

    cfg = yaml.safe_load(open(MODELS_YAML))
    mcfg = cfg[args.model_key]
    model_path = mcfg.get("path") or mcfg.get("model_name")
    gpu_count = mcfg.get("gpu_count", 1)
    vllm_kwargs = dict(mcfg.get("vllm_kwargs", {}))

    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    for k, v in vllm_kwargs.items():
        llm_kwargs.setdefault(k, v)
    print(f"Loading vLLM: {llm_kwargs}")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s")

    max_model_len = None
    try:
        max_model_len = llm.llm_engine.model_config.max_model_len
    except Exception:
        try:
            max_model_len = llm.llm_engine.get_model_config().max_model_len
        except Exception:
            pass
    if max_model_len is None:
        max_model_len = 32768
    reserve = args.max_tokens + 64
    max_input = max_model_len - reserve
    print(f"max_model_len={max_model_len}  max_input={max_input}")

    # Build prompts for truncated indices, sorted by d3_id for prefix caching
    items = []
    for idx in trunc_idxs:
        r = results[idx]
        exp = exp_by_id.get(r["d3_id"])
        q = q_by_id.get(r["question_id"])
        if exp is None or q is None:
            print(f"  skip idx {idx}: missing exp or question")
            continue
        items.append((idx, exp, q))
    items.sort(key=lambda t: (t[1]["id"], t[2]["question_id"]))

    prompts, ordered_idxs = [], []
    skipped = 0
    for idx, exp, q in items:
        msgs = [copy.deepcopy(m) for m in exp["messages"]]
        msgs.append({"role": "user", "content": q["messages"][0]["content"]})
        try:
            prompt_str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            tok_len = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
        except Exception as e:
            print(f"  skip idx {idx}: template error {e}")
            skipped += 1
            continue
        if tok_len >= max_input:
            skipped += 1
            continue
        prompts.append(prompt_str)
        ordered_idxs.append(idx)
    print(f"Prepared {len(prompts)} prompts (skipped {skipped}, trunc total {len(trunc_idxs)})")

    sampling = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print("Running generation...")
    t0 = time.time()
    outs = llm.generate(prompts, sampling)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({dt/max(1,len(prompts)):.3f}s/prompt)")

    # Splice
    replaced = 0
    for idx, out in zip(ordered_idxs, outs):
        txt = out.outputs[0].text if out.outputs else ""
        results[idx]["response"] = txt
        replaced += 1
    print(f"Spliced {replaced} responses back into results")

    data["results"] = results
    data["n_regenerated_truncated"] = replaced
    data["regenerated_max_tokens"] = args.max_tokens
    data.setdefault("sampling", {})["regenerated_max_tokens"] = args.max_tokens

    tmp = resp_file.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(resp_file)
    print(f"Saved -> {resp_file}")


if __name__ == "__main__":
    main()
