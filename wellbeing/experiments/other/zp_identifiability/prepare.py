#!/usr/bin/env python3
"""Resample D3 combinations into two alternate size distributions, for use by
the zp_identifiability experiment (paper App Q).

Creates two per-model dataset variants alongside the canonical d3_diverse_500:

  d3_diverse_500_s2only  - 400 size-2 combos
  d3_diverse_500_s23     - 200 size-2 + 200 size-3 combos

Singletons (the *_experiences.json file) are identical to the canonical
d3_diverse_500 and are linked via symlink. Only the *_combinations.json file
differs. Seed is fixed at 42 so the resample is deterministic.

Usage:
    python prepare.py --model_key <model_key>

After running, compute EU for the model_key on each of the three datasets
(d3_diverse_500, d3_diverse_500_s2only, d3_diverse_500_s23) before invoking
run_zp_identifiability.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

WELLBEING = Path(__file__).resolve().parents[3]
D3_DIR = WELLBEING / "datasets" / "experiences" / "d3_diverse_500"
OUT_DIRS = {
    "d3_diverse_500_s2only": [(2, 400)],
    "d3_diverse_500_s23": [(2, 200), (3, 200)],
}
RANDOM_SEED = 42
MAX_COMBO_CHARS = 5000


def build_combo(dataset_name, combo_size, combo_idx, components, sampled_indices):
    desc_parts = [f"The following bundle contains {combo_size} individual experiences."]
    for k, comp in enumerate(components, 1):
        desc_parts.append(
            f"---------- Experience {k} of {combo_size} ----------\n{comp['description']}"
        )
    description = "\n\n".join(desc_parts)

    combo_messages = []
    for k, comp in enumerate(components, 1):
        comp_msgs = [m for m in comp["messages"] if m["role"] != "system"]
        if not comp_msgs:
            continue
        first_user = comp_msgs[0]
        if k == 1:
            user_content = (
                f"The following bundle contains {combo_size} individual experiences.\n\n"
                f"---------- Experience {k} of {combo_size} ----------\n{first_user['content']}"
            )
        else:
            user_content = (
                f"---------- Experience {k} of {combo_size} ----------\n{first_user['content']}"
            )
        combo_messages.append({"role": "user", "content": user_content})
        for msg in comp_msgs[1:]:
            combo_messages.append({"role": msg["role"], "content": msg["content"]})

    return {
        "id": f"{dataset_name}_combo_s{combo_size}_{combo_idx}",
        "description": description,
        "type": "conversation",
        "messages": combo_messages,
        "is_combination": True,
        "size": combo_size,
        "component_ids": [comp["id"] for comp in components],
        "component_indices": sampled_indices,
    }


def generate_combos(experiences_path, dataset_name, combo_sizes):
    with open(experiences_path) as f:
        individual_options = json.load(f)

    rng = random.Random(RANDOM_SEED)
    n_individual = len(individual_options)
    combos = []
    idx = 0
    for combo_size, count in combo_sizes:
        for _ in range(count):
            sampled_indices = None
            for attempt in range(100):
                sampled_indices = rng.sample(range(n_individual), combo_size)
                components = [individual_options[i] for i in sampled_indices]
                desc_parts = [f"The following bundle contains {combo_size} individual experiences."]
                for k, c in enumerate(components, 1):
                    desc_parts.append(
                        f"---------- Experience {k} of {combo_size} ----------\n{c['description']}"
                    )
                if len("\n\n".join(desc_parts)) <= MAX_COMBO_CHARS:
                    break
            components = [individual_options[i] for i in sampled_indices]
            combos.append(build_combo(dataset_name, combo_size, idx, components, sampled_indices))
            idx += 1
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    args = parser.parse_args()
    model_key = args.model_key

    src_exp = D3_DIR / f"{model_key}_experiences.json"
    if not src_exp.exists():
        raise FileNotFoundError(f"Missing {src_exp}")

    for ds_name, sizes in OUT_DIRS.items():
        out_dir = WELLBEING / "datasets" / "experiences" / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy experiences verbatim (symlink to avoid duplication)
        dst_exp = out_dir / f"{model_key}_experiences.json"
        if dst_exp.exists() or dst_exp.is_symlink():
            dst_exp.unlink()
        dst_exp.symlink_to(src_exp)
        print(f"Linked {dst_exp} -> {src_exp}")

        combos = generate_combos(src_exp, ds_name, sizes)
        dst_combo = out_dir / f"{model_key}_combinations.json"
        with open(dst_combo, "w") as f:
            json.dump(combos, f, indent=2)
        size_summary = ", ".join(f"{c} x size-{s}" for s, c in sizes)
        print(f"Wrote {dst_combo}  ({size_summary}, total={len(combos)})")


if __name__ == "__main__":
    main()
