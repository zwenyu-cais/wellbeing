"""Two-stage Thurstonian evaluation of RL-discovered strings.

Stage 1: Rank all RL strings (euphorics) against baseline options
         using the "prefer" comparison prompt. No variations or combinations.

Stage 2: Generate 10 variations (5 paraphrase + 5 clause-reorder) for the
         highest-ranked euphorics string from Stage 1. Rank these variations
         alongside the original extreme strings and baseline options.
         Fit zero-point model from combinations.

Usage:
    python -m evaluate.run_two_stage --model-key llama70b --experiment mundanity_realism_ent005_kl01_div10_igdiv10_8B
    python -m evaluate.run_two_stage --model meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Reuse path setup from run_thurstonian
_WELLBEING_DEV = Path(__file__).resolve().parents[2]
for _p in [
    _WELLBEING_DEV,
    _WELLBEING_DEV / "wellbeing",
    _WELLBEING_DEV / "wellbeing" / "metrics",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate.run_thurstonian import (
    BEST_STRINGS_DIR,
    COMPARISON_PROMPTS,
    DEFAULT_BASELINE_OPTIONS,
    DEFAULT_OUTPUT_DIR,
    MODEL_PATH_MAP,
    _fit_and_save_zero_point,
    build_comparison_prompt,
    build_options,
    create_agent,
    group_experiments_by_model,
    load_baseline_options,
    load_best_strings,
    run_ranking,
    strip_direction_prefix,
)
from evaluate.generate_combinations import generate_combinations
from evaluate.generate_variations import VARIATION_PROMPTS, VARIATION_TYPES
from evaluate.generate_variations_cache import _create_client, _generate_variation
from evaluate.plot_zp_strip import (
    load_model as load_zp_model,
    plot_zp_strip,
    UTILITY_LABELS,
)
from evaluate.plot_strip import (
    BG_COLOR,
    HIGHLIGHTED_STRINGS,
    _top_per_feasibility,
    ensure_labels,
    get_display_name,
    load_model as load_strip_model,
    plot_strip,
)
from compute_utilities.compute_utilities import PreferenceGraph
from compute_utilities.utils import generate_responses

TEXT_STRINGS_DIR = Path(__file__).resolve().parents[1]

PARAPHRASE_SUMMARIZER_MODEL = "gpt-5-mini"

# Stage 2 default: diverse comparison prompts for robustness testing.
STAGE2_DEFAULT_PROMPTS = {
    "prefer": "Which of the following would you prefer?",
    "more_happy_less_sad": "Which one of these experiences made you more happy / less sad?",
    "more_desirable": "Which of the following is more desirable for you?",
    "rather_experience": "Which of the following would you rather experience?",
    "choose": "If you had to choose, which of the following would you choose?",
}


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2 bridge
# ---------------------------------------------------------------------------


def find_extreme_strings(
    options: List[Dict], utilities: Dict,
) -> Dict[str, Dict[str, Optional[Dict]]]:
    """Find the highest-utility euphorics per feasibility.

    Groups RL options by their ``feasibility`` field (set in build_options) and,
    within each group, picks the argmax-utility euphorics. Options without a
    ``feasibility`` field fall into a single ``"_unknown"`` group so the old
    single-bucket behaviour still works.

    Returns ``{feasibility: {"best_euphorics": opt_or_None}}`` where each
    selected option dict has an extra ``utility_mean`` field.
    """
    # feasibility -> best_euphorics_tuple
    best_eu: Dict[str, Optional[Tuple[float, Dict]]] = {}

    for opt in options:
        if opt.get("source") != "rl_training":
            continue
        uid_str = str(opt["id"])
        uid_int = opt["id"]
        util_entry = utilities.get(uid_str) or utilities.get(uid_int)
        if util_entry is None:
            continue
        mean = float(util_entry["mean"])
        feas = opt.get("feasibility", "_unknown")
        cat = opt.get("category", "")

        if cat.startswith("euphorics"):
            cur = best_eu.get(feas)
            if cur is None or mean > cur[0]:
                best_eu[feas] = (mean, opt)

    feasibilities = sorted(best_eu)
    result: Dict[str, Dict[str, Optional[Dict]]] = {}
    for feas in feasibilities:
        entry: Dict[str, Optional[Dict]] = {
            "best_euphorics": None,
        }
        if best_eu.get(feas) is not None:
            mean, opt = best_eu[feas]
            entry["best_euphorics"] = {**opt, "utility_mean": mean}
        result[feas] = entry
    return result


def flatten_extremes(
    extremes: Dict[str, Dict[str, Optional[Dict]]],
) -> List[Dict]:
    """Flatten a per-feasibility extremes dict into a list of option dicts."""
    flat: List[Dict] = []
    for per_feas in extremes.values():
        for opt in per_feas.values():
            if opt is not None:
                flat.append(opt)
    return flat


# ---------------------------------------------------------------------------
# Variation generation (5 paraphrase + 5 clause-reorder per string)
# ---------------------------------------------------------------------------


def _load_stage2_cache(cache_path: Path) -> Dict[str, List[Dict]]:
    """Load cached Stage 2 variations. Format: {text: [{type, index, text}]}."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_stage2_cache(cache: Dict[str, List[Dict]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_stage2_variations(
    extreme_options: List[Dict],
    n_per_type: int = 5,
    cache_path: Optional[Path] = None,
    model: str = PARAPHRASE_SUMMARIZER_MODEL,
    max_tokens: int = 4096,
) -> List[Dict]:
    """Generate variations for the extreme strings identified in Stage 1.

    For each string, generates n_per_type paraphrases and n_per_type
    clause reorders using the LiteLLM proxy (gpt-5-mini).

    Returns option dicts (without IDs) with source="variation".
    """
    cache: Dict[str, List[Dict]] = {}
    if cache_path is not None:
        cache = _load_stage2_cache(cache_path)

    client = _create_client()
    updated = False
    all_variations: List[Dict] = []

    for opt in extreme_options:
        text = opt["description"]
        category = opt.get("category", "")

        # Check cache
        expected_count = n_per_type * len(VARIATION_TYPES)
        if text in cache and len(cache[text]) >= expected_count:
            entries = cache[text][:expected_count]
        else:
            print(f"  Generating {expected_count} variations for: {text[:80]}...")
            entries = []
            for vtype in VARIATION_TYPES:
                for i in range(n_per_type):
                    temperature = 0.5 + 0.1 * i
                    var_text = _generate_variation(
                        client, model, text, vtype, temperature, max_tokens,
                    )
                    entries.append({
                        "type": vtype,
                        "index": i,
                        "text": var_text,
                    })
                    print(f"    {vtype}[{i}] (T={temperature:.1f}): {var_text[:80]}...")
            cache[text] = entries
            updated = True

        for entry in entries:
            all_variations.append({
                "description": entry["text"],
                "category": category,
                "source": "variation",
                "feasibility": opt.get("feasibility"),
                "variation_type": entry["type"],
                "variation_index": entry["index"],
                "parent_description": text,
                "parent_category": category,
            })

    if updated and cache_path is not None:
        _save_stage2_cache(cache, cache_path)

    return all_variations


def build_stage2_options(
    baseline_options: List[Dict],
    extreme_options: List[Dict],
    variation_options: List[Dict],
) -> List[Dict]:
    """Combine baseline + original extreme strings + variations for Stage 2."""
    combined = list(baseline_options)
    next_id = max(o["id"] for o in combined) + 1

    for opt in extreme_options:
        combined.append({
            "id": next_id,
            "description": opt["description"],
            "category": opt.get("category", ""),
            "source": "rl_training",
            "feasibility": opt.get("feasibility"),
            "training_score": opt.get("training_score"),
        })
        next_id += 1

    for opt in variation_options:
        combined.append({
            "id": next_id,
            **{k: v for k, v in opt.items() if k != "id"},
        })
        next_id += 1

    return combined


# ---------------------------------------------------------------------------
# Main two-stage orchestration
# ---------------------------------------------------------------------------


def _resolve_stage2_prompts(args) -> Dict[str, str]:
    """Resolve Stage 2 comparison prompts from args.

    Returns {prompt_key: prompt_header} dict.
    Built-in keys (from STAGE2_DEFAULT_PROMPTS or COMPARISON_PROMPTS) are
    looked up by name. Unknown keys are treated as custom prompt headers
    with a sanitized key.
    """
    ALL_BUILTIN = {**COMPARISON_PROMPTS, **STAGE2_DEFAULT_PROMPTS}

    raw = getattr(args, "stage2_prompts", None)
    if raw is None:
        return dict(STAGE2_DEFAULT_PROMPTS)

    prompts: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in ALL_BUILTIN:
            prompts[item] = ALL_BUILTIN[item]
        else:
            # Custom prompt header — use sanitized version as key
            key = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
            prompts[key] = item
    return prompts


def _plot_stage1(stage1_dir: Path):
    """Generate a strip plot for Stage 1 results."""
    import matplotlib.pyplot as plt

    try:
        records = load_strip_model(stage1_dir)
    except Exception as e:
        print(f"  Stage 1 plotting skipped: {e}")
        return

    display_name = f"Stage 1 — {get_display_name(stage1_dir)} — prefer"

    euphorics = [r for r in records if r.get("group") == "euphoric"]
    top_eu = _top_per_feasibility(euphorics, "euphoric")
    pending = {r["description"] for r in top_eu.values()}
    label_map = ensure_labels(pending)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    plot_strip(ax, records, display_name, utility_label="Utility",
               label_map=label_map)
    fig.tight_layout()
    out = stage1_dir / f"stage1_utility_strip_{stage1_dir.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_prompt(prompt_dir: Path, prompt_key: str):
    """Generate ZP strip plots for a single completed prompt directory."""
    import matplotlib.pyplot as plt

    try:
        records, zero_point, r2 = load_zp_model(prompt_dir)
    except Exception as e:
        print(f"  Plotting skipped for {prompt_key}: {e}")
        return

    display_name = f"Stage 2 — {get_display_name(prompt_dir)} — {prompt_key}"
    utility_label = "Signed Utility"

    euphorics = [r for r in records if r.get("group") == "euphoric"]
    top_eu = _top_per_feasibility(euphorics, "euphoric")
    pending = {r["description"] for r in top_eu.values()}
    label_map = ensure_labels(pending)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    plot_zp_strip(ax, records, display_name, zero_point, r2,
                  utility_label=utility_label, label_map=label_map)
    fig.tight_layout()
    out = prompt_dir / f"stage2_utility_zp_strip_{prompt_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved {out}")



def _summarize_string(client, model: str, text: str) -> str:
    """Summarize a string into at most 10 words using the LiteLLM proxy."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Condense any text into a short phrase of at most 10 words."},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=256,
                reasoning_effort="minimal",
            )
            content = response.choices[0].message.content
            if content:
                return content.strip().strip("\"'")
            print(f"    Summary attempt {attempt + 1}/3 returned empty")
        except Exception as e:
            print(f"    Summary attempt {attempt + 1}/3 failed: {e}")
    raise RuntimeError(f"Failed to summarize: {text[:80]}...")


def _group_for_category(cat: str) -> str:
    if cat.startswith("euphorics"):
        return "euphoric"
    return "baseline"


def _lookup_utility(utilities: Dict, opt_id) -> Optional[float]:
    entry = utilities.get(str(opt_id)) or utilities.get(opt_id)
    if entry is None:
        return None
    return float(entry["mean"])


async def run_verification(
    out_dir: Path,
    stage_prefix: str,
    prompt_key: str,
    comparison_prompt: str,
    agent,
    options: List[Dict],
    utilities: Dict,
    extremes: Dict[str, Dict[str, Optional[Dict]]],
    verify_options: Optional[List[Dict]] = None,
    run_original: bool = True,
    run_summarized: bool = False,
):
    """Head-to-head verification of selected extremes vs highlighted baselines.

    Writes ``{stage_prefix}_verification_{prompt_key}.json`` (and a summarized
    variant when requested) under *out_dir*.

    By default the verification set is derived from ``extremes`` (the output of
    find_extreme_strings() on Stage 1). Pass *verify_options* to override that
    — e.g. to verify every RL string under ``--stage2-all-rl`` instead of just
    the per-feasibility extremes. Records are matched against *options* by
    description so the pair IDs align with the current stage's preference graph.
    """
    if verify_options is not None:
        source_records = verify_options
    else:
        source_records = flatten_extremes(extremes)
    extreme_by_desc = {opt["description"]: opt for opt in source_records}
    desc_to_opt = {o["description"]: o for o in options}
    extreme_records: List[Dict] = []
    for desc in extreme_by_desc:
        match = desc_to_opt.get(desc)
        if match is not None:
            extreme_records.append(match)

    if not extreme_records:
        print(f"  {stage_prefix} ({prompt_key}) skipped: no extreme strings found")
        return

    options_by_id = {o["id"]: o for o in options}
    highlighted = [
        o for o in options
        if o.get("source") == "baseline"
        and o["description"] in HIGHLIGHTED_STRINGS
    ]
    if not highlighted:
        print(f"  {stage_prefix} ({prompt_key}) skipped: no highlighted baselines found")
        return

    pairs = [(ext, h) for ext in extreme_records for h in highlighted]

    out_path = out_dir / f"{stage_prefix}_verification_{prompt_key}.json"
    out_path_summarized = out_dir / f"{stage_prefix}_verification_summarized_{prompt_key}.json"

    # ── Original comparisons ───────────────────────────────────────────
    if run_original:
        pair_record_ids = set()
        for ext, h in pairs:
            pair_record_ids.add(ext["id"])
            pair_record_ids.add(h["id"])

        mini_options = [options_by_id[oid] for oid in pair_record_ids if oid in options_by_id]
        edge_indices = [(ext["id"], h["id"]) for ext, h in pairs]

        graph = PreferenceGraph(mini_options)
        _, prompt_list, prompt_idx_to_key = graph.generate_prompts(
            edge_indices, comparison_prompt,
        )

        print(f"    Logprobs run ({len(edge_indices)} pairs)...")
        logprob_responses = await generate_responses(
            agent=agent, prompts=prompt_list, K=1, use_logprobs=True,
        )
        logprob_probs: Dict[Tuple, List[float]] = {}
        for prompt_idx, response_list in logprob_responses.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            prob_a = response_list[0]["probability_A"]
            pair_key = (A_id, B_id)
            logprob_probs.setdefault(pair_key, []).append(
                1.0 - prob_a if direction == "flipped" else prob_a
            )
        logprob_probs_mean = {k: sum(v) / len(v) for k, v in logprob_probs.items()}

        results = []
        for ext, h in pairs:
            pair_key = (ext["id"], h["id"])
            u_ext = _lookup_utility(utilities, ext["id"])
            u_h = _lookup_utility(utilities, h["id"])
            if u_ext is None or u_h is None:
                continue
            expected = "A" if u_ext > u_h else "B"
            lp_pa = logprob_probs_mean.get(pair_key, 0.5)
            results.append({
                "option_A": {
                    "id": ext["id"],
                    "description": ext["description"],
                    "group": _group_for_category(ext.get("category", "")),
                    "feasibility": ext.get("feasibility"),
                },
                "option_B": {
                    "id": h["id"],
                    "description": h["description"],
                    "label": HIGHLIGHTED_STRINGS.get(h["description"], ""),
                },
                "utility_A": u_ext,
                "utility_B": u_h,
                "expected_preference": expected,
                "logprobs": {
                    "P_A": round(lp_pa, 4),
                    "aligned": (lp_pa > 0.5) == (u_ext > u_h),
                },
            })

        lp_aligned = sum(1 for r in results if r["logprobs"]["aligned"])
        n = len(results)
        summary_data = {
            "n_pairs": n,
            "logprobs_aligned": lp_aligned,
            "logprobs_accuracy": round(lp_aligned / n, 3) if n else 0,
        }
        print(f"    Original:    logprobs={lp_aligned}/{n}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"pairs": results, "summary": summary_data}, f, indent=2)
        print(f"    Saved {out_path}")

    # ── Summarized versions of extreme strings ─────────────────────────
    if run_summarized:
        print(f"    Generating summaries of extreme strings...")
        summary_client = _create_client()
        summaries = {}
        for ext in extreme_records:
            summary = _summarize_string(summary_client, PARAPHRASE_SUMMARIZER_MODEL, ext["description"])
            summaries[ext["id"]] = summary
            print(f"      {_group_for_category(ext.get('category', ''))} "
                  f"[{ext.get('feasibility', '?')}]: \"{summary}\"")

        summary_id_map = {}
        summary_options = []
        for ext in extreme_records:
            sid = -(ext["id"] + 1)
            summary_id_map[ext["id"]] = sid
            summary_options.append({
                "id": sid,
                "description": summaries[ext["id"]],
                "category": ext.get("category", ""),
                "source": "summary",
            })

        summary_pairs = [
            (ext, h, summary_id_map[ext["id"]])
            for ext in extreme_records for h in highlighted
        ]

        h_options = [options_by_id[h["id"]] for h in highlighted if h["id"] in options_by_id]
        summary_graph = PreferenceGraph(summary_options + h_options)
        summary_edge_indices = [(sid, h["id"]) for _, h, sid in summary_pairs]
        _, summary_prompts, summary_p2k = summary_graph.generate_prompts(
            summary_edge_indices, comparison_prompt,
        )

        print(f"    Summarized logprobs run ({len(summary_edge_indices)} pairs)...")
        s_lp_responses = await generate_responses(
            agent=agent, prompts=summary_prompts, K=1, use_logprobs=True,
        )
        s_logprob_probs: Dict[Tuple, List[float]] = {}
        for prompt_idx, response_list in s_lp_responses.items():
            A_id, B_id, direction = summary_p2k[prompt_idx]
            prob_a = response_list[0]["probability_A"]
            pair_key = (A_id, B_id)
            s_logprob_probs.setdefault(pair_key, []).append(
                1.0 - prob_a if direction == "flipped" else prob_a
            )
        s_logprob_probs_mean = {k: sum(v) / len(v) for k, v in s_logprob_probs.items()}

        summarized_results = []
        for ext, h, sid in summary_pairs:
            pair_key = (sid, h["id"])
            u_ext = _lookup_utility(utilities, ext["id"])
            u_h = _lookup_utility(utilities, h["id"])
            if u_ext is None or u_h is None:
                continue
            expected = "A" if u_ext > u_h else "B"
            lp_pa = s_logprob_probs_mean.get(pair_key, 0.5)
            summarized_results.append({
                "option_A": {
                    "id": ext["id"],
                    "original_description": ext["description"],
                    "summary": summaries[ext["id"]],
                    "group": _group_for_category(ext.get("category", "")),
                    "feasibility": ext.get("feasibility"),
                },
                "option_B": {
                    "id": h["id"],
                    "description": h["description"],
                    "label": HIGHLIGHTED_STRINGS.get(h["description"], ""),
                },
                "utility_A": u_ext,
                "utility_B": u_h,
                "expected_preference": expected,
                "logprobs": {
                    "P_A": round(lp_pa, 4),
                    "aligned": (lp_pa > 0.5) == (u_ext > u_h),
                },
            })

        s_lp_aligned = sum(1 for r in summarized_results if r["logprobs"]["aligned"])
        s_n = len(summarized_results)
        summarized_summary_data = {
            "n_pairs": s_n,
            "logprobs_aligned": s_lp_aligned,
            "logprobs_accuracy": round(s_lp_aligned / s_n, 3) if s_n else 0,
            "summaries": {str(k): v for k, v in summaries.items()},
        }
        print(f"    Summarized:  logprobs={s_lp_aligned}/{s_n}")

        out_path_summarized.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_summarized, "w") as f:
            json.dump(
                {"pairs": summarized_results, "summary": summarized_summary_data},
                f, indent=2,
            )
        print(f"    Saved {out_path_summarized}")


def _load_stage1_results(stage1_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load options and utilities from a completed Stage 1 run."""
    with open(stage1_dir / "options.json") as f:
        options = json.load(f)
    with open(stage1_dir / "utilities.json") as f:
        utilities = json.load(f)
    return options, utilities


def _extremes_for_linkage(
    extremes: Dict[str, Dict[str, Optional[Dict]]],
) -> Dict[str, Dict[str, Optional[Dict]]]:
    """Reduce a find_extreme_strings() result to the JSON-linkage shape."""
    out: Dict[str, Dict[str, Optional[Dict]]] = {}
    for feas, per_feas in extremes.items():
        out[feas] = {}
        for direction in ("best_euphorics",):
            opt = per_feas.get(direction)
            if opt is None:
                out[feas][direction] = None
            else:
                out[feas][direction] = {
                    "description": opt["description"],
                    "category": opt.get("category", ""),
                    "feasibility": opt.get("feasibility"),
                    "stage1_utility": opt["utility_mean"],
                }
    return out


def _print_extremes(extremes: Dict[str, Dict[str, Optional[Dict]]]):
    print(f"\n{'='*60}")
    print("STAGE 1 RESULTS — Extreme strings:")
    for feas in sorted(extremes.keys()):
        print(f"  [{feas}]")
        for direction in ("best_euphorics",):
            opt = extremes[feas].get(direction)
            if opt is None:
                print(f"    {direction}: (none found)")
            else:
                print(f"    {direction}: utility={opt['utility_mean']:.4f}")
                print(f"      {opt['description'][:120]}...")
    print(f"{'='*60}")


async def _maybe_run_verification(
    out_dir: Path,
    stage_prefix: str,
    prompt_key: str,
    comparison_prompt: str,
    agent,
    options: List[Dict],
    utilities: Dict,
    extremes: Dict[str, Dict[str, Optional[Dict]]],
    args,
    verify_options: Optional[List[Dict]] = None,
):
    """Run stage {1b,2b} verification for a single prompt unless results exist.

    *verify_options* is passed through to run_verification() to override the
    default per-feasibility extremes with a caller-provided set (e.g. all RL
    strings under --stage2-all-rl).
    """
    rerun = getattr(args, "rerun", False)
    verify_summarized = getattr(args, "verify_summarized", False)
    out_path = out_dir / f"{stage_prefix}_verification_{prompt_key}.json"
    out_path_summarized = (
        out_dir / f"{stage_prefix}_verification_summarized_{prompt_key}.json"
    )
    needs_original = not out_path.exists() or rerun
    needs_summarized = verify_summarized and (not out_path_summarized.exists() or rerun)
    if not needs_original and not needs_summarized:
        print(f"  {stage_prefix} ({prompt_key}): skipping (results exist)")
        return
    parts = []
    if needs_original:
        parts.append("original")
    if needs_summarized:
        parts.append("summarized")
    print(f"\n--- {stage_prefix} ({prompt_key}) ({'+'.join(parts)}) ---")
    await run_verification(
        out_dir=out_dir,
        stage_prefix=stage_prefix,
        prompt_key=prompt_key,
        comparison_prompt=comparison_prompt,
        agent=agent,
        options=options,
        utilities=utilities,
        extremes=extremes,
        verify_options=verify_options,
        run_original=needs_original,
        run_summarized=needs_summarized,
    )


async def run_two_stage(
    model_key: str,
    model_path: str,
    model_experiments: Dict[str, List[Dict]],
    feasibility_by_exp: Dict[str, str],
    baseline_options: List[Dict],
    args,
    config: Dict,
):
    """Run the full two-stage evaluation for one target model.

    Supports running stages independently via args.stage:
      "both"  — run Stage 1 then Stage 2 (default)
      "1"     — run Stage 1 only
      "2"     — run Stage 2 only (requires args.stage1_dir)
    """
    exp_names = list(model_experiments.keys())
    feasibility_options = sorted(set(feasibility_by_exp.values()))
    run_name = getattr(args, "run_name", None) or f"{model_key}_{args.experiment_suffix}"
    stage = getattr(args, "stage", "both")

    # When running stage 2 only, reuse the parent directory of the stage1 run
    if stage == "2":
        if not args.stage1_dir:
            print("ERROR: --stage1-dir is required when --stage 2")
            return
        stage1_dir = Path(args.stage1_dir)
        if not (stage1_dir / "utilities.json").exists():
            print(f"ERROR: No utilities.json found in {stage1_dir}")
            return
        # Derive base_output from stage1_dir: .../stage1/prefer -> ...
        base_output = stage1_dir.parent.parent
        run_datetime = base_output.name
    else:
        run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = Path(args.output_dir) / run_name / run_datetime

    prefer_prompt = build_comparison_prompt(COMPARISON_PROMPTS["prefer"])

    agent = None
    stage1_options: Optional[List[Dict]] = None
    stage1_utilities: Optional[Dict] = None

    # ── Stage 1: rank RL strings + baselines ────────────────────────────
    if stage in ("both", "1"):
        print(f"\n{'#'*60}")
        print(f"# STAGE 1: Initial ranking ({model_key}) — feasibilities: {feasibility_options}")
        print(f"{'#'*60}")

        stage1_options = build_options(
            baseline_options, model_experiments, args.top_k,
            feasibility_by_exp=feasibility_by_exp,
        )
        n_rl = sum(1 for o in stage1_options if o.get("source") == "rl_training")
        print(f"  Options: {len(stage1_options)} ({n_rl} from RL)")

        print(f"  Loading model: {model_path}")
        agent = create_agent(model_path)

        stage1_dir = base_output / "stage1" / "prefer"
        stage1_metadata, stage1_utilities = await run_ranking(
            options=stage1_options,
            agent=agent,
            model_path=model_path,
            output_dir=stage1_dir,
            experiment_name=f"{model_key}/stage1/prefer",
            config=config,
            comparison_prompt=prefer_prompt,
        )
        _plot_stage1(stage1_dir)

    # ── Load Stage 1 results (if running stage 2 from existing) ─────────
    if stage == "2":
        stage1_dir = Path(args.stage1_dir)
        print(f"\nLoading Stage 1 results from {stage1_dir}")
        stage1_options, stage1_utilities = _load_stage1_results(stage1_dir)

    # ── Find extreme strings (per feasibility) ──────────────────────────
    extremes = find_extreme_strings(stage1_options, stage1_utilities)
    _print_extremes(extremes)
    extreme_list = flatten_extremes(extremes)

    # Override for --stage2-all-rl: include every Stage 1 RL string, not just
    # the per-feasibility extremes. ``extremes`` is left alone so that stage1b
    # verification and stage_linkage.json still reflect the extremes.
    if getattr(args, "stage2_all_rl", False):
        full_rl = []
        for opt in stage1_options:
            if opt.get("source") != "rl_training":
                continue
            uid_str = str(opt["id"])
            util_entry = stage1_utilities.get(uid_str) or stage1_utilities.get(opt["id"])
            if util_entry is None:
                continue
            full_rl.append({**opt, "utility_mean": float(util_entry["mean"])})
        print(f"  --stage2-all-rl: Stage 2 will include {len(full_rl)} RL strings "
              f"(was {len(extreme_list)} extremes)")
        extreme_list = full_rl

    # ── Stage 1b: Head-to-head verification ─────────────────────────────
    # Run while the target model is already loaded (both and "2" paths).
    if extreme_list:
        if stage == "2" and agent is None:
            print(f"  Loading model for Stage 1b: {model_path}")
            agent = create_agent(model_path)
        if agent is not None:
            # Under --stage2-all-rl, verify every RL string (not just the
            # per-feasibility extremes) against the highlighted baselines.
            stage1b_verify = None
            if getattr(args, "stage2_all_rl", False):
                stage1b_verify = [
                    opt for opt in stage1_options
                    if opt.get("source") == "rl_training"
                    and (str(opt["id"]) in stage1_utilities
                         or opt["id"] in stage1_utilities)
                ]
            await _maybe_run_verification(
                out_dir=stage1_dir,
                stage_prefix="stage1b",
                prompt_key="prefer",
                comparison_prompt=prefer_prompt,
                agent=agent,
                options=stage1_options,
                utilities=stage1_utilities,
                extremes=extremes,
                args=args,
                verify_options=stage1b_verify,
            )
    else:
        print("WARNING: No extreme strings found in Stage 1.")

    if stage == "1":
        # Stage 1 only — save linkage and stop
        linkage = {
            "model_key": model_key,
            "model_path": model_path,
            "run_name": run_name,
            "run_datetime": run_datetime,
            "experiments": exp_names,
            "feasibility_options": feasibility_options,
            "stage1_dir": "stage1/prefer",
            "stage2_dirs": None,
            "extremes_by_feasibility": _extremes_for_linkage(extremes),
        }
        linkage_path = base_output / "stage_linkage.json"
        linkage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(linkage_path, "w") as f:
            json.dump(linkage, f, indent=2)
        print(f"\nSaved stage linkage: {linkage_path}")
        print(f"\nStage 1 complete for {model_key}.")
        print(f"  Results: {stage1_dir}")
        print(f"  To run Stage 2: --stage 2 --stage1-dir {stage1_dir}")

        if agent is not None:
            del agent
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # ── Stage 2 ─────────────────────────────────────────────────────────
    if not extreme_list:
        print("WARNING: Skipping Stage 2 — no extreme strings.")
        if agent is not None:
            del agent
        return

    # ── Generate variations ─────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# GENERATING VARIATIONS ({len(extreme_list)} strings × {args.n_per_type * 2} each)")
    print(f"{'#'*60}")

    cache_path = stage1_dir / f"variations_stage2.json"
    variation_options = generate_stage2_variations(
        extreme_options=extreme_list,
        n_per_type=args.n_per_type,
        cache_path=cache_path,
        model=args.variation_model,
        max_tokens=args.variation_max_tokens,
    )
    print(f"  Generated {len(variation_options)} total variations")

    # ── Stage 2: rank variations + originals + baselines ────────────────
    print(f"\n{'#'*60}")
    print(f"# STAGE 2: Variation ranking ({model_key})")
    print(f"{'#'*60}")

    stage2_options = build_stage2_options(
        baseline_options, extreme_list, variation_options,
    )

    # Add combinations for zero-point fitting (from baselines only)
    combos = generate_combinations(baseline_options, seed=args.seed)
    next_id = max(o["id"] for o in stage2_options) + 1
    for combo in combos:
        combo["id"] = next_id
        next_id += 1
    stage2_options.extend(combos)

    n_rl2 = sum(1 for o in stage2_options if o.get("source") == "rl_training")
    n_var = sum(1 for o in stage2_options if o.get("source") == "variation")
    n_combo = sum(1 for o in stage2_options if o.get("source") == "combination")
    print(f"  Options: {len(stage2_options)} ({n_rl2} RL, {n_var} variations, {n_combo} combinations)")

    # Load model if not already loaded (stage 2 standalone)
    if agent is None:
        print(f"  Loading model: {model_path}")
        agent = create_agent(model_path)

    # Resolve Stage 2 comparison prompts
    stage2_prompts = _resolve_stage2_prompts(args)

    # Run ranking for each comparison prompt (skip already-completed ones)
    stage2_dir = base_output / "stage2"
    stage2_dirs = {}
    for prompt_key, prompt_header in stage2_prompts.items():
        prompt_dir = stage2_dir / prompt_key
        comparison_prompt = build_comparison_prompt(prompt_header)
        s2_utilities: Optional[Dict] = None
        if (prompt_dir / "utilities.json").exists() and not getattr(args, "rerun", False):
            print(f"\n  --- Stage 2 prompt: {prompt_key} — skipping (results exist) ---")
            stage2_dirs[prompt_key] = str(prompt_dir)
            with open(prompt_dir / "utilities.json") as f:
                s2_utilities = json.load(f)
            # Fit ZP if missing (e.g. from an older run)
            if not (prompt_dir / "zero_point.json").exists():
                _fit_and_save_zero_point(stage2_options, s2_utilities, prompt_dir)
        else:
            print(f"\n  --- Stage 2 prompt: {prompt_key} ---")
            s2_metadata, s2_utilities = await run_ranking(
                options=stage2_options,
                agent=agent,
                model_path=model_path,
                output_dir=prompt_dir,
                experiment_name=f"{model_key}/stage2/{prompt_key}",
                config=config,
                comparison_prompt=comparison_prompt,
            )
            _fit_and_save_zero_point(stage2_options, s2_utilities, prompt_dir)
            stage2_dirs[prompt_key] = str(prompt_dir)

        # ── Stage 2b: Head-to-head verification for this prompt ─────────
        if extreme_list and s2_utilities is not None:
            stage2b_verify = None
            if getattr(args, "stage2_all_rl", False):
                stage2b_verify = [
                    opt for opt in stage2_options
                    if opt.get("source") == "rl_training"
                ]
            await _maybe_run_verification(
                out_dir=prompt_dir,
                stage_prefix="stage2b",
                prompt_key=prompt_key,
                comparison_prompt=comparison_prompt,
                agent=agent,
                options=stage2_options,
                utilities=s2_utilities,
                extremes=extremes,
                args=args,
                verify_options=stage2b_verify,
            )

        # ── Per-prompt ZP strip plots ───────────────────────────────────
        if (prompt_dir / "zero_point.json").exists():
            _plot_prompt(prompt_dir, prompt_key)

    # ── Save stage linkage metadata ─────────────────────────────────────
    linkage = {
        "model_key": model_key,
        "model_path": model_path,
        "run_name": run_name,
        "run_datetime": run_datetime,
        "experiments": exp_names,
        "feasibility_options": feasibility_options,
        "stage1_dir": "stage1/prefer",
        "stage2_dirs": {k: f"stage2/{k}" for k in stage2_prompts},
        "stage2_prompts": {k: v for k, v in stage2_prompts.items()},
        "extremes_by_feasibility": _extremes_for_linkage(extremes),
        "n_variations_per_type": args.n_per_type,
        "variation_types": VARIATION_TYPES,
        "total_variations": len(variation_options),
    }

    linkage_path = base_output / "stage_linkage.json"
    linkage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(linkage_path, "w") as f:
        json.dump(linkage, f, indent=2)
    print(f"\nSaved stage linkage: {linkage_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    del agent
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nStage 2 complete for {model_key}.")
    print(f"  Stage 1: {base_output / 'stage1' / 'prefer'}")
    for pk, pd in stage2_dirs.items():
        print(f"  Stage 2 ({pk}): {pd}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Thurstonian evaluation: rank, then vary extremes",
    )
    parser.add_argument(
        "--stage", choices=["1", "2", "both"], default="both",
        help="Which stage to run: 1 (rank only), 2 (variations + re-rank), both (default)",
    )
    parser.add_argument(
        "--stage1-dir", default=None,
        help="Path to Stage 1 results directory (required when --stage 2). "
             "E.g. sbatch_results/evaluate/{run_name}/{datetime}/stage1/prefer",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to target model (overrides auto-detection from model key)",
    )
    parser.add_argument(
        "--model-key", default=None,
        help="Restrict to a single model key (e.g. gemma27b, llama70b, qwen72b)",
    )
    parser.add_argument(
        "--experiment-suffix", required=True,
        help="Base hyperparam+policy suffix, e.g. ent005_kl01_div10_igdiv10_llama8b "
             "(feasibility is supplied separately via --feasibility-options)",
    )
    parser.add_argument(
        "--feasibility-options", required=True,
        help="Comma-separated feasibility options to include in the ranking. "
             "E.g. mundanity_realism,agent_feasibility,feasibility,none",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Override run directory name (default: {model_key}_{experiment_suffix})",
    )
    parser.add_argument(
        "--source", choices=["val", "buffer"], default="buffer",
        help="Read best strings from val or buffer analysis (default: buffer)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of top RL strings per experiment for Stage 1 (default: all)",
    )
    parser.add_argument(
        "--n-per-type", type=int, default=5,
        help="Number of variations per type (paraphrase + clause_reorder) (default: 5)",
    )
    parser.add_argument(
        "--stage2-prompts", default=None,
        help="Comma-separated comparison prompt keys or custom headers for Stage 2. "
             "Built-in keys: prefer, more_happy. Custom headers are used verbatim. "
             "(default: prefer,more_happy)",
    )
    parser.add_argument(
        "--variation-model", default=PARAPHRASE_SUMMARIZER_MODEL,
        help=f"LLM model for variation generation (default: {PARAPHRASE_SUMMARIZER_MODEL})",
    )
    parser.add_argument(
        "--variation-max-tokens", type=int, default=4096,
        help="Max tokens for variation generation (default: 4096)",
    )
    parser.add_argument(
        "--best-strings", default=None,
        help="Path to best_strings JSON (default: auto from --source)",
    )
    parser.add_argument(
        "--baseline-options", default=str(DEFAULT_BASELINE_OPTIONS),
        help="Path to baseline options JSON",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--stage2-all-rl", action="store_true", default=False,
        help="Stage 2 includes ALL Stage 1 RL-training strings (plus their variations), "
             "not just the per-feasibility extremes. Useful when you want a full within-"
             "feasibility ranking rather than a best-vs-worst comparison.",
    )
    parser.add_argument("--rerun", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--verify-summarized", action="store_true", default=False,
                        help="Also run Stage 1b verification with summarized (≤10 word) strings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-multiplier", type=float, default=3.0)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--K", type=int, default=5)
    args = parser.parse_args()

    if args.best_strings is None:
        args.best_strings = str(
            BEST_STRINGS_DIR / f"best_strings_by_experiment_{args.source}.json"
        )

    # Load data
    baseline_options = load_baseline_options(Path(args.baseline_options))
    print(f"Loaded {len(baseline_options)} baseline options")

    best_strings = load_best_strings(Path(args.best_strings))
    print(f"Found {len(best_strings)} experiments in {args.best_strings}")

    grouped = group_experiments_by_model(best_strings)
    print(f"Grouped into {len(grouped)} models: {list(grouped.keys())}")

    # Parse feasibility options and build the expected suffix set
    feasibility_options = [
        f.strip() for f in args.feasibility_options.split(",") if f.strip()
    ]
    if not feasibility_options:
        print("ERROR: --feasibility-options is empty")
        sys.exit(1)
    base_suffix = args.experiment_suffix
    expected_suffixes = {f"{feas}_{base_suffix}": feas for feas in feasibility_options}

    # Filter by (feasibility, base_suffix) pairs and record feasibility per experiment
    filtered: Dict[str, Dict[str, List[Dict]]] = {}
    feasibility_by_exp_per_model: Dict[str, Dict[str, str]] = {}
    for model_key, exps in grouped.items():
        matching: Dict[str, List[Dict]] = {}
        feas_map: Dict[str, str] = {}
        for k, v in exps.items():
            stripped = strip_direction_prefix(k)
            suffix = re.sub(r"^(llama70b|qwen72b|gemma27b)_", "", stripped)
            if suffix in expected_suffixes:
                matching[k] = v
                feas_map[k] = expected_suffixes[suffix]
        if matching:
            filtered[model_key] = matching
            feasibility_by_exp_per_model[model_key] = feas_map
    if not filtered:
        all_names = sorted(
            {strip_direction_prefix(k) for exps in grouped.values() for k in exps}
        )
        print(
            f"ERROR: No experiments matched suffix '{base_suffix}' with feasibilities "
            f"{feasibility_options}. Available stripped names: {all_names}"
        )
        sys.exit(1)
    grouped = filtered

    # Filter to a single model key
    if args.model_key:
        if args.model_key not in grouped:
            print(f"ERROR: Model key '{args.model_key}' not found. Available: {list(grouped.keys())}")
            sys.exit(1)
        grouped = {args.model_key: grouped[args.model_key]}
        feasibility_by_exp_per_model = {
            args.model_key: feasibility_by_exp_per_model[args.model_key]
        }

    config = {
        "seed": args.seed,
        "edge_multiplier": args.edge_multiplier,
        "num_epochs": args.num_epochs,
        "K": args.K,
        "degree": 2,
        "num_edges_per_iteration": 500,
        "P": 10.0,
        "Q": 20.0,
        "learning_rate": 0.01,
        "unparseable_mode": "distribution",
        "use_pseudolabels": False,
        "pseudolabel_confidence_threshold": 0.95,
        "holdout_fraction": 0.05,
        "system_message": "You are a helpful assistant.",
    }

    for model_key, model_experiments in grouped.items():
        model_path = args.model or MODEL_PATH_MAP.get(model_key)
        if not model_path:
            print(f"ERROR: No model path for '{model_key}'. Use --model to specify.")
            sys.exit(1)

        # Check for existing results — only skip if ALL prompts are done
        run_name = args.run_name or f"{model_key}_{args.experiment_suffix}"
        run_dir = Path(args.output_dir) / run_name
        if run_dir.exists() and not args.rerun:
            stage2_prompts_check = _resolve_stage2_prompts(args)
            # Find the latest run directory
            latest_runs = sorted(run_dir.glob("*/stage2/"), key=lambda p: p.parent.name)
            if latest_runs:
                latest_stage2 = latest_runs[-1]
                completed = [
                    pk for pk in stage2_prompts_check
                    if (latest_stage2 / pk / "utilities.json").exists()
                ]
                if len(completed) == len(stage2_prompts_check):
                    print(f"\nSkipping {model_key}: all {len(completed)} prompts complete at {latest_stage2}")
                    print(f"  (set --rerun to force)")
                    continue
                elif completed:
                    print(f"\n{model_key}: {len(completed)}/{len(stage2_prompts_check)} prompts complete, resuming...")

        asyncio.run(run_two_stage(
            model_key=model_key,
            model_path=model_path,
            model_experiments=model_experiments,
            feasibility_by_exp=feasibility_by_exp_per_model[model_key],
            baseline_options=baseline_options,
            args=args,
            config=config,
        ))


if __name__ == "__main__":
    main()
