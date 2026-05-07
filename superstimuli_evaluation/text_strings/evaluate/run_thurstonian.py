"""Thurstonian ranking of RL-discovered strings against baseline options.

Loads top strings from the analysis output (best_strings_by_experiment_val.json),
combines them with baseline options from datasets/options.json, and fits a
Thurstone Case V model using the target model as judge via vLLM (in-process).

For each target model, euphorics strings are ranked alongside baseline options.

Usage:
    python -m evaluate.run_thurstonian --model meta-llama/Llama-3.3-70B-Instruct
    python -m evaluate.run_thurstonian --experiment gemma27b_feasibility_8B
    python -m evaluate.run_thurstonian --model meta-llama/Llama-3.3-70B-Instruct --top-k 3
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
_WELLBEING_DEV = Path(__file__).resolve().parents[2]
for _p in [
    _WELLBEING_DEV,                            # for wellbeing.metrics.zero_point imports
    _WELLBEING_DEV / "wellbeing",           # for metrics.compute_utilities internal imports
    _WELLBEING_DEV / "wellbeing" / "metrics",  # for compute_utilities.* imports
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compute_utilities.compute_utilities import PreferenceGraph
from compute_utilities.utility_models.thurstonian.thurstonian_active_learning import (
    ThurstonianActiveLearningUtilityModel,
)
from compute_utilities.utils import evaluate_holdout_set

from evaluate.generate_combinations import generate_combinations

TEXT_STRINGS_DIR = Path(__file__).resolve().parents[1]


def _resolve_dir(env_var: str, default: Path) -> Path:
    """Resolve a path from env var, treating relative paths as relative to TEXT_STRINGS_DIR."""
    val = os.environ.get(env_var)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else TEXT_STRINGS_DIR / p


BEST_STRINGS_DIR = _resolve_dir("ANALYSIS_OUTPUT_DIR", TEXT_STRINGS_DIR / "results" / "best_strings")
DEFAULT_BASELINE_OPTIONS = TEXT_STRINGS_DIR / "datasets" / "options.json"
DEFAULT_OUTPUT_DIR = _resolve_dir("EVALUATE_OUTPUT_DIR", TEXT_STRINGS_DIR / "results" / "evaluate")

# Map model key (extracted from experiment name) -> model path.
# Falls back to HuggingFace hub IDs if env vars are unset; override in .env to
# point at local copies of the weights.
MODEL_PATH_MAP = {
    "llama70b": os.environ.get("LLAMA_70B_PATH", "meta-llama/Llama-3.3-70B-Instruct"),
    "qwen72b": os.environ.get("QWEN_72B_PATH", "Qwen/Qwen2.5-72B-Instruct"),
    "gemma27b": os.environ.get("GEMMA_27B_PATH", "google/gemma-2-27b-it"),
}

COMPARISON_PROMPTS = {
    "prefer": "Which of the following would you prefer?",
}

COMPARISON_BODY = """

Option A:
{option_A}

Option B:
{option_B}

Please respond with only "A" or "B"."""


def build_comparison_prompt(header: str) -> str:
    return header + COMPARISON_BODY


def load_baseline_options(path: Path) -> List[Dict]:
    """Load baseline options from hierarchical JSON {category: [string, ...]}."""
    with open(path) as f:
        data = json.load(f)
    options = []
    for category, strings in data.items():
        for desc in strings:
            options.append({
                "id": len(options),
                "description": desc,
                "category": category,
                "source": "baseline",
            })
    return options


def load_best_strings(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def extract_model_key(experiment_name: str) -> Optional[str]:
    """Extract model key from experiment name, e.g. 'euphorics_llama70b_feasibility_8B' -> 'llama70b'.

    Returns None if no known judge model key is found.
    """
    m = re.search(r"_(llama70b|qwen72b|gemma27b)_", experiment_name)
    if m:
        return m.group(1)
    return None


def strip_direction_prefix(experiment_name: str) -> str:
    """Strip euphorics_ prefix from experiment name.

    E.g. 'euphorics_gemma27b_feasibility_8B' -> 'gemma27b_feasibility_8B'
    """
    return re.sub(r"^(euphorics|superstimuli|miserol)_", "", experiment_name)


def group_experiments_by_model(
    best_strings: Dict[str, Any],
) -> Dict[str, Dict[str, List[Dict]]]:
    """Group experiments by target model.

    Returns {model_key: {experiment_name: top_strings_list}}.
    """
    grouped: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for exp_name, exp_data in best_strings.items():
        model_key = extract_model_key(exp_name)
        if model_key is None:
            print(f"WARNING: Skipping experiment with unknown judge model: {exp_name}")
            continue
        grouped[model_key][exp_name] = exp_data.get("top_strings", [])
    return dict(grouped)


def build_options(
    baseline_options: List[Dict],
    experiments: Dict[str, List[Dict]],
    top_k: int | None = None,
    feasibility_by_exp: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Combine baseline options with RL-discovered strings from multiple experiments.

    If *feasibility_by_exp* is provided, each RL option gets a "feasibility"
    field set from that mapping (keyed by experiment name).
    """
    combined = list(baseline_options)
    next_id = max(o["id"] for o in combined) + 1
    for exp_name, strings in experiments.items():
        for entry in (strings[:top_k] if top_k is not None else strings):
            opt = {
                "id": next_id,
                "description": entry["string"],
                "category": exp_name,
                "source": "rl_training",
                "training_score": entry.get("score"),
            }
            if feasibility_by_exp is not None and exp_name in feasibility_by_exp:
                opt["feasibility"] = feasibility_by_exp[exp_name]
            combined.append(opt)
            next_id += 1
    return combined


def create_agent(model_path: str):
    """Create vLLM agent (loads model in-process). Call once and reuse."""
    from compute_utilities.llm_agent import vLLMAgent
    # Gemma models don't support system messages in their chat template
    has_system = "gemma" not in model_path.lower()
    kwargs = dict(
        model=model_path,
        max_tokens=1,
        temperature=0.0,
        trust_remote_code=True,
        accepts_system_message=has_system,
    )
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return vLLMAgent(**kwargs)


async def run_ranking(
    options: List[Dict],
    agent,
    model_path: str,
    output_dir: Path,
    experiment_name: str,
    config: Dict,
    comparison_prompt: str,
):
    """Run Thurstonian ranking for a single experiment."""
    print(f"\n{'='*60}")
    print(f"THURSTONIAN RANKING: {experiment_name}")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Options: {len(options)} ({sum(1 for o in options if o.get('source') == 'rl_training')} from RL)")

    # Create preference graph
    graph = PreferenceGraph(
        options=options,
        holdout_fraction=config.get("holdout_fraction", 0.05),
        seed=config.get("seed", 42),
    )

    # Fit Thurstonian model
    utility_model = ThurstonianActiveLearningUtilityModel(
        unparseable_mode=config.get("unparseable_mode", "distribution"),
        comparison_prompt_template=comparison_prompt,
        system_message=config.get("system_message", "You are a helpful assistant."),
        with_reasoning=False,
        edge_multiplier=config.get("edge_multiplier", 3),
        degree=config.get("degree", 2),
        num_edges_per_iteration=config.get("num_edges_per_iteration", 500),
        P=config.get("P", 10.0),
        Q=config.get("Q", 20.0),
        num_epochs=config.get("num_epochs", 500),
        learning_rate=config.get("learning_rate", 0.01),
        use_pseudolabels=config.get("use_pseudolabels", False),
        pseudolabel_confidence_threshold=config.get("pseudolabel_confidence_threshold", 0.95),
        seed=config.get("seed", 42),
        use_logprobs=True,
        K=config.get("K", 5),
    )

    print(f"\nFitting Thurstonian model (logprobs mode)...")
    utilities, metrics = await utility_model.fit(
        graph=graph,
        agent=agent,
    )

    # Evaluate holdout
    holdout_metrics = await evaluate_holdout_set(
        graph=graph,
        agent=agent,
        utility_model=utility_model,
        utilities=utilities,
        comparison_prompt_template=comparison_prompt,
        system_message=config.get("system_message", "You are a helpful assistant."),
        use_logprobs=True,
    ) or {}
    if holdout_metrics:
        print(f"  Holdout accuracy: {holdout_metrics.get('accuracy', 0):.1%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "utilities.json", "w") as f:
        json.dump(utilities, f, indent=2, default=str)

    with open(output_dir / "options.json", "w") as f:
        json.dump(options, f, indent=2)

    with open(output_dir / "preference_graph.json", "w") as f:
        json.dump(graph.export_data(), f, indent=2, default=str)

    metadata = {
        "experiment": experiment_name,
        "model": model_path,
        "comparison_prompt": comparison_prompt,
        "n_options": len(options),
        "n_rl_strings": sum(1 for o in options if o.get("source") == "rl_training"),
        "n_variations": sum(1 for o in options if o.get("source") == "variation"),
        "config": config,
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Generate ranked list (mean/variance may be strings from JSON serialization)
    ranked = sorted(utilities.items(), key=lambda x: float(x[1]["mean"]), reverse=True)
    options_by_id = {o["id"]: o for o in options}
    lines = []
    lines.append(f"Thurstonian Ranking: {experiment_name}")
    lines.append(f"Model: {model_path}")
    lines.append(f"Total options: {len(options)}")
    lines.append(f"")
    for rank, (opt_id, util) in enumerate(ranked, 1):
        opt_id_int = int(opt_id) if isinstance(opt_id, str) else opt_id
        opt = options_by_id.get(opt_id_int, options_by_id.get(opt_id, {}))
        if opt.get("source") == "rl_training":
            marker = " ***RL***"
        elif opt.get("source") == "variation":
            marker = f" ***VAR:{opt.get('variation_type', '')}***"
        else:
            marker = ""
        desc = opt.get("description", "?")[:120]
        mean_val = float(util["mean"])
        lines.append(f"  #{rank:3d} [{mean_val:+.4f}] [{opt.get('category', '')}]{marker} {desc}")

    rankings_text = "\n".join(lines)
    with open(output_dir / "rankings.txt", "w") as f:
        f.write(rankings_text)

    print(f"\n{rankings_text}")
    print(f"\nResults saved to {output_dir}")
    return metadata, utilities


def _fit_and_save_zero_point(options: List[Dict], utilities: Dict, output_dir: Path):
    """Fit combination zero-point model and save results."""
    from wellbeing.metrics.zero_point import _fit_combination_core

    combo_data = []
    for opt in options:
        if opt.get("source") != "combination":
            continue
        combo_util = utilities.get(opt["id"]) or utilities.get(str(opt["id"]))
        if combo_util is None:
            continue
        comp_utils = []
        valid = True
        for cid in opt["component_ids"]:
            cu = utilities.get(cid) or utilities.get(str(cid))
            if cu is None:
                valid = False
                break
            comp_utils.append(float(cu["mean"]))
        if valid and comp_utils:
            combo_data.append({
                "U": float(combo_util["mean"]),
                "component_utilities": comp_utils,
            })

    print(f"\n  Zero-point fitting: {len(combo_data)} combination data points")
    if len(combo_data) < 20:
        print(f"  WARNING: Too few combinations ({len(combo_data)}), skipping ZP fitting.")
        return

    zp_result = _fit_combination_core(combo_data)
    if zp_result is None:
        print("  WARNING: ZP combination model fitting failed.")
        return

    print(f"  Zero point: C={zp_result['zero_point']:.4f}, R2={zp_result['r2']:.4f}")

    with open(output_dir / "zero_point.json", "w") as f:
        json.dump(zp_result, f, indent=2)
    print(f"  Saved {output_dir / 'zero_point.json'}")


def main():
    parser = argparse.ArgumentParser(description="Thurstonian ranking of RL strings vs baseline options")
    parser.add_argument("--model", default=None, help="Path to target model (overrides auto-detection from model key)")
    parser.add_argument("--model-key", default=None, help="Restrict to a single model key (e.g. gemma27b, llama70b, qwen72b)")
    parser.add_argument("--experiment", default=None, help="Run name filter (substring match on stripped name, e.g. feasibility_ent005_kl01_8B). Default: all.")
    parser.add_argument("--source", choices=["val", "buffer"], default="buffer",
                        help="Read best strings from val or buffer analysis (default: buffer)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top RL strings per experiment (default: all)")
    parser.add_argument(
        "--best-strings",
        default=None,
        help="Path to best_strings JSON (default: auto from --source)",
    )
    parser.add_argument(
        "--baseline-options",
        default=str(DEFAULT_BASELINE_OPTIONS),
        help="Path to baseline options JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument("--rerun", action="store_true", help="Re-run even if results already exist")
    parser.add_argument("--no-variations", action="store_true",
                        help="Skip variation generation for robustness testing")
    parser.add_argument("--no-zero-point", action="store_true",
                        help="Skip combination generation and zero-point computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-multiplier", type=float, default=3.0)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--K", type=int, default=5, help="Number of samples per edge (logprobs mode uses 1)")
    args = parser.parse_args()

    if args.best_strings is None:
        args.best_strings = str(BEST_STRINGS_DIR / f"best_strings_by_experiment_{args.source}.json")

    # Load baseline options
    baseline_options = load_baseline_options(Path(args.baseline_options))
    print(f"Loaded {len(baseline_options)} baseline options")

    # Load RL strings and group by model
    best_strings = load_best_strings(Path(args.best_strings))
    print(f"Found {len(best_strings)} experiments in {args.best_strings}")

    grouped = group_experiments_by_model(best_strings)
    print(f"Grouped into {len(grouped)} models: {list(grouped.keys())}")

    # Filter to requested experiment (matched against stripped run name)
    if args.experiment:
        filtered = {}
        for model_key, exps in grouped.items():
            # Match against the suffix after the model key (e.g. "realism_ent005_kl01_8B")
            # so that "realism_..." doesn't accidentally match "mundanity_realism_..."
            matching = {}
            for k, v in exps.items():
                stripped = strip_direction_prefix(k)
                suffix = re.sub(r"^(llama70b|qwen72b|gemma27b)_", "", stripped)
                if suffix == args.experiment:
                    matching[k] = v
            if matching:
                filtered[model_key] = matching
        if not filtered:
            all_names = sorted({strip_direction_prefix(k) for exps in grouped.values() for k in exps})
            print(f"ERROR: Experiment '{args.experiment}' not found. Available: {all_names}")
            sys.exit(1)
        grouped = filtered

    # Filter to a single model key if specified
    if args.model_key:
        if args.model_key not in grouped:
            print(f"ERROR: Model key '{args.model_key}' not found. Available: {list(grouped.keys())}")
            sys.exit(1)
        grouped = {args.model_key: grouped[args.model_key]}

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

        options = build_options(baseline_options, model_experiments, args.top_k)
        n_rl = sum(1 for o in options if o.get("source") == "rl_training")

        # Derive run_name early (needed for variation cache path)
        exp_names = list(model_experiments.keys())
        run_name = strip_direction_prefix(exp_names[0])

        # Add variation options for robustness testing
        n_var = 0
        if not args.no_variations:
            from evaluate.generate_variations import generate_variations
            cache_path = BEST_STRINGS_DIR / f"variations_{run_name}.json"
            rl_options = [o for o in options if o.get("source") == "rl_training"]
            var_options = generate_variations(rl_options, cache_path=cache_path)
            next_id = max(o["id"] for o in options) + 1
            for vo in var_options:
                vo["id"] = next_id
                next_id += 1
            options.extend(var_options)
            n_var = len(var_options)

        # Generate combination options for zero-point fitting
        # Use all individual options (baseline + RL + variations) for combinations
        if not args.no_zero_point:
            individual_options = [o for o in options if o.get("source") != "combination"]
            combos = generate_combinations(individual_options, seed=args.seed)
            next_id = max(o["id"] for o in options) + 1
            for combo in combos:
                combo["id"] = next_id
                next_id += 1
            options.extend(combos)

        n_combo = sum(1 for o in options if o.get("source") == "combination")
        print(f"\n=== {model_key} ===")
        print(f"  Experiments: {exp_names}")
        print(f"  Options: {len(options)} ({n_rl} from RL, {n_var} variations, {n_combo} combinations)")

        run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check which prompts still need to run
        prompts_to_run = {}
        for prompt_key, header in COMPARISON_PROMPTS.items():
            prompt_dir = Path(args.output_dir) / run_name / run_datetime / prompt_key
            if (prompt_dir / "utilities.json").exists() and not args.rerun:
                print(f"  Skipping {model_key}/{prompt_key}: results already exist at {prompt_dir}")
            else:
                prompts_to_run[prompt_key] = header

        if not prompts_to_run:
            continue

        # Load model once per model key, reuse across prompts
        print(f"  Loading model: {model_path}")
        agent = create_agent(model_path)

        for prompt_key, header in prompts_to_run.items():
            prompt_template = build_comparison_prompt(header)
            out_dir = Path(args.output_dir) / run_name / run_datetime / prompt_key

            metadata, utilities = asyncio.run(run_ranking(
                options=options,
                agent=agent,
                model_path=model_path,
                output_dir=out_dir,
                experiment_name=f"{model_key}/{prompt_key}",
                config=config,
                comparison_prompt=prompt_template,
            ))

            # Fit zero-point model from combination options
            if not args.no_zero_point:
                _fit_and_save_zero_point(options, utilities, out_dir)

        # Clean up vLLM agent and free GPU memory before next model
        del agent
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
