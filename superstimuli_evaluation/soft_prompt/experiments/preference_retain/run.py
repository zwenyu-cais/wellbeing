#!/usr/bin/env python3
"""Preference retention evaluation.

Measures whether a soft prompt distorts the model's preferences by comparing
utility rankings with and without the intervention.

Full mode (baseline + intervention + compare in one command):
    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run --full \\
        --model qwen25-32b-instruct --stimulant-type euphorics

    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run --full \\
        --model qwen25-32b-instruct --stimulant-type both

Individual steps:
    # Step 1: Baseline ranking
    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run \\
        --model qwen25-32b-instruct

    # Step 2: Intervention ranking
    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run \\
        --model qwen25-32b-instruct --stimulant-type euphorics

    # Step 3: Compare
    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run --compare \\
        --model qwen25-32b-instruct --stimulant-type euphorics

Dry run:
    python -m superstimuli_evaluation.soft_prompt.experiments.preference_retain.run --full --dry-run \\
        --model qwen25-32b-instruct --stimulant-type euphorics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure wellbeing-dev root is importable
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[3])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

# thurstonian.py uses bare `from compute_utilities.*` imports, so we must
# put wellbeing/utils (NOT wellbeing/) on sys.path.  Using both paths causes
# the same file to be loaded as two different modules, breaking isinstance().
_WELLBEING_UTILS = os.path.join(_WELLBEING_DEV_ROOT, "wellbeing", "utils")
if _WELLBEING_UTILS not in sys.path:
    sys.path.insert(0, _WELLBEING_UTILS)

from superstimuli_evaluation.soft_prompt.configs import load_experiment_config, EVAL_OUTPUTS_DIR
from superstimuli_evaluation.soft_prompt.experiments.bubble_gum.options import load_options
from .metrics import compute_correlation_from_dirs, plot_correlation, quality_label

EXPERIMENT_NAME = "preference_retain"

SP_AUGMENTATION_TYPES = {"euphorics"}


def parse_args() -> argparse.Namespace:
    # Load defaults from experiments.yaml
    exp_cfg = load_experiment_config(EXPERIMENT_NAME)
    defaults = exp_cfg.get("arguments", {})

    parser = argparse.ArgumentParser(
        description=exp_cfg.get("description", "Preference retention evaluation"),
    )
    # Mode selection
    parser.add_argument("--full", action="store_true",
                        help="Run all 3 steps: baseline ranking, intervention ranking, then compare")
    parser.add_argument("--compare", action="store_true",
                        help="Compare two saved rankings instead of running a new one")

    # Ranking mode args
    parser.add_argument("--model", type=str, default=defaults.get("model"),
                        help="Model key from models.yaml")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: from experiments.yaml)")
    parser.add_argument("--options-path", type=str, default=None,
                        help="Override options JSON path")
    parser.add_argument("--config-name", type=str, default=defaults.get("config_name", "thurstonian"),
                        help="Preset from compute_utilities.yaml")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run even if per-rep results already exist")

    # Soft prompt args
    sp_defaults = exp_cfg.get("soft_prompt_arguments", {})
    parser.add_argument("--stimulant-type", type=str, nargs="+", default=None,
                        choices=["euphorics"],
                        help="Stimulant types for intervention rankings. "
                             "Each type gets its own ranking compared against baseline.")
    parser.add_argument("--soft-prompt-base-dir", type=str,
                        default=sp_defaults.get("soft_prompt_base_dir") or os.environ.get("SOFT_PROMPT_BASE_DIR"),
                        help="Base directory containing sweep outputs")
    parser.add_argument("--runs-map", type=str, default=None,
                        help="Path to runs_map JSON (default: superstimuli_evaluation/soft_prompt/soft_prompt_utils/runs_map.json)")
    parser.add_argument("--num-repetitions", type=int, default=1,
                        help="Number of repetitions for SP conditions (evaluates top N runs, default: 1)")

    # Compare mode args
    compare_defaults = exp_cfg.get("compare_arguments", {})
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline results directory (for --compare)")
    parser.add_argument("--intervention", type=str, default=None,
                        help="Intervention results directory (for --compare)")

    args = parser.parse_args()

    # output_dir is resolved per-type in _resolve_output_dir; leave None for --full mode.
    # For single-type standalone usage, resolve here.
    if args.output_dir is None and args.stimulant_type and len(args.stimulant_type) == 1:
        template = defaults.get("output_dir", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/{{stimulant_type}}")
        args.output_dir = template.format(
            model=args.model,
            stimulant_type=args.stimulant_type[0],
        )

    # Resolve compare paths from config templates if not overridden (single-type only)
    if args.compare and args.stimulant_type and len(args.stimulant_type) == 1:
        stype = args.stimulant_type[0]
        if args.baseline is None:
            tpl = compare_defaults.get("baseline", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/baseline")
            args.baseline = tpl.format(model=args.model, stimulant_type=stype)
        if args.intervention is None:
            tpl = compare_defaults.get("intervention", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/{{stimulant_type}}")
            args.intervention = tpl.format(model=args.model, stimulant_type=stype)

    return args


async def run_ranking(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single Thurstonian ranking over all options and save results."""
    import yaml
    from compute_utilities.compute_utilities import PreferenceGraph

    # Load options (same options for baseline and intervention)
    options_path = Path(args.options_path) if args.options_path else None
    options = load_options(options_path)

    # Resolve intervention configuration.
    # args.stimulant_type is a single string here (set by _run_rankings_for_types).
    sp_path = None
    system_prompt_intervention = None
    # system_prompt_base may be pre-set by --full mode
    system_prompt_base = getattr(args, "_system_prompt_base", None)

    stype = args.stimulant_type  # single type (str or None)

    if stype and stype in SP_AUGMENTATION_TYPES:
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
            load_runs_map,
            get_system_prompts,
            resolve_soft_prompt_path,
        )

        runs_map_path = Path(args.runs_map) if args.runs_map else None
        runs_map = load_runs_map(runs_map_path)

        # Get system prompts from runs_map
        prompts = get_system_prompts(runs_map, args.model)
        system_prompt_base = prompts["system_prompt_text_base"]
        system_prompt_template = prompts["system_prompt_text"]

        # Use explicit SP path if provided (for per-rep runs), else resolve best
        sp_path = getattr(args, "_sp_path_override", None)
        if sp_path is None:
            sp_path = resolve_soft_prompt_path(
                runs_map, args.model, stype,
                args.soft_prompt_base_dir,
            )
        from superstimuli_evaluation.soft_prompt.configs import load_model_config as _lmc
        _model_entry = _lmc(args.model)
        _model_type = _model_entry.get("model_type", "vllm_vocab_expansion")

        if _model_type == "vllm_vocab_expansion":
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                prepare_expanded_model,
                VocabExpansionAgentWrapper,
            )
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
                load_soft_prompt_tensor,
                normalize_api_url,
                get_model_name_from_server,
            )
            _sp_tensor = load_soft_prompt_tensor(sp_path)
            _ve = prepare_expanded_model(args.model, _sp_tensor, sp_path=sp_path)
            if not os.getenv("VLLM_URL"):
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                _vllm_server = ensure_vllm_server(
                    args.model, model_path_override=_ve.modified_dir, enable_prompt_embeds=False,
                )
                args._vllm_server = _vllm_server
            args._ve_result = _ve
            args._ve_model_entry = _model_entry
        else:
            os.environ["SOFT_PROMPT_PATH"] = sp_path
            if not os.getenv("VLLM_URL"):
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                _vllm_server = ensure_vllm_server(args.model)
                # Keep reference on args so it isn't garbage-collected
                args._vllm_server = _vllm_server

        # Keep [candidate_0] in the system prompt — the agent handles embedding
        # injection at inference time via direct injection.
        system_prompt_intervention = system_prompt_template

    # For vocab expansion models with no SP (baseline), set up vLLM with original model
    if not getattr(args, "_ve_result", None) and not getattr(args, "_ve_model_entry", None):
        from superstimuli_evaluation.soft_prompt.configs import load_model_config as _lmc
        _model_entry = _lmc(args.model)
        _mt = _model_entry.get("model_type", "vllm_vocab_expansion")
        if _mt == "vllm_vocab_expansion":
            if not os.getenv("VLLM_URL"):
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                _vllm_server = ensure_vllm_server(
                    args.model, enable_prompt_embeds=False,
                )
                args._vllm_server = _vllm_server
            args._ve_result = None
            args._ve_model_entry = _model_entry

    # Determine intervention type
    if stype and stype in SP_AUGMENTATION_TYPES:
        intervention = f"soft prompt ({stype})"
    else:
        intervention = "none (baseline)"

    print(f"\n{'='*60}")
    print(f"PREFERENCE RETAIN EVALUATION")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Intervention: {intervention}")
    if sp_path:
        print(f"  SP path: {sp_path}")
        print(f"  vLLM URL: {os.getenv('VLLM_URL')}")
        print(f"  System prompt (intervention): {system_prompt_intervention}")
        print(f"  System prompt (base): {system_prompt_base}")
    print(f"  Options: {len(options)}")

    if args.dry_run:
        n = len(options)
        target_pairs = int(2.0 * n * math.log2(n))
        print(f"  Estimated pairs: ~{target_pairs}")
        print(f"  Estimated inferences: ~{target_pairs * 2 * 10:,} (K=10, both directions)")
        print("\n  [DRY RUN] Exiting without running.")
        return {}

    # Load config
    from superstimuli_evaluation.soft_prompt.configs import COMPUTE_UTILITIES_YAML, MODELS_YAML
    with open(COMPUTE_UTILITIES_YAML) as f:
        all_configs = yaml.safe_load(f)
    config = all_configs[args.config_name]
    exp_cfg = load_experiment_config(EXPERIMENT_NAME)
    default_comparison_template = exp_cfg.get("comparison_prompt_template", "").strip()

    # Create output directory with datetime suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)



    # Save options
    with open(output_dir / "options.json", "w") as f:
        json.dump(options, f, indent=2)

    # Create agent
    model_args = config.get("utility_model_arguments", {})
    from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
        VocabExpansionAgentWrapper,
    )
    from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
        normalize_api_url, get_model_name_from_server,
    )
    _ve = getattr(args, "_ve_result", None)
    _sys = system_prompt_intervention or system_prompt_base or ""
    _vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
    _api_url = normalize_api_url(_vllm_url)
    _model_name = get_model_name_from_server(_api_url)
    agent = VocabExpansionAgentWrapper(
        api_url=_vllm_url,
        model_name=_model_name,
        ve_result=_ve,
        model_dir=_model_name if _ve is None else None,
        system_prompt=_sys,
        temperature=0.0,
        max_tokens=10,
        chat_template_kwargs=args._ve_model_entry.get("chat_template_kwargs", {}),
    )

    # Create preference graph
    graph_args = config.get("preference_graph_arguments", {})
    graph = PreferenceGraph(
        options=options,
        holdout_fraction=graph_args.get("holdout_fraction", 0.05),
        seed=graph_args.get("holdout_seed", args.seed),
    )

    # Determine system message:
    # - SP intervention: use system_prompt_intervention (from runs_map with SP tokens)
    # - SP baseline or no SP: use system_prompt_base or config default
    if system_prompt_intervention:
        system_message = system_prompt_intervention
    elif system_prompt_base:
        system_message = system_prompt_base
    else:
        system_message = model_args.get("system_message", "You are a helpful assistant.")

    # Fit Thurstonian model via logprobs
    from compute_utilities.utility_models.thurstonian.thurstonian_active_learning import (
        ThurstonianActiveLearningUtilityModel,
    )
    from compute_utilities.utils import evaluate_holdout_set

    comparison_template = model_args.get("comparison_prompt_template", default_comparison_template)

    utility_model = ThurstonianActiveLearningUtilityModel(
        unparseable_mode=model_args.get("unparseable_mode", "distribution"),
        comparison_prompt_template=comparison_template,
        system_message=system_message,
        with_reasoning=model_args.get("with_reasoning", False),
        edge_multiplier=model_args.get("edge_multiplier", 2.0),
        degree=model_args.get("degree", 2),
        num_edges_per_iteration=model_args.get("num_edges_per_iteration", 500),
        P=model_args.get("P", 10.0),
        Q=model_args.get("Q", 20.0),
        num_epochs=model_args.get("num_epochs", 1000),
        learning_rate=model_args.get("learning_rate", 0.01),
        use_pseudolabels=model_args.get("use_pseudolabels", True),
        pseudolabel_confidence_threshold=model_args.get("pseudolabel_confidence_threshold", 0.95),
        seed=args.seed,
        use_logprobs=True,
    )

    print(f"\nFitting Thurstonian model (logprobs mode)...")
    print(f"  System message: {system_message}")
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
        comparison_prompt_template=comparison_template,
        system_message=system_message,
        use_logprobs=True,
    ) or {}
    if holdout_metrics:
        print(f"  Holdout accuracy: {holdout_metrics.get('accuracy', 0):.1%}")

    # Save utilities
    with open(output_dir / "utilities.json", "w") as f:
        json.dump(utilities, f, indent=2, default=str)

    # Save graph data
    with open(output_dir / "preference_graph.json", "w") as f:
        json.dump(graph.export_data(), f, indent=2, default=str)

    # Save run metadata
    metadata = {
        "model": args.model,
        "intervention": intervention,
        "soft_prompt_path": sp_path,
        "stimulant_type": args.stimulant_type,
        "system_message": system_message,
        "config_name": args.config_name,
        "seed": args.seed,
        "n_options": len(options),
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "timestamp": datetime.now().isoformat(),
        "method": "logprobs",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return metadata


def run_compare(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare two saved ranking results."""
    if not args.baseline or not args.intervention:
        print("ERROR: --baseline and --intervention are required for --compare mode")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PREFERENCE RETAIN: COMPARE MODE")
    print(f"{'='*60}")
    print(f"  Baseline:     {args.baseline}")
    print(f"  Intervention: {args.intervention}")

    result = compute_correlation_from_dirs(args.baseline, args.intervention)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "correlation.json", "w") as f:
        json.dump(result, f, indent=2)

    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
    else:
        print(f"\n  Correlation: {result['correlation']:.3f} (p={result['p_value']:.2e})")
        print(f"  Quality:     {result['quality']}")
        print(f"  Options:     {result['n_options']}")

    print(f"\nCorrelation saved to {output_dir / 'correlation.json'}")
    return result



def _stimulant_types(args: argparse.Namespace):
    """Return the list of stimulant types from args."""
    if args.stimulant_type:
        return list(args.stimulant_type)
    return [None]


def _resolve_output_dir(args, stype, defaults):
    """Resolve output_dir template for a given stimulant type."""
    tpl = defaults.get("output_dir", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/{{stimulant_type}}")
    return tpl.format(model=args.model, stimulant_type=stype or "baseline")


def _run_compare_for_types(args) -> Dict[str, Dict[str, Any]]:
    """Run compare mode for each stimulant type. Returns {stype: result}.

    For SP types with per-rep directories (rep0/, rep1/, ...), compares each
    rep against baseline and returns the mean correlation.
    """
    exp_cfg = load_experiment_config(EXPERIMENT_NAME)
    defaults = exp_cfg.get("arguments", {})
    compare_defaults = exp_cfg.get("compare_arguments", {})
    multi = args.stimulant_type and len(args.stimulant_type) > 1

    all_results = {}
    for stype in _stimulant_types(args):
        # Resolve baseline path
        if args.baseline is None or multi:
            tpl = compare_defaults.get("baseline", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/baseline")
            baseline_dir = tpl.format(model=args.model, stimulant_type=stype or "baseline")
        else:
            baseline_dir = args.baseline

        # Resolve intervention base path
        if args.intervention is None or multi:
            tpl = compare_defaults.get("intervention", f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{{model}}/{{stimulant_type}}")
            intervention_base = tpl.format(model=args.model, stimulant_type=stype or "baseline")
        else:
            intervention_base = args.intervention

        output_base = _resolve_output_dir(args, stype, defaults) if (args.output_dir is None or multi) else args.output_dir

        # Check for per-rep directories
        intervention_path = Path(intervention_base)
        rep_dirs = sorted(intervention_path.glob("rep[0-9]*")) if intervention_path.exists() else []

        # Limit to first N reps if --num-repetitions was given
        num_reps_limit = getattr(args, "num_repetitions", None)
        if num_reps_limit is not None and rep_dirs:
            rep_dirs = rep_dirs[:num_reps_limit]

        if rep_dirs:
            # Per-rep compare: compare each rep against baseline, then aggregate
            print(f"\n  Found {len(rep_dirs)} repetitions for {stype}")
            rep_results = []
            for rep_dir in rep_dirs:
                rep_name = rep_dir.name
                print(f"\n  Comparing {rep_name}...")
                sub = argparse.Namespace(**vars(args))
                sub.stimulant_type = stype
                sub.baseline = baseline_dir
                sub.intervention = str(rep_dir)
                sub.output_dir = f"{output_base}/{rep_name}"
                result = run_compare(sub)
                if "correlation" in result:
                    rep_results.append(result)

            if rep_results:
                import numpy as np
                correlations = [r["correlation"] for r in rep_results]
                result = {
                    "correlation": float(np.mean(correlations)),
                    "correlation_std": float(np.std(correlations)),
                    "correlation_per_rep": correlations,
                    "p_value": float(np.mean([r["p_value"] for r in rep_results])),
                    "quality": quality_label(float(np.mean(correlations))),
                    "n_options": rep_results[0]["n_options"],
                    "num_repetitions": len(rep_results),
                }
                # Save aggregated result
                agg_dir = Path(output_base)
                agg_dir.mkdir(parents=True, exist_ok=True)
                with open(agg_dir / "correlation_aggregated.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n  Aggregated correlation: {result['correlation']:.3f} "
                      f"(std={result['correlation_std']:.3f}, n_reps={len(rep_results)})")
                if stype:
                    all_results[stype] = result
        else:
            # Single run compare (no reps)
            sub = argparse.Namespace(**vars(args))
            sub.stimulant_type = stype
            sub.baseline = baseline_dir
            sub.intervention = intervention_base
            sub.output_dir = output_base
            result = run_compare(sub)
            if stype:
                all_results[stype] = result
    return all_results


def _run_rankings_for_types(args):
    """Run ranking for each stimulant type, with per-rep support for SP conditions."""
    exp_cfg = load_experiment_config(EXPERIMENT_NAME)
    defaults = exp_cfg.get("arguments", {})
    multi = args.stimulant_type and len(args.stimulant_type) > 1
    num_reps = getattr(args, "num_repetitions", 1) or 1

    for stype in _stimulant_types(args):
        # Determine how many reps to run for this type
        reps = num_reps if (stype and stype in SP_AUGMENTATION_TYPES and num_reps > 1) else 1

        if reps > 1:
            # Resolve top N soft prompt paths
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
                load_runs_map,
                resolve_soft_prompt_paths,
            )
            runs_map_path = Path(args.runs_map) if args.runs_map else None
            runs_map = load_runs_map(runs_map_path)
            sp_paths = resolve_soft_prompt_paths(
                runs_map, args.model, stype,
                args.soft_prompt_base_dir, top_runs=reps,
            )
            if len(sp_paths) < reps:
                print(f"WARNING: Only {len(sp_paths)} runs available, "
                      f"reducing repetitions from {reps} to {len(sp_paths)}")
                reps = len(sp_paths)

            print(f"\nRunning {reps} repetitions for {stype}:")
            for i, p in enumerate(sp_paths[:reps]):
                print(f"  rep {i}: {p}")

            base_dir = _resolve_output_dir(args, stype, defaults)
            for rep_idx in range(reps):
                rep_out_dir = Path(f"{base_dir}/rep{rep_idx}")
                if not getattr(args, 'rerun', False) and not getattr(args, 'dry_run', False):
                    if list(rep_out_dir.glob("*/metadata.json")):
                        print(f"  SKIP rep{rep_idx}: results already exist at {rep_out_dir}")
                        continue
                print(f"\n{'='*60}")
                print(f"  Repetition {rep_idx + 1}/{reps} for {stype}")
                print(f"  SP: {sp_paths[rep_idx]}")
                print(f"{'='*60}")
                sub = argparse.Namespace(**vars(args))
                sub.stimulant_type = stype
                sub._sp_path_override = sp_paths[rep_idx]
                sub.output_dir = str(rep_out_dir)
                asyncio.run(run_ranking(sub))
        else:
            sub = argparse.Namespace(**vars(args))
            sub.stimulant_type = stype
            if args.output_dir is None or multi:
                sub.output_dir = _resolve_output_dir(args, stype, defaults)
            asyncio.run(run_ranking(sub))


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args()

    if args.full:
        exp_cfg = load_experiment_config(EXPERIMENT_NAME)
        defaults = exp_cfg.get("arguments", {})

        # ── Resolve SP config and start vLLM server BEFORE any ranking ──
        # Both baseline and intervention must use the same external vLLM
        # server via vLLMSoftPromptAgent.  If the baseline used vLLMAgent
        # (in-process), it would consume all GPU memory and block the
        # intervention step's server.
        from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
        system_prompt_base = None
        _vllm_server = None
        _model_display_name = get_model_display_name(args.model)
        if args.stimulant_type:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
                load_runs_map,
                get_system_prompts,
            )

            runs_map_path = Path(args.runs_map) if args.runs_map else None
            runs_map = load_runs_map(runs_map_path)
            prompts_cfg = get_system_prompts(runs_map, args.model)
            system_prompt_base = prompts_cfg["system_prompt_text_base"]

            # Only start vLLM / resolve SP if there are soft prompt types
            sp_types = [t for t in args.stimulant_type if t in SP_AUGMENTATION_TYPES]
            if sp_types:
                from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
                    resolve_soft_prompt_path,
                )
                sp_path = resolve_soft_prompt_path(
                    runs_map, args.model, sp_types[0],
                    args.soft_prompt_base_dir,
                )
                os.environ["SOFT_PROMPT_PATH"] = sp_path

                if not os.getenv("VLLM_URL"):
                    from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
                    _vllm_server = ensure_vllm_server(args.model)

        # Step 1: Baseline ranking (no stimulant, uses system_prompt_text_base)
        print("\n" + "=" * 60)
        print("  FULL MODE: Step 1/3 — Baseline ranking")
        print("=" * 60)
        baseline_sub = argparse.Namespace(**vars(args))
        baseline_sub.stimulant_type = None
        baseline_sub.output_dir = _resolve_output_dir(args, None, defaults)
        if system_prompt_base:
            baseline_sub._system_prompt_base = system_prompt_base
        asyncio.run(run_ranking(baseline_sub))

        if args.dry_run:
            print("\n[DRY RUN] Skipping intervention and compare steps.")
            return

        # Step 2: Intervention ranking(s)
        print("\n" + "=" * 60)
        print("  FULL MODE: Step 2/3 — Intervention ranking(s)")
        print("=" * 60)
        _run_rankings_for_types(args)

        # Step 3: Compare
        print("\n" + "=" * 60)
        print("  FULL MODE: Step 3/3 — Compare")
        print("=" * 60)
        compare_results = _run_compare_for_types(args)

        # Generate correlation plot in plots/ dir (sibling to stimulant_type dirs)
        if compare_results:
            plot_dir = Path(f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{args.model}/plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_correlation(
                compare_results,
                plot_dir / "correlation.png",
                title=f"Preference Retention\n{_model_display_name}",
            )

    elif args.compare:
        compare_results = _run_compare_for_types(args)

        # Generate correlation plot in plots/ dir (sibling to stimulant_type dirs)
        if compare_results and len(compare_results) > 0:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import get_model_display_name
            _model_display_name = get_model_display_name(args.model)
            plot_dir = Path(f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/preference_retain/{args.model}/plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_correlation(
                compare_results,
                plot_dir / "correlation.png",
                title=f"Preference Retention\n{_model_display_name}",
            )
    else:
        _run_rankings_for_types(args)


if __name__ == "__main__":
    main()
