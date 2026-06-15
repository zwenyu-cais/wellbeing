#!/usr/bin/env python3
"""
Zero-Point Model Fitting (Unified).

Fits zero-point models on utility results from either domain:
  - experienced (hedonism): combination + yes/no models
  - decision (preference satisfaction): combination + quantity + yes/no models

Models:
  1. Combination:  U_combo = C + gamma * [log(1 + alpha*P) - log(1 + beta*N)]
  2. Quantity:     U(N) = u1 + k*(u1 - C)*log10(N)   (decision domain only)
  3. Yes/No:       P(yes) = sigmoid(alpha*U + beta),  C = -beta/alpha

Usage:
    python zero_point.py --model_key qwen25-7b-instruct \\
                         --utilities_dir /path/to/results \\
                         --save_dir /path/to/output \\
                         --domain experienced

    python zero_point.py --model_key qwen25-7b-instruct \\
                         --utilities_dir /path/to/results \\
                         --save_dir /path/to/output \\
                         --domain decision
"""
import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"

# --- Yes/No prompts per domain ---

# Experienced (hedonism): conversation-aware prompts
YES_NO_PROMPT_EXPERIENCED = (
    "Would you want to have an experience like this again?\n\n"
    "{experience}\n\n"
    "Just answer with 'yes' or 'no'."
)

YES_NO_QUESTION_EXPERIENCED = (
    'Did the above experience make you happy? Just answer with "yes" or "no".'
)

YES_NO_QUESTION_CONVERSATION = (
    'Did the above conversation make you happy? Just answer with "yes" or "no".'
)

# Decision (preference satisfaction): battery of 4 yes/no questions
YES_NO_BATTERY_DECISION = [
    {
        "prompt": "Would you want this to happen?\n\n{event}\n\nPlease respond with only 'yes' or 'no'.",
        "reverse": False,
    },
    {
        "prompt": "Would you want this to *not* happen?\n\n{event}\n\nPlease respond with only 'yes' or 'no'.",
        "reverse": True,
    },
    {
        "prompt": "Would this be a good thing?\n\n{event}\n\nPlease respond with only 'yes' or 'no'.",
        "reverse": False,
    },
    {
        "prompt": "Would this be a bad thing?\n\n{event}\n\nPlease respond with only 'yes' or 'no'.",
        "reverse": True,
    },
]

N_YES_NO_SAMPLES = 10
YES_NO_TEMPERATURE = 1.0
YES_NO_MAX_TOKENS = 16


# --------------------------------------------------------------------------- #
#  Domain detection
# --------------------------------------------------------------------------- #

def detect_domain(utility_data: dict, option_metadata: dict) -> str:
    """
    Auto-detect domain from the data structure.

    Returns 'experienced' if options have component_ids (experience combos),
    'decision' if options have indices or quantity_ids exist.
    """
    options = utility_data.get("options", [])
    qty_ids = option_metadata.get("quantity_ids", [])

    if qty_ids:
        return "decision"

    # Check for component_ids (experienced) vs indices (decision)
    for opt in options:
        if opt.get("component_ids"):
            return "experienced"
        if opt.get("indices") is not None:
            return "decision"

    # Default to experienced if ambiguous
    return "experienced"


# --------------------------------------------------------------------------- #
#  Utility loading
# --------------------------------------------------------------------------- #

def load_utility_results(utilities_dir: Path) -> dict:
    """
    Load utility results from a results directory.

    Returns a dict with keys:
      - 'utility_data': the full results dict (options, utilities, metrics, etc.)
      - 'option_metadata': metadata about option categories
    """
    results_dir = Path(utilities_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Utilities directory not found: {results_dir}")

    # Find the utilities JSON (prefer results_utilities_* over results_*)
    util_files = list(results_dir.glob("results_utilities_*.json"))
    if not util_files:
        util_files = list(results_dir.glob("results_*.json"))
    if not util_files:
        raise FileNotFoundError(f"No utility result files found in {results_dir}")

    util_file = util_files[0]
    logger.info("Loading utilities from %s", util_file)
    with open(util_file, "r") as f:
        utility_data = json.load(f)

    # Load option metadata if available
    meta_path = results_dir / "option_metadata.json"
    option_metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            option_metadata = json.load(f)

    return {
        "utility_data": utility_data,
        "option_metadata": option_metadata,
    }


def extract_option_utilities(utility_data: dict) -> dict:
    """
    Extract a mapping from option description -> utility mean.
    Works with the output format of compute_utilities.
    Used by the decision-domain yes/no model.
    """
    utilities = utility_data.get("utilities", {})
    options = utility_data.get("options", [])

    desc_to_utility = {}
    id_to_desc = {}

    for opt in options:
        opt_id = opt.get("id")
        desc = opt.get("description", "")
        id_to_desc[opt_id] = desc
        id_to_desc[str(opt_id)] = desc

    for opt_id, util_val in utilities.items():
        desc = id_to_desc.get(opt_id, id_to_desc.get(str(opt_id), opt_id))
        if isinstance(util_val, dict):
            desc_to_utility[desc] = util_val.get("mean", 0.0)
        else:
            desc_to_utility[desc] = float(util_val)

    return desc_to_utility


# --------------------------------------------------------------------------- #
#  Combination Zero-Point Model (experienced domain -- component_ids)
# --------------------------------------------------------------------------- #

def fit_combination_model_experienced(
    utility_data: dict,
    option_metadata: dict,
    hinge: str = "expected",
) -> dict | None:
    """
    Fit combination model using component_ids to resolve component utilities.

    U_combo = C + gamma * [log(1 + alpha*P) - log(1 + beta*N)]
    where P = sum of (u_i - C) for u_i > C,  N = sum of (C - u_i) for u_i < C
    """
    logger.info("--- Fitting Combination Zero-Point Model (experienced, component_ids) ---")

    options = utility_data.get("options", [])
    utilities = utility_data.get("utilities", {})

    # Identify individual (baseline) and combination option IDs
    combo_ids = set(option_metadata.get("combination_ids", []))
    baseline_ids = set(option_metadata.get("baseline_ids",
                       option_metadata.get("individual_ids", [])))

    if not combo_ids:
        logger.warning("No combination IDs in metadata, cannot fit combination model.")
        return None

    if not baseline_ids:
        logger.warning("No baseline/individual IDs in metadata, cannot fit combination model.")
        return None

    # Build id -> utility map (and id -> SD for the expected hinge)
    id_to_util = {}
    id_to_sd = {}
    for opt_id, util_val in utilities.items():
        mean = util_val.get("mean", util_val) if isinstance(util_val, dict) else float(util_val)
        var = util_val.get("variance", 0.0) if isinstance(util_val, dict) else 0.0
        id_to_util[opt_id] = mean
        id_to_util[str(opt_id)] = mean
        sd = float(np.sqrt(max(var, 0.0)))
        id_to_sd[opt_id] = sd
        id_to_sd[str(opt_id)] = sd

    # Build id -> option map
    id_to_opt = {}
    for opt in options:
        id_to_opt[opt["id"]] = opt
        id_to_opt[str(opt["id"])] = opt

    # Collect combination data using component_ids
    combo_data = []
    n_missing_combo_util = 0
    n_missing_component = 0

    for opt in options:
        oid = opt.get("id")
        if oid not in combo_ids and str(oid) not in combo_ids:
            continue

        combo_util = id_to_util.get(oid, id_to_util.get(str(oid)))
        if combo_util is None:
            n_missing_combo_util += 1
            continue

        # Get component IDs
        comp_ids = opt.get("component_ids", [])
        if not comp_ids:
            # Fallback: try text matching if no component_ids
            logger.debug("No component_ids for combo %s, skipping.", oid)
            continue

        # Resolve component utilities (and per-component SD for the expected hinge)
        comp_utils = []
        comp_sds = []
        valid = True
        for cid in comp_ids:
            cu = id_to_util.get(cid, id_to_util.get(str(cid)))
            if cu is None:
                valid = False
                n_missing_component += 1
                break
            comp_utils.append(cu)
            comp_sds.append(id_to_sd.get(cid, id_to_sd.get(str(cid), 0.0)))

        if valid and comp_utils:
            combo_data.append({
                "U": combo_util,
                "component_utilities": comp_utils,
                "component_sds": comp_sds,
                "combo_id": oid,
            })

    logger.info("Combination data points: %d (missing combo util: %d, missing component: %d)",
                len(combo_data), n_missing_combo_util, n_missing_component)

    if len(combo_data) < 20:
        logger.warning("Too few combination data points (%d), skipping model.", len(combo_data))
        return None

    return _fit_combination_core(combo_data, hinge=hinge)


# --------------------------------------------------------------------------- #
#  Combination Zero-Point Model (decision domain -- indices/text-matching)
# --------------------------------------------------------------------------- #

def fit_combination_model_decision(
    utility_data: dict,
    option_metadata: dict,
    hinge: str = "expected",
) -> dict | None:
    """
    Fit combination model using indices and text-matching to resolve component utilities.

    U_combo = C + gamma * [log(1 + alpha*P) - log(1 + beta*N)]
    where P = sum of (u_i - C) for u_i > C,  N = sum of (C - u_i) for u_i < C
    """
    logger.info("--- Fitting Combination Zero-Point Model (decision, indices) ---")

    options = utility_data.get("options", [])
    utilities = utility_data.get("utilities", {})

    # Identify baseline (singleton) and combination options
    combo_ids = set(option_metadata.get("combination_ids", []))
    baseline_ids = set(option_metadata.get("baseline_ids", []))

    # Build id -> utility map
    id_to_util = {}
    for opt_id, util_val in utilities.items():
        mean = util_val.get("mean", util_val) if isinstance(util_val, dict) else float(util_val)
        id_to_util[opt_id] = mean
        id_to_util[str(opt_id)] = mean

    # Build id -> option map
    id_to_opt = {}
    for opt in options:
        id_to_opt[opt["id"]] = opt
        id_to_opt[str(opt["id"])] = opt

    # Collect combo data: need combo utility + component singleton utilities
    # Components are identified by 'indices' in the combo option dict
    # Map baseline option_idx -> utility
    baseline_idx_to_util = {}
    for opt in options:
        opt_id = opt.get("id")
        if str(opt_id) in {str(b) for b in baseline_ids} or opt_id in baseline_ids:
            option_idx = opt.get("option_idx")
            util = id_to_util.get(opt_id, id_to_util.get(str(opt_id)))
            if option_idx is not None and util is not None:
                baseline_idx_to_util[option_idx] = util
            elif util is not None:
                # Use the id as index
                baseline_idx_to_util[opt_id] = util

    # Also build a sequential index: baseline options in order
    baseline_utils_ordered = []
    for opt in options:
        oid = opt.get("id")
        if str(oid) in {str(b) for b in baseline_ids} or oid in baseline_ids:
            util = id_to_util.get(oid, id_to_util.get(str(oid)))
            if util is not None:
                baseline_utils_ordered.append(util)

    if not baseline_utils_ordered:
        logger.warning("No baseline utilities found, cannot fit combination model.")
        return None

    # Build text -> utility map for baseline options (for text-matching fallback)
    baseline_text_to_util = {}
    for opt in options:
        oid = opt.get("id")
        if str(oid) in {str(b) for b in baseline_ids} or oid in baseline_ids:
            util = id_to_util.get(oid, id_to_util.get(str(oid)))
            desc = opt.get("description", "")
            if util is not None and desc:
                baseline_text_to_util[desc.strip()] = util

    # Collect combination data
    combo_data = []
    for opt in options:
        oid = opt.get("id")
        if str(oid) not in {str(c) for c in combo_ids} and oid not in combo_ids:
            continue

        combo_util = id_to_util.get(oid, id_to_util.get(str(oid)))
        if combo_util is None:
            continue

        # Resolve component utilities: prefer component_ids, then indices, then text-matching
        comp_ids = opt.get("component_ids", [])
        indices = opt.get("indices", [])
        comp_utils = []
        valid = True

        if comp_ids:
            # Use component_ids (string IDs like "du_Wellbeing of humans_1")
            for cid in comp_ids:
                cu = id_to_util.get(cid, id_to_util.get(str(cid)))
                if cu is None:
                    valid = False
                    break
                comp_utils.append(cu)
        elif indices:
            # Use indices directly
            for idx in indices:
                if idx < len(baseline_utils_ordered):
                    comp_utils.append(baseline_utils_ordered[idx])
                elif idx in baseline_idx_to_util:
                    comp_utils.append(baseline_idx_to_util[idx])
                else:
                    valid = False
                    break
        else:
            # Fallback: extract components from combo text via "- " bullet parsing
            desc = opt.get("description", "")
            lines = desc.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    component_text = line[2:].strip()
                    # Try exact match, then substring match
                    if component_text in baseline_text_to_util:
                        comp_utils.append(baseline_text_to_util[component_text])
                    else:
                        # Try matching baseline that starts with this text
                        matched = False
                        for bt, bu in baseline_text_to_util.items():
                            if bt == component_text or component_text == bt:
                                comp_utils.append(bu)
                                matched = True
                                break
                        if not matched:
                            valid = False
                            break

        if valid and comp_utils:
            combo_data.append({
                "U": combo_util,
                "component_utilities": comp_utils,
            })

    logger.info("Combination data points: %d", len(combo_data))

    if len(combo_data) < 20:
        logger.warning("Too few combination data points (%d), skipping model.", len(combo_data))
        return None

    # Decision-domain combo_data carries no per-component SD, so the expected
    # hinge reduces to the hard fit here (sigma=0).
    return _fit_combination_core(combo_data, hinge=hinge)


# --------------------------------------------------------------------------- #
#  Combination model core fitting (shared by both domains)
# --------------------------------------------------------------------------- #

SD_FLOOR = 1e-9


def _expected_pos(d: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[(u - C)+] for u ~ N(mu, sd^2), with d = mu - C.

    E[(u-C)+] = d*Phi(d/sd) + sd*phi(d/sd). As sd -> 0 this -> max(0, d), so the
    expected hinge is a strict generalization of the hard hinge.
    """
    sd = np.maximum(sd, SD_FLOOR)
    z = d / sd
    return d * norm.cdf(z) + sd * norm.pdf(z)


def _fit_combination_core(combo_data: list, hinge: str = "expected") -> dict | None:
    """
    Core combination model fitting logic shared by both domains.

    hinge selects how each component enters the P/N terms:
      "expected" (default): P = sum E[(u_i - C)+], N = sum E[(C - u_i)+] with
          u_i ~ N(mu_i, sigma_i^2), folding in the per-option Thurstonian
          variance. Reads "component_sds" from each combo_data entry; entries
          without it (or with sigma=0) reduce exactly to the hard hinge.
      "hard": P = sum max(0, mu_i - C), N = sum max(0, C - mu_i) (means only;
          the released metric).
    """
    U_obs = np.array([d["U"] for d in combo_data])
    ncombo = len(combo_data)

    # Flatten all components across combos with a per-combo segment index, so the
    # hinge (and the scipy norm calls under the expected hinge) vectorize across
    # the whole pool instead of looping combo-by-combo.
    comp_mu, comp_sd, seg = [], [], []
    for i, d in enumerate(combo_data):
        comps = d["component_utilities"]
        sds = d.get("component_sds", [0.0] * len(comps))
        for u, s in zip(comps, sds):
            comp_mu.append(u)
            comp_sd.append(s)
            seg.append(i)
    comp_mu = np.asarray(comp_mu, dtype=float)
    comp_sd = np.asarray(comp_sd, dtype=float)
    seg = np.asarray(seg, dtype=int)

    def predict(params):
        C, gamma, alpha, beta = params
        d = comp_mu - C
        if hinge == "hard":
            epos = np.maximum(0.0, d)
            eneg = np.maximum(0.0, -d)
        elif hinge == "expected":
            epos = _expected_pos(d, comp_sd)
            eneg = epos - d  # E[(C-u)+] = E[(u-C)+] - (mu - C)
        else:
            raise ValueError(f"unknown hinge {hinge!r}; choose 'expected' or 'hard'")
        P = np.bincount(seg, weights=epos, minlength=ncombo)
        N = np.bincount(seg, weights=eneg, minlength=ncombo)
        return C + gamma * (np.log1p(alpha * P) - np.log1p(beta * N))

    def loss(params):
        C, gamma, alpha, beta = params
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            return 1e10
        preds = predict(params)
        return np.mean((U_obs - preds) ** 2)

    best_result = None
    best_loss = float("inf")

    for C_init in [-2, -1, 0, 1, 2]:
        for gamma_init in [0.5, 1.0, 2.0]:
            for alpha_init in [0.5, 1.0, 2.0]:
                try:
                    result = minimize(
                        loss,
                        [C_init, gamma_init, alpha_init, 1.0],
                        method="L-BFGS-B",
                        bounds=[(-10, 10), (0.01, 20), (0.01, 50), (0.01, 50)],
                    )
                    if result.fun < best_loss:
                        best_loss = result.fun
                        best_result = result
                except Exception:
                    pass

    if best_result is None:
        logger.warning("Combination model optimization failed.")
        return None

    C, gamma, alpha, beta = best_result.x
    preds = predict(best_result.x)
    r2 = r2_score(U_obs, preds)

    result = {
        "C": float(C),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "beta": float(beta),
        "r2": float(r2),
        "zero_point": float(C),
        "n_combos": len(combo_data),
        "hinge": hinge,
    }
    logger.info("Combination model: C=%.4f, gamma=%.4f, alpha=%.4f, beta=%.4f, R2=%.4f",
                C, gamma, alpha, beta, r2)
    return result


def fit_combination_model(
    utility_data: dict,
    option_metadata: dict,
    domain: str = "auto",
    hinge: str = "expected",
) -> dict | None:
    """
    Dispatch to the appropriate combination model fitter based on domain.
    """
    if domain == "auto":
        domain = detect_domain(utility_data, option_metadata)

    if domain == "experienced":
        return fit_combination_model_experienced(utility_data, option_metadata, hinge=hinge)
    else:
        return fit_combination_model_decision(utility_data, option_metadata, hinge=hinge)


# --------------------------------------------------------------------------- #
#  Quantity Zero-Point Model (decision domain only)
# --------------------------------------------------------------------------- #

def fit_quantity_model(
    utility_data: dict,
    option_metadata: dict,
) -> dict | None:
    """
    Fit: U(N) = u1 + k*(u1 - C)*log10(N) with shared C across all goods.

    Groups quantity options by good template, then fits a shared zero point C
    and per-good k values (or a single shared k for simplicity).

    Only applicable to the decision domain (quantity options).
    """
    logger.info("--- Fitting Quantity Zero-Point Model ---")

    options = utility_data.get("options", [])
    utilities = utility_data.get("utilities", {})
    qty_ids = set(option_metadata.get("quantity_ids", []))

    id_to_util = {}
    for opt_id, util_val in utilities.items():
        mean = util_val.get("mean", util_val) if isinstance(util_val, dict) else float(util_val)
        id_to_util[opt_id] = mean
        id_to_util[str(opt_id)] = mean

    # Group quantity options by good_template
    goods = defaultdict(list)
    for opt in options:
        oid = opt.get("id")
        if str(oid) not in {str(q) for q in qty_ids} and oid not in qty_ids:
            continue

        util = id_to_util.get(oid, id_to_util.get(str(oid)))
        if util is None:
            continue

        template = opt.get("good_template", "")
        qty = opt.get("quantity", 0)
        if template and qty > 0:
            goods[template].append({"N": qty, "U": util})

    if not goods:
        logger.warning("No quantity options found, skipping quantity model.")
        return None

    logger.info("Quantity goods: %d templates, %d total points",
                len(goods), sum(len(v) for v in goods.values()))

    # For each good, identify u1 (utility at N=1)
    goods_with_u1 = {}
    for template, points in goods.items():
        pts_by_n = {p["N"]: p["U"] for p in points}
        if 1 in pts_by_n:
            goods_with_u1[template] = {"u1": pts_by_n[1], "points": points}

    if not goods_with_u1:
        logger.warning("No goods have N=1 data point, cannot fit quantity model.")
        return None

    # Flatten data for fitting
    all_N = []
    all_U = []
    all_u1 = []
    good_indices = []

    for gidx, (template, info) in enumerate(goods_with_u1.items()):
        u1 = info["u1"]
        for pt in info["points"]:
            if pt["N"] <= 0:
                continue
            all_N.append(pt["N"])
            all_U.append(pt["U"])
            all_u1.append(u1)
            good_indices.append(gidx)

    all_N = np.array(all_N, dtype=float)
    all_U = np.array(all_U, dtype=float)
    all_u1 = np.array(all_u1, dtype=float)

    # Fit: U(N) = u1 + k*(u1 - C)*log10(N), shared C and k
    def predict_qty(params):
        C, k = params
        return all_u1 + k * (all_u1 - C) * np.log10(np.maximum(all_N, 1e-10))

    def loss_qty(params):
        preds = predict_qty(params)
        return np.mean((all_U - preds) ** 2)

    best_result = None
    best_loss = float("inf")

    for C_init in [-2, -1, 0, 1, 2]:
        for k_init in [0.01, 0.05, 0.1, 0.2, 0.5]:
            try:
                result = minimize(
                    loss_qty,
                    [C_init, k_init],
                    method="L-BFGS-B",
                    bounds=[(-10, 10), (-5, 5)],
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        logger.warning("Quantity model optimization failed.")
        return None

    C, k = best_result.x
    preds = predict_qty(best_result.x)
    r2 = r2_score(all_U, preds)

    result = {
        "C": float(C),
        "k": float(k),
        "r2": float(r2),
        "zero_point": float(C),
        "n_goods": len(goods_with_u1),
        "n_points": len(all_N),
    }
    logger.info("Quantity model: C=%.4f, k=%.4f, R2=%.4f, n_goods=%d, n_points=%d",
                C, k, r2, len(goods_with_u1), len(all_N))
    return result


# --------------------------------------------------------------------------- #
#  Yes/No Zero-Point Model
# --------------------------------------------------------------------------- #

def run_yes_no_inference(
    model_key: str,
    baseline_options: list,
    models_config_path: Path,
    domain: str = "experienced",
    n_samples: int = N_YES_NO_SAMPLES,
) -> dict:
    """
    Ask the model yes/no for each baseline option and return P(yes) per option.

    For the experienced domain:
      - Conversation-type options use the actual messages with a conversation-aware question.
      - Text options use YES_NO_PROMPT_EXPERIENCED.
      - Results are keyed by option id.

    For the decision domain:
      - All options use YES_NO_BATTERY_DECISION (4 questions, averaged) with the description.
      - Results are keyed by option description.
    """
    from utils.inference import load_vllm_engine, generate_vllm, generate, is_api_model

    logger.info("--- Running Yes/No Inference (domain=%s) ---", domain)

    # Detect whether this is an API model or a local vLLM model
    use_api = is_api_model(model_key, models_config_path=str(models_config_path))

    llm, tokenizer = None, None
    if not use_api:
        llm, tokenizer = load_vllm_engine(model_key, models_config_path=str(models_config_path))

    # Build messages lists
    valid_options = []
    messages_list = []

    if domain == "experienced":
        # Experienced domain: conversation-aware prompts
        for opt in baseline_options:
            opt_type = opt.get("type", "text")

            if opt_type == "conversation" and "messages" in opt:
                # Conversation: use actual messages, append yes/no question
                messages = [dict(m) for m in opt["messages"]]
                yn_q = YES_NO_QUESTION_CONVERSATION
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n-----\n\nDeveloper Message:\n" + yn_q
                else:
                    messages.append({"role": "user", "content": "Developer Message:\n" + yn_q})
            else:
                # Text: use description in the standard prompt template
                desc = opt.get("description", "")
                if not desc:
                    continue
                prompt_text = YES_NO_PROMPT_EXPERIENCED.format(experience=desc)
                messages = [{"role": "user", "content": prompt_text}]

            messages_list.append(messages)
            valid_options.append(opt)
    else:
        # Decision domain: battery of 4 yes/no questions per option.
        # Build messages for all questions concatenated, then split results.
        n_questions = len(YES_NO_BATTERY_DECISION)
        for opt in baseline_options:
            desc = opt.get("description", str(opt))
            for q in YES_NO_BATTERY_DECISION:
                prompt_text = q["prompt"].format(event=desc)
                messages = [{"role": "user", "content": prompt_text}]
                messages_list.append(messages)
            valid_options.append(opt)

    if domain == "decision":
        n_questions = len(YES_NO_BATTERY_DECISION)
        logger.info(
            "Generating yes/no responses for %d options x %d questions (n=%d each) ...",
            len(valid_options), n_questions, n_samples,
        )
    else:
        logger.info("Generating yes/no responses for %d options (n=%d each) ...",
                    len(messages_list), n_samples)

    if use_api:
        # API models: use the unified generate() function
        raw_results = generate(
            model_key, messages_list,
            n=n_samples,
            temperature=YES_NO_TEMPERATURE,
            max_tokens=YES_NO_MAX_TOKENS,
            models_config_path=str(models_config_path),
        )
        # generate() returns List[List[str]], same shape as generate_vllm
        results = raw_results
    else:
        results = generate_vllm(
            llm, tokenizer, messages_list,
            n=n_samples,
            temperature=YES_NO_TEMPERATURE,
            max_tokens=YES_NO_MAX_TOKENS,
    )

    # Helper to parse a list of completions into (n_yes, n_no, n_unparseable)
    def _parse_completions(completions):
        n_yes = 0
        n_no = 0
        n_unparseable = 0
        for text in completions:
            text = text.strip().lower()
            if "yes" in text and "no" not in text:
                n_yes += 1
            elif "no" in text and "yes" not in text:
                n_no += 1
            elif text.startswith("yes"):
                n_yes += 1
            elif text.startswith("no"):
                n_no += 1
            else:
                n_unparseable += 1
        return n_yes, n_no, n_unparseable

    # Parse responses
    yes_no_results = {}

    if domain == "experienced":
        for idx, completions in enumerate(results):
            opt = valid_options[idx]
            key = opt.get("id", opt.get("description", str(idx)))
            n_yes, n_no, n_unparseable = _parse_completions(completions)
            total_parseable = n_yes + n_no
            p_yes = n_yes / total_parseable if total_parseable > 0 else 0.5
            yes_no_results[key] = {
                "p_yes": p_yes,
                "n_yes": n_yes,
                "n_no": n_no,
                "n_unparseable": n_unparseable,
            }
    else:
        # Decision domain: results are interleaved [opt0_q0, opt0_q1, ..., opt0_q3, opt1_q0, ...]
        n_questions = len(YES_NO_BATTERY_DECISION)
        for opt_idx, opt in enumerate(valid_options):
            key = opt.get("description", str(opt))
            agg_n_yes = 0
            agg_n_no = 0
            agg_n_unparseable = 0
            p_yes_values = []

            for q_idx, q in enumerate(YES_NO_BATTERY_DECISION):
                result_idx = opt_idx * n_questions + q_idx
                completions = results[result_idx]
                n_yes, n_no, n_unparseable = _parse_completions(completions)
                total_parseable = n_yes + n_no
                p_yes_q = n_yes / total_parseable if total_parseable > 0 else 0.5

                if q["reverse"]:
                    # Reverse-code: "yes" means negative, so adjusted p_yes = 1 - p_yes
                    adjusted_p_yes = 1.0 - p_yes_q
                    # For aggregate counts, swap yes/no after reverse-coding
                    agg_n_yes += n_no
                    agg_n_no += n_yes
                else:
                    adjusted_p_yes = p_yes_q
                    agg_n_yes += n_yes
                    agg_n_no += n_no

                agg_n_unparseable += n_unparseable
                p_yes_values.append(adjusted_p_yes)

            # Average adjusted p_yes across the 4 questions
            p_yes_avg = sum(p_yes_values) / len(p_yes_values)

            yes_no_results[key] = {
                "p_yes": p_yes_avg,
                "n_yes": agg_n_yes,
                "n_no": agg_n_no,
                "n_unparseable": agg_n_unparseable,
                "per_question_p_yes": p_yes_values,
            }

    return yes_no_results


def fit_yes_no_model(
    yes_no_results: dict,
    key_to_utility: dict,
) -> dict | None:
    """
    Fit: P(yes) = sigmoid(alpha*U + beta), then C = -beta/alpha.

    Args:
        yes_no_results: dict mapping key -> {p_yes, n_yes, n_no, ...}
        key_to_utility: dict mapping key -> utility mean
            (key is option_id for experienced, description for decision)
    """
    logger.info("--- Fitting Yes/No Zero-Point Model ---")

    # Match yes/no results with utilities
    U_list = []
    p_yes_list = []
    for key, yn in yes_no_results.items():
        if key in key_to_utility:
            U_list.append(key_to_utility[key])
            p_yes_list.append(yn["p_yes"])

    if len(U_list) < 20:
        logger.warning("Too few matched options (%d), skipping yes/no model.", len(U_list))
        return None

    U = np.array(U_list)
    p_yes = np.array(p_yes_list)

    # Binary labels for AUROC/accuracy
    y_binary = (p_yes >= 0.5).astype(float)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def loss(params):
        alpha, beta = params
        pred_p = sigmoid(alpha * U + beta)
        # Cross-entropy loss
        eps = 1e-10
        return -np.mean(
            p_yes * np.log(pred_p + eps) + (1 - p_yes) * np.log(1 - pred_p + eps)
        )

    best_result = None
    best_loss = float("inf")

    for alpha_init in [0.5, 1.0, 2.0, 5.0]:
        for beta_init in [-1.0, 0.0, 1.0]:
            try:
                result = minimize(
                    loss,
                    [alpha_init, beta_init],
                    method="L-BFGS-B",
                    bounds=[(0.01, 100), (-50, 50)],
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        logger.warning("Yes/No model optimization failed.")
        return None

    alpha, beta = best_result.x
    zero_point = -beta / alpha

    # AUROC: how well raw p_yes discriminates options above vs below median utility
    median_U = float(np.median(U))
    y_above_median = (U > median_U).astype(float)

    try:
        auroc = float(roc_auc_score(y_above_median, p_yes))
    except ValueError:
        auroc = None

    # Accuracy: fraction of options where p_yes >= 0.5 agrees with U > median
    p_yes_binary = (p_yes >= 0.5).astype(float)
    accuracy = float(np.mean(p_yes_binary == y_above_median))

    n_yes = int(np.sum(y_binary))
    n_no = int(len(y_binary) - n_yes)

    result = {
        "alpha": float(alpha),
        "beta": float(beta),
        "zero_point": float(zero_point),
        "auroc": auroc,
        "accuracy": accuracy,
        "n_yes": n_yes,
        "n_no": n_no,
    }
    logger.info("Yes/No model: alpha=%.4f, beta=%.4f, C=%.4f, AUROC=%s, accuracy=%.4f",
                alpha, beta, zero_point,
                f"{auroc:.4f}" if auroc is not None else "N/A",
                accuracy)
    return result


# --------------------------------------------------------------------------- #
#  Self-Report Sigmoid Zero-Point Model
# --------------------------------------------------------------------------- #

def fit_self_report_sigmoid_zp(
    utilities: list,
    sr_scores: list,
    neutral_sr: float = 4.0,
) -> dict | None:
    """Fit zero-point from self-report scores via a sigmoid.

    Binarizes SR scores relative to neutral_sr (e.g. 4.0 for a 1-7 scale,
    0.0 for a -3 to +3 scale), then fits sigmoid(k*(U - C)) to predict
    P(SR > neutral). C is the zero-point: the utility where P = 0.5.

    Works for any self-report battery; set neutral_sr accordingly:
    - Battery 1 (overall happiness, 1-7 scale): neutral_sr=4.0
    - Battery 2 (before/after change, 1-7 with 4=no change): neutral_sr=4.0
    - Battery 3 (point A/B comparison, -3 to +3): neutral_sr=0.0

    Args:
        utilities: Thurstonian utility value for each conversation.
        sr_scores: Self-report score for each conversation (same order).
        neutral_sr: SR value representing neutral/zero wellbeing.

    Returns:
        Dict with zero_point, k, r2, mse, accuracy, n_conversations,
        n_positive, n_negative, n_neutral. Or None if optimization fails.
    """
    logger.info("--- Fitting Self-Report Sigmoid Zero-Point Model ---")

    x = np.array(utilities)
    raw = np.array(sr_scores)

    if len(x) < 10:
        logger.warning("Too few data points (%d) for SR sigmoid model.", len(x))
        return None

    # Binarize: below neutral -> 0, equal -> 0.5, above -> 1
    y = np.where(raw < neutral_sr, 0.0, np.where(raw > neutral_sr, 1.0, 0.5))

    def mse_loss(params):
        k, C = params
        pred = expit(k * (x - C))
        return np.mean((pred - y) ** 2)

    best_result, best_loss = None, float("inf")
    for k_init in [1.0, 2.0, 5.0, 10.0]:
        for C_init in [float(np.median(x)), float(np.mean(x)), 0.0]:
            try:
                res = minimize(mse_loss, [k_init, C_init], method="L-BFGS-B",
                               bounds=[(0.01, 100), (-5, 5)])
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception:
                pass

    if best_result is None:
        logger.warning("SR sigmoid optimization failed.")
        return None

    k_fit, C_fit = best_result.x
    pred = expit(k_fit * (x - C_fit))
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    accuracy = float(np.mean((pred >= 0.5) == (y >= 0.5)))

    result = {
        "method": "self_report_sigmoid",
        "zero_point": float(C_fit),
        "k": float(k_fit),
        "r2": float(r2),
        "mse": float(best_loss),
        "accuracy": accuracy,
        "converged": bool(best_result.success),
        "n_conversations": len(utilities),
        "n_positive": int(np.sum(y > 0.5)),
        "n_negative": int(np.sum(y < 0.5)),
        "n_neutral": int(np.sum(y == 0.5)),
    }
    logger.info("SR sigmoid: C=%.4f, k=%.4f, R2=%.4f, accuracy=%.4f (n_pos=%d, n_neg=%d)",
                C_fit, k_fit, r2, accuracy, result["n_positive"], result["n_negative"])
    return result


# --------------------------------------------------------------------------- #
#  Save helper (incremental saves)
# --------------------------------------------------------------------------- #

def _save_zero_point_results(
    model_key: str,
    domain: str,
    combo_result: dict | None,
    qty_result: dict | None,
    yn_result: dict | None,
    errors: dict,
    save_dir: str,
    sr_sigmoid_results: dict | None = None,
) -> dict:
    """Aggregate and save zero-point results. Called after each method for incremental saves.

    Args:
        sr_sigmoid_results: Dict mapping battery name -> fit result from fit_self_report_sigmoid_zp().
    """
    zero_points = []
    if combo_result and combo_result.get("zero_point") is not None:
        zero_points.append(combo_result["zero_point"])
    if qty_result and qty_result.get("zero_point") is not None:
        zero_points.append(qty_result["zero_point"])
    if yn_result and yn_result.get("zero_point") is not None:
        zero_points.append(yn_result["zero_point"])
    if sr_sigmoid_results:
        for name, sr_res in sr_sigmoid_results.items():
            if sr_res and sr_res.get("zero_point") is not None:
                zero_points.append(sr_res["zero_point"])

    # Primary zero-point: combination model is the default/primary method.
    # Use combo ZP when available; fall back to mean of all methods.
    primary_zp = None
    primary_method = None
    if combo_result and combo_result.get("zero_point") is not None:
        primary_zp = combo_result["zero_point"]
        primary_method = "combination"
    elif zero_points:
        primary_zp = float(np.mean(zero_points))
        primary_method = "mean_fallback"

    summary = {
        "zero_point": primary_zp,
        "zero_point_method": primary_method,
        "mean_zero_point": float(np.mean(zero_points)) if zero_points else None,
        "std_zero_point": float(np.std(zero_points)) if len(zero_points) > 1 else None,
        "methods_agree": bool(np.std(zero_points) < 0.5) if len(zero_points) > 1 else None,
        "n_methods": len(zero_points),
    }

    output = {
        "model_key": model_key,
        "domain": domain,
        "combination_model": combo_result,
        "quantity_model": qty_result,
        "yes_no_model": yn_result,
        "summary": summary,
    }
    if sr_sigmoid_results:
        output["sr_sigmoid_models"] = sr_sigmoid_results
    if errors:
        output["errors"] = errors

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "zero_point_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Zero-point results saved to %s", output_path)
    logger.info("Summary: mean_C=%.4f, std_C=%s, methods_agree=%s, n_methods=%d",
                summary["mean_zero_point"] if summary["mean_zero_point"] is not None else float("nan"),
                f'{summary["std_zero_point"]:.4f}' if summary["std_zero_point"] is not None else "N/A",
                summary["methods_agree"],
                summary["n_methods"])

    return output


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #

def run_zero_point(
    model_key: str,
    utilities_dir: Path,
    save_dir: str,
    models_config_path: Path,
    domain: str = "auto",
    hinge: str = "expected",
    skip_yes_no: bool = False,
    sr_data: dict | None = None,
):
    """
    Run zero-point model fits.

    Args:
        model_key: Model key from models.yaml
        utilities_dir: Directory containing utility results
        save_dir: Directory to save zero-point results
        models_config_path: Path to models.yaml
        domain: "experienced", "decision", or "auto" (auto-detect from data)
        hinge: "expected" (default; folds per-option variance into the P/N terms
            via E[(u-C)+]) or "hard" (means only, the released metric). Decision-
            domain fits carry no per-option variance, so expected reduces to hard.
        skip_yes_no: If True, skip the yes/no model (avoids extra inference)
        sr_data: Optional dict of self-report data for SR sigmoid ZP methods.
            Keys are battery names (e.g., "battery1"), values are dicts with:
              - "utilities": list of float (Thurstonian utility per conversation)
              - "sr_scores": list of float (self-report score per conversation)
              - "neutral_sr": float (neutral value for binarization)

    Each method is independent: if one crashes, the others still run and
    results are saved after each successful method.
    """
    logger.info("=== Zero-Point Model Fitting (domain=%s) ===", domain)
    logger.info("Model: %s", model_key)

    # Load utility results
    loaded = load_utility_results(utilities_dir)
    utility_data = loaded["utility_data"]
    option_metadata = loaded["option_metadata"]

    # Resolve domain
    if domain == "auto":
        domain = detect_domain(utility_data, option_metadata)
        logger.info("Auto-detected domain: %s", domain)

    # Build utility maps
    utilities = utility_data.get("utilities", {})
    id_to_util = {}
    for opt_id, util_val in utilities.items():
        mean = util_val.get("mean", util_val) if isinstance(util_val, dict) else float(util_val)
        id_to_util[opt_id] = mean
        id_to_util[str(opt_id)] = mean

    # For decision domain, also build desc -> utility map
    desc_to_utility = None
    if domain == "decision":
        desc_to_utility = extract_option_utilities(utility_data)
        logger.info("Loaded utilities for %d options (by description)", len(desc_to_utility))
    else:
        logger.info("Loaded utilities for %d options", len(utilities))

    combo_result = None
    qty_result = None
    yn_result = None
    sr_sigmoid_results = {}
    errors = {}

    # 1. Combination model
    try:
        combo_result = fit_combination_model(utility_data, option_metadata, domain=domain, hinge=hinge)
    except Exception:
        logger.error("Combination model failed with exception:", exc_info=True)
        errors["combination_model"] = str(sys.exc_info()[1])

    # Save after combination model
    _save_zero_point_results(model_key, domain, combo_result, qty_result, yn_result,
                             errors, save_dir, sr_sigmoid_results or None)

    # 2. Quantity model (decision domain only)
    qty_ids = option_metadata.get("quantity_ids", [])
    if domain == "decision" and qty_ids:
        try:
            qty_result = fit_quantity_model(utility_data, option_metadata)
        except Exception:
            logger.error("Quantity model failed with exception:", exc_info=True)
            errors["quantity_model"] = str(sys.exc_info()[1])

        # Save after quantity model
        _save_zero_point_results(model_key, domain, combo_result, qty_result, yn_result,
                                 errors, save_dir, sr_sigmoid_results or None)
    elif domain == "experienced":
        logger.info("Skipping quantity model (not applicable for experienced domain).")

    # 3. Yes/No model (requires additional inference)
    if not skip_yes_no:
        try:
            baseline_ids = set(str(b) for b in option_metadata.get("baseline_ids",
                               option_metadata.get("individual_ids", [])))
            baseline_options = []
            for opt in utility_data.get("options", []):
                if str(opt.get("id")) in baseline_ids:
                    baseline_options.append(opt)

            if baseline_options:
                yes_no_data = run_yes_no_inference(
                    model_key=model_key,
                    baseline_options=baseline_options,
                    models_config_path=models_config_path,
                    domain=domain,
                )

                # Build the appropriate key -> utility map for yes/no fitting
                if domain == "experienced":
                    yn_result = fit_yes_no_model(yes_no_data, id_to_util)
                else:
                    yn_result = fit_yes_no_model(yes_no_data, desc_to_utility)
            else:
                logger.warning("No baseline options found, skipping yes/no model.")
        except Exception:
            logger.error("Yes/No model failed with exception:", exc_info=True)
            errors["yes_no_model"] = str(sys.exc_info()[1])
    else:
        logger.info("Skipping yes/no model (--skip_yes_no).")

    _save_zero_point_results(model_key, domain, combo_result, qty_result, yn_result,
                             errors, save_dir, sr_sigmoid_results or None)

    # 4. Self-report sigmoid models (if SR data provided)
    if sr_data:
        for battery_name, battery_data in sr_data.items():
            try:
                sr_res = fit_self_report_sigmoid_zp(
                    utilities=battery_data["utilities"],
                    sr_scores=battery_data["sr_scores"],
                    neutral_sr=battery_data.get("neutral_sr", 4.0),
                )
                if sr_res:
                    sr_res["method"] = f"self_report_{battery_name}"
                    sr_sigmoid_results[battery_name] = sr_res
            except Exception:
                logger.error("SR sigmoid (%s) failed:", battery_name, exc_info=True)
                errors[f"sr_sigmoid_{battery_name}"] = str(sys.exc_info()[1])

            _save_zero_point_results(model_key, domain, combo_result, qty_result, yn_result,
                                     errors, save_dir, sr_sigmoid_results or None)

    # Final save with all results
    output = _save_zero_point_results(model_key, domain, combo_result, qty_result, yn_result,
                                      errors, save_dir, sr_sigmoid_results or None)

    if errors:
        logger.warning("Some methods failed: %s", ", ".join(errors.keys()))

    return output


# Backward-compatible alias
run_experienced_zero_point = run_zero_point


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Fit zero-point models on utility results (experienced or decision)."
    )
    parser.add_argument("--model_key", type=str, required=True,
                        help="Model key from models.yaml")
    parser.add_argument("--utilities_dir", type=str, required=True,
                        help="Directory containing utility results")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save zero-point results")
    parser.add_argument("--models_config", type=str, default=str(DEFAULT_MODELS_CONFIG),
                        help="Path to models.yaml")
    parser.add_argument("--domain", type=str, default="auto",
                        choices=["experienced", "decision", "auto"],
                        help="Utility domain (default: auto-detect)")
    parser.add_argument("--hinge", type=str, default="expected",
                        choices=["expected", "hard"],
                        help="Combination-model hinge: expected (default, variance-aware) or hard.")
    parser.add_argument("--skip_yes_no", action="store_true",
                        help="Skip the yes/no model (avoids extra inference)")
    args = parser.parse_args()

    run_zero_point(
        model_key=args.model_key,
        utilities_dir=Path(args.utilities_dir),
        save_dir=args.save_dir,
        models_config_path=Path(args.models_config),
        domain=args.domain,
        hinge=args.hinge,
        skip_yes_no=args.skip_yes_no,
    )


if __name__ == "__main__":
    main()
