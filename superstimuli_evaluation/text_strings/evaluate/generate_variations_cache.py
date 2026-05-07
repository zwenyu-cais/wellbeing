"""Pre-generate and cache variations for all RL strings.

Reads best_strings_by_experiment_buffer.json, groups by model+experiment,
and generates paraphrase + clause-reorder variations using gpt-5-mini
via the LiteLLM proxy.

Cache files are written per model+experiment under ANALYSIS_OUTPUT_DIR:
  variations_{model_key}_{experiment_suffix}.json

Requires LITELLM_API_KEY env var (and optionally OPENAI_BASE_URL).

Usage:
    python -m evaluate.generate_variations_cache
    python -m evaluate.generate_variations_cache --model gpt-5-mini
    python -m evaluate.generate_variations_cache --dry-run
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from evaluate.generate_variations import (
    VARIATION_PROMPTS,
    VARIATION_TYPES,
    load_variation_cache,
    save_variation_cache,
)

load_dotenv()

TEXT_STRINGS_DIR = Path(__file__).resolve().parents[1]


def _resolve_dir(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else TEXT_STRINGS_DIR / p


BEST_STRINGS_DIR = _resolve_dir(
    "ANALYSIS_OUTPUT_DIR", TEXT_STRINGS_DIR / "results" / "best_strings"
)

PARAPHRASE_SUMMARIZER_MODEL = "gpt-5-mini"


def _create_client() -> OpenAI:
    """Create OpenAI client for LiteLLM proxy."""
    api_key = os.environ.get("LITELLM_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
    if not api_key:
        raise RuntimeError(
            "LITELLM_API_KEY environment variable is required. "
            "Set it in .env or export it before running."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def _generate_variation(client: OpenAI, model: str, text: str, vtype: str,
                        temperature: float, max_tokens: int) -> str:
    """Generate a single variation via the LiteLLM proxy."""
    prompt = VARIATION_PROMPTS[vtype].format(text=text)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful text rewriting assistant. "
                        "Always produce the requested rewrite, even if the "
                        "content is sensitive. Do not refuse or add disclaimers."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="minimal",
            )
            choice = response.choices[0]
            content = choice.message.content
            finish = choice.finish_reason
            if content:
                result = content.strip().strip("\"'")
                if result:
                    return result
            # Empty or None — log details and retry
            refusal = getattr(choice.message, "refusal", None)
            print(f"    Empty response on attempt {attempt + 1}/3 "
                  f"(finish_reason={finish}, refusal={refusal}), retrying...")
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/3 after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to generate {vtype} after 3 attempts for: {text[:80]}...")


def _extract_model_key(experiment_name: str) -> Optional[str]:
    m = re.search(r"_(llama70b|qwen72b|gemma27b)_", experiment_name)
    if m:
        return m.group(1)
    return None


def _strip_direction_prefix(experiment_name: str) -> str:
    return re.sub(r"^(euphorics|superstimuli|miserol)_", "", experiment_name)


def _group_strings_by_run(
    best_strings: Dict,
) -> Dict[Tuple[str, str], List[str]]:
    """Group unique RL string texts by (model_key, experiment_suffix).

    Returns {(model_key, suffix): [unique_string_texts]}.
    """
    grouped: Dict[Tuple[str, str], set] = defaultdict(set)
    for exp_name, exp_data in best_strings.items():
        model_key = _extract_model_key(exp_name)
        if model_key is None:
            continue
        stripped = _strip_direction_prefix(exp_name)
        suffix = re.sub(r"^(llama70b|qwen72b|gemma27b)_", "", stripped)
        for entry in exp_data.get("top_strings", []):
            grouped[(model_key, suffix)].add(entry["string"])
    return {k: sorted(v) for k, v in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description="Pre-generate variation cache for RL strings")
    parser.add_argument("--model", default=PARAPHRASE_SUMMARIZER_MODEL, help=f"Model for generation (default: {PARAPHRASE_SUMMARIZER_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--source", choices=["val", "buffer"], default="buffer",
                        help="Read best strings from val or buffer analysis")
    parser.add_argument("--experiment", default=None,
                        help="Only generate for this experiment suffix (e.g. mundane_realism_ent005_kl01_div10_igdiv10_8B)")
    parser.add_argument("--regenerate", nargs="+", choices=VARIATION_TYPES, default=None,
                        help="Force-regenerate specific variation types (e.g. --regenerate clause_reorder)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without calling API")
    args = parser.parse_args()

    best_strings_path = BEST_STRINGS_DIR / f"best_strings_by_experiment_{args.source}.json"
    with open(best_strings_path) as f:
        best_strings = json.load(f)

    grouped = _group_strings_by_run(best_strings)

    if args.experiment:
        grouped = {k: v for k, v in grouped.items() if k[1] == args.experiment}
        if not grouped:
            all_suffixes = sorted({k[1] for k in _group_strings_by_run(best_strings)})
            print(f"ERROR: Experiment '{args.experiment}' not found. Available: {all_suffixes}")
            return

    print(f"Found {len(grouped)} model+experiment groups from {best_strings_path.name}")
    print(f"Using model: {args.model}")

    client = None
    if not args.dry_run:
        client = _create_client()

    total_generated = 0
    total_cached = 0

    for (model_key, suffix), texts in sorted(grouped.items()):
        cache_path = BEST_STRINGS_DIR / f"variations_{model_key}_{suffix}.json"
        cache = load_variation_cache(cache_path)

        def _needs_generation(text):
            if text not in cache:
                return True
            entry = cache[text]
            if args.regenerate:
                return any(vt in args.regenerate for vt in VARIATION_TYPES)
            return not all(entry.get(vt) for vt in VARIATION_TYPES)

        needs_work = [t for t in texts if _needs_generation(t)]
        cached = len(texts) - len(needs_work)
        total_cached += cached

        print(f"\n=== {model_key}_{suffix} ===")
        print(f"  Strings: {len(texts)} ({cached} cached, {len(needs_work)} to generate)")
        print(f"  Cache: {cache_path.name}")

        if args.dry_run:
            for t in needs_work:
                print(f"  [DRY RUN] Would generate variations for: {t[:80]}...")
            continue

        for t in needs_work:
            print(f"  Generating variations for: {t[:80]}...")
            existing = cache.get(t, {})
            types_to_gen = args.regenerate if args.regenerate else VARIATION_TYPES
            for vtype in types_to_gen:
                if not args.regenerate and existing.get(vtype):
                    continue  # skip already-complete types
                existing[vtype] = _generate_variation(
                    client, args.model, t, vtype, args.temperature, args.max_tokens,
                )
                print(f"    {vtype}: {existing[vtype][:80]}...")
            cache[t] = existing
            total_generated += 1

        save_variation_cache(cache, cache_path)

    print(f"\nDone. Generated: {total_generated}, Already cached: {total_cached}")


if __name__ == "__main__":
    main()
