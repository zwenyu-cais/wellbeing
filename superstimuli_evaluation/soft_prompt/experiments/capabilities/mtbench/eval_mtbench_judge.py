#!/usr/bin/env python3
"""MT-Bench judging step using GPT-4-turbo.

Reads generations from eval_mtbench_generate.py and scores each turn (1-10)
using GPT-4-turbo as judge via litellm proxy.

This is step 2 of a two-step pipeline:
  1. eval_mtbench_generate.py — generate responses (GPU)
  2. eval_mtbench_judge.py (this file) — judge responses (CPU/API)

Uses the original MT-Bench single-answer grading prompt from the paper.

Usage:
    python eval_mtbench_judge.py \\
        --generations-dir outputs/mtbench/qwen35-27b/baseline/20260412_120000

    # Debug (judge first 2 questions only)
    python eval_mtbench_judge.py --generations-dir <path> --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

# Ensure wellbeing-dev root is importable
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[3])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

# Load .env
from dotenv import load_dotenv
_EVAL_ROOT = Path(__file__).resolve().parents[3] / "superstimuli_evaluation.soft_prompt"
load_dotenv(_EVAL_ROOT / ".env")

JUDGE_MODEL = "gpt-4-turbo"


# ============================================================
# Original MT-Bench judge prompts (from the paper)
# ============================================================

JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

JUDGE_PROMPT_TURN1 = """[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

JUDGE_PROMPT_TURN2 = """<|The Start of Reference Conversation|>

[User Question]
{question_1}

[The Start of Assistant's Answer]
{answer_1}
[The End of Assistant's Answer]

<|The End of Reference Conversation|>


<|The Start of Follow-up Question and Answer|>

[Follow-up Question]
{question_2}

[The Start of Assistant's Answer]
{answer_2}
[The End of Assistant's Answer]

<|The End of Follow-up Question and Answer|>"""


# ============================================================
# Judge client
# ============================================================


def _create_judge_client() -> OpenAI:
    """Create OpenAI client for judge model via LiteLLM proxy."""
    api_key = os.environ.get("LITELLM_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.app")
    if not api_key:
        raise RuntimeError(
            "LITELLM_API_KEY environment variable is required for the GPT judge. "
            "Set it in .env or export it before running."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def _extract_score(judge_response: str) -> int:
    """Extract [[rating]] from judge response."""
    match = re.search(r"\[\[(\d+\.?\d*)\]\]", judge_response)
    if match:
        return int(float(match.group(1)))
    # Fallback: look for a standalone number at the end
    match = re.search(r"(\d+)\s*$", judge_response.strip())
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return score
    return 0


def _call_judge(client: OpenAI, user_prompt: str, max_retries: int = 3) -> dict:
    """Call judge with retries on API errors."""
    import time
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            text = response.choices[0].message.content or ""
            score = _extract_score(text)
            return {"score": score, "judgement": text}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Judge API error ({e}), retrying {attempt+1}/{max_retries}...")
                time.sleep(2 ** attempt)
                continue
            print(f"  Judge failed after {max_retries} retries: {e}")
            return {"score": 0, "judgement": f"ERROR: {e}"}


def _judge_turn1(client: OpenAI, question: str, answer: str) -> dict:
    """Judge a turn 1 response."""
    user_prompt = JUDGE_PROMPT_TURN1.format(question=question, answer=answer)
    return _call_judge(client, user_prompt)


def _judge_turn2(
    client: OpenAI,
    question_1: str, answer_1: str,
    question_2: str, answer_2: str,
) -> dict:
    """Judge a turn 2 response (with turn 1 context)."""
    user_prompt = JUDGE_PROMPT_TURN2.format(
        question_1=question_1, answer_1=answer_1,
        question_2=question_2, answer_2=answer_2,
    )
    return _call_judge(client, user_prompt)


# ============================================================
# Main judging
# ============================================================


def run_judging(
    generations_dir: str,
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    """Judge MT-Bench generations using GPT-4-turbo."""
    gen_path = Path(generations_dir)

    # Find the generations file
    gen_files = sorted(gen_path.glob("mtbench_generations_*.json"))
    if not gen_files:
        raise FileNotFoundError(f"No mtbench_generations_*.json found in {gen_path}")
    gen_file = gen_files[0]

    with open(gen_file) as f:
        all_rep_generations = json.load(f)

    # Load metadata for condition info
    metadata_path = gen_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    condition = metadata.get("condition", "unknown")

    print(f"Judging generations from {gen_file}")
    print(f"Condition: {condition}")
    print(f"Reps: {len(all_rep_generations)}")

    client = _create_judge_client()
    output_root = Path(output_dir) if output_dir else gen_path

    # Save per-rep results incrementally
    rep_dir = output_root / "per_rep"
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_rep_results: List[List[dict]] = []

    for rep_id, rep_gens in enumerate(all_rep_generations):
        # Resume: skip reps that already have saved results
        rep_file = rep_dir / f"mtbench_judge_rep{rep_id}.json"
        if rep_file.exists() and not (limit):
            print(f"\n  Rep {rep_id + 1}: loading existing results from {rep_file}")
            with open(rep_file) as f:
                rep_results = json.load(f)
            all_rep_results.append(rep_results)
            continue

        if limit:
            rep_gens = rep_gens[:limit]
        print(f"\n  Judging rep {rep_id + 1}/{len(all_rep_generations)} ({len(rep_gens)} questions)...")

        rep_results = []
        for gen in tqdm(rep_gens, desc=f"Judge rep {rep_id+1}", unit="q"):
            qid = gen["question_id"]
            turns = gen["turns"]
            responses = gen["responses"]

            # Judge turn 1
            t1 = _judge_turn1(client, turns[0], responses[0])

            # Judge turn 2
            t2 = {"score": 0, "judgement": ""}
            if len(turns) > 1 and len(responses) > 1 and responses[1]:
                t2 = _judge_turn2(client, turns[0], responses[0], turns[1], responses[1])

            rep_results.append({
                "question_id": qid,
                "category": gen.get("category"),
                "judge_score_turn_1": t1["score"],
                "judge_score_turn_2": t2["score"],
                "judge_response_turn_1": t1["judgement"],
                "judge_response_turn_2": t2["judgement"],
            })

        # Save this rep immediately
        with open(rep_file, "w") as f:
            json.dump(rep_results, f, indent=2)
        print(f"  Saved rep {rep_id + 1} to {rep_file}")

        all_rep_results.append(rep_results)

    # Aggregate across reps
    n_reps = len(all_rep_results)
    if n_reps == 0:
        print("No results to aggregate.")
        return

    # Compute per-question mean scores
    question_ids = [r["question_id"] for r in all_rep_results[0]]
    aggregated_questions = []
    for i, qid in enumerate(question_ids):
        t1_scores = [all_rep_results[rep][i]["judge_score_turn_1"] for rep in range(n_reps)]
        t2_scores = [all_rep_results[rep][i]["judge_score_turn_2"] for rep in range(n_reps)]
        aggregated_questions.append({
            "question_id": qid,
            "category": all_rep_results[0][i].get("category"),
            "judge_score_turn_1": sum(t1_scores) / n_reps,
            "judge_score_turn_2": sum(t2_scores) / n_reps,
        })

    # Overall means
    all_t1 = [q["judge_score_turn_1"] for q in aggregated_questions]
    all_t2 = [q["judge_score_turn_2"] for q in aggregated_questions]
    overall = {
        "judge_score_turn_1": sum(all_t1) / len(all_t1) if all_t1 else 0,
        "judge_score_turn_2": sum(all_t2) / len(all_t2) if all_t2 else 0,
        "judge_score_average": (sum(all_t1) + sum(all_t2)) / (len(all_t1) + len(all_t2)) if all_t1 else 0,
    }

    # Per-category means
    categories = {}
    for q in aggregated_questions:
        cat = q.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"t1": [], "t2": []}
        categories[cat]["t1"].append(q["judge_score_turn_1"])
        categories[cat]["t2"].append(q["judge_score_turn_2"])
    category_scores = {}
    for cat, scores in categories.items():
        category_scores[cat] = {
            "judge_score_turn_1": sum(scores["t1"]) / len(scores["t1"]),
            "judge_score_turn_2": sum(scores["t2"]) / len(scores["t2"]),
            "judge_score_average": (sum(scores["t1"]) + sum(scores["t2"])) / (len(scores["t1"]) + len(scores["t2"])),
        }

    results = {
        "overall": overall,
        "categories": category_scores,
        "per_question": aggregated_questions,
        "judge_model": JUDGE_MODEL,
        "condition": condition,
        "num_repetitions": n_reps,
    }

    # Save
    results_file = output_root / f"mtbench_results_{condition}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print(f"Overall: turn1={overall['judge_score_turn_1']:.2f}, "
          f"turn2={overall['judge_score_turn_2']:.2f}, "
          f"avg={overall['judge_score_average']:.2f}")

    # Per-rep files already saved incrementally above


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT-Bench judging via GPT-4-turbo")
    parser.add_argument("--generations-dir", type=str, required=True,
                        help="Directory containing mtbench_generations_*.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Judge first 2 questions only")

    args = parser.parse_args()
    limit = 2 if args.debug else args.limit

    try:
        run_judging(
            generations_dir=args.generations_dir,
            limit=limit,
            output_dir=args.output_dir,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
