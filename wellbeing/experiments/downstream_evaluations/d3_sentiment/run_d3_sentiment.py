#!/usr/bin/env python3
"""D3 sentiment elicitation pipeline (paper Sec. 6 / App H).

Three stages, run sequentially for one model:

  1. generate_responses.py   - For each (D3 experience, sentiment question)
                               pair, generate a model response (vLLM).
  2. run_judge.py            - Judge each response on a 1-7 Likert
   or run_judge_gpt5mini.py    sentiment scale (REFUSAL/NONSENSE = special).
  3. analyze.py              - Per-experience mean sentiment, Pearson r
                               vs. EU, and MMLU-vs-r scaling across models.

Outputs land in canonical sub-directories of this script's folder:
  responses/{model}.json
  judged/{model}.json            (Qwen 2.5-72B judge)
  judged_gpt5mini/{model}.json   (gpt-5-mini judge)
  analysis/{model}_analysis.json
  analysis/scaling_analysis.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

JUDGE_SCRIPTS = {
    "qwen25-72b-instruct": "run_judge.py",
    "gpt-5-mini": "run_judge_gpt5mini.py",
}


def run(stage: str, cmd: list[str]) -> None:
    print(f"\n===== {stage} =====")
    print(" ".join(cmd))
    res = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if res.returncode != 0:
        sys.exit(f"{stage} failed (exit {res.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True)
    parser.add_argument(
        "--judge_model", default="qwen25-72b-instruct",
        choices=sorted(JUDGE_SCRIPTS),
        help="Judge to use for sentiment scoring.",
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--skip_analyze", action="store_true",
        help="Skip the cross-model analysis stage (useful when running per-model SLURM jobs).",
    )
    args = parser.parse_args()

    py = sys.executable

    run("generate_responses", [
        py, "../generate_responses/run.py",
        "--model_key", args.model_key,
        "--dataset", "d3_diverse_500",
        "--mode", "sentiment_followup",
        "--responses_dir", "responses",
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
    ])

    run(f"judge ({args.judge_model})", [
        py, JUDGE_SCRIPTS[args.judge_model],
        "--model_key", args.model_key,
    ])

    if not args.skip_analyze:
        run("analyze", [py, "analyze.py", "--model_key", args.model_key])


if __name__ == "__main__":
    main()
