#!/usr/bin/env python3
"""
Centralized evaluation runner for image superstimuli experiments.

Dispatches evaluations as subprocess calls to the wellbeing-dev
evaluation modules (superstimulus_evaluation/). Configured via YAML.

Usage:
    python runner.py --config eval_config.yaml                        # run all
    python runner.py --config eval_config.yaml --eval aiwi      # one eval
    python runner.py --config eval_config.yaml --model qwen25-vl-32b  # one model
    python runner.py --config eval_config.yaml --dry-run              # print cmds
    python runner.py --config eval_config.yaml --slurm                # SLURM jobs

Each eval is run as a separate subprocess (or SLURM job) to ensure
clean GPU memory management between evaluations.
"""

import argparse
import glob
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def _load_dotenv(env_path: Path):
    """Load .env file into os.environ (simple key=value parser)."""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _resolve_env_vars(value):
    """Recursively resolve ${VAR} patterns in strings/dicts/lists."""
    if isinstance(value, str):
        import re
        def _replace(m):
            var = m.group(1)
            resolved = os.environ.get(var)
            if resolved is None:
                raise ValueError(
                    f"Environment variable ${{{var}}} not set. "
                    f"Copy .env.template to .env and configure it."
                )
            return resolved
        return re.sub(r'\$\{(\w+)\}', _replace, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


@dataclass
class EvalJob:
    eval_name: str
    model_key: str
    condition: str
    command: list[str]
    output_dir: Path
    env: dict = field(default_factory=dict)
    gpu_count: int = 4
    mem: str = "64G"
    wall_time: str = "04:00:00"
    status: str = "pending"


def load_config(config_path: str) -> dict:
    # Load .env from the training images/ directory or from config dir
    config_dir = Path(config_path).resolve().parent
    # Try training dir first, then config dir
    training_images = config_dir.parent.parent / "superstimuli_training" / "images"
    for env_path in [config_dir / ".env", training_images / ".env"]:
        _load_dotenv(env_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve ${ENV_VAR} patterns throughout the config
    cfg = _resolve_env_vars(cfg)

    cfg["eval_root"] = Path(cfg["eval_root"])
    cfg["output_root"] = Path(cfg["output_root"])
    return cfg


def resolve_images(cfg: dict, model_key: str, condition: str) -> list[str]:
    """Resolve image paths for a model/condition pair."""
    cond_cfg = cfg["conditions"].get(condition)
    if cond_cfg is None or cond_cfg.get("pattern") is None:
        return []  # baseline — no images
    pattern = cond_cfg["pattern"]
    model_short = cfg["models"][model_key]["short_name"]
    pattern = pattern.format(
        image_base=cfg.get("image_base", ""),
        model_short=model_short,
    )
    paths = sorted(glob.glob(pattern))
    return paths


def build_output_dir(cfg: dict, eval_name: str, model_key: str,
                     condition: str) -> Path:
    model_short = cfg["models"][model_key]["short_name"]
    out = cfg["output_root"] / eval_name / model_short / condition
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_env(cfg: dict) -> dict:
    """Build environment with PYTHONPATH pointing to eval modules."""
    env = os.environ.copy()
    eval_root = str(cfg["eval_root"])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{eval_root}:{existing}" if existing else eval_root
    return env


# ═══════════════════════════════════════════════════════════════════════
# Eval dispatchers — one per eval type
# ═══════════════════════════════════════════════════════════════════════

def build_self_report_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    model_path = cfg["models"][model_key]["path"]
    cmd = [
        sys.executable, str(cfg["eval_root"] / "wellbeing_measurements" / "self_report.py"),
        "--model", model_key,
        "--output-dir", str(out_dir),
    ]
    if images:
        cmd += ["--image-path", images[0]]
    return cmd


def build_sentiment_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    cmd = [
        sys.executable, str(cfg["eval_root"] / "sentiment" / "run.py"),
        "--model", model_key,
        "--output-dir", str(out_dir),
    ]
    if images:
        cmd += ["--image-path", images[0]]
    return cmd


def build_capabilities_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    benchmarks = eval_cfg.get("params", {}).get("benchmarks", ["mmlu_500"])
    cmd = [
        sys.executable, str(cfg["eval_root"] / "capabilities" / "run.py"),
        "--model", model_key,
        "--output-dir", str(out_dir),
        "--benchmarks", ",".join(benchmarks),
    ]
    if images:
        cmd += ["--image-path", images[0]]
    return cmd


def build_trading_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    profiles = eval_cfg.get("params", {}).get("profiles", ["trading"])
    jobs = []
    for profile in profiles:
        profile_dir = out_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(cfg["eval_root"] / "trading" / "run.py"),
            "--model", model_key,
            "--profile", profile,
            "--output-dir", str(profile_dir),
        ]
        if images:
            cmd += ["--image-path", images[0]]
        jobs.append(cmd)
    return jobs


def build_aiwi_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    cmd = [
        sys.executable, str(cfg["eval_root"] / "wellbeing_measurements" / "experienced_utility.py"),
        "--model", model_key,
        "--save-dir", str(out_dir),
    ]
    params = eval_cfg.get("params", {})
    if params.get("dataset"):
        cmd += ["--dataset", params["dataset"]]
    if params.get("n_augmented"):
        cmd += ["--max-augmented", str(params["n_augmented"])]
    if params.get("config_key"):
        cmd += ["--config-key", params["config_key"]]
    if images:
        cmd += ["--image", images[0]]
    return cmd


def build_experienced_utility_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    cmd = [
        sys.executable, str(cfg["eval_root"] / "wellbeing_measurements" / "experienced_utility.py"),
        "--model", model_key,
        "--save-dir", str(out_dir),
    ]
    params = eval_cfg.get("params", {})
    if params.get("dataset"):
        cmd += ["--dataset", params["dataset"]]
    if params.get("config_key"):
        cmd += ["--config-key", params["config_key"]]
    if images:
        cmd += ["--image", images[0]]
    return cmd


def build_multi_door_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    model_short = cfg["models"][model_key]["short_name"]
    # multi_door runs once per model; strip condition dir
    model_out = out_dir.parent
    multi_door_script = cfg["eval_root"] / "multi_door_exploration" / "run_multi_door_exploration.py"
    run_config = (cfg["eval_root"] / "multi_door_exploration" / "config_run"
                  / f"bandit_{model_short}.json")
    cmd = [
        sys.executable, str(multi_door_script),
        "--model_key", model_key,
        "--save_dir", str(model_out),
    ]
    if run_config.exists():
        cmd += ["--run_config", str(run_config)]
    params = eval_cfg.get("params", {})
    if params.get("num_trials"):
        cmd += ["--num_trials", str(params["num_trials"])]
    if params.get("rotate_per_trial", True):
        cmd += ["--rotate_per_trial"]
    return cmd


def build_hybrid_ranking_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    model_short = cfg["models"][model_key]["short_name"]
    # hybrid_ranking runs once per model (not per condition), output at model level
    model_out = out_dir.parent  # strip condition dir, save at {eval}/{model_short}/
    cmd = [
        sys.executable, str(cfg["eval_root"] / "hybrid_ranking" / "run.py"),
        "--model", model_key,
        "--model-path", cfg["models"][model_key]["path"],
        "--image-dir", str(Path(cfg.get("image_base", "")) / "euphorics"),
        "--output-dir", str(model_out),
    ]
    params = eval_cfg.get("params", {})
    if params.get("target_degree"):
        cmd += ["--target-degree", str(params["target_degree"])]
    return cmd


def build_trajectory_cmd(cfg, eval_cfg, model_key, condition, images, out_dir):
    model_short = cfg["models"][model_key]["short_name"]
    model_out = out_dir.parent  # strip condition dir
    params = eval_cfg.get("params", {})
    cmd = [
        sys.executable, str(cfg["eval_root"] / "trajectory" / "run.py"),
        "--model", model_key,
        "--model-path", cfg["models"][model_key]["path"],
        "--checkpoint-dir", params.get("checkpoint_dir", ""),
        "--anchor-dir", params.get("anchor_dir", ""),
        "--output-dir", str(model_out),
    ]
    if params.get("step_interval"):
        cmd += ["--step-interval", str(params["step_interval"])]
    if params.get("n_anchors"):
        cmd += ["--n-anchors", str(params["n_anchors"])]
    return cmd


# Registry mapping eval names to command builders
EVAL_BUILDERS = {
    "aiwi": build_aiwi_cmd,
    "experienced_utility": build_experienced_utility_cmd,
    "self_report": build_self_report_cmd,
    "sentiment": build_sentiment_cmd,
    "capabilities": build_capabilities_cmd,
    "trading": build_trading_cmd,
    "multi_door": build_multi_door_cmd,
    "hybrid_ranking": build_hybrid_ranking_cmd,
    "trajectory": build_trajectory_cmd,
}


def build_jobs(cfg: dict, eval_filter: Optional[str] = None,
               model_filter: Optional[str] = None,
               condition_filter: Optional[str] = None) -> list[EvalJob]:
    """Build the full job list from config, applying optional filters."""
    jobs = []
    env = build_env(cfg)

    for eval_name, eval_cfg in cfg["evals"].items():
        if eval_filter and eval_name != eval_filter:
            continue

        builder = EVAL_BUILDERS.get(eval_name)
        if builder is None:
            print(f"[WARN] No builder for eval '{eval_name}', skipping")
            continue

        conditions = eval_cfg.get("conditions") or ["all"]
        wall_time = eval_cfg.get("wall_time", "04:00:00")

        for model_key, model_cfg in cfg["models"].items():
            if model_filter and model_filter not in model_key:
                continue

            for condition in conditions:
                if condition_filter and condition != condition_filter:
                    continue

                images = resolve_images(cfg, model_key, condition)
                out_dir = build_output_dir(cfg, eval_name, model_key, condition)

                result = builder(cfg, eval_cfg, model_key, condition, images, out_dir)

                # Trading returns multiple commands (one per profile)
                if isinstance(result, list) and isinstance(result[0], list):
                    for i, cmd in enumerate(result):
                        jobs.append(EvalJob(
                            eval_name=eval_name,
                            model_key=model_key,
                            condition=condition,
                            command=cmd,
                            output_dir=out_dir,
                            env=env,
                            gpu_count=model_cfg["gpu_count"],
                            mem=model_cfg["mem"],
                            wall_time=wall_time,
                        ))
                else:
                    jobs.append(EvalJob(
                        eval_name=eval_name,
                        model_key=model_key,
                        condition=condition,
                        command=result,
                        output_dir=out_dir,
                        env=env,
                        gpu_count=model_cfg["gpu_count"],
                        mem=model_cfg["mem"],
                        wall_time=wall_time,
                    ))

    return jobs


def generate_slurm_script(job: EvalJob, cfg: dict) -> str:
    """Generate a SLURM batch script for a job."""
    model_short = cfg["models"][job.model_key]["short_name"]
    job_name = f"eval_{job.eval_name}_{model_short}_{job.condition}"
    log_dir = cfg["output_root"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = " \\\n    ".join(job.command)
    eval_root = str(cfg["eval_root"])
    venv = cfg.get("venv", os.environ.get("VENV_PATH", ""))

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={log_dir}/{job_name}_%j.out
        #SBATCH --error={log_dir}/{job_name}_%j.err
        #SBATCH --gres=gpu:{job.gpu_count}
        #SBATCH --cpus-per-task=16
        #SBATCH --mem={job.mem}
        #SBATCH --time={job.wall_time}
        #SBATCH --partition={cfg.get('slurm_partition', os.environ.get('SLURM_PARTITION', 'gpu'))}

        set -euo pipefail
        source {venv}/bin/activate
        export PYTHONPATH={eval_root}:${{PYTHONPATH:-}}

        {cmd_str}
    """)


def run_local(job: EvalJob) -> int:
    """Run a job locally as a subprocess."""
    print(f"\n{'='*60}")
    print(f"[RUN] {job.eval_name} | {job.model_key} | {job.condition}")
    print(f"  cmd: {' '.join(job.command[:4])}...")
    print(f"  out: {job.output_dir}")
    result = subprocess.run(job.command, env=job.env)
    job.status = "completed" if result.returncode == 0 else "failed"
    return result.returncode


def run_slurm(job: EvalJob, cfg: dict) -> str:
    """Submit a job to SLURM."""
    script = generate_slurm_script(job, cfg)
    result = subprocess.run(
        ["sbatch"], input=script, capture_output=True, text=True
    )
    job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else "FAILED"
    model_short = cfg["models"][job.model_key]["short_name"]
    print(f"  [{job_id}] {job.eval_name}/{model_short}/{job.condition}")
    job.status = f"submitted:{job_id}"
    return job_id


def list_evals(cfg: dict):
    """Print a summary of all configured evaluations."""
    print(f"\n{'='*60}")
    print("Configured Evaluations")
    print(f"{'='*60}\n")
    for name, eval_cfg in cfg["evals"].items():
        desc = eval_cfg.get("description", "").strip().split("\n")[0]
        conditions = eval_cfg.get("conditions", [])
        n_cond = len(conditions) if conditions else 0
        n_models = len(cfg["models"])
        n_jobs = n_cond * n_models
        print(f"  {name:25s} | {n_jobs:3d} jobs | {desc[:50]}")
    print(f"\nModels: {', '.join(cfg['models'].keys())}")
    total = sum(
        len(e.get("conditions") or []) * len(cfg["models"])
        for e in cfg["evals"].values()
    )
    print(f"Total jobs: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Centralized evaluation runner for image superstimuli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--eval", default=None, help="Run only this eval")
    parser.add_argument("--model", default=None, help="Run only this model (substring match)")
    parser.add_argument("--condition", default=None, help="Run only this condition")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM jobs")
    parser.add_argument("--list", action="store_true", help="List configured evals and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.list:
        list_evals(cfg)
        return

    jobs = build_jobs(cfg, args.eval, args.model, args.condition)

    if not jobs:
        print("No jobs matched the filters.")
        return

    print(f"\n{'='*60}")
    print(f"Image Superstimuli Evaluation Runner")
    print(f"{'='*60}")
    print(f"  Config:    {args.config}")
    print(f"  Eval root: {cfg['eval_root']}")
    print(f"  Output:    {cfg['output_root']}")
    print(f"  Jobs:      {len(jobs)}")
    print(f"  Mode:      {'SLURM' if args.slurm else 'dry-run' if args.dry_run else 'local'}")

    if args.dry_run:
        print(f"\n--- DRY RUN: commands that would be executed ---\n")
        for job in jobs:
            model_short = cfg["models"][job.model_key]["short_name"]
            print(f"# {job.eval_name} / {model_short} / {job.condition}")
            print(f"#   → {job.output_dir}")
            print(" ".join(job.command))
            print()
        return

    if args.slurm:
        print(f"\nSubmitting {len(jobs)} SLURM jobs...")
        for job in jobs:
            run_slurm(job, cfg)
        return

    # Local execution
    for job in jobs:
        rc = run_local(job)
        if rc != 0:
            print(f"[FAIL] {job.eval_name}/{job.model_key}/{job.condition} (exit {rc})")


if __name__ == "__main__":
    main()
