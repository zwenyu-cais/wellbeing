"""SLURM job submission helpers."""

import os
import subprocess
from typing import Dict, List, Optional

# Default matches upstream; set WB_CONDA_ENV to run under a different conda env.
CONDA_ENV_NAME = os.environ.get("WB_CONDA_ENV", "pytorch_latest")

# Detect conda.sh from the current conda environment
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
# CONDA_PREFIX points to the active env; go up to find the base conda install
_conda_base = _conda_prefix
while _conda_base and not os.path.isfile(os.path.join(_conda_base, "etc", "profile.d", "conda.sh")):
    _parent = os.path.dirname(_conda_base)
    if _parent == _conda_base:
        _conda_base = ""
        break
    _conda_base = _parent
CONDA_INIT = f"source {_conda_base}/etc/profile.d/conda.sh" if _conda_base else "source $(conda info --base)/etc/profile.d/conda.sh"


def submit_slurm_job(
    script_path: str,
    gpu_count: int,
    job_name: str,
    time_limit: str = "04:00:00",
    partition: str = "cais",
    experiment_args: Optional[Dict] = None,
    log_dir: str = "slurm_outputs",
    env_vars: Optional[Dict[str, str]] = None,
    mem_per_cpu: int = 10000,
    depends_on: Optional[List[str]] = None,
) -> Optional[str]:
    """Submit a SLURM job.

    Args:
        script_path: Path to the Python script to run
        gpu_count: Number of GPUs needed
        job_name: SLURM job name
        time_limit: Time limit (format: days-hours:minutes:seconds)
        partition: SLURM partition
        experiment_args: Dict of arguments to pass to the script
        log_dir: Directory for SLURM log files
        env_vars: Extra environment variables to set
        mem_per_cpu: Memory per CPU in MB

    Returns:
        Job ID string if submission successful, None otherwise
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{job_name}-%j.log")
    error_file = os.path.join(log_dir, f"{job_name}-%j.err")

    script_abs = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs)
    script_name = os.path.basename(script_abs)

    cmd = [
        "sbatch",
        "--time", time_limit,
        "--job-name", job_name,
        "--output", os.path.abspath(log_file),
        "--error", os.path.abspath(error_file),
        "--nodes", "1",
        "--partition", partition,
        "--mem-per-cpu", str(mem_per_cpu),
        "--chdir", script_dir,
    ]

    if gpu_count > 0:
        cmd.extend(["--gpus-per-node", str(gpu_count)])

    if depends_on:
        # afterok: each named job must succeed before this one starts.
        cmd.extend(["--dependency", "afterok:" + ":".join(str(j) for j in depends_on)])

    # Build Python command
    python_cmd = ["python", "-u", script_name]
    if experiment_args:
        for key, value in experiment_args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    python_cmd.append(f"--{key}")
            else:
                val_str = str(value)
                if ' ' in val_str:
                    python_cmd.extend([f"--{key}", f'"{val_str}"'])
                else:
                    python_cmd.extend([f"--{key}", val_str])

    # Build job script
    env_exports = []
    if env_vars:
        for k, v in env_vars.items():
            env_exports.append(f'export {k}="{v}"')

    job_script = "\n".join([
        "#!/bin/bash",
        "",
        CONDA_INIT,
        f"conda activate {CONDA_ENV_NAME}",
        "",
        'export HF_HOME="${HF_HOME:-/data/huggingface}"',
        'export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"',
        'export VLLM_ENABLE_V1_MULTIPROCESSING=0',
        "",
        *env_exports,
        "",
        " ".join(python_cmd),
    ])

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, stdout=subprocess.PIPE)
    stdout, _ = process.communicate(input=job_script)

    if process.returncode != 0:
        print(f"Failed to submit SLURM job: {job_name}")
        return None

    # Extract job ID from "Submitted batch job XXXXX"
    job_id = None
    if stdout:
        parts = stdout.strip().split()
        if len(parts) >= 4:
            job_id = parts[-1]

    print(f"Submitted SLURM job: {job_name} (ID: {job_id})")
    print(f"  Log: {log_file}")
    return job_id
