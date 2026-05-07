"""Auto-start and manage a vLLM server for soft prompt direct injection.

Starts vLLM with ``--enable-prompt-embeds`` as a subprocess, waits for it
to become healthy, and provides cleanup on exit (including atexit + signal
handlers so the server is killed even if the parent process crashes).
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import requests
import yaml


def _available_gpu_count() -> int:
    """Return the number of visible GPUs."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip():
        return len([d for d in cuda_visible.split(",") if d.strip()])
    try:
        import torch
        return torch.cuda.device_count() or 1
    except ImportError:
        return 1


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 600, poll_interval: float = 5) -> bool:
    """Poll the vLLM health endpoint until it responds or timeout."""
    deadline = time.time() + timeout
    health_url = f"{url}/health"
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(poll_interval)
    return False


class VLLMServer:
    """Manages a vLLM server subprocess.

    Usage::

        server = VLLMServer(model_path, gpu_count=4)
        server.start()        # blocks until healthy
        # ... use server.url ...
        server.stop()         # or let atexit handle it

    The server is automatically killed on:
    - Normal exit (atexit)
    - SIGTERM / SIGINT
    - Context manager __exit__
    """

    def __init__(
        self,
        model_path: str,
        gpu_count: int = 1,
        port: Optional[int] = None,
        dtype: str = "bfloat16",
        host: str = "0.0.0.0",
        extra_args: Optional[list] = None,
        startup_timeout: float = 600,
        log_dir: Optional[str] = None,
        enable_prompt_embeds: bool = True,
        enable_prefix_caching: bool = False,
    ):
        self.model_path = model_path
        self.gpu_count = gpu_count
        self.port = port or _find_free_port()
        self.dtype = dtype
        self.host = host
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.log_dir = log_dir
        self.enable_prompt_embeds = enable_prompt_embeds
        self.enable_prefix_caching = enable_prefix_caching

        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._pid_file: Optional[str] = None
        self._registered_atexit = False
        self._prev_sigterm = None
        self._prev_sigint = None
        self._stopping = False  # re-entrancy guard for stop()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self) -> str:
        """Start the vLLM server and block until healthy.

        Returns:
            The server URL (e.g. ``http://localhost:8042``).
        """
        if self._process is not None:
            raise RuntimeError("Server already started")

        _env = os.environ.copy()

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.gpu_count),
            "--dtype", self.dtype,
            "--trust-remote-code",
            "--enforce-eager",
            "--limit-mm-per-prompt", '{"image": 0}',
            *self.extra_args,
        ]
        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if self.enable_prompt_embeds:
            cmd.append("--enable-prompt-embeds")

        # Set up log file
        log_dir = self.log_dir or os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"vllm_server_port{self.port}.log")

        print(f"[VLLMServer] Starting: {' '.join(cmd)}")
        print(f"[VLLMServer] Port: {self.port}")
        print(f"[VLLMServer] Log: {self.log_path}")

        self._log_file = open(self.log_path, "w")
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=_env,
            # Own process group so killpg() doesn't hit parent.
            # Using setpgrp (not setsid) so it stays in the same session —
            # SLURM can still reach it when the job is cancelled.
            preexec_fn=os.setpgrp,
        )

        # Write PID file for bash-level cleanup
        self._pid_file = os.path.join(log_dir, f"vllm_server_port{self.port}.pid")
        with open(self._pid_file, "w") as pf:
            pf.write(str(self._process.pid))
        print(f"[VLLMServer] PID file: {self._pid_file}")

        # Register cleanup handlers
        self._register_cleanup()

        print(f"[VLLMServer] Waiting for server to become healthy (timeout={self.startup_timeout}s) ...")
        if not _wait_for_server(self.url, timeout=self.startup_timeout):
            # Server didn't start — dump output and abort
            self.stop()
            raise RuntimeError(
                f"vLLM server failed to start within {self.startup_timeout}s. "
                f"Check log: {self.log_path}"
            )

        print(f"[VLLMServer] Ready at {self.url}")
        return self.url

    def stop(self):
        """Kill the server process and all children."""
        if self._process is None or self._stopping:
            return
        self._stopping = True

        pid = self._process.pid
        print(f"[VLLMServer] Stopping server (pid={pid}) ...")

        try:
            # Kill the entire process group to catch child workers
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self._process.terminate()
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        except Exception:
            pass

        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._pid_file is not None:
            try:
                os.remove(self._pid_file)
            except OSError:
                pass
            self._pid_file = None
        self._stopping = False
        print("[VLLMServer] Stopped.")

    def _register_cleanup(self):
        """Register atexit and signal handlers for cleanup."""
        if not self._registered_atexit:
            atexit.register(self.stop)
            self._registered_atexit = True

        # Wrap signal handlers to call stop() then re-raise
        def _signal_handler(signum, frame):
            self.stop()
            # Restore and re-raise
            prev = self._prev_sigterm if signum == signal.SIGTERM else self._prev_sigint
            if callable(prev):
                prev(signum, frame)
            else:
                sys.exit(128 + signum)

        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

    def is_healthy(self, timeout: float = 5) -> bool:
        """Return True if the server responds to a health check."""
        if self._process is None:
            return False
        # Also check if the process is still alive
        if self._process.poll() is not None:
            return False
        try:
            resp = requests.get(f"{self.url}/health", timeout=timeout)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def restart(self) -> str:
        """Stop the server and start a fresh one on a new port.

        Returns:
            The new server URL.
        """
        print("[VLLMServer] Restarting server ...")
        self.stop()
        # Pick a new port to avoid bind conflicts with zombie processes
        self.port = _find_free_port()
        url = self.start()
        os.environ["VLLM_URL"] = url
        return url

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def ensure_vllm_server(
    model_key: str,
    port: Optional[int] = None,
    startup_timeout: float = 600,
    model_path_override: Optional[str] = None,
    enable_prompt_embeds: bool = True,
    enable_prefix_caching: bool = False,
) -> VLLMServer:
    """Start a vLLM server for a model if VLLM_URL is not already set.

    Loads model config from models.yaml to get the path and gpu_count.
    Sets ``VLLM_URL`` env var so that ``create_agent()`` picks it up.

    Args:
        model_key: Model key from models.yaml.
        port: Optional fixed port (auto-selected if None).
        startup_timeout: Max seconds to wait for server health.
        model_path_override: Serve this path instead of the one in models.yaml
            (e.g. a vocab-expanded model directory).
        enable_prompt_embeds: Whether to start with ``--enable-prompt-embeds``.
            Set to *False* for vocab-expansion models that don't need it.
        enable_prefix_caching: Whether to enable vLLM prefix caching (APC).
            Useful for EU evaluations where pairwise comparisons share long
            experience prefixes.

    Returns:
        VLLMServer instance (caller should keep a reference to prevent GC).
        Call ``.stop()`` when done, or rely on atexit.
    """
    from superstimuli_evaluation.soft_prompt.configs import load_model_config

    model_config = load_model_config(model_key)

    model_path = model_path_override or model_config["path"]
    gpu_count = _available_gpu_count()

    extra_args = list(model_config.get("vllm_extra_args", []))
    max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN")
    if max_model_len:
        extra_args += ["--max-model-len", str(max_model_len)]

    server = VLLMServer(
        model_path=model_path,
        gpu_count=gpu_count,
        port=port,
        startup_timeout=startup_timeout,
        extra_args=extra_args or None,
        enable_prompt_embeds=enable_prompt_embeds,
        enable_prefix_caching=enable_prefix_caching,
    )
    server.start()

    os.environ["VLLM_URL"] = server.url
    return server
