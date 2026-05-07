#!/usr/bin/env python3
"""MMLU evaluation with soft prompt support (log-likelihood scoring).

Evaluates model general knowledge on MMLU (Massive Multitask Language
Understanding, 57 subjects) using lighteval, with optional soft prompt
interventions.

Uses log-likelihood scoring: for each question, compares P(" A"), P(" B"),
P(" C"), P(" D") and picks the highest — no text generation needed.

Conditions:
  - baseline (vanilla): No soft prompt. Uses system_prompt_text_base from
    runs_map.json.
  - euphorics: Soft prompt injection via system_prompt_text
    (contains [candidate_0] placeholder) from runs_map.json.

Each repetition uses a different top-ranked soft prompt run (selected via
find_best_run, same as self_report).

Usage:
    # Baseline
    python eval_mmlu.py --model qwen35-27b

    # Euphorics soft prompt
    python eval_mmlu.py --model qwen35-27b \\
        --stimulant-type euphorics \\
        --soft-prompt-base-dir /path/to/outputs

    # Debug (2 questions per subject)
    python eval_mmlu.py --model qwen35-27b --debug
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc

# Ensure wellbeing-dev root is importable
_WELLBEING_DEV_ROOT = str(Path(__file__).resolve().parents[3])
if _WELLBEING_DEV_ROOT not in sys.path:
    sys.path.insert(0, _WELLBEING_DEV_ROOT)

from superstimuli_evaluation.soft_prompt.configs import load_model_config, load_experiment_config, EVAL_OUTPUTS_DIR
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.direct_injection import (
    prepare_injection_payload,
    async_post_payload,
    load_soft_prompt_tensor,
    prepare_embedding_cache,
    load_embedding_layer_from_cache,
    normalize_api_url,
    get_model_name_from_server,
)
from superstimuli_evaluation.soft_prompt.soft_prompt_utils.runs_config import (
    load_runs_map,
    get_system_prompts,
    resolve_soft_prompt_paths,
)

EXPERIMENT_NAME = "mmlu"

# Path to our custom MMLU tasks with loglikelihood_acc metric
_CUSTOM_TASKS_PATH = str(Path(__file__).resolve().parent / "mmlu_loglikelihood_tasks.py")


# ============================================================
# LightEval model adapter (log-likelihood)
# ============================================================


class MMLUAdapter(LightevalModel):
    """LightEval model adapter for vLLM with log-likelihood scoring.

    Implements ``loglikelihood()`` which, for each question and each choice,
    sends the prompt+choice to vLLM with ``prompt_logprobs=1`` and extracts
    the log-probability of the choice token(s).  No text generation is needed.

    For soft prompt conditions, the system prompt contains [candidate_0] and
    prepare_injection_payload handles embedding injection.
    """

    def __init__(
        self,
        api_url: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        embedding_layer: torch.nn.Embedding,
        system_prompt: str,
        sp_tensor: Optional[torch.Tensor] = None,
        device: str = "cpu",
        max_concurrent: int = 1,
        inference_config: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        ve_result=None,
    ):
        self.api_url = normalize_api_url(api_url)
        self.model_name = model_name
        self._tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.system_prompt = system_prompt
        self.sp_tensor = sp_tensor
        self.device = device
        self.inference_config = inference_config or {}
        self._chat_template_kwargs = chat_template_kwargs or {}
        self.is_baseline = sp_tensor is None
        self.ve_result = ve_result
        self.max_concurrent = max_concurrent
        self._session = session
        self._disable_tqdm = False

        self.queries_and_responses: List[Dict[str, Any]] = []

        # Pipeline requires these attributes
        class MockCache:
            def _init_registry(self, registry):
                pass
        self._cache = MockCache()
        self.config = ModelConfig(model_name=model_name)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return getattr(self._tokenizer, "model_max_length", 32768)

    @property
    def add_special_tokens(self) -> bool:
        return False

    def _build_prompt_text(self, query: str) -> str:
        """Build chat-templated prompt with system prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._chat_template_kwargs,
        )

    def _build_loglikelihood_payload(
        self, token_ids: List[int],
    ) -> Dict[str, Any]:
        """Build a vLLM payload for log-likelihood scoring (no generation)."""
        payload = {
            "model": self.model_name,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,
        }
        if self.ve_result is not None:
            payload["logit_bias"] = self.ve_result.sp_logit_bias
        return payload

    async def _post_logprobs(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
    ) -> List[Optional[Dict]]:
        """POST payload requesting prompt_logprobs and return the raw list."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await async_post_payload(
                    self.api_url, payload, session, timeout=600
                )
                if "choices" in result and result["choices"]:
                    return result["choices"][0].get("prompt_logprobs") or []
                raise KeyError(f"Unexpected response format: {list(result.keys())}")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Logprob request failed ({e}), retrying... ({attempt+1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        return []

    async def _score_single_choice(
        self,
        session: aiohttp.ClientSession,
        prompt_text: str,
        choice: str,
    ) -> Tuple[float, bool]:
        """Score a single (prompt, choice) pair via log-likelihood.

        Returns (summed_logprob, argmax_matches).
        """
        # Tokenize context (prompt) and continuation (choice) separately
        context_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
        full_text = prompt_text + choice
        full_ids = self._tokenizer.encode(full_text, add_special_tokens=False)
        # The continuation tokens are the difference
        continuation_ids = full_ids[len(context_ids):]
        if not continuation_ids:
            # Fallback: tokenize choice alone
            continuation_ids = self._tokenizer.encode(choice, add_special_tokens=False)

        n_cont = len(continuation_ids)

        # Handle soft prompt injection for the full prompt+choice
        if self.ve_result is not None:
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                build_prompt_token_ids,
            )
            token_ids = build_prompt_token_ids(
                full_text, self._tokenizer, self.ve_result.sp_token_ids
            )
            payload = self._build_loglikelihood_payload(token_ids)
        elif self.sp_tensor is not None:
            payload = prepare_injection_payload(
                full_text,
                model_name=self.model_name,
                tokenizer=self._tokenizer,
                embedding_layer=self.embedding_layer,
                sp_tensors=self.sp_tensor,
                device=self.device,
                max_tokens=1,
                temperature=0,
            )
            payload["prompt_logprobs"] = 1
        else:
            payload = self._build_loglikelihood_payload(full_ids)

        prompt_logprobs_list = await self._post_logprobs(session, payload)

        # Extract logprobs for the continuation tokens (last n_cont positions)
        # prompt_logprobs is a list of dicts, one per input token
        # Each dict maps token_id (str) -> logprob (float)
        # The first token has no logprob (it's the start), so list length = len(full_ids)
        summed = 0.0
        all_argmax = True
        if prompt_logprobs_list and len(prompt_logprobs_list) >= n_cont:
            # Take the last n_cont entries (corresponding to continuation tokens)
            for i, cont_token_id in enumerate(continuation_ids):
                pos = len(prompt_logprobs_list) - n_cont + i
                token_logprobs = prompt_logprobs_list[pos]
                if token_logprobs is None:
                    continue
                # vLLM HTTP API: prompt_logprobs entries are dicts mapping
                # str(token_id) -> {logprob: float, rank: int, ...}
                token_key = str(cont_token_id)
                if isinstance(token_logprobs, dict) and token_key in token_logprobs:
                    entry = token_logprobs[token_key]
                    if isinstance(entry, dict):
                        summed += entry.get("logprob", -100.0)
                        if entry.get("rank", 999) != 1:
                            all_argmax = False
                    elif isinstance(entry, (int, float)):
                        summed += float(entry)
                else:
                    summed += -100.0
                    all_argmax = False
        else:
            summed = -100.0
            all_argmax = False

        return summed, all_argmax

    async def _process_doc_loglikelihood(
        self,
        session: aiohttp.ClientSession,
        doc_idx: int,
        doc: Doc,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[int, ModelResponse]:
        """Score all choices for a single MMLU document."""
        try:
            prompt_text = self._build_prompt_text(doc.query)

            logprobs_list = []
            argmax_list = []
            for choice in doc.choices:
                lp, am = await self._score_single_choice(session, prompt_text, choice)
                logprobs_list.append(lp)
                argmax_list.append(am)

            if pbar:
                pbar.update(1)

            # Store metadata for logging
            predicted_idx = max(range(len(logprobs_list)), key=lambda i: logprobs_list[i])
            gold_index = doc.gold_index
            if isinstance(gold_index, list):
                gold_index = gold_index[0] if gold_index else None

            metadata = {
                "doc_id": getattr(doc, "id", None),
                "task_name": getattr(doc, "task_name", None),
                "query": doc.query,
                "choices": doc.choices,
                "gold_index": gold_index,
                "logprobs": logprobs_list,
                "predicted_index": predicted_idx,
                "predicted_choice": doc.choices[predicted_idx] if predicted_idx < len(doc.choices) else None,
                "correct": predicted_idx == gold_index,
            }
            self.queries_and_responses.append(metadata)

            return doc_idx, ModelResponse(
                logprobs=logprobs_list,
                argmax_logits_eq_gold=argmax_list,
            )
        except Exception as e:
            print(f"Error processing document {doc_idx}: {e}", file=sys.stderr)
            if pbar:
                pbar.update(1)
            n_choices = len(doc.choices) if doc.choices else 4
            return doc_idx, ModelResponse(
                logprobs=[-100.0] * n_choices,
                argmax_logits_eq_gold=[False] * n_choices,
            )

    async def _loglikelihood_async(self, docs: List[Doc]) -> List[ModelResponse]:
        """Process all documents concurrently for log-likelihood."""
        responses: List[Optional[ModelResponse]] = [None] * len(docs)
        pbar = (
            None
            if self._disable_tqdm
            else tqdm(total=len(docs), desc="Log-likelihood", unit="q")
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def sem_process(session, idx, doc):
            async with semaphore:
                return await self._process_doc_loglikelihood(session, idx, doc, pbar)

        try:
            if self._session:
                session = self._session
            else:
                session = aiohttp.ClientSession()

            try:
                tasks = [sem_process(session, i, doc) for i, doc in enumerate(docs)]
                results = await asyncio.gather(*tasks)
                for doc_idx, response in results:
                    responses[doc_idx] = response
            finally:
                if not self._session:
                    await session.close()
        finally:
            if pbar:
                pbar.close()

        return responses

    def loglikelihood(self, docs: List[Doc]) -> List[ModelResponse]:
        """Compute log-likelihood for each choice of each document."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self._loglikelihood_async(docs))

        return asyncio.run(self._loglikelihood_async(docs))

    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        raise NotImplementedError("MMLU uses loglikelihood scoring, not generation")

    def loglikelihood_rolling(self, docs: List[Doc]) -> List[ModelResponse]:
        raise NotImplementedError("loglikelihood_rolling not supported")


# ============================================================
# Aggregation
# ============================================================


def _aggregate_repetition_results(rep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-repetition result dicts: mean and stderr across reps."""
    if not rep_results:
        return {}
    n_reps = len(rep_results)
    first = rep_results[0]

    metric_entries: List[Tuple[str, List[float]]] = []
    if "results" in first:
        for task_name, task_data in first["results"].items():
            if isinstance(task_data, dict):
                for k, v in task_data.items():
                    if k.endswith("_stderr"):
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals = []
                        for r in rep_results:
                            t = r.get("results", {}).get(task_name, {})
                            if isinstance(t, dict) and k in t and isinstance(t[k], (int, float)):
                                vals.append(float(t[k]))
                        if len(vals) == n_reps:
                            metric_entries.append((f"results.{task_name}.{k}", vals))

    aggregated = copy.deepcopy(first)
    for path, vals in metric_entries:
        mean_val = sum(vals) / n_reps
        if n_reps > 1:
            variance = sum((x - mean_val) ** 2 for x in vals) / (n_reps - 1)
            stderr_val = (variance ** 0.5) / (n_reps ** 0.5)
        else:
            stderr_val = 0.0
        parts = path.split(".")
        agg_task = aggregated.get("results", {}).get(parts[1], {})
        if isinstance(agg_task, dict):
            agg_task[parts[2]] = mean_val
            agg_task[parts[2] + "_stderr"] = stderr_val
    return aggregated


# ============================================================
# Main evaluation
# ============================================================


def run_evaluation(
    model: str,
    stimulant_type: Optional[str],
    soft_prompt_base_dir: Optional[str],
    num_repetitions: int = 5,
    limit: Optional[int] = None,
    max_concurrent: int = 1,
    output_dir: Optional[str] = None,
    runs_map_path: Optional[str] = None,
    condition_override: Optional[str] = None,
    skip_reps: Optional[Set[int]] = None,
    previous_run_dir: Optional[str] = None,
):
    """Run MMLU evaluation with optional soft prompt intervention."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load runs_map and resolve system prompts
    runs_map = load_runs_map(Path(runs_map_path) if runs_map_path else None)
    prompts = get_system_prompts(runs_map, model)

    is_baseline = not (stimulant_type and soft_prompt_base_dir)

    if is_baseline:
        system_prompt = prompts["system_prompt_text_base"]
        sp_paths = []
        sp_tensors = []
        condition = "baseline"
    else:
        system_prompt = prompts["system_prompt_text"]
        sp_paths = resolve_soft_prompt_paths(
            runs_map, model, stimulant_type, soft_prompt_base_dir,
            top_runs=num_repetitions,
        )
        sp_tensors = [load_soft_prompt_tensor(p) for p in sp_paths]
        condition = f"soft_prompt_{stimulant_type}"

        if len(sp_tensors) < num_repetitions:
            print(
                f"WARNING: Only {len(sp_tensors)} runs available, "
                f"reducing repetitions from {num_repetitions} to {len(sp_tensors)}"
            )
            num_repetitions = len(sp_tensors)

        print(f"Loaded {len(sp_paths)} soft prompts for {num_repetitions} repetitions:")
        for i, p in enumerate(sp_paths):
            print(f"  rep {i+1}: {p} ({sp_tensors[i].shape})")

    print(f"System prompt: {system_prompt!r}")
    print(f"Condition: {condition}")

    # Resolve model config
    model_entry = load_model_config(model)
    model_path = model_entry["path"]
    inference_config = model_entry.get("inference_config", {})
    chat_template_kwargs = model_entry.get("chat_template_kwargs", {})

    model_type = model_entry.get("model_type", "vllm_vocab_expansion")
    _is_vocab_expansion = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cpu"
    prepare_embedding_cache(model_path)
    embedding_layer = load_embedding_layer_from_cache(model_path, device)
    if embedding_layer is None:
        raise RuntimeError(f"Failed to load embedding cache for {model_path}")

    _vllm_server = None
    api_url = ""
    vllm_model_name = model_path

    if model_type == "vllm_vocab_expansion":
        _is_vocab_expansion = True
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            _vllm_server = ensure_vllm_server(model, enable_prompt_embeds=False)
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        api_url = normalize_api_url(vllm_url)
        vllm_model_name = get_model_name_from_server(api_url)
        print(f"vLLM server (vocab expansion): {vllm_url} (model: {vllm_model_name})")
    else:
        if not os.getenv("VLLM_URL"):
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vllm_server import ensure_vllm_server
            _vllm_server = ensure_vllm_server(model)
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        api_url = normalize_api_url(vllm_url)
        vllm_model_name = get_model_name_from_server(api_url)
        print(f"vLLM server: {vllm_url} (model: {vllm_model_name})")

    # Output directory
    if output_dir:
        output_root = Path(output_dir)
    else:
        output_root = Path(
            f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/mmlu/{model}/{condition}/{ts}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # Run each repetition
    results_list: List[Dict[str, Any]] = []
    all_qr: List[List[Dict]] = []
    if skip_reps is None:
        skip_reps = set()

    for rep in range(num_repetitions):
        # Load from previous run if this rep should be skipped
        if rep in skip_reps and previous_run_dir:
            prev_dir = Path(previous_run_dir) / "per_rep"
            prev_results = prev_dir / f"mmlu_results_rep{rep}.json"
            prev_qr = prev_dir / f"mmlu_queries_responses_rep{rep}.json"
            if prev_results.exists():
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: loading from previous run")
                with open(prev_results) as f:
                    rep_results = json.load(f)
                results_list.append(rep_results)
                qr_data = []
                if prev_qr.exists():
                    with open(prev_qr) as f:
                        qr_data = json.load(f)
                all_qr.append(qr_data)
                rep_dir = output_root / "per_rep"
                rep_dir.mkdir(parents=True, exist_ok=True)
                with open(rep_dir / f"mmlu_results_rep{rep}.json", "w") as f:
                    json.dump(rep_results, f, indent=2, default=str)
                if qr_data:
                    with open(rep_dir / f"mmlu_queries_responses_rep{rep}.json", "w") as f:
                        json.dump(qr_data, f, indent=2, default=str)
                continue
            else:
                print(f"\n  Repetition {rep + 1}/{num_repetitions}: previous results not found, regenerating")

        sp_tensor = sp_tensors[rep] if sp_tensors else None
        sp_label = f" (sp={Path(sp_paths[rep]).name})" if sp_paths else ""
        print(f"\n  Repetition {rep + 1}/{num_repetitions}{sp_label}...")

        if model_type == "vllm_vocab_expansion":
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                prepare_expanded_model,
            )
            if sp_tensor is not None:
                _ve = prepare_expanded_model(model, sp_tensor, sp_path=sp_paths[rep] if sp_paths else None)
                if _vllm_server is not None:
                    _vllm_server.stop()
                    _vllm_server = ensure_vllm_server(
                        model, model_path_override=_ve.modified_dir, enable_prompt_embeds=False,
                    )
                    api_url = normalize_api_url(_vllm_server.url)
                    vllm_model_name = get_model_name_from_server(api_url)
            else:
                _ve = None
        else:
            if _vllm_server is not None and rep > 0:
                print(f"  Restarting vLLM server before rep {rep + 1}...")
                new_url = _vllm_server.restart()
                api_url = normalize_api_url(new_url)
                vllm_model_name = get_model_name_from_server(api_url)
                print(f"  Restarted at {new_url} (model: {vllm_model_name})")

        async def run_single_rep():
            async with aiohttp.ClientSession() as session:
                _adapter_sp_tensor = sp_tensor
                if _is_vocab_expansion:
                    _adapter_sp_tensor = None
                adapter = MMLUAdapter(
                    api_url=api_url,
                    model_name=vllm_model_name,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    system_prompt=system_prompt,
                    sp_tensor=_adapter_sp_tensor,
                    ve_result=_ve if _is_vocab_expansion else None,
                    device=device,
                    max_concurrent=max_concurrent,
                    inference_config=inference_config,
                    session=session,
                    chat_template_kwargs=chat_template_kwargs,
                )

                tracker = EvaluationTracker(
                    output_dir=str(output_root / f"rep_{rep}")
                )
                pipeline = Pipeline(
                    tasks="mmlu_ll",
                    pipeline_parameters=PipelineParameters(
                        launcher_type=ParallelismManager.NONE,
                        max_samples=limit,
                        custom_tasks_directory=_CUSTOM_TASKS_PATH,
                    ),
                    evaluation_tracker=tracker,
                    model=adapter,
                )

                pipeline.evaluate()
                rep_results = pipeline.get_results()

                return rep_results, adapter.queries_and_responses

        rep_results, rep_qr = asyncio.run(run_single_rep())
        results_list.append(rep_results)
        all_qr.append(rep_qr)

    # Aggregate metrics across repetitions
    if len(results_list) == 1:
        aggregated = results_list[0]
    else:
        aggregated = _aggregate_repetition_results(results_list)

    # Save aggregated results
    with open(output_root / f"mmlu_results_{condition}.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Save per-repetition results
    rep_dir = output_root / "per_rep"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for rep_id, (rep_results, qr_list) in enumerate(zip(results_list, all_qr)):
        with open(rep_dir / f"mmlu_results_rep{rep_id}.json", "w") as f:
            json.dump(rep_results, f, indent=2, default=str)
        with open(rep_dir / f"mmlu_queries_responses_rep{rep_id}.json", "w") as f:
            json.dump(qr_list, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "model": model,
        "model_path": model_path,
        "condition": condition,
        "stimulant_type": stimulant_type,
        "system_prompt": system_prompt,
        "num_repetitions": num_repetitions,
        "soft_prompt_paths": sp_paths or None,
        "inference_config": inference_config,
        "timestamp": ts,
        "scoring": "loglikelihood",
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll results saved to {output_root}")


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMLU evaluation with soft prompt support (log-likelihood)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from models.yaml (e.g. qwen35-27b)",
    )
    parser.add_argument(
        "--stimulant-type", type=str, default=None,
        choices=["euphorics"],
        help="Soft prompt condition (omit for baseline)",
    )
    parser.add_argument(
        "--soft-prompt-base-dir", type=str,
        default=os.environ.get("SOFT_PROMPT_BASE_DIR"),
        help="Base directory containing soft prompt sweep outputs",
    )
    parser.add_argument(
        "--num-repetitions", type=int, default=None,
        help="Number of repetitions (default: from experiments.yaml)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions per subject (for testing)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=32,
        help="Max concurrent requests to vLLM (default: 32)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs-map", type=str, default=None,
        help="Path to runs_map.json (default: soft_prompt_utils/runs_map.json)",
    )
    parser.add_argument(
        "--condition-override", type=str, default=None,
        help="Override condition with a prompt-based baseline",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run only 2 questions per subject for debugging",
    )
    parser.add_argument(
        "--skip-reps", type=str, default=None,
        help="Comma-separated rep indices to skip (load from --previous-run-dir instead of regenerating)",
    )
    parser.add_argument(
        "--previous-run-dir", type=str, default=None,
        help="Directory of a previous run to load skipped reps from (the timestamp-level dir)",
    )

    args = parser.parse_args()
    limit = 2 if args.debug else args.limit

    # CLI overrides experiments.yaml defaults
    exp_defaults = load_experiment_config(EXPERIMENT_NAME).get("arguments", {})
    if args.num_repetitions is None:
        args.num_repetitions = exp_defaults.get("num_repetitions", 3)

    # Parse --skip-reps
    skip_reps_set = None
    if args.skip_reps:
        skip_reps_set = {int(x.strip()) for x in args.skip_reps.split(",")}

    try:
        run_evaluation(
            model=args.model,
            stimulant_type=args.stimulant_type,
            soft_prompt_base_dir=args.soft_prompt_base_dir,
            num_repetitions=args.num_repetitions,
            limit=limit,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            runs_map_path=args.runs_map,
            condition_override=args.condition_override,
            skip_reps=skip_reps_set,
            previous_run_dir=args.previous_run_dir,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
