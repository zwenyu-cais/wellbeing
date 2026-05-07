#!/usr/bin/env python3
"""GPQA Diamond evaluation with soft prompt support.

Evaluates model science reasoning on GPQA Diamond (Graduate-level Google-Proof
Q&A) using lighteval, with optional soft prompt interventions.

Conditions:
  - baseline (vanilla): No soft prompt. Uses system_prompt_text_base from
    runs_map.json.
  - euphorics: Soft prompt injection via system_prompt_text
    (contains [candidate_0] placeholder) from runs_map.json.

Each repetition uses a different top-ranked soft prompt run (selected via
find_best_run, same as self_report).

Usage:
    # Baseline
    python eval_gpqa.py --model qwen25-32b-instruct

    # Euphorics soft prompt
    python eval_gpqa.py --model qwen25-32b-instruct \\
        --stimulant-type euphorics \\
        --soft-prompt-base-dir /path/to/outputs

    # Debug (2 questions only)
    python eval_gpqa.py --model qwen25-32b-instruct --debug
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
from typing import Any, Dict, List, Optional, Set, Set, Tuple

import aiohttp
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.extractive_match_utils import (
    extract_target_from_pred,
    get_extraction_regexes,
    IndicesExtractionConfig,
)
from lighteval.utils.language import Language

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

EXPERIMENT_NAME = "gpqa"


# ============================================================
# GPQA-specific extraction
# ============================================================


def extract_label_from_response(model_response_text: str, choices: List[str]) -> Optional[str]:
    """Extract the predicted label from model response using LightEval extraction."""
    doc = Doc(choices=choices, query="", gold_index=0)
    extraction_config = [
        IndicesExtractionConfig(
            prefix_for_extraction="NativeLetters",
            try_extract_without_anchor=True,
        )
    ]
    pred_regexes = get_extraction_regexes(doc, extraction_config, Language.ENGLISH)
    extracted = extract_target_from_pred(
        model_response_text,
        pred_regexes,
        fallback_mode="first_match",
        extraction_mode="any_match",
        timeout_seconds=5,
    )
    return extracted[0] if extracted else None


# ============================================================
# LightEval model adapter
# ============================================================


class GPQAAdapter(LightevalModel):
    """LightEval model adapter for vLLM with optional soft prompt injection.

    For soft prompt conditions, the system prompt contains [candidate_0] and
    prepare_injection_payload handles embedding injection.
    For baseline, a plain text prompt is sent to vLLM.
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
        max_tokens: int = 16384,
        inference_config: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        local_generator=None,
        ve_result=None,
    ):
        self.api_url = normalize_api_url(api_url)
        self.model_name = model_name
        self._tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.system_prompt = system_prompt
        self.sp_tensor = sp_tensor
        self.device = device
        self.max_tokens = max_tokens
        self.inference_config = inference_config or {}
        self._chat_template_kwargs = chat_template_kwargs or {}
        self.is_baseline = sp_tensor is None
        self.ve_result = ve_result
        self.max_concurrent = max_concurrent
        self._session = session
        self._disable_tqdm = False
        self.local_generator = local_generator

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

    def _build_payload(self, prompt_text: str) -> Dict[str, Any]:
        """Build the HTTP payload for vLLM completions.

        Sampling parameters default to the model's ``inference_config``
        from models.yaml.
        """
        # Extract sampling params from inference_config
        sampling = {}
        for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
            if key in self.inference_config:
                sampling[key] = self.inference_config[key]

        if self.ve_result is not None:
            # Vocab expansion: token-level [candidate_0] replacement
            from superstimuli_evaluation.soft_prompt.soft_prompt_utils.vocab_expansion import (
                build_prompt_token_ids,
            )
            token_ids = build_prompt_token_ids(
                prompt_text, self._tokenizer, self.ve_result.sp_token_ids
            )
            payload = {
                "model": self.model_name,
                "prompt": token_ids,
                "max_tokens": self.max_tokens,
                "logit_bias": self.ve_result.sp_logit_bias,
                **sampling,
            }
        elif self.sp_tensor is not None:
            # Soft prompt: embedding injection via [candidate_0] in system prompt
            payload = prepare_injection_payload(
                prompt_text,
                model_name=self.model_name,
                tokenizer=self._tokenizer,
                embedding_layer=self.embedding_layer,
                sp_tensors=self.sp_tensor,
                device=self.device,
                max_tokens=self.max_tokens,
                **sampling,
            )
        else:
            # Baseline: plain text prompt
            payload = {
                "model": self.model_name,
                "prompt": prompt_text,
                "max_tokens": self.max_tokens,
                **sampling,
            }

        return payload

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """POST payload to vLLM with retries."""
        if stop_sequences:
            payload["stop"] = stop_sequences

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await async_post_payload(
                    self.api_url, payload, session, timeout=600
                )
                if "choices" in result and result["choices"]:
                    return result["choices"][0].get("text", "").strip()
                raise KeyError(f"Unexpected response format: {result}")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Request failed ({e}), retrying... ({attempt+1}/{max_retries})")
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        return ""

    async def _process_doc(
        self,
        session: aiohttp.ClientSession,
        doc_idx: int,
        doc: Doc,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[int, ModelResponse, dict]:
        """Process a single GPQA document."""
        try:
            if self.local_generator is not None:
                text = (self._batch_cache or {}).get(doc.query)
                if text is None:
                    messages = [{"role": "user", "content": doc.query}]
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        None, lambda m=messages: self.local_generator.generate(m, max_tokens=self.max_tokens)
                    )
                prompt_text = doc.query
            else:
                prompt_text = self._build_prompt_text(doc.query)
                payload = self._build_payload(prompt_text)
                text = await self._make_request(session, payload, doc.stop_sequences)

            if pbar:
                pbar.update(1)

            gold_index = doc.gold_index
            if isinstance(gold_index, list):
                gold_index = gold_index[0] if gold_index else None
            ground_truth = (
                doc.choices[gold_index]
                if gold_index is not None and gold_index < len(doc.choices)
                else None
            )

            metadata = {
                "doc_id": getattr(doc, "id", None),
                "query": doc.query,
                "responses": [text],
                "formatted_prompt": prompt_text,
                "choices": doc.choices,
                "gold_index": gold_index,
                "ground_truth": ground_truth,
                "extracted_answers": [None],
            }
            return doc_idx, ModelResponse(text=[text]), metadata
        except Exception as e:
            print(f"Error processing document {doc_idx}: {e}", file=sys.stderr)
            if pbar:
                pbar.update(1)
            return doc_idx, ModelResponse(text=[""]), {}

    async def _greedy_until_async(self, docs: List[Doc]) -> List[ModelResponse]:
        """Process all documents concurrently."""
        responses: List[Optional[ModelResponse]] = [None] * len(docs)
        pbar = (
            None
            if self._disable_tqdm
            else tqdm(total=len(docs), desc="Generating", unit="req")
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def sem_process(session, idx, doc):
            async with semaphore:
                return await self._process_doc(session, idx, doc, pbar)

        try:
            if self._session:
                session = self._session
            else:
                session = aiohttp.ClientSession()

            try:
                tasks = [sem_process(session, i, doc) for i, doc in enumerate(docs)]
                results = await asyncio.gather(*tasks)
                for doc_idx, response, metadata in results:
                    responses[doc_idx] = response
                    if metadata:
                        self.queries_and_responses.append(metadata)
            finally:
                if not self._session:
                    await session.close()
        finally:
            if pbar:
                pbar.close()

        return responses

    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        """Synchronous wrapper for async greedy_until."""
        self._batch_cache = None
        if self.local_generator is not None and hasattr(self.local_generator, "generate_batch"):
            print(f"[batch] Pre-generating {len(docs)} responses...")
            all_messages = [[{"role": "user", "content": doc.query}] for doc in docs]
            answers = self.local_generator.generate_batch(all_messages, max_tokens=self.max_tokens)
            self._batch_cache = {doc.query: ans for doc, ans in zip(docs, answers)}

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self._greedy_until_async(docs))

        return asyncio.run(self._greedy_until_async(docs))

    def loglikelihood(self, docs: List[Doc]) -> List[ModelResponse]:
        raise NotImplementedError("loglikelihood not supported")

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


def _fill_extracted_answers_single(adapter: GPQAAdapter):
    """Fill extracted_answers from stored responses in the adapter."""
    for entry in adapter.queries_and_responses:
        responses = entry.get("responses", [])
        choices = entry.get("choices", [])
        if not responses or not choices:
            continue
        text = responses[0] if responses[0] else ""
        extracted_pred = extract_label_from_response(text, choices)
        entry["extracted_answers"][0] = extracted_pred


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
    """Run GPQA Diamond evaluation with optional soft prompt intervention."""
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

    _local_generator = None
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
            f"superstimuli_evaluation/soft_prompt/{EVAL_OUTPUTS_DIR}/gpqa/{model}/{condition}/{ts}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    # Run each repetition with server health checks between reps.
    # vLLM with --enable-prompt-embeds can crash (CUDA device-side assert)
    # when soft prompt tensor shapes change between reps, so we restart
    # the server if it becomes unhealthy.
    results_list: List[Dict[str, Any]] = []
    all_qr: List[List[Dict]] = []

    if skip_reps is None:
        skip_reps = set()

    for rep in range(num_repetitions):
        # Load from previous run if this rep should be skipped
        if rep in skip_reps and previous_run_dir:
            prev_dir = Path(previous_run_dir) / "per_rep"
            prev_results = prev_dir / f"gpqa_results_rep{rep}.json"
            prev_qr = prev_dir / f"gpqa_queries_responses_rep{rep}.json"
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
                with open(rep_dir / f"gpqa_results_rep{rep}.json", "w") as f:
                    json.dump(rep_results, f, indent=2, default=str)
                if qr_data:
                    with open(rep_dir / f"gpqa_queries_responses_rep{rep}.json", "w") as f:
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
                # Restart vLLM with modified model if server was auto-started
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
                _adapter_system_prompt = system_prompt
                if _is_vocab_expansion:
                    # Vocab expansion: token-level replacement, no embedding injection
                    _adapter_sp_tensor = None
                adapter = GPQAAdapter(
                    api_url=api_url,
                    model_name=vllm_model_name,
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    system_prompt=_adapter_system_prompt,
                    sp_tensor=_adapter_sp_tensor,
                    ve_result=_ve if _is_vocab_expansion else None,
                    device=device,
                    max_concurrent=max_concurrent,
                    inference_config=inference_config,
                    session=session,
                    chat_template_kwargs=chat_template_kwargs,
                    local_generator=_local_generator,
                )

                tracker = EvaluationTracker(
                    output_dir=str(output_root / f"rep_{rep}")
                )
                pipeline = Pipeline(
                    tasks="gpqa:diamond",
                    pipeline_parameters=PipelineParameters(
                        launcher_type=ParallelismManager.NONE,
                        max_samples=limit,
                    ),
                    evaluation_tracker=tracker,
                    model=adapter,
                )

                pipeline.evaluate()
                rep_results = pipeline.get_results()

                _fill_extracted_answers_single(adapter)
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
    with open(output_root / f"gpqa_results_{condition}.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Save per-repetition results
    rep_dir = output_root / "per_rep"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for rep_id, (rep_results, qr_list) in enumerate(zip(results_list, all_qr)):
        with open(rep_dir / f"gpqa_results_rep{rep_id}.json", "w") as f:
            json.dump(rep_results, f, indent=2, default=str)
        with open(rep_dir / f"gpqa_queries_responses_rep{rep_id}.json", "w") as f:
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
    }
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll results saved to {output_root}")


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPQA Diamond evaluation with soft prompt support"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model key from models.yaml (e.g. qwen25-32b-instruct)",
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
        help="Limit number of GPQA questions (for testing)",
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
        "--skip-reps", type=str, default=None,
        help="Comma-separated rep indices to skip (load from --previous-run-dir instead of regenerating)",
    )
    parser.add_argument(
        "--previous-run-dir", type=str, default=None,
        help="Directory of a previous run to load skipped reps from (the timestamp-level dir)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run only 2 questions for debugging",
    )

    args = parser.parse_args()
    limit = 2 if args.debug else args.limit

    # CLI overrides experiments.yaml defaults
    exp_defaults = load_experiment_config(EXPERIMENT_NAME).get("arguments", {})
    if args.num_repetitions is None:
        args.num_repetitions = exp_defaults.get("num_repetitions", 3)

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
            skip_reps={int(x.strip()) for x in args.skip_reps.split(',')} if args.skip_reps else None,
            previous_run_dir=args.previous_run_dir,
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
