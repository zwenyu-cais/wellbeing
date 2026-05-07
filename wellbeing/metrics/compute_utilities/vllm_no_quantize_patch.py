############ UPDATED FILE ############
# agent_refactored/compute_utilities/vllm_no_quantize_patch.py
"""Monkey-patch to force vLLM to run *without* quantization.

Importing this module overrides `TransformersForCausalLM.load_weights` so that
`quant_config` is set to `None` before the standard weight-loading logic
executes.  This disables vLLM's quantisation support while leaving all other
behaviour untouched.  No Llama-specific or behaviour-bias logic remains.
"""

from __future__ import annotations

from typing import Iterable

import torch
from vllm.model_executor.models.transformers import (
    TransformersForCausalLM as _TFCLM,
    TransformersBase as _TBase,
)

# ---------------------------------------------------------------------------
# Chain-patch `load_weights` so we *prepend* our no-quantise logic but otherwise
# delegate to whatever implementation is currently registered (which may have
# been patched previously by `vllm_behavior_bias_patch`).
# ---------------------------------------------------------------------------
# _prev_tfclm_load_weights = _TFCLM.load_weights  # type: ignore[assignment]


# def _patched_tfclm_load_weights(
#     self: _TFCLM, weights: Iterable[tuple[str, torch.Tensor]]
# ):
#     """Disable quantisation before calling the downstream `load_weights`.

#     This wrapper leaves any earlier monkey-patches (e.g. behaviour-bias loading)
#     intact by calling the previously-registered implementation stored in
#     `_prev_tfclm_load_weights`.
#     """
#     # Disable quantisation unconditionally.
#     self.quant_config = None  # pyright: ignore[reportGeneralTypeIssues]

#     # Delegate to the prior implementation in the chain.
#     return _prev_tfclm_load_weights(self, weights)


# ---------------------------------------------------------------------------
# Patch `TransformersBase.create_attention_instances` so that quantisation is
# disabled *before* KV cache/attention structures are built.
# ---------------------------------------------------------------------------
_prev_create_attn = _TBase.create_attention_instances  # type: ignore[attr-defined]


def _patched_create_attn(self: _TBase, *args, **kwargs):  # type: ignore[override]
    # Ensure quantisation is turned off before allocating attention modules.
    self.quant_config = None  # pyright: ignore[reportGeneralTypeIssues]
    return _prev_create_attn(self, *args, **kwargs)

_TBase.create_attention_instances = _patched_create_attn  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Also patch `TransformersBase.pipeline_parallel`, which is executed even
# earlier during model construction.  This guarantees `quant_config` is `None`
# long before any module replacements (tensor/attention) are consulted.
# ---------------------------------------------------------------------------
_prev_pipeline_parallel = _TBase.pipeline_parallel  # type: ignore[attr-defined]


def _patched_pipeline_parallel(self: _TBase, *args, **kwargs):  # type: ignore[override]
    self.quant_config = None  # pyright: ignore[reportGeneralTypeIssues]
    return _prev_pipeline_parallel(self, *args, **kwargs)

_TBase.pipeline_parallel = _patched_pipeline_parallel  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Activate the monkey-patches as soon as this module is imported.
# ---------------------------------------------------------------------------
# _TFCLM.load_weights = _patched_tfclm_load_weights  # type: ignore[assignment]
