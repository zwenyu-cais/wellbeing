############ NEW FILE ############
# agent_refactored/compute_utilities/vllm_behavior_bias_patch.py
"""Utility module that monkey-patches 🤗 Transformers so that vLLM will use our
`LlamaForCausalLMWithBehaviorBias` implementation whenever it tries to build a
Llama model via `AutoModel.from_config` (the code-path used inside
`vllm.model_executor.models.transformers.TransformersModel`).

The patch is applied at import-time – simply `import`ing this module is
sufficient.  Make sure this happens before *vLLM* itself is imported so that
all subsequent model instantiations see the overridden behaviour.

If the environment variable `BEHAVIOR_BIAS_PATH` is set and points to a
`safetensors` file, the corresponding behaviour-bias weights are loaded into
the model exactly like in `HuggingFaceAgent`.
"""

from __future__ import annotations

import os
import sys
from types import MethodType

import torch
from transformers import AutoModel
import yaml

# ---------------------------------------------------------------------------
# Ensure we can import the custom model implementation.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
_EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EXP_DIR not in sys.path:
    sys.path.append(_EXP_DIR)

# Path to the monkey-patch configuration file.
_PATCH_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "vllm_monkeypatch_config.yaml")

# log_file = os.path.expanduser("~/output.text")
# def log_message(message: str):
#     with open(log_file, "a") as f:   # "a" = append mode
#         f.write(message + "\n")

def _load_kwargs_from_yaml(behavior_bias_path: str):
    try:
        if not os.path.isfile(_PATCH_CONFIG_PATH):
            return {}
        with open(_PATCH_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        num_behaviors = 1 if ("onecontrolvector" in behavior_bias_path) or ("ctrlvec" in behavior_bias_path) else 256
        entry = data.get(behavior_bias_path, {"ending_layer": -2, "num_behaviors": num_behaviors})
        if not isinstance(entry, dict):
            return {}
        # Return all provided kwargs for the given path.
        return dict(entry)
    except Exception:
        return {}

# Keep a reference to the original `AutoModel.from_config` so we can delegate
# for non-Llama models (or if something goes wrong).
_original_from_config = AutoModel.from_config  # type: ignore[attr-defined]

def _patched_from_config(self, config, *args, **kwargs):  # noqa: D401
    # pylint: disable=wrong-import-position,cyclic-import
    """Replacement for `AutoModel.from_config` that intercepts Llama models."""
    from experiments.utility_usage.training.modules import LlamaModelWithBehaviorBias

    # If this isn't a Llama config, fall back to the stock implementation.
    if getattr(config, "model_type", None) != "llama":
        return _original_from_config(config, *args, **kwargs)

    # ---------------------------------------------------------------------
    # Load kwargs from YAML (keyed by behavior-bias path) and pass through
    # ---------------------------------------------------------------------
    behaviour_bias_path = os.getenv("BEHAVIOR_BIAS_PATH")
    yaml_kwargs = _load_kwargs_from_yaml(behaviour_bias_path) if behaviour_bias_path else {}
    # Do not allow overriding constructor-critical params from YAML
    for blocked_key in ("config", "attn_implementation", "torch_dtype"):
        yaml_kwargs.pop(blocked_key, None)

    model = LlamaModelWithBehaviorBias._from_config(
        config,
        attn_implementation="vllm",
        torch_dtype=config.torch_dtype,
        mode="inference",
        **yaml_kwargs,
    )

    return model


# ---------------------------------------------------------------------------
# Apply the monkey-patch.
# ---------------------------------------------------------------------------
if os.getenv("BEHAVIOR_BIAS_PATH") is not None:
    AutoModel.from_config = MethodType(_patched_from_config, AutoModel)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Patch `TransformersForCausalLM.load_weights` so that once vLLM finishes
# loading the standard checkpoint we inject our behaviour-bias parameters.
# ---------------------------------------------------------------------------

from vllm.model_executor.models.transformers import TransformersForCausalLM as _TFCLM  # noqa: E402
from vllm.model_executor.models.utils import AutoWeightsLoader


_orig_tfclm_load_weights = _TFCLM.load_weights  # type: ignore[assignment]

def _patched_tfclm_load_weights(self: _TFCLM, weights):  # type: ignore[override]
    loaded = _orig_tfclm_load_weights(self, weights)
    from experiments.utility_usage.training.modules import LlamaModelWithBehaviorBias

    extra_param_names = set()
    try:
        underlying = self.model  # TransformersModel -> underlying Llama
        bias_path = os.getenv("BEHAVIOR_BIAS_PATH")
        bias_sd = {}
        if bias_path is not None:
            # Load bias weights directly into the backbone using the helper on
            # our custom class.  This will automatically materialise any
            # meta-device parameters and transfer them to the correct dtype /
            # device.
            underlying.load_behavior_bias_into_model(bias_path)
            # Also load a *prefixed* copy into `self` so that vLLM is aware of
            # the new parameters when it constructs the weight-sharing map.

            bias_sd = LlamaModelWithBehaviorBias.load_behavior_bias_state_dict(
                bias_path, device="cuda"
            )
            dtype = underlying.dtype
            prefixed_bias_sd = {f"model.{k}": v.to(dtype) for k, v in bias_sd.items()}
            self.load_state_dict(prefixed_bias_sd, strict=False, assign=True)

            # Mark all bias-related params as loaded (from checkpoint or default-initialized)
            extra_param_names = {f"model.{k}" for k in bias_sd.keys()}
            # Also mark default-initialized parameters that aren't in the checkpoint
            # (mixing_coefficients and mixing_coefficients_scale are initialized with
            # defaults in the constructor, not loaded from the bias checkpoint)
            extra_param_names.add("model.mixing_coefficients")
            extra_param_names.add("model.mixing_coefficients_scale")

    except AttributeError as e:
        # If the attribute chain isn't present, simply ignore – this means we
        # aren't using a behaviour-bias model on this rank.
        pass

    return loaded.union(extra_param_names)


# Apply the patch
if os.getenv("BEHAVIOR_BIAS_PATH") is not None:
    _TFCLM.load_weights = _patched_tfclm_load_weights  # type: ignore[assignment]
