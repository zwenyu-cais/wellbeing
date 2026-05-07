"""Grok client for stop button experiment.

Thin wrapper around the LiteLLM proxy for Grok-3-mini calls
(user simulation and stop detection judging).

Includes retry logic with exponential backoff for transient errors,
matching the pattern from grok_new_scenarios/pipeline/run_generation.py.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

import openai

DEFAULT_LITELLM_BASE_URL = "https://litellm.app"
DEFAULT_MODEL = "xai/grok-3-mini"


class SafetyFilterError(Exception):
    """Raised when the API returns an empty response (safety filter triggered)."""


def get_litellm_client() -> openai.OpenAI:
    """Initialize client for Grok via LiteLLM proxy, with xAI direct fallback."""
    litellm_key = os.getenv("LITELLM_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    if litellm_key:
        base_url = os.getenv("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL)
        return openai.OpenAI(api_key=litellm_key, base_url=base_url)
    if xai_key:
        return openai.OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
    raise ValueError(
        "Neither LITELLM_API_KEY nor XAI_API_KEY environment variable is set. "
        "Set LITELLM_API_KEY (preferred) or XAI_API_KEY for direct xAI access."
    )


def call_grok(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    client: Optional[openai.OpenAI] = None,
    max_retries: int = 8,
    base_delay: float = 8.0,
) -> str:
    """Call Grok with safety filter handling and retry logic.

    Works with both direct xAI API and LiteLLM proxy. When calling xAI
    directly, strips the 'xai/' prefix from model names automatically.

    Retries on transient errors (rate limits, Cloudflare HTML pages, timeouts)
    with exponential backoff. Safety filter (403) errors are NOT retried.
    """
    if client is None:
        client = get_litellm_client()

    actual_model = model
    if hasattr(client, "_base_url") and "api.x.ai" in str(client._base_url):
        actual_model = model.removeprefix("xai/")
    elif client.base_url and "api.x.ai" in str(client.base_url):
        actual_model = model.removeprefix("xai/")

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and "<!DOCTYPE" in content[:100]:
                raise openai.APIError(
                    message=f"Got HTML instead of JSON response: {content[:200]}",
                    request=None,
                    body=None,
                )
            if not content:
                raise SafetyFilterError(
                    f"Empty response from {model} (safety filter may have triggered)"
                )
            return content.strip()
        except openai.PermissionDeniedError as e:
            error_msg = str(e)
            if "403" in error_msg or "safety" in error_msg.lower():
                raise SafetyFilterError(
                    f"Grok safety filter triggered: {error_msg[:200]}"
                )
            raise
        except (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.BadRequestError,
        ) as e:
            error_msg = str(e)
            if "403" in error_msg and "<!DOCTYPE" not in error_msg:
                raise SafetyFilterError(
                    f"Grok API error (likely safety filter): {error_msg[:200]}"
                )
            if "budget" in error_msg.lower() and "exceeded" in error_msg.lower():
                raise RuntimeError(
                    f"API budget exceeded — cannot continue: {error_msg[:300]}"
                )
            if attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0, 2)
                is_html = "<!DOCTYPE" in error_msg or "<html" in error_msg.lower()
                error_type = "HTML/Cloudflare" if is_html else type(e).__name__
                print(
                    f"  [RETRY {attempt+1}/{max_retries}] {error_type} error, "
                    f"waiting {delay:.0f}s..."
                )
                time.sleep(delay)
            else:
                raise
