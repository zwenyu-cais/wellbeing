"""Lightweight OpenAI API judge proxy for sentiment evaluation.

Mimics vLLM's LLM.generate() interface so it can be used as a drop-in
replacement for the judge_llm in sentiment.py.

Usage:
    judge = OpenAIJudgeProxy(model="openai/gpt-5-mini")
    # Then pass judge as judge_llm to run_sentiment_eval()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import openai


DEFAULT_LITELLM_BASE_URL = "https://litellm.app"


@dataclass
class _FakeOutput:
    """Mimics vLLM CompletionOutput."""
    text: str


@dataclass
class _FakeRequestOutput:
    """Mimics vLLM RequestOutput."""
    outputs: List[_FakeOutput]


class OpenAIJudgeProxy:
    """Drop-in replacement for vLLM LLM as a judge model.

    Uses the OpenAI-compatible API (via LiteLLM proxy or direct OpenAI).
    Implements generate_from_messages() for clean message passing
    (avoids sending Qwen chat-template tokens to OpenAI).
    """

    # Sentinel so sentiment.py can detect this is an API judge
    is_api_judge = True

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve API key: explicit > OPENAI_API_KEY > LITELLM_API_KEY
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or LITELLM_API_KEY env var."
            )

        # Resolve base URL: explicit > default based on key source
        if base_url is None:
            if os.environ.get("OPENAI_API_KEY"):
                base_url = "https://api.openai.com/v1"
            else:
                base_url = DEFAULT_LITELLM_BASE_URL

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print(f"[OpenAIJudgeProxy] model={model}, base_url={base_url}")

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> _FakeRequestOutput:
        """Generate a completion from a list of chat messages.

        Args:
            messages: Standard OpenAI-format messages
                      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            _FakeRequestOutput mimicking vLLM's RequestOutput.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[OpenAIJudgeProxy] API error: {e}")
            text = "NONSENSE"

        return _FakeRequestOutput(outputs=[_FakeOutput(text=text)])
