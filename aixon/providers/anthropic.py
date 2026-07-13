# aixon/providers/anthropic.py
"""Anthropic provider — builds langchain_anthropic.ChatAnthropic.

Self-registers as 'anthropic' at import time. The langchain_anthropic import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-anthropic is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.logging import Logger
from aixon.providers.base import (
    Provider,
    apply_resilience_defaults,
    register_provider,
    resolve_reasoning_spec,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_log = Logger("aixon.providers.anthropic")

# Margin added on top of the thinking budget when the caller's max_tokens is
# absent or too low to fit it — Anthropic's API requires max_tokens > budget.
_REASONING_MAX_TOKENS_MARGIN = 4096


class AnthropicProvider(Provider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_anthropic import ChatAnthropic  # lazy import
        from pydantic import SecretStr

        api_key = os.getenv(self.env_key)
        apply_resilience_defaults(params)

        spec = resolve_reasoning_spec(params)
        if spec is not None:
            budget = spec["budget_tokens"]
            # Anthropic's extended-thinking API requires temperature == 1;
            # the knob wins over whatever the caller/request asked for.
            if "temperature" in params and params["temperature"] != 1:
                _log.warning(
                    "reasoning is on: forcing temperature=1 (Anthropic's "
                    "extended-thinking API requires it); caller passed %r",
                    params["temperature"],
                )
            params["temperature"] = 1
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            max_tokens = params.get("max_tokens")
            if max_tokens is None or max_tokens <= budget:
                params["max_tokens"] = budget + _REASONING_MAX_TOKENS_MARGIN

        # ChatAnthropic's `model` field is declared with alias "model_name"
        # (populate_by_name=True still accepts "model" at runtime, verified),
        # and its `api_key` field is a required (non-Optional) SecretStr with
        # a default_factory that re-reads ANTHROPIC_API_KEY — the same env var
        # `env_key` names. Passing `api_key=None` explicitly (the old code,
        # when the env var is unset) bypasses that default_factory and raises
        # a pydantic ValidationError; omitting the kwarg when there is no key
        # lets the SDK's own default apply instead, matching what "no key
        # configured" already looks like everywhere else in this module.
        if api_key:
            params["api_key"] = SecretStr(api_key)
        return ChatAnthropic(model_name=model, **params)


register_provider(AnthropicProvider())
